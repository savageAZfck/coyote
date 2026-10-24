//! coyote repl — the language's interactive shell.
//!
//! `coyote repl [file.cy]` loads the organism once (tensor programs compile
//! to silicon or the cpu backend exactly as `run` would), then hands you a
//! prompt. Two machines under the hood:
//!
//!   * the **live** VM — the loaded organism. `:call`, `:main`, `:progs`,
//!     `:params`, `:w` all operate on it; witness entries and mutated
//!     state are real.
//!   * the **scratch** VM — a host-stratum-only session built from the
//!     file's source plus whatever declarations you add at the prompt.
//!     Bare expressions eval here; `host fn`/`reflex`/`struct`/`enum`
//!     declarations accumulate in the session source.
//!
//! Scratch has no tensor programs loaded — evaluating a line that calls an
//! `ane(...)` program errors; drive those through the live VM instead.

use crate::parser;
use crate::typeck;
use crate::vm::{Val, Vm};
use crate::{load_vm, lower::sha256_hex};
use std::cell::RefCell;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;

const BANNER: &str = "coyote repl — source to silicon, interactively
  :help            this text
  :fns             host fns + signatures
  :progs           tensor programs + backend + io
  :params          persistent tensors (params/tables/state) + dims
  name(a, b, \"s\")    call a host fn on the live VM  (:call works too)
  :main            run host fn main
  :step tok [tok…] feed decoder tokens one step each (:decode N)
  :w [n]           last n witness entries
  :reset           reset program state surfaces
  :load file.cy    reload organism + reset session
  :src             show session source size / accumulated decls
  :q               quit
  anything else    eval as a host-fn body, or a decl to keep
";

fn usage() {
    eprintln!("coyote repl [file.cy] [-o dir] [--cpu] [--rebuild]");
}

/// depth-aware split on top-level commas (strings, brackets, braces).
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut in_str, mut start) = (0i32, false, 0usize);
    for (i, ch) in s.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '(' | '[' | '{' if !in_str => depth += 1,
            ')' | ']' | '}' if !in_str => depth -= 1,
            ',' if !in_str && depth == 0 => {
                out.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// scalar/vector literal → Val. i64, f64, bool, "str", [..], {k: v}.
fn parse_val(s: &str) -> Result<Val, String> {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        return Ok(Val::S(Rc::from(&s[1..s.len() - 1])));
    }
    match s {
        "true" => return Ok(Val::B(true)),
        "false" => return Ok(Val::B(false)),
        _ => {}
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let items: Vec<Val> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            split_args(inner)
                .iter()
                .map(|p| parse_val(p))
                .collect::<Result<_, _>>()?
        };
        return Ok(Val::V(Rc::new(RefCell::new(items))));
    }
    if let Ok(i) = s.parse::<i64>() {
        return Ok(Val::I(i));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Ok(Val::F(f));
    }
    Err(format!(
        "can't parse `{s}` — try i64, f64, \"str\", bool, or [..]"
    ))
}

/// split a statement line at the last top-level `;` — `(head, tail)` when
/// the tail looks like a value position (`let x = 1; x + 1` → `x + 1`).
fn split_last_stmt(line: &str) -> Option<(String, String)> {
    let (mut depth, mut in_str, mut last) = (0i32, false, None);
    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '(' | '[' | '{' if !in_str => depth += 1,
            ')' | ']' | '}' if !in_str => depth -= 1,
            ';' if !in_str && depth == 0 => last = Some(i),
            _ => {}
        }
    }
    let i = last?;
    let tail = line[i + 1..].trim();
    if tail.is_empty() || tail.starts_with('}') {
        return None;
    }
    Some((line[..=i].to_string(), tail.to_string()))
}

/// `name(a, b, …)` → (name, args). Requires the full line be one call.
fn parse_call(line: &str) -> Option<(String, Vec<Val>)> {
    let line = line.trim();
    let open = line.find('(')?;
    if !line.ends_with(')') {
        return None;
    }
    let name = line[..open].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    let inner = &line[open + 1..line.len() - 1];
    let args = if inner.trim().is_empty() {
        Vec::new()
    } else {
        split_args(inner)
            .iter()
            .map(|p| parse_val(p))
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    Some((name.to_string(), args))
}

/// lines that extend the session source verbatim rather than evaluating.
fn looks_like_decl(line: &str) -> bool {
    let t = line.trim_start();
    [
        "host",
        "reflex",
        "struct",
        "enum",
        "param",
        "table",
        "state",
        "io",
        "extern",
        "invariant",
        "fn",
    ]
    .iter()
    .any(|p| t == *p || t.starts_with(&format!("{p} ")))
}

struct Session {
    /// the live organism VM (what :call/:main/witness drive)
    vm: Vm,
    /// file source + accumulated repl decls — scratch compiles from this
    src: String,
    /// decls added at the prompt, for :src listing
    decls: Vec<String>,
    /// autoregressive position for :step
    pos: usize,
}

impl Session {
    fn load(file: &Path, out: &Path, cpu: bool, rebuild: bool) -> Result<Self, String> {
        let vm = load_vm(file, out, cpu, rebuild, false)?;
        let src = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        Ok(Self {
            vm,
            src,
            decls: Vec::new(),
            pos: 0,
        })
    }

    fn bare() -> Result<Self, String> {
        // no file — session starts empty; scratch eval only. the ping fn
        // guarantees a host stratum so the VM can always be built.
        let src = "// coyote repl session\n\nhost fn __repl_ping() -> i64 {\n    return 0;\n}\n"
            .to_string();
        let h = check_src(&src)?;
        let host = h.host.ok_or("internal: no host stratum")?;
        let vm = Vm::new(&host, Vec::new(), &sha256_hex(src.as_bytes()))?;
        Ok(Self {
            vm,
            src,
            decls: Vec::new(),
            pos: 0,
        })
    }

    /// extend session source; returns error and rolls back on failure.
    fn add_decl(&mut self, decl: &str) -> Result<String, String> {
        let start = appended_line(&self.src);
        let candidate = format!("{}\n\n{decl}\n", self.src);
        check_src(&candidate).map_err(|e| rebase(&e, start))?; // validate before committing
        self.src = candidate;
        // first line of the decl as the label
        let label = decl.lines().next().unwrap_or(decl).trim().to_string();
        self.decls.push(label.clone());
        Ok(label)
    }
}

fn check_src(src: &str) -> Result<typeck::Checked, String> {
    let ast = parser::parse(src).map_err(|d| d.to_string())?;
    typeck::check_all(&ast).map_err(|d| d.to_string())
}

/// count of '\n' — absolute line a freshly-appended decl/body lands on
/// (`{src}\n\n` + text → line nl(src) + 3).
fn appended_line(src: &str) -> usize {
    src.matches('\n').count() + 3
}

/// `line:col: msg` → rebased against the typed input. Errors landing
/// above the input point back at accumulated session code.
fn rebase(err: &str, input_start: usize) -> String {
    let Some((l, rest)) = err.split_once(':') else {
        return err.to_string();
    };
    let Some((c, m)) = rest.split_once(": ") else {
        return err.to_string();
    };
    let (Ok(l), Ok(c)) = (l.parse::<usize>(), c.parse::<usize>()) else {
        return err.to_string();
    };
    if l < input_start {
        format!("session line {l}: {m}")
    } else {
        let rel = (l - input_start + 1).max(1);
        format!("{rel}:{c}: {m}")
    }
}

fn show_call(vm: &mut Vm, line: &str) -> Result<(), String> {
    let Some((name, args)) = parse_call(line.strip_prefix(":call").unwrap_or(line)) else {
        return Err("usage: name(arg, …) — scalars: i64, f64, \"str\", bool, [..]".into());
    };
    let v = vm.call_fn(&name, args)?;
    println!("{}", vm.stringify(&v));
    Ok(())
}

fn show_progs(vm: &Vm) {
    let names = vm.prog_names();
    if names.is_empty() {
        println!("  (no tensor programs)");
        return;
    }
    for n in names {
        let backend = if vm.is_ane() { "ane" } else { "cpu" };
        match vm.prog_ios(n) {
            Some(ios) => {
                let io = ios
                    .iter()
                    .map(|(n, d)| format!("{n}{d:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  {n:<16} {backend}  io: {io}");
            }
            None => println!("  {n:<16} {backend}"),
        }
    }
}

fn show_params(vm: &Vm) {
    let mut names = vm.param_names();
    names.sort();
    if names.is_empty() {
        println!("  (no persistent tensors)");
        return;
    }
    for n in names {
        match vm.param_slice(&n) {
            Some(s) => {
                let head: Vec<String> = s.iter().take(4).map(|v| format!("{v:.4}")).collect();
                println!("  {n:<24} [{} elems]  {head:?}…", s.len());
            }
            None => println!("  {n:<24} (unresolved)"),
        }
    }
}

fn show_witness(vm: &Vm, tail: Option<usize>) {
    let w = vm.witness_log();
    if w.is_empty() {
        println!("  (witness log empty)");
        return;
    }
    let start = tail.map(|n| w.len().saturating_sub(n)).unwrap_or(0);
    for e in &w[start..] {
        println!(
            "  #{:<3} {:<20} {:<40} {}",
            e.seq,
            e.tag,
            e.val,
            &e.hash[..16]
        );
    }
    if start > 0 {
        println!("  … {} earlier entries", start);
    }
}

pub fn run(file: Option<&Path>, out: &Path, cpu: bool, rebuild: bool) -> Result<(), String> {
    let mut s = match file {
        Some(f) => Session::load(f, out, cpu, rebuild)?,
        None => Session::bare()?,
    };
    let where_str = if file.is_some() {
        format!("{}", file.unwrap().display())
    } else {
        "empty session".to_string()
    };
    println!("coyote repl — {where_str}  (:help, :q)");

    let stdin = std::io::stdin();
    loop {
        print!("cy> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) | Err(_) => break, // eof / read error
            _ => {}
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let r: Result<(), String> = (|| {
            if let Some(rest) = line.strip_prefix(':') {
                let mut it = rest.split_whitespace();
                match it.next().unwrap_or("") {
                    "q" | "quit" | "exit" => return Err(String::new()), // sentinel: clean quit
                    "h" | "help" | "?" => print!("{BANNER}"),
                    "fns" => {
                        let fns = s.vm.host_fns();
                        if fns.is_empty() {
                            println!("  (no host fns)");
                        }
                        for (_, sig, reflex) in fns {
                            let tag = if reflex { "  [reflex]" } else { "" };
                            println!("  {sig}{tag}");
                        }
                    }
                    "progs" => show_progs(&s.vm),
                    "params" => show_params(&s.vm),
                    "w" | "witness" => {
                        let n = it.next().and_then(|x| x.parse::<usize>().ok());
                        show_witness(&s.vm, n);
                    }
                    "main" => {
                        let v = s.vm.run_main()?;
                        println!("main() → {}", s.vm.stringify(&v));
                    }
                    "reset" => {
                        s.vm.reset_states();
                        s.pos = 0;
                        println!("✓ state surfaces zeroed");
                    }
                    "step" => {
                        // feed each token id as a decoder step
                        let toks: Vec<u32> = it.filter_map(|t| t.parse::<u32>().ok()).collect();
                        if toks.is_empty() {
                            return Err("usage: :step <token_id> [token_id…]".into());
                        }
                        for (i, t) in toks.iter().enumerate() {
                            let project = i + 1 == toks.len();
                            match s.vm.step_decoder_token(*t, s.pos, project) {
                                Ok(Some(tok)) => println!("  pos {} tok {t} → {tok}", s.pos),
                                Ok(None) => println!("  pos {} tok {t} → (no projection)", s.pos),
                                Err(e) => return Err(e),
                            }
                            s.pos += 1;
                        }
                    }
                    "load" => {
                        let f = it.next().ok_or("usage: :load <file.cy>")?;
                        s = Session::load(Path::new(f), out, cpu, rebuild)?;
                        println!("✓ loaded {f}");
                    }
                    "src" => {
                        println!("  {} bytes session source", s.src.len());
                        for d in &s.decls {
                            println!("    + {d}");
                        }
                    }
                    "call" => show_call(&mut s.vm, line)?,
                    other => return Err(format!("unknown :{other} — :help lists commands")),
                }
                return Ok(());
            }
            // bare `name(...)` → live call when the organism has the fn;
            // session-declared fns fall through to scratch eval below.
            if let Some((name, _)) = parse_call(line) {
                if !line.contains(" let ") && s.vm.host_fns().iter().any(|(n, _, _)| n == &name) {
                    return show_call(&mut s.vm, line);
                }
            }
            // decl or eval through the scratch VM
            if looks_like_decl(line) {
                match s.add_decl(line) {
                    Ok(label) => println!("✓ kept `{label}`"),
                    Err(e) => return Err(format!("decl rejected: {e}")),
                }
                return Ok(());
            }
            // bare expression — wrap in a synthetic host fn, eval in scratch.
            // `-> str` declared so `return str(expr)` handles any type;
            // `let a = 1; a + 1` → `{ let a = 1; return str(a + 1); }`.
            let n = s.decls.len();
            let (body, ret) = match split_last_stmt(line) {
                Some((head, tail)) => (format!("{head} return str({tail});"), " -> str"),
                None if line.trim_end().ends_with(';') || line.trim_end().ends_with('}') => {
                    (line.to_string(), "")
                }
                None => (format!("return str({line});"), " -> str"),
            };
            let eval_src = format!(
                "{}\n\nhost fn __repl_eval_{n}(){ret} {{\n    {body}\n}}\n",
                s.src
            );
            // if the return-wrap fails to check, try the raw line as
            // statements before giving up.
            let eval_src = if check_src(&eval_src).is_err() {
                format!(
                    "{}\n\nhost fn __repl_eval_{n}() {{\n    {line}\n}}\n",
                    s.src
                )
            } else {
                eval_src
            };
            match check_src(&eval_src) {
                Ok(h) => {
                    let host = h.host.ok_or("internal: no host stratum")?;
                    let mut vm = Vm::new(&host, Vec::new(), &sha256_hex(eval_src.as_bytes()))?;
                    let v = vm.call_fn(&format!("__repl_eval_{n}"), Vec::new())?;
                    println!("{}", vm.stringify(&v));
                }
                Err(e) => {
                    if line.contains('(') && parse_call(line).is_some() {
                        return show_call(&mut s.vm, line);
                    }
                    // errors point into the typed line, not the
                    // accumulated session source
                    return Err(rebase(&e, appended_line(&s.src) + 1));
                }
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_empty() => break, // :q
            Err(e) => println!("✗ {e}"),
        }
    }
    println!("✓ session closed");
    Ok(())
}

pub fn usage_err() -> ! {
    usage();
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_args_top_level_commas() {
        assert_eq!(split_args("1, 2, 3"), vec!["1", "2", "3"]);
        assert_eq!(
            split_args("f(1, 2), [3, 4], \"a,b\""),
            vec!["f(1, 2)", "[3, 4]", "\"a,b\""]
        );
        assert_eq!(split_args(""), Vec::<String>::new());
        assert_eq!(split_args("x"), vec!["x"]);
    }

    #[test]
    fn parse_val_scalars() {
        assert!(matches!(parse_val("42"), Ok(Val::I(42))));
        assert!(matches!(parse_val("-7"), Ok(Val::I(-7))));
        assert!(matches!(parse_val("1.5"), Ok(Val::F(v)) if v == 1.5));
        assert!(matches!(parse_val("true"), Ok(Val::B(true))));
        match parse_val("\"hi\"") {
            Ok(Val::S(s)) => assert_eq!(&*s, "hi"),
            _ => panic!("string literal didn't parse"),
        }
        assert!(parse_val("hello").is_err());
    }

    #[test]
    fn parse_val_vector() {
        match parse_val("[1, 2, 3]") {
            Ok(Val::V(v)) => assert_eq!(v.borrow().len(), 3),
            _ => panic!("vec literal didn't parse"),
        }
    }

    #[test]
    fn parse_call_forms() {
        let (n, a) = parse_call("fib(10)").unwrap();
        assert_eq!(n, "fib");
        assert_eq!(a.len(), 1);
        let (n, a) = parse_call("main()").unwrap();
        assert_eq!(n, "main");
        assert!(a.is_empty());
        assert!(parse_call("not a call").is_none());
        assert!(parse_call("f(").is_none());
        // nested call args aren't scalars — parse_call declines, eval handles
        assert!(parse_call("f(g(1))").is_none());
    }

    #[test]
    fn split_last_stmt_finds_tail() {
        let (h, t) = split_last_stmt("let x = 40; x + 2").unwrap();
        assert_eq!(h, "let x = 40;");
        assert_eq!(t, "x + 2");
        assert!(split_last_stmt("x + 2").is_none());
        assert!(split_last_stmt("let x = 1;").is_none());
        // `;` inside a for-body is nested — not top level
        assert!(split_last_stmt("for i in 0..3 { x += i; }").is_none());
    }

    #[test]
    fn decl_detection() {
        for l in [
            "host fn f() { }",
            "reflex r() { }",
            "param w: f16[8, 8]",
            "struct P { x: i64 }",
            "fn g(x: f16[64]) -> f16[64] { x }",
        ] {
            assert!(looks_like_decl(l), "should be a decl: {l}");
        }
        for l in ["fib(1)", "x + 1", "let a = 1"] {
            assert!(!looks_like_decl(l), "not a decl: {l}");
        }
    }
}
