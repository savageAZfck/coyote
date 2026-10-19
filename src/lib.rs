//! coyote — bare-metal language for sovereign computation.
//!
//! source → AST → typed HIR (shape solving) → MIL text → mil_infer → HWX → ANE.
//! No pointers, no aliasing, tensors are values. Everything that parses
//! typechecks; everything that typechecks lowers.

pub mod ast;
pub mod eval;
pub mod hostck;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod typeck;
pub mod vm;
pub mod weights;

#[derive(Debug, Clone)]
pub struct Diag {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for Diag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.msg)
    }
}

pub type CResult<T> = Result<T, Diag>;

pub fn diag<T>(msg: impl Into<String>, line: usize, col: usize) -> CResult<T> {
    Err(Diag {
        msg: msg.into(),
        line,
        col,
    })
}

// ------------------------------------------------ Rust Session & Build API

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::Path;
use std::rc::Rc;

/// Emit a checked `TProgram` to `dir` (`net.mil`, weight blobs,
/// `coyote_build.json`, and `coyote_plan.txt`), using the `CYPLAN1`
/// warm-cache sidecar unless `rebuild` is true.
pub fn emit_program(
    src: &str,
    p: &typeck::TProgram,
    dir: &Path,
    rebuild: bool,
) -> Result<lower::MilOut, String> {
    use ast::DType;
    use typeck::CTy;
    let src_sha = lower::sha256_hex(src.as_bytes());
    let cache = dir.join("coyote_plan.txt");
    if !rebuild {
        if let Ok(text) = std::fs::read_to_string(&cache) {
            if let Some(out) = parse_plan_cache(&text, &src_sha) {
                if dir.join("net.mil").exists() {
                    return Ok(out);
                }
            }
        }
    }
    let out = lower::lower(p, dir.to_str().unwrap_or("."), &src_sha).map_err(|d| d.to_string())?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (fname, bytes) in &out.blobs {
        std::fs::write(dir.join(fname), bytes).map_err(|e| format!("{fname}: {e}"))?;
    }
    std::fs::write(dir.join("net.mil"), &out.mil).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("coyote_build.json"), &out.manifest).map_err(|e| e.to_string())?;
    let dt_ch = |d: &DType| match d {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::Q4 => "q4",
    };
    let dims_s = |t: &CTy| {
        t.dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let mut pl = format!("CYPLAN1\nsrc_sha {src_sha}\nmil_sha {}\n", out.mil_sha);
    for (n, t, contig) in &out.inputs {
        pl.push_str(&format!(
            "input {n} {} {} {}\n",
            dt_ch(&t.dt),
            dims_s(t),
            *contig as u8
        ));
    }
    for (pn, port, t, s) in &out.trainable {
        pl.push_str(&format!(
            "trainable {pn} {port} {} {} {}\n",
            dt_ch(&t.dt),
            dims_s(t),
            s.as_deref().unwrap_or("-")
        ));
    }
    for (pn, mn, t) in &out.param_outs {
        pl.push_str(&format!(
            "param_out {pn} {mn} {} {}\n",
            dt_ch(&t.dt),
            dims_s(t)
        ));
    }
    for (pn, lr) in &out.adam_lrs {
        pl.push_str(&format!("adam_lr {pn} {lr}\n"));
    }
    for (k, port, t) in &out.states {
        pl.push_str(&format!(
            "state {k} {port} {} {}\n",
            dt_ch(&t.dt),
            dims_s(t)
        ));
    }
    for (port, mn, t) in &out.state_outs {
        pl.push_str(&format!(
            "state_out {port} {mn} {} {}\n",
            dt_ch(&t.dt),
            dims_s(t)
        ));
    }
    pl.push_str(&format!(
        "output {} {} {}\n",
        out.output.0,
        dt_ch(&out.output.1.dt),
        dims_s(&out.output.1)
    ));
    std::fs::write(&cache, pl).map_err(|e| e.to_string())?;
    Ok(out)
}

fn parse_plan_cache(text: &str, src_sha: &str) -> Option<lower::MilOut> {
    use ast::DType;
    use typeck::CTy;
    let mut lines = text.lines();
    if lines.next()? != "CYPLAN1" {
        return None;
    }
    if lines.next()?.strip_prefix("src_sha ")? != src_sha {
        return None;
    }
    let mil_sha = lines.next()?.strip_prefix("mil_sha ")?.to_string();
    let rd_ty = |dt: &str, ds: &str| -> Option<CTy> {
        Some(CTy {
            dt: match dt {
                "f16" => DType::F16,
                "f32" => DType::F32,
                "i32" => DType::I32,
                _ => DType::Q4,
            },
            dims: ds
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|d| d.parse().ok())
                .collect::<Option<Vec<i64>>>()?,
        })
    };
    let (mut inputs, mut trainable, mut param_outs) = (Vec::new(), Vec::new(), Vec::new());
    let mut adam_lrs = Vec::new();
    let (mut states, mut state_outs) = (Vec::new(), Vec::new());
    let mut output = None;
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        match f.first().copied() {
            Some("input") if f.len() >= 5 => {
                inputs.push((f[1].into(), rd_ty(f[2], f[3])?, f[4] == "1"));
            }
            Some("trainable") if f.len() >= 6 => {
                let src = (f[5] != "-").then(|| f[5].to_string());
                trainable.push((f[1].into(), f[2].into(), rd_ty(f[3], f[4])?, src));
            }
            Some("param_out") if f.len() >= 5 => {
                param_outs.push((f[1].into(), f[2].into(), rd_ty(f[3], f[4])?));
            }
            Some("adam_lr") if f.len() >= 3 => {
                adam_lrs.push((f[1].into(), f[2].parse().ok()?));
            }
            Some("state") if f.len() >= 5 => {
                states.push((f[1].into(), f[2].into(), rd_ty(f[3], f[4])?));
            }
            Some("state_out") if f.len() >= 5 => {
                state_outs.push((f[1].into(), f[2].into(), rd_ty(f[3], f[4])?));
            }
            Some("output") if f.len() >= 4 => {
                output = Some((f[1].to_string(), rd_ty(f[2], f[3])?));
            }
            _ => {}
        }
    }
    Some(lower::MilOut {
        mil: String::new(),
        blobs: Vec::new(),
        inputs,
        trainable,
        param_outs,
        adam_lrs,
        states,
        state_outs,
        output: output?,
        manifest: String::new(),
        mil_sha,
    })
}

/// Parse, typecheck, compile all tensor `fn`s, and return a ready-to-run
/// [`vm::Vm`] for an organism `.cy` file.
pub fn load_vm(
    file: &Path,
    out_dir: &Path,
    use_cpu: bool,
    rebuild: bool,
    quiet: bool,
) -> Result<vm::Vm, String> {
    let (src, ast) = parser::parse_file(file).map_err(|d| d.to_string())?;
    let checked = typeck::check_all(&ast).map_err(|d| d.to_string())?;
    let host = checked
        .host
        .as_ref()
        .ok_or_else(|| format!("{}: no host stratum declared", file.display()))?;
    let mut progs = Vec::new();
    for (name, tp) in &checked.ane {
        let dir2 = out_dir.join(name);
        let t0 = std::time::Instant::now();
        let out = emit_program(&src, tp, &dir2, rebuild)?;
        let t_emit = t0.elapsed();
        let backend = if use_cpu {
            if !quiet {
                println!(
                    "✓ {name} — cpu backend  (emit {:.1}s)",
                    t_emit.as_secs_f64()
                );
            }
            vm::ProgBackend::Cpu(tp.clone())
        } else {
            let plan = mil_infer::graph::compile_graph(
                dir2.join("net.mil").to_str().unwrap(),
                dir2.to_str().unwrap(),
            )
            .map_err(|e| format!("compile_graph ({name}): {e}"))?;
            if !quiet {
                println!(
                    "✓ {name} — {} program nodes  (emit {:.1}s, compile {:.1}s)",
                    plan.nodes.len(),
                    t_emit.as_secs_f64(),
                    t0.elapsed().as_secs_f64() - t_emit.as_secs_f64()
                );
            }
            let rt = mil_infer::graph::GraphRuntime::load(plan)
                .map_err(|e| format!("runtime load ({name}): {e}"))?;
            vm::ProgBackend::Ane(rt)
        };
        progs.push(vm::AneProg {
            name: name.clone(),
            backend,
            ins: out.inputs,
            trainable: out
                .trainable
                .into_iter()
                .map(|(p, port, ty, _)| (p, port, ty))
                .collect(),
            states: out.states,
            param_outs: out.param_outs,
            state_outs: out.state_outs,
            output: out.output,
        });
    }
    let src_sha = lower::sha256_hex(src.as_bytes());
    vm::Vm::new(host, progs, &src_sha)
}

// ------------------------------------------------ C / Swift FFI (`cdylib`)

thread_local! {
    static LAST_ERR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_err(e: impl Into<String>) {
    let s = CString::new(e.into().replace('\0', " ")).unwrap_or_default();
    LAST_ERR.with(|slot| *slot.borrow_mut() = s);
}

fn c_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(p) }.to_str().ok()
    }
}

/// Last error message on the current thread (valid until the next FFI error).
#[no_mangle]
pub extern "C" fn coyote_last_error() -> *const c_char {
    LAST_ERR.with(|slot| slot.borrow().as_ptr())
}

/// Compile and load a `.cy` organism into an opaque `Vm` handle.
/// Returns NULL on error (inspect `coyote_last_error()`).
#[no_mangle]
pub extern "C" fn coyote_open(
    cy_path: *const c_char,
    out_dir: *const c_char,
    use_cpu: bool,
) -> *mut vm::Vm {
    let Some(path) = c_str(cy_path) else {
        set_err("coyote_open: null cy_path");
        return std::ptr::null_mut();
    };
    let dir = c_str(out_dir).unwrap_or("/tmp/coyote_ffi");
    match load_vm(Path::new(path), Path::new(dir), use_cpu, false, true) {
        Ok(v) => Box::into_raw(Box::new(v)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `Vm` handle opened by `coyote_open`.
#[no_mangle]
pub extern "C" fn coyote_close(vm: *mut vm::Vm) {
    if !vm.is_null() {
        unsafe {
            drop(Box::from_raw(vm));
        }
    }
}

/// Bind a 2-arg `(i64, i64) -> i64` C/Swift function (`@_cdecl`) to an
/// `extern fn` declared in the `.cy` file.
#[no_mangle]
pub extern "C" fn coyote_bind_extern_i64(
    vm: *mut vm::Vm,
    name: *const c_char,
    cb: extern "C" fn(i64, i64) -> i64,
) -> bool {
    let (Some(vm), Some(name)) = (unsafe { vm.as_mut() }, c_str(name)) else {
        return false;
    };
    vm.bind_extern(name, move |args| {
        let a = match args.first() {
            Some(vm::Val::I(i)) => *i,
            _ => 0,
        };
        let b = match args.get(1) {
            Some(vm::Val::I(i)) => *i,
            _ => 0,
        };
        Ok(vm::Val::I(cb(a, b)))
    });
    true
}

/// Bind a `(str) -> str` C/Swift callback (`cb(in_cstr, out_buf, cap) -> len`)
/// to an `extern fn` declared in the `.cy` file.
#[no_mangle]
pub extern "C" fn coyote_bind_extern_str(
    vm: *mut vm::Vm,
    name: *const c_char,
    cb: extern "C" fn(*const c_char, *mut c_char, usize) -> isize,
) -> bool {
    let (Some(vm), Some(name)) = (unsafe { vm.as_mut() }, c_str(name)) else {
        return false;
    };
    vm.bind_extern(name, move |args| {
        let s = match args.first() {
            Some(vm::Val::S(rc)) => rc.to_string(),
            _ => String::new(),
        };
        let cs = CString::new(s).unwrap_or_default();
        let mut buf = vec![0u8; 4096];
        let n = cb(cs.as_ptr(), buf.as_mut_ptr() as *mut c_char, buf.len());
        if n < 0 {
            return Err("extern str callback returned < 0".into());
        }
        let len = (n as usize).min(buf.len());
        let out = String::from_utf8_lossy(&buf[..len]).to_string();
        Ok(vm::Val::S(Rc::from(out.as_str())))
    });
    true
}

/// Run `host fn main()` and write its `i64` return value to `*out_val`.
#[no_mangle]
pub extern "C" fn coyote_run_main_i64(vm: *mut vm::Vm, out_val: *mut i64) -> bool {
    let Some(vm) = (unsafe { vm.as_mut() }) else {
        return false;
    };
    match vm.run_main() {
        Ok(v) => {
            if !out_val.is_null() {
                unsafe {
                    *out_val = match v {
                        vm::Val::I(i) => i,
                        _ => 0,
                    };
                }
            }
            true
        }
        Err(e) => {
            set_err(e);
            false
        }
    }
}

/// Call any `host fn` by name with `nargs` `i64` arguments and write its
/// `i64` result to `*out_val`.
#[no_mangle]
pub extern "C" fn coyote_call_i64(
    vm: *mut vm::Vm,
    fn_name: *const c_char,
    args: *const i64,
    nargs: usize,
    out_val: *mut i64,
) -> bool {
    let (Some(vm), Some(fname)) = (unsafe { vm.as_mut() }, c_str(fn_name)) else {
        return false;
    };
    let slice = if args.is_null() || nargs == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(args, nargs) }
    };
    let vargs: Vec<vm::Val> = slice.iter().map(|&x| vm::Val::I(x)).collect();
    match vm.call_fn(fname, vargs) {
        Ok(v) => {
            if !out_val.is_null() {
                unsafe {
                    *out_val = match v {
                        vm::Val::I(i) => i,
                        _ => 0,
                    };
                }
            }
            true
        }
        Err(e) => {
            set_err(e);
            false
        }
    }
}

/// Call any `host fn` or `reflex` taking a single `str` argument and
/// write its stringified return value (e.g. `Reflex::Quarantine(...)`)
/// into `out_buf`. Returns byte length written, or `-1` on error.
#[no_mangle]
pub extern "C" fn coyote_call_str(
    vm: *mut vm::Vm,
    fn_name: *const c_char,
    arg: *const c_char,
    out_buf: *mut c_char,
    cap: usize,
) -> isize {
    let (Some(vm), Some(fname), Some(arg_s)) =
        (unsafe { vm.as_mut() }, c_str(fn_name), c_str(arg))
    else {
        return -1;
    };
    let vargs = vec![vm::Val::S(Rc::from(arg_s))];
    match vm.call_fn(fname, vargs) {
        Ok(v) => {
            let s = vm.stringify(&v);
            write_c_buf(&s, out_buf, cap)
        }
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// Resume a migrated snapshot into `vm` and run to completion.
#[no_mangle]
pub extern "C" fn coyote_resume(
    vm: *mut vm::Vm,
    snap_path: *const c_char,
    out_val: *mut i64,
) -> bool {
    let (Some(vm), Some(path)) = (unsafe { vm.as_mut() }, c_str(snap_path)) else {
        return false;
    };
    if let Err(e) = vm.resume(path) {
        set_err(e);
        return false;
    }
    match vm.run_resumed() {
        Ok(v) => {
            if !out_val.is_null() {
                unsafe {
                    *out_val = match v {
                        vm::Val::I(i) => i,
                        _ => 0,
                    };
                }
            }
            true
        }
        Err(e) => {
            set_err(e);
            false
        }
    }
}

/// Number of hash-chained witness entries recorded by `vm`.
#[no_mangle]
pub extern "C" fn coyote_witness_count(vm: *const vm::Vm) -> usize {
    match unsafe { vm.as_ref() } {
        Some(vm) => vm.witness_log().len(),
        None => 0,
    }
}

/// Write the JSONL witness chain into `out_buf`. Returns byte length written.
#[no_mangle]
pub extern "C" fn coyote_witness_json(
    vm: *const vm::Vm,
    out_buf: *mut c_char,
    cap: usize,
) -> isize {
    let Some(vm) = (unsafe { vm.as_ref() }) else {
        return -1;
    };
    let mut wl = String::new();
    for w in vm.witness_log() {
        wl.push_str(&format!(
            "{{\"seq\":{},\"tag\":\"{}\",\"val\":\"{}\",\"hash\":\"{}\"}}\n",
            w.seq, w.tag, w.val, w.hash
        ));
    }
    write_c_buf(&wl, out_buf, cap)
}

fn write_c_buf(s: &str, out_buf: *mut c_char, cap: usize) -> isize {
    let bytes = s.as_bytes();
    if !out_buf.is_null() && cap > 0 {
        let n = bytes.len().min(cap - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, n);
            *out_buf.add(n) = 0;
        }
    }
    bytes.len() as isize
}
