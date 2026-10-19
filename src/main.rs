//! coyote — source to silicon.
//!
//!   coyote check  <file.cy>          parse + shape-check only
//!   coyote build  <file.cy> [-o dir] emit .mil + blobs, compile to ANE binaries
//!   coyote run    <file.cy> [-o dir] [--steps N]
//!                                   build + execute on ANE + diff vs CPU;
//!                                   with trainable params, N steps of the
//!                                   weight-feedback loop = on-chip training.

use coyote::emit_program;
use coyote::eval::Evaluator;
use coyote::lower::{init_weight_pub, sha256_hex};
use coyote::parser::parse_file;
use coyote::typeck::check_all;
use coyote::vm::{AneProg, ProgBackend, Vm};
use std::collections::HashMap;
use std::path::PathBuf;

fn out_dir(args: &[String]) -> PathBuf {
    args.windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| PathBuf::from("coyote_out"))
}

fn steps_arg(args: &[String]) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == "--steps")
        .and_then(|w| w[1].parse().ok())
}

fn resume_arg(args: &[String]) -> Option<PathBuf> {
    args.windows(2)
        .find(|w| w[0] == "--resume")
        .map(|w| PathBuf::from(&w[1]))
}

fn rebuild_arg(args: &[String]) -> bool {
    args.iter().any(|a| a == "--rebuild")
}

fn cpu_arg(args: &[String]) -> bool {
    args.iter().any(|a| a == "--cpu")
}

/// elementwise f16 read of an output surface through its desc.
fn read_surface(
    raw: &[u8],
    desc: &mil_infer::espresso::TensorDesc,
    n: usize,
    k: usize,
) -> Vec<f32> {
    if k <= 1 {
        return (0..n)
            .map(|i| {
                // vectors may ride c (strided) or w (contig) — follow the desc
                let o = if desc.w as usize >= n && desc.w > 1 {
                    desc.offset(0, 0, 0, 0, i as u64) as usize
                } else {
                    desc.offset(0, 0, i as u64, 0, 0) as usize
                };
                if o + 2 <= raw.len() {
                    half::f16::from_le_bytes([raw[o], raw[o + 1]]).to_f32()
                } else {
                    f32::NAN
                }
            })
            .collect();
    }
    let mut v = Vec::with_capacity(n * k);
    for c in 0..n {
        for w in 0..k {
            let o = desc.offset(0, 0, c as u64, 0, w as u64) as usize;
            v.push(if o + 2 <= raw.len() {
                half::f16::from_le_bytes([raw[o], raw[o + 1]]).to_f32()
            } else {
                f32::NAN
            });
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("coyote — bare-metal language for sovereign computation");
        eprintln!("  coyote check <file.cy>");
        eprintln!("  coyote build <file.cy> [-o dir]");
        eprintln!("  coyote run   <file.cy> [-o dir] [--steps N]");
        std::process::exit(2);
    }
    let cmd = &args[1];
    let file = PathBuf::from(&args[2]);
    let dir = out_dir(&args[3..]);
    let steps = steps_arg(&args[3..]);
    let rebuild = rebuild_arg(&args[3..]);
    let use_cpu = cpu_arg(&args[3..]);

    let r = (|| -> Result<(), String> {
        let (src, ast) = parse_file(&file).map_err(|d| format!("{d}"))?;
        let checked = check_all(&ast).map_err(|d| format!("{d}"))?;
        // ---- organism mode: `host fn main` drives the programs ----
        if checked.host.is_some() && cmd != "check" {
            let host = checked.host.as_ref().unwrap();
            if !checked.organism {
                return Err("host fns declared but no `host fn main` entry".into());
            }
            let mut progs = Vec::new();
            for (name, tp) in &checked.ane {
                let dir2 = dir.join(name);
                let t0 = std::time::Instant::now();
                let out = emit_program(&src, tp, &dir2, rebuild)?;
                let t_emit = t0.elapsed();
                let backend = if use_cpu {
                    println!(
                        "✓ {name} — cpu backend  (emit {:.1}s)",
                        t_emit.as_secs_f64()
                    );
                    ProgBackend::Cpu(tp.clone())
                } else {
                    let plan = mil_infer::graph::compile_graph(
                        dir2.join("net.mil").to_str().unwrap(),
                        dir2.to_str().unwrap(),
                    )
                    .map_err(|e| format!("compile_graph ({name}): {e}"))?;
                    println!(
                        "✓ {name} — {} program nodes  (emit {:.1}s, compile {:.1}s)",
                        plan.nodes.len(),
                        t_emit.as_secs_f64(),
                        t0.elapsed().as_secs_f64() - t_emit.as_secs_f64()
                    );
                    let rt = mil_infer::graph::GraphRuntime::load(plan)
                        .map_err(|e| format!("runtime load ({name}): {e}"))?;
                    ProgBackend::Ane(rt)
                };
                progs.push(AneProg {
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
            if cmd == "build" {
                return Ok(());
            }
            let src_sha = sha256_hex(src.as_bytes());
            let mut vm = Vm::new(host, progs, &src_sha).map_err(|e| e)?;
            let ret = if let Some(snap) = resume_arg(&args[3..]) {
                vm.resume(snap.to_str().unwrap_or(""))?;
                vm.run_resumed()?
            } else {
                vm.run_main()?
            };
            // the witness log is part of the artifact — hash-chained,
            // written next to the binaries like her ledger.
            let mut wl = String::new();
            for w in vm.witness_log() {
                wl.push_str(&format!(
                    "{{\"seq\":{},\"tag\":\"{}\",\"val\":\"{}\",\"hash\":\"{}\"}}\n",
                    w.seq, w.tag, w.val, w.hash
                ));
            }
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join("witness.jsonl"), &wl).map_err(|e| e.to_string())?;
            println!("✓ organism returned {}", vm.stringify(&ret));
            if !vm.witness_log().is_empty() {
                println!("  witness: {} entries → {}", vm.witness_log().len(), dir.join("witness.jsonl").display());
            }
            return Ok(());
        }
        match cmd.as_str() {
            "check" => {
                println!("✓ {} — checked", file.display());
                for (name, p) in &checked.ane {
                    println!("  program {name}: {} io, {} params, {} stmts", p.ios.len(), p.params.len(), p.body.len());
                }
                if let Some(h) = &checked.host {
                    let nreflex = h.fns.iter().filter(|f| f.is_reflex).count();
                    println!(
                        "  host:     {} fns ({} reflex), {} globals, {} structs, {} enums",
                        h.fns.len(),
                        nreflex,
                        h.globals.len(),
                        h.structs.len(),
                        h.enums.len()
                    );
                }
                if !checked.invariants.is_empty() {
                    println!("  invariants: {}", checked.invariants.join(", "));
                }
                if !checked.caps.is_empty() {
                    println!("  caps:       {}", checked.caps.join(", "));
                }
                Ok(())
            }
            "build" | "run" => {
                let prog = &checked
                    .ane
                    .first()
                    .ok_or("no ane programs and no host entry — nothing to run")?
                    .1;
                let out = emit_program(&src, prog, &dir, rebuild)?;
                let verb = if out.mil.is_empty() { "cached" } else { "lowered" };
                println!("✓ {verb} → {}", dir.join("net.mil").display());
                println!("  mil sha256: {}", &out.mil_sha[..16]);
                let plan = mil_infer::graph::compile_graph(
                    dir.join("net.mil").to_str().unwrap(),
                    dir.to_str().unwrap(),
                )
                .map_err(|e| format!("compile_graph: {e}"))?;
                println!("✓ compiled — {} program nodes", plan.nodes.len());
                if cmd == "build" {
                    return Ok(());
                }
                // ---- run on ANE ----
                let mut rt = mil_infer::graph::GraphRuntime::load(plan)
                    .map_err(|e| format!("runtime load: {e}"))?;
                let mut inputs: HashMap<String, Vec<f32>> = HashMap::new();
                for (name, ty, _contig) in &out.inputs {
                    let n = ty.numel() as usize;
                    inputs.insert(name.clone(), init_weight_pub(&format!("io::{name}"), n));
                }
                for (name, vals) in &inputs {
                    let vc = vals.clone();
                    rt.write_ext(name, move |b, desc| {
                        for (i, v) in vc.iter().enumerate() {
                            // contig vectors live on the w axis; strided on c
                            let o = if desc.w as usize >= vc.len() && desc.w > 1 {
                                desc.offset(0, 0, 0, 0, i as u64)
                            } else {
                                desc.offset(0, 0, i as u64, 0, 0)
                            };
                            if o + 2 <= b.len() {
                                b[o..o + 2]
                                    .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                            }
                        }
                    });
                }
                // trainable params: file source or deterministic init,
                // [N, K] storage written into packed [1, N, 1, K] surfaces.
                let mut ane_w: HashMap<String, Vec<f32>> = HashMap::new();
                for (pname, port, cty, src) in &out.trainable {
                    let (n, k) = (cty.dims[1] as usize, cty.dims[0] as usize);
                    let w = coyote::weights::resolve(
                        pname,
                        Some(&[n, k]),
                        src.as_deref(),
                        init_weight_pub,
                    )
                    .map_err(|e| e.to_string())?;
                    let wc = w.clone();
                    let (nn, kk) = (n, k);
                    rt.write_ext(port, move |b, desc| {
                        for c in 0..nn {
                            for w2 in 0..kk {
                                let o = desc.offset(0, 0, c as u64, 0, w2 as u64);
                                if o + 2 <= b.len() {
                                    b[o..o + 2].copy_from_slice(
                                        &half::f16::from_f32(wc[c * kk + w2]).to_le_bytes(),
                                    );
                                }
                            }
                        }
                    });
                    ane_w.insert(pname.clone(), w);
                }
                // state fields: zero-init surfaces ([T, D] row-major
                // inside packed [1, T, 1, D]).
                let mut ane_s: HashMap<String, Vec<f32>> = HashMap::new();
                for (_key, port, cty) in &out.states {
                    let (t, d) = (cty.dims[0] as usize, cty.dims[1] as usize);
                    rt.write_ext(port, move |b, desc| {
                        for r in 0..t {
                            for c2 in 0..d {
                                let o = desc.offset(0, 0, r as u64, 0, c2 as u64);
                                if o + 2 <= b.len() {
                                    b[o..o + 2]
                                        .copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
                                }
                            }
                        }
                    });
                    ane_s.insert(port.clone(), vec![0.0; t * d]);
                }
                // per-step io conventions for the decode loop: `pos` is
                // the one-hot write position, `mask` the causal mask.
                let pos_io = inputs.iter().find_map(|(n, v)| {
                    (n == "pos").then(|| v.len())
                });
                let mask_io = inputs.iter().find_map(|(n, v)| {
                    (n == "mask").then(|| v.len())
                });
                let training = !out.trainable.is_empty() || !out.states.is_empty();
                // decode mode: an io out named `logits` means the loop
                // argmaxes, gathers the next token's embedding row from
                // the checkpoint (host-side table read), and feeds it
                // back into `x`. seed: BOS.
                let decode = prog
                    .ios
                    .iter()
                    .any(|(n, d, _)| n == "logits" && *d == coyote::ast::IoDir::Out);
                let mut embed_st = None;
                let mut next_tok: Option<usize> = None;
                if decode {
                    let st_path = prog
                        .params
                        .iter()
                        .filter_map(|p| p.src.as_deref())
                        .find_map(|s| s.split_once("::").map(|(p, _)| p.to_string()))
                        .ok_or("decode: no safetensors param source for embeddings")?;
                    embed_st = Some(coyote::weights::SafeTensors::load(&st_path)?);
                    next_tok = Some(151643); // BOS
                }
                let nsteps = steps.unwrap_or(if training { 20 } else { 1 });
                let (oname, oty) = &out.output;
                let n = oty.numel() as usize;
                // target for host mse: io in named t/target, else none
                let target = inputs.get("t").or_else(|| inputs.get("target")).cloned();
                // cpu-side weights track the same init
                let mut cpu_w = ane_w.clone();
                let mut cpu_s = ane_s.clone();
                let mut ane_adam: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
                let mut cpu_adam: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
                for step in 0..nsteps {
                    // token feedback: embed the pending token's row
                    // into x (host gather — the same split Apple's own
                    // path uses for sampling).
                    if let (Some(st), Some(tok)) = (&embed_st, next_tok) {
                        let row = st
                            .f32s_slice("model.embed_tokens.weight", tok, 1024)
                            .map_err(|e| format!("embed {tok}: {e}"))?;
                        let rowc = row.clone();
                        inputs.insert("x".into(), row);
                        rt.write_ext("x", move |b, desc| {
                            for (i, v) in rowc.iter().enumerate() {
                                let o = if desc.w as usize >= rowc.len() && desc.w > 1 {
                                    desc.offset(0, 0, 0, 0, i as u64)
                                } else {
                                    desc.offset(0, 0, i as u64, 0, 0)
                                };
                                if o + 2 <= b.len() {
                                    b[o..o + 2]
                                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                                }
                            }
                        });
                    }
                    // io conventions: one-hot `pos`, causal `mask`
                    if let Some(t) = pos_io {
                        let mut one = vec![0f32; t];
                        one[step % t] = 1.0;
                        inputs.insert("pos".into(), one.clone());
                        rt.write_ext("pos", move |b, desc| {
                            for (i, v) in one.iter().enumerate() {
                                let o = desc.offset(0, 0, i as u64, 0, 0);
                                if o + 2 <= b.len() {
                                    b[o..o + 2]
                                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                                }
                            }
                        });
                    }
                    if let Some(t) = mask_io {
                        let m: Vec<f32> = (0..t)
                            .map(|j| if j <= step % t { 0.0 } else { -1e4 })
                            .collect();
                        inputs.insert("mask".into(), m.clone());
                        rt.write_ext("mask", move |b, desc| {
                            for (i, v) in m.iter().enumerate() {
                                let o = desc.offset(0, 0, i as u64, 0, 0);
                                if o + 2 <= b.len() {
                                    b[o..o + 2]
                                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                                }
                            }
                        });
                    }
                    // rope pairs: `cos`/`sin` io ins of width d get the
                    // rotate-half table for the current position,
                    // θ_i = base^(-2i/d), base = 1e6 (Qwen3 rope).
                    for (nm, base) in [("cos", 1e6f64), ("sin", 1e6f64)] {
                        if let Some(v) = inputs.get(nm) {
                            let d = v.len();
                            let p = step as f64;
                            let v: Vec<f32> = (0..d)
                                .map(|i| {
                                    let th = base.powf(-2.0 * i as f64 / (2.0 * d as f64));
                                    let ang = p * th;
                                    if nm == "cos" { ang.cos() as f32 } else { ang.sin() as f32 }
                                })
                                .collect();
                            inputs.insert(nm.to_string(), v.clone());
                            rt.write_ext(nm, move |b, desc| {
                                for (i, v) in v.iter().enumerate() {
                                    let o = desc.offset(0, 0, i as u64, 0, 0);
                                    if o + 2 <= b.len() {
                                        b[o..o + 2].copy_from_slice(
                                            &half::f16::from_f32(*v).to_le_bytes(),
                                        );
                                    }
                                }
                            });
                        }
                    }
                    rt.run().map_err(|e| format!("ane run: {e}"))?;
                    let idx = rt
                        .out_node(oname)
                        .ok_or_else(|| format!("output node {oname} not found"))?;
                    let desc = rt.out_desc(idx);
                    let raw = rt.output(idx).to_vec();
                    let ane = read_surface(&raw, &desc, n, 1);
                    if decode {
                        let (tok, logit) = ane
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, v)| (i, *v))
                            .unwrap_or((0, 0.0));
                        println!("  step {step:>3}  tok {tok}  (logit {logit:.3})");
                        next_tok = Some(tok);
                    }
                    // ---- CPU reference ----
                    let mut ev = Evaluator::new(&prog);
                    if training {
                        ev.set_params(cpu_w.clone());
                        ev.set_states(cpu_s.clone());
                    }
                    let cpu = ev.run(&inputs).map_err(|d| format!("cpu eval: {d}"))?;
                    let maxdiff = ane
                        .iter()
                        .zip(&cpu)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0f32, f32::max);
                    if training {
                        let mse = target
                            .as_ref()
                            .map(|t| {
                                ane.iter()
                                    .zip(t)
                                    .map(|(a, b)| (a - b) * (a - b))
                                    .sum::<f32>()
                                    / n as f32
                            })
                            .unwrap_or(f32::NAN);
                        // read updated weights, feed back
                        let mut wdiff = 0f32;
                        for (pname, mname, cty) in &out.param_outs {
                            let pidx = rt
                                .out_node(mname)
                                .ok_or_else(|| format!("update node {mname} missing"))?;
                            let pdesc = rt.out_desc(pidx);
                            let praw = rt.output(pidx).to_vec();
                            let (nn, kk) =
                                (cty.dims[1] as usize, cty.dims[0] as usize);
                            let mut wnew = read_surface(&praw, &pdesc, nn, kk);
                            // cpu-side update from the same step
                            let mut cupd = ev
                                .param_updates()
                                .iter()
                                .find(|(p, _)| p == pname)
                                .map(|(_, v)| v.clone())
                                .unwrap_or_default();
                            if let Some(&(_, lr)) = out.adam_lrs.iter().find(|(p, _)| p == pname) {
                                let g_ane = wnew;
                                let mut w_cur = ane_w.get(pname).cloned().unwrap_or_default();
                                let (m, v) = ane_adam
                                    .entry(pname.clone())
                                    .or_insert_with(|| (vec![0.0; w_cur.len()], vec![0.0; w_cur.len()]));
                                coyote::eval::adam_step(&mut w_cur, &g_ane, m, v, step, lr);
                                wnew = w_cur;

                                if !cupd.is_empty() {
                                    let g_cpu = cupd;
                                    let mut wc_cur = cpu_w.get(pname).cloned().unwrap_or_default();
                                    let (mc, vc) = cpu_adam
                                        .entry(pname.clone())
                                        .or_insert_with(|| (vec![0.0; wc_cur.len()], vec![0.0; wc_cur.len()]));
                                    coyote::eval::adam_step(&mut wc_cur, &g_cpu, mc, vc, step, lr);
                                    cupd = wc_cur;
                                }
                            }
                            if !cupd.is_empty() {
                                let d = wnew
                                    .iter()
                                    .zip(&cupd)
                                    .map(|(a, c)| (a - c).abs())
                                    .fold(0f32, f32::max);
                                wdiff = wdiff.max(d);
                                cpu_w.insert(pname.clone(), cupd);
                            }
                            ane_w.insert(pname.clone(), wnew.clone());
                            // feed back into the param surface
                            rt.write_ext(pname, move |b, d2| {
                                for c in 0..nn {
                                    for w2 in 0..kk {
                                        let o = d2.offset(0, 0, c as u64, 0, w2 as u64);
                                        if o + 2 <= b.len() {
                                            b[o..o + 2].copy_from_slice(
                                                &half::f16::from_f32(wnew[c * kk + w2])
                                                    .to_le_bytes(),
                                            );
                                        }
                                    }
                                }
                            });
                        }
                        // state fields — same loop, row-major [T, D]
                        for (port, mname, cty) in &out.state_outs {
                            let pidx = rt
                                .out_node(mname)
                                .ok_or_else(|| format!("state node {mname} missing"))?;
                            let pdesc = rt.out_desc(pidx);
                            let praw = rt.output(pidx).to_vec();
                            let (t, d) = (cty.dims[0] as usize, cty.dims[1] as usize);
                            let snew = read_surface(&praw, &pdesc, t, d);
                            let cupd = ev
                                .param_updates()
                                .iter()
                                .find(|(p, _)| p == port)
                                .map(|(_, v)| v.clone())
                                .unwrap_or_default();
                            if !cupd.is_empty() {
                                let d2 = snew
                                    .iter()
                                    .zip(&cupd)
                                    .map(|(a, c)| (a - c).abs())
                                    .fold(0f32, f32::max);
                                wdiff = wdiff.max(d2);
                                cpu_s.insert(port.clone(), cupd);
                            }
                            ane_s.insert(port.clone(), snew.clone());
                            let port2 = port.clone();
                            rt.write_ext(&port2, move |b, d2| {
                                for r in 0..t {
                                    for c2 in 0..d {
                                        let o = d2.offset(0, 0, r as u64, 0, c2 as u64);
                                        if o + 2 <= b.len() {
                                            b[o..o + 2].copy_from_slice(
                                                &half::f16::from_f32(snew[r * d + c2])
                                                    .to_le_bytes(),
                                            );
                                        }
                                    }
                                }
                            });
                        }
                        let kvnorm: f32 = ane_s
                            .values()
                            .map(|v| v.iter().map(|x| x.abs()).sum::<f32>())
                            .sum();
                        if ane_s.is_empty() {
                            println!(
                                "  step {step:>3}  mse {mse:.6}  |ane−cpu| pred {maxdiff:.6}  w {wdiff:.6}"
                            );
                        } else {
                            println!(
                                "  step {step:>3}  |ane−cpu| pred {maxdiff:.6}  kv {wdiff:.6}  |kv| {kvnorm:.3}"
                            );
                        }
                    } else {
                        println!("✓ executed on ANE — {} outputs", n);
                        println!("  out[:8]  ane: {:?}", &ane[..ane.len().min(8)]);
                        println!("  out[:8]  cpu: {:?}", &cpu[..cpu.len().min(8)]);
                        println!("  max |ane - cpu| = {maxdiff:.6}");
                        if maxdiff < 0.05 {
                            println!("✓ verified — silicon agrees with spec");
                        } else {
                            println!("⚠ divergence — check op semantics");
                        }
                    }
                }
                if training {
                    println!("✓ {nsteps} steps — weights live on-chip, fed back each step");
                }
                Ok(())
            }
            _ => Err(format!("unknown command {cmd:?}")),
        }
    })();
    if let Err(e) = r {
        eprintln!("✗ {e}");
        std::process::exit(1);
    }
}
