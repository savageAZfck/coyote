//! CPU reference evaluator — executes a TProgram in f32 with
//! f16-quantized weights, so `coyote run` can diff silicon vs. spec.
//! Implements the same `grad` backward rules the lowerer emits.

use crate::ast::{AssignOp, BinOp, DType, Qual, UnOp};
use crate::lower::{init_weight_pub, q4_quantize};
use crate::typeck::{CTy, TExpr, TIndex, TProgram, TStmt, Typed};
use crate::{diag, CResult, Diag};
use std::collections::HashMap;

/// Standard Adam optimizer step ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$),
/// with updated weights rounded to fp16 to match the ANE surface representation.
pub fn adam_step(
    w: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    step: usize,
    lr: f64,
) {
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    // fp16-safe epsilon (1e-2 prevents sub-ULP fp16 gradient noise near 0
    // from amplifying into a ±lr sign flip on step 0).
    let eps = 1e-2f32;
    let t = (step + 1) as i32;
    let bc1 = 1.0 - b1.powi(t);
    let bc2 = 1.0 - b2.powi(t);
    let lr = lr as f32;
    for i in 0..w.len().min(g.len()) {
        let gi = g[i];
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        let nw = w[i] - lr * mhat / (vhat.sqrt() + eps);
        w[i] = half::f16::from_f32(nw).to_f32();
    }
}

#[derive(Debug, Clone)]
pub enum Val {
    V(Vec<f32>),
    S(f64),
}

impl Val {
    fn vec(&self, n: usize) -> Vec<f32> {
        match self {
            Val::V(v) => v.clone(),
            Val::S(x) => vec![*x as f32; n],
        }
    }
}

/// f16-quantized weights — identical bytes to the emitted blobs.
fn qw(name: &str, numel: usize) -> Vec<f32> {
    init_weight_pub(name, numel)
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

pub struct Evaluator<'a> {
    p: &'a TProgram,
    env: Vec<HashMap<String, Val>>,
    /// host-provided trainable weights ([N, K] storage)
    pvals: HashMap<String, Vec<f32>>,
    /// host-provided state surfaces ([T, D] row-major, keyed by mil port)
    svals: HashMap<String, Vec<f32>>,
    /// Typed node → forward value (AD tape)
    defs: HashMap<usize, Val>,
    tape: HashMap<String, Typed>,
    grads: HashMap<String, Val>,
    in_grad: bool,
    /// trainable param → updated value (post-run)
    updates: Vec<(String, Vec<f32>)>,
}

impl<'a> Evaluator<'a> {
    pub fn new(p: &'a TProgram) -> Self {
        Self {
            p,
            env: vec![HashMap::new()],
            pvals: HashMap::new(),
            svals: HashMap::new(),
            defs: HashMap::new(),
            tape: HashMap::new(),
            grads: HashMap::new(),
            in_grad: false,
            updates: Vec::new(),
        }
    }

    /// provide current trainable weights ([N, K] storage) — the same
    /// bytes the driver writes into the param surfaces.
    pub fn set_params(&mut self, vals: HashMap<String, Vec<f32>>) {
        self.pvals = vals;
    }

    /// provide current state surfaces ([T,D] row-major, keyed by mil
    /// port `kv_k`) — the same bytes the driver feeds back.
    pub fn set_states(&mut self, vals: HashMap<String, Vec<f32>>) {
        self.svals = vals;
    }

    /// trainable updates produced by the run: (param, [N,K] values).
    pub fn param_updates(&self) -> &[(String, Vec<f32>)] {
        &self.updates
    }

    pub fn run(&mut self, inputs: &HashMap<String, Vec<f32>>) -> CResult<Vec<f32>> {
        for (arg, port) in &self.p.arg_ports {
            if let Some(v) = inputs.get(port) {
                self.env[0].insert(arg.clone(), Val::V(v.clone()));
            } else {
                return diag(format!("missing input {port}"), 0, 0);
            }
        }
        // bind trainable params to their current weights ([N, K] storage)
        for tp in &self.p.params {
            if tp.qual == Qual::Trainable {
                let n = (tp.ty.dims[0] * tp.ty.dims[1]) as usize;
                let v = self
                    .pvals
                    .get(&tp.name)
                    .cloned()
                    .unwrap_or_else(|| qw(&tp.name, n));
                self.env[0].insert(tp.name.clone(), Val::V(v));
            }
        }
        // bind state fields — zero-init unless the host fed back updates
        for (key, port, ty) in &self.p.states {
            let n = ty.numel().max(1) as usize;
            let v = self
                .svals
                .get(port)
                .cloned()
                .unwrap_or_else(|| vec![0.0; n]);
            self.env[0].insert(key.clone(), Val::V(v));
        }
        for s in &self.p.body {
            self.stmt(s)?;
        }
        let v = self.expr(&self.p.ret)?;
        Ok(v.vec(self.p.ret.ty.numel().max(1) as usize))
    }

    fn lookup(&self, name: &str) -> Option<Val> {
        for s in self.env.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    fn is_param(&self, name: &str) -> Option<&CTy> {
        self.p
            .params
            .iter()
            .find(|t| t.name == name)
            .map(|t| &t.ty)
    }

    fn param_val(&self, name: &str) -> CResult<Val> {
        let t = self
            .p
            .params
            .iter()
            .find(|t| t.name == name)
            .ok_or(())
            .map_err(|_| Diag { msg: format!("no param {name}"), line: 0, col: 0 })?;
        // storage is always [N, K] for matrices — matches emitted blobs
        // and the packed runtime layout.
        let expect: Vec<usize> = if t.ty.dims.len() == 2 {
            vec![t.ty.dims[1] as usize, t.ty.dims[0] as usize]
        } else {
            vec![t.ty.numel() as usize]
        };
        let v = crate::weights::resolve(name, Some(&expect), t.src.as_deref(), qw)
            .map_err(|m| Diag { msg: m, line: 0, col: 0 })?;
        let v = if t.ty.dt == DType::Q4 {
            q4_quantize(&v)
        } else {
            v.iter()
                .map(|&x| half::f16::from_f32(x).to_f32())
                .collect()
        };
        Ok(Val::V(v))
    }

    fn fwd(&mut self, e: &Typed) -> CResult<Val> {
        if let TExpr::Ident(s) = &e.e {
            if let Some(v) = self.lookup(s) {
                return Ok(v);
            }
            if self.is_param(s).is_some() {
                return self.param_val(s);
            }
        }
        if let Some(v) = self.defs.get(&(e as *const Typed as usize)) {
            return Ok(v.clone());
        }
        self.expr(e)
    }

    fn expr(&mut self, e: &Typed) -> CResult<Val> {
        let v = self.expr_inner(e)?;
        self.defs.insert(e as *const Typed as usize, v.clone());
        Ok(v)
    }

    fn expr_inner(&mut self, e: &Typed) -> CResult<Val> {
        match &e.e {
            TExpr::Lit(f) => Ok(Val::S(*f)),
            TExpr::Int(n) => Ok(Val::S(*n as f64)),
            TExpr::Ident(s) => {
                if let Some(v) = self.lookup(s) {
                    Ok(v)
                } else {
                    self.param_val(s)
                }
            }
            TExpr::Unary(UnOp::Neg, a) => {
                let v = self.expr(a)?;
                Ok(match v {
                    Val::S(x) => Val::S(-x),
                    Val::V(v) => Val::V(v.iter().map(|x| -x).collect()),
                })
            }
            TExpr::Unary(UnOp::Not, _) => diag("eval: `!` is host-stratum", 0, 0),
            TExpr::Bin(op, a, b) => {
                let n = e.ty.numel().max(1) as usize;
                match op {
                    BinOp::MatMul => {
                        let x = self.expr(a)?.vec(a.ty.numel() as usize);
                        // state-field rhs: `s @ kv.v` contracts dim 0
                        // (scores→values), `q @ kv.k.T` contracts dim 1
                        // of the [T,D] storage (query→scores).
                        let sf: Option<(String, bool)> = match &b.e {
                            TExpr::Field(bx, f) => match &bx.e {
                                TExpr::Ident(s) => Some((format!("{s}.{f}"), false)),
                                _ => None,
                            },
                            TExpr::Transpose(i) => match &i.e {
                                TExpr::Field(bx, f) => match &bx.e {
                                    TExpr::Ident(s) => Some((format!("{s}.{f}"), true)),
                                    _ => None,
                                },
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some((key, transposed)) = sf {
                            // field storage is [T, D] row-major
                            let (t, d) = match &b.e {
                                TExpr::Field(..) => (b.ty.dims[0] as usize, b.ty.dims[1] as usize),
                                TExpr::Transpose(i) => (i.ty.dims[0] as usize, i.ty.dims[1] as usize),
                                _ => unreachable!(),
                            };
                            let v = self
                                .lookup(&key)
                                .ok_or(())
                                .map_err(|_| Diag { msg: format!("unbound state {key}"), line: 0, col: 0 })?
                                .vec(t * d);
                            let nn = b.ty.dims[1] as usize;
                            let mut out = vec![0f32; nn];
                            if transposed {
                                // q @ kv.k.T → out[t] = Σ_d q_d·v[t,d]
                                for t2 in 0..t {
                                    let mut acc = 0f32;
                                    for d2 in 0..d {
                                        acc += x[d2] * v[t2 * d + d2];
                                    }
                                    out[t2] = acc;
                                }
                            } else {
                                // s @ kv.v → out[d] = Σ_t s_t·v[t,d]
                                for d2 in 0..d {
                                    let mut acc = 0f32;
                                    for t2 in 0..t {
                                        acc += x[t2] * v[t2 * d + d2];
                                    }
                                    out[d2] = acc;
                                }
                            }
                            return Ok(Val::V(out));
                        }
                        let (pname, is_trans) = match &b.e {
                            TExpr::Ident(s) => (s.clone(), false),
                            TExpr::Transpose(i) => match &i.e {
                                TExpr::Ident(s) => (s.clone(), true),
                                _ => return diag("eval: bad matmul rhs", 0, 0),
                            },
                            _ => return diag("eval: bad matmul rhs", 0, 0),
                        };
                        // [N, K] storage — runtime weights come via env
                        let mut w = match self.lookup(&pname) {
                            Some(v) => v.vec(0),
                            None => self.param_val(&pname)?.vec(0),
                        };
                        let k = *a.ty.dims.last().unwrap_or(&1) as usize;
                        let nn = b.ty.dims[1] as usize;
                        if is_trans {
                            if let Some(pty) = self.is_param(&pname) {
                                if pty.dims[0] == b.ty.dims[1]
                                    && pty.dims[1] == b.ty.dims[0]
                                    && pty.dims[0] != pty.dims[1]
                                {
                                    // `w` was stored `[K, N]`, transpose to `[N, K]`
                                    let mut wt = vec![0.0f32; nn * k];
                                    for r in 0..k {
                                        for c in 0..nn {
                                            wt[c * k + r] = w[r * nn + c];
                                        }
                                    }
                                    w = wt;
                                }
                            }
                        }
                        let m = if a.ty.dims.len() == 2 {
                            a.ty.dims[0] as usize
                        } else {
                            1
                        };
                        let mut out = vec![0f32; m * nn];
                        for br in 0..m {
                            for j in 0..nn {
                                let mut acc = 0f32;
                                for i in 0..k {
                                    acc += x[br * k + i] * w[j * k + i];
                                }
                                out[br * nn + j] = acc;
                            }
                        }
                        Ok(Val::V(out))
                    }
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        let x = self.expr(a)?.vec(n);
                        let y = self.expr(b)?.vec(n);
                        let f: fn(f32, f32) -> f32 = match op {
                            BinOp::Add => |x, y| x + y,
                            BinOp::Sub => |x, y| x - y,
                            BinOp::Mul => |x, y| x * y,
                            _ => |x, y| x / y,
                        };
                        Ok(Val::V(x.iter().zip(&y).map(|(&a, &b)| f(a, b)).collect()))
                    }
                    _ => diag("eval: comparisons unsupported", 0, 0),
                }
            }
            TExpr::Transpose(inner) => {
                let v = self.expr(inner)?.vec(inner.ty.numel() as usize);
                let (r, c) = (inner.ty.dims[0] as usize, inner.ty.dims[1] as usize);
                let mut out = vec![0.0f32; r * c];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = v[i * c + j];
                    }
                }
                Ok(Val::V(out))
            }
            TExpr::Call(name, args, _) => {
                if name == "__ret" {
                    return self.expr(&args[0]);
                }
                if name == "gather" || name == "embed" {
                    let synth = Typed {
                        e: TExpr::Bin(
                            BinOp::MatMul,
                            Box::new(args[1].clone()),
                            Box::new(args[0].clone()),
                        ),
                        ty: e.ty.clone(),
                    };
                    return self.expr(&synth);
                }
                let n = e.ty.numel().max(1) as usize;
                match name.as_str() {
                    "relu" | "sigmoid" | "tanh" | "silu" | "softmax" | "gelu" => {
                        let x = self.expr(&args[0])?.vec(n);
                        let out: Vec<f32> = match name.as_str() {
                            "relu" => x.iter().map(|v| v.max(0.0)).collect(),
                            "sigmoid" => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
                            "tanh" => x.iter().map(|&v| v.tanh()).collect(),
                            "silu" => x.iter().map(|&v| v / (1.0 + (-v).exp())).collect(),
                            "gelu" => x
                                .iter()
                                .map(|&v| {
                                    let u = 0.7978845608028654f32 * (v + 0.044715 * v * v * v);
                                    0.5 * v * (1.0 + u.tanh())
                                })
                                .collect(),
                            _ => {
                                let m = x.iter().cloned().fold(f32::MIN, f32::max);
                                let ex: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
                                let s: f32 = ex.iter().sum();
                                ex.iter().map(|&v| v / s).collect()
                            }
                        };
                        Ok(Val::V(out))
                    }
                    "concat" => {
                        let mut a = self.expr(&args[0])?.vec(args[0].ty.numel() as usize);
                        let b = self.expr(&args[1])?.vec(args[1].ty.numel() as usize);
                        a.extend_from_slice(&b);
                        Ok(Val::V(a))
                    }
                    "rmsnorm" => {
                        let x = self.expr(&args[0])?.vec(n);
                        let w = match &args[1].e {
                            TExpr::Ident(s) if self.lookup(s).is_none() => {
                                self.param_val(s)?.vec(n)
                            }
                            _ => self.expr(&args[1])?.vec(n),
                        };
                        let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
                        let r = (ms + 1e-5).sqrt().recip();
                        Ok(Val::V(x.iter().zip(&w).map(|(&a, &b)| a * r * b).collect()))
                    }
                    "outer" => {
                        // [T,D] row-major: out[t,d] = a_t · b_d
                        let a = self.expr(&args[0])?.vec(args[0].ty.numel() as usize);
                        let b = self.expr(&args[1])?.vec(args[1].ty.numel() as usize);
                        let mut out = vec![0f32; a.len() * b.len()];
                        for (t, av) in a.iter().enumerate() {
                            for (d, bv) in b.iter().enumerate() {
                                out[t * b.len() + d] = av * bv;
                            }
                        }
                        Ok(Val::V(out))
                    }
                    "mse" => {
                        let x = self.expr(&args[0])?.vec(n);
                        let y = self.expr(&args[1])?.vec(n);
                        let m = x
                            .iter()
                            .zip(&y)
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            / n as f32;
                        Ok(Val::S(m as f64))
                    }
                    _ => diag(format!("eval: {name}"), 0, 0),
                }
            }
            TExpr::Grad { loss, wrt } => self.grad(loss, wrt),
            TExpr::Index(base, idxs) => {
                if base.ty.dims.len() == 1 && idxs.len() == 1 {
                    if let TIndex::Range(lo, hi) = idxs[0] {
                        let v = self.expr(base)?.vec(base.ty.numel() as usize);
                        return Ok(Val::V(v[lo as usize..hi as usize].to_vec()));
                    }
                }
                if base.ty.dims.len() == 2 && idxs.len() == 1 {
                    if let TIndex::Lit(row) = idxs[0] {
                        // param `w` is stored `[N, K]` where `K = dims[0], N = dims[1]`
                        let v_dim = base.ty.dims[0] as usize;
                        let d_dim = base.ty.dims[1] as usize;
                        let mut oh = vec![0.0f32; v_dim];
                        oh[row as usize] = 1.0;
                        let w = self.expr(base)?.vec(v_dim * d_dim);
                        let mut out = vec![0.0f32; d_dim];
                        for j in 0..d_dim {
                            let mut acc = 0.0f32;
                            for i in 0..v_dim {
                                acc += oh[i] * w[j * v_dim + i];
                            }
                            out[j] = acc;
                        }
                        return Ok(Val::V(out));
                    }
                }
                diag("eval: unsupported index form", 0, 0)
            }
            TExpr::Field(bx, f) => {
                if let TExpr::Ident(s) = &bx.e {
                    let key = format!("{s}.{f}");
                    return self
                        .lookup(&key)
                        .ok_or(())
                        .map_err(|_| Diag { msg: format!("unbound state {key}"), line: 0, col: 0 });
                }
                diag("eval: field access only on state blocks", 0, 0)
            }
            TExpr::Block(stmts, ret) => {
                if !self.in_grad {
                    self.env.push(HashMap::new());
                }
                for s in stmts {
                    if let TStmt::Let { name, expr } = s {
                        let v = self.expr(expr)?;
                        self.tape.insert(name.clone(), expr.clone());
                        self.env.last_mut().unwrap().insert(name.clone(), v);
                    } else {
                        self.stmt(s)?;
                    }
                }
                let v = self.expr(ret)?;
                if !self.in_grad {
                    self.env.pop();
                }
                Ok(v)
            }
        }
    }

    /// CPU mirror of the lowerer's tape backward.
    fn grad(&mut self, loss: &Typed, wrt: &str) -> CResult<Val> {
        self.env.push(HashMap::new());
        self.tape.clear();
        self.grads.clear();
        self.in_grad = true;
        let terminal: Typed = match &loss.e {
            TExpr::Block(stmts, ret) => {
                for s in stmts {
                    if let TStmt::Let { name, expr } = s {
                        let v = self.expr(expr)?;
                        self.tape.insert(name.clone(), expr.clone());
                        self.env.last_mut().unwrap().insert(name.clone(), v);
                    } else {
                        self.stmt(s)?;
                    }
                }
                (**ret).clone()
            }
            _ => loss.clone(),
        };
        match &terminal.e {
            TExpr::Call(name, args, _) if name == "mse" => {
                let n = args[0].ty.numel().max(1) as usize;
                let p = self.fwd(&args[0])?.vec(n);
                let t = self.fwd(&args[1])?.vec(n);
                let seed: Vec<f32> = p
                    .iter()
                    .zip(&t)
                    .map(|(a, b)| 2.0 * (a - b) / n as f32)
                    .collect();
                self.back(&args[0], &seed)?;
            }
            _ => return diag("grad: v1 losses must end in mse", 0, 0),
        }
        let out = self
            .grads
            .get(wrt)
            .cloned()
            .ok_or(())
            .map_err(|_| Diag { msg: format!("grad: loss does not depend on {wrt}"), line: 0, col: 0 })?;
        self.in_grad = false;
        self.tape.clear();
        self.env.pop();
        Ok(out)
    }

    /// adjoint on a subtree; adj is a Vec matching e.ty ([N,K] storage
    /// for matrices — same convention as everything else).
    fn back(&mut self, e: &Typed, adj: &[f32]) -> CResult<()> {
        match &e.e {
            TExpr::Ident(s) => {
                if self.is_param(s).is_some() {
                    self.acc(s, adj.to_vec());
                } else if let Some(def) = self.tape.get(s).cloned() {
                    self.back(&def, adj)?;
                }
            }
            TExpr::Lit(_) | TExpr::Int(_) => {}
            TExpr::Unary(UnOp::Neg, a) => {
                let neg: Vec<f32> = adj.iter().map(|x| -x).collect();
                self.back(a, &neg)?;
            }
            TExpr::Unary(UnOp::Not, _) => {
                return diag("back: `!` is host-stratum", 0, 0)
            }
            TExpr::Bin(op, a, b) => match op {
                BinOp::Add => {
                    self.back(a, adj)?;
                    self.back(b, adj)?;
                }
                BinOp::Sub => {
                    self.back(a, adj)?;
                    let neg: Vec<f32> = adj.iter().map(|x| -x).collect();
                    self.back(b, &neg)?;
                }
                BinOp::Mul => {
                    let n = adj.len();
                    if self.needs_grad(a) {
                        let bv = self.fwd(b)?.vec(n);
                        let d: Vec<f32> = adj.iter().zip(&bv).map(|(x, y)| x * y).collect();
                        self.back(a, &d)?;
                    }
                    if self.needs_grad(b) {
                        let av = self.fwd(a)?.vec(n);
                        let d: Vec<f32> = adj.iter().zip(&av).map(|(x, y)| x * y).collect();
                        self.back(b, &d)?;
                    }
                }
                BinOp::Div => {
                    let n = adj.len();
                    if let TExpr::Lit(s) = b.e {
                        if self.needs_grad(a) {
                            let d: Vec<f32> = adj.iter().map(|x| x / s as f32).collect();
                            self.back(a, &d)?;
                        }
                    } else {
                        let bv = self.fwd(b)?.vec(n);
                        if self.needs_grad(a) {
                            let da: Vec<f32> = adj.iter().zip(&bv).map(|(d, y)| d / y).collect();
                            self.back(a, &da)?;
                        }
                        if self.needs_grad(b) {
                            let av = self.fwd(a)?.vec(n);
                            let db: Vec<f32> = adj
                                .iter()
                                .zip(&av)
                                .zip(&bv)
                                .map(|((d, x), y)| -(d * x) / (y * y))
                                .collect();
                            self.back(b, &db)?;
                        }
                    }
                }
                BinOp::MatMul => {
                    // grad through state-field contractions lands in M3b
                    let is_state_rhs = matches!(
                        &b.e,
                        TExpr::Field(..)
                    ) || matches!(&b.e, TExpr::Transpose(i) if matches!(&i.e, TExpr::Field(..)));
                    if is_state_rhs {
                        if self.needs_grad(a) {
                            return diag("grad through cache/state ops lands in M3b", 0, 0);
                        }
                        return Ok(());
                    }
                    let k = a.ty.dims[0] as usize;
                    let nn = b.ty.dims[1] as usize;
                    let pname = match &b.e {
                        TExpr::Ident(s) => Some(s.clone()),
                        TExpr::Transpose(i) => match &i.e {
                            TExpr::Ident(s) => Some(s.clone()),
                            _ => None,
                        },
                        _ => None,
                    };
                    if self.needs_grad(a) {
                        // dx_i = Σ_j adj_j · w[j,k+i] — w in [N,K] storage
                        let w = self.fwd(b)?.vec(nn * k);
                        let mut dx = vec![0f32; k];
                        for i in 0..k {
                            let mut acc = 0f32;
                            for j in 0..nn {
                                acc += adj[j] * w[j * k + i];
                            }
                            dx[i] = acc;
                        }
                        self.back(a, &dx)?;
                    }
                    if let Some(p) = pname {
                        if self.is_param(&p).is_some() {
                            let x = self.fwd(a)?.vec(k);
                            // dW[n,k] = adj_n · x_k — [N,K] storage
                            let mut dw = vec![0f32; nn * k];
                            for j in 0..nn {
                                for i in 0..k {
                                    dw[j * k + i] = adj[j] * x[i];
                                }
                            }
                            self.acc(&p, dw);
                        }
                    }
                }
                _ => {}
            },
            TExpr::Call(name, args, _) => {
                let x = &args[0];
                let n = adj.len();
                match name.as_str() {
                    "sigmoid" => {
                        let h = self.fwd(e)?.vec(n);
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&h)
                            .map(|(d, h)| d * h * (1.0 - h))
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "tanh" => {
                        let t = self.fwd(e)?.vec(n);
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&t)
                            .map(|(d, t)| d * (1.0 - t * t))
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "silu" => {
                        let xv = self.fwd(x)?.vec(n);
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&xv)
                            .map(|(d, x)| {
                                let s = 1.0 / (1.0 + (-x).exp());
                                d * s * (1.0 + x * (1.0 - s))
                            })
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "relu" => {
                        let xv = self.fwd(x)?.vec(n);
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&xv)
                            .map(|(&d, &v)| if v > 0.0 { d } else { 0.0 })
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "gelu" => {
                        let xv = self.fwd(x)?.vec(n);
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&xv)
                            .map(|(&d, &v)| {
                                let a = 0.7978845608028654f32;
                                let b = 0.044715f32;
                                let u = a * (v + b * v * v * v);
                                let t = u.tanh();
                                let dg = 0.5 * (1.0 + t)
                                    + 0.5 * v * (1.0 - t * t) * a * (1.0 + 3.0 * b * v * v);
                                d * dg
                            })
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "softmax" => {
                        let s = self.fwd(e)?.vec(n);
                        let dot: f32 = adj.iter().zip(&s).map(|(d, si)| d * si).sum();
                        let dx: Vec<f32> = adj
                            .iter()
                            .zip(&s)
                            .map(|(d, si)| si * (d - dot))
                            .collect();
                        self.back(x, &dx)?;
                    }
                    "rmsnorm" => {
                        let xv = self.fwd(x)?.vec(n);
                        let wv = match &args[1].e {
                            TExpr::Ident(s) if self.lookup(s).is_none() => {
                                self.param_val(s)?.vec(n)
                            }
                            _ => self.fwd(&args[1])?.vec(n),
                        };
                        let ms = xv.iter().map(|v| v * v).sum::<f32>() / n as f32;
                        let r = (ms + 1e-5).sqrt().recip();
                        let xhat: Vec<f32> = xv.iter().map(|v| v * r).collect();
                        if let TExpr::Ident(wp) = &args[1].e {
                            if self.is_param(wp).is_some() {
                                let dw: Vec<f32> =
                                    adj.iter().zip(&xhat).map(|(d, xh)| d * xh).collect();
                                self.acc(wp, dw);
                            }
                        }
                        if self.needs_grad(x) {
                            let g: Vec<f32> = adj.iter().zip(&wv).map(|(d, w)| d * w).collect();
                            let mean_gx: f32 =
                                g.iter().zip(&xhat).map(|(gi, xh)| gi * xh).sum::<f32>()
                                    / n as f32;
                            let dx: Vec<f32> = g
                                .iter()
                                .zip(&xhat)
                                .map(|(gi, xh)| r * (gi - xh * mean_gx))
                                .collect();
                            self.back(x, &dx)?;
                        }
                    }
                    "concat" => {
                        let k1 = args[0].ty.dims[0] as usize;
                        if self.needs_grad(&args[0]) {
                            self.back(&args[0], &adj[..k1])?;
                        }
                        if self.needs_grad(&args[1]) {
                            self.back(&args[1], &adj[k1..])?;
                        }
                    }
                    "gather" | "embed" => {
                        let synth = Typed {
                            e: TExpr::Bin(
                                BinOp::MatMul,
                                Box::new(args[1].clone()),
                                Box::new(args[0].clone()),
                            ),
                            ty: e.ty.clone(),
                        };
                        self.back(&synth, adj)?;
                    }
                    "mse" => {
                        let p = self.fwd(&args[0])?.vec(n);
                        let t = self.fwd(&args[1])?.vec(n);
                        let dx: Vec<f32> = p
                            .iter()
                            .zip(&t)
                            .map(|(a, b)| 2.0 * (a - b) / n as f32)
                            .collect();
                        self.back(&args[0], &dx)?;
                    }
                    _ => return diag(format!("eval grad: no rule for {name}"), 0, 0),
                }
            }
            TExpr::Transpose(inner) => {
                if self.needs_grad(inner) {
                    let (r, c) = (inner.ty.dims[0] as usize, inner.ty.dims[1] as usize);
                    let mut dt = vec![0.0f32; r * c];
                    for i in 0..r {
                        for j in 0..c {
                            dt[i * c + j] = adj[j * r + i];
                        }
                    }
                    self.back(inner, &dt)?;
                }
            }
            TExpr::Index(base, idxs) => {
                if self.needs_grad(base) && base.ty.dims.len() == 1 && idxs.len() == 1 {
                    if let TIndex::Range(lo, hi) = idxs[0] {
                        let k = base.ty.dims[0] as usize;
                        let mut dx = vec![0.0f32; k];
                        dx[lo as usize..hi as usize].copy_from_slice(adj);
                        self.back(base, &dx)?;
                    }
                }
            }
            TExpr::Block(_stmts, ret) => {
                self.back(ret, adj)?;
            }
            TExpr::Grad { .. } | TExpr::Field(..) => {
                return diag("eval grad: unsupported node", 0, 0)
            }
        }
        Ok(())
    }

    fn acc(&mut self, name: &str, dw: Vec<f32>) {
        match self.grads.get(name) {
            None => {
                self.grads.insert(name.to_string(), Val::V(dw));
            }
            Some(Val::V(prev)) => {
                let sum: Vec<f32> = prev.iter().zip(&dw).map(|(a, b)| a + b).collect();
                self.grads.insert(name.to_string(), Val::V(sum));
            }
            _ => {}
        }
    }

    fn needs_grad(&self, e: &Typed) -> bool {
        self.needs_grad_d(e, 0)
    }
    fn needs_grad_d(&self, e: &Typed, depth: usize) -> bool {
        if depth > 32 {
            return false;
        }
        match &e.e {
            TExpr::Ident(s) => {
                self.is_param(s).is_some()
                    || self
                        .tape
                        .get(s)
                        .map(|t| self.needs_grad_d(t, depth + 1))
                        .unwrap_or(false)
            }
            TExpr::Unary(_, a) | TExpr::Transpose(a) => self.needs_grad_d(a, depth + 1),
            TExpr::Bin(_, a, b) => {
                self.needs_grad_d(a, depth + 1) || self.needs_grad_d(b, depth + 1)
            }
            TExpr::Call(_, args, named) => {
                args.iter().any(|a| self.needs_grad_d(a, depth + 1))
                    || named.iter().any(|(_, a)| self.needs_grad_d(a, depth + 1))
            }
            TExpr::Index(a, _) | TExpr::Field(a, _) => self.needs_grad_d(a, depth + 1),
            TExpr::Block(stmts, ret) => {
                stmts.iter().any(|s| match s {
                    TStmt::Let { expr, .. } => self.needs_grad_d(expr, depth + 1),
                    TStmt::Assign { expr, .. } => self.needs_grad_d(expr, depth + 1),
                    TStmt::ParamUpdate { grad, .. } | TStmt::AdamUpdate { grad, .. } => {
                        self.needs_grad_d(grad, depth + 1)
                    }
                    TStmt::StateUpdate { expr, .. } => self.needs_grad_d(expr, depth + 1),
                    TStmt::Expr(t) => self.needs_grad_d(t, depth + 1),
                }) || self.needs_grad_d(ret, depth + 1)
            }
            TExpr::Grad { .. } => false,
            _ => false,
        }
    }

    fn stmt(&mut self, s: &TStmt) -> CResult<()> {
        match s {
            TStmt::Let { name, expr } => {
                let v = self.expr(expr)?;
                self.env.last_mut().unwrap().insert(name.clone(), v);
            }
            TStmt::ParamUpdate { param, scale, grad } => {
                if let TExpr::Grad { loss, wrt } = &grad.e {
                    let g = self.grad(loss, wrt)?.vec(grad.ty.numel() as usize);
                    let cur = self.lookup(param).unwrap_or(Val::S(0.0)).vec(g.len());
                    let nv: Vec<f32> = cur
                        .iter()
                        .zip(&g)
                        .map(|(w, d)| w + (*scale as f32) * d)
                        .collect();
                    if let Some(e) = self.updates.iter_mut().find(|(p, _)| p == param) {
                        e.1 = nv.clone();
                    } else {
                        self.updates.push((param.clone(), nv.clone()));
                    }
                    self.env
                        .last_mut()
                        .unwrap()
                        .insert(param.clone(), Val::V(nv));
                }
            }
            TStmt::AdamUpdate { param, grad, .. } => {
                if let TExpr::Grad { loss, wrt } = &grad.e {
                    // Record the raw gradient `g` into `updates`; the driver's
                    // Adam moment loop (`adam_step`) updates `w`, `m`, `v` across steps.
                    let g = self.grad(loss, wrt)?.vec(grad.ty.numel() as usize);
                    if let Some(e) = self.updates.iter_mut().find(|(p, _)| p == param) {
                        e.1 = g;
                    } else {
                        self.updates.push((param.clone(), g));
                    }
                }
            }
            TStmt::Assign { name, op, expr } => {
                let cur = self.lookup(name).unwrap_or(Val::S(0.0));
                let v = self.expr(expr)?;
                let nv = match op {
                    AssignOp::Set => v,
                    AssignOp::Add | AssignOp::Sub | AssignOp::Mul | AssignOp::Div => {
                        let (cn, vn) = match (&cur, &v) {
                            (Val::V(c), _) => (c.len(), v.vec(c.len())),
                            (Val::S(_), Val::V(x)) => (x.len(), v.vec(x.len())),
                            _ => (1, v.vec(1)),
                        };
                        let c = cur.vec(cn);
                        let f: fn(f32, f32) -> f32 = match op {
                            AssignOp::Add => |a, b| a + b,
                            AssignOp::Sub => |a, b| a - b,
                            AssignOp::Mul => |a, b| a * b,
                            _ => |a, b| a / b,
                        };
                        Val::V(c.iter().zip(&vn).map(|(&a, &b)| f(a, b)).collect())
                    }
                };
                // record trainable updates for the feedback loop
                if self.is_param(name).is_some() {
                    if let Val::V(up) = &nv {
                        if let Some(e) = self.updates.iter_mut().find(|(p, _)| p == name) {
                            e.1 = up.clone();
                        } else {
                            self.updates.push((name.clone(), up.clone()));
                        }
                    }
                }
                self.env.last_mut().unwrap().insert(name.clone(), nv);
            }
            TStmt::StateUpdate { key, port, expr } => {
                let v = self.expr(expr)?;
                if let Val::V(up) = &v {
                    if let Some(e) = self.updates.iter_mut().find(|(p, _)| p == port) {
                        e.1 = up.clone();
                    } else {
                        self.updates.push((port.clone(), up.clone()));
                    }
                }
                self.env.last_mut().unwrap().insert(key.clone(), v);
            }
            TStmt::Expr(e) => {
                self.expr(e)?;
            }
        }
        Ok(())
    }
}
