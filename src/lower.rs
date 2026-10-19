//! Lowerer: concrete TProgram → MIL text + weight blobs + provenance.
//!
//! Layout convention (the contract `mil_infer::graph::compile_graph`
//! understands): a coyote `f16[C]` rides the wire as `tensor<fp16,
//! [1, C, 1, 1]>`; a matvec result is `[1, 1, 1, N]`; matmul weight blobs
//! are stored `[N, K]` and consumed with `transpose_y`.

use crate::ast::{AssignOp, BinOp, DType, IoDir, Qual, UnOp};
use crate::typeck::{CTy, TExpr, TParam, TProgram, TStmt, Typed};
use crate::{diag, CResult, Diag};
use std::collections::HashMap;
use std::fmt::Write as _;

/// a lowered value: either a named mil tensor or a host scalar.
#[derive(Debug, Clone)]
enum V {
    Named { name: String, ty: CTy, taint: bool },
    Scalar(f64),
}

impl V {
    /// mil name + taint — only valid on `Named` (scalars go through
    /// `named`, which materializes a broadcast blob).
    fn name(&self) -> (String, bool) {
        match self {
            V::Named { name, taint, .. } => (name.clone(), *taint),
            V::Scalar(_) => (String::new(), false),
        }
    }
}

pub struct MilOut {
    pub mil: String,
    /// blob filename → f16 LE bytes
    pub blobs: Vec<(String, Vec<u8>)>,
    /// io ins: (mil port name, type, contig layout)
    pub inputs: Vec<(String, CTy, bool)>,
    /// trainable params: (param name, mil port name, [K,N] type, file src)
    /// — storage [N,K]; `src` is `file.safetensors::tensor` when declared.
    pub trainable: Vec<(String, String, CTy, Option<String>)>,
    /// trainable outputs: (param name, mil value name producing the update)
    pub param_outs: Vec<(String, String, CTy)>,
    /// state fields: (key `kv.k`, mil port `kv_k`, ty) — zero-init surfaces
    pub states: Vec<(String, String, CTy)>,
    /// state outputs: (mil port name, mil value producing the update, ty)
    /// — row-major [T, D] storage, unlike params' transposed convention.
    pub state_outs: Vec<(String, String, CTy)>,
    /// the output value's mil name + type
    pub output: (String, CTy),
    pub manifest: String,
    /// sha256 of the emitted mil text (artifact birthmark)
    pub mil_sha: String,
}

struct Lo {
    ops: String,
    consts: String,
    blobs: Vec<(String, Vec<u8>)>,
    env: Vec<HashMap<String, V>>,
    /// param name → emitted const name, per role
    param_emitted: HashMap<(String, &'static str), String>,
    /// param name → declared layout + qual + optional file source
    params: HashMap<String, (Qual, CTy, Option<String>)>,
    /// Typed node → emitted mil value name (for the AD tape)
    defs: HashMap<usize, String>,
    /// trainable param → mil value producing its update
    param_outs: Vec<(String, String, CTy)>,
    /// state key `kv.k` → mil port `kv_k`
    state_ports: HashMap<String, String>,
    /// state port → mil value producing its update
    state_outs: Vec<(String, String, CTy)>,
    /// accumulated gradients during a `grad` lowering
    grads: HashMap<String, V>,
    /// let-bound names → defining expr (tape for backward continuation)
    tape: HashMap<String, crate::typeck::Typed>,
    scalar_blobs: usize,
    n: usize,
}

fn mil_shape(ty: &CTy) -> String {
    match ty.dims.len() {
        0 => "[]".into(),
        1 => format!("[1, {}, 1, 1]", ty.dims[0]),
        // rank-2 rides the packed 4-D frame `[1, T, 1, D]` — the add4d /
        // outer / attn·V surface layout; a literal `[T, D]` spelling
        // drops to the flat elementwise route and currency-mismatches.
        2 => format!("[1, {}, 1, {}]", ty.dims[0], ty.dims[1]),
        _ => format!(
            "[{}]",
            ty.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

/// deterministic weight init — xorshift64 seeded from the param's
/// name and shape so the same program always compiles the same artifact.
pub fn init_weight_pub(name: &str, numel: usize) -> Vec<f32> {
    init_weight(name, numel)
}

fn init_weight(name: &str, numel: usize) -> Vec<f32> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for b in name.bytes() {
        s ^= b as u64;
        s = s.wrapping_mul(0x100000001B3);
    }
    s ^= numel as u64;
    (0..numel)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64 - 0.5) as f32
        })
        .collect()
}

impl Lo {
    fn lookup(&self, name: &str) -> Option<V> {
        for scope in self.env.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }
    fn bind(&mut self, name: &str, v: V) {
        self.env.last_mut().unwrap().insert(name.to_string(), v);
    }
    fn fresh(&mut self) -> String {
        self.n += 1;
        format!("v{}", self.n)
    }

    /// materialize a scalar into a broadcast const blob of `elems` copies.
    fn bcast(&mut self, x: f64, elems: usize) -> String {
        self.scalar_blobs += 1;
        let name = format!("__bc{}", self.scalar_blobs);
        let fname = format!("{name}.bin");
        self.blobs
            .push((fname.clone(), f16_bytes(&vec![x as f32; elems])));
        let _ = writeln!(
            self.consts,
            "        tensor<fp16, [1, {elems}, 1, 1]> {name} = const()[val = tensor<fp16, [1, {elems}, 1, 1]>(BLOBFILE(path = string(\"{fname}\"), offset = uint64(0)))];"
        );
        name
    }

    /// emit (or reuse) a param as a mil const; returns its mil name.
    fn param_ref(&mut self, name: &str, role: &'static str) -> CResult<(String, CTy, bool)> {
        let (qual, ty, src) = self
            .params
            .get(name)
            .cloned()
            .ok_or(())
            .map_err(|_| Diag { msg: format!("{name} is not a param"), line: 0, col: 0 })?;
        if let Some(n) = self.param_emitted.get(&(name.to_string(), role)) {
            return Ok((n.clone(), ty, qual == Qual::Sealed));
        }
        let mil_name = name.to_string();
        // real weights via `@ "file::tensor"`; deterministic init otherwise
        let init = |expect: &[usize]| -> CResult<Vec<f32>> {
            crate::weights::resolve(name, Some(expect), src.as_deref(), init_weight)
                .map_err(|m| Diag { msg: m, line: 0, col: 0 })
        };
        match role {
            // [N, K] matmul weight for `x @ w` — blob is stored transposed
            // relative to the declared [K, N] layout.
            "mat" => {
                if ty.dims.len() != 2 {
                    return diag(format!("param {name} is not rank-2"), 0, 0);
                }
                let (k, n) = (ty.dims[0] as usize, ty.dims[1] as usize);
                // file tensors are [out, in] = [N, K] row-major — the
                // blob's storage layout exactly, no transpose needed.
                let w = init(&[n, k])?;
                let fname = format!("{name}.bin");
                self.blobs.push((fname.clone(), f16_bytes(&w)));
                let _ = writeln!(
                    self.consts,
                    "        tensor<fp16, [{n}, {k}]> {mil_name} = const()[val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string(\"{fname}\"), offset = uint64(0)))];"
                );
            }
            // [1, C, 1, 1] vector for elementwise / norm weights.
            "vec" => {
                let c = *ty.dims.first().unwrap_or(&1) as usize;
                let w = init(&[c])?;
                let fname = format!("{name}.bin");
                self.blobs.push((fname.clone(), f16_bytes(&w)));
                let _ = writeln!(
                    self.consts,
                    "        tensor<fp16, [1, {c}, 1, 1]> {mil_name} = const()[val = tensor<fp16, [1, {c}, 1, 1]>(BLOBFILE(path = string(\"{fname}\"), offset = uint64(0)))];"
                );
            }
            _ => unreachable!(),
        }
        self.param_emitted
            .insert((name.to_string(), role), mil_name.clone());
        Ok((mil_name, ty, qual == Qual::Sealed))
    }

    fn emit_op(&mut self, ty: &CTy, line: String, taint: bool) -> V {
        let name = self.fresh();
        let _ = writeln!(
            self.ops,
            "        tensor<fp16, {}> {name} = {line}[name = string(\"{name}\")];",
            mil_shape(ty)
        );
        V::Named {
            name,
            ty: ty.clone(),
            taint,
        }
    }

    /// σ(x) as a mil name. The apply-LUT kernel is only probed to
    /// n=1024 — larger widths decompose into a clamped odd polynomial
    /// (deg-13 minimax on [−6,6], max err 4e-4 — under fp16 noise) over
    /// minimum/maximum/mul/add, which the ew family covers at any n.
    fn sigmoid_of(&mut self, x: &str, ty: &CTy) -> String {
        let elems = ty.numel().max(1) as usize;
        if elems <= 1024 {
            return self
                .emit_op(ty, format!("sigmoid(x = {x})"), false)
                .name()
                .0;
        }
        const C: [f64; 7] = [
            0.249020496788,
            -0.019419997668,
            0.001468364205,
            -0.000077364380,
            0.000002501550,
            -0.000000043871,
            0.000000000317,
        ];
        let c6 = self.bcast(6.0, elems);
        let cn6 = self.bcast(-6.0, elems);
        let s = self
            .emit_op(ty, format!("minimum(x = {x}, y = {c6})"), false)
            .name()
            .0;
        let s = self
            .emit_op(ty, format!("maximum(x = {s}, y = {cn6})"), false)
            .name()
            .0;
        let s2 = self
            .emit_op(ty, format!("mul(x = {s}, y = {s})"), false)
            .name()
            .0;
        // Horner in u = s²: h = c0 + u(c1 + u(… + u·c6))
        let mut h = self.bcast(C[6], elems);
        for c in C[..6].iter().rev() {
            let m = self
                .emit_op(ty, format!("mul(x = {s2}, y = {h})"), false)
                .name()
                .0;
            let cb = self.bcast(*c, elems);
            h = self
                .emit_op(ty, format!("add(x = {cb}, y = {m})"), false)
                .name()
                .0;
        }
        let sh = self
            .emit_op(ty, format!("mul(x = {s}, y = {h})"), false)
            .name()
            .0;
        let half = self.bcast(0.5, elems);
        self.emit_op(ty, format!("add(x = {sh}, y = {half})"), false)
            .name()
            .0
    }

    /// emit with an explicit mil shape string (packed 4-D weights etc.).
    fn emit_shaped(&mut self, shape: &str, line: String, ty: &CTy, taint: bool) -> V {
        let name = self.fresh();
        let _ = writeln!(
            self.ops,
            "        tensor<fp16, {shape}> {name} = {line}[name = string(\"{name}\")];"
        );
        V::Named {
            name,
            ty: ty.clone(),
            taint,
        }
    }

    /// the mil name of an already-emitted value (forward values in AD).
    fn fwd(&mut self, e: &Typed) -> CResult<V> {
        if let TExpr::Ident(s) = &e.e {
            if let Some(v) = self.lookup(s) {
                return Ok(v);
            }
        }
        if let Some(n) = self.defs.get(&(e as *const Typed as usize)) {
            return Ok(V::Named {
                name: n.clone(),
                ty: e.ty.clone(),
                taint: false,
            });
        }
        self.expr(e)
    }

    /// mil name + taint for a forward value in backward expressions.
    fn fwd_name(&mut self, e: &Typed) -> CResult<(String, bool)> {
        let v = self.fwd(e)?;
        self.named(v, &e.ty)
    }

    /// force a value to a named mil tensor (materializing scalars).
    fn named(&mut self, v: V, ty: &CTy) -> CResult<(String, bool)> {
        match v {
            V::Named { name, taint, .. } => Ok((name, taint)),
            V::Scalar(x) => Ok((self.bcast(x, ty.numel().max(1) as usize), false)),
        }
    }

    fn expr(&mut self, e: &Typed) -> CResult<V> {
        let v = self.expr_inner(e)?;
        if let V::Named { name, .. } = &v {
            self.defs.insert(e as *const Typed as usize, name.clone());
        }
        Ok(v)
    }

    fn expr_inner(&mut self, e: &Typed) -> CResult<V> {
        match &e.e {
            TExpr::Lit(f) => Ok(V::Scalar(*f)),
            TExpr::Int(n) => Ok(V::Scalar(*n as f64)),
            TExpr::Ident(s) => {
                if let Some(v) = self.lookup(s) {
                    Ok(v)
                } else if self.params.contains_key(s) {
                    // param used positionally — pick role by context-free
                    // default: vectors and 1-D params are `vec`.
                    let ty = &self.params[s].1;
                    let role = if ty.dims.len() == 2 { "mat" } else { "vec" };
                    let (n, ty, taint) = self.param_ref(s, role)?;
                    Ok(V::Named { name: n, ty, taint })
                } else {
                    diag(format!("unbound {s} at lower"), 0, 0)
                }
            }
            TExpr::Unary(UnOp::Neg, a) => {
                let a = self.expr(a)?;
                match a {
                    V::Scalar(x) => Ok(V::Scalar(-x)),
                    named => {
                        let (an, taint) = self.named(named.clone(), &e.ty)?;
                        let neg = self.bcast(-1.0, e.ty.numel().max(1) as usize);
                        Ok(self.emit_op(&e.ty, format!("mul(x = {an}, y = {neg})"), taint))
                    }
                }
            }
            TExpr::Bin(op, a, b) => self.binop(*op, a, b, &e.ty),
            TExpr::Transpose(_a) => {
                // standalone .T — only survives into v1 mil inside `x @ w.T`,
                // which `binop` handles without materializing.
                diag("`.T` is only supported as the right side of `@` in v1", 0, 0)
            }
            TExpr::Call(name, args, named) => self.call(name, args, named, &e.ty),
            TExpr::Grad { loss, wrt } => self.grad(loss, wrt),
            TExpr::Index(..) => diag("indexing lands with state/KV in M5", 0, 0),
            TExpr::Field(base, f) => {
                // `kv.k` — bound under the dotted key; reads see the
                // latest StateUpdate value.
                if let TExpr::Ident(s) = &base.e {
                    let key = format!("{s}.{f}");
                    return self
                        .lookup(&key)
                        .ok_or(())
                        .map_err(|_| Diag { msg: format!("unbound state field {key}"), line: 0, col: 0 });
                }
                diag("field access only on state blocks", 0, 0)
            }
            TExpr::Block(stmts, ret) => {
                self.env.push(HashMap::new());
                for s in stmts {
                    self.stmt(s)?;
                }
                let v = self.expr(ret)?;
                self.env.pop();
                Ok(v)
            }
        }
    }

    fn binop(&mut self, op: BinOp, a: &Typed, b: &Typed, ty: &CTy) -> CResult<V> {
        match op {
            BinOp::MatMul => {
                let av = self.expr(a)?;
                let (an, ataint) = self.named(av, &a.ty)?;
                if a.ty.dims.len() != 1 {
                    return diag("v1 matmul: lhs must be a vector (batched lands in M2)", 0, 0);
                }
                let _k = a.ty.dims[0];
                let n = b.ty.dims[1];
                // rhs: `w` or `w.T` (param, [N,K] storage, transpose_y),
                // `kv.v` (state, [T,D] packed, transpose_x — attn·V), or
                // `kv.k.T` (state, transpose_y — avw scores over rows).
                // (operand, flag) — flag is transpose_y for param matvecs
                // and transposed state fields, transpose_x for cache
                // contractions `s @ kv.v`.
                let (wn, flag) = match &b.e {
                    TExpr::Field(base, f) => {
                        let port = match &base.e {
                            TExpr::Ident(s) => self
                                .state_ports
                                .get(&format!("{s}.{f}"))
                                .cloned(),
                            _ => None,
                        }
                        .ok_or(())
                        .map_err(|_| Diag { msg: "matmul rhs state field not found".into(), line: 0, col: 0 })?;
                        (port, "transpose_x")
                    }
                    TExpr::Transpose(inner) => match &inner.e {
                        TExpr::Field(base, f) => {
                            let port = match &base.e {
                                TExpr::Ident(s) => self
                                    .state_ports
                                    .get(&format!("{s}.{f}"))
                                    .cloned(),
                                _ => None,
                            }
                            .ok_or(())
                            .map_err(|_| Diag { msg: "matmul rhs state field not found".into(), line: 0, col: 0 })?;
                            (port, "transpose_y")
                        }
                        _ => (self.w_name(b)?, "transpose_y"),
                    },
                    _ => (self.w_name(b)?, "transpose_y"),
                };
                let pname = match &b.e {
                    TExpr::Ident(s) => s.clone(),
                    TExpr::Transpose(inner) => match &inner.e {
                        TExpr::Ident(s) => s.clone(),
                        _ => String::new(),
                    },
                    _ => String::new(),
                };
                let wtaint = self
                    .params
                    .get(&pname)
                    .map(|(q, _, _)| *q == Qual::Sealed)
                    .unwrap_or(false);
                let out = CTy {
                    dt: DType::F16,
                    dims: vec![n],
                };
                let name = self.fresh();
                let _ = writeln!(
                    self.ops,
                    "        tensor<fp16, [1, 1, 1, {n}]> {name} = matmul(x = {an}, y = {wn}, {flag} = __bt)[name = string(\"{name}\")];"
                );
                Ok(V::Named {
                    name,
                    ty: out,
                    taint: ataint || wtaint,
                })
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                let elems = ty.numel().max(1) as usize;
                let av = self.expr(a)?;
                if op == BinOp::Div {
                    // only scalar divisors in v1 — `x / s` → `x * (1/s)`
                    if let V::Scalar(s) = self.expr(b)? {
                        if s == 0.0 {
                            return diag("division by zero", 0, 0);
                        }
                        let (an, taint) = self.named(av, &a.ty)?;
                        let recip = self.bcast(1.0 / s, elems);
                        return Ok(self.emit_op(ty, format!("mul(x = {an}, y = {recip})"), taint));
                    }
                    return diag("tensor ÷ tensor lands in M2 — divide by scalars only", 0, 0);
                }
                let bv = self.expr(b)?;
                let (an, at) = self.named(av, ty)?;
                let (bn, bt) = self.named(bv, ty)?;
                let opname = match op {
                    BinOp::Add => "add",
                    BinOp::Sub => "sub",
                    BinOp::Mul => "mul",
                    _ => unreachable!(),
                };
                // mil arg order matters for sub: (x - y)
                Ok(self.emit_op(ty, format!("{opname}(x = {an}, y = {bn})"), at || bt))
            }
            _ => diag("comparison ops land with control flow in M4", 0, 0),
        }
    }

    fn call(&mut self, name: &str, args: &[Typed], named: &[(String, Typed)], ty: &CTy) -> CResult<V> {
        if name == "__ret" {
            return self.expr(&args[0]);
        }
        let vals: Vec<V> = args.iter().map(|a| self.expr(a)).collect::<CResult<_>>()?;
        let taint = vals.iter().any(|v| matches!(v, V::Named { taint: true, .. }));
        let arg = |i: usize, vals: &[V], lo: &mut Lo| -> CResult<String> {
            let v = vals[i].clone();
            let ty = args[i].ty.clone();
            Ok(lo.named(v, &ty)?.0)
        };
        match name {
            "outer" => {
                // packed row×col — the cache-write / weight-grad shape.
                // `matmul(x=[1,T,1,1] strided, y=D-vec, transpose_x)` →
                // emit_outer_t → Packed(t, d).
                let an = arg(0, &vals, self)?;
                let bn = arg(1, &vals, self)?;
                let (t, d) = (args[0].ty.dims[0], args[1].ty.dims[0]);
                Ok(self.emit_shaped(
                    &format!("[1, {t}, 1, {d}]"),
                    format!("matmul(x = {an}, y = {bn}, transpose_x = __bt)"),
                    ty,
                    taint,
                ))
            }
            "relu" | "sigmoid" | "tanh" | "silu" | "softmax" => {
                let x = arg(0, &vals, self)?;
                if name == "relu" && ty.numel() != 64 {
                    return diag(
                        format!("relu n={} off skeleton (n=64 only in v1)", ty.numel()),
                        0,
                        0,
                    );
                }
                match name {
                    "sigmoid" if ty.numel() > 1024 => {
                        let sn = self.sigmoid_of(&x, ty);
                        Ok(V::Named { name: sn, ty: ty.clone(), taint })
                    }
                    "silu" if ty.numel() > 1024 => {
                        let sn = self.sigmoid_of(&x, ty);
                        Ok(self.emit_op(ty, format!("mul(x = {x}, y = {sn})"), taint))
                    }
                    _ => Ok(self.emit_op(ty, format!("{name}(x = {x})"), taint)),
                }
            }
            "rmsnorm" => {
                let x = arg(0, &vals, self)?;
                // w: param gamma → vector const, or any named value
                let w = match &vals[1] {
                    V::Named { name, .. } if self.params.contains_key(name) => {
                        self.param_ref(name, "vec")?.0
                    }
                    v => self.named(v.clone(), &args[1].ty)?.0,
                };
                let eps = named
                    .iter()
                    .find(|(k, _)| k == "eps")
                    .and_then(|(_, v)| match v.e {
                        TExpr::Lit(f) => Some(f),
                        _ => None,
                    })
                    .unwrap_or(1e-5);
                Ok(self.emit_op(
                    ty,
                    format!("rms_norm(x = {x}, w = {w}, eps = fp16({eps}))"),
                    taint,
                ))
            }
            "mse" => diag(
                "mse produces a host scalar — it only lowers as a `grad` target",
                0,
                0,
            ),
            _ => diag(format!("builtin {name} has no v1 lowering"), 0, 0),
        }
    }

    /// does this subtree reach a param (through idents and let-defs)?
    fn needs_grad(&self, e: &Typed, depth: usize) -> bool {
        if depth > 32 {
            return false;
        }
        match &e.e {
            TExpr::Ident(s) => {
                self.params.contains_key(s)
                    || self
                        .tape
                        .get(s)
                        .map(|t| self.needs_grad(t, depth + 1))
                        .unwrap_or(false)
            }
            TExpr::Unary(_, a) | TExpr::Transpose(a) => self.needs_grad(a, depth + 1),
            TExpr::Bin(_, a, b) => {
                self.needs_grad(a, depth + 1) || self.needs_grad(b, depth + 1)
            }
            TExpr::Call(_, args, named) => {
                args.iter().any(|a| self.needs_grad(a, depth + 1))
                    || named.iter().any(|(_, a)| self.needs_grad(a, depth + 1))
            }
            TExpr::Index(a, _) | TExpr::Field(a, _) => self.needs_grad(a, depth + 1),
            TExpr::Block(stmts, ret) => {
                // let-bound exprs inside the block may carry params
                stmts.iter().any(|s| match s {
                    TStmt::Let { expr, .. } => self.needs_grad(expr, depth + 1),
                    TStmt::Assign { expr, .. } => self.needs_grad(expr, depth + 1),
                    TStmt::ParamUpdate { grad, .. } => self.needs_grad(grad, depth + 1),
                    TStmt::StateUpdate { expr, .. } => self.needs_grad(expr, depth + 1),
                    TStmt::Expr(t) => self.needs_grad(t, depth + 1),
                }) || self.needs_grad(ret, depth + 1)
            }
            TExpr::Grad { .. } => false,
            _ => false,
        }
    }

    /// `grad(loss, w)` — tape-based reverse mode. The loss's forward
    /// values are emitted normally; the backward pass emits the same
    /// op classes train.mil used: `transpose_x` matmuls for input-grads
    /// and outer products, elementwise for activation derivatives.
    /// `scale` pre-scales the adjoint seed — `w -= lr*grad` folds the
    /// negated learning rate into the seed before the outer product,
    /// so the update lands as a single packed add (train.mil's ndy).
    fn grad(&mut self, loss: &Typed, wrt: &str) -> CResult<V> {
        self.grad_scaled(loss, wrt, 1.0)
    }

    fn grad_scaled(&mut self, loss: &Typed, wrt: &str, scale: f64) -> CResult<V> {
        self.env.push(HashMap::new());
        self.tape.clear();
        self.grads.clear();
        // forward: emit the loss block's lets, recording each name's
        // defining expr on the tape for backward continuation.
        let terminal: &Typed = match &loss.e {
            TExpr::Block(stmts, ret) => {
                for s in stmts {
                    if let TStmt::Let { name, expr } = s {
                        let v = self.expr(expr)?;
                        self.tape.insert(name.clone(), expr.clone());
                        self.bind(name, v);
                    } else {
                        self.stmt(s)?;
                    }
                }
                ret
            }
            _ => loss,
        };
        // seed: v1 losses end in mse(pred, target). The scalar itself
        // never materializes — the seed is dL/dp = 2(p - t)/n, an
        // elementwise chain, no reduction needed.
        match &terminal.e {
            TExpr::Call(name, args, _) if name == "mse" => {
                let n = args[0].ty.numel().max(1) as usize;
                let (pn, pt) = self.fwd_name(&args[0])?;
                let (tn, _tt) = self.fwd_name(&args[1])?;
                let sub = self.emit_op(
                    &args[0].ty,
                    format!("sub(x = {pn}, y = {tn})"),
                    pt,
                );
                let sn = self.named(sub, &args[0].ty)?.0;
                let scale = self.bcast(scale * 2.0 / n as f64, n);
                let seed = self.emit_op(
                    &args[0].ty,
                    format!("mul(x = {sn}, y = {scale})"),
                    pt,
                );
                let seedn = self.named(seed.clone(), &args[0].ty)?.0;
                self.back(&args[0], &seedn)?;
            }
            _ => {
                return diag(
                    "grad: v1 losses must end in mse(pred, target)",
                    0,
                    0,
                )
            }
        }
        let out = self
            .grads
            .get(wrt)
            .cloned()
            .ok_or(())
            .map_err(|_| Diag {
                msg: format!("grad: loss does not depend on param {wrt:?}"),
                line: 0,
                col: 0,
            })?;
        self.tape.clear();
        self.env.pop();
        Ok(out)
    }

    /// accumulate an adjoint contribution for a param.
    fn acc(&mut self, name: &str, v: V) -> CResult<()> {
        match self.grads.get(name) {
            None => {
                self.grads.insert(name.to_string(), v);
                Ok(())
            }
            Some(prev) => {
                let ty = match prev {
                    V::Named { ty, .. } => ty.clone(),
                    V::Scalar(_) => CTy::scalar(DType::F16),
                };
                let (a, at) = self.named(prev.clone(), &ty)?;
                let (b, bt) = self.named(v, &ty)?;
                let sum = if ty.dims.len() == 2 {
                    let shape = format!("[1, {}, 1, {}]", ty.dims[1], ty.dims[0]);
                    self.emit_shaped(&shape, format!("add(x = {a}, y = {b})"), &ty, at || bt)
                } else {
                    self.emit_op(&ty, format!("add(x = {a}, y = {b})"), at || bt)
                };
                self.grads.insert(name.to_string(), sum);
                Ok(())
            }
        }
    }

    /// reverse-mode walk: `adj` is the mil name of dL/d(node), with
    /// layout matching node.ty.
    fn back(&mut self, e: &Typed, adj: &str) -> CResult<()> {
        let adjty = e.ty.clone();
        let adjv = || V::Named {
            name: adj.to_string(),
            ty: adjty.clone(),
            taint: false,
        };
        match &e.e {
            TExpr::Ident(s) => {
                if self.params.contains_key(s) {
                    self.acc(s, adjv())?;
                } else if let Some(def) = self.tape.get(s).cloned() {
                    self.back(&def, adj)?;
                }
            }
            TExpr::Lit(_) | TExpr::Int(_) => {}
            TExpr::Unary(UnOp::Neg, a) => {
                let neg = self.bcast(-1.0, adjty.numel().max(1) as usize);
                let v = self.emit_op(&adjty, format!("mul(x = {adj}, y = {neg})"), false);
                let vn = self.named(v, &adjty)?.0;
                self.back(a, &vn)?;
            }
            TExpr::Bin(op, a, b) => match op {
                BinOp::Add => {
                    self.back(a, adj)?;
                    self.back(b, adj)?;
                }
                BinOp::Sub => {
                    self.back(a, adj)?;
                    let neg = self.bcast(-1.0, adjty.numel().max(1) as usize);
                    let v = self.emit_op(&adjty, format!("mul(x = {adj}, y = {neg})"), false);
                    let vn = self.named(v, &adjty)?.0;
                    self.back(b, &vn)?;
                }
                BinOp::Mul => {
                    if self.needs_grad(a, 0) {
                        let (bn, _) = self.fwd_name(b)?;
                        let v = self.emit_op(&adjty, format!("mul(x = {adj}, y = {bn})"), false);
                        let vn = self.named(v, &adjty)?.0;
                        self.back(a, &vn)?;
                    }
                    if self.needs_grad(b, 0) {
                        let (an, _) = self.fwd_name(a)?;
                        let v = self.emit_op(&adjty, format!("mul(x = {adj}, y = {an})"), false);
                        let vn = self.named(v, &adjty)?.0;
                        self.back(b, &vn)?;
                    }
                }
                BinOp::Div => {
                    // only scalar divisors survive to lowering
                    if self.needs_grad(a, 0) {
                        if let TExpr::Lit(s) = b.e {
                            let r = self.bcast(1.0 / s, adjty.numel().max(1) as usize);
                            let v = self.emit_op(&adjty, format!("mul(x = {adj}, y = {r})"), false);
                            let vn = self.named(v, &adjty)?.0;
                            self.back(a, &vn)?;
                        }
                    }
                }
                BinOp::MatMul => {
                    // forward: out[N] = a[K] @ W[K,N] (storage [N,K])
                    let k = a.ty.dims[0];
                    let n = b.ty.dims[1];
                    if self.needs_grad(a, 0) {
                        // dx = adj @ Wᵀ — transpose_x matvec (attn·V route)
                        let wn = self.w_name(b)?;
                        let dty = CTy { dt: DType::F16, dims: vec![k] };
                        let v = self.emit_shaped(
                            &format!("[1, 1, 1, {k}]"),
                            format!("matmul(x = {adj}, y = {wn}, transpose_x = __bt)"),
                            &dty,
                            false,
                        );
                        let vn = self.named(v, &dty)?.0;
                        self.back(a, &vn)?;
                    }
                    // dW = outer(adj, a) — transpose_x outer (packed route)
                    let pname = self.matmul_param(b)?;
                    if let Some(p) = pname {
                        let (an, _) = self.fwd_name(a)?;
                        let dty = CTy { dt: DType::F16, dims: vec![k, n] };
                        let v = self.emit_shaped(
                            &format!("[1, {n}, 1, {k}]"),
                            format!("matmul(x = {adj}, y = {an}, transpose_x = __bt)"),
                            &dty,
                            false,
                        );
                        self.acc(&p, v)?;
                    }
                }
                _ => {}
            },
            TExpr::Call(name, args, _) => {
                let x = &args[0];
                let dty = x.ty.clone();
                let elems = dty.numel().max(1) as usize;
                match name.as_str() {
                    "sigmoid" => {
                        // dx = adj · h·(1-h), h the forward value
                        let (hn, _) = self.fwd_name(e)?;
                        let one = self.bcast(1.0, elems);
                        let omh = self.emit_op(&dty, format!("sub(x = {one}, y = {hn})"), false);
                        let omhn = self.named(omh, &dty)?.0;
                        let ds = self.emit_op(&dty, format!("mul(x = {hn}, y = {omhn})"), false);
                        let dsn = self.named(ds, &dty)?.0;
                        let dx = self.emit_op(&dty, format!("mul(x = {adj}, y = {dsn})"), false);
                        let dxn = self.named(dx, &dty)?.0;
                        self.back(x, &dxn)?;
                    }
                    "tanh" => {
                        let (tn, _) = self.fwd_name(e)?;
                        let tt = self.emit_op(&dty, format!("mul(x = {tn}, y = {tn})"), false);
                        let ttn = self.named(tt, &dty)?.0;
                        let one = self.bcast(1.0, elems);
                        let omtt =
                            self.emit_op(&dty, format!("sub(x = {one}, y = {ttn})"), false);
                        let omttn = self.named(omtt, &dty)?.0;
                        let dx = self.emit_op(&dty, format!("mul(x = {adj}, y = {omttn})"), false);
                        let dxn = self.named(dx, &dty)?.0;
                        self.back(x, &dxn)?;
                    }
                    "silu" => {
                        // d[x·σ(x)] = σ(x)·(1 + x·(1-σ(x)))
                        let (xn, _) = self.fwd_name(x)?;
                        let sn = self.sigmoid_of(&xn, &dty);
                        let one = self.bcast(1.0, elems);
                        let oms = self.emit_op(&dty, format!("sub(x = {one}, y = {sn})"), false);
                        let omsn = self.named(oms, &dty)?.0;
                        let xoms = self.emit_op(&dty, format!("mul(x = {xn}, y = {omsn})"), false);
                        let xomsn = self.named(xoms, &dty)?.0;
                        let inner =
                            self.emit_op(&dty, format!("add(x = {one}, y = {xomsn})"), false);
                        let innern = self.named(inner, &dty)?.0;
                        let d = self.emit_op(&dty, format!("mul(x = {sn}, y = {innern})"), false);
                        let dn = self.named(d, &dty)?.0;
                        let dx = self.emit_op(&dty, format!("mul(x = {adj}, y = {dn})"), false);
                        let dxn = self.named(dx, &dty)?.0;
                        self.back(x, &dxn)?;
                    }
                    "mse" => {
                        // reachable when loss was spelled inline
                        let n2 = dty.numel().max(1) as usize;
                        let (pn, _) = self.fwd_name(&args[0])?;
                        let (tn, _) = self.fwd_name(&args[1])?;
                        let sub = self.emit_op(&dty, format!("sub(x = {pn}, y = {tn})"), false);
                        let sn = self.named(sub, &dty)?.0;
                        let scale = self.bcast(2.0 / n2 as f64, n2);
                        let dx = self.emit_op(&dty, format!("mul(x = {sn}, y = {scale})"), false);
                        let dxn = self.named(dx, &dty)?.0;
                        self.back(&args[0], &dxn)?;
                    }
                    "relu" => {
                        return diag("grad through relu needs a mask op — lands in M3b", 0, 0)
                    }
                    "softmax" => {
                        return diag("grad through softmax needs a reduce — lands in M3b", 0, 0)
                    }
                    "rmsnorm" => {
                        return diag("grad through rmsnorm lands in M3b", 0, 0)
                    }
                    _ => return diag(format!("grad: no backward rule for {name}"), 0, 0),
                }
            }
            TExpr::Transpose(_) => {
                return diag("grad through .T lands in M3b", 0, 0)
            }
            TExpr::Block(..) => {
                return diag("grad: nested blocks in loss land in M3b", 0, 0)
            }
            TExpr::Grad { .. } => return diag("higher-order grad lands later", 0, 0),
            TExpr::Index(..) | TExpr::Field(..) => {}
        }
        Ok(())
    }

    /// mil name of a matmul rhs (param or param.T) — runtime name for
    /// trainables, const for frozen.
    fn w_name(&mut self, b: &Typed) -> CResult<String> {
        match &b.e {
            TExpr::Ident(s) => {
                if self.params.get(s).map(|(q, _, _)| *q) == Some(Qual::Trainable) {
                    if let Some(v) = self.lookup(s) {
                        return self.named(v, &b.ty).map(|x| x.0);
                    }
                }
                Ok(self.param_ref(s, "mat")?.0)
            }
            TExpr::Transpose(inner) => match &inner.e {
                TExpr::Ident(s) => Ok(self.param_ref(s, "mat")?.0),
                _ => diag("v1 matmul rhs must be a param or param.T", 0, 0),
            },
            _ => diag("v1 matmul rhs must be a param or param.T", 0, 0),
        }
    }

    /// the param name feeding a matmul rhs, if differentiating it is
    /// meaningful (frozen params get grads too — they're just read-only).
    fn matmul_param(&mut self, b: &Typed) -> CResult<Option<String>> {
        Ok(match &b.e {
            TExpr::Ident(s) if self.params.contains_key(s) => Some(s.clone()),
            TExpr::Transpose(inner) => match &inner.e {
                TExpr::Ident(s) if self.params.contains_key(s) => Some(s.clone()),
                _ => None,
            },
            _ => None,
        })
    }

    fn stmt(&mut self, s: &TStmt) -> CResult<()> {
        match s {
            TStmt::ParamUpdate { param, scale, grad } => {
                if let TExpr::Grad { loss, wrt } = &grad.e {
                    if wrt != param {
                        return diag(
                            format!("param update for {param} takes grad w.r.t. {wrt} — they must match"),
                            0,
                            0,
                        );
                    }
                    let dw = self.grad_scaled(loss, wrt, *scale)?;
                    let ty = grad.ty.clone();
                    let cur = self.lookup(param).ok_or(()).map_err(|_| Diag {
                        msg: format!("update to undeclared param {param}"),
                        line: 0,
                        col: 0,
                    })?;
                    let (cn, ct) = self.named(cur, &ty)?;
                    let (dn, dt) = self.named(dw, &ty)?;
                    let (k, n) = (ty.dims[0], ty.dims[1]);
                    let nv = self.emit_shaped(
                        &format!("[1, {n}, 1, {k}]"),
                        format!("add(x = {cn}, y = {dn})"),
                        &ty,
                        ct || dt,
                    );
                    if let V::Named { name: mn, .. } = &nv {
                        if let Some(e) = self.param_outs.iter_mut().find(|(p, _, _)| p == param) {
                            e.1 = mn.clone();
                        } else {
                            self.param_outs
                                .push((param.to_string(), mn.clone(), ty.clone()));
                        }
                    }
                    self.bind(param, nv);
                }
            }
            TStmt::StateUpdate { key, port, expr } => {
                // the desugared expr IS the new field value — its mil
                // name becomes the state output the host feeds back.
                let v = self.expr(expr)?;
                let (en, et) = self.named(v, &expr.ty)?;
                let ty = expr.ty.clone();
                if let Some(e) = self.state_outs.iter_mut().find(|(p, _, _)| p == port) {
                    e.1 = en.clone();
                } else {
                    self.state_outs.push((port.clone(), en.clone(), ty.clone()));
                }
                self.bind(key, V::Named { name: en, ty, taint: et });
            }
            TStmt::Let { name, expr } => {
                let v = self.expr(expr)?;
                self.bind(name, v);
            }
            TStmt::Assign { name, op, expr } => {
                let cur = self
                    .lookup(name)
                    .ok_or(())
                    .map_err(|_| Diag { msg: format!("assign to undeclared {name}"), line: 0, col: 0 })?;
                let ty = match &cur {
                    V::Named { ty, .. } => ty.clone(),
                    V::Scalar(_) => expr.ty.clone(),
                };
                let v = self.expr(expr)?;
                let (en, et) = self.named(v.clone(), &ty)?;
                if self.params.contains_key(name) {
                    // trainable param update → packed [1,N,1,K] add route
                    // (the same in-place-accumulator path train.mil used);
                    // `-=` folds the negation into the operand.
                    let (cn, ct) = self.named(cur, &ty)?;
                    let nv = match op {
                        AssignOp::Set => V::Named { name: en, ty: ty.clone(), taint: et },
                        _ => {
                            if ty.dims.len() != 2 {
                                return diag("v1 param updates are rank-2 weights", 0, 0);
                            }
                            let (k, n) = (ty.dims[0], ty.dims[1]);
                            let shape = format!("[1, {n}, 1, {k}]");
                            let (rn, rt) = match op {
                                AssignOp::Add => (en, et),
                                AssignOp::Sub => {
                                    let neg = self.bcast(-1.0, ty.numel().max(1) as usize);
                                    let v2 = self.emit_op(&ty, format!("mul(x = {en}, y = {neg})"), et);
                                    self.named(v2, &ty)?
                                }
                                _ => return diag("param updates support =, +=, -= in v1", 0, 0),
                            };
                            self.emit_shaped(
                                &shape,
                                format!("add(x = {cn}, y = {rn})"),
                                &ty,
                                ct || rt,
                            )
                        }
                    };
                    if let V::Named { name: mn, .. } = &nv {
                        if let Some(e) = self.param_outs.iter_mut().find(|(p, _, _)| p == name) {
                            e.1 = mn.clone();
                        } else {
                            self.param_outs
                                .push((name.to_string(), mn.clone(), ty.clone()));
                        }
                    }
                    self.bind(name, nv);
                    return Ok(());
                }
                let (cn, ct) = self.named(cur, &ty)?;
                let opname = match op {
                    AssignOp::Set => {
                        self.bind(name, V::Named { name: en, ty, taint: et });
                        return Ok(());
                    }
                    AssignOp::Add => "add",
                    AssignOp::Sub => "sub",
                    AssignOp::Mul => "mul",
                    AssignOp::Div => return diag("compound `/=` lands in M2", 0, 0),
                };
                let nv = self.emit_op(&ty, format!("{opname}(x = {cn}, y = {en})"), ct || et);
                self.bind(name, nv);
            }
            TStmt::Expr(e) => {
                self.expr(e)?;
            }
        }
        Ok(())
    }
}

/// scan the TIR for io-arg names used as the lhs of a matmul against a
/// trainable (runtime) weight — the avw route wants its x Contig, and
/// ext surfaces can't be repacked, so those ports must be declared
/// `[1, 1, 1, K]` instead of the strided `[1, K, 1, 1]`.
fn contig_ios(p: &TProgram) -> HashMap<String, bool> {
    let trainable: std::collections::HashSet<String> = p
        .params
        .iter()
        .filter(|t| t.qual == Qual::Trainable)
        .map(|t| t.name.clone())
        .collect();
    let statenames: std::collections::HashSet<String> = p
        .states
        .iter()
        .map(|(k, _, _)| k.split('.').next().unwrap_or("").to_string())
        .collect();
    let mut marks: HashMap<String, bool> = HashMap::new(); // arg → contig?
    fn walk(
        e: &Typed,
        trainable: &std::collections::HashSet<String>,
        statenames: &std::collections::HashSet<String>,
        marks: &mut HashMap<String, bool>,
        p: &TProgram,
    ) {
        match &e.e {
            TExpr::Bin(BinOp::MatMul, a, b) => {
                // contig iff the rhs is a runtime weight read avw-style
                // (transpose_y): trainable params and `.T`'d state fields.
                let is_train = match &b.e {
                    TExpr::Ident(s) => trainable.contains(s),
                    TExpr::Transpose(i) => match &i.e {
                        TExpr::Ident(s) => trainable.contains(s),
                        TExpr::Field(fb, _) => match &fb.e {
                            TExpr::Ident(s) => statenames.contains(s),
                            _ => false,
                        },
                        _ => false,
                    },
                    _ => false,
                };
                if let TExpr::Ident(s) = &a.e {
                    if p.arg_ports.iter().any(|(an, _)| an == s) {
                        let prev = marks.insert(s.clone(), is_train);
                        if let Some(p2) = prev {
                            if p2 != is_train {
                                marks.insert(s.clone(), true); // conflict → contig wins
                            }
                        }
                    }
                }
                walk(a, trainable, statenames, marks, p);
                walk(b, trainable, statenames, marks, p);
            }
            TExpr::Unary(_, a) | TExpr::Transpose(a) => walk(a, trainable, statenames, marks, p),
            TExpr::Bin(_, a, b) => {
                walk(a, trainable, statenames, marks, p);
                walk(b, trainable, statenames, marks, p);
            }
            TExpr::Call(_, args, named) => {
                for a in args {
                    walk(a, trainable, statenames, marks, p);
                }
                for (_, a) in named {
                    walk(a, trainable, statenames, marks, p);
                }
            }
            TExpr::Index(a, _) | TExpr::Field(a, _) => walk(a, trainable, statenames, marks, p),
            TExpr::Block(stmts, ret) => {
                for s in stmts {
                    match s {
                        TStmt::Let { expr, .. } | TStmt::Assign { expr, .. } => {
                            walk(expr, trainable, statenames, marks, p)
                        }
                        TStmt::ParamUpdate { grad, .. } => walk(grad, trainable, statenames, marks, p),
                        TStmt::StateUpdate { expr, .. } => walk(expr, trainable, statenames, marks, p),
                        TStmt::Expr(t) => walk(t, trainable, statenames, marks, p),
                    }
                }
                walk(ret, trainable, statenames, marks, p);
            }
            TExpr::Grad { loss, .. } => walk(loss, trainable, statenames, marks, p),
            _ => {}
        }
    }
    for s in &p.body {
        match s {
            TStmt::Let { expr, .. } | TStmt::Assign { expr, .. } => {
                walk(expr, &trainable, &statenames, &mut marks, p)
            }
            TStmt::ParamUpdate { grad, .. } => walk(grad, &trainable, &statenames, &mut marks, p),
            TStmt::StateUpdate { expr, .. } => walk(expr, &trainable, &statenames, &mut marks, p),
            TStmt::Expr(t) => walk(t, &trainable, &statenames, &mut marks, p),
        }
    }
    walk(&p.ret, &trainable, &statenames, &mut marks, p);
    marks
}

/// Lower a checked program into MIL text + blobs + manifest.
pub fn lower(p: &TProgram, _out_dir: &str, source_sha: &str) -> CResult<MilOut> {
    let mut lo = Lo {
        ops: String::new(),
        consts: String::new(),
        blobs: Vec::new(),
        env: vec![HashMap::new()],
        param_emitted: HashMap::new(),
        params: p
            .params
            .iter()
            .map(|t| (t.name.clone(), (t.qual, t.ty.clone(), t.src.clone())))
            .collect(),
        defs: HashMap::new(),
        param_outs: Vec::new(),
        state_ports: p
            .states
            .iter()
            .map(|(k, port, _)| (k.clone(), port.clone()))
            .collect(),
        state_outs: Vec::new(),
        grads: HashMap::new(),
        tape: HashMap::new(),
        scalar_blobs: 0,
        n: 0,
    };
    // bind main's args to io port names
    for (arg, port) in &p.arg_ports {
        let (_, _, ty) = p
            .ios
            .iter()
            .find(|(n, d, _)| n == port && *d == IoDir::In)
            .ok_or(())
            .map_err(|_| Diag { msg: format!("io {port} not found"), line: 0, col: 0 })?;
        lo.bind(
            arg,
            V::Named {
                name: port.clone(),
                ty: ty.clone(),
                taint: false,
            },
        );
    }
    // trainable params are runtime tensors, not consts: [N, K] storage
    // packed [1, N, 1, K] — the same layout train.mil's W args used, so
    // forward hits the avw route and backward hits attn·V/outer.
    for tp in &p.params {
        if tp.qual == Qual::Trainable {
            lo.bind(
                &tp.name,
                V::Named {
                    name: tp.name.clone(),
                    ty: tp.ty.clone(),
                    taint: false,
                },
            );
        }
    }
    // state fields ride as runtime tensors — packed [1, T, 1, D] like
    // trainable weights, bound under their dotted keys so `kv.k` reads
    // see (and `StateUpdate` rebinds) the current value.
    for (key, port, ty) in &p.states {
        lo.bind(
            key,
            V::Named {
                name: port.clone(),
                ty: ty.clone(),
                taint: false,
            },
        );
    }
    // bool const for transpose_y / transpose_x
    lo.consts
        .push_str("        tensor<bool, []> __bt = const()[val = bool(true)];\n");
    for s in &p.body {
        lo.stmt(s)?;
    }
    let ret_v = lo.expr(&p.ret)?;
    let out_ty = p
        .ios
        .iter()
        .find(|(_, d, _)| *d == IoDir::Out)
        .map(|(_, _, t)| t.clone())
        .unwrap_or_else(|| p.ret.ty.clone());
    let (ret_name, ret_taint) = lo.named(ret_v, &out_ty)?;
    if ret_taint {
        return diag("sealed value reaches the io boundary — constitution holds", 0, 0);
    }
    // func signature: io ins + trainable params as runtime tensors
    let marks = contig_ios(p);
    let mut ins: Vec<String> = Vec::new();
    for (n, _d, t) in p.ios.iter().filter(|(_, d, _)| *d == IoDir::In) {
        // contig-declared port if any arg bound to it lhs-feeds a
        // trainable matmul — the avw route reads x as a packed vector.
        let arg = p
            .arg_ports
            .iter()
            .find(|(_, port)| port == n)
            .map(|(a, _)| a.clone());
        let contig = arg.map(|a| marks.get(&a).copied().unwrap_or(false)).unwrap_or(false);
        let shape = if contig && t.dims.len() == 1 {
            format!("[1, 1, 1, {}]", t.dims[0])
        } else {
            mil_shape(t)
        };
        ins.push(format!("tensor<fp16, {shape}> {n}"));
    }
    let mut trainable = Vec::new();
    for tp in &p.params {
        if tp.qual == Qual::Trainable {
            let shape = if tp.ty.dims.len() == 2 {
                format!("[1, {}, 1, {}]", tp.ty.dims[1], tp.ty.dims[0])
            } else {
                mil_shape(&tp.ty)
            };
            ins.push(format!("tensor<fp16, {shape}> {}", tp.name));
            trainable.push((tp.name.clone(), tp.name.clone(), tp.ty.clone(), tp.src.clone()));
        }
    }
    // state fields: packed [1, T, 1, D] runtime tensors
    for (_, port, ty) in &p.states {
        let shape = if ty.dims.len() == 2 {
            format!("[1, {}, 1, {}]", ty.dims[0], ty.dims[1])
        } else {
            mil_shape(ty)
        };
        ins.push(format!("tensor<fp16, {shape}> {port}"));
    }
    let mut outs = vec![ret_name.clone()];
    outs.extend(lo.param_outs.iter().map(|(_, mn, _)| mn.clone()));
    outs.extend(lo.state_outs.iter().map(|(_, mn, _)| mn.clone()));
    let mil = format!(
        "program(1.3)\n{{\n    func main<ios18>({}) {{\n{}{}\n    }} -> ({});\n}}\n",
        ins.join(", "),
        lo.consts,
        lo.ops,
        outs.join(", ")
    );
    let mil_sha = sha256_hex(mil.as_bytes());
    // provenance manifest — the artifact's birth certificate.
    let mut manifest = String::from("{\n");
    let _ = writeln!(manifest, "  \"language\": \"coyote 0.1\",");
    let _ = writeln!(manifest, "  \"source_sha256\": \"{source_sha}\",");
    let _ = writeln!(manifest, "  \"mil_sha256\": \"{mil_sha}\",");
    let _ = writeln!(
        manifest,
        "  \"invariants\": [{}],",
        p.invariants.iter().map(|i| format!("\"{i}\"")).collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(
        manifest,
        "  \"caps\": [{}],",
        p.caps.iter().map(|i| format!("\"{i}\"")).collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(
        manifest,
        "  \"provenance\": {{{}}},",
        p.provenance
            .iter()
            .map(|(k, v)| format!("\"{k}\": \"{v}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        manifest,
        "  \"io\": [{}],",
        p.ios
            .iter()
            .map(|(n, d, t)| format!(
                "{{\"name\": \"{n}\", \"dir\": \"{}\", \"shape\": [{}]}}",
                if *d == IoDir::In { "in" } else { "out" },
                t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        manifest,
        "  \"params\": [{}],",
        p.params
            .iter()
            .map(|TParam { name, qual, ty, src }| format!(
                "{{\"name\": \"{name}\", \"qual\": \"{qual:?}\", \"shape\": [{}]{}}}",
                ty.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
                src.as_ref()
                    .map(|s| format!(", \"src\": \"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")))
                    .unwrap_or_default(),
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // trainable plumbing for the host feedback loop
    let _ = writeln!(
        manifest,
        "  \"param_updates\": {{{}}}",
        lo.param_outs
            .iter()
            .map(|(pn, mn, _)| format!("\"{pn}\": \"{mn}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    manifest.push_str("}\n");
    Ok(MilOut {
        mil,
        blobs: lo.blobs,
        inputs: p
            .ios
            .iter()
            .filter(|(_, d, _)| *d == IoDir::In)
            .map(|(n, _, t)| {
                let arg = p
                    .arg_ports
                    .iter()
                    .find(|(_, port)| port == n)
                    .map(|(a, _)| a.clone());
                let contig = arg
                    .map(|a| marks.get(&a).copied().unwrap_or(false))
                    .unwrap_or(false);
                (n.clone(), t.clone(), contig)
            })
            .collect(),
        trainable,
        param_outs: lo.param_outs,
        states: p
            .states
            .iter()
            .map(|(k, port, ty)| (k.clone(), port.clone(), ty.clone()))
            .collect(),
        state_outs: lo.state_outs,
        output: (ret_name, out_ty),
        manifest,
        mil_sha,
    })
}

/// tiny sha256 — hex digest. (Self-contained so the driver does not
/// depend on mil_spec's private module layout.)
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
