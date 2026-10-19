//! Typechecker + monomorphizer.
//!
//! Walks `main`, binds generic dims at each callsite, inlines user-fn calls
//! into `TExpr::Block`s, and produces a fully concrete `TProgram` — every
//! tensor type resolved to literal dims. This is where shape mismatches die.

use crate::ast::*;
use crate::{diag, CResult, Diag};
use std::collections::HashMap;

/// Concrete type: dtype + literal dims.
#[derive(Debug, Clone, PartialEq)]
pub struct CTy {
    pub dt: DType,
    pub dims: Vec<i64>,
}

impl CTy {
    pub fn scalar(dt: DType) -> Self {
        Self { dt, dims: vec![] }
    }
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    pub fn numel(&self) -> i64 {
        self.dims.iter().product()
    }
    pub fn show(&self) -> String {
        if self.dims.is_empty() {
            format!("{:?}", self.dt).to_lowercase()
        } else {
            format!(
                "{:?}[{}]",
                self.dt,
                self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
            )
            .to_lowercase()
        }
    }
}

#[derive(Debug, Clone)]
pub enum TIndex {
    Lit(i64),
    Range(i64, i64),
}

#[derive(Debug, Clone)]
pub enum TExpr {
    Lit(f64),
    Int(i64),
    Ident(String),
    Unary(UnOp, Box<Typed>),
    Bin(BinOp, Box<Typed>, Box<Typed>),
    Transpose(Box<Typed>),
    /// builtins only — user fns are inlined into `Block`.
    Call(String, Vec<Typed>, Vec<(String, Typed)>),
    Index(Box<Typed>, Vec<TIndex>),
    Field(Box<Typed>, String),
    /// inlined user-fn body.
    Block(Vec<TStmt>, Box<Typed>),
    /// `grad(loss_expr, wrt_param)` — reverse-mode AD target.
    Grad { loss: Box<Typed>, wrt: String },
}

#[derive(Debug, Clone)]
pub struct Typed {
    pub e: TExpr,
    pub ty: CTy,
}

#[derive(Debug, Clone)]
pub enum TStmt {
    Let { name: String, expr: Typed },
    Assign { name: String, op: AssignOp, expr: Typed },
    /// `w op= s * grad(loss, w)` — the SGD step, folded: the seed is
    /// pre-scaled so the update lowers to a single packed add.
    ParamUpdate { param: String, scale: f64, grad: Typed },
    /// `kv.k = e` / `kv.k += e` / `kv.k[pos] = e` — a state field takes
    /// a new value; the result becomes an output the host feeds back.
    /// `key` is `kv.k`, `port` the mil arg `kv_k`.
    StateUpdate { key: String, port: String, expr: Typed },
    Expr(Typed),
}

#[derive(Debug, Clone)]
pub struct TParam {
    pub name: String,
    pub qual: Qual,
    pub ty: CTy,
    /// `@ "file.safetensors::tensor"` source, if declared
    pub src: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TProgram {
    pub ios: Vec<(String, IoDir, CTy)>,
    pub params: Vec<TParam>,
    pub invariants: Vec<String>,
    pub caps: Vec<String>,
    pub provenance: Vec<(String, String)>,
    /// main arg name → io port name, in signature order.
    pub arg_ports: Vec<(String, String)>,
    /// typed body of `main`, all calls inlined.
    pub body: Vec<TStmt>,
    pub ret: Typed,
    /// state fields as runtime tensors: (key `kv.k`, port `kv_k`, ty)
    pub states: Vec<(String, String, CTy)>,
}

type Env = HashMap<String, CTy>;
type Dims = HashMap<String, i64>;

struct Ck {
    fns: HashMap<String, Sp<Fn>>,
    params: HashMap<String, (Qual, Ty, Option<String>)>,
    states: HashMap<String, Vec<(String, Ty)>>,
    ios: Vec<(String, IoDir, CTy)>,
}

fn bind_dim(dims: &mut Dims, d: &Dim, v: i64, l: usize, c: usize) -> CResult<()> {
    match d {
        Dim::Lit(n) => {
            if *n != v {
                diag(format!("dim mismatch: expected {n}, found {v}"), l, c)
            } else {
                Ok(())
            }
        }
        Dim::Name(s) => {
            if let Some(&prev) = dims.get(s) {
                if prev != v {
                    return diag(format!("dim {s} bound to both {prev} and {v}"), l, c);
                }
            } else {
                dims.insert(s.clone(), v);
            }
            Ok(())
        }
    }
}

fn resolve_ty(dims: &Dims, ty: &Ty, l: usize, c: usize) -> CResult<CTy> {
    let mut out = Vec::new();
    for d in &ty.dims {
        match d {
            Dim::Lit(n) => out.push(*n),
            Dim::Name(s) => out.push(
                *dims
                    .get(s)
                    .ok_or(())
                    .map_err(|_| Diag { msg: format!("unbound dim {s}"), line: l, col: c })?,
            ),
        }
    }
    Ok(CTy { dt: ty.dt, dims: out })
}

/// unify a declared (possibly generic) arg type with a concrete arg type.
fn unify(dims: &mut Dims, decl: &Ty, act: &CTy, l: usize, c: usize) -> CResult<()> {
    if decl.dims.len() != act.dims.len() {
        return diag(
            format!("rank mismatch: declared {:?}[{}] vs actual {}", decl.dt, decl.dims.len(), act.show()),
            l,
            c,
        );
    }
    for (d, &v) in decl.dims.iter().zip(&act.dims) {
        bind_dim(dims, d, v, l, c)?;
    }
    Ok(())
}

fn is_numeric(dt: DType) -> bool {
    matches!(dt, DType::F16 | DType::F32 | DType::I32 | DType::Q4)
}

impl Ck {
    fn expr(
        &mut self,
        e: &Sp<Expr>,
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
    ) -> CResult<Typed> {
        if depth > 64 {
            return diag("inlining depth exceeded — recursive fns are not supported", e.line, e.col);
        }
        let (l, c) = (e.line, e.col);
        match &e.v {
            Expr::Lit(f) => Ok(Typed {
                e: TExpr::Lit(*f),
                ty: CTy::scalar(DType::F32),
            }),
            Expr::Int(n) => Ok(Typed {
                e: TExpr::Int(*n),
                ty: CTy::scalar(DType::I32),
            }),
            Expr::Ident(s) => {
                if let Some(ty) = env.get(s) {
                    Ok(Typed {
                        e: TExpr::Ident(s.clone()),
                        ty: ty.clone(),
                    })
                } else if let Some((_, ty, _)) = self.params.get(s) {
                    Ok(Typed {
                        e: TExpr::Ident(s.clone()),
                        ty: resolve_ty(dims, ty, l, c)?,
                    })
                } else {
                    diag(format!("unknown name {s:?}"), l, c)
                }
            }
            Expr::Unary(op, a) => {
                let a = self.expr(a, env, dims, depth)?;
                if !is_numeric(a.ty.dt) {
                    return diag("unary op on non-numeric", l, c);
                }
                let ty = a.ty.clone();
                Ok(Typed {
                    e: TExpr::Unary(*op, Box::new(a)),
                    ty,
                })
            }
            Expr::Bin(op, a, b) => {
                let a = self.expr(a, env, dims, depth)?;
                let b = self.expr(b, env, dims, depth)?;
                let ty = match op {
                    BinOp::MatMul => self.matmul_ty(&a.ty, &b.ty, l, c)?,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        self.elementwise_ty(&a.ty, &b.ty, l, c)?
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        CTy::scalar(DType::I32)
                    }
                };
                Ok(Typed {
                    e: TExpr::Bin(*op, Box::new(a), Box::new(b)),
                    ty,
                })
            }
            Expr::Transpose(a) => {
                let a = self.expr(a, env, dims, depth)?;
                if a.ty.dims.len() != 2 {
                    return diag(
                        format!(".T requires rank-2, found {}", a.ty.show()),
                        l,
                        c,
                    );
                }
                let ty = CTy {
                    dt: a.ty.dt,
                    dims: vec![a.ty.dims[1], a.ty.dims[0]],
                };
                Ok(Typed {
                    e: TExpr::Transpose(Box::new(a)),
                    ty,
                })
            }
            Expr::Call(name, pos, named) => self.call(name, pos, named, env, dims, depth, l, c),
            Expr::Index(base, idxs) => {
                let base = self.expr(base, env, dims, depth)?;
                if idxs.len() > base.ty.dims.len() {
                    return diag("too many indices", l, c);
                }
                let mut tidx = Vec::new();
                let mut out_dims = Vec::new();
                for (i, ix) in idxs.iter().enumerate() {
                    let dim = base.ty.dims[i];
                    match &ix.v {
                        Index::Expr(ie) => {
                            let v = self.const_int(ie, dims)?;
                            if v < 0 || v >= dim {
                                return diag(format!("index {v} out of bounds for dim {dim}"), ix.line, ix.col);
                            }
                            tidx.push(TIndex::Lit(v));
                        }
                        Index::Range(lo, hi) => {
                            let lv = match lo {
                                Some(e) => self.const_int(e, dims)?,
                                None => 0,
                            };
                            let hv = match hi {
                                Some(e) => self.const_int(e, dims)?,
                                None => dim,
                            };
                            if lv < 0 || hv > dim || lv >= hv {
                                return diag(format!("range {lv}..{hv} invalid for dim {dim}"), ix.line, ix.col);
                            }
                            tidx.push(TIndex::Range(lv, hv));
                            out_dims.push(hv - lv);
                        }
                    }
                }
                // remaining (unspecified) dims carry through
                out_dims.extend_from_slice(&base.ty.dims[idxs.len()..]);
                let ty = CTy {
                    dt: base.ty.dt,
                    dims: out_dims,
                };
                Ok(Typed {
                    e: TExpr::Index(Box::new(base), tidx),
                    ty,
                })
            }
            Expr::Field(base, f) => {
                // `kv.k` — a bare ident naming a state block. Checked
                // before `self.expr` because `kv` itself isn't a value.
                if let Expr::Ident(s) = &base.v {
                    if let Some(fields) = self.states.get(s) {
                        for (fname, fty) in fields {
                            if fname == &f.v {
                                return Ok(Typed {
                                    e: TExpr::Field(
                                        Box::new(Typed {
                                            e: TExpr::Ident(s.clone()),
                                            ty: CTy::scalar(DType::I32),
                                        }),
                                        f.v.clone(),
                                    ),
                                    ty: resolve_ty(dims, fty, l, c)?,
                                });
                            }
                        }
                        return diag(format!("state {s} has no field {}", f.v), l, c);
                    }
                }
                let _b = self.expr(base, env, dims, depth)?;
                diag("field access only on state blocks", l, c)
            }
        }
    }

    fn const_int(&mut self, e: &Sp<Expr>, dims: &Dims) -> CResult<i64> {
        match &e.v {
            Expr::Int(n) => Ok(*n),
            Expr::Ident(s) => dims
                .get(s)
                .copied()
                .ok_or(())
                .map_err(|_| Diag { msg: format!("index {s} is not a compile-time dim"), line: e.line, col: e.col }),
            Expr::Unary(UnOp::Neg, a) => Ok(-self.const_int(a, dims)?),
            _ => diag("index must be a constant or generic dim", e.line, e.col),
        }
    }

    fn matmul_ty(&self, a: &CTy, b: &CTy, l: usize, c: usize) -> CResult<CTy> {
        // v1: vector @ matrix and matrix @ matrix; b rank-2.
        if b.dims.len() != 2 {
            return diag(format!("matmul rhs must be rank-2, found {}", b.show()), l, c);
        }
        let (k1, n) = (b.dims[0], b.dims[1]);
        match a.dims.len() {
            1 => {
                if a.dims[0] != k1 {
                    return diag(format!("matmul: a[K={}] vs b[K={k1}, N={n}] — K mismatch", a.dims[0]), l, c);
                }
                Ok(CTy { dt: a.dt, dims: vec![n] })
            }
            2 => {
                if a.dims[1] != k1 {
                    return diag(format!("matmul: a[M={}, K={}] vs b[K={k1}, N={n}] — K mismatch", a.dims[0], a.dims[1]), l, c);
                }
                Ok(CTy { dt: a.dt, dims: vec![a.dims[0], n] })
            }
            _ => diag(format!("matmul lhs must be rank-1 or 2, found {}", a.show()), l, c),
        }
    }

    fn elementwise_ty(&self, a: &CTy, b: &CTy, l: usize, c: usize) -> CResult<CTy> {
        if a.dims == b.dims {
            return Ok(a.clone());
        }
        if a.is_scalar() {
            return Ok(b.clone());
        }
        if b.is_scalar() {
            return Ok(a.clone());
        }
        diag(format!("shape mismatch: {} vs {}", a.show(), b.show()), l, c)
    }

    fn call(
        &mut self,
        name: &str,
        pos: &[Sp<Expr>],
        named: &[(Sp<String>, Sp<Expr>)],
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
        l: usize,
        c: usize,
    ) -> CResult<Typed> {
        // `grad(loss_expr, param)` — reverse-mode AD.
        if name == "grad" {
            return self.grad_call(pos, env, dims, depth, l, c);
        }
        // user fn → inline
        if self.fns.contains_key(name) {
            let f = self.fns[name].clone();
            return self.inline(&f.v, pos, env, dims, depth, l, c);
        }
        let mut targs = Vec::new();
        for a in pos {
            targs.push(self.expr(a, env, dims, depth)?);
        }
        let mut tnamed = Vec::new();
        for (n, a) in named {
            tnamed.push((n.v.clone(), self.expr(a, env, dims, depth)?));
        }
        let ty = self.builtin_ret(name, &targs, &tnamed, l, c)?;
        Ok(Typed {
            e: TExpr::Call(name.to_string(), targs, tnamed),
            ty,
        })
    }

    /// recognize `s * grad(...)`, `grad(...)`, or a let-bound name for
    /// either — used to fold `w op= s*grad` into a ParamUpdate.
    fn grad_of<'x>(
        &self,
        mut e: &'x Sp<Expr>,
        letmap: &'x HashMap<String, Sp<Expr>>,
    ) -> Option<(f64, &'x Vec<Sp<Expr>>)> {
        let mut scale = 1.0;
        for _ in 0..4 {
            match &e.v {
                Expr::Call(n, pos, _) if n == "grad" => return Some((scale, pos)),
                Expr::Ident(s) => match letmap.get(s) {
                    Some(d) => e = d,
                    None => return None,
                },
                Expr::Bin(BinOp::Mul, a, b) => {
                    let sc = |e: &Sp<Expr>| match &e.v {
                        Expr::Lit(f) => Some(*f),
                        Expr::Int(n) => Some(*n as f64),
                        Expr::Unary(UnOp::Neg, i) => match &i.v {
                            Expr::Lit(f) => Some(-*f),
                            Expr::Int(n) => Some(-*n as f64),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(s) = sc(a) {
                        scale *= s;
                        e = b;
                    } else if let Some(s) = sc(b) {
                        scale *= s;
                        e = a;
                    } else {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        None
    }

    /// `grad(loss(x, t, w), w)` — differentiate a scalar-producing call
    /// w.r.t. a named param. Result is a tensor shaped like the param.
    fn grad_call(
        &mut self,
        pos: &[Sp<Expr>],
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
        l: usize,
        c: usize,
    ) -> CResult<Typed> {
        if pos.len() != 2 {
            return diag("grad(loss_expr, param_name) expects 2 args", l, c);
        }
        let wrt = match &pos[1].v {
            Expr::Ident(s) => s.clone(),
            _ => return diag("grad's second arg must be a param name", pos[1].line, pos[1].col),
        };
        let (_, pty, _) = self
            .params
            .get(&wrt)
            .cloned()
            .ok_or(())
            .map_err(|_| Diag {
                msg: format!("grad target {wrt:?} is not a param"),
                line: pos[1].line,
                col: pos[1].col,
            })?;
        let loss = self.expr(&pos[0], env, dims, depth)?;
        if !loss.ty.is_scalar() {
            return diag(
                format!("grad requires a scalar loss, found {}", loss.ty.show()),
                pos[0].line,
                pos[0].col,
            );
        }
        let pty = resolve_ty(dims, &pty, l, c)?;
        Ok(Typed {
            e: TExpr::Grad {
                loss: Box::new(loss),
                wrt,
            },
            ty: pty,
        })
    }

    fn builtin_ret(
        &self,
        name: &str,
        args: &[Typed],
        _named: &[(String, Typed)],
        l: usize,
        c: usize,
    ) -> CResult<CTy> {
        let arity = |n: usize| -> CResult<()> {
            if args.len() != n {
                diag(format!("{name} expects {n} args, found {}", args.len()), l, c)
            } else {
                Ok(())
            }
        };
        match name {
            // v1 lowerable set — the invariant is "everything that
            // typechecks lowers", so the admitted set IS the backend set.
            "relu" | "sigmoid" | "tanh" | "silu" | "softmax" => {
                arity(1)?;
                if args[0].ty.dims.is_empty() {
                    return diag(format!("{name} on scalar"), l, c);
                }
                Ok(args[0].ty.clone())
            }
            "rmsnorm" => {
                arity(2)?;
                if args[0].ty.dims.len() != 1 || args[1].ty.dims.len() != 1
                    || args[0].ty.dims[0] != args[1].ty.dims[0]
                {
                    return diag("rmsnorm(x[C], w[C]) expected", l, c);
                }
                Ok(args[0].ty.clone())
            }
            // scalar loss — only meaningful as a `grad` target; the scalar
            // itself never materializes on-chip.
            "mse" => {
                arity(2)?;
                if args[0].ty.dims != args[1].ty.dims {
                    return diag("mse args must match shapes", l, c);
                }
                Ok(CTy::scalar(DType::F32))
            }
            // `outer(a[T], b[D]) -> [T, D]` — packed row×col product, the
            // cache-write primitive (and the weight-grad shape).
            "outer" => {
                arity(2)?;
                if args[0].ty.dims.len() != 1 || args[1].ty.dims.len() != 1 {
                    return diag("outer expects two vectors", l, c);
                }
                Ok(CTy {
                    dt: args[0].ty.dt,
                    dims: vec![args[0].ty.dims[0], args[1].ty.dims[0]],
                })
            }
            "__ret" => {
                arity(1)?;
                Ok(args[0].ty.clone())
            }
            _ => diag(
                format!("unknown function {name:?} — v1 builtins: relu sigmoid tanh silu softmax rmsnorm mse outer"),
                l,
                c,
            ),
        }
    }

    fn inline(
        &mut self,
        f: &Fn,
        pos: &[Sp<Expr>],
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
        l: usize,
        c: usize,
    ) -> CResult<Typed> {
        if pos.len() != f.args.len() {
            return diag(
                format!("{} expects {} args, found {}", f.name, f.args.len(), pos.len()),
                l,
                c,
            );
        }
        // bind callee dims from concrete arg types
        let mut cdims: Dims = dims.clone();
        let mut cenv: Env = HashMap::new();
        let mut targs = Vec::new();
        for ((an, aty), ae) in f.args.iter().zip(pos.iter()) {
            let ta = self.expr(ae, env, dims, depth)?;
            unify(&mut cdims, &aty.v, &ta.ty, ae.line, ae.col)?;
            cenv.insert(an.v.clone(), ta.ty.clone());
            targs.push(ta);
        }
        // declared generic dims must all get bound by arg unification
        for d in &f.dims {
            if !cdims.contains_key(&d.v) {
                return diag(format!("dim {} of {} is not bound by arguments", d.v, f.name), d.line, d.col);
            }
        }
        let mut letmap = HashMap::new();
        let (body, last) = self.body(&f.body, &mut cenv, &mut cdims, depth + 1, &mut letmap)?;
        let ret = match last {
            Some(t) => t,
            None => return diag(format!("{} has no result expression", f.name), l, c),
        };
        if let Some(rt) = &f.ret {
            let want = resolve_ty(&cdims, &rt.v, rt.line, rt.col)?;
            if want.dims != ret.ty.dims {
                return diag(
                    format!("{} declared -> {} but returns {}", f.name, want.show(), ret.ty.show()),
                    rt.line,
                    rt.col,
                );
            }
        }
        // move actual args into the block so evaluation order is respected
        let mut stmts = Vec::new();
        for (i, ta) in targs.into_iter().enumerate() {
            let nm = format!("__arg{i}");
            cenv.insert(nm.clone(), ta.ty.clone());
            stmts.push(TStmt::Let { name: nm, expr: ta });
        }
        // rebind arg names to __argN names inside the block? — instead: emit
        // aliases mapping arg names to the evaluated values.
        let mut aliased = Vec::new();
        for (i, (an, _)) in f.args.iter().enumerate() {
            aliased.push(TStmt::Let {
                name: an.v.clone(),
                expr: Typed {
                    e: TExpr::Ident(format!("__arg{i}")),
                    ty: cenv[&format!("__arg{i}")].clone(),
                },
            });
        }
        stmts.extend(aliased);
        stmts.extend(body);
        Ok(Typed {
            e: TExpr::Block(stmts, Box::new(ret.clone())),
            ty: ret.ty,
        })
    }

    /// typecheck a statement list; returns (stmts, trailing-expr).
    /// `letmap` records let-bound source exprs so param updates like
    /// `w -= lr * dw` can fold through the binding to a grad call.
    fn body(
        &mut self,
        stmts: &[Sp<Stmt>],
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
        letmap: &mut HashMap<String, Sp<Expr>>,
    ) -> CResult<(Vec<TStmt>, Option<Typed>)> {
        let mut out = Vec::new();
        let mut last = None;
        for (i, s) in stmts.iter().enumerate() {
            let terminal = i + 1 == stmts.len();
            match &s.v {
                Stmt::Let { name, ty, expr } => {
                    letmap.insert(name.v.clone(), expr.clone());
                    let te = self.expr(expr, env, dims, depth)?;
                    if let Some(dt) = ty {
                        let want = resolve_ty(dims, &dt.v, dt.line, dt.col)?;
                        if want.dims != te.ty.dims {
                            return diag(
                                format!("{}: declared {} but expression is {}", name.v, want.show(), te.ty.show()),
                                dt.line,
                                dt.col,
                            );
                        }
                    }
                    env.insert(name.v.clone(), te.ty.clone());
                    out.push(TStmt::Let { name: name.v.clone(), expr: te });
                    if terminal {
                        last = None;
                    }
                }
                Stmt::Assign { name, op, expr } => {
                    // `frozen`/`sealed` params are immutable identity —
                    // only `trainable` params may be reassigned.
                    if let Some((q, _, _)) = self.params.get(&name.v) {
                        if *q != Qual::Trainable {
                            return diag(
                                format!("cannot assign to {:?} param {:?}", q, name.v),
                                name.line,
                                name.col,
                            );
                        }
                        // `w op= s * grad(loss, w)` → folded ParamUpdate.
                        if let Some((scale, pos)) = self.grad_of(expr, letmap) {
                            let sign = match op {
                                AssignOp::Add => 1.0,
                                AssignOp::Sub => -1.0,
                                _ => return diag("param updates support += and -= in v1", s.line, s.col),
                            };
                            let grad = self.grad_call(pos, env, dims, depth, expr.line, expr.col)?;
                            out.push(TStmt::ParamUpdate {
                                param: name.v.clone(),
                                scale: sign * scale,
                                grad,
                            });
                            continue;
                        }
                    }
                    let cur = env.get(&name.v).cloned().ok_or(()).map_err(|_| Diag {
                        msg: format!("assignment to undeclared {}", name.v),
                        line: name.line,
                        col: name.col,
                    })?;
                    let te = self.expr(expr, env, dims, depth)?;
                    if te.ty.dims != cur.dims {
                        return diag(
                            format!("assign: {} is {}, expr is {}", name.v, cur.show(), te.ty.show()),
                            s.line,
                            s.col,
                        );
                    }
                    out.push(TStmt::Assign { name: name.v.clone(), op: *op, expr: te });
                }
                Stmt::Store { target, op, expr } => {
                    let ts = self.store(target, op, expr, env, dims, depth, s.line, s.col)?;
                    out.push(ts);
                }
                Stmt::Expr(e) => {
                    let te = self.expr(e, env, dims, depth)?;
                    if terminal {
                        // the trailing expression is the block's result —
                        // it is evaluated once as `last`, not again as a stmt.
                        last = Some(te);
                    } else {
                        out.push(TStmt::Expr(te));
                    }
                }
            }
        }
        Ok((out, last))
    }

    /// `kv.k = e` | `kv.k += e` | `kv.k[pos] = e` — state-field writes.
    /// Index writes desugar to `field + outer(pos, val)` — additive by
    /// construction because packed tensors only route `add` on-chip.
    /// The `pos` index is a one-hot vector, not a scalar — scatter by
    /// mask is the hardware's native scatter.
    fn store(
        &mut self,
        target: &Sp<Expr>,
        op: &AssignOp,
        expr: &Sp<Expr>,
        env: &mut Env,
        dims: &mut Dims,
        depth: usize,
        l: usize,
        c: usize,
    ) -> CResult<TStmt> {
        // target → (state, field, index-exprs?)
        let (sname, fname, idxs) = match &target.v {
            Expr::Field(b, f) => match &b.v {
                Expr::Ident(s) => (s.clone(), f.v.clone(), None),
                _ => return diag("store target must be a state field", target.line, target.col),
            },
            Expr::Index(b, idxs) => match &b.v {
                Expr::Field(fb, f) => match &fb.v {
                    Expr::Ident(s) => (s.clone(), f.v.clone(), Some(idxs)),
                    _ => return diag("store target must be a state field", target.line, target.col),
                },
                _ => return diag("scatter writes target state fields — `kv.k[pos] = e`", target.line, target.col),
            },
            _ => return diag("store target must be `state.field` or `state.field[i]`", target.line, target.col),
        };
        let fields = self
            .states
            .get(&sname)
            .cloned()
            .ok_or(())
            .map_err(|_| Diag {
                msg: format!("{sname:?} is not a state block"),
                line: target.line,
                col: target.col,
            })?;
        let fty = fields
            .iter()
            .find(|(n, _)| n == &fname)
            .map(|(_, t)| t.clone())
            .ok_or(())
            .map_err(|_| Diag {
                msg: format!("state {sname} has no field {fname}"),
                line: target.line,
                col: target.col,
            })?;
        let fct = resolve_ty(dims, &fty, l, c)?;
        let key = format!("{sname}.{fname}");
        let port = format!("{sname}_{fname}");
        // `s.f` as a source expr (for += / scatter desugars)
        let fref = |line: usize, col: usize| -> Sp<Expr> {
            sp(
                Expr::Field(
                    Box::new(sp(Expr::Ident(sname.clone()), line, col)),
                    sp(fname.clone(), line, col),
                ),
                line,
                col,
            )
        };
        let desugared: Sp<Expr> = match (idxs, op) {
            // `s.f = e` — the name just rebinds to the new value
            (None, AssignOp::Set) => expr.clone(),
            // `s.f += e` → `s.f + e` — the packed add route
            (None, AssignOp::Add) => sp(
                Expr::Bin(BinOp::Add, Box::new(fref(l, c)), Box::new(expr.clone())),
                l,
                c,
            ),
            (None, _) => {
                return diag(
                    "state fields support `=` and `+=` — the packed route is add-only",
                    target.line,
                    target.col,
                )
            }
            // `s.f[p] = e` / `s.f[p] += e` → `s.f + outer(p, e)` —
            // additive write; fresh cache slots start zeroed.
            (Some(ix), AssignOp::Set) | (Some(ix), AssignOp::Add) => {
                if ix.len() != 1 {
                    return diag("v1 scatter writes one index — `s.f[p] = e`", target.line, target.col);
                }
                let pos = match &ix[0].v {
                    Index::Expr(e) => e.clone(),
                    Index::Range(..) => {
                        return diag("scatter index is a one-hot vector, not a range", ix[0].line, ix[0].col)
                    }
                };
                // the index must be a runtime vector matching dim 0
                let pos_t = self.expr(&pos, env, dims, depth)?;
                if pos_t.ty.dims.len() != 1 || pos_t.ty.dims[0] != fct.dims[0] {
                    return diag(
                        format!(
                            "scatter index {} must be a one-hot vector [{}]",
                            pos_t.ty.show(),
                            fct.dims[0]
                        ),
                        ix[0].line,
                        ix[0].col,
                    );
                }
                let val_t = self.expr(expr, env, dims, depth)?;
                let want = &fct.dims[1..];
                if val_t.ty.dims != want {
                    return diag(
                        format!("scatter value {} must be [{}]", val_t.ty.show(), want.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
                        expr.line,
                        expr.col,
                    );
                }
                let outer = sp(
                    Expr::Call("outer".into(), vec![pos, expr.clone()], vec![]),
                    l,
                    c,
                );
                sp(
                    Expr::Bin(BinOp::Add, Box::new(fref(l, c)), Box::new(outer)),
                    l,
                    c,
                )
            }
            (Some(ix), AssignOp::Sub) => {
                if ix.len() != 1 {
                    return diag("v1 scatter writes one index — `s.f[p] -= e`", target.line, target.col);
                }
                let pos = match &ix[0].v {
                    Index::Expr(e) => e.clone(),
                    Index::Range(..) => {
                        return diag("scatter index is a one-hot vector, not a range", ix[0].line, ix[0].col)
                    }
                };
                let outer = sp(
                    Expr::Call(
                        "outer".into(),
                        vec![
                            pos,
                            sp(Expr::Unary(UnOp::Neg, Box::new(expr.clone())), expr.line, expr.col),
                        ],
                        vec![],
                    ),
                    l,
                    c,
                );
                sp(
                    Expr::Bin(BinOp::Add, Box::new(fref(l, c)), Box::new(outer)),
                    l,
                    c,
                )
            }
            (Some(_), _) => return diag("state scatter supports `=`, `+=`, `-=`", target.line, target.col),
        };
        let te = self.expr(&desugared, env, dims, depth)?;
        if te.ty.dims != fct.dims {
            return diag(
                format!("state update {} produces {} — field is {}", key, te.ty.show(), fct.show()),
                l,
                c,
            );
        }
        Ok(TStmt::StateUpdate { key, port, expr: te })
    }
}

/// typecheck a whole program → concrete TProgram.
pub fn check(p: &Program) -> CResult<TProgram> {
    let mut ck = Ck {
        fns: HashMap::new(),
        params: HashMap::new(),
        states: HashMap::new(),
        ios: Vec::new(),
    };
    let mut invariants = Vec::new();
    let mut caps = Vec::new();
    let mut provenance = Vec::new();
    for d in &p.decls {
        match d {
            Decl::Invariant(n) => invariants.push(n.v.clone()),
            Decl::Cap(cs) => caps.extend(cs.iter().map(|c| c.v.clone())),
            Decl::Provenance(kvs) => {
                provenance.extend(kvs.iter().map(|(k, v)| (k.v.clone(), v.v.clone())))
            }
            Decl::Param { name, qual, ty, src } => {
                ck.params.insert(
                    name.v.clone(),
                    (*qual, ty.v.clone(), src.as_ref().map(|s| s.v.clone())),
                );
            }
            Decl::Io { name, dir, ty } => {
                let dims = Dims::new();
                let ct = resolve_ty(&dims, &ty.v, ty.line, ty.col)?;
                ck.ios.push((name.v.clone(), *dir, ct));
            }
            Decl::State { name, fields } => {
                ck.states.insert(
                    name.v.clone(),
                    fields.iter().map(|(f, t)| (f.v.clone(), t.v.clone())).collect(),
                );
            }
            Decl::Fn(f) => {
                ck.fns.insert(f.v.name.clone(), f.clone());
            }
        }
    }
    let main = ck
        .fns
        .get("main")
        .cloned()
        .ok_or(())
        .map_err(|_| Diag { msg: "no `fn main` — a coyote program needs an entry".into(), line: 0, col: 0 })?;
    // bind main's args to `io in` decls
    let ins: Vec<&(String, IoDir, CTy)> = ck.ios.iter().filter(|(_, d, _)| *d == IoDir::In).collect();
    if ins.len() != main.v.args.len() {
        return diag(
            format!("main takes {} args but {} `io in` declared", main.v.args.len(), ins.len()),
            main.line,
            main.col,
        );
    }
    let mut dims = Dims::new();
    let mut env = Env::new();
    let mut arg_ports = Vec::new();
    for ((an, aty), (ion, _, ict)) in main.v.args.iter().zip(ins.iter()) {
        unify(&mut dims, &aty.v, ict, aty.line, aty.col)?;
        // env name is the fn's arg name; io name is the mil port name — the
        // lowerer maps arg name → io port.
        env.insert(an.v.clone(), ict.clone());
        arg_ports.push((an.v.clone(), ion.clone()));
    }
    let mut tparams = Vec::new();
    let mut pdims = Dims::new();
    // deterministic param order — the mil signature and blob set must
    // be reproducible across runs.
    let mut sorted: Vec<_> = ck.params.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (n, (q, t, s)) in sorted {
        let ct = resolve_ty(&mut pdims, t, 0, 0)?;
        tparams.push(TParam { name: n.clone(), qual: *q, ty: ct, src: s.clone() });
    }
    // main sees param types under its own dim env
    for (n, (_, t, _)) in ck.params.clone() {
        if let Ok(ct) = resolve_ty(&dims, &t, 0, 0) {
            env.insert(n, ct);
        }
    }
    let mut letmap = HashMap::new();
    let (body, last) = ck.body(&main.v.body, &mut env, &mut dims, 0, &mut letmap)?;
    let ret = last.ok_or(()).map_err(|_| Diag {
        msg: "main must end in a result expression".into(),
        line: main.line,
        col: main.col,
    })?;
    let outs: Vec<&(String, IoDir, CTy)> = ck.ios.iter().filter(|(_, d, _)| *d == IoDir::Out).collect();
    if let Some((_, _, oct)) = outs.first() {
        if oct.dims != ret.ty.dims {
            return diag(
                format!("main returns {} but `io out` declares {}", ret.ty.show(), oct.show()),
                main.line,
                main.col,
            );
        }
    }
    // state blocks → flat field tensors: (key `kv.k`, port `kv_k`, ty)
    let mut states = Vec::new();
    let mut snames: Vec<_> = ck.states.iter().collect();
    snames.sort_by(|a, b| a.0.cmp(b.0));
    for (sname, fields) in snames {
        for (fname, fty) in fields {
            let ct = resolve_ty(&Dims::new(), fty, 0, 0)?;
            states.push((format!("{sname}.{fname}"), format!("{sname}_{fname}"), ct));
        }
    }
    Ok(TProgram {
        ios: ck.ios,
        params: tparams,
        invariants,
        caps,
        provenance,
        arg_ports,
        body,
        ret,
        states,
    })
}
