//! hostck — the organism-stratum typechecker.
//!
//! Where the tensor stratum checks shapes, this stratum checks the
//! organism's constitution: sealed params cannot be named here at all
//! (the boundary is total, not a lint), `token` is the identity type
//! Bad Apple asked for — immutable, witnessed — and ANE programs are
//! first-class values of type `ane(args) -> ret`.

use crate::ast::*;
use crate::typeck::{CTy, Ck, TProgram};
use crate::{diag, CResult, Diag};
use std::collections::HashMap;

/// resolved host type.
#[derive(Debug, Clone, PartialEq)]
pub enum RTy {
    I64,
    F64,
    Bool,
    Str,
    Unit,
    Token,
    Tensor(CTy),
    Vec(Box<RTy>),
    Struct(u32),
    /// `ane(args) -> ret` — a program value.
    AneFn(Vec<RTy>, Box<RTy>),
}

impl RTy {
    pub fn show(&self, structs: &[(String, Vec<(String, RTy)>)]) -> String {
        match self {
            RTy::I64 => "i64".into(),
            RTy::F64 => "f64".into(),
            RTy::Bool => "bool".into(),
            RTy::Str => "str".into(),
            RTy::Unit => "unit".into(),
            RTy::Token => "token".into(),
            RTy::Tensor(t) => t.show(),
            RTy::Vec(t) => format!("[{}]", t.show(structs)),
            RTy::Struct(i) => structs[*i as usize].0.clone(),
            RTy::AneFn(a, r) => format!(
                "ane({}) -> {}",
                a.iter().map(|t| t.show(structs)).collect::<Vec<_>>().join(", "),
                r.show(structs),
            ),
        }
    }
}

/// native function table — indices are stable; vm.rs matches by index.
pub const NATIVE_NAMES: &[&str] = &[
    "print",      // (any...) -> unit — stdout
    "str",        // (any) -> str
    "len",        // (vec|str) -> i64
    "push",       // (vec<T>, T) -> unit
    "argmax",     // (f16[N]) -> i64
    "i64",        // (i64|f64|bool|token) -> i64
    "f64",        // (i64|f64) -> f64
    "onehot",     // (i64 i, i64 const n) -> f16[n]
    "causal",     // (i64 upto, i64 const n) -> f16[n]
    "rope_cos",   // (i64 pos, i64 const d) -> f16[d]
    "rope_sin",   // (i64 pos, i64 const d) -> f16[d]
    "zeros",      // (i64 const n) -> f16[n]
    "self_token", // () -> token — witnessed identity (id, epoch)
    "witness",    // (str tag, any val) -> unit — hash-chained log entry
    "die",        // () -> unit — halt the organism with attestation
    "migrate",    // (str path) -> unit — serialize full VM state
    "rand",       // () -> f64 — deterministic, seeded by lineage
];

#[derive(Debug, Clone)]
pub enum HE {
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// local variable — name resolved to a slot at bytecode compile.
    Var(String),
    /// a global: param tensor, state-field mirror, or program value.
    Global(u32),
    Un(UnOp, Box<HT>),
    Bin(BinOp, Box<HT>, Box<HT>),
    Call(u32, Vec<HT>),
    /// direct call of a known program — `layer(x, pos)`.
    CallAne(u32, Vec<HT>),
    /// call through a program-typed value — `p(x)` where p: ane(...).
    CallDyn(Box<HT>, Vec<HT>),
    Native(u32, Vec<HT>),
    Index(Box<HT>, Box<HT>),
    Field(Box<HT>, u32),
    VecLit(Vec<HT>),
    /// struct idx + field values in declaration order.
    StructLit(u32, Vec<HT>),
}

#[derive(Debug, Clone)]
pub struct HT {
    pub e: HE,
    pub ty: RTy,
}

#[derive(Debug, Clone)]
pub enum StoreT {
    /// `v[i] = e` — vec element or tensor element write.
    Index(HT, HT),
    /// `s.f = e` — struct field write.
    Field(HT, u32),
}

#[derive(Debug, Clone)]
pub enum HS {
    Let {
        name: String,
        mutable: bool,
        expr: HT,
    },
    Assign {
        name: String,
        op: AssignOp,
        expr: HT,
    },
    Store {
        target: StoreT,
        expr: HT,
    },
    If {
        cond: HT,
        then: Vec<HS>,
        els: Vec<HS>,
    },
    While {
        cond: HT,
        body: Vec<HS>,
    },
    For {
        var: String,
        lo: HT,
        hi: HT,
        body: Vec<HS>,
    },
    Break,
    Continue,
    Return(Option<HT>),
    Expr(HT),
}

#[derive(Debug, Clone)]
pub struct HFnT {
    pub name: String,
    pub args: Vec<(String, RTy)>,
    pub ret: RTy,
    pub body: Vec<HS>,
}

/// a global slot's source: param tensor (resolved weights), a live
/// state-field mirror, a host data table, or an ane program value.
#[derive(Debug, Clone)]
pub enum GVal {
    Param { name: String, ty: CTy, src: Option<String> },
    Table { name: String, ty: CTy, src: Option<String> },
    State { key: String, port: String, ty: CTy },
    Prog { idx: u32 },
}

#[derive(Debug)]
pub struct HProgram {
    pub fns: Vec<HFnT>,
    pub structs: Vec<(String, Vec<(String, RTy)>)>,
    /// globals in slot order — tensors and program values.
    pub globals: Vec<(String, GVal, RTy)>,
    /// `main`'s index into fns, if present.
    pub entry: Option<usize>,
}

struct Hk {
    fns: HashMap<String, u32>,
    fndecl: Vec<HFn>,
    structs: Vec<(String, Vec<(String, RTy)>)>,
    /// ane program name → (index, arg types, ret type)
    progs: HashMap<String, (u32, Vec<RTy>, RTy)>,
    /// param/state name → (global idx, RTy, sealed)
    globals: HashMap<String, (u32, RTy)>,
    /// sealed params — naming one here is the constitutional error.
    sealed: std::collections::HashSet<String>,
    glist: Vec<(String, GVal, RTy)>,
    /// state block name → field name → global idx
    states: HashMap<String, HashMap<String, u32>>,
}

type HEnv = Vec<HashMap<String, (RTy, bool)>>; // name → (ty, mutable)

fn lookup<'a>(env: &'a HEnv, n: &str) -> Option<&'a (RTy, bool)> {
    env.iter().rev().find_map(|s| s.get(n))
}

impl Hk {
    fn resolve(&self, t: &HTy, l: usize, c: usize) -> CResult<RTy> {
        Ok(match t {
            HTy::I64 => RTy::I64,
            HTy::F64 => RTy::F64,
            HTy::Bool => RTy::Bool,
            HTy::Str => RTy::Str,
            HTy::Unit => RTy::Unit,
            HTy::Token => RTy::Token,
            HTy::Tensor(ty) => {
                let mut dims = Vec::new();
                for d in &ty.dims {
                    match d {
                        Dim::Lit(n) => dims.push(*n),
                        Dim::Name(s) => {
                            return diag(
                                format!("host tensors need literal dims — {s} is generic"),
                                l,
                                c,
                            )
                        }
                    }
                }
                RTy::Tensor(CTy { dt: ty.dt, dims })
            }
            HTy::Vec(inner) => RTy::Vec(Box::new(self.resolve(inner, l, c)?)),
            HTy::Struct(name) => {
                let i = self
                    .structs
                    .iter()
                    .position(|(n, _)| n == name)
                    .ok_or(())
                    .map_err(|_| Diag { msg: format!("no struct {name:?}"), line: l, col: c })?;
                RTy::Struct(i as u32)
            }
            HTy::AneFn(args, ret) => RTy::AneFn(
                args.iter()
                    .map(|a| self.resolve(a, l, c))
                    .collect::<CResult<Vec<_>>>()?,
                Box::new(self.resolve(ret, l, c)?),
            ),
        })
    }

    fn expr(
        &mut self,
        e: &Sp<Expr>,
        env: &mut HEnv,
        loops: usize,
    ) -> CResult<HT> {
        let _ = loops;
        let (l, c) = (e.line, e.col);
        match &e.v {
            Expr::Int(n) => Ok(HT { e: HE::I64(*n), ty: RTy::I64 }),
            Expr::Lit(f) => Ok(HT { e: HE::F64(*f), ty: RTy::F64 }),
            Expr::Bool(b) => Ok(HT { e: HE::Bool(*b), ty: RTy::Bool }),
            Expr::Str(s) => Ok(HT {
                e: HE::Str(s.clone()),
                ty: RTy::Str,
            }),
            Expr::VecLit(items) => {
                if items.is_empty() {
                    return diag("empty vec literal needs a type annotation — `let v: [i64] = []`", l, c);
                }
                let mut titems = Vec::new();
                let mut ety: Option<RTy> = None;
                for it in items {
                    let t = self.expr(it, env, loops)?;
                    match &ety {
                        None => ety = Some(t.ty.clone()),
                        Some(w) if *w == t.ty => {}
                        Some(w) => {
                            return diag(
                                format!("vec literal mixes {} and {}", w.show(&self.structs), t.ty.show(&self.structs)),
                                it.line,
                                it.col,
                            )
                        }
                    }
                    titems.push(t);
                }
                let ty = RTy::Vec(Box::new(ety.unwrap()));
                Ok(HT {
                    e: HE::VecLit(titems),
                    ty,
                })
            }
            Expr::StructLit(name, fields) => {
                let si = self
                    .structs
                    .iter()
                    .position(|(n, _)| n == &name.v)
                    .ok_or(())
                    .map_err(|_| Diag {
                        msg: format!("no struct {:?}", name.v),
                        line: name.line,
                        col: name.col,
                    })? as u32;
                let decl = self.structs[si as usize].1.clone();
                let mut ordered: Vec<Option<HT>> = vec![None; decl.len()];
                for (fname, fe) in fields {
                    let fi = decl
                        .iter()
                        .position(|(n, _)| n == &fname.v)
                        .ok_or(())
                        .map_err(|_| Diag {
                            msg: format!("{} has no field {}", name.v, fname.v),
                            line: fname.line,
                            col: fname.col,
                        })?;
                    let t = self.expr(fe, env, loops)?;
                    if t.ty != decl[fi].1 {
                        return diag(
                            format!(
                                "field {} wants {}, found {}",
                                fname.v,
                                decl[fi].1.show(&self.structs),
                                t.ty.show(&self.structs)
                            ),
                            fe.line,
                            fe.col,
                        );
                    }
                    ordered[fi] = Some(t);
                }
                let mut vals = Vec::new();
                for (i, o) in ordered.into_iter().enumerate() {
                    match o {
                        Some(t) => vals.push(t),
                        None => {
                            return diag(
                                format!("struct {} missing field {}", name.v, decl[i].0),
                                l,
                                c,
                            )
                        }
                    }
                }
                Ok(HT {
                    e: HE::StructLit(si, vals),
                    ty: RTy::Struct(si),
                })
            }
            Expr::Ident(s) => {
                if let Some((ty, _)) = lookup(env, s) {
                    let ty = ty.clone();
                    return Ok(HT {
                        e: HE::Var(s.clone()),
                        ty,
                    });
                }
                if let Some(&(gi, ref ty)) = self.globals.get(s) {
                    return Ok(HT {
                        e: HE::Global(gi),
                        ty: ty.clone(),
                    });
                }
                if self.sealed.contains(s) {
                    return diag(
                        format!("sealed param {s:?} cannot be named in host code — the boundary holds"),
                        l,
                        c,
                    );
                }
                diag(format!("unknown name {s:?}"), l, c)
            }
            Expr::Unary(op, a) => {
                let a = self.expr(a, env, loops)?;
                match op {
                    UnOp::Neg => match a.ty {
                        RTy::I64 | RTy::F64 => Ok(HT {
                            ty: a.ty.clone(),
                            e: HE::Un(*op, Box::new(a)),
                        }),
                        _ => diag(format!("- requires i64/f64, found {}", a.ty.show(&self.structs)), l, c),
                    },
                    UnOp::Not => match a.ty {
                        RTy::Bool => Ok(HT {
                            e: HE::Un(*op, Box::new(a)),
                            ty: RTy::Bool,
                        }),
                        _ => diag(format!("! requires bool, found {}", a.ty.show(&self.structs)), l, c),
                    },
                }
            }
            Expr::Bin(op, a, b) => {
                let a = self.expr(a, env, loops)?;
                let b = self.expr(b, env, loops)?;
                let same = a.ty == b.ty;
                let ty = match op {
                    BinOp::Add => match (&a.ty, &b.ty) {
                        (RTy::I64, RTy::I64) => RTy::I64,
                        (RTy::F64, RTy::F64) => RTy::F64,
                        (RTy::Str, RTy::Str) => RTy::Str,
                        _ => {
                            return diag(
                                format!("+ needs matching i64/f64/str, found {} and {}", a.ty.show(&self.structs), b.ty.show(&self.structs)),
                                l,
                                c,
                            )
                        }
                    },
                    BinOp::Sub | BinOp::Mul | BinOp::Div => match (&a.ty, &b.ty) {
                        (RTy::I64, RTy::I64) => RTy::I64,
                        (RTy::F64, RTy::F64) => RTy::F64,
                        _ => {
                            return diag(
                                format!("arithmetic needs matching i64/f64, found {} and {}", a.ty.show(&self.structs), b.ty.show(&self.structs)),
                                l,
                                c,
                            )
                        }
                    },
                    BinOp::Mod => match (&a.ty, &b.ty) {
                        (RTy::I64, RTy::I64) => RTy::I64,
                        _ => return diag("% needs i64 operands", l, c),
                    },
                    BinOp::Eq | BinOp::Ne => {
                        if !same {
                            return diag(
                                format!("== needs matching types, found {} and {}", a.ty.show(&self.structs), b.ty.show(&self.structs)),
                                l,
                                c,
                            );
                        }
                        RTy::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => match (&a.ty, &b.ty) {
                        (RTy::I64, RTy::I64) | (RTy::F64, RTy::F64) | (RTy::Str, RTy::Str) => RTy::Bool,
                        _ => {
                            return diag(
                                format!("ordering needs i64/f64/str, found {} and {}", a.ty.show(&self.structs), b.ty.show(&self.structs)),
                                l,
                                c,
                            )
                        }
                    },
                    BinOp::And | BinOp::Or => match (&a.ty, &b.ty) {
                        (RTy::Bool, RTy::Bool) => RTy::Bool,
                        _ => return diag("&& and || need bool operands", l, c),
                    },
                    BinOp::MatMul => return diag("@ is tensor-stratum — belongs in an `fn`", l, c),
                };
                Ok(HT {
                    e: HE::Bin(*op, Box::new(a), Box::new(b)),
                    ty,
                })
            }
            Expr::Transpose(_) => diag(".T is tensor-stratum — belongs in an `fn`", l, c),
            Expr::Call(name, pos, named) => {
                if !named.is_empty() {
                    return diag("host calls are positional — no named args", l, c);
                }
                self.call(name, pos, env, loops, l, c)
            }
            Expr::Index(base, idxs) => {
                if idxs.len() != 1 {
                    return diag("host indexing takes one index — `v[i]`", l, c);
                }
                let idx = match &idxs[0].v {
                    Index::Expr(ie) => ie.clone(),
                    Index::Range(..) => {
                        return diag("host slices aren't supported — use a for loop", idxs[0].line, idxs[0].col)
                    }
                };
                let tb = self.expr(base, env, loops)?;
                let ti = self.expr(&idx, env, loops)?;
                if ti.ty != RTy::I64 {
                    return diag(format!("index must be i64, found {}", ti.ty.show(&self.structs)), idx.line, idx.col);
                }
                let ty = match &tb.ty {
                    RTy::Vec(t) => (**t).clone(),
                    RTy::Tensor(ct) => {
                        if ct.dims.len() == 1 {
                            RTy::F64
                        } else if ct.dims.len() == 2 {
                            // row gather — `embed[tok]` reads a [N] row
                            // at runtime; this is the decode feedback.
                            RTy::Tensor(CTy { dt: ct.dt, dims: vec![ct.dims[1]] })
                        } else {
                            return diag("host indexing supports rank 1-2 tensors", l, c);
                        }
                    }
                    RTy::Str => RTy::Str,
                    _ => {
                        return diag(
                            format!("cannot index {}", tb.ty.show(&self.structs)),
                            l,
                            c,
                        )
                    }
                };
                Ok(HT {
                    e: HE::Index(Box::new(tb), Box::new(ti)),
                    ty,
                })
            }
            Expr::Field(bx, f) => {
                // `state.field` — the on-chip mirror, readable as a tensor.
                if let Expr::Ident(s) = &bx.v {
                    if let Some(fields) = self.states.get(s).cloned() {
                        if let Some(&gi) = fields.get(&f.v) {
                            let ty = self.globals.get(&format!("{s}.{}", f.v)).map(|(_, t)| t.clone()).unwrap_or(RTy::Unit);
                            return Ok(HT {
                                e: HE::Global(gi),
                                ty,
                            });
                        }
                        return diag(format!("state {s} has no field {}", f.v), f.line, f.col);
                    }
                }
                let tb = self.expr(bx, env, loops)?;
                match &tb.ty {
                    RTy::Struct(si) => {
                        let decl = &self.structs[*si as usize].1;
                        let fi = decl
                            .iter()
                            .position(|(n, _)| n == &f.v)
                            .ok_or(())
                            .map_err(|_| Diag {
                                msg: format!("struct {} has no field {}", self.structs[*si as usize].0, f.v),
                                line: f.line,
                                col: f.col,
                            })?;
                        Ok(HT {
                            ty: decl[fi].1.clone(),
                            e: HE::Field(Box::new(tb), fi as u32),
                        })
                    }
                    RTy::Token => match f.v.as_str() {
                        "id" => Ok(HT { ty: RTy::I64, e: HE::Field(Box::new(tb), 0) }),
                        "epoch" => Ok(HT { ty: RTy::I64, e: HE::Field(Box::new(tb), 1) }),
                        _ => diag("token fields are .id and .epoch", f.line, f.col),
                    },
                    _ => diag(
                        format!("{} has no fields", tb.ty.show(&self.structs)),
                        f.line,
                        f.col,
                    ),
                }
            }
        }
    }

    fn call(
        &mut self,
        name: &str,
        pos: &[Sp<Expr>],
        env: &mut HEnv,
        loops: usize,
        l: usize,
        c: usize,
    ) -> CResult<HT> {
        // host fn?
        if let Some(&fi) = self.fns.get(name) {
            let f = self.fndecl[fi as usize].clone();
            if pos.len() != f.args.len() {
                return diag(
                    format!("{name} expects {} args, found {}", f.args.len(), pos.len()),
                    l,
                    c,
                );
            }
            let mut targs = Vec::new();
            for ((an, aty), ae) in f.args.iter().zip(pos.iter()) {
                let want = self.resolve(&aty.v, aty.line, aty.col)?;
                let t = self.expr(ae, env, loops)?;
                if t.ty != want {
                    return diag(
                        format!("{name} arg {} wants {}, found {}", an.v, want.show(&self.structs), t.ty.show(&self.structs)),
                        ae.line,
                        ae.col,
                    );
                }
                targs.push(t);
            }
            let ret = match &f.ret {
                Some(rt) => self.resolve(&rt.v, rt.line, rt.col)?,
                None => RTy::Unit,
            };
            return Ok(HT {
                e: HE::Call(fi, targs),
                ty: ret,
            });
        }
        // ane program — first-class value called directly.
        if let Some(&(pi, ref pargs, ref pret)) = self.progs.get(name) {
            let (pargs, pret) = (pargs.clone(), pret.clone());
            if pos.len() != pargs.len() {
                return diag(
                    format!("{name} expects {} args, found {}", pargs.len(), pos.len()),
                    l,
                    c,
                );
            }
            let mut targs = Vec::new();
            for (i, (want, ae)) in pargs.iter().zip(pos.iter()).enumerate() {
                let t = self.expr(ae, env, loops)?;
                if &t.ty != want {
                    return diag(
                        format!("{name} arg {i} wants {}, found {}", want.show(&self.structs), t.ty.show(&self.structs)),
                        ae.line,
                        ae.col,
                    );
                }
                targs.push(t);
            }
            return Ok(HT {
                e: HE::CallAne(pi, targs),
                ty: pret,
            });
        }
        // native?
        if let Some(ni) = NATIVE_NAMES.iter().position(|n| *n == name) {
            let mut targs = Vec::new();
            for a in pos {
                targs.push(self.expr(a, env, loops)?);
            }
            let ty = self.native_ret(name, &targs, pos, l, c)?;
            return Ok(HT {
                e: HE::Native(ni as u32, targs),
                ty,
            });
        }
        // program-typed variable — `p(x)` where p: ane(...) -> ...
        if let Some((RTy::AneFn(pargs, pret), _)) = lookup(env, name) {
            let (pargs, pret) = (pargs.clone(), pret.clone());
            if pos.len() != pargs.len() {
                return diag(
                    format!("{name} expects {} args, found {}", pargs.len(), pos.len()),
                    l,
                    c,
                );
            }
            let callee = HT {
                e: HE::Var(name.to_string()),
                ty: RTy::AneFn(pargs.clone(), pret.clone()),
            };
            let mut targs = Vec::new();
            for (i, (want, ae)) in pargs.iter().zip(pos.iter()).enumerate() {
                let t = self.expr(ae, env, loops)?;
                if &t.ty != want {
                    return diag(
                        format!("{name} arg {i} wants {}, found {}", want.show(&self.structs), t.ty.show(&self.structs)),
                        ae.line,
                        ae.col,
                    );
                }
                targs.push(t);
            }
            return Ok(HT {
                ty: (*pret).clone(),
                e: HE::CallDyn(Box::new(callee), targs),
            });
        }
        diag(format!("unknown function {name:?}"), l, c)
    }

    fn native_ret(
        &self,
        name: &str,
        args: &[HT],
        pos: &[Sp<Expr>],
        l: usize,
        c: usize,
    ) -> CResult<RTy> {
        let arity = |n: usize| -> CResult<()> {
            if args.len() != n {
                diag(format!("{name} expects {n} args, found {}", args.len()), l, c)
            } else {
                Ok(())
            }
        };
        let s = &self.structs;
        match name {
            "print" => Ok(RTy::Unit),
            "str" => {
                arity(1)?;
                Ok(RTy::Str)
            }
            "len" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::Vec(_) | RTy::Str | RTy::Tensor(_) => Ok(RTy::I64),
                    t => diag(format!("len on {}", t.show(s)), l, c),
                }
            }
            "push" => {
                arity(2)?;
                match &args[0].ty {
                    RTy::Vec(t) if **t == args[1].ty => Ok(RTy::Unit),
                    _ => diag("push(vec<T>, T) expected", l, c),
                }
            }
            "argmax" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::Tensor(t) if t.dims.len() == 1 => Ok(RTy::I64),
                    _ => diag("argmax needs a rank-1 tensor", l, c),
                }
            }
            "i64" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::I64 | RTy::F64 | RTy::Bool | RTy::Token => Ok(RTy::I64),
                    t => diag(format!("i64() on {}", t.show(s)), l, c),
                }
            }
            "f64" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::I64 | RTy::F64 => Ok(RTy::F64),
                    t => diag(format!("f64() on {}", t.show(s)), l, c),
                }
            }
            "onehot" | "causal" | "rope_cos" | "rope_sin" => {
                arity(2)?;
                if args[0].ty != RTy::I64 {
                    return diag(format!("{name}'s first arg must be i64"), l, c);
                }
                let n = match &pos[1].v {
                    Expr::Int(n) => *n,
                    _ => {
                        return diag(
                            format!("{name}'s size must be a literal — tensor shapes are static"),
                            pos[1].line,
                            pos[1].col,
                        )
                    }
                };
                if n <= 0 {
                    return diag(format!("{name} size must be positive"), pos[1].line, pos[1].col);
                }
                Ok(RTy::Tensor(CTy {
                    dt: DType::F16,
                    dims: vec![n],
                }))
            }
            "zeros" => {
                arity(1)?;
                let n = match &pos[0].v {
                    Expr::Int(n) => *n,
                    _ => {
                        return diag(
                            "zeros's size must be a literal — tensor shapes are static",
                            pos[0].line,
                            pos[0].col,
                        )
                    }
                };
                Ok(RTy::Tensor(CTy {
                    dt: DType::F16,
                    dims: vec![n],
                }))
            }
            "self_token" => {
                arity(0)?;
                Ok(RTy::Token)
            }
            "witness" => {
                arity(2)?;
                if args[0].ty != RTy::Str {
                    return diag("witness(tag: str, val)", l, c);
                }
                Ok(RTy::Unit)
            }
            "die" => {
                arity(0)?;
                Ok(RTy::Unit)
            }
            "migrate" => {
                arity(1)?;
                if args[0].ty != RTy::Str {
                    return diag("migrate(path: str)", l, c);
                }
                Ok(RTy::Unit)
            }
            "rand" => {
                arity(0)?;
                Ok(RTy::F64)
            }
            _ => diag(format!("unknown native {name}"), l, c),
        }
    }

    fn stmts(
        &mut self,
        body: &[Sp<Stmt>],
        env: &mut HEnv,
        ret: &RTy,
        loops: usize,
    ) -> CResult<Vec<HS>> {
        let mut out = Vec::new();
        for s in body {
            out.push(self.stmt(s, env, ret, loops)?);
        }
        Ok(out)
    }

    fn stmt(&mut self, s: &Sp<Stmt>, env: &mut HEnv, ret: &RTy, loops: usize) -> CResult<HS> {
        let (l, c) = (s.line, s.col);
        match &s.v {
            Stmt::Let {
                name,
                hty,
                ty,
                mutable,
                expr,
            } => {
                if ty.is_some() {
                    return diag(
                        "tensor annotation inside a host fn — use host types (i64, f16[N], ...)",
                        name.line,
                        name.col,
                    );
                }
                // `let xs: [i64] = []` — the annotation reaches the
                // empty-literal before inference does.
                if let (Some(ht), Expr::VecLit(items)) = (hty, &expr.v) {
                    if items.is_empty() {
                        let want = self.resolve(&ht.v, ht.line, ht.col)?;
                        if let RTy::Vec(_) = want {
                            env.last_mut()
                                .unwrap()
                                .insert(name.v.clone(), (want.clone(), *mutable));
                            return Ok(HS::Let {
                                name: name.v.clone(),
                                mutable: *mutable,
                                expr: HT {
                                    e: HE::VecLit(vec![]),
                                    ty: want,
                                },
                            });
                        }
                    }
                }
                let te = self.expr(expr, env, loops)?;
                if let Some(ht) = hty {
                    let want = self.resolve(&ht.v, ht.line, ht.col)?;
                    if want != te.ty {
                        return diag(
                            format!(
                                "{}: declared {} but expression is {}",
                                name.v,
                                want.show(&self.structs),
                                te.ty.show(&self.structs)
                            ),
                            ht.line,
                            ht.col,
                        );
                    }
                }
                env.last_mut()
                    .unwrap()
                    .insert(name.v.clone(), (te.ty.clone(), *mutable));
                Ok(HS::Let {
                    name: name.v.clone(),
                    mutable: *mutable,
                    expr: te,
                })
            }
            Stmt::Assign { name, op, expr } => {
                let (vty, mutable) = lookup(env, &name.v).cloned().ok_or(()).map_err(|_| Diag {
                    msg: format!("assignment to undeclared {}", name.v),
                    line: name.line,
                    col: name.col,
                })?;
                if !mutable {
                    return diag(
                        format!("{} is immutable — declare it with `var`", name.v),
                        name.line,
                        name.col,
                    );
                }
                let te = self.expr(expr, env, loops)?;
                if te.ty != vty {
                    return diag(
                        format!(
                            "assign: {} is {}, expr is {}",
                            name.v,
                            vty.show(&self.structs),
                            te.ty.show(&self.structs)
                        ),
                        l,
                        c,
                    );
                }
                match op {
                    AssignOp::Set => {}
                    AssignOp::Add => match vty {
                        RTy::I64 | RTy::F64 | RTy::Str => {}
                        _ => return diag("+= needs i64/f64/str", l, c),
                    },
                    _ => match vty {
                        RTy::I64 | RTy::F64 => {}
                        _ => return diag("compound assignment needs i64/f64", l, c),
                    },
                }
                Ok(HS::Assign {
                    name: name.v.clone(),
                    op: *op,
                    expr: te,
                })
            }
            Stmt::Store { target, op, expr } => {
                if *op != AssignOp::Set {
                    return diag("store targets support `=` — write the compound form", l, c);
                }
                let st = match &target.v {
                    Expr::Index(b, idxs) => {
                        if idxs.len() != 1 {
                            return diag("store indexing takes one index", l, c);
                        }
                        let idx = match &idxs[0].v {
                            Index::Expr(ie) => ie.clone(),
                            Index::Range(..) => return diag("host slices aren't supported", l, c),
                        };
                        let tb = self.expr(b, env, loops)?;
                        let ti = self.expr(&idx, env, loops)?;
                        if ti.ty != RTy::I64 {
                            return diag("index must be i64", l, c);
                        }
                        match &tb.ty {
                            RTy::Vec(_) | RTy::Tensor(_) => StoreT::Index(tb, ti),
                            _ => return diag("can only store into vecs and tensors", l, c),
                        }
                    }
                    Expr::Field(b, f) => {
                        let tb = self.expr(b, env, loops)?;
                        match &tb.ty {
                            RTy::Struct(si) => {
                                let decl = &self.structs[*si as usize].1;
                                let fi = decl
                                    .iter()
                                    .position(|(n, _)| n == &f.v)
                                    .ok_or(())
                                    .map_err(|_| Diag {
                                        msg: format!("no field {}", f.v),
                                        line: f.line,
                                        col: f.col,
                                    })?;
                                StoreT::Field(tb, fi as u32)
                            }
                            _ => return diag("field stores need a struct", l, c),
                        }
                    }
                    _ => return diag("store target must be `v[i]` or `s.f`", l, c),
                };
                let te = self.expr(expr, env, loops)?;
                // check the stored type matches the slot type
                let want = match &st {
                    StoreT::Index(tb, _) => match &tb.ty {
                        RTy::Vec(t) => (**t).clone(),
                        RTy::Tensor(_) => RTy::F64,
                        _ => unreachable!(),
                    },
                    StoreT::Field(tb, fi) => match &tb.ty {
                        RTy::Struct(si) => self.structs[*si as usize].1[*fi as usize].1.clone(),
                        _ => unreachable!(),
                    },
                };
                if te.ty != want {
                    return diag(
                        format!("store wants {}, found {}", want.show(&self.structs), te.ty.show(&self.structs)),
                        expr.line,
                        expr.col,
                    );
                }
                Ok(HS::Store { target: st, expr: te })
            }
            Stmt::If { cond, then, els } => {
                let tc = self.expr(cond, env, loops)?;
                if tc.ty != RTy::Bool {
                    return diag(format!("if needs bool, found {}", tc.ty.show(&self.structs)), cond.line, cond.col);
                }
                env.push(HashMap::new());
                let t = self.stmts(then, env, ret, loops)?;
                env.pop();
                env.push(HashMap::new());
                let e = self.stmts(els, env, ret, loops)?;
                env.pop();
                Ok(HS::If { cond: tc, then: t, els: e })
            }
            Stmt::While { cond, body } => {
                let tc = self.expr(cond, env, loops)?;
                if tc.ty != RTy::Bool {
                    return diag(format!("while needs bool, found {}", tc.ty.show(&self.structs)), cond.line, cond.col);
                }
                env.push(HashMap::new());
                let b = self.stmts(body, env, ret, loops + 1)?;
                env.pop();
                Ok(HS::While { cond: tc, body: b })
            }
            Stmt::For { var, lo, hi, body } => {
                let tlo = self.expr(lo, env, loops)?;
                let thi = self.expr(hi, env, loops)?;
                if tlo.ty != RTy::I64 || thi.ty != RTy::I64 {
                    return diag("for ranges need i64 bounds", l, c);
                }
                env.push(HashMap::new());
                env.last_mut()
                    .unwrap()
                    .insert(var.v.clone(), (RTy::I64, false));
                let b = self.stmts(body, env, ret, loops + 1)?;
                env.pop();
                Ok(HS::For {
                    var: var.v.clone(),
                    lo: tlo,
                    hi: thi,
                    body: b,
                })
            }
            Stmt::Break => {
                if loops == 0 {
                    return diag("break outside a loop", l, c);
                }
                Ok(HS::Break)
            }
            Stmt::Continue => {
                if loops == 0 {
                    return diag("continue outside a loop", l, c);
                }
                Ok(HS::Continue)
            }
            Stmt::Return(e) => {
                let te = match e {
                    Some(e) => Some(self.expr(e, env, loops)?),
                    None => None,
                };
                match (ret, &te) {
                    (RTy::Unit, None) => {}
                    (RTy::Unit, Some(_)) => {
                        return diag("returning a value from a unit fn", l, c)
                    }
                    (w, Some(t)) if *w == t.ty => {}
                    (w, Some(t)) => {
                        return diag(
                            format!("return wants {}, found {}", w.show(&self.structs), t.ty.show(&self.structs)),
                            l,
                            c,
                        )
                    }
                    (_, None) => {
                        return diag("bare return in a fn that returns a value", l, c)
                    }
                }
                Ok(HS::Return(te))
            }
            Stmt::Expr(e) => Ok(HS::Expr(self.expr(e, env, loops)?)),
        }
    }
}

/// does this statement list definitely exit with a value?
fn definitely_returns(stmts: &[HS]) -> bool {
    match stmts.last() {
        Some(HS::Return(Some(_))) | Some(HS::Expr(_)) => true,
        Some(HS::If { then, els, .. }) => {
            !els.is_empty() && definitely_returns(then) && definitely_returns(els)
        }
        _ => false,
    }
}

/// check the host stratum against the ane programs it can call.
pub(crate) fn check(
    _p: &Program,
    hostfns: &[Sp<HFn>],
    structs: &[(String, Vec<(String, HTy)>)],
    ck: &Ck,
    ane: &[(String, TProgram)],
    tables: &[(String, Ty, Option<String>)],
) -> CResult<HProgram> {
    let mut hk = Hk {
        fns: HashMap::new(),
        fndecl: hostfns.iter().map(|f| f.v.clone()).collect(),
        structs: Vec::new(),
        progs: HashMap::new(),
        globals: HashMap::new(),
        sealed: std::collections::HashSet::new(),
        glist: Vec::new(),
        states: HashMap::new(),
    };
    // struct table — fields resolved now so recursive references fail
    for (name, fields) in structs {
        let mut rfields = Vec::new();
        for (fname, fty) in fields {
            rfields.push((fname.clone(), hk.resolve(fty, 0, 0)?));
        }
        if hk.structs.iter().any(|(n, _)| n == name) {
            return diag(format!("duplicate struct {name}"), 0, 0);
        }
        hk.structs.push((name.clone(), rfields));
    }
    // ane program values — first-class, callable
    for (i, (name, prog)) in ane.iter().enumerate() {
        let pargs: Vec<RTy> = prog
            .ios
            .iter()
            .filter(|(_, d, _)| *d == IoDir::In)
            .map(|(_, _, t)| RTy::Tensor(t.clone()))
            .collect();
        let pret = prog
            .ios
            .iter()
            .find(|(_, d, _)| *d == IoDir::Out)
            .map(|(_, _, t)| RTy::Tensor(t.clone()))
            .unwrap_or(RTy::Unit);
        let ty = RTy::AneFn(pargs.clone(), Box::new(pret.clone()));
        hk.progs.insert(name.clone(), (i as u32, pargs, pret));
        let gi = hk.glist.len() as u32;
        hk.glist.push((name.clone(), GVal::Prog { idx: i as u32 }, ty.clone()));
        hk.globals.insert(name.clone(), (gi, ty));
    }
    // params: host-visible unless sealed — the constitutional boundary.
    let mut pnames: Vec<_> = ck.params.keys().cloned().collect();
    pnames.sort();
    for pname in pnames {
        let (qual, ty, src) = ck.params.get(&pname).unwrap().clone();
        let ct = resolve_ty_pub(&ty)?;
        if qual == Qual::Sealed {
            // sealed params are physically unnameable in host code —
            // they never enter the global table.
            hk.sealed.insert(pname);
            continue;
        }
        let rty = RTy::Tensor(ct.clone());
        let gi = hk.glist.len() as u32;
        hk.glist.push(
            (pname.clone(), GVal::Param { name: pname.clone(), ty: ct, src }, rty.clone()),
        );
        hk.globals.insert(pname, (gi, rty));
    }
    // tables — host data sources, indexable like params.
    for (tname, tty, tsrc) in tables {
        let ct = resolve_ty_pub(tty)?;
        if hk.globals.contains_key(tname) {
            return diag(format!("{tname} declared twice"), 0, 0);
        }
        let rty = RTy::Tensor(ct.clone());
        let gi = hk.glist.len() as u32;
        hk.glist.push((
            tname.clone(),
            GVal::Table { name: tname.clone(), ty: ct, src: tsrc.clone() },
            rty.clone(),
        ));
        hk.globals.insert(tname.clone(), (gi, rty));
    }
    // state fields — the on-chip mirror, readable by the organism.
    let mut snames: Vec<_> = ck.states.keys().cloned().collect();
    snames.sort();
    for sname in snames {
        let fields = ck.states.get(&sname).unwrap().clone();
        let mut fmap = HashMap::new();
        for (fname, fty) in fields {
            let ct = resolve_ty_pub(&fty)?;
            let key = format!("{sname}.{fname}");
            let port = format!("{sname}_{fname}");
            let rty = RTy::Tensor(ct.clone());
            let gi = hk.glist.len() as u32;
            hk.glist.push(
                (
                    key.clone(),
                    GVal::State { key: key.clone(), port, ty: ct },
                    rty.clone(),
                ),
            );
            hk.globals.insert(key.clone(), (gi, rty));
            fmap.insert(fname, gi);
        }
        hk.states.insert(sname, fmap);
    }
    // fn name → index
    for (i, f) in hk.fndecl.iter().enumerate() {
        if hk.fns.insert(f.name.clone(), i as u32).is_some() {
            return diag(format!("duplicate host fn {}", f.name), 0, 0);
        }
        if hk.progs.contains_key(&f.name) {
            return diag(format!("{} is both an ane program and a host fn", f.name), 0, 0);
        }
    }
    // check each fn body
    let mut fns = Vec::new();
    for (i, f) in hk.fndecl.clone().iter().enumerate() {
        let mut env: HEnv = vec![HashMap::new()];
        let mut args = Vec::new();
        for (an, aty) in &f.args {
            let rt = hk.resolve(&aty.v, aty.line, aty.col)?;
            env[0].insert(an.v.clone(), (rt.clone(), false));
            args.push((an.v.clone(), rt));
        }
        let ret = match &f.ret {
            Some(rt) => hk.resolve(&rt.v, rt.line, rt.col)?,
            None => RTy::Unit,
        };
        let decl = hostfns[i].clone();
        let body = hk.stmts(&decl.v.body, &mut env, &ret, 0)?;
        if ret != RTy::Unit && !definitely_returns(&body) {
            return diag(
                format!("{} may reach its end without returning {}", f.name, ret.show(&hk.structs)),
                decl.line,
                decl.col,
            );
        }
        fns.push(HFnT {
            name: f.name.clone(),
            args,
            ret,
            body,
        });
    }
    let entry = hk.fns.get("main").copied().map(|i| i as usize);
    Ok(HProgram {
        fns,
        structs: hk.structs,
        globals: hk.glist,
        entry,
    })
}

/// literal-dims resolution for param/state types in host context.
fn resolve_ty_pub(ty: &Ty) -> CResult<CTy> {
    let mut dims = Vec::new();
    for d in &ty.dims {
        match d {
            Dim::Lit(n) => dims.push(*n),
            Dim::Name(s) => {
                return diag(format!("unbound dim {s} in host-visible type"), 0, 0)
            }
        }
    }
    Ok(CTy { dt: ty.dt, dims })
}
