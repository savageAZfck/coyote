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
use std::collections::{HashMap, HashSet};

/// resolved host type.
#[derive(Debug, Clone, PartialEq)]
pub enum RTy {
    I64,
    F64,
    Bool,
    Str,
    Unit,
    Token,
    Dfa,
    Tensor(CTy),
    Vec(Box<RTy>),
    Map(Box<RTy>, Box<RTy>),
    Struct(u32),
    Enum(u32),
    /// `ane(args) -> ret` — a program value.
    AneFn(Vec<RTy>, Box<RTy>),
}

pub type StructDef = (String, Vec<(String, RTy)>);
pub type EnumDef = (String, Vec<(String, Vec<RTy>)>);

impl RTy {
    pub fn show(&self, structs: &[StructDef], enums: &[EnumDef]) -> String {
        match self {
            RTy::I64 => "i64".into(),
            RTy::F64 => "f64".into(),
            RTy::Bool => "bool".into(),
            RTy::Str => "str".into(),
            RTy::Unit => "unit".into(),
            RTy::Token => "token".into(),
            RTy::Dfa => "dfa".into(),
            RTy::Tensor(t) => t.show(),
            RTy::Vec(t) => format!("[{}]", t.show(structs, enums)),
            RTy::Map(k, v) => format!("{{{}: {}}}", k.show(structs, enums), v.show(structs, enums)),
            RTy::Struct(i) => structs[*i as usize].0.clone(),
            RTy::Enum(i) => enums[*i as usize].0.clone(),
            RTy::AneFn(a, r) => format!(
                "ane({}) -> {}",
                a.iter().map(|t| t.show(structs, enums)).collect::<Vec<_>>().join(", "),
                r.show(structs, enums),
            ),
        }
    }
}

/// native function table — indices are stable; vm.rs matches by index.
pub const NATIVE_NAMES: &[&str] = &[
    "print",         // (any...) -> unit — stdout
    "str",           // (any) -> str
    "len",           // (vec|str|tensor|map) -> i64
    "push",          // (vec<T>, T) -> unit
    "argmax",        // (f16[N]) -> i64
    "i64",           // (i64|f64|bool|token) -> i64
    "f64",           // (i64|f64) -> f64
    "onehot",        // (i64 i, i64 const n) -> f16[n]
    "causal",        // (i64 upto, i64 const n) -> f16[n]
    "rope_cos",      // (i64 pos, i64 const d) -> f16[d]
    "rope_sin",      // (i64 pos, i64 const d) -> f16[d]
    "zeros",         // (i64 const n) -> f16[n]
    "self_token",    // () -> token — witnessed identity (id, epoch)
    "witness",       // (str tag, any val) -> unit — hash-chained log entry
    "die",           // () -> unit — halt the organism with attestation
    "migrate",       // (str path) -> unit — serialize full VM state
    "rand",          // () -> f64 — deterministic PRNG, stoch-typed
    "has",           // (map<K,V>, K) -> bool
    "keys",          // (map<K,V>) -> [K]
    "contains",      // (str, str) -> bool
    "split",         // (str, str) -> [str]
    "slice_str",     // (str, i64, i64) -> str
    "dfa_compile",   // ([str]) -> dfa — Aho-Corasick compiler (macuahuitl)
    "dfa_scan",      // (dfa, str) -> i64 — streaming DFA match index (-1 if clean)
    "quorum",        // (str topic, [Vote]) -> QuorumVerdict — dissenting council (tokala)
    "propose_patch", // (str target, str old, str new, str reason) -> Decision (akicita/xipe, cap `patch`)
    "erase",         // (str subject, str payload) -> str — certified erasure (camazotz, cap `erase`)
    "read_file",     // (str path) -> str (cap `fs`)
    "write_file",    // (str path, str content) -> unit (cap `fs`)
    "cortex_ask",    // (str prompt) -> str — SLICKS bridge (cap `cortex`, stoch)
    "sha256",        // (str) -> str — hex sha-256, det (firing chain, dedupe fingerprints)
    "parse_num",     // (str) -> f64 — NaN on non-numeric, det (threshold triggers)
];

/// Capability required by a native, if any.
pub fn native_cap(name: &str) -> Option<&'static str> {
    Some(match name {
        "propose_patch" => "patch",
        "erase" => "erase",
        "read_file" | "write_file" => "fs",
        "cortex_ask" => "cortex",
        _ => return None,
    })
}

/// Whether a native is intrinsically stochastic (`stoch`).
pub fn native_is_stoch(name: &str) -> bool {
    matches!(name, "rand" | "cortex_ask")
}

/// Canonicalize `cap::net` and `net` to the same capability token.
pub fn norm_cap(s: &str) -> String {
    s.strip_prefix("cap::").unwrap_or(s).to_string()
}

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
    /// call to an `extern` FFI function bound from Rust or Swift.
    CallExt(u32, Vec<HT>),
    /// direct call of a known program — `layer(x, pos)`.
    CallAne(u32, Vec<HT>),
    /// call through a program-typed value — `p(x)` where p: ane(...).
    CallDyn(Box<HT>, Vec<HT>),
    Native(u32, Vec<HT>),
    Index(Box<HT>, Box<HT>),
    Field(Box<HT>, u32),
    VecLit(Vec<HT>),
    MapLit(Vec<(HT, HT)>),
    /// struct idx + field values in declaration order.
    StructLit(u32, Vec<HT>),
    /// enum idx + variant idx + payload values.
    EnumLit(u32, u32, Vec<HT>),
}

#[derive(Debug, Clone)]
pub struct HT {
    pub e: HE,
    pub ty: RTy,
}

#[derive(Debug, Clone)]
pub enum StoreT {
    /// `v[i] = e` — vec element, map entry, or tensor element write.
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
    Match {
        expr: HT,
        /// `(Some(variant_idx) | None for wildcard, bound vars, body)`
        arms: Vec<(Option<u32>, Vec<(String, RTy)>, Vec<HS>)>,
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
    pub is_reflex: bool,
    pub det: bool,
    pub caps: Vec<String>,
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

#[derive(Debug, Clone)]
pub struct HExtern {
    pub name: String,
    pub args: Vec<RTy>,
    pub ret: RTy,
    pub det: bool,
    pub caps: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HProgram {
    pub fns: Vec<HFnT>,
    pub externs: Vec<HExtern>,
    pub structs: Vec<StructDef>,
    pub enums: Vec<EnumDef>,
    /// globals in slot order — tensors and program values.
    pub globals: Vec<(String, GVal, RTy)>,
    /// `main`'s index into fns, if present.
    pub entry: Option<usize>,
}

struct Hk {
    fns: HashMap<String, u32>,
    fndecl: Vec<HFn>,
    fnlocs: Vec<(usize, usize)>,
    externs: Vec<HExtern>,
    ext_map: HashMap<String, u32>,
    structs: Vec<StructDef>,
    enums: Vec<EnumDef>,
    /// ane program name → (index, arg types, ret type)
    progs: HashMap<String, (u32, Vec<RTy>, RTy)>,
    /// param/state name → (global idx, RTy, sealed)
    globals: HashMap<String, (u32, RTy)>,
    /// sealed params — naming one here is the constitutional error.
    sealed: HashSet<String>,
    glist: Vec<(String, GVal, RTy)>,
    /// state block name → field name → global idx
    states: HashMap<String, HashMap<String, u32>>,
    /// call-graph tracking per current fn index
    cur_fn: usize,
    callees: Vec<HashSet<(usize, usize, usize)>>,
    direct_caps: Vec<Vec<(String, usize, usize)>>,
    direct_stoch: Vec<Option<(String, usize, usize)>>,
    direct_steering: Vec<Option<(String, usize, usize)>>,
}

type HEnv = Vec<HashMap<String, (RTy, bool)>>; // name → (ty, mutable)

fn lookup<'a>(env: &'a HEnv, n: &str) -> Option<&'a (RTy, bool)> {
    env.iter().rev().find_map(|s| s.get(n))
}

impl Hk {
    fn show(&self, t: &RTy) -> String {
        t.show(&self.structs, &self.enums)
    }

    fn resolve(&self, t: &HTy, l: usize, c: usize) -> CResult<RTy> {
        Ok(match t {
            HTy::I64 => RTy::I64,
            HTy::F64 => RTy::F64,
            HTy::Bool => RTy::Bool,
            HTy::Str => RTy::Str,
            HTy::Unit => RTy::Unit,
            HTy::Token => RTy::Token,
            HTy::Dfa => RTy::Dfa,
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
            HTy::Map(kt, vt) => {
                let rk = self.resolve(kt, l, c)?;
                if !matches!(rk, RTy::Str | RTy::I64) {
                    return diag(
                        format!("map keys must be str or i64, found {}", self.show(&rk)),
                        l,
                        c,
                    );
                }
                let rv = self.resolve(vt, l, c)?;
                RTy::Map(Box::new(rk), Box::new(rv))
            }
            HTy::Struct(name) => {
                if let Some(i) = self.structs.iter().position(|(n, _)| n == name) {
                    RTy::Struct(i as u32)
                } else if let Some(i) = self.enums.iter().position(|(n, _)| n == name) {
                    RTy::Enum(i as u32)
                } else {
                    return diag(format!("no struct or enum {name:?}"), l, c);
                }
            }
            HTy::AneFn(args, ret) => RTy::AneFn(
                args.iter()
                    .map(|a| self.resolve(a, l, c))
                    .collect::<CResult<Vec<_>>>()?,
                Box::new(self.resolve(ret, l, c)?),
            ),
        })
    }

    fn find_enum_variant(&self, s: &str) -> Option<(u32, u32, Vec<RTy>)> {
        let (ename, vname) = s.split_once("::")?;
        let ei = self.enums.iter().position(|(n, _)| n == ename)?;
        let vi = self.enums[ei].1.iter().position(|(vn, _)| vn == vname)?;
        let ptys = self.enums[ei].1[vi].1.clone();
        Some((ei as u32, vi as u32, ptys))
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
                                format!("vec literal mixes {} and {}", self.show(w), self.show(&t.ty)),
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
            Expr::MapLit(items) => {
                if items.is_empty() {
                    return diag("empty map literal needs a type annotation — `let m: {str: i64} = {}`", l, c);
                }
                let mut titems = Vec::new();
                let mut kty: Option<RTy> = None;
                let mut vty: Option<RTy> = None;
                for (ke, ve) in items {
                    let tk = self.expr(ke, env, loops)?;
                    if !matches!(tk.ty, RTy::Str | RTy::I64) {
                        return diag(
                            format!("map keys must be str or i64, found {}", self.show(&tk.ty)),
                            ke.line,
                            ke.col,
                        );
                    }
                    let tv = self.expr(ve, env, loops)?;
                    match &kty {
                        None => kty = Some(tk.ty.clone()),
                        Some(w) if *w == tk.ty => {}
                        Some(w) => {
                            return diag(
                                format!("map literal mixes key types {} and {}", self.show(w), self.show(&tk.ty)),
                                ke.line,
                                ke.col,
                            )
                        }
                    }
                    match &vty {
                        None => vty = Some(tv.ty.clone()),
                        Some(w) if *w == tv.ty => {}
                        Some(w) => {
                            return diag(
                                format!("map literal mixes value types {} and {}", self.show(w), self.show(&tv.ty)),
                                ve.line,
                                ve.col,
                            )
                        }
                    }
                    titems.push((tk, tv));
                }
                let ty = RTy::Map(Box::new(kty.unwrap()), Box::new(vty.unwrap()));
                Ok(HT {
                    e: HE::MapLit(titems),
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
                                self.show(&decl[fi].1),
                                self.show(&t.ty)
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
                if let Some((ei, vi, ptys)) = self.find_enum_variant(s) {
                    if !ptys.is_empty() {
                        return diag(
                            format!("variant {s} expects {} payload arg(s)", ptys.len()),
                            l,
                            c,
                        );
                    }
                    return Ok(HT {
                        e: HE::EnumLit(ei, vi, Vec::new()),
                        ty: RTy::Enum(ei),
                    });
                }
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
                        _ => diag(format!("- requires i64/f64, found {}", self.show(&a.ty)), l, c),
                    },
                    UnOp::Not => match a.ty {
                        RTy::Bool => Ok(HT {
                            e: HE::Un(*op, Box::new(a)),
                            ty: RTy::Bool,
                        }),
                        _ => diag(format!("! requires bool, found {}", self.show(&a.ty)), l, c),
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
                                format!("+ needs matching i64/f64/str, found {} and {}", self.show(&a.ty), self.show(&b.ty)),
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
                                format!("arithmetic needs matching i64/f64, found {} and {}", self.show(&a.ty), self.show(&b.ty)),
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
                                format!("== needs matching types, found {} and {}", self.show(&a.ty), self.show(&b.ty)),
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
                                format!("ordering needs i64/f64/str, found {} and {}", self.show(&a.ty), self.show(&b.ty)),
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
                let ty = match &tb.ty {
                    RTy::Map(kt, vt) => {
                        if ti.ty != **kt {
                            return diag(
                                format!("map key must be {}, found {}", self.show(kt), self.show(&ti.ty)),
                                idx.line,
                                idx.col,
                            );
                        }
                        (**vt).clone()
                    }
                    RTy::Vec(t) => {
                        if ti.ty != RTy::I64 {
                            return diag(format!("index must be i64, found {}", self.show(&ti.ty)), idx.line, idx.col);
                        }
                        (**t).clone()
                    }
                    RTy::Tensor(ct) => {
                        if ti.ty != RTy::I64 {
                            return diag(format!("index must be i64, found {}", self.show(&ti.ty)), idx.line, idx.col);
                        }
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
                    RTy::Str => {
                        if ti.ty != RTy::I64 {
                            return diag(format!("index must be i64, found {}", self.show(&ti.ty)), idx.line, idx.col);
                        }
                        RTy::Str
                    }
                    _ => {
                        return diag(
                            format!("cannot index {}", self.show(&tb.ty)),
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
                // `prog.state.field` — a qualified state mirror.
                if let Expr::Field(ibx, is) = &bx.v {
                    if let Expr::Ident(p) = &ibx.v {
                        let qkey = format!("{p}.{}", is.v);
                        if let Some(fields) = self.states.get(&qkey).cloned() {
                            if let Some(&gi) = fields.get(&f.v) {
                                let ty = self
                                    .glist
                                    .get(gi as usize)
                                    .map(|(_, _, t)| t.clone())
                                    .unwrap_or(RTy::Unit);
                                return Ok(HT {
                                    e: HE::Global(gi),
                                    ty,
                                });
                            }
                            return diag(
                                format!("state {qkey} has no field {}", f.v),
                                f.line,
                                f.col,
                            );
                        }
                    }
                }
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
                        format!("{} has no fields", self.show(&tb.ty)),
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
        // enum variant constructor: `Decision::Hold(reason)`
        if let Some((ei, vi, ptys)) = self.find_enum_variant(name) {
            if pos.len() != ptys.len() {
                return diag(
                    format!("{name} expects {} arg(s), found {}", ptys.len(), pos.len()),
                    l,
                    c,
                );
            }
            let mut targs = Vec::new();
            for (i, (want, ae)) in ptys.iter().zip(pos.iter()).enumerate() {
                let t = self.expr(ae, env, loops)?;
                if &t.ty != want {
                    return diag(
                        format!("{name} arg {i} wants {}, found {}", self.show(want), self.show(&t.ty)),
                        ae.line,
                        ae.col,
                    );
                }
                targs.push(t);
            }
            return Ok(HT {
                e: HE::EnumLit(ei, vi, targs),
                ty: RTy::Enum(ei),
            });
        }
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
            self.callees[self.cur_fn].insert((fi as usize, l, c));
            let mut targs = Vec::new();
            for ((an, aty), ae) in f.args.iter().zip(pos.iter()) {
                let want = self.resolve(&aty.v, aty.line, aty.col)?;
                let t = self.expr(ae, env, loops)?;
                if t.ty != want {
                    return diag(
                        format!("{name} arg {} wants {}, found {}", an.v, self.show(&want), self.show(&t.ty)),
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
        // extern fn bound from Rust or Swift?
        if let Some(&ei) = self.ext_map.get(name) {
            let ext = self.externs[ei as usize].clone();
            if pos.len() != ext.args.len() {
                return diag(
                    format!("{name} expects {} args, found {}", ext.args.len(), pos.len()),
                    l,
                    c,
                );
            }
            for cap in &ext.caps {
                self.direct_caps[self.cur_fn].push((cap.clone(), l, c));
            }
            if !ext.det && self.direct_stoch[self.cur_fn].is_none() {
                self.direct_stoch[self.cur_fn] = Some((name.to_string(), l, c));
            }
            let mut targs = Vec::new();
            for (i, (want, ae)) in ext.args.iter().zip(pos.iter()).enumerate() {
                let t = self.expr(ae, env, loops)?;
                if &t.ty != want {
                    return diag(
                        format!("{name} arg {i} wants {}, found {}", self.show(want), self.show(&t.ty)),
                        ae.line,
                        ae.col,
                    );
                }
                targs.push(t);
            }
            return Ok(HT {
                e: HE::CallExt(ei, targs),
                ty: ext.ret,
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
            if self.direct_steering[self.cur_fn].is_none() {
                self.direct_steering[self.cur_fn] = Some((format!("ane dispatch `{name}`"), l, c));
            }
            let mut targs = Vec::new();
            for (i, (want, ae)) in pargs.iter().zip(pos.iter()).enumerate() {
                let t = self.expr(ae, env, loops)?;
                if &t.ty != want {
                    return diag(
                        format!("{name} arg {i} wants {}, found {}", self.show(want), self.show(&t.ty)),
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
            if let Some(cap) = native_cap(name) {
                self.direct_caps[self.cur_fn].push((cap.to_string(), l, c));
            }
            if native_is_stoch(name) && self.direct_stoch[self.cur_fn].is_none() {
                self.direct_stoch[self.cur_fn] = Some((name.to_string(), l, c));
            }
            if matches!(name, "migrate" | "die") && self.direct_steering[self.cur_fn].is_none() {
                self.direct_steering[self.cur_fn] = Some((name.to_string(), l, c));
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
            if self.direct_steering[self.cur_fn].is_none() {
                self.direct_steering[self.cur_fn] = Some((format!("dynamic ane call `{name}`"), l, c));
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
                        format!("{name} arg {i} wants {}, found {}", self.show(want), self.show(&t.ty)),
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
        match name {
            "print" => Ok(RTy::Unit),
            "str" => {
                arity(1)?;
                Ok(RTy::Str)
            }
            "len" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::Vec(_) | RTy::Map(_, _) | RTy::Str | RTy::Tensor(_) => Ok(RTy::I64),
                    t => diag(format!("len on {}", self.show(t)), l, c),
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
                    t => diag(format!("i64() on {}", self.show(t)), l, c),
                }
            }
            "f64" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::I64 | RTy::F64 => Ok(RTy::F64),
                    t => diag(format!("f64() on {}", self.show(t)), l, c),
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
            "has" => {
                arity(2)?;
                match &args[0].ty {
                    RTy::Map(kt, _) if **kt == args[1].ty => Ok(RTy::Bool),
                    _ => diag("has(map<K, V>, K) expected", l, c),
                }
            }
            "keys" => {
                arity(1)?;
                match &args[0].ty {
                    RTy::Map(kt, _) => Ok(RTy::Vec(kt.clone())),
                    _ => diag("keys(map<K, V>) expected", l, c),
                }
            }
            "contains" => {
                arity(2)?;
                if args[0].ty != RTy::Str || args[1].ty != RTy::Str {
                    return diag("contains(s: str, sub: str) expected", l, c);
                }
                Ok(RTy::Bool)
            }
            "split" => {
                arity(2)?;
                if args[0].ty != RTy::Str || args[1].ty != RTy::Str {
                    return diag("split(s: str, sep: str) expected", l, c);
                }
                Ok(RTy::Vec(Box::new(RTy::Str)))
            }
            "slice_str" => {
                arity(3)?;
                if args[0].ty != RTy::Str || args[1].ty != RTy::I64 || args[2].ty != RTy::I64 {
                    return diag("slice_str(s: str, lo: i64, hi: i64) expected", l, c);
                }
                Ok(RTy::Str)
            }
            "sha256" | "parse_num" => {
                arity(1)?;
                if args[0].ty != RTy::Str {
                    return diag(format!("{name}(s: str) expected"), l, c);
                }
                Ok(if name == "sha256" { RTy::Str } else { RTy::F64 })
            }
            "dfa_compile" => {
                arity(1)?;
                if args[0].ty != RTy::Vec(Box::new(RTy::Str)) {
                    return diag("dfa_compile(patterns: [str]) expected", l, c);
                }
                Ok(RTy::Dfa)
            }
            "dfa_scan" => {
                arity(2)?;
                if args[0].ty != RTy::Dfa || args[1].ty != RTy::Str {
                    return diag("dfa_scan(machine: dfa, text: str) expected", l, c);
                }
                Ok(RTy::I64)
            }
            "quorum" => {
                arity(2)?;
                let vote_si = self
                    .structs
                    .iter()
                    .position(|(n, _)| n == "Vote")
                    .unwrap_or(0) as u32;
                let qv_si = self
                    .structs
                    .iter()
                    .position(|(n, _)| n == "QuorumVerdict")
                    .unwrap_or(1) as u32;
                if args[0].ty != RTy::Str || args[1].ty != RTy::Vec(Box::new(RTy::Struct(vote_si))) {
                    return diag("quorum(topic: str, votes: [Vote]) expected", l, c);
                }
                Ok(RTy::Struct(qv_si))
            }
            "propose_patch" => {
                arity(4)?;
                if args.iter().any(|a| a.ty != RTy::Str) {
                    return diag("propose_patch(target: str, old: str, new: str, reason: str) expected", l, c);
                }
                let dec_ei = self
                    .enums
                    .iter()
                    .position(|(n, _)| n == "Decision")
                    .unwrap_or(1) as u32;
                Ok(RTy::Enum(dec_ei))
            }
            "erase" => {
                arity(2)?;
                if args[0].ty != RTy::Str || args[1].ty != RTy::Str {
                    return diag("erase(subject: str, payload: str) expected", l, c);
                }
                Ok(RTy::Str)
            }
            "read_file" | "cortex_ask" => {
                arity(1)?;
                if args[0].ty != RTy::Str {
                    return diag(format!("{name}(arg: str) expected"), l, c);
                }
                Ok(RTy::Str)
            }
            "write_file" => {
                arity(2)?;
                if args[0].ty != RTy::Str || args[1].ty != RTy::Str {
                    return diag("write_file(path: str, content: str) expected", l, c);
                }
                Ok(RTy::Unit)
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
                // `let m: {str: i64} = {}` — empty map literal with annotation.
                if let (Some(ht), Expr::MapLit(items)) = (hty, &expr.v) {
                    if items.is_empty() {
                        let want = self.resolve(&ht.v, ht.line, ht.col)?;
                        if let RTy::Map(_, _) = want {
                            env.last_mut()
                                .unwrap()
                                .insert(name.v.clone(), (want.clone(), *mutable));
                            return Ok(HS::Let {
                                name: name.v.clone(),
                                mutable: *mutable,
                                expr: HT {
                                    e: HE::MapLit(vec![]),
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
                                self.show(&want),
                                self.show(&te.ty)
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
                            self.show(&vty),
                            self.show(&te.ty)
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
                        match &tb.ty {
                            RTy::Map(kt, _) => {
                                if ti.ty != **kt {
                                    return diag(
                                        format!("map key must be {}, found {}", self.show(kt), self.show(&ti.ty)),
                                        l,
                                        c,
                                    );
                                }
                                StoreT::Index(tb, ti)
                            }
                            RTy::Vec(_) | RTy::Tensor(_) => {
                                if ti.ty != RTy::I64 {
                                    return diag("index must be i64", l, c);
                                }
                                StoreT::Index(tb, ti)
                            }
                            _ => return diag("can only store into vecs, maps, and tensors", l, c),
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
                        RTy::Map(_, vt) => (**vt).clone(),
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
                        format!("store wants {}, found {}", self.show(&want), self.show(&te.ty)),
                        expr.line,
                        expr.col,
                    );
                }
                Ok(HS::Store { target: st, expr: te })
            }
            Stmt::If { cond, then, els } => {
                let tc = self.expr(cond, env, loops)?;
                if tc.ty != RTy::Bool {
                    return diag(format!("if needs bool, found {}", self.show(&tc.ty)), cond.line, cond.col);
                }
                env.push(HashMap::new());
                let t = self.stmts(then, env, ret, loops)?;
                env.pop();
                env.push(HashMap::new());
                let e = self.stmts(els, env, ret, loops)?;
                env.pop();
                Ok(HS::If { cond: tc, then: t, els: e })
            }
            Stmt::Match { expr, arms } => {
                let te = self.expr(expr, env, loops)?;
                let ei = match te.ty {
                    RTy::Enum(ei) => ei as usize,
                    _ => {
                        return diag(
                            format!("match requires an enum value, found {}", self.show(&te.ty)),
                            expr.line,
                            expr.col,
                        )
                    }
                };
                let (ename, variants) = self.enums[ei].clone();
                let mut covered = HashSet::new();
                let mut has_wild = false;
                let mut tarms = Vec::new();
                for (pat, body) in arms {
                    match pat {
                        Pat::Wildcard(_, _) => {
                            has_wild = true;
                            env.push(HashMap::new());
                            let tb = self.stmts(body, env, ret, loops)?;
                            env.pop();
                            tarms.push((None, Vec::new(), tb));
                        }
                        Pat::Variant { name, binds } => {
                            let vname = match name.v.split_once("::") {
                                Some((en, vn)) => {
                                    if en != ename {
                                        return diag(
                                            format!("pattern {} is not a variant of {ename}", name.v),
                                            name.line,
                                            name.col,
                                        );
                                    }
                                    vn
                                }
                                None => name.v.as_str(),
                            };
                            let vi = variants
                                .iter()
                                .position(|(vn, _)| vn == vname)
                                .ok_or(())
                                .map_err(|_| Diag {
                                    msg: format!("enum {ename} has no variant {vname}"),
                                    line: name.line,
                                    col: name.col,
                                })?;
                            let ptys = &variants[vi].1;
                            if binds.len() != ptys.len() {
                                return diag(
                                    format!(
                                        "{ename}::{vname} has {} payload field(s), pattern binds {}",
                                        ptys.len(),
                                        binds.len()
                                    ),
                                    name.line,
                                    name.col,
                                );
                            }
                            covered.insert(vi);
                            env.push(HashMap::new());
                            let mut tbinds = Vec::new();
                            for (b, pty) in binds.iter().zip(ptys.iter()) {
                                env.last_mut()
                                    .unwrap()
                                    .insert(b.v.clone(), (pty.clone(), false));
                                tbinds.push((b.v.clone(), pty.clone()));
                            }
                            let tb = self.stmts(body, env, ret, loops)?;
                            env.pop();
                            tarms.push((Some(vi as u32), tbinds, tb));
                        }
                    }
                }
                if !has_wild {
                    for (vi, (vname, _)) in variants.iter().enumerate() {
                        if !covered.contains(&vi) {
                            return diag(
                                format!(
                                    "non-exhaustive match on {ename}: missing variant `{ename}::{vname}`"
                                ),
                                l,
                                c,
                            );
                        }
                    }
                }
                Ok(HS::Match {
                    expr: te,
                    arms: tarms,
                })
            }
            Stmt::While { cond, body } => {
                let tc = self.expr(cond, env, loops)?;
                if tc.ty != RTy::Bool {
                    return diag(format!("while needs bool, found {}", self.show(&tc.ty)), cond.line, cond.col);
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
                            format!("return wants {}, found {}", self.show(w), self.show(&t.ty)),
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
        Some(HS::Match { arms, .. }) => {
            !arms.is_empty() && arms.iter().all(|(_, _, b)| definitely_returns(b))
        }
        _ => false,
    }
}

/// check the host stratum against the ane programs it can call.
pub(crate) fn check(
    _p: &Program,
    hostfns: &[Sp<HFn>],
    externs: &[Sp<ExternFn>],
    structs: &[(String, Vec<(String, HTy)>)],
    enums: &[(String, Vec<(String, Vec<HTy>)>)],
    declared_caps: &[String],
    ck: &Ck,
    ane: &[(String, TProgram)],
    tables: &[(String, Ty, Option<String>)],
) -> CResult<HProgram> {
    let nfns = hostfns.len();
    let mut hk = Hk {
        fns: HashMap::new(),
        fndecl: hostfns.iter().map(|f| f.v.clone()).collect(),
        fnlocs: hostfns.iter().map(|f| (f.line, f.col)).collect(),
        externs: Vec::new(),
        ext_map: HashMap::new(),
        structs: Vec::new(),
        enums: Vec::new(),
        progs: HashMap::new(),
        globals: HashMap::new(),
        sealed: HashSet::new(),
        glist: Vec::new(),
        states: HashMap::new(),
        cur_fn: 0,
        callees: vec![HashSet::new(); nfns],
        direct_caps: vec![Vec::new(); nfns],
        direct_stoch: vec![None; nfns],
        direct_steering: vec![None; nfns],
    };
    // Built-in sovereign organ enums: `Reflex` (axon) and `Decision` (akicita),
    // unless overridden by an explicit declaration in source.
    if !enums.iter().any(|(n, _)| n == "Reflex") {
        hk.enums.push((
            "Reflex".into(),
            vec![
                ("Pass".into(), vec![]),
                ("Alert".into(), vec![RTy::Str]),
                ("Brake".into(), vec![RTy::Str]),
                ("Quarantine".into(), vec![RTy::Str]),
            ],
        ));
    }
    if !enums.iter().any(|(n, _)| n == "Decision") {
        hk.enums.push((
            "Decision".into(),
            vec![
                ("Permit".into(), vec![]),
                ("Hold".into(), vec![RTy::Str]),
                ("Deny".into(), vec![RTy::Str]),
            ],
        ));
    }
    // Built-in sovereign organ structs: `Vote` and `QuorumVerdict` (tokala).
    if !structs.iter().any(|(n, _)| n == "Vote") {
        hk.structs.push((
            "Vote".into(),
            vec![
                ("seat".into(), RTy::Str),
                ("approve".into(), RTy::Bool),
                ("reason".into(), RTy::Str),
            ],
        ));
    }
    if !structs.iter().any(|(n, _)| n == "QuorumVerdict") {
        hk.structs.push((
            "QuorumVerdict".into(),
            vec![
                ("approved".into(), RTy::Bool),
                ("yes".into(), RTy::I64),
                ("no".into(), RTy::I64),
                ("dissent".into(), RTy::Vec(Box::new(RTy::Str))),
                ("hash".into(), RTy::Str),
            ],
        ));
    }
    // Pre-register user enum names with empty variants so structs/enums can
    // reference each other, then resolve payload types.
    let enum_base = hk.enums.len();
    for (name, _) in enums {
        if hk.enums.iter().any(|(n, _)| n == name) || hk.structs.iter().any(|(n, _)| n == name) {
            return diag(format!("duplicate type {name}"), 0, 0);
        }
        hk.enums.push((name.clone(), Vec::new()));
    }
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
    for (idx, (_, variants)) in enums.iter().enumerate() {
        let mut rvars = Vec::new();
        for (vname, ptys) in variants {
            let mut rptys = Vec::new();
            for pty in ptys {
                rptys.push(hk.resolve(pty, 0, 0)?);
            }
            rvars.push((vname.clone(), rptys));
        }
        hk.enums[enum_base + idx].1 = rvars;
    }
    // extern fn table — resolved after structs/enums so extern signatures
    // can reference any type.
    for ef in externs {
        if hk.ext_map.contains_key(&ef.v.name) {
            return diag(format!("duplicate extern fn {}", ef.v.name), ef.line, ef.col);
        }
        let mut rargs = Vec::new();
        for (_, aty) in &ef.v.args {
            rargs.push(hk.resolve(&aty.v, aty.line, aty.col)?);
        }
        let rret = match &ef.v.ret {
            Some(rt) => hk.resolve(&rt.v, rt.line, rt.col)?,
            None => RTy::Unit,
        };
        let caps: Vec<String> = ef.v.caps.iter().map(|c| norm_cap(&c.v)).collect();
        let idx = hk.externs.len() as u32;
        hk.ext_map.insert(ef.v.name.clone(), idx);
        hk.externs.push(HExtern {
            name: ef.v.name.clone(),
            args: rargs,
            ret: rret,
            det: ef.v.det != DetMode::Stoch,
            caps,
        });
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
    let mut used_by: HashMap<String, Vec<String>> = HashMap::new(); // key → programs
    for (pname, tp) in ane {
        for (key, port, ty) in &tp.states {
            used_by.entry(key.clone()).or_default().push(pname.clone());
            let gkey = format!("{pname}.{key}");
            let rty = RTy::Tensor(ty.clone());
            let gi = hk.glist.len() as u32;
            hk.glist.push((
                gkey.clone(),
                GVal::State { key: gkey.clone(), port: port.clone(), ty: ty.clone() },
                rty.clone(),
            ));
            hk.globals.insert(gkey, (gi, rty));
        }
    }
    for (pname, tp) in ane {
        for (key, _, ty) in &tp.states {
            let (sname, fname) = match key.split_once('.') {
                Some(p) => p,
                None => continue,
            };
            // qualified: `layer0.kv.k0lo`
            hk.states
                .entry(format!("{pname}.{sname}"))
                .or_default()
                .insert(fname.to_string(), hk.globals[&format!("{pname}.{key}")].0);
            // unqualified `kv.k0lo` — only when unambiguous
            if used_by.get(key).map(|v| v.len()) == Some(1) {
                let gkey = key.to_string();
                if !hk.globals.contains_key(&gkey) {
                    let rty = RTy::Tensor(ty.clone());
                    let gi = hk.glist.len() as u32;
                    let port = format!("{sname}_{fname}");
                    hk.glist.push((
                        gkey.clone(),
                        GVal::State { key: format!("{pname}.{key}"), port, ty: ty.clone() },
                        rty.clone(),
                    ));
                    hk.globals.insert(gkey, (gi, rty));
                }
                hk.states
                    .entry(sname.to_string())
                    .or_default()
                    .insert(fname.to_string(), hk.globals[key].0);
            }
        }
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
    let reflex_ei = hk
        .enums
        .iter()
        .position(|(n, _)| n == "Reflex")
        .unwrap_or(0) as u32;
    // check each fn body
    let mut fns = Vec::new();
    for (i, f) in hk.fndecl.clone().iter().enumerate() {
        hk.cur_fn = i;
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
        if f.is_reflex && ret != RTy::Enum(reflex_ei) {
            return diag(
                format!(
                    "reflex `{}` must return `Reflex`, found {}",
                    f.name,
                    hk.show(&ret)
                ),
                decl.line,
                decl.col,
            );
        }
        // If a host fn has an explicit `cap a, b` signature, those caps
        // are also required by anyone calling it.
        if let Some(cs) = &f.caps {
            for c in cs {
                hk.direct_caps[i].push((norm_cap(&c.v), c.line, c.col));
            }
        }
        let body = hk.stmts(&decl.v.body, &mut env, &ret, 0)?;
        if ret != RTy::Unit && !definitely_returns(&body) {
            return diag(
                format!("{} may reach its end without returning {}", f.name, hk.show(&ret)),
                decl.line,
                decl.col,
            );
        }
        fns.push(HFnT {
            name: f.name.clone(),
            args,
            ret,
            is_reflex: f.is_reflex,
            det: true,
            caps: Vec::new(),
            body,
        });
    }

    // ---- Constitutional Pass: Determinism (`det`/`stoch`), Reflex Bounds,
    //      and Transitive Capability Soundness (`cap`) ----
    let mut is_stoch: Vec<Option<(String, usize, usize)>> = (0..nfns)
        .map(|i| {
            if let Some(s) = &hk.direct_stoch[i] {
                Some(s.clone())
            } else if hk.fndecl[i].det == DetMode::Stoch {
                Some((hk.fndecl[i].name.clone(), hk.fnlocs[i].0, hk.fnlocs[i].1))
            } else {
                None
            }
        })
        .collect();
    let mut steers: Vec<Option<(String, usize, usize)>> = hk.direct_steering.clone();
    let mut fn_caps: Vec<HashMap<String, (usize, usize)>> = (0..nfns)
        .map(|i| {
            let mut m = HashMap::new();
            for (c, l, col) in &hk.direct_caps[i] {
                m.entry(c.clone()).or_insert((*l, *col));
            }
            m
        })
        .collect();

    loop {
        let mut changed = false;
        for i in 0..nfns {
            for &(callee, cl, cc) in &hk.callees[i] {
                if is_stoch[i].is_none() {
                    if let Some((why, _, _)) = &is_stoch[callee] {
                        is_stoch[i] = Some((
                            format!("{} -> {why}", hk.fndecl[callee].name),
                            cl,
                            cc,
                        ));
                        changed = true;
                    }
                }
                if steers[i].is_none() {
                    if let Some((why, _, _)) = &steers[callee] {
                        steers[i] = Some((
                            format!("{} -> {why}", hk.fndecl[callee].name),
                            cl,
                            cc,
                        ));
                        changed = true;
                    }
                }
                let callee_caps: Vec<(String, (usize, usize))> = fn_caps[callee]
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                for (cap_name, _) in callee_caps {
                    if !fn_caps[i].contains_key(&cap_name) {
                        fn_caps[i].insert(cap_name, (cl, cc));
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let top_caps: HashSet<String> = declared_caps.iter().map(|c| norm_cap(c)).collect();
    for i in 0..nfns {
        let f = &hk.fndecl[i];
        // 1. Reflex non-steering invariant (`axon`: reflexes brake, never steer)
        if f.is_reflex {
            if let Some((why, l, c)) = &steers[i] {
                return diag(
                    format!(
                        "constitutional violation: reflex `{}` cannot steer or halt ({why}) — reflexes brake, never steer",
                        f.name
                    ),
                    *l,
                    *c,
                );
            }
        }
        // 2. Determinism soundness (`det` functions admit no `stoch` interior)
        if f.det == DetMode::Det {
            if let Some((why, l, c)) = &is_stoch[i] {
                let kind = if f.is_reflex { "reflex" } else { "deterministic fn" };
                return diag(
                    format!(
                        "constitutional violation: {kind} `{}` calls stochastic `{why}` — declare `host stoch fn` or remove nondeterminism",
                        f.name
                    ),
                    *l,
                    *c,
                );
            }
        }
        // 3. Per-function explicit `cap` bound (and reflex zero-cap invariant)
        if let Some(explicit) = &f.caps {
            let allowed: HashSet<String> = explicit.iter().map(|c| norm_cap(&c.v)).collect();
            let mut sorted_used: Vec<_> = fn_caps[i].iter().collect();
            sorted_used.sort_by(|a, b| a.0.cmp(b.0));
            for (cap_name, &(l, c)) in sorted_used {
                if !allowed.contains(cap_name) {
                    if f.is_reflex {
                        return diag(
                            format!(
                                "constitutional violation: reflex `{}` uses capability `{cap_name}` — reflexes cannot hold capabilities",
                                f.name
                            ),
                            l,
                            c,
                        );
                    }
                    return diag(
                        format!(
                            "constitutional violation: fn `{}` uses capability `{cap_name}` not in its declared `cap` set",
                            f.name
                        ),
                        l,
                        c,
                    );
                }
            }
        }
        // 4. Program-wide capability soundness (`no cap::X -> no X`)
        let mut sorted_used: Vec<_> = fn_caps[i].iter().collect();
        sorted_used.sort_by(|a, b| a.0.cmp(b.0));
        for (cap_name, &(l, c)) in &sorted_used {
            if !top_caps.contains(*cap_name) {
                return diag(
                    format!(
                        "constitutional violation: fn `{}` requires capability `{cap_name}` — declare `cap {cap_name}` in the program header",
                        f.name
                    ),
                    l,
                    c,
                );
            }
        }
        fns[i].det = is_stoch[i].is_none();
        fns[i].caps = sorted_used.into_iter().map(|(k, _)| k.clone()).collect();
    }

    let entry = hk.fns.get("main").copied().map(|i| i as usize);
    Ok(HProgram {
        fns,
        externs: hk.externs,
        structs: hk.structs,
        enums: hk.enums,
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
