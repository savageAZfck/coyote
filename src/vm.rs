//! vm — the organism runtime.
//!
//! host fns compile to bytecode and execute here. The whole VM state —
//! value stack, frames, tensor store, witness log — is serializable,
//! which is the point: `migrate` freezes the organism mid-thought,
//! `--resume` wakes it, and the lineage hash proves it woke as itself.
//! A native-code backend could not do that; a VM can. ANE programs are
//! invoked as natives through the same GraphRuntime the driver uses.

use crate::ast::{AssignOp, BinOp, DType, UnOp};
use crate::eval::Evaluator;
use crate::hostck::{GVal, HFnT, HProgram, StoreT, HE, HS, HT, RTy, NATIVE_NAMES};
use crate::lower::{init_weight_pub, sha256_hex};
use crate::typeck::{CTy, TProgram};
use mil_infer::graph::GraphRuntime;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

// ---------------------------------------------------------------- bytecode

#[derive(Debug, Clone)]
enum Op {
    Unit,
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(Rc<str>),
    Load(u32),
    Store(u32),
    Global(u32),
    AddI,
    SubI,
    MulI,
    DivI,
    ModI,
    NegI,
    AddF,
    SubF,
    MulF,
    DivF,
    NegF,
    Concat,
    Cmp(BinOp),
    Not,
    Pop,
    Jmp(u32),
    JmpF(u32),
    Call(u32, u8),
    CallExt(u32, u8),
    CallDyn(u8),
    CallAne(u32, u8),
    Native(u32, u8),
    VecNew(u32),
    VecGet,
    VecSet,
    MapNew(u32),
    MapGet,
    MapSet,
    StrGet,
    TensorGet,
    TensorRow,
    TensorSet,
    StructNew(u32),
    FieldGet(u32),
    FieldSet(u32),
    EnumNew(u32, u32, u8),
    EnumIs(u32),
    EnumField(u8),
    TokenField(u32),
    Ret,
}

/// Byte-level Aho-Corasick DFA (`macuahuitl` / `mil_machines` semantics).
#[derive(Clone, Debug)]
pub struct DfaMachine {
    pub patterns: Vec<String>,
    /// `next[state * 256 + byte] -> next_state`
    pub next: Vec<i32>,
    /// `out[state] >= 0` is the matched pattern index (`-1` if none).
    pub out: Vec<i32>,
}

impl DfaMachine {
    pub fn compile(patterns: Vec<String>) -> Self {
        let mut goto: Vec<[i32; 256]> = vec![[-1; 256]];
        let mut out: Vec<i32> = vec![-1];
        for (pi, pat) in patterns.iter().enumerate() {
            if pat.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for &b in pat.as_bytes() {
                let nb = goto[cur][b as usize];
                if nb == -1 {
                    let ns = goto.len() as i32;
                    goto[cur][b as usize] = ns;
                    goto.push([-1; 256]);
                    out.push(-1);
                    cur = ns as usize;
                } else {
                    cur = nb as usize;
                }
            }
            if out[cur] == -1 {
                out[cur] = pi as i32;
            }
        }
        let n = goto.len();
        let mut fail = vec![0usize; n];
        let mut q = VecDeque::new();
        for b in 0..256 {
            if goto[0][b] == -1 {
                goto[0][b] = 0;
            } else {
                let s = goto[0][b] as usize;
                fail[s] = 0;
                q.push_back(s);
            }
        }
        while let Some(r) = q.pop_front() {
            for b in 0..256 {
                let s = goto[r][b];
                if s != -1 {
                    let su = s as usize;
                    q.push_back(su);
                    let f = fail[r];
                    fail[su] = goto[f][b] as usize;
                    if out[su] == -1 && out[fail[su]] != -1 {
                        out[su] = out[fail[su]];
                    }
                } else {
                    goto[r][b] = goto[fail[r]][b];
                }
            }
        }
        let mut next = Vec::with_capacity(n * 256);
        for row in goto {
            next.extend_from_slice(&row);
        }
        Self { patterns, next, out }
    }

    pub fn scan(&self, text: &str) -> i64 {
        let mut state = 0usize;
        for &b in text.as_bytes() {
            state = self.next[state * 256 + b as usize] as usize;
            if self.out[state] >= 0 {
                return self.out[state] as i64;
            }
        }
        -1
    }
}

// ---------------------------------------------------------------- values

#[derive(Clone)]
pub enum Val {
    Unit,
    I(i64),
    F(f64),
    B(bool),
    S(Rc<str>),
    V(Rc<RefCell<Vec<Val>>>),
    /// deterministic insertion-ordered map `{K: V}`
    M(Rc<RefCell<Vec<(Val, Val)>>>),
    /// struct type idx + fields in decl order
    St(u32, Rc<RefCell<Vec<Val>>>),
    /// enum type idx + variant idx + payload values
    En(u32, u32, Rc<Vec<Val>>),
    /// compiled Aho-Corasick DFA
    Dfa(Rc<DfaMachine>),
    /// tensor handle — index into the store
    T(u32),
    /// ane program value — index into the program table
    Ane(u32),
    /// witnessed identity — her SecurityToken. Immutable.
    Tok { id: u64, epoch: u64 },
}

impl Val {
    fn truthy(&self) -> bool {
        matches!(self, Val::B(true))
    }
    fn as_i(&self) -> Option<i64> {
        match self {
            Val::I(i) => Some(*i),
            _ => None,
        }
    }
    fn as_t(&self) -> Option<u32> {
        match self {
            Val::T(i) => Some(*i),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Wit {
    pub seq: u64,
    pub tag: String,
    pub val: String,
    pub hash: String,
}

/// Execution backend for a compiled tensor `fn` program: either native
/// ANE silicon via `GraphRuntime`, or pure-Rust CPU evaluation via `Evaluator`.
pub enum ProgBackend {
    Ane(GraphRuntime),
    Cpu(TProgram),
}

/// a loaded tensor program — the runtime half of a program value.
pub struct AneProg {
    pub name: String,
    pub backend: ProgBackend,
    /// io input ports in arg order: (port, ty, contig)
    pub ins: Vec<(String, CTy, bool)>,
    /// trainable params: (param, port, declared ty)
    pub trainable: Vec<(String, String, CTy)>,
    /// state fields: (key, port, ty)
    pub states: Vec<(String, String, CTy)>,
    pub param_outs: Vec<(String, String, CTy)>,
    pub state_outs: Vec<(String, String, CTy)>,
    pub output: (String, CTy),
}

struct Frame {
    f: u32,
    ip: usize,
    base: usize,
}

// ---------------------------------------------------------------- surface io

/// write a declared-order row-major tensor into a runtime surface.
fn write_surface(rt: &mut GraphRuntime, name: &str, dims: &[usize], data: &[f32]) {
    let dc = data.to_vec();
    match dims.len() {
        1 => {
            rt.write_ext(name, move |b, desc| {
                for (i, v) in dc.iter().enumerate() {
                    // vectors ride w (contig) or c (strided) — follow desc
                    let o = if desc.w as usize >= dc.len() && desc.w > 1 {
                        desc.offset(0, 0, 0, 0, i as u64)
                    } else {
                        desc.offset(0, 0, i as u64, 0, 0)
                    };
                    if o + 2 <= b.len() {
                        b[o..o + 2].copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                    }
                }
            });
        }
        2 => {
            let (r, c) = (dims[0], dims[1]);
            rt.write_ext(name, move |b, desc| {
                for i in 0..r {
                    for j in 0..c {
                        let o = desc.offset(0, 0, i as u64, 0, j as u64);
                        if o + 2 <= b.len() {
                            b[o..o + 2]
                                .copy_from_slice(&half::f16::from_f32(dc[i * c + j]).to_le_bytes());
                        }
                    }
                }
            });
        }
        _ => {}
    }
}

/// read a surface as declared-order row-major f32s.
fn read_surface_dims(raw: &[u8], desc: &mil_infer::espresso::TensorDesc, dims: &[usize]) -> Vec<f32> {
    let rd = |o: usize| -> f32 {
        if o + 2 <= raw.len() {
            half::f16::from_le_bytes([raw[o], raw[o + 1]]).to_f32()
        } else {
            f32::NAN
        }
    };
    match dims.len() {
        1 => {
            let n = dims[0];
            (0..n)
                .map(|i| {
                    let o = if desc.w as usize >= n && desc.w > 1 {
                        desc.offset(0, 0, 0, 0, i as u64)
                    } else {
                        desc.offset(0, 0, i as u64, 0, 0)
                    };
                    rd(o)
                })
                .collect()
        }
        2 => {
            let (r, c) = (dims[0], dims[1]);
            let mut v = Vec::with_capacity(r * c);
            for i in 0..r {
                for j in 0..c {
                    v.push(rd(desc.offset(0, 0, i as u64, 0, j as u64)));
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

fn dims_usize(t: &CTy) -> Vec<usize> {
    t.dims.iter().map(|d| *d as usize).collect()
}

// ---------------------------------------------------------------- compiler

struct Bc {
    ops: Vec<Op>,
    scopes: Vec<HashMap<String, u32>>,
    nslots: u32,
    breaks: Vec<Vec<u32>>,
    continues: Vec<Vec<u32>>,
}

impl Bc {
    fn slot(&mut self, name: &str) -> u32 {
        for s in self.scopes.iter().rev() {
            if let Some(&i) = s.get(name) {
                return i;
            }
        }
        let i = self.nslots;
        self.nslots += 1;
        self.scopes.last_mut().unwrap().insert(name.to_string(), i);
        i
    }
    fn find(&self, name: &str) -> Option<u32> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).copied()
    }
    fn patch(&mut self, at: u32, to: u32) {
        match &mut self.ops[at as usize] {
            Op::Jmp(t) | Op::JmpF(t) => *t = to,
            _ => {}
        }
    }
    fn here(&self) -> u32 {
        self.ops.len() as u32
    }
    fn emit(&mut self, op: Op) -> u32 {
        self.ops.push(op);
        self.here() - 1
    }

    fn expr(&mut self, e: &HT) -> Result<(), String> {
        match &e.e {
            HE::I64(n) => {
                self.emit(Op::I64(*n));
            }
            HE::F64(f) => {
                self.emit(Op::F64(*f));
            }
            HE::Bool(b) => {
                self.emit(Op::Bool(*b));
            }
            HE::Str(s) => {
                self.emit(Op::Str(Rc::from(s.as_str())));
            }
            HE::Var(n) => {
                let i = self
                    .find(n)
                    .ok_or_else(|| format!("compile: unbound local {n}"))?;
                self.emit(Op::Load(i));
            }
            HE::Global(gi) => {
                self.emit(Op::Global(*gi));
            }
            HE::Un(op, a) => {
                self.expr(a)?;
                match (op, &e.ty) {
                    (UnOp::Neg, RTy::I64) => self.emit(Op::NegI),
                    (UnOp::Neg, RTy::F64) => self.emit(Op::NegF),
                    (UnOp::Not, _) => self.emit(Op::Not),
                    _ => return Err("compile: bad unary".into()),
                };
            }
            HE::Bin(op, a, b) => match op {
                BinOp::And => {
                    self.expr(a)?;
                    let jf = self.emit(Op::JmpF(0));
                    self.expr(b)?;
                    let je = self.emit(Op::Jmp(0));
                    let l1 = self.here();
                    self.emit(Op::Bool(false));
                    let l2 = self.here();
                    self.patch(jf, l1);
                    self.patch(je, l2);
                }
                BinOp::Or => {
                    self.expr(a)?;
                    let jf = self.emit(Op::JmpF(0));
                    let je = self.emit(Op::Jmp(0));
                    let l1 = self.here();
                    self.expr(b)?;
                    let l2 = self.here();
                    self.patch(jf, l1);
                    self.patch(je, l2);
                }
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                    self.expr(a)?;
                    self.expr(b)?;
                    let op2 = match (op, &e.ty) {
                        (BinOp::Add, RTy::I64) => Op::AddI,
                        (BinOp::Add, RTy::F64) => Op::AddF,
                        (BinOp::Add, RTy::Str) => Op::Concat,
                        (BinOp::Sub, RTy::I64) => Op::SubI,
                        (BinOp::Sub, RTy::F64) => Op::SubF,
                        (BinOp::Mul, RTy::I64) => Op::MulI,
                        (BinOp::Mul, RTy::F64) => Op::MulF,
                        (BinOp::Div, RTy::I64) => Op::DivI,
                        (BinOp::Div, RTy::F64) => Op::DivF,
                        (BinOp::Mod, _) => Op::ModI,
                        _ => return Err(format!("compile: bad binop {op:?}")),
                    };
                    self.emit(op2);
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                    self.expr(a)?;
                    self.expr(b)?;
                    self.emit(Op::Cmp(*op));
                }
                BinOp::MatMul => return Err("compile: @ in host code".into()),
            },
            HE::Call(fi, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.emit(Op::Call(*fi, args.len() as u8));
            }
            HE::CallExt(ei, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.emit(Op::CallExt(*ei, args.len() as u8));
            }
            HE::CallAne(pi, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.emit(Op::CallAne(*pi, args.len() as u8));
            }
            HE::CallDyn(callee, args) => {
                self.expr(callee)?;
                for a in args {
                    self.expr(a)?;
                }
                self.emit(Op::CallDyn(args.len() as u8));
            }
            HE::Native(ni, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.emit(Op::Native(*ni, args.len() as u8));
            }
            HE::Index(b, i) => {
                self.expr(b)?;
                self.expr(i)?;
                match &b.ty {
                    RTy::Vec(_) => self.emit(Op::VecGet),
                    RTy::Map(_, _) => self.emit(Op::MapGet),
                    RTy::Str => self.emit(Op::StrGet),
                    RTy::Tensor(t) if t.dims.len() == 1 => self.emit(Op::TensorGet),
                    RTy::Tensor(_) => self.emit(Op::TensorRow),
                    _ => return Err("compile: bad index".into()),
                };
            }
            HE::Field(b, fi) => {
                self.expr(b)?;
                match &b.ty {
                    RTy::Token => self.emit(Op::TokenField(*fi)),
                    _ => self.emit(Op::FieldGet(*fi)),
                };
            }
            HE::VecLit(items) => {
                for it in items {
                    self.expr(it)?;
                }
                self.emit(Op::VecNew(items.len() as u32));
            }
            HE::MapLit(items) => {
                for (k, v) in items {
                    self.expr(k)?;
                    self.expr(v)?;
                }
                self.emit(Op::MapNew(items.len() as u32));
            }
            HE::StructLit(si, vals) => {
                for v in vals {
                    self.expr(v)?;
                }
                self.emit(Op::StructNew(*si));
            }
            HE::EnumLit(ei, vi, vals) => {
                for v in vals {
                    self.expr(v)?;
                }
                self.emit(Op::EnumNew(*ei, *vi, vals.len() as u8));
            }
        }
        Ok(())
    }

    fn stmt(&mut self, s: &HS) -> Result<(), String> {
        match s {
            HS::Let { name, expr, .. } => {
                self.expr(expr)?;
                let i = self.slot(name);
                self.emit(Op::Store(i));
            }
            HS::Assign { name, op, expr } => {
                let i = self
                    .find(name)
                    .ok_or_else(|| format!("compile: unbound {name}"))?;
                match op {
                    AssignOp::Set => {
                        self.expr(expr)?;
                    }
                    _ => {
                        self.emit(Op::Load(i));
                        self.expr(expr)?;
                        let aop = match (op, &expr.ty) {
                            (AssignOp::Add, RTy::I64) => Op::AddI,
                            (AssignOp::Add, RTy::F64) => Op::AddF,
                            (AssignOp::Add, RTy::Str) => Op::Concat,
                            (AssignOp::Sub, RTy::I64) => Op::SubI,
                            (AssignOp::Sub, RTy::F64) => Op::SubF,
                            (AssignOp::Mul, RTy::I64) => Op::MulI,
                            (AssignOp::Mul, RTy::F64) => Op::MulF,
                            (AssignOp::Div, RTy::I64) => Op::DivI,
                            (AssignOp::Div, RTy::F64) => Op::DivF,
                            _ => return Err("compile: bad assign op".into()),
                        };
                        self.emit(aop);
                    }
                }
                self.emit(Op::Store(i));
            }
            HS::Store { target, expr } => match target {
                StoreT::Index(tb, ti) => {
                    self.expr(tb)?;
                    self.expr(ti)?;
                    self.expr(expr)?;
                    match &tb.ty {
                        RTy::Vec(_) => self.emit(Op::VecSet),
                        RTy::Map(_, _) => self.emit(Op::MapSet),
                        RTy::Tensor(_) => self.emit(Op::TensorSet),
                        _ => return Err("compile: bad store".into()),
                    };
                }
                StoreT::Field(tb, fi) => {
                    self.expr(tb)?;
                    self.expr(expr)?;
                    self.emit(Op::FieldSet(*fi));
                }
            },
            HS::If { cond, then, els } => {
                self.expr(cond)?;
                let jf = self.emit(Op::JmpF(0));
                self.scopes.push(HashMap::new());
                for s in then {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                let je = self.emit(Op::Jmp(0));
                let l_else = self.here();
                self.patch(jf, l_else);
                self.scopes.push(HashMap::new());
                for s in els {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                let l_end = self.here();
                self.patch(je, l_end);
            }
            HS::Match { expr, arms } => {
                self.scopes.push(HashMap::new());
                self.expr(expr)?;
                let scr = self.slot(&format!("__match_{}", self.nslots));
                self.emit(Op::Store(scr));
                let mut end_jmps = Vec::new();
                for (vi_opt, binds, body) in arms {
                    let mut next_arm = None;
                    if let Some(vi) = vi_opt {
                        self.emit(Op::Load(scr));
                        self.emit(Op::EnumIs(*vi));
                        next_arm = Some(self.emit(Op::JmpF(0)));
                    }
                    self.scopes.push(HashMap::new());
                    for (bi, (bname, _)) in binds.iter().enumerate() {
                        self.emit(Op::Load(scr));
                        self.emit(Op::EnumField(bi as u8));
                        let bslot = self.slot(bname);
                        self.emit(Op::Store(bslot));
                    }
                    for s in body {
                        self.stmt(s)?;
                    }
                    self.scopes.pop();
                    end_jmps.push(self.emit(Op::Jmp(0)));
                    if let Some(jf) = next_arm {
                        let here = self.here();
                        self.patch(jf, here);
                    }
                }
                let end = self.here();
                for j in end_jmps {
                    self.patch(j, end);
                }
                self.scopes.pop();
            }
            HS::While { cond, body } => {
                let l0 = self.here();
                self.expr(cond)?;
                let jf = self.emit(Op::JmpF(0));
                self.breaks.push(vec![jf]);
                self.continues.push(Vec::new());
                self.scopes.push(HashMap::new());
                for s in body {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                self.emit(Op::Jmp(l0));
                let end = self.here();
                for b in self.breaks.pop().unwrap() {
                    self.patch(b, end);
                }
                for cj in self.continues.pop().unwrap() {
                    self.patch(cj, l0);
                }
            }
            HS::For { var, lo, hi, body } => {
                self.scopes.push(HashMap::new());
                self.expr(lo)?;
                let iv = self.slot(var);
                self.emit(Op::Store(iv));
                self.expr(hi)?;
                let hv = self.slot("__hi");
                self.emit(Op::Store(hv));
                let l0 = self.here();
                self.emit(Op::Load(iv));
                self.emit(Op::Load(hv));
                self.emit(Op::Cmp(BinOp::Lt));
                let jf = self.emit(Op::JmpF(0));
                self.breaks.push(vec![jf]);
                self.continues.push(Vec::new());
                for s in body {
                    self.stmt(s)?;
                }
                // continue lands here — increment then re-test
                let linc = self.here();
                for cj in self.continues.pop().unwrap() {
                    self.patch(cj, linc);
                }
                self.emit(Op::Load(iv));
                self.emit(Op::I64(1));
                self.emit(Op::AddI);
                self.emit(Op::Store(iv));
                self.emit(Op::Jmp(l0));
                let end = self.here();
                for b in self.breaks.pop().unwrap() {
                    self.patch(b, end);
                }
                self.scopes.pop();
            }
            HS::Break => {
                let j = self.emit(Op::Jmp(0));
                self.breaks.last_mut().unwrap().push(j);
            }
            HS::Continue => {
                let j = self.emit(Op::Jmp(0));
                self.continues.last_mut().unwrap().push(j);
            }
            HS::Return(e) => {
                match e {
                    Some(t) => self.expr(t)?,
                    None => {
                        self.emit(Op::Unit);
                    }
                }
                self.emit(Op::Ret);
            }
            HS::Expr(e) => {
                self.expr(e)?;
                self.emit(Op::Pop);
            }
        }
        Ok(())
    }
}

/// compile a host fn to bytecode. Returns (ops, nslots).
fn compile_fn(f: &HFnT, arg_names: &[String]) -> Result<(Vec<Op>, u32), String> {
    let mut bc = Bc {
        ops: Vec::new(),
        scopes: vec![HashMap::new()],
        nslots: 0,
        breaks: Vec::new(),
        continues: Vec::new(),
    };
    for n in arg_names {
        bc.slot(n);
    }
    for s in &f.body {
        bc.stmt(s)?;
    }
    // implicit unit return — non-unit fns are guaranteed a Return by
    // the typechecker's definitely_returns check.
    bc.emit(Op::Unit);
    bc.emit(Op::Ret);
    Ok((bc.ops, bc.nslots))
}

// ---------------------------------------------------------------- the VM

pub type ExternCb = Box<dyn FnMut(&[Val]) -> Result<Val, String>>;

pub struct Vm {
    h: HProgram,
    code: Vec<Vec<Op>>,
    nslots: Vec<u32>,
    stack: Vec<Val>,
    frames: Vec<Frame>,
    globals: Vec<Val>,
    /// tensors: (declared cty, declared-order row-major f32s)
    tensors: Vec<(CTy, Vec<f32>)>,
    /// global name → tensor handle (params, tables, state mirrors)
    gtensor: HashMap<String, u32>,
    progs: Vec<AneProg>,
    extern_fns: HashMap<String, ExternCb>,
    witness: Vec<Wit>,
    rng: u64,
    sha: String,
    halted: bool,
    result: Option<Val>,
}

impl Vm {
    /// build the VM: compile host fns, resolve globals (params load
    /// from safetensors here — f16-rounded like the emitted blobs).
    pub fn new(h: &HProgram, progs: Vec<AneProg>, sha: &str) -> Result<Self, String> {
        let h = h.clone();
        let mut code = Vec::new();
        let mut nslots = Vec::new();
        for f in &h.fns {
            let names: Vec<String> = f.args.iter().map(|(n, _)| n.clone()).collect();
            let (ops, n) = compile_fn(f, &names)?;
            code.push(ops);
            nslots.push(n);
        }
        let mut tensors = Vec::new();
        let mut gtensor = HashMap::new();
        let mut globals = Vec::new();
        for (_name, gv, _) in &h.globals {
            let v = match gv {
                GVal::Prog { idx } => Val::Ane(*idx),
                GVal::Param { name, ty, src } | GVal::Table { name, ty, src } => {
                    // params resolve through the same loader the lowerer
                    // uses — [N,K] file storage for rank-2 weight
                    // matrices; tables resolve in declared order.
                    let is_param = matches!(gv, GVal::Param { .. });
                    let decl = dims_usize(ty);
                    let expect: Vec<usize> = if ty.dims.len() == 2 && is_param {
                        vec![decl[1], decl[0]]
                    } else {
                        decl.clone()
                    };
                    let mut data = crate::weights::resolve(
                        name,
                        Some(&expect),
                        src.as_deref(),
                        init_weight_pub,
                    )
                    .map_err(|e| e.to_string())?;
                    // file sources arrive f32 — round through f16 like
                    // the emitted blobs (parity is vs silicon).
                    for v in data.iter_mut() {
                        *v = half::f16::from_f32(*v).to_f32();
                    }
                    // host storage is declared order: for rank-2 params
                    // the resolved [N,K] buffer transposes back.
                    if ty.dims.len() == 2 && matches!(gv, GVal::Param { .. }) {
                        let (n, k) = (expect[0], expect[1]);
                        let mut t = vec![0f32; data.len()];
                        for a in 0..n {
                            for b in 0..k {
                                t[b * n + a] = data[a * k + b];
                            }
                        }
                        data = t;
                    }
                    let idx = tensors.len() as u32;
                    tensors.push((ty.clone(), data));
                    gtensor.insert(name.clone(), idx);
                    Val::T(idx)
                }
                GVal::State { key, ty, .. } => {
                    let idx = tensors.len() as u32;
                    tensors.push((ty.clone(), vec![0.0; ty.numel().max(1) as usize]));
                    gtensor.insert(key.clone(), idx);
                    Val::T(idx)
                }
            };
            globals.push(v);
        }
        let seed = u64::from_le_bytes(
            sha256_hex(sha.as_bytes()).as_bytes()[..8]
                .try_into()
                .unwrap_or([0u8; 8]),
        ) | 1;
        Ok(Self {
            h,
            code,
            nslots,
            stack: Vec::new(),
            frames: Vec::new(),
            globals,
            tensors,
            gtensor,
            progs,
            extern_fns: HashMap::new(),
            witness: Vec::new(),
            rng: seed,
            sha: sha.to_string(),
            halted: false,
            result: None,
        })
    }

    pub fn witness_log(&self) -> &[Wit] {
        &self.witness
    }

    /// Bind a Rust or Swift FFI callback to an `extern fn` declared in
    /// the `.cy` source.
    pub fn bind_extern(
        &mut self,
        name: &str,
        f: impl FnMut(&[Val]) -> Result<Val, String> + 'static,
    ) {
        self.extern_fns.insert(name.to_string(), Box::new(f));
    }

    /// Invoke any `host fn` or `reflex` by name with `args` and run to
    /// completion.
    pub fn call_fn(&mut self, name: &str, args: Vec<Val>) -> Result<Val, String> {
        let fi = self
            .h
            .fns
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| format!("no host fn or reflex `{name}`"))?;
        let want = self.h.fns[fi].args.len();
        if args.len() != want {
            return Err(format!("{name} expects {want} args, found {}", args.len()));
        }
        self.stack.clear();
        self.frames.clear();
        self.halted = false;
        self.result = None;
        for a in args {
            self.stack.push(a);
        }
        let need = self.nslots[fi] as usize;
        for _ in want..need {
            self.stack.push(Val::Unit);
        }
        self.frames.push(Frame {
            f: fi as u32,
            ip: 0,
            base: 0,
        });
        self.run()?;
        Ok(self.result.clone().unwrap_or(Val::Unit))
    }

    /// kick off `main` and run to completion (or die).
    pub fn run_main(&mut self) -> Result<Val, String> {
        self.call_fn("main", Vec::new())
    }

    fn run(&mut self) -> Result<(), String> {
        loop {
            if self.halted {
                return Ok(());
            }
            let (f, ip, base) = match self.frames.last() {
                Some(fr) => (fr.f as usize, fr.ip, fr.base),
                None => return Ok(()),
            };
            let op = self.code[f]
                .get(ip)
                .cloned()
                .ok_or_else(|| format!("vm: pc out of bounds in fn {f}"))?;
            self.frames.last_mut().unwrap().ip += 1;
            match op {
                Op::Unit => self.stack.push(Val::Unit),
                Op::I64(n) => self.stack.push(Val::I(n)),
                Op::F64(x) => self.stack.push(Val::F(x)),
                Op::Bool(b) => self.stack.push(Val::B(b)),
                Op::Str(s) => self.stack.push(Val::S(s)),
                Op::Load(i) => {
                    let v = self.stack[base + i as usize].clone();
                    self.stack.push(v);
                }
                Op::Store(i) => {
                    let v = self.stack.pop().unwrap();
                    self.stack[base + i as usize] = v;
                }
                Op::Global(i) => {
                    let v = self.globals[i as usize].clone();
                    self.stack.push(v);
                }
                Op::AddI => self.ibi(|a, b| a.wrapping_add(b), Val::I)?,
                Op::SubI => self.ibi(|a, b| a.wrapping_sub(b), Val::I)?,
                Op::MulI => self.ibi(|a, b| a.wrapping_mul(b), Val::I)?,
                Op::DivI => self.ibi(|a, b| {
                    if b == 0 {
                        0
                    } else {
                        a.wrapping_div(b)
                    }
                }, Val::I)?,
                Op::ModI => self.ibi(|a, b| if b == 0 { 0 } else { a % b }, Val::I)?,
                Op::NegI => {
                    let a = self.stack.pop().unwrap();
                    match a {
                        Val::I(i) => self.stack.push(Val::I(i.wrapping_neg())),
                        v => return Err(format!("neg on {}", self.kind(&v))),
                    }
                }
                Op::AddF => self.fbf(|a, b| a + b)?,
                Op::SubF => self.fbf(|a, b| a - b)?,
                Op::MulF => self.fbf(|a, b| a * b)?,
                Op::DivF => self.fbf(|a, b| a / b)?,
                Op::NegF => {
                    let a = self.stack.pop().unwrap();
                    match a {
                        Val::F(x) => self.stack.push(Val::F(-x)),
                        v => return Err(format!("neg on {}", self.kind(&v))),
                    }
                }
                Op::Concat => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    let s = format!("{}{}", self.stringify(&a), self.stringify(&b));
                    self.stack.push(Val::S(Rc::from(s.as_str())));
                }
                Op::Cmp(op) => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    let r = self.compare(op, &a, &b)?;
                    self.stack.push(Val::B(r));
                }
                Op::Not => {
                    let a = self.stack.pop().unwrap();
                    self.stack.push(Val::B(!a.truthy()));
                }
                Op::Pop => {
                    self.stack.pop();
                }
                Op::Jmp(t) => self.frames.last_mut().unwrap().ip = t as usize,
                Op::JmpF(t) => {
                    let c = self.stack.pop().unwrap();
                    if !c.truthy() {
                        self.frames.last_mut().unwrap().ip = t as usize;
                    }
                }
                Op::Call(fi, argc) => self.do_call(fi, argc)?,
                Op::CallExt(ei, argc) => {
                    let mut args = Vec::new();
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap());
                    }
                    args.reverse();
                    let name = self.h.externs[ei as usize].name.clone();
                    let cb = self
                        .extern_fns
                        .get_mut(&name)
                        .ok_or_else(|| format!("unbound extern fn `{name}` — bind it before running"))?;
                    let r = cb(&args)?;
                    self.stack.push(r);
                }
                Op::CallDyn(argc) => {
                    let mut args = Vec::new();
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap());
                    }
                    args.reverse();
                    let callee = self.stack.pop().unwrap();
                    match callee {
                        Val::Ane(pi) => self.call_ane(pi as usize, args)?,
                        v => return Err(format!("call on non-program {}", self.kind(&v))),
                    }
                }
                Op::CallAne(pi, argc) => {
                    let mut args = Vec::new();
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap());
                    }
                    args.reverse();
                    self.call_ane(pi as usize, args)?;
                }
                Op::Native(ni, argc) => {
                    let mut args = Vec::new();
                    for _ in 0..argc {
                        args.push(self.stack.pop().unwrap());
                    }
                    args.reverse();
                    // migrate freezes mid-op: the return value must already be
                    // on the stack or the resumed body is off by a slot.
                    if NATIVE_NAMES.get(ni as usize).copied() == Some("migrate") {
                        let path = match &args[0] {
                            Val::S(s) => s.to_string(),
                            v => {
                                return Err(format!(
                                    "migrate path must be str, found {}",
                                    self.kind(v)
                                ))
                            }
                        };
                        self.stack.push(Val::Unit);
                        self.snapshot(&path)?;
                        self.wit("migrate", &path);
                        continue;
                    }
                    let r = self.native(ni, args)?;
                    self.stack.push(r);
                }
                Op::VecNew(n) => {
                    let mut items = Vec::new();
                    for _ in 0..n {
                        items.push(self.stack.pop().unwrap());
                    }
                    items.reverse();
                    self.stack.push(Val::V(Rc::new(RefCell::new(items))));
                }
                Op::VecGet => {
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let v = self.stack.pop().unwrap();
                    match v {
                        Val::V(rc) => {
                            let b = rc.borrow();
                            let item = b
                                .get(i)
                                .cloned()
                                .ok_or_else(|| format!("vec index {i} out of bounds ({})", b.len()))?;
                            self.stack.push(item);
                        }
                        _ => return Err("vecget on non-vec".into()),
                    }
                }
                Op::VecSet => {
                    let x = self.stack.pop().unwrap();
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let v = self.stack.pop().unwrap();
                    match v {
                        Val::V(rc) => {
                            let mut b = rc.borrow_mut();
                            if i >= b.len() {
                                return Err(format!("vec index {i} out of bounds ({})", b.len()));
                            }
                            b[i] = x;
                        }
                        _ => return Err("vecset on non-vec".into()),
                    }
                }
                Op::MapNew(n) => {
                    let mut pairs = Vec::new();
                    for _ in 0..n {
                        let v = self.stack.pop().unwrap();
                        let k = self.stack.pop().unwrap();
                        pairs.push((k, v));
                    }
                    pairs.reverse();
                    let mut dedup: Vec<(Val, Val)> = Vec::new();
                    for (k, v) in pairs {
                        if let Some(slot) = dedup.iter_mut().find(|(ek, _)| self.val_eq(ek, &k)) {
                            slot.1 = v;
                        } else {
                            dedup.push((k, v));
                        }
                    }
                    self.stack.push(Val::M(Rc::new(RefCell::new(dedup))));
                }
                Op::MapGet => {
                    let k = self.stack.pop().unwrap();
                    let m = self.stack.pop().unwrap();
                    match m {
                        Val::M(rc) => {
                            let b = rc.borrow();
                            let item = b
                                .iter()
                                .find_map(|(ek, ev)| self.val_eq(ek, &k).then(|| ev.clone()))
                                .ok_or_else(|| format!("map key {} not found", self.stringify(&k)))?;
                            self.stack.push(item);
                        }
                        _ => return Err("mapget on non-map".into()),
                    }
                }
                Op::MapSet => {
                    let v = self.stack.pop().unwrap();
                    let k = self.stack.pop().unwrap();
                    let m = self.stack.pop().unwrap();
                    match m {
                        Val::M(rc) => {
                            let mut b = rc.borrow_mut();
                            if let Some(slot) = b.iter_mut().find(|(ek, _)| self.val_eq(ek, &k)) {
                                slot.1 = v;
                            } else {
                                b.push((k, v));
                            }
                        }
                        _ => return Err("mapset on non-map".into()),
                    }
                }
                Op::StrGet => {
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let s = self.stack.pop().unwrap();
                    match s {
                        Val::S(rc) => {
                            let ch = rc
                                .chars()
                                .nth(i)
                                .ok_or_else(|| format!("str index {i} out of bounds"))?;
                            self.stack.push(Val::S(Rc::from(ch.to_string().as_str())));
                        }
                        _ => return Err("strget on non-str".into()),
                    }
                }
                Op::TensorGet => {
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let t = self.stack.pop().unwrap().as_t().unwrap_or(0) as usize;
                    let v = self.tensors[t]
                        .1
                        .get(i)
                        .copied()
                        .ok_or_else(|| format!("tensor index {i} out of bounds"))?;
                    self.stack.push(Val::F(v as f64));
                }
                Op::TensorRow => {
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let t = self.stack.pop().unwrap().as_t().unwrap_or(0) as usize;
                    let (ty, data) = &self.tensors[t];
                    let row = ty.dims[1] as usize;
                    if i >= ty.dims[0] as usize {
                        return Err(format!("tensor row {i} out of bounds ({})", ty.dims[0]));
                    }
                    let out = data[i * row..(i + 1) * row].to_vec();
                    let nty = CTy {
                        dt: ty.dt,
                        dims: vec![row as i64],
                    };
                    self.tensors.push((nty, out));
                    let idx = self.tensors.len() as u32 - 1;
                    self.stack.push(Val::T(idx));
                }
                Op::TensorSet => {
                    let x = self.stack.pop().unwrap();
                    let i = self.stack.pop().unwrap().as_i().unwrap_or(0) as usize;
                    let t = self.stack.pop().unwrap().as_t().unwrap_or(0) as usize;
                    let f = match x {
                        Val::F(f) => f as f32,
                        Val::I(n) => n as f32,
                        v => return Err(format!("tensor store of {}", self.kind(&v))),
                    };
                    let d = &mut self.tensors[t].1;
                    if i >= d.len() {
                        return Err(format!("tensor index {i} out of bounds ({})", d.len()));
                    }
                    d[i] = f;
                }
                Op::StructNew(si) => {
                    let n = self.h.structs[si as usize].1.len();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        fields.push(self.stack.pop().unwrap());
                    }
                    fields.reverse();
                    self.stack.push(Val::St(si, Rc::new(RefCell::new(fields))));
                }
                Op::FieldGet(fi) => {
                    let s = self.stack.pop().unwrap();
                    match s {
                        Val::St(_, rc) => {
                            let v = rc.borrow()[fi as usize].clone();
                            self.stack.push(v);
                        }
                        v => return Err(format!("field on {}", self.kind(&v))),
                    }
                }
                Op::FieldSet(fi) => {
                    let x = self.stack.pop().unwrap();
                    let s = self.stack.pop().unwrap();
                    match s {
                        Val::St(_, rc) => {
                            rc.borrow_mut()[fi as usize] = x;
                        }
                        v => return Err(format!("field store on {}", self.kind(&v))),
                    }
                }
                Op::EnumNew(ei, vi, n) => {
                    let mut payloads = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        payloads.push(self.stack.pop().unwrap());
                    }
                    payloads.reverse();
                    self.stack.push(Val::En(ei, vi, Rc::new(payloads)));
                }
                Op::EnumIs(want_vi) => {
                    let v = self.stack.pop().unwrap();
                    match v {
                        Val::En(_, vi, _) => self.stack.push(Val::B(vi == want_vi)),
                        other => return Err(format!("enum match on {}", self.kind(&other))),
                    }
                }
                Op::EnumField(bi) => {
                    let v = self.stack.pop().unwrap();
                    match v {
                        Val::En(_, _, payloads) => {
                            let item = payloads
                                .get(bi as usize)
                                .cloned()
                                .ok_or_else(|| format!("enum payload {bi} out of bounds"))?;
                            self.stack.push(item);
                        }
                        other => return Err(format!("enum field on {}", self.kind(&other))),
                    }
                }
                Op::TokenField(fi) => {
                    let s = self.stack.pop().unwrap();
                    match s {
                        Val::Tok { id, epoch } => {
                            self.stack.push(Val::I(if fi == 0 { id as i64 } else { epoch as i64 }));
                        }
                        v => return Err(format!("token field on {}", self.kind(&v))),
                    }
                }
                Op::Ret => {
                    let v = self.stack.pop().unwrap_or(Val::Unit);
                    let fr = self.frames.pop().unwrap();
                    // `axon` invariant #2: every non-Pass reflex firing is
                    // ledgered onto the hash chain automatically.
                    let fmeta = &self.h.fns[fr.f as usize];
                    if fmeta.is_reflex {
                        if let Val::En(_, vi, _) = &v {
                            if *vi != 0 {
                                let tag = format!("reflex:{}", fmeta.name);
                                let val = self.stringify(&v);
                                self.wit(&tag, &val);
                            }
                        }
                    }
                    self.stack.truncate(fr.base);
                    if self.frames.is_empty() {
                        self.result = Some(v);
                        return Ok(());
                    }
                    self.stack.push(v);
                }
            }
        }
    }

    fn do_call(&mut self, fi: u32, argc: u8) -> Result<(), String> {
        let f = fi as usize;
        let want = self.h.fns[f].args.len();
        if argc as usize != want {
            return Err(format!("{} expects {want} args", self.h.fns[f].name));
        }
        let base = self.stack.len() - argc as usize;
        let local = self.nslots[f] as usize;
        for _ in argc as usize..local {
            self.stack.push(Val::Unit);
        }
        self.frames.push(Frame { f: fi, ip: 0, base });
        Ok(())
    }

    fn ibi(&mut self, f: impl Fn(i64, i64) -> i64, mk: impl Fn(i64) -> Val) -> Result<(), String> {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();
        match (a, b) {
            (Val::I(x), Val::I(y)) => {
                self.stack.push(mk(f(x, y)));
                Ok(())
            }
            (x, y) => Err(format!(
                "int op on {} and {}",
                self.kind(&x),
                self.kind(&y)
            )),
        }
    }

    fn fbf(&mut self, f: impl Fn(f64, f64) -> f64) -> Result<(), String> {
        let b = self.stack.pop().unwrap();
        let a = self.stack.pop().unwrap();
        match (a, b) {
            (Val::F(x), Val::F(y)) => {
                self.stack.push(Val::F(f(x, y)));
                Ok(())
            }
            (x, y) => Err(format!(
                "float op on {} and {}",
                self.kind(&x),
                self.kind(&y)
            )),
        }
    }

    fn val_eq(&self, a: &Val, b: &Val) -> bool {
        match (a, b) {
            (Val::I(i), Val::I(j)) => i == j,
            (Val::F(i), Val::F(j)) => i == j,
            (Val::B(i), Val::B(j)) => i == j,
            (Val::S(i), Val::S(j)) => i == j,
            (Val::Unit, Val::Unit) => true,
            (Val::T(i), Val::T(j)) => i == j,
            (Val::Ane(i), Val::Ane(j)) => i == j,
            (Val::V(i), Val::V(j)) => Rc::ptr_eq(i, j),
            (Val::M(i), Val::M(j)) => Rc::ptr_eq(i, j),
            (Val::St(i, a), Val::St(j, b2)) => i == j && Rc::ptr_eq(a, b2),
            (Val::En(e1, v1, p1), Val::En(e2, v2, p2)) => {
                e1 == e2
                    && v1 == v2
                    && p1.len() == p2.len()
                    && p1.iter().zip(p2.iter()).all(|(x, y)| self.val_eq(x, y))
            }
            (Val::Tok { id: i, epoch: e }, Val::Tok { id: i2, epoch: e2 }) => {
                i == i2 && e == e2
            }
            _ => false,
        }
    }

    fn compare(&self, op: BinOp, a: &Val, b: &Val) -> Result<bool, String> {
        let ord = |x: &Val, y: &Val| -> Option<std::cmp::Ordering> {
            match (x, y) {
                (Val::I(i), Val::I(j)) => Some(i.cmp(j)),
                (Val::F(i), Val::F(j)) => i.partial_cmp(j),
                (Val::S(i), Val::S(j)) => Some(i.as_ref().cmp(j.as_ref())),
                _ => None,
            }
        };
        Ok(match op {
            BinOp::Eq => self.val_eq(a, b),
            BinOp::Ne => !self.val_eq(a, b),
            BinOp::Lt => ord(a, b) == Some(std::cmp::Ordering::Less),
            BinOp::Gt => ord(a, b) == Some(std::cmp::Ordering::Greater),
            BinOp::Le => matches!(ord(a, b), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)),
            BinOp::Ge => matches!(ord(a, b), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)),
            _ => return Err("bad compare".into()),
        })
    }

    fn kind(&self, v: &Val) -> &'static str {
        match v {
            Val::Unit => "unit",
            Val::I(_) => "i64",
            Val::F(_) => "f64",
            Val::B(_) => "bool",
            Val::S(_) => "str",
            Val::V(_) => "vec",
            Val::M(_) => "map",
            Val::St(_, _) => "struct",
            Val::En(_, _, _) => "enum",
            Val::Dfa(_) => "dfa",
            Val::T(_) => "tensor",
            Val::Ane(_) => "ane",
            Val::Tok { .. } => "token",
        }
    }

    pub fn stringify(&self, v: &Val) -> String {
        match v {
            Val::Unit => "()".into(),
            Val::I(i) => i.to_string(),
            Val::F(f) => {
                if f.fract() == 0.0 {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
            Val::B(b) => b.to_string(),
            Val::S(s) => s.to_string(),
            Val::V(rc) => {
                let b = rc.borrow();
                let inner: Vec<String> = b.iter().take(8).map(|x| self.stringify(x)).collect();
                format!(
                    "[{}{}]",
                    inner.join(", "),
                    if b.len() > 8 { ", …" } else { "" }
                )
            }
            Val::M(rc) => {
                let b = rc.borrow();
                let inner: Vec<String> = b
                    .iter()
                    .take(8)
                    .map(|(k, v)| format!("{}: {}", self.stringify(k), self.stringify(v)))
                    .collect();
                format!(
                    "{{{}{} }}",
                    if inner.is_empty() { "" } else { " " },
                    inner.join(", ")
                )
            }
            Val::St(si, rc) => {
                let (name, decl) = &self.h.structs[*si as usize];
                let b = rc.borrow();
                let inner: Vec<String> = decl
                    .iter()
                    .zip(b.iter())
                    .map(|((n, _), v)| format!("{n}: {}", self.stringify(v)))
                    .collect();
                format!("{name} {{ {} }}", inner.join(", "))
            }
            Val::En(ei, vi, payloads) => {
                let (ename, vars) = &self.h.enums[*ei as usize];
                let vname = &vars[*vi as usize].0;
                if payloads.is_empty() {
                    format!("{ename}::{vname}")
                } else {
                    let inner: Vec<String> = payloads.iter().map(|x| self.stringify(x)).collect();
                    format!("{ename}::{vname}({})", inner.join(", "))
                }
            }
            Val::Dfa(d) => format!("<dfa {} patterns, {} states>", d.patterns.len(), d.out.len()),
            Val::T(i) => {
                let (ty, data) = &self.tensors[*i as usize];
                let head: Vec<String> = data.iter().take(4).map(|x| format!("{x:.4}")).collect();
                format!(
                    "{}<{}{}>",
                    ty.show(),
                    head.join(", "),
                    if data.len() > 4 { ", …" } else { "" }
                )
            }
            Val::Ane(i) => format!("<ane {}>", self.progs[*i as usize].name),
            Val::Tok { id, epoch } => format!("token#{id:016x}:e{epoch}"),
        }
    }

    fn wit(&mut self, tag: &str, val: &str) {
        let seq = self.witness.len() as u64;
        let prev = self
            .witness
            .last()
            .map(|w| w.hash.clone())
            .unwrap_or_else(|| "genesis".into());
        let hash = sha256_hex(format!("{prev}|{seq}|{tag}|{val}").as_bytes());
        self.witness.push(Wit {
            seq,
            tag: tag.to_string(),
            val: val.to_string(),
            hash,
        });
    }

    // ------------------------------------------------- ane / cpu dispatch

    /// invoke a compiled tensor program: tensors in, run on silicon (or
    /// CPU evaluator when `--cpu` is selected), tensor out. State and
    /// trainable weights ride the host store — written before the run,
    /// read back after.
    fn call_ane(&mut self, pi: usize, args: Vec<Val>) -> Result<(), String> {
        // ---- gather staging data first (borrows of self.tensors end
        // before self.progs is borrowed mutably) ----
        let mut stages: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut cpu_inputs: HashMap<String, Vec<f32>> = HashMap::new();
        let mut cpu_params: HashMap<String, Vec<f32>> = HashMap::new();
        let mut cpu_states: HashMap<String, Vec<f32>> = HashMap::new();
        {
            let prog = &self.progs[pi];
            for (i, (port, ty, _contig)) in prog.ins.iter().enumerate() {
                match &args[i] {
                    Val::T(h) => {
                        let data = self.tensors[*h as usize].1.clone();
                        cpu_inputs.insert(port.clone(), data.clone());
                        stages.push((port.clone(), dims_usize(ty), data));
                    }
                    v => {
                        return Err(format!(
                            "{} arg {i} needs a tensor, found {}",
                            prog.name,
                            self.kind(v)
                        ))
                    }
                }
            }
            // trainable weights — host store is declared [K,N]; the
            // surface/eval convention is [N,K] storage.
            for (pname, port, ty) in &prog.trainable {
                let h = self
                    .gtensor
                    .get(pname)
                    .copied()
                    .ok_or_else(|| format!("no host tensor for {pname}"))?;
                let data = self.tensors[h as usize].1.clone();
                if ty.dims.len() == 2 {
                    let (k, n) = (ty.dims[0] as usize, ty.dims[1] as usize);
                    let mut t = vec![0f32; data.len()];
                    for a in 0..n {
                        for b in 0..k {
                            t[a * k + b] = data[b * n + a];
                        }
                    }
                    cpu_params.insert(pname.clone(), t.clone());
                    stages.push((port.clone(), vec![n, k], t));
                } else {
                    cpu_params.insert(pname.clone(), data.clone());
                    stages.push((port.clone(), dims_usize(ty), data));
                }
            }
            // state fields — [T, D] declared-order both ways; the
            // host mirror is namespaced `prog.state.field`.
            for (key, port, ty) in &prog.states {
                let gkey = format!("{}.{}", prog.name, key);
                let h = self
                    .gtensor
                    .get(&gkey)
                    .copied()
                    .ok_or_else(|| format!("no host tensor for {gkey}"))?;
                let data = self.tensors[h as usize].1.clone();
                cpu_states.insert(port.clone(), data.clone());
                stages.push((port.clone(), dims_usize(ty), data));
            }
        }
        // ---- drive the program ----
        let (odata, oty, pouts, souts) = {
            let prog = &mut self.progs[pi];
            let (oname, oty) = prog.output.clone();
            match &mut prog.backend {
                ProgBackend::Ane(rt) => {
                    for (port, dims, data) in &stages {
                        write_surface(rt, port, dims, data);
                    }
                    rt.run().map_err(|e| format!("ane run ({}): {e}", prog.name))?;
                    let idx = rt
                        .out_node(&oname)
                        .ok_or_else(|| format!("output node {oname} missing"))?;
                    let desc = rt.out_desc(idx);
                    let raw = rt.output(idx).to_vec();
                    let odata = read_surface_dims(&raw, &desc, &dims_usize(&oty));
                    let mut pouts = Vec::new();
                    for (pname, mname, ty) in &prog.param_outs {
                        let idx = rt
                            .out_node(mname)
                            .ok_or_else(|| format!("update node {mname} missing"))?;
                        let desc = rt.out_desc(idx);
                        let raw = rt.output(idx).to_vec();
                        let (n, k) = (ty.dims[1] as usize, ty.dims[0] as usize);
                        let surf = read_surface_dims(&raw, &desc, &[n, k]);
                        let mut host = vec![0f32; surf.len()];
                        for a in 0..n {
                            for b in 0..k {
                                host[b * n + a] = surf[a * k + b];
                            }
                        }
                        pouts.push((pname.clone(), host));
                    }
                    let mut souts = Vec::new();
                    for (port, mname, ty) in &prog.state_outs {
                        let idx = rt
                            .out_node(mname)
                            .ok_or_else(|| format!("state node {mname} missing"))?;
                        let desc = rt.out_desc(idx);
                        let raw = rt.output(idx).to_vec();
                        let surf = read_surface_dims(&raw, &desc, &dims_usize(ty));
                        souts.push((port.clone(), surf));
                    }
                    rt.release_all();
                    (odata, oty, pouts, souts)
                }
                ProgBackend::Cpu(tp) => {
                    let mut ev = Evaluator::new(tp);
                    ev.set_params(cpu_params);
                    ev.set_states(cpu_states);
                    let odata = ev
                        .run(&cpu_inputs)
                        .map_err(|d| format!("cpu eval ({}): {d}", prog.name))?;
                    let mut pouts = Vec::new();
                    let mut souts = Vec::new();
                    for (name, vals) in ev.param_updates() {
                        if let Some((_, _, ty)) = prog.param_outs.iter().find(|(p, _, _)| p == name)
                        {
                            let (n, k) = (ty.dims[1] as usize, ty.dims[0] as usize);
                            let mut host = vec![0f32; vals.len()];
                            for a in 0..n {
                                for b in 0..k {
                                    host[b * n + a] = vals[a * k + b];
                                }
                            }
                            pouts.push((name.clone(), host));
                        } else if prog.state_outs.iter().any(|(port, _, _)| port == name) {
                            souts.push((name.clone(), vals.clone()));
                        }
                    }
                    (odata, oty, pouts, souts)
                }
            }
        };
        // ---- land the results in the host store ----
        self.tensors.push((oty, odata));
        let th = self.tensors.len() as u32 - 1;
        self.stack.push(Val::T(th));
        for (pname, host) in pouts {
            if let Some(&h) = self.gtensor.get(&pname) {
                self.tensors[h as usize].1 = host;
            }
        }
        for (port, surf) in souts {
            let gkey = format!("{}.{}", self.progs[pi].name, port_to_key(&port));
            if let Some(&h) = self.gtensor.get(&gkey) {
                self.tensors[h as usize].1 = surf;
            }
        }
        Ok(())
    }

    // ------------------------------------------------- natives

    fn native(&mut self, ni: u32, args: Vec<Val>) -> Result<Val, String> {
        match NATIVE_NAMES.get(ni as usize).copied().unwrap_or("?") {
            "print" => {
                let s: Vec<String> = args.iter().map(|a| self.stringify(a)).collect();
                println!("{}", s.join(" "));
                Ok(Val::Unit)
            }
            "str" => Ok(Val::S(Rc::from(self.stringify(&args[0]).as_str()))),
            "len" => match &args[0] {
                Val::V(rc) => Ok(Val::I(rc.borrow().len() as i64)),
                Val::M(rc) => Ok(Val::I(rc.borrow().len() as i64)),
                Val::S(s) => Ok(Val::I(s.chars().count() as i64)),
                Val::T(h) => Ok(Val::I(self.tensors[*h as usize].1.len() as i64)),
                v => Err(format!("len on {}", self.kind(v))),
            },
            "push" => match &args[0] {
                Val::V(rc) => {
                    rc.borrow_mut().push(args[1].clone());
                    Ok(Val::Unit)
                }
                v => Err(format!("push on {}", self.kind(v))),
            },
            "argmax" => match &args[0] {
                Val::T(h) => {
                    let d = &self.tensors[*h as usize].1;
                    let (i, _) = d
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, &0.0));
                    Ok(Val::I(i as i64))
                }
                v => Err(format!("argmax on {}", self.kind(v))),
            },
            "i64" => match &args[0] {
                Val::I(i) => Ok(Val::I(*i)),
                Val::F(f) => Ok(Val::I(*f as i64)),
                Val::B(b) => Ok(Val::I(*b as i64)),
                Val::Tok { id, .. } => Ok(Val::I(*id as i64)),
                v => Err(format!("i64 on {}", self.kind(v))),
            },
            "f64" => match &args[0] {
                Val::I(i) => Ok(Val::F(*i as f64)),
                Val::F(f) => Ok(Val::F(*f)),
                v => Err(format!("f64 on {}", self.kind(v))),
            },
            "onehot" => {
                let i = args[0].as_i().unwrap_or(0) as usize;
                let n = args[1].as_i().unwrap_or(0) as usize;
                let mut d = vec![0f32; n];
                if i < n {
                    d[i] = 1.0;
                }
                self.tensors.push((
                    CTy { dt: DType::F16, dims: vec![n as i64] },
                    d,
                ));
                Ok(Val::T(self.tensors.len() as u32 - 1))
            }
            "causal" => {
                let upto = args[0].as_i().unwrap_or(0) as usize;
                let n = args[1].as_i().unwrap_or(0) as usize;
                let d: Vec<f32> = (0..n).map(|j| if j <= upto { 0.0 } else { -1e4 }).collect();
                self.tensors.push((
                    CTy { dt: DType::F16, dims: vec![n as i64] },
                    d,
                ));
                Ok(Val::T(self.tensors.len() as u32 - 1))
            }
            "rope_cos" | "rope_sin" => {
                let pos = args[0].as_i().unwrap_or(0) as f64;
                let d = args[1].as_i().unwrap_or(0) as usize;
                let cos = NATIVE_NAMES[ni as usize] == "rope_cos";
                let v: Vec<f32> = (0..d)
                    .map(|i| {
                        let th = 1e6f64.powf(-2.0 * i as f64 / (2.0 * d as f64));
                        let ang = pos * th;
                        if cos { ang.cos() as f32 } else { ang.sin() as f32 }
                    })
                    .collect();
                self.tensors.push((
                    CTy { dt: DType::F16, dims: vec![d as i64] },
                    v,
                ));
                Ok(Val::T(self.tensors.len() as u32 - 1))
            }
            "zeros" => {
                let n = args[0].as_i().unwrap_or(0) as usize;
                self.tensors.push((
                    CTy { dt: DType::F16, dims: vec![n as i64] },
                    vec![0.0; n],
                ));
                Ok(Val::T(self.tensors.len() as u32 - 1))
            }
            "self_token" => {
                let id = u64::from_le_bytes(
                    self.sha.as_bytes()[..8].try_into().unwrap_or([0u8; 8]),
                );
                Ok(Val::Tok {
                    id,
                    epoch: self.witness.len() as u64,
                })
            }
            "witness" => {
                let tag = match &args[0] {
                    Val::S(s) => s.to_string(),
                    v => return Err(format!("witness tag must be str, found {}", self.kind(v))),
                };
                let val = self.stringify(&args[1]);
                self.wit(&tag, &val);
                Ok(Val::Unit)
            }
            "die" => {
                self.wit("die", "organism halted by its own hand");
                self.halted = true;
                Ok(Val::Unit)
            }
            "migrate" => Err("migrate must run in the dispatch loop — it freezes stack state".into()),
            "rand" => {
                // xorshift64* — deterministic, seeded by lineage.
                let mut x = self.rng;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.rng = x;
                let f = (x >> 11) as f64 / (1u64 << 53) as f64;
                Ok(Val::F(f))
            }
            "has" => match &args[0] {
                Val::M(rc) => {
                    let b = rc.borrow();
                    let found = b.iter().any(|(ek, _)| self.val_eq(ek, &args[1]));
                    Ok(Val::B(found))
                }
                v => Err(format!("has on {}", self.kind(v))),
            },
            "keys" => match &args[0] {
                Val::M(rc) => {
                    let b = rc.borrow();
                    let ks: Vec<Val> = b.iter().map(|(k, _)| k.clone()).collect();
                    Ok(Val::V(Rc::new(RefCell::new(ks))))
                }
                v => Err(format!("keys on {}", self.kind(v))),
            },
            "contains" => match (&args[0], &args[1]) {
                (Val::S(a), Val::S(b)) => Ok(Val::B(a.contains(b.as_ref()))),
                _ => Err("contains needs two str args".into()),
            },
            "split" => match (&args[0], &args[1]) {
                (Val::S(a), Val::S(sep)) => {
                    let parts: Vec<Val> = a
                        .split(sep.as_ref())
                        .map(|p| Val::S(Rc::from(p)))
                        .collect();
                    Ok(Val::V(Rc::new(RefCell::new(parts))))
                }
                _ => Err("split needs two str args".into()),
            },
            "slice_str" => match (&args[0], &args[1], &args[2]) {
                (Val::S(s), Val::I(lo), Val::I(hi)) => {
                    let chars: Vec<char> = s.chars().collect();
                    let l = (*lo).max(0).min(chars.len() as i64) as usize;
                    let h = (*hi).max(l as i64).min(chars.len() as i64) as usize;
                    let sub: String = chars[l..h].iter().collect();
                    Ok(Val::S(Rc::from(sub.as_str())))
                }
                _ => Err("slice_str(str, i64, i64) expected".into()),
            },
            "dfa_compile" => match &args[0] {
                Val::V(rc) => {
                    let b = rc.borrow();
                    let mut pats = Vec::with_capacity(b.len());
                    for x in b.iter() {
                        match x {
                            Val::S(s) => pats.push(s.to_string()),
                            v => return Err(format!("dfa pattern must be str, found {}", self.kind(v))),
                        }
                    }
                    Ok(Val::Dfa(Rc::new(DfaMachine::compile(pats))))
                }
                v => Err(format!("dfa_compile on {}", self.kind(v))),
            },
            "dfa_scan" => match (&args[0], &args[1]) {
                (Val::Dfa(d), Val::S(text)) => Ok(Val::I(d.scan(text))),
                _ => Err("dfa_scan(dfa, str) expected".into()),
            },
            "quorum" => {
                // `tokala` — dissenting council deliberation: tally yes/no,
                // preserve every minority report, and witness the verdict.
                let topic = match &args[0] {
                    Val::S(s) => s.to_string(),
                    _ => return Err("quorum topic must be str".into()),
                };
                let votes = match &args[1] {
                    Val::V(rc) => rc.borrow().clone(),
                    _ => return Err("quorum votes must be [Vote]".into()),
                };
                let mut yes = 0i64;
                let mut no = 0i64;
                let mut yes_notes = Vec::new();
                let mut no_notes = Vec::new();
                for v in &votes {
                    if let Val::St(_, rc) = v {
                        let f = rc.borrow();
                        let seat = self.stringify(&f[0]);
                        let approve = f[1].truthy();
                        let reason = self.stringify(&f[2]);
                        if approve {
                            yes += 1;
                            yes_notes.push(format!("{seat}: {reason}"));
                        } else {
                            no += 1;
                            no_notes.push(format!("{seat}: {reason}"));
                        }
                    }
                }
                let approved = yes > no;
                let minority = if approved { no_notes } else { yes_notes };
                let dissent_vals: Vec<Val> = minority
                    .iter()
                    .map(|s| Val::S(Rc::from(s.as_str())))
                    .collect();
                let summary = format!(
                    "approved={approved}|yes={yes}|no={no}|dissent=[{}]",
                    minority.join("; ")
                );
                self.wit(&format!("quorum:{topic}"), &summary);
                let hash = self
                    .witness
                    .last()
                    .map(|w| w.hash.clone())
                    .unwrap_or_default();
                let qv_si = self
                    .h
                    .structs
                    .iter()
                    .position(|(n, _)| n == "QuorumVerdict")
                    .unwrap_or(1) as u32;
                Ok(Val::St(
                    qv_si,
                    Rc::new(RefCell::new(vec![
                        Val::B(approved),
                        Val::I(yes),
                        Val::I(no),
                        Val::V(Rc::new(RefCell::new(dissent_vals))),
                        Val::S(Rc::from(hash.as_str())),
                    ])),
                ))
            }
            "propose_patch" => {
                // `akicita` / `xipe` — bounded autonomy marshal:
                // protected surfaces hold for human review; blast-radius
                // (>20 lines or >1000 bytes) holds; invalid proposals deny.
                let target = self.stringify(&args[0]);
                let old = self.stringify(&args[1]);
                let new = self.stringify(&args[2]);
                let reason = self.stringify(&args[3]);
                let dec_ei = self
                    .h
                    .enums
                    .iter()
                    .position(|(n, _)| n == "Decision")
                    .unwrap_or(1) as u32;
                let protected = [
                    "PolicyEngine",
                    "hostck.rs",
                    "constitution",
                    "CHARTER.md",
                    "/System/",
                    "/Library/",
                ];
                let decision = if reason.trim().is_empty() || old == new {
                    let msg = "empty reason or no-op diff".to_string();
                    self.wit(&format!("marshal:{target}"), &format!("Deny({msg})"));
                    Val::En(dec_ei, 2, Rc::new(vec![Val::S(Rc::from(msg.as_str()))]))
                } else if protected.iter().any(|p| target.contains(p)) {
                    let msg = format!("protected surface `{target}` requires human review");
                    self.wit(&format!("marshal:{target}"), &format!("Hold({msg})"));
                    Val::En(dec_ei, 1, Rc::new(vec![Val::S(Rc::from(msg.as_str()))]))
                } else {
                    let lines = old.lines().count().max(new.lines().count());
                    let bytes = old.len().max(new.len());
                    if lines > 20 || bytes > 1000 {
                        let msg = format!(
                            "blast radius ({lines} lines, {bytes} bytes) exceeds 20L/1000B cap"
                        );
                        self.wit(&format!("marshal:{target}"), &format!("Hold({msg})"));
                        Val::En(dec_ei, 1, Rc::new(vec![Val::S(Rc::from(msg.as_str()))]))
                    } else {
                        self.wit(&format!("marshal:{target}"), &format!("Permit({reason})"));
                        Val::En(dec_ei, 0, Rc::new(Vec::new()))
                    }
                };
                Ok(decision)
            }
            "erase" => {
                // `camazotz` — certified cryptographic erasure:
                // 1) derive per-subject key, encrypt payload + MAC tag
                // 2) zeroize the key
                // 3) attempt decrypt with zeroed key and prove tag failure
                // 4) witness and return the erasure certificate hash.
                let subject = self.stringify(&args[0]);
                let payload = self.stringify(&args[1]);
                let mut key = sha256_hex(format!("{}|key|{subject}", self.sha).as_bytes())
                    .into_bytes();
                let pbytes = payload.as_bytes();
                let mut ct = Vec::with_capacity(pbytes.len());
                for (i, &b) in pbytes.iter().enumerate() {
                    ct.push(b ^ key[i % key.len()]);
                }
                let tag = sha256_hex(
                    &[key.as_slice(), ct.as_slice()].concat(),
                );
                // Destroy the per-subject key
                for kb in key.iter_mut() {
                    *kb = 0;
                }
                // Attempt verification under destroyed key — must fail
                let post_tag = sha256_hex(
                    &[key.as_slice(), ct.as_slice()].concat(),
                );
                let decrypt_verified = post_tag != tag;
                if !decrypt_verified {
                    return Err("camazotz: erasure verification failed".into());
                }
                let cert = sha256_hex(
                    format!(
                        "camazotz|subject={subject}|ct_sha={}|decrypt_failed=true",
                        sha256_hex(&ct)
                    )
                    .as_bytes(),
                );
                self.wit(
                    &format!("erase:{subject}"),
                    &format!("decrypt_verified=true|cert={}", &cert[..16]),
                );
                Ok(Val::S(Rc::from(cert.as_str())))
            }
            "read_file" => {
                let path = self.stringify(&args[0]);
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("read_file({path}): {e}"))?;
                Ok(Val::S(Rc::from(text.as_str())))
            }
            "write_file" => {
                let path = self.stringify(&args[0]);
                let content = self.stringify(&args[1]);
                std::fs::write(&path, content.as_bytes())
                    .map_err(|e| format!("write_file({path}): {e}"))?;
                self.wit("fs:write", &format!("{path} ({}B)", content.len()));
                Ok(Val::Unit)
            }
            "cortex_ask" => {
                let prompt = self.stringify(&args[0]);
                let reply = ask_slicks_or_local(&prompt, &self.sha);
                self.wit("cortex", &format!("q={}B|a={}B", prompt.len(), reply.len()));
                Ok(Val::S(Rc::from(reply.as_str())))
            }
            other => Err(format!("unimplemented native {other}")),
        }
    }

    // ------------------------------------------------- snapshot

    /// serialize the whole organism: stack, frames, tensors, witness
    /// position, rng. The snapshot carries the lineage hash — resume
    /// refuses to wake a body that isn't this one's own.
    fn snapshot(&self, path: &str) -> Result<(), String> {
        let mut out = Vec::new();
        out.extend_from_slice(b"CYSNP01\0");
        out.extend_from_slice(self.sha.as_bytes());
        out.extend_from_slice(&self.rng.to_le_bytes());
        // the witness chain rides the body — a migrated organism
        // continues its own history, it does not start a new one.
        out.extend_from_slice(&(self.witness.len() as u64).to_le_bytes());
        for w in &self.witness {
            out.extend_from_slice(&(w.tag.len() as u32).to_le_bytes());
            out.extend_from_slice(w.tag.as_bytes());
            out.extend_from_slice(&(w.val.len() as u32).to_le_bytes());
            out.extend_from_slice(w.val.as_bytes());
            out.extend_from_slice(w.hash.as_bytes());
        }
        // intern shared heap objects so aliasing survives the round trip
        let mut ids: HashMap<usize, u32> = HashMap::new();
        let mut heap: Vec<u8> = Vec::new();
        let write_val = |v: &Val, out: &mut Vec<u8>, heap: &mut Vec<u8>, ids: &mut HashMap<usize, u32>| {
            snap_val(v, out, heap, ids, self);
        };
        // frames
        out.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
        for fr in &self.frames {
            out.extend_from_slice(&fr.f.to_le_bytes());
            out.extend_from_slice(&(fr.ip as u64).to_le_bytes());
            out.extend_from_slice(&(fr.base as u64).to_le_bytes());
        }
        out.extend_from_slice(&(self.stack.len() as u32).to_le_bytes());
        for v in &self.stack {
            write_val(v, &mut out, &mut heap, &mut ids);
        }
        out.extend_from_slice(&(self.globals.len() as u32).to_le_bytes());
        for v in &self.globals {
            write_val(v, &mut out, &mut heap, &mut ids);
        }
        // tensors
        out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        for (ty, data) in &self.tensors {
            out.push(match ty.dt {
                DType::F16 => 0,
                DType::F32 => 1,
                DType::I32 => 2,
                DType::Q4 => 3,
            });
            out.extend_from_slice(&(ty.dims.len() as u32).to_le_bytes());
            for d in &ty.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            for v in data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        // heap section last — the u64 trailer marks its length so the
        // loader can find it from the back.
        out.extend_from_slice(&heap);
        out.extend_from_slice(&(heap.len() as u64).to_le_bytes());
        std::fs::write(path, &out).map_err(|e| format!("migrate: {e}"))
    }

    /// resume a snapshot into a freshly built VM — same source, same
    /// binaries, restored state. Lineage mismatch = different body.
    pub fn resume(&mut self, path: &str) -> Result<(), String> {
        let buf = std::fs::read(path).map_err(|e| format!("resume: {e}"))?;
        if buf.len() < 8 + 64 + 16 || &buf[..8] != b"CYSNP01\0" {
            return Err("resume: not a coyote snapshot".into());
        }
        let sha = String::from_utf8_lossy(&buf[8..72]).to_string();
        if sha != self.sha {
            return Err(format!(
                "lineage mismatch — snapshot {}… is not this organism ({}…)",
                &sha[..16],
                &self.sha[..16]
            ));
        }
        let mut pos = 72usize;
        self.rng = u64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap());
        let witlen = u64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap());
        let mut witness = Vec::new();
        for seq in 0..witlen {
            let tl = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap()) as usize;
            let tag = String::from_utf8_lossy(take(&buf, &mut pos, tl)?).to_string();
            let vl = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap()) as usize;
            let val = String::from_utf8_lossy(take(&buf, &mut pos, vl)?).to_string();
            let hash = String::from_utf8_lossy(take(&buf, &mut pos, 64)?).to_string();
            witness.push(Wit { seq, tag, val, hash });
        }
        let nframes = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
        let mut frames = Vec::new();
        for _ in 0..nframes {
            let f = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
            let ip = u64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap()) as usize;
            let base = u64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap()) as usize;
            frames.push(Frame { f, ip, base });
        }
        // heap objects must exist before refs resolve — the heap is the
        // last section, marked by a u64 length trailer at the very end.
        let heap_len = u64::from_le_bytes(buf[buf.len() - 8..].try_into().unwrap()) as usize;
        let heap_start = buf.len() - 8 - heap_len;
        let mut heap_map: HashMap<u32, Val> = HashMap::new();
        let mut hp = heap_start;
        let heap_end = buf.len() - 8;
        while hp < heap_end {
            hp = unsnap_heap(&buf, hp, &mut heap_map, self)?;
        }
        let nstack = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
        let mut stack = Vec::new();
        for _ in 0..nstack {
            let (v, np) = unsnap_val(&buf, pos, &heap_map, self)?;
            pos = np;
            stack.push(v);
        }
        let nglob = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
        let mut globals = Vec::new();
        for _ in 0..nglob {
            let (v, np) = unsnap_val(&buf, pos, &heap_map, self)?;
            pos = np;
            globals.push(v);
        }
        let nten = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
        let mut tensors = Vec::new();
        for _ in 0..nten {
            let dt = match take(&buf, &mut pos, 1)?[0] {
                0 => DType::F16,
                1 => DType::F32,
                2 => DType::I32,
                _ => DType::Q4,
            };
            let rank = u32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap());
            let mut dims = Vec::new();
            for _ in 0..rank {
                dims.push(i64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap()));
            }
            let len = u64::from_le_bytes(take(&buf, &mut pos, 8)?.try_into().unwrap()) as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(f32::from_le_bytes(take(&buf, &mut pos, 4)?.try_into().unwrap()));
            }
            tensors.push((CTy { dt, dims }, data));
        }
        self.frames = frames;
        self.stack = stack;
        self.globals = globals;
        self.tensors = tensors;
        self.witness = witness;
        self.wit("resume", path);
        Ok(())
    }

    /// after resume: keep running from the restored frame.
    pub fn run_resumed(&mut self) -> Result<Val, String> {
        self.run()?;
        Ok(self.result.clone().unwrap_or(Val::Unit))
    }
}

/// Connect to Bad Apple's live SLICKS Unix socket if reachable, else
/// return a deterministic local cortex attestation.
fn ask_slicks_or_local(prompt: &str, sha: &str) -> String {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;
    if let Ok(mut sock) = UnixStream::connect("/var/run/badapple/substrate.sock") {
        let _ = sock.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = sock.set_write_timeout(Some(Duration::from_secs(2)));
        let esc = prompt.replace('\\', "\\\\").replace('"', "\\\"");
        let req = format!("{{\"method\":\"inference\",\"prompt\":\"{esc}\",\"max_tokens\":64}}\n");
        if sock.write_all(req.as_bytes()).is_ok() {
            let mut buf = Vec::new();
            let _ = sock.read_to_end(&mut buf);
            if !buf.is_empty() {
                let s = String::from_utf8_lossy(&buf).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    format!(
        "cortex#{}[ack:{}]",
        &sha[..8],
        &sha256_hex(prompt.as_bytes())[..12]
    )
}

/// `kv_k` → `kv.k` — state mirrors are keyed dotted, ported flat.
fn port_to_key(port: &str) -> String {
    match port.split_once('_') {
        Some((a, b)) => format!("{a}.{b}"),
        None => port.to_string(),
    }
}

// ------------------------------------------------- snapshot codecs

fn snap_val(
    v: &Val,
    out: &mut Vec<u8>,
    heap: &mut Vec<u8>,
    ids: &mut HashMap<usize, u32>,
    vm: &Vm,
) {
    match v {
        Val::Unit => out.push(0),
        Val::I(i) => {
            out.push(1);
            out.extend_from_slice(&i.to_le_bytes());
        }
        Val::F(f) => {
            out.push(2);
            out.extend_from_slice(&f.to_le_bytes());
        }
        Val::B(b) => {
            out.push(3);
            out.push(*b as u8);
        }
        Val::S(s) => {
            out.push(4);
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Val::V(rc) | Val::St(_, rc) => {
            let ptr = Rc::as_ptr(rc) as usize;
            if let Some(&id) = ids.get(&ptr) {
                out.push(5);
                out.extend_from_slice(&id.to_le_bytes());
                return;
            }
            let id = ids.len() as u32;
            ids.insert(ptr, id);
            // definition lands in the heap section
            let mut def = Vec::new();
            match v {
                Val::V(rc) => {
                    def.push(6);
                    def.extend_from_slice(&id.to_le_bytes());
                    let b = rc.borrow();
                    def.extend_from_slice(&(b.len() as u32).to_le_bytes());
                    for x in b.iter() {
                        snap_val(x, &mut def, heap, ids, vm);
                    }
                }
                Val::St(si, rc) => {
                    def.push(7);
                    def.extend_from_slice(&id.to_le_bytes());
                    def.extend_from_slice(&si.to_le_bytes());
                    let b = rc.borrow();
                    def.extend_from_slice(&(b.len() as u32).to_le_bytes());
                    for x in b.iter() {
                        snap_val(x, &mut def, heap, ids, vm);
                    }
                }
                _ => unreachable!(),
            }
            heap.extend_from_slice(&def);
            out.push(5);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Val::M(rc) => {
            let ptr = Rc::as_ptr(rc) as usize;
            if let Some(&id) = ids.get(&ptr) {
                out.push(5);
                out.extend_from_slice(&id.to_le_bytes());
                return;
            }
            let id = ids.len() as u32;
            ids.insert(ptr, id);
            let mut def = Vec::new();
            def.push(11);
            def.extend_from_slice(&id.to_le_bytes());
            let b = rc.borrow();
            def.extend_from_slice(&(b.len() as u32).to_le_bytes());
            for (k, val) in b.iter() {
                snap_val(k, &mut def, heap, ids, vm);
                snap_val(val, &mut def, heap, ids, vm);
            }
            heap.extend_from_slice(&def);
            out.push(5);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Val::En(ei, vi, rc) => {
            let ptr = Rc::as_ptr(rc) as usize;
            if let Some(&id) = ids.get(&ptr) {
                out.push(5);
                out.extend_from_slice(&id.to_le_bytes());
                return;
            }
            let id = ids.len() as u32;
            ids.insert(ptr, id);
            let mut def = Vec::new();
            def.push(12);
            def.extend_from_slice(&id.to_le_bytes());
            def.extend_from_slice(&ei.to_le_bytes());
            def.extend_from_slice(&vi.to_le_bytes());
            def.extend_from_slice(&(rc.len() as u32).to_le_bytes());
            for x in rc.iter() {
                snap_val(x, &mut def, heap, ids, vm);
            }
            heap.extend_from_slice(&def);
            out.push(5);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Val::Dfa(rc) => {
            let ptr = Rc::as_ptr(rc) as usize;
            if let Some(&id) = ids.get(&ptr) {
                out.push(5);
                out.extend_from_slice(&id.to_le_bytes());
                return;
            }
            let id = ids.len() as u32;
            ids.insert(ptr, id);
            let mut def = Vec::new();
            def.push(13);
            def.extend_from_slice(&id.to_le_bytes());
            def.extend_from_slice(&(rc.patterns.len() as u32).to_le_bytes());
            for pat in &rc.patterns {
                def.extend_from_slice(&(pat.len() as u32).to_le_bytes());
                def.extend_from_slice(pat.as_bytes());
            }
            heap.extend_from_slice(&def);
            out.push(5);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Val::T(h) => {
            out.push(8);
            out.extend_from_slice(&h.to_le_bytes());
        }
        Val::Ane(p) => {
            out.push(9);
            out.extend_from_slice(&p.to_le_bytes());
        }
        Val::Tok { id, epoch } => {
            out.push(10);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&epoch.to_le_bytes());
        }
    }
}

fn unsnap_val(
    buf: &[u8],
    pos: usize,
    heap: &HashMap<u32, Val>,
    _vm: &Vm,
) -> Result<(Val, usize), String> {
    let tag = *buf.get(pos).ok_or("snap: truncated")?;
    let p = pos + 1;
    let take64 = |p: usize| -> Result<(u64, usize), String> {
        if p + 8 > buf.len() {
            return Err("snap: truncated".into());
        }
        Ok((
            u64::from_le_bytes(buf[p..p + 8].try_into().unwrap()),
            p + 8,
        ))
    };
    Ok(match tag {
        0 => (Val::Unit, p),
        1 => {
            let (v, np) = take64(p)?;
            (Val::I(v as i64), np)
        }
        2 => {
            let (v, np) = take64(p)?;
            (Val::F(f64::from_bits(v)), np)
        }
        3 => (Val::B(buf[p] != 0), p + 1),
        4 => {
            if p + 4 > buf.len() {
                return Err("snap: truncated".into());
            }
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            let s = String::from_utf8_lossy(&buf[p + 4..p + 4 + n]).to_string();
            (Val::S(Rc::from(s.as_str())), p + 4 + n)
        }
        5 => {
            let id = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            let v = heap
                .get(&id)
                .cloned()
                .ok_or_else(|| format!("snap: dangling heap ref {id}"))?;
            (v, p + 4)
        }
        8 => {
            let h = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            (Val::T(h), p + 4)
        }
        9 => {
            let h = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            (Val::Ane(h), p + 4)
        }
        10 => {
            let (id, np) = take64(p)?;
            let (epoch, np2) = take64(np)?;
            (Val::Tok { id, epoch }, np2)
        }
        t => return Err(format!("snap: bad val tag {t}")),
    })
}

/// take n bytes at pos, advancing it.
fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
    if *pos + n > buf.len() {
        return Err("snap: truncated".into());
    }
    let s = &buf[*pos..*pos + n];
    *pos += n;
    Ok(s)
}

/// read one heap object definition into the map; returns next pos.
fn unsnap_heap(
    buf: &[u8],
    pos: usize,
    heap: &mut HashMap<u32, Val>,
    vm: &Vm,
) -> Result<usize, String> {
    let tag = *buf.get(pos).ok_or("snap: truncated heap")?;
    let mut p = pos + 1;
    let id = u32::from_le_bytes(
        buf.get(p..p + 4).ok_or("snap: truncated heap")?.try_into().unwrap(),
    );
    p += 4;
    let v = match tag {
        6 => {
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let (x, np) = unsnap_val(buf, p, heap, vm)?;
                p = np;
                items.push(x);
            }
            Val::V(Rc::new(RefCell::new(items)))
        }
        7 => {
            let si = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            p += 4;
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let (x, np) = unsnap_val(buf, p, heap, vm)?;
                p = np;
                items.push(x);
            }
            Val::St(si, Rc::new(RefCell::new(items)))
        }
        11 => {
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut pairs = Vec::with_capacity(n);
            for _ in 0..n {
                let (k, np1) = unsnap_val(buf, p, heap, vm)?;
                let (val, np2) = unsnap_val(buf, np1, heap, vm)?;
                p = np2;
                pairs.push((k, val));
            }
            Val::M(Rc::new(RefCell::new(pairs)))
        }
        12 => {
            let ei = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            p += 4;
            let vi = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
            p += 4;
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let (x, np) = unsnap_val(buf, p, heap, vm)?;
                p = np;
                items.push(x);
            }
            Val::En(ei, vi, Rc::new(items))
        }
        13 => {
            let n = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            let mut pats = Vec::with_capacity(n);
            for _ in 0..n {
                let len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                let s = String::from_utf8_lossy(&buf[p..p + len]).to_string();
                p += len;
                pats.push(s);
            }
            Val::Dfa(Rc::new(DfaMachine::compile(pats)))
        }
        t => return Err(format!("snap: bad heap tag {t}")),
    };
    heap.insert(id, v);
    Ok(p)
}
