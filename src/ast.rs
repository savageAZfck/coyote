//! coyote AST — the parsed surface language.

#[derive(Debug, Clone)]
pub struct Sp<T> {
    pub v: T,
    pub line: usize,
    pub col: usize,
}

pub fn sp<T>(v: T, line: usize, col: usize) -> Sp<T> {
    Sp { v, line, col }
}

/// A dimension: literal or generic name bound at instantiation.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Lit(i64),
    Name(String),
}

/// Element dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    I32,
    Q4,
}

impl DType {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "f16" => Self::F16,
            "f32" => Self::F32,
            "i32" => Self::I32,
            "q4" => Self::Q4,
            _ => return None,
        })
    }
    pub fn mil(self) -> &'static str {
        match self {
            Self::F16 => "fp16",
            Self::F32 => "fp32",
            Self::I32 => "int32",
            Self::Q4 => "int8",
        }
    }
}

/// A type: `dtype[dims]` tensor, or bare `dtype` scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct Ty {
    pub dt: DType,
    pub dims: Vec<Dim>,
}

impl Ty {
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
}

/// Host-stratum types — the organism layer. Tensor types ride through
/// as opaque handles; everything else is CPU data.
#[derive(Debug, Clone, PartialEq)]
pub enum HTy {
    I64,
    F64,
    Bool,
    Str,
    Unit,
    /// `token` — witnessed organism identity (id + epoch).
    Token,
    /// `f16[N, M]` — a handle into the tensor store.
    Tensor(Ty),
    /// `[T]` — a growable vec.
    Vec(Box<HTy>),
    /// a declared `struct` type.
    Struct(String),
    /// `ane(t1, t2) -> r` — a first-class compiled program value.
    AneFn(Vec<HTy>, Box<HTy>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qual {
    Frozen,
    Trainable,
    /// `sealed` — cannot cross the io boundary; enforced at lower time.
    Sealed,
    Plain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoDir {
    In,
    Out,
}

#[derive(Debug, Clone)]
pub enum Decl {
    /// `invariant name` — constitutional clause, name recorded into artifact.
    Invariant(Sp<String>),
    /// `cap hear, see, net::send` — declared capabilities.
    Cap(Vec<Sp<String>>),
    /// `provenance { k: "v" ... }` — signed into the artifact.
    Provenance(Vec<(Sp<String>, Sp<String>)>),
    /// `param w: frozen f16[K, N]` — weight tensor, const or trainable.
    /// Optional `@ "file.safetensors::tensor"` source — without it the
    /// param inits deterministically from its name.
    Param {
        name: Sp<String>,
        qual: Qual,
        ty: Sp<Ty>,
        src: Option<Sp<String>>,
    },
    /// `io x: in f16[K]` / `io y: out f16[N]` — function boundary.
    Io {
        name: Sp<String>,
        dir: IoDir,
        ty: Sp<Ty>,
    },
    /// `state kv { k: f16[L, C, D], len: i32 }` — persistent state block.
    State {
        name: Sp<String>,
        fields: Vec<(Sp<String>, Sp<Ty>)>,
    },
    /// `table embed: f16[V, N] @ "file::tensor"` — a host-stratum data
    /// source: readable, indexable, never lowered to silicon.
    Table {
        name: Sp<String>,
        ty: Sp<Ty>,
        src: Option<Sp<String>>,
    },
    /// `struct Point { x: f64, y: f64 }` — host-stratum data.
    Struct {
        name: Sp<String>,
        fields: Vec<(Sp<String>, Sp<HTy>)>,
    },
    Fn(Sp<Fn>),
    HostFn(Sp<HFn>),
}

#[derive(Debug, Clone)]
pub struct Fn {
    pub name: String,
    /// generic dim names, e.g. `<M, K, N>`
    pub dims: Vec<Sp<String>>,
    pub args: Vec<(Sp<String>, Sp<Ty>)>,
    pub ret: Option<Sp<Ty>>,
    pub body: Vec<Sp<Stmt>>,
}

/// A host-stratum function — bytecode-compiled, runs on the CPU,
/// calls ANE programs as first-class values.
#[derive(Debug, Clone)]
pub struct HFn {
    pub name: String,
    pub args: Vec<(Sp<String>, Sp<HTy>)>,
    pub ret: Option<Sp<HTy>>,
    pub body: Vec<Sp<Stmt>>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: Sp<String>,
        /// host-stratum annotation (None on the tensor stratum)
        hty: Option<Sp<HTy>>,
        ty: Option<Sp<Ty>>,
        mutable: bool,
        expr: Sp<Expr>,
    },
    Assign {
        name: Sp<String>,
        op: AssignOp,
        expr: Sp<Expr>,
    },
    /// assignment to a postfix target — `kv.k = e` or `kv.k[pos] = e`.
    Store {
        target: Sp<Expr>,
        op: AssignOp,
        expr: Sp<Expr>,
    },
    /// host stratum: `if c { } else { }`
    If {
        cond: Sp<Expr>,
        then: Vec<Sp<Stmt>>,
        els: Vec<Sp<Stmt>>,
    },
    /// host stratum: `while c { }`
    While {
        cond: Sp<Expr>,
        body: Vec<Sp<Stmt>>,
    },
    /// host stratum: `for i in lo..hi { }`
    For {
        var: Sp<String>,
        lo: Sp<Expr>,
        hi: Sp<Expr>,
        body: Vec<Sp<Stmt>>,
    },
    Break,
    Continue,
    /// `return e;` — real control flow on the host stratum. On the
    /// tensor stratum it desugars to the `__ret` marker call.
    Return(Option<Sp<Expr>>),
    Expr(Sp<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    /// `[a, b, c]` — vec literal (host stratum).
    VecLit(Vec<Sp<Expr>>),
    /// `Name { f: e, ... }` — struct literal (host stratum).
    StructLit(Sp<String>, Vec<(Sp<String>, Sp<Expr>)>),
    Ident(String),
    Unary(UnOp, Box<Sp<Expr>>),
    Bin(BinOp, Box<Sp<Expr>>, Box<Sp<Expr>>),
    /// `x.T`
    Transpose(Box<Sp<Expr>>),
    /// `f(a, b, dim = last)` — named args allowed.
    Call(String, Vec<Sp<Expr>>, Vec<(Sp<String>, Sp<Expr>)>),
    /// `x[i]` / `x[a..b]` — index/slice (v1: literal or generic indices only).
    Index(Box<Sp<Expr>>, Vec<Sp<Index>>),
    /// `s.field` — struct/state field access.
    Field(Box<Sp<Expr>>, Sp<String>),
}

#[derive(Debug, Clone)]
pub enum Index {
    Expr(Sp<Expr>),
    Range(Option<Box<Sp<Expr>>>, Option<Box<Sp<Expr>>>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    MatMul,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}
