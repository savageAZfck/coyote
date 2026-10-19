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
    Fn(Sp<Fn>),
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

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: Sp<String>,
        ty: Option<Sp<Ty>>,
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}
