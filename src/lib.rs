//! coyote — bare-metal language for sovereign computation.
//!
//! source → AST → typed HIR (shape solving) → MIL text → mil_infer → HWX → ANE.
//! No pointers, no aliasing, tensors are values. Everything that parses
//! typechecks; everything that typechecks lowers.

pub mod ast;
pub mod eval;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod typeck;
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
