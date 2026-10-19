//! Recursive-descent parser: tokens → AST.

use crate::ast::*;
use crate::lexer::{Lexed, Tok};
use crate::{diag, CResult};

pub struct P {
    t: Vec<Lexed>,
    i: usize,
}

impl P {
    pub fn new(t: Vec<Lexed>) -> Self {
        Self { t, i: 0 }
    }
    fn peek(&self) -> &Tok {
        &self.t[self.i].tok
    }
    fn at(&self) -> (usize, usize) {
        (self.t[self.i].line, self.t[self.i].col)
    }
    fn bump(&mut self) -> Tok {
        let t = self.t[self.i].tok.clone();
        if self.i + 1 < self.t.len() {
            self.i += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &Tok) -> CResult<()> {
        if self.eat(t) {
            Ok(())
        } else {
            let (l, c) = self.at();
            diag(format!("expected {t:?}, found {:?}", self.peek()), l, c)
        }
    }
    fn ident(&mut self) -> CResult<Sp<String>> {
        let (l, c) = self.at();
        match self.bump() {
            Tok::Ident(s) => Ok(sp(s, l, c)),
            t => diag(format!("expected identifier, found {t:?}"), l, c),
        }
    }

    fn ty(&mut self) -> CResult<Sp<Ty>> {
        let (l, c) = self.at();
        let name = match self.bump() {
            Tok::Ident(s) => s,
            t => return diag(format!("expected dtype, found {t:?}"), l, c),
        };
        let dt = DType::parse(&name).ok_or(())
            .map_err(|_| crate::Diag { msg: format!("unknown dtype {name:?}"), line: l, col: c })?;
        let mut dims = Vec::new();
        if self.eat(&Tok::LBracket) {
            loop {
                let (dl, dc) = self.at();
                let d = match self.bump() {
                    Tok::Int(n) => Dim::Lit(n),
                    Tok::Ident(s) => Dim::Name(s),
                    t => return diag(format!("expected dim, found {t:?}"), dl, dc),
                };
                dims.push(d);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
            self.expect(&Tok::RBracket)?;
        }
        Ok(sp(Ty { dt, dims }, l, c))
    }

    // ---- expressions (precedence climbing) ----

    fn expr(&mut self) -> CResult<Sp<Expr>> {
        self.cmp()
    }
    fn cmp(&mut self) -> CResult<Sp<Expr>> {
        let mut a = self.add()?;
        loop {
            let (l, c) = self.at();
            let op = match self.peek() {
                Tok::EqEq => BinOp::Eq,
                Tok::Ne => BinOp::Ne,
                Tok::Lt => BinOp::Lt,
                Tok::Gt => BinOp::Gt,
                Tok::Le => BinOp::Le,
                Tok::Ge => BinOp::Ge,
                _ => break,
            };
            self.bump();
            let b = self.add()?;
            a = sp(Expr::Bin(op, Box::new(a), Box::new(b)), l, c);
        }
        Ok(a)
    }
    fn add(&mut self) -> CResult<Sp<Expr>> {
        let mut a = self.mul()?;
        loop {
            let (l, c) = self.at();
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let b = self.mul()?;
            a = sp(Expr::Bin(op, Box::new(a), Box::new(b)), l, c);
        }
        Ok(a)
    }
    fn mul(&mut self) -> CResult<Sp<Expr>> {
        let mut a = self.unary()?;
        loop {
            let (l, c) = self.at();
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::At => BinOp::MatMul,
                _ => break,
            };
            self.bump();
            let b = self.unary()?;
            a = sp(Expr::Bin(op, Box::new(a), Box::new(b)), l, c);
        }
        Ok(a)
    }
    fn unary(&mut self) -> CResult<Sp<Expr>> {
        let (l, c) = self.at();
        if self.eat(&Tok::Minus) {
            let e = self.unary()?;
            return Ok(sp(Expr::Unary(UnOp::Neg, Box::new(e)), l, c));
        }
        self.postfix()
    }
    fn postfix(&mut self) -> CResult<Sp<Expr>> {
        let mut e = self.atom()?;
        loop {
            let (l, c) = self.at();
            match self.peek() {
                Tok::T => {
                    self.bump();
                    e = sp(Expr::Transpose(Box::new(e)), l, c);
                }
                Tok::Dot => {
                    self.bump();
                    let f = self.ident()?;
                    e = sp(Expr::Field(Box::new(e), f), l, c);
                }
                Tok::LBracket => {
                    self.bump();
                    let mut idx = Vec::new();
                    loop {
                        let (il, ic) = self.at();
                        if self.eat(&Tok::DotDot) {
                            if self.peek() == &Tok::RBracket || self.peek() == &Tok::Comma {
                                idx.push(sp(Index::Range(None, None), il, ic));
                            } else {
                                let hi = self.expr()?;
                                idx.push(sp(Index::Range(None, Some(Box::new(hi))), il, ic));
                            }
                        } else {
                            let lo = self.expr()?;
                            if self.eat(&Tok::DotDot) {
                                if self.peek() == &Tok::RBracket || self.peek() == &Tok::Comma {
                                    idx.push(sp(Index::Range(Some(Box::new(lo)), None), il, ic));
                                } else {
                                    let hi = self.expr()?;
                                    idx.push(sp(Index::Range(Some(Box::new(lo)), Some(Box::new(hi))), il, ic));
                                }
                            } else {
                                idx.push(sp(Index::Expr(lo), il, ic));
                            }
                        }
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        break;
                    }
                    self.expect(&Tok::RBracket)?;
                    e = sp(Expr::Index(Box::new(e), idx), l, c);
                }
                _ => break,
            }
        }
        Ok(e)
    }
    fn atom(&mut self) -> CResult<Sp<Expr>> {
        let (l, c) = self.at();
        match self.bump() {
            Tok::Float(f) => Ok(sp(Expr::Lit(f), l, c)),
            Tok::Int(n) => Ok(sp(Expr::Int(n), l, c)),
            Tok::Ident(s) => {
                if self.peek() == &Tok::LParen {
                    self.bump();
                    let mut pos = Vec::new();
                    let mut named = Vec::new();
                    if self.peek() != &Tok::RParen {
                        loop {
                            // named arg: `ident =` prefix
                            if let Tok::Ident(n) = self.peek().clone() {
                                let save = self.i;
                                self.bump();
                                if self.eat(&Tok::Eq) {
                                    let e = self.expr()?;
                                    named.push((sp(n, l, c), e));
                                } else {
                                    self.i = save;
                                    pos.push(self.expr()?);
                                }
                            } else {
                                pos.push(self.expr()?);
                            }
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(sp(Expr::Call(s, pos, named), l, c))
                } else {
                    Ok(sp(Expr::Ident(s), l, c))
                }
            }
            Tok::LParen => {
                let e = self.expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            t => diag(format!("unexpected {t:?} in expression"), l, c),
        }
    }

    // ---- statements ----

    fn stmt(&mut self) -> CResult<Sp<Stmt>> {
        let (l, c) = self.at();
        match self.peek().clone() {
            Tok::Let | Tok::Var => {
                self.bump();
                let name = self.ident()?;
                let ty = if self.eat(&Tok::Colon) { Some(self.ty()?) } else { None };
                self.expect(&Tok::Eq)?;
                let e = self.expr()?;
                self.eat(&Tok::Semi);
                Ok(sp(
                    Stmt::Let {
                        name,
                        ty,
                        expr: e,
                    },
                    l,
                    c,
                ))
            }
            Tok::Return => {
                self.bump();
                let e = self.expr()?;
                self.eat(&Tok::Semi);
                // return lowers as an expression statement marking exit
                Ok(sp(Stmt::Expr(sp(Expr::Call("__ret".into(), vec![e], vec![]), l, c)), l, c))
            }
            Tok::Ident(_) => {
                // assignment or expression statement — lookahead through
                // the postfix chain: `w -= e`, `kv.k = e`, `kv.k[p] = e`.
                let save = self.i;
                let target = self.postfix()?;
                let (l2, c2) = self.at();
                let op = match self.peek() {
                    Tok::Eq => Some(AssignOp::Set),
                    Tok::PlusEq => Some(AssignOp::Add),
                    Tok::MinusEq => Some(AssignOp::Sub),
                    Tok::StarEq => Some(AssignOp::Mul),
                    Tok::SlashEq => Some(AssignOp::Div),
                    _ => None,
                };
                if let Some(op) = op {
                    self.bump();
                    let e = self.expr()?;
                    self.eat(&Tok::Semi);
                    return match &target.v {
                        Expr::Ident(n) => Ok(sp(
                            Stmt::Assign {
                                name: sp(n.clone(), target.line, target.col),
                                op,
                                expr: e,
                            },
                            l2,
                            c2,
                        )),
                        _ => Ok(sp(Stmt::Store { target, op, expr: e }, l2, c2)),
                    };
                }
                self.i = save;
                let e = self.expr()?;
                self.eat(&Tok::Semi);
                Ok(sp(Stmt::Expr(e), l, c))
            }
            _ => {
                let e = self.expr()?;
                self.eat(&Tok::Semi);
                Ok(sp(Stmt::Expr(e), l, c))
            }
        }
    }

    // ---- declarations ----

    fn fn_decl(&mut self) -> CResult<Sp<Fn>> {
        let (l, c) = self.at(); // at `fn`
        self.expect(&Tok::Fn)?;
        let name = self.ident()?.v;
        let mut dims = Vec::new();
        if self.eat(&Tok::Lt) {
            loop {
                dims.push(self.ident()?);
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
            self.expect(&Tok::Gt)?;
        }
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                let an = self.ident()?;
                self.expect(&Tok::Colon)?;
                let at = self.ty()?;
                args.push((an, at));
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) { Some(self.ty()?) } else { None };
        self.expect(&Tok::LBrace)?;
        let mut body = Vec::new();
        while self.peek() != &Tok::RBrace && self.peek() != &Tok::Eof {
            body.push(self.stmt()?);
        }
        self.expect(&Tok::RBrace)?;
        Ok(sp(Fn { name, dims, args, ret, body }, l, c))
    }

    pub fn program(&mut self) -> CResult<Program> {
        let mut decls = Vec::new();
        loop {
            let (l, c) = self.at();
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Fn => decls.push(Decl::Fn(self.fn_decl()?)),
                Tok::Invariant => {
                    self.bump();
                    let n = self.ident()?;
                    decls.push(Decl::Invariant(n));
                }
                Tok::Cap => {
                    self.bump();
                    let mut caps = vec![self.ident()?];
                    while self.eat(&Tok::Comma) {
                        caps.push(self.ident()?);
                    }
                    decls.push(Decl::Cap(caps));
                }
                Tok::Provenance => {
                    self.bump();
                    self.expect(&Tok::LBrace)?;
                    let mut kvs = Vec::new();
                    while self.peek() != &Tok::RBrace {
                        let k = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        let (vl, vc) = self.at();
                        let v = match self.bump() {
                            Tok::Str(s) => sp(s, vl, vc),
                            Tok::Ident(s) => sp(s, vl, vc),
                            Tok::Int(n) => sp(n.to_string(), vl, vc),
                            t => return diag(format!("provenance value must be a string, found {t:?}"), vl, vc),
                        };
                        kvs.push((k, v));
                        self.eat(&Tok::Comma);
                    }
                    self.expect(&Tok::RBrace)?;
                    decls.push(Decl::Provenance(kvs));
                }
                Tok::Param => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::Colon)?;
                    let qual = match self.peek() {
                        Tok::Frozen => {
                            self.bump();
                            Qual::Frozen
                        }
                        Tok::Trainable => {
                            self.bump();
                            Qual::Trainable
                        }
                        Tok::Sealed => {
                            self.bump();
                            Qual::Sealed
                        }
                        _ => Qual::Plain,
                    };
                    let ty = self.ty()?;
                    // `@ "file.safetensors::tensor"` — a real weight source
                    let src = if self.eat(&Tok::At) {
                        match self.bump() {
                            Tok::Str(s) => Some(sp(s, l, c)),
                            t => return diag(format!("param source must be \"file::tensor\", found {t:?}"), l, c),
                        }
                    } else {
                        None
                    };
                    decls.push(Decl::Param { name: n, qual, ty, src });
                }
                Tok::Io => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::Colon)?;
                    let dir = match self.bump() {
                        Tok::In => IoDir::In,
                        Tok::Out => IoDir::Out,
                        t => return diag(format!("io direction must be `in` or `out`, found {t:?}"), l, c),
                    };
                    let ty = self.ty()?;
                    decls.push(Decl::Io { name: n, dir, ty });
                }
                Tok::State => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::LBrace)?;
                    let mut fields = Vec::new();
                    while self.peek() != &Tok::RBrace {
                        let f = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        let ft = self.ty()?;
                        fields.push((f, ft));
                        self.eat(&Tok::Comma);
                    }
                    self.expect(&Tok::RBrace)?;
                    decls.push(Decl::State { name: n, fields });
                }
                t => return diag(format!("expected declaration, found {t:?}"), l, c),
            }
        }
        Ok(Program { decls })
    }
}

pub fn parse(src: &str) -> CResult<Program> {
    let toks = crate::lexer::lex(src)?;
    P::new(toks).program()
}
