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

    /// host-stratum type: `i64` `f64` `bool` `str` `token` `dfa` `unit`,
    /// `[T]` vec, `{K: V}` map, `Name` struct/enum, `f16[N]` tensor handle,
    /// `ane(t, t) -> r` program value.
    fn hty(&mut self) -> CResult<Sp<HTy>> {
        let (l, c) = self.at();
        let t = match self.bump() {
            Tok::LBracket => {
                let inner = self.hty()?;
                self.expect(&Tok::RBracket)?;
                HTy::Vec(Box::new(inner.v))
            }
            Tok::LBrace => {
                let kt = self.hty()?;
                self.expect(&Tok::Colon)?;
                let vt = self.hty()?;
                self.expect(&Tok::RBrace)?;
                HTy::Map(Box::new(kt.v), Box::new(vt.v))
            }
            Tok::Ident(s) => match s.as_str() {
                "i64" => HTy::I64,
                "f64" => HTy::F64,
                "bool" => HTy::Bool,
                "str" => HTy::Str,
                "token" => HTy::Token,
                "dfa" => HTy::Dfa,
                "unit" => HTy::Unit,
                "ane" => {
                    self.expect(&Tok::LParen)?;
                    let mut args = Vec::new();
                    if self.peek() != &Tok::RParen {
                        loop {
                            args.push(self.hty()?.v);
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Arrow)?;
                    let ret = self.hty()?;
                    HTy::AneFn(args, Box::new(ret.v))
                }
                _ => {
                    if let Some(dt) = DType::parse(&s) {
                        let mut dims = Vec::new();
                        if self.eat(&Tok::LBracket) {
                            loop {
                                let (dl, dc) = self.at();
                                let d = match self.bump() {
                                    Tok::Int(n) => Dim::Lit(n),
                                    Tok::Ident(dn) => Dim::Name(dn),
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
                        HTy::Tensor(Ty { dt, dims })
                    } else {
                        HTy::Struct(s)
                    }
                }
            },
            t => return diag(format!("expected host type, found {t:?}"), l, c),
        };
        Ok(sp(t, l, c))
    }

    // ---- expressions (precedence climbing) ----

    fn expr(&mut self) -> CResult<Sp<Expr>> {
        self.or()
    }
    fn or(&mut self) -> CResult<Sp<Expr>> {
        let mut a = self.and()?;
        loop {
            let (l, c) = self.at();
            if self.eat(&Tok::Or) {
                let b = self.and()?;
                a = sp(Expr::Bin(BinOp::Or, Box::new(a), Box::new(b)), l, c);
            } else {
                break;
            }
        }
        Ok(a)
    }
    fn and(&mut self) -> CResult<Sp<Expr>> {
        let mut a = self.cmp()?;
        loop {
            let (l, c) = self.at();
            if self.eat(&Tok::And) {
                let b = self.cmp()?;
                a = sp(Expr::Bin(BinOp::And, Box::new(a), Box::new(b)), l, c);
            } else {
                break;
            }
        }
        Ok(a)
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
                Tok::Percent => BinOp::Mod,
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
        if self.eat(&Tok::Not) {
            let e = self.unary()?;
            return Ok(sp(Expr::Unary(UnOp::Not, Box::new(e)), l, c));
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
            Tok::True => Ok(sp(Expr::Bool(true), l, c)),
            Tok::False => Ok(sp(Expr::Bool(false), l, c)),
            Tok::Str(s) => Ok(sp(Expr::Str(s), l, c)),
            Tok::LBracket => {
                let mut items = Vec::new();
                if self.peek() != &Tok::RBracket {
                    loop {
                        items.push(self.expr()?);
                        if self.eat(&Tok::Comma) {
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RBracket)?;
                Ok(sp(Expr::VecLit(items), l, c))
            }
            Tok::LBrace => {
                let mut items = Vec::new();
                if self.peek() != &Tok::RBrace {
                    loop {
                        let k = self.expr()?;
                        self.expect(&Tok::Colon)?;
                        let v = self.expr()?;
                        items.push((k, v));
                        if self.eat(&Tok::Comma) {
                            if self.peek() == &Tok::RBrace {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                }
                self.expect(&Tok::RBrace)?;
                Ok(sp(Expr::MapLit(items), l, c))
            }
            Tok::Ident(s) => {
                let is_struct_lit = self.peek() == &Tok::LBrace
                    && !s.contains("::")
                    && matches!(
                        (
                            self.t.get(self.i + 1).map(|x| &x.tok),
                            self.t.get(self.i + 2).map(|x| &x.tok),
                        ),
                        (Some(Tok::Ident(_)), Some(Tok::Colon))
                    );
                if is_struct_lit {
                    // `Name { f: e, ... }` — a struct literal
                    self.bump();
                    let mut fields = Vec::new();
                    while self.peek() != &Tok::RBrace {
                        let fname = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        let fe = self.expr()?;
                        fields.push((fname, fe));
                        self.eat(&Tok::Comma);
                    }
                    self.expect(&Tok::RBrace)?;
                    return Ok(sp(Expr::StructLit(sp(s, l, c), fields), l, c));
                }
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

    /// `{ stmt* }` — a block in the current stratum.
    fn block(&mut self, host: bool) -> CResult<Vec<Sp<Stmt>>> {
        self.expect(&Tok::LBrace)?;
        let mut body = Vec::new();
        while self.peek() != &Tok::RBrace && self.peek() != &Tok::Eof {
            body.push(self.stmt(host)?);
        }
        self.expect(&Tok::RBrace)?;
        Ok(body)
    }

    fn stmt(&mut self, host: bool) -> CResult<Sp<Stmt>> {
        let (l, c) = self.at();
        match self.peek().clone() {
            Tok::If | Tok::Match | Tok::While | Tok::For | Tok::Break | Tok::Continue if !host => {
                diag(
                    "control flow is host-stratum — wrap it in `host fn`",
                    l,
                    c,
                )
            }
            Tok::Match => {
                self.bump();
                let expr = self.expr()?;
                self.expect(&Tok::LBrace)?;
                let mut arms = Vec::new();
                while self.peek() != &Tok::RBrace && self.peek() != &Tok::Eof {
                    let (pl, pc) = self.at();
                    let pat = match self.bump() {
                        Tok::Ident(s) if s == "_" => Pat::Wildcard(pl, pc),
                        Tok::Ident(s) => {
                            let mut binds = Vec::new();
                            if self.eat(&Tok::LParen) {
                                if self.peek() != &Tok::RParen {
                                    loop {
                                        binds.push(self.ident()?);
                                        if self.eat(&Tok::Comma) {
                                            continue;
                                        }
                                        break;
                                    }
                                }
                                self.expect(&Tok::RParen)?;
                            }
                            Pat::Variant {
                                name: sp(s, pl, pc),
                                binds,
                            }
                        }
                        t => {
                            return diag(
                                format!("expected pattern (`Enum::Variant` or `_`), found {t:?}"),
                                pl,
                                pc,
                            )
                        }
                    };
                    self.expect(&Tok::FatArrow)?;
                    let body = if self.peek() == &Tok::LBrace {
                        self.block(host)?
                    } else {
                        vec![self.stmt(host)?]
                    };
                    self.eat(&Tok::Comma);
                    arms.push((pat, body));
                }
                self.expect(&Tok::RBrace)?;
                Ok(sp(Stmt::Match { expr, arms }, l, c))
            }
            Tok::If => {
                self.bump();
                let cond = self.expr()?;
                let then = self.block(host)?;
                let els = if self.eat(&Tok::Else) {
                    if self.peek() == &Tok::If {
                        let s = self.stmt(host)?;
                        vec![s]
                    } else {
                        self.block(host)?
                    }
                } else {
                    Vec::new()
                };
                Ok(sp(Stmt::If { cond, then, els }, l, c))
            }
            Tok::While => {
                self.bump();
                let cond = self.expr()?;
                let body = self.block(host)?;
                Ok(sp(Stmt::While { cond, body }, l, c))
            }
            Tok::For => {
                self.bump();
                let var = self.ident()?;
                self.expect(&Tok::In)?;
                let lo = self.expr()?;
                self.expect(&Tok::DotDot)?;
                let hi = self.expr()?;
                let body = self.block(host)?;
                Ok(sp(Stmt::For { var, lo, hi, body }, l, c))
            }
            Tok::Break => {
                self.bump();
                self.eat(&Tok::Semi);
                Ok(sp(Stmt::Break, l, c))
            }
            Tok::Continue => {
                self.bump();
                self.eat(&Tok::Semi);
                Ok(sp(Stmt::Continue, l, c))
            }
            Tok::Let | Tok::Var => {
                let mutable = self.peek() == &Tok::Var;
                self.bump();
                let name = self.ident()?;
                let (hty, ty) = if self.eat(&Tok::Colon) {
                    if host {
                        (Some(self.hty()?), None)
                    } else {
                        (None, Some(self.ty()?))
                    }
                } else {
                    (None, None)
                };
                self.expect(&Tok::Eq)?;
                let e = self.expr()?;
                self.eat(&Tok::Semi);
                Ok(sp(
                    Stmt::Let {
                        name,
                        hty,
                        ty,
                        mutable,
                        expr: e,
                    },
                    l,
                    c,
                ))
            }
            Tok::Return => {
                self.bump();
                let e = if self.peek() == &Tok::Semi || self.peek() == &Tok::RBrace {
                    None
                } else {
                    Some(self.expr()?)
                };
                self.eat(&Tok::Semi);
                Ok(sp(Stmt::Return(e), l, c))
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
        let body = self.block(false)?;
        Ok(sp(Fn { name, dims, args, ret, body }, l, c))
    }

    /// `host [det|stoch] fn name(arg: hty, ...) [-> hty] [cap c1, c2] { stmt* }`
    fn host_fn_decl(&mut self, pre_det: DetMode) -> CResult<Sp<HFn>> {
        let (l, c) = self.at(); // at `host`
        self.expect(&Tok::Host)?;
        let det = if pre_det != DetMode::Auto {
            pre_det
        } else if self.eat(&Tok::Det) {
            DetMode::Det
        } else if self.eat(&Tok::Stoch) {
            DetMode::Stoch
        } else {
            DetMode::Auto
        };
        self.expect(&Tok::Fn)?;
        let name = self.ident()?.v;
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                let an = self.ident()?;
                self.expect(&Tok::Colon)?;
                let at = self.hty()?;
                args.push((an, at));
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) { Some(self.hty()?) } else { None };
        let caps = if self.eat(&Tok::Cap) {
            let mut cs = Vec::new();
            if self.peek() != &Tok::LBrace {
                cs.push(self.ident()?);
                while self.eat(&Tok::Comma) {
                    cs.push(self.ident()?);
                }
            }
            Some(cs)
        } else {
            None
        };
        let body = self.block(true)?;
        Ok(sp(
            HFn {
                name,
                args,
                ret,
                det,
                caps,
                is_reflex: false,
                body,
            },
            l,
            c,
        ))
    }

    /// `reflex name(arg: hty, ...) [-> Reflex] { stmt* }` — an audited
    /// sub-deliberative reflex arc (`axon`): statically forced to be
    /// `det`, zero-capability (`caps = Some([])`), and returning `Reflex`.
    fn reflex_decl(&mut self) -> CResult<Sp<HFn>> {
        let (l, c) = self.at();
        self.expect(&Tok::Reflex)?;
        let name = self.ident()?.v;
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                let an = self.ident()?;
                self.expect(&Tok::Colon)?;
                let at = self.hty()?;
                args.push((an, at));
                if self.eat(&Tok::Comma) {
                    continue;
                }
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) {
            Some(self.hty()?)
        } else {
            Some(sp(HTy::Struct("Reflex".into()), l, c))
        };
        if self.peek() == &Tok::Cap {
            return diag(
                "reflex declarations cannot hold capabilities — reflexes brake, never steer",
                l,
                c,
            );
        }
        let body = self.block(true)?;
        Ok(sp(
            HFn {
                name,
                args,
                ret,
                det: DetMode::Det,
                caps: Some(Vec::new()),
                is_reflex: true,
                body,
            },
            l,
            c,
        ))
    }

    pub fn program(&mut self) -> CResult<Program> {
        let mut decls = Vec::new();
        loop {
            let (l, c) = self.at();
            match self.peek().clone() {
                Tok::Eof => break,
                Tok::Import => {
                    self.bump();
                    let (sl, sc) = self.at();
                    let path = match self.bump() {
                        Tok::Str(s) => sp(s, sl, sc),
                        t => return diag(format!("import expects a path string, found {t:?}"), sl, sc),
                    };
                    self.eat(&Tok::Semi);
                    decls.push(Decl::Import(path));
                }
                Tok::Fn => decls.push(Decl::Fn(self.fn_decl()?)),
                Tok::Host => decls.push(Decl::HostFn(self.host_fn_decl(DetMode::Auto)?)),
                Tok::Det => {
                    self.bump();
                    decls.push(Decl::HostFn(self.host_fn_decl(DetMode::Det)?));
                }
                Tok::Stoch => {
                    self.bump();
                    decls.push(Decl::HostFn(self.host_fn_decl(DetMode::Stoch)?));
                }
                Tok::Reflex => decls.push(Decl::HostFn(self.reflex_decl()?)),
                Tok::Extern => {
                    self.bump();
                    let det = if self.eat(&Tok::Det) {
                        DetMode::Det
                    } else if self.eat(&Tok::Stoch) {
                        DetMode::Stoch
                    } else {
                        DetMode::Det
                    };
                    self.expect(&Tok::Fn)?;
                    let name = self.ident()?.v;
                    self.expect(&Tok::LParen)?;
                    let mut args = Vec::new();
                    if self.peek() != &Tok::RParen {
                        loop {
                            let an = self.ident()?;
                            self.expect(&Tok::Colon)?;
                            let at = self.hty()?;
                            args.push((an, at));
                            if self.eat(&Tok::Comma) {
                                continue;
                            }
                            break;
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    let ret = if self.eat(&Tok::Arrow) {
                        Some(self.hty()?)
                    } else {
                        None
                    };
                    let mut caps = Vec::new();
                    if self.eat(&Tok::Cap) {
                        if self.peek() != &Tok::Semi {
                            caps.push(self.ident()?);
                            while self.eat(&Tok::Comma) {
                                caps.push(self.ident()?);
                            }
                        }
                    }
                    self.eat(&Tok::Semi);
                    decls.push(Decl::ExternFn(sp(
                        ExternFn {
                            name,
                            args,
                            ret,
                            det,
                            caps,
                        },
                        l,
                        c,
                    )));
                }
                Tok::Struct => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::LBrace)?;
                    let mut fields = Vec::new();
                    while self.peek() != &Tok::RBrace {
                        let f = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        let ft = self.hty()?;
                        fields.push((f, ft));
                        self.eat(&Tok::Comma);
                    }
                    self.expect(&Tok::RBrace)?;
                    decls.push(Decl::Struct { name: n, fields });
                }
                Tok::Enum => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::LBrace)?;
                    let mut variants = Vec::new();
                    while self.peek() != &Tok::RBrace {
                        let vn = self.ident()?;
                        let mut payloads = Vec::new();
                        if self.eat(&Tok::LParen) {
                            if self.peek() != &Tok::RParen {
                                loop {
                                    payloads.push(self.hty()?);
                                    if self.eat(&Tok::Comma) {
                                        continue;
                                    }
                                    break;
                                }
                            }
                            self.expect(&Tok::RParen)?;
                        }
                        variants.push((vn, payloads));
                        self.eat(&Tok::Comma);
                    }
                    self.expect(&Tok::RBrace)?;
                    decls.push(Decl::Enum { name: n, variants });
                }
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
                Tok::Table => {
                    self.bump();
                    let n = self.ident()?;
                    self.expect(&Tok::Colon)?;
                    let ty = self.ty()?;
                    let src = if self.eat(&Tok::At) {
                        match self.bump() {
                            Tok::Str(s) => Some(sp(s, l, c)),
                            t => return diag(format!("table source must be \"file::tensor\", found {t:?}"), l, c),
                        }
                    } else {
                        None
                    };
                    decls.push(Decl::Table { name: n, ty, src });
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

/// Parse a `.cy` file and recursively splice `import "other.cy"`
/// declarations (resolved relative to the importing file's directory,
/// deduplicated in encounter order). Returns `(combined_src, Program)`
/// so the manifest hash covers every imported module.
pub fn parse_file(path: &std::path::Path) -> CResult<(String, Program)> {
    let mut seen = std::collections::HashSet::new();
    let mut combined = String::new();
    let mut decls = Vec::new();
    load_recursive(path, 0, 0, &mut seen, &mut combined, &mut decls)?;
    Ok((combined, Program { decls }))
}

fn load_recursive(
    path: &std::path::Path,
    l: usize,
    c: usize,
    seen: &mut std::collections::HashSet<std::path::PathBuf>,
    combined: &mut String,
    out: &mut Vec<Decl>,
) -> CResult<()> {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(canon) {
        return Ok(());
    }
    let src = std::fs::read_to_string(path)
        .map_err(|e| crate::Diag { msg: format!("{}: {e}", path.display()), line: l, col: c })?;
    let prog = parse(&src)?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    for d in prog.decls {
        match d {
            Decl::Import(rel) => {
                let next = base.join(&rel.v);
                load_recursive(&next, rel.line, rel.col, seen, combined, out)?;
            }
            other => out.push(other),
        }
    }
    combined.push_str(&src);
    combined.push('\n');
    Ok(())
}
