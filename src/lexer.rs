//! Tokenizer for coyote source.

use crate::{diag, CResult};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),

    // keywords
    Fn,
    Let,
    Var,
    If,
    Else,
    For,
    In,
    Param,
    State,
    Io,
    Cap,
    Invariant,
    Provenance,
    Frozen,
    Trainable,
    Sealed,
    Det,
    Stoch,
    Out,
    DirIn,
    Return,

    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semi,
    Arrow,
    DotDot,
    Dot,
    T, // .T transpose
    At,
    Plus,
    Minus,
    Star,
    Slash,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    EqEq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    LtAngle, // < after fn name (generic dims)
    GtAngle, // > — same token as Gt; parser disambiguates
    Eof,
}

#[derive(Debug, Clone)]
pub struct Lexed {
    pub tok: Tok,
    pub line: usize,
    pub col: usize,
}

fn kw(s: &str) -> Option<Tok> {
    Some(match s {
        "fn" => Tok::Fn,
        "let" => Tok::Let,
        "var" => Tok::Var,
        "if" => Tok::If,
        "else" => Tok::Else,
        "for" => Tok::For,
        "in" => Tok::In,
        "param" => Tok::Param,
        "state" => Tok::State,
        "io" => Tok::Io,
        "cap" => Tok::Cap,
        "invariant" => Tok::Invariant,
        "provenance" => Tok::Provenance,
        "frozen" => Tok::Frozen,
        "trainable" => Tok::Trainable,
        "sealed" => Tok::Sealed,
        "det" => Tok::Det,
        "stoch" => Tok::Stoch,
        "out" => Tok::Out,
        "dirin" => Tok::DirIn,
        "return" => Tok::Return,
        _ => return None,
    })
}

pub fn lex(src: &str) -> CResult<Vec<Lexed>> {
    let mut toks = Vec::new();
    let mut it = src.chars().peekable();
    let (mut line, mut col) = (1usize, 1usize);
    macro_rules! push {
        ($t:expr) => {
            toks.push(Lexed {
                tok: $t,
                line,
                col,
            })
        };
    }
    let bump = |it: &mut std::iter::Peekable<std::str::Chars>, line: &mut usize, col: &mut usize| -> Option<char> {
        let c = it.next()?;
        if c == '\n' {
            *line += 1;
            *col = 1;
        } else {
            *col += 1;
        }
        Some(c)
    };
    while let Some(&c) = it.peek() {
        if c.is_whitespace() {
            bump(&mut it, &mut line, &mut col);
            continue;
        }
        if c == '#' || (c == '/' && matches!(it.clone().nth(1), Some('/'))) {
            while let Some(&c2) = it.peek() {
                bump(&mut it, &mut line, &mut col);
                if c2 == '\n' {
                    break;
                }
            }
            continue;
        }
        let (l, cl) = (line, col);
        macro_rules! pushat {
            ($t:expr) => {
                toks.push(Lexed {
                    tok: $t,
                    line: l,
                    col: cl,
                })
            };
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut s = String::new();
            while let Some(&c2) = it.peek() {
                if c2.is_ascii_alphanumeric() || c2 == '_' || c2 == ':' {
                    if c2 == ':' && it.clone().nth(1) != Some(':') {
                        break;
                    }
                    if c2 == ':' {
                        bump(&mut it, &mut line, &mut col);
                        bump(&mut it, &mut line, &mut col);
                        s.push_str("::");
                        continue;
                    }
                    s.push(c2);
                    bump(&mut it, &mut line, &mut col);
                } else {
                    break;
                }
            }
            pushat!(kw(&s).unwrap_or(Tok::Ident(s)));
            continue;
        }
        if c.is_ascii_digit() {
            let mut s = String::new();
            let mut is_f = false;
            while let Some(&c2) = it.peek() {
                if c2.is_ascii_digit() || c2 == '_' {
                    s.push(c2);
                    bump(&mut it, &mut line, &mut col);
                } else if c2 == '.' && it.clone().nth(1).map_or(false, |c3| c3.is_ascii_digit()) {
                    is_f = true;
                    s.push('.');
                    bump(&mut it, &mut line, &mut col);
                } else {
                    break;
                }
            }
            let s2: String = s.chars().filter(|&x| x != '_').collect();
            if is_f {
                pushat!(Tok::Float(s2.parse().unwrap_or(0.0)));
            } else {
                pushat!(Tok::Int(s2.parse().unwrap_or(0)));
            }
            continue;
        }
        if c == '"' {
            bump(&mut it, &mut line, &mut col);
            let mut s = String::new();
            while let Some(&c2) = it.peek() {
                if c2 == '"' {
                    bump(&mut it, &mut line, &mut col);
                    break;
                }
                s.push(c2);
                bump(&mut it, &mut line, &mut col);
            }
            pushat!(Tok::Str(s));
            continue;
        }
        bump(&mut it, &mut line, &mut col);
        match c {
            '(' => pushat!(Tok::LParen),
            ')' => pushat!(Tok::RParen),
            '{' => pushat!(Tok::LBrace),
            '}' => pushat!(Tok::RBrace),
            '[' => pushat!(Tok::LBracket),
            ']' => pushat!(Tok::RBracket),
            ',' => pushat!(Tok::Comma),
            ':' => pushat!(Tok::Colon),
            ';' => pushat!(Tok::Semi),
            '@' => pushat!(Tok::At),
            '+' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::PlusEq } else { Tok::Plus }),
            '-' => pushat!(if it.peek() == Some(&'>') { bump(&mut it, &mut line, &mut col); Tok::Arrow }
                          else if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::MinusEq }
                          else { Tok::Minus }),
            '*' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::StarEq } else { Tok::Star }),
            '/' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::SlashEq } else { Tok::Slash }),
            '.' => pushat!(if it.peek() == Some(&'.') { bump(&mut it, &mut line, &mut col); Tok::DotDot }
                          else if it.peek() == Some(&'T') { bump(&mut it, &mut line, &mut col); Tok::T }
                          else { Tok::Dot }),
            '=' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::EqEq } else { Tok::Eq }),
            '!' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::Ne } else { return diag("unexpected '!'", l, cl) }),
            '<' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::Le } else { Tok::Lt }),
            '>' => pushat!(if it.peek() == Some(&'=') { bump(&mut it, &mut line, &mut col); Tok::Ge } else { Tok::Gt }),
            _ => return diag(format!("unexpected char {c:?}"), l, cl),
        }
    }
    push!(Tok::Eof);
    Ok(toks)
}
