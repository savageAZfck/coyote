//! wasm — compile Coyote's host/organism stratum to a standalone
//! WebAssembly (`\0asm`) binary targeting Bad Apple's `WasmCage` ABI.
//!
//! Bad Apple's `WasmCage` (`wasmi` 0.35, 500k fuel, 1 initial 64 KiB page,
//! max 8 pages) provides four host imports under module `"bad_apple"`:
//!   0: `input_size() -> i32`
//!   1: `input_read(dst: i32)`
//!   2: `alloc(len: i32) -> i32`  (bump allocator starting at offset 1024)
//!   3: `output_write(src: i32, len: i32)`
//! and invokes the exported `"run": () -> ()` function with exported `"memory"`.
//!
//! All Coyote values inside the WASM codegen ride as `i64`:
//! - `i64`: native `i64`
//! - `f64`: bitcast via `i64.reinterpret_f64` / `f64.reinterpret_i64`
//! - `bool`: `0` or `1` (`i64`)
//! - `str`: fat pointer `(len << 32) | ptr` in linear memory
//! - `struct` / `enum` / `vec`: linear memory offset (`bad_apple::alloc`)

use crate::ast::{AssignOp, BinOp, UnOp};
use crate::hostck::{HProgram, RTy, StoreT, HE, HS, HT, NATIVE_NAMES};
use std::collections::HashMap;

const FN_INPUT_SIZE: u32 = 0;
const FN_INPUT_READ: u32 = 1;
const FN_ALLOC: u32 = 2;
const FN_OUTPUT_WRITE: u32 = 3;
const FN_WRITE_STR: u32 = 4;
const FN_I64_TO_STR: u32 = 5;
const FN_STR_CONCAT: u32 = 6;
const FN_STR_EQ: u32 = 7;
const FN_STR_CONTAINS: u32 = 8;
const FN_READ_INPUT: u32 = 9;
const USER_FN_BASE: u32 = 10;

fn leb_u32(mut v: u32, out: &mut Vec<u8>) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn leb_i64(mut v: i64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let done = (v == 0 && (b & 0x40) == 0) || (v == -1 && (b & 0x40) != 0);
        if done {
            out.push(b);
            break;
        } else {
            out.push(b | 0x80);
        }
    }
}

fn leb_i32(v: i32, out: &mut Vec<u8>) {
    leb_i64(v as i64, out);
}

fn push_name(s: &str, out: &mut Vec<u8>) {
    leb_u32(s.len() as u32, out);
    out.extend_from_slice(s.as_bytes());
}

fn push_section(id: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    leb_u32(payload.len() as u32, out);
    out.extend_from_slice(payload);
}

struct DataPool {
    bytes: Vec<u8>,
    intern: HashMap<String, i64>,
}

impl DataPool {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            intern: HashMap::new(),
        }
    }

    /// Intern a string literal in static data (offsets `0..1024`, below
    /// `WasmCage`'s `ALLOC_BASE = 1024`) and return packed `(len << 32) | off`.
    fn str_lit(&mut self, s: &str) -> Result<i64, String> {
        if s.is_empty() {
            return Ok(0);
        }
        if let Some(&v) = self.intern.get(s) {
            return Ok(v);
        }
        let off = self.bytes.len();
        let len = s.len();
        if off + len > 1024 {
            return Err("wasm static string pool exceeded 1024B pre-alloc region".into());
        }
        self.bytes.extend_from_slice(s.as_bytes());
        let packed = ((len as i64) << 32) | (off as i64);
        self.intern.insert(s.to_string(), packed);
        Ok(packed)
    }
}

struct FnGen<'a> {
    h: &'a HProgram,
    pool: &'a mut DataPool,
    code: Vec<u8>,
    scopes: Vec<HashMap<String, u32>>,
    nlocals: u32,
    nargs: u32,
    /// Stack of `(break_depth_from_current, continue_depth_from_current)`
    /// represented as the nesting `depth` at which the loop's `(block (loop` sits.
    loops: Vec<(u32, u32)>,
    depth: u32,
    is_reflex: bool,
    fn_name: String,
}

impl<'a> FnGen<'a> {
    fn slot(&mut self, name: &str) -> u32 {
        for s in self.scopes.iter().rev() {
            if let Some(&i) = s.get(name) {
                return i;
            }
        }
        let i = self.nargs + self.nlocals;
        self.nlocals += 1;
        self.scopes.last_mut().unwrap().insert(name.to_string(), i);
        i
    }

    fn tmp(&mut self) -> u32 {
        let i = self.nargs + self.nlocals;
        self.nlocals += 1;
        i
    }

    fn find(&self, name: &str) -> Option<u32> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).copied()
    }

    fn i64_const(&mut self, v: i64) {
        self.code.push(0x42);
        leb_i64(v, &mut self.code);
    }

    fn i32_const(&mut self, v: i32) {
        self.code.push(0x41);
        leb_i32(v, &mut self.code);
    }

    fn f64_const(&mut self, v: f64) {
        self.code.push(0x44);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn local_get(&mut self, idx: u32) {
        self.code.push(0x20);
        leb_u32(idx, &mut self.code);
    }

    fn local_set(&mut self, idx: u32) {
        self.code.push(0x21);
        leb_u32(idx, &mut self.code);
    }

    fn call(&mut self, fn_idx: u32) {
        self.code.push(0x10);
        leb_u32(fn_idx, &mut self.code);
    }

    fn i64_load(&mut self, offset: u32) {
        self.code.push(0x29);
        leb_u32(3, &mut self.code); // align 8
        leb_u32(offset, &mut self.code);
    }

    fn i64_store(&mut self, offset: u32) {
        self.code.push(0x37);
        leb_u32(3, &mut self.code); // align 8
        leb_u32(offset, &mut self.code);
    }

    fn emit_stringify(&mut self, e: &HT) -> Result<(), String> {
        self.expr(e)?;
        self.emit_val_to_str(&e.ty)
    }

    fn emit_val_to_str(&mut self, ty: &RTy) -> Result<(), String> {
        match ty {
            RTy::Str => Ok(()),
            RTy::I64 | RTy::Token => {
                self.call(FN_I64_TO_STR);
                Ok(())
            }
            RTy::Bool => {
                let t = self.pool.str_lit("true")?;
                let f = self.pool.str_lit("false")?;
                self.code.push(0xa7); // i32.wrap_i64
                self.code.extend_from_slice(&[0x04, 0x7e]); // if (result i64)
                self.depth += 1;
                self.i64_const(t);
                self.code.push(0x05); // else
                self.i64_const(f);
                self.code.push(0x0b); // end
                self.depth -= 1;
                Ok(())
            }
            RTy::Enum(ei) => {
                // Format `Enum::Variant` or `Enum::Variant(payload)`
                let p_loc = self.tmp();
                self.local_set(p_loc);
                let out_loc = self.tmp();
                self.i64_const(0);
                self.local_set(out_loc);
                let (ename, vars) = &self.h.enums[*ei as usize];
                for (vi, (vname, ptys)) in vars.iter().enumerate() {
                    self.local_get(p_loc);
                    self.code.push(0xa7); // i32.wrap_i64
                    self.i64_load(0);
                    self.i64_const(vi as i64);
                    self.code.push(0x51); // i64.eq
                    self.code.extend_from_slice(&[0x04, 0x40]); // if
                    self.depth += 1;
                    if ptys.is_empty() {
                        let s = self.pool.str_lit(&format!("{ename}::{vname}"))?;
                        self.i64_const(s);
                        self.local_set(out_loc);
                    } else {
                        let head = self.pool.str_lit(&format!("{ename}::{vname}("))?;
                        let tail = self.pool.str_lit(")")?;
                        self.i64_const(head);
                        self.local_get(p_loc);
                        self.code.push(0xa7);
                        self.i64_load(8);
                        self.emit_val_to_str(&ptys[0])?;
                        self.call(FN_STR_CONCAT);
                        self.i64_const(tail);
                        self.call(FN_STR_CONCAT);
                        self.local_set(out_loc);
                    }
                    self.code.push(0x0b); // end
                    self.depth -= 1;
                }
                self.local_get(out_loc);
                Ok(())
            }
            _ => {
                let s = self.pool.str_lit("<val>")?;
                self.code.push(0x1a); // drop
                self.i64_const(s);
                Ok(())
            }
        }
    }

    fn expr(&mut self, e: &HT) -> Result<(), String> {
        match &e.e {
            HE::I64(n) => self.i64_const(*n),
            HE::F64(f) => {
                self.f64_const(*f);
                self.code.push(0xbd); // i64.reinterpret_f64
            }
            HE::Bool(b) => self.i64_const(if *b { 1 } else { 0 }),
            HE::Str(s) => {
                let packed = self.pool.str_lit(s)?;
                self.i64_const(packed);
            }
            HE::Var(n) => {
                let idx = self
                    .find(n)
                    .ok_or_else(|| format!("wasm: unbound local {n}"))?;
                self.local_get(idx);
            }
            HE::Global(_) => {
                return Err("wasm cage target does not support ANE/global tensors".into())
            }
            HE::Un(op, a) => {
                self.expr(a)?;
                match (op, &e.ty) {
                    (UnOp::Neg, RTy::I64) => {
                        let tmp = self.tmp();
                        self.local_set(tmp);
                        self.i64_const(0);
                        self.local_get(tmp);
                        self.code.push(0x7d); // i64.sub
                    }
                    (UnOp::Neg, RTy::F64) => {
                        self.code.push(0xbf); // f64.reinterpret_i64
                        self.code.push(0x9a); // f64.neg
                        self.code.push(0xbd); // i64.reinterpret_f64
                    }
                    (UnOp::Not, _) => {
                        self.code.push(0x50); // i64.eqz -> i32
                        self.code.push(0xad); // i64.extend_i32_u
                    }
                    _ => return Err("wasm: unsupported unary".into()),
                }
            }
            HE::Bin(op, a, b) => match op {
                BinOp::And => {
                    self.expr(a)?;
                    self.code.push(0xa7); // i32.wrap_i64
                    self.code.extend_from_slice(&[0x04, 0x7e]); // if (result i64)
                    self.depth += 1;
                    self.expr(b)?;
                    self.code.push(0x05); // else
                    self.i64_const(0);
                    self.code.push(0x0b); // end
                    self.depth -= 1;
                }
                BinOp::Or => {
                    self.expr(a)?;
                    self.code.push(0xa7); // i32.wrap_i64
                    self.code.extend_from_slice(&[0x04, 0x7e]); // if (result i64)
                    self.depth += 1;
                    self.i64_const(1);
                    self.code.push(0x05); // else
                    self.expr(b)?;
                    self.code.push(0x0b); // end
                    self.depth -= 1;
                }
                _ => {
                    self.expr(a)?;
                    self.expr(b)?;
                    match (&a.ty, op) {
                        (RTy::Str, BinOp::Add) => {
                            self.call(FN_STR_CONCAT);
                        }
                        (RTy::Str, BinOp::Eq) => {
                            self.call(FN_STR_EQ);
                        }
                        (RTy::Str, BinOp::Ne) => {
                            self.call(FN_STR_EQ);
                            self.code.push(0x50); // i64.eqz
                            self.code.push(0xad); // i64.extend_i32_u
                        }
                        (RTy::I64, BinOp::Add) => self.code.push(0x7c),
                        (RTy::I64, BinOp::Sub) => self.code.push(0x7d),
                        (RTy::I64, BinOp::Mul) => self.code.push(0x7e),
                        (RTy::I64, BinOp::Div) => self.code.push(0x7f),
                        (RTy::I64, BinOp::Mod) => self.code.push(0x81),
                        (RTy::I64 | RTy::Bool | RTy::Enum(_), BinOp::Eq) => {
                            self.code.push(0x51);
                            self.code.push(0xad);
                        }
                        (RTy::I64 | RTy::Bool | RTy::Enum(_), BinOp::Ne) => {
                            self.code.push(0x52);
                            self.code.push(0xad);
                        }
                        (RTy::I64, BinOp::Lt) => {
                            self.code.push(0x53);
                            self.code.push(0xad);
                        }
                        (RTy::I64, BinOp::Gt) => {
                            self.code.push(0x55);
                            self.code.push(0xad);
                        }
                        (RTy::I64, BinOp::Le) => {
                            self.code.push(0x57);
                            self.code.push(0xad);
                        }
                        (RTy::I64, BinOp::Ge) => {
                            self.code.push(0x59);
                            self.code.push(0xad);
                        }
                        (RTy::F64, _) => {
                            let tb = self.tmp();
                            let ta = self.tmp();
                            self.local_set(tb);
                            self.local_set(ta);
                            self.local_get(ta);
                            self.code.push(0xbf); // f64.reinterpret_i64
                            self.local_get(tb);
                            self.code.push(0xbf);
                            match op {
                                BinOp::Add => {
                                    self.code.push(0xa0);
                                    self.code.push(0xbd);
                                }
                                BinOp::Sub => {
                                    self.code.push(0xa1);
                                    self.code.push(0xbd);
                                }
                                BinOp::Mul => {
                                    self.code.push(0xa2);
                                    self.code.push(0xbd);
                                }
                                BinOp::Div => {
                                    self.code.push(0xa3);
                                    self.code.push(0xbd);
                                }
                                BinOp::Eq => {
                                    self.code.push(0x61);
                                    self.code.push(0xad);
                                }
                                BinOp::Ne => {
                                    self.code.push(0x62);
                                    self.code.push(0xad);
                                }
                                BinOp::Lt => {
                                    self.code.push(0x63);
                                    self.code.push(0xad);
                                }
                                BinOp::Gt => {
                                    self.code.push(0x64);
                                    self.code.push(0xad);
                                }
                                BinOp::Le => {
                                    self.code.push(0x65);
                                    self.code.push(0xad);
                                }
                                BinOp::Ge => {
                                    self.code.push(0x66);
                                    self.code.push(0xad);
                                }
                                _ => return Err("wasm: bad f64 binop".into()),
                            }
                        }
                        _ => return Err(format!("wasm: unsupported binop {op:?}")),
                    }
                }
            },
            HE::Call(fi, args) => {
                for a in args {
                    self.expr(a)?;
                }
                self.call(USER_FN_BASE + *fi);
            }
            HE::CallAne(_, _) | HE::CallDyn(_, _) | HE::CallExt(_, _) => {
                return Err(
                    "wasm cage modules are pure sandboxed compute (no ANE/extern calls)".into(),
                )
            }
            HE::Native(ni, args) => {
                let name = NATIVE_NAMES.get(*ni as usize).copied().unwrap_or("?");
                match name {
                    "print" => {
                        let sp = self.pool.str_lit(" ")?;
                        let nl = self.pool.str_lit("\n")?;
                        for (i, a) in args.iter().enumerate() {
                            if i > 0 {
                                self.i64_const(sp);
                                self.call(FN_WRITE_STR);
                                self.code.push(0x1a); // drop
                            }
                            self.emit_stringify(a)?;
                            self.call(FN_WRITE_STR);
                            self.code.push(0x1a);
                        }
                        self.i64_const(nl);
                        self.call(FN_WRITE_STR);
                    }
                    "str" => {
                        self.emit_stringify(&args[0])?;
                    }
                    "len" => match &args[0].ty {
                        RTy::Str => {
                            self.expr(&args[0])?;
                            self.i64_const(32);
                            self.code.push(0x88); // i64.shr_u
                        }
                        RTy::Vec(_) => {
                            self.expr(&args[0])?;
                            self.code.push(0xa7); // i32.wrap_i64
                            self.i64_load(0);
                        }
                        _ => return Err("wasm: len supports str and vec".into()),
                    },
                    "push" => {
                        // vec header at `v`: `[len: i64, data_ptr: i64]`
                        let v_loc = self.tmp();
                        let x_loc = self.tmp();
                        self.expr(&args[0])?;
                        self.local_set(v_loc);
                        self.expr(&args[1])?;
                        self.local_set(x_loc);
                        let len_loc = self.tmp();
                        self.local_get(v_loc);
                        self.code.push(0xa7);
                        self.i64_load(0);
                        self.local_set(len_loc);
                        // store x at data_ptr + len * 8
                        self.local_get(v_loc);
                        self.code.push(0xa7);
                        self.i64_load(8);
                        self.local_get(len_loc);
                        self.i64_const(8);
                        self.code.push(0x7e); // i64.mul
                        self.code.push(0x7c); // i64.add
                        self.code.push(0xa7); // i32.wrap_i64
                        self.local_get(x_loc);
                        self.i64_store(0);
                        // increment len
                        self.local_get(v_loc);
                        self.code.push(0xa7);
                        self.local_get(len_loc);
                        self.i64_const(1);
                        self.code.push(0x7c);
                        self.i64_store(0);
                        self.i64_const(0);
                    }
                    "contains" => {
                        self.expr(&args[0])?;
                        self.expr(&args[1])?;
                        self.call(FN_STR_CONTAINS);
                    }
                    "slice_str" => {
                        let s_loc = self.tmp();
                        let lo_loc = self.tmp();
                        let hi_loc = self.tmp();
                        self.expr(&args[0])?;
                        self.local_set(s_loc);
                        self.expr(&args[1])?;
                        self.local_set(lo_loc);
                        self.expr(&args[2])?;
                        self.local_set(hi_loc);
                        // ptr = (s & 0xffff_ffff) + lo, len = hi - lo
                        self.local_get(hi_loc);
                        self.local_get(lo_loc);
                        self.code.push(0x7d); // sub
                        self.i64_const(32);
                        self.code.push(0x86); // shl
                        self.local_get(s_loc);
                        self.i64_const(0xffff_ffff);
                        self.code.push(0x83); // and
                        self.local_get(lo_loc);
                        self.code.push(0x7c); // add
                        self.code.push(0x84); // or
                    }
                    "i64" => match &args[0].ty {
                        RTy::I64 | RTy::Bool | RTy::Token => self.expr(&args[0])?,
                        RTy::F64 => {
                            self.expr(&args[0])?;
                            self.code.push(0xbf); // f64.reinterpret_i64
                            self.code.push(0xb0); // i64.trunc_f64_s
                        }
                        _ => return Err("wasm: bad i64() cast".into()),
                    },
                    "f64" => match &args[0].ty {
                        RTy::F64 => self.expr(&args[0])?,
                        RTy::I64 => {
                            self.expr(&args[0])?;
                            self.code.push(0xb9); // f64.convert_i64_s
                            self.code.push(0xbd); // i64.reinterpret_f64
                        }
                        _ => return Err("wasm: bad f64() cast".into()),
                    },
                    "witness" => {
                        let pfx = self.pool.str_lit("[witness:")?;
                        let mid = self.pool.str_lit("] ")?;
                        let nl = self.pool.str_lit("\n")?;
                        self.i64_const(pfx);
                        self.call(FN_WRITE_STR);
                        self.code.push(0x1a);
                        self.emit_stringify(&args[0])?;
                        self.call(FN_WRITE_STR);
                        self.code.push(0x1a);
                        self.i64_const(mid);
                        self.call(FN_WRITE_STR);
                        self.code.push(0x1a);
                        self.emit_stringify(&args[1])?;
                        self.call(FN_WRITE_STR);
                        self.code.push(0x1a);
                        self.i64_const(nl);
                        self.call(FN_WRITE_STR);
                    }
                    "self_token" => {
                        self.i64_const(1);
                    }
                    other => {
                        return Err(format!("wasm: native `{other}` not available in WasmCage"))
                    }
                }
            }
            HE::Index(base, idx) => match &base.ty {
                RTy::Vec(_) => {
                    self.expr(base)?;
                    self.code.push(0xa7);
                    self.i64_load(8); // data_ptr
                    self.expr(idx)?;
                    self.i64_const(8);
                    self.code.push(0x7e); // mul
                    self.code.push(0x7c); // add
                    self.code.push(0xa7);
                    self.i64_load(0);
                }
                RTy::Str => {
                    let s_loc = self.tmp();
                    self.expr(base)?;
                    self.local_set(s_loc);
                    self.i64_const(1i64 << 32);
                    self.local_get(s_loc);
                    self.i64_const(0xffff_ffff);
                    self.code.push(0x83); // and
                    self.expr(idx)?;
                    self.code.push(0x7c); // add
                    self.code.push(0x84); // or
                }
                _ => return Err("wasm: index supports vec and str".into()),
            },
            HE::Field(base, fi) => match &base.ty {
                RTy::Struct(_) => {
                    self.expr(base)?;
                    self.code.push(0xa7); // i32.wrap_i64
                    self.i64_load(*fi * 8);
                }
                RTy::Token => {
                    self.expr(base)?;
                }
                _ => return Err("wasm: bad field access".into()),
            },
            HE::VecLit(items) => {
                // Header (16B): `[len: i64, data_ptr: i64]`, data buffer: `max(items.len(), 64) * 8`
                let cap = items.len().max(64) as i32;
                let hdr_loc = self.tmp();
                let dat_loc = self.tmp();
                self.i32_const(16);
                self.call(FN_ALLOC);
                self.code.push(0xad); // i64.extend_i32_u
                self.local_set(hdr_loc);
                self.i32_const(cap * 8);
                self.call(FN_ALLOC);
                self.code.push(0xad);
                self.local_set(dat_loc);
                // store len and data_ptr
                self.local_get(hdr_loc);
                self.code.push(0xa7);
                self.i64_const(items.len() as i64);
                self.i64_store(0);
                self.local_get(hdr_loc);
                self.code.push(0xa7);
                self.local_get(dat_loc);
                self.i64_store(8);
                for (i, it) in items.iter().enumerate() {
                    self.local_get(dat_loc);
                    self.code.push(0xa7);
                    self.expr(it)?;
                    self.i64_store((i * 8) as u32);
                }
                self.local_get(hdr_loc);
            }
            HE::StructLit(_, vals) => {
                let ptr_loc = self.tmp();
                self.i32_const((vals.len().max(1) * 8) as i32);
                self.call(FN_ALLOC);
                self.code.push(0xad);
                self.local_set(ptr_loc);
                for (i, v) in vals.iter().enumerate() {
                    self.local_get(ptr_loc);
                    self.code.push(0xa7);
                    self.expr(v)?;
                    self.i64_store((i * 8) as u32);
                }
                self.local_get(ptr_loc);
            }
            HE::EnumLit(_, vi, vals) => {
                let ptr_loc = self.tmp();
                self.i32_const(((1 + vals.len()) * 8) as i32);
                self.call(FN_ALLOC);
                self.code.push(0xad);
                self.local_set(ptr_loc);
                self.local_get(ptr_loc);
                self.code.push(0xa7);
                self.i64_const(*vi as i64);
                self.i64_store(0);
                for (i, v) in vals.iter().enumerate() {
                    self.local_get(ptr_loc);
                    self.code.push(0xa7);
                    self.expr(v)?;
                    self.i64_store(((i + 1) * 8) as u32);
                }
                self.local_get(ptr_loc);
            }
            HE::MapLit(_) => return Err("wasm: map literals not supported in minimal cage".into()),
        }
        Ok(())
    }

    fn stmt(&mut self, s: &HS) -> Result<(), String> {
        match s {
            HS::Let { name, expr, .. } => {
                self.expr(expr)?;
                let idx = self.slot(name);
                self.local_set(idx);
            }
            HS::Assign { name, op, expr } => {
                let idx = self
                    .find(name)
                    .ok_or_else(|| format!("wasm: unbound {name}"))?;
                match op {
                    AssignOp::Set => {
                        self.expr(expr)?;
                    }
                    _ => {
                        self.local_get(idx);
                        self.expr(expr)?;
                        match (&expr.ty, op) {
                            (RTy::Str, AssignOp::Add) => self.call(FN_STR_CONCAT),
                            (RTy::I64, AssignOp::Add) => self.code.push(0x7c),
                            (RTy::I64, AssignOp::Sub) => self.code.push(0x7d),
                            (RTy::I64, AssignOp::Mul) => self.code.push(0x7e),
                            (RTy::I64, AssignOp::Div) => self.code.push(0x7f),
                            (RTy::F64, _) => {
                                let tb = self.tmp();
                                let ta = self.tmp();
                                self.local_set(tb);
                                self.local_set(ta);
                                self.local_get(ta);
                                self.code.push(0xbf);
                                self.local_get(tb);
                                self.code.push(0xbf);
                                match op {
                                    AssignOp::Add => self.code.push(0xa0),
                                    AssignOp::Sub => self.code.push(0xa1),
                                    AssignOp::Mul => self.code.push(0xa2),
                                    AssignOp::Div => self.code.push(0xa3),
                                    _ => {}
                                }
                                self.code.push(0xbd);
                            }
                            _ => return Err("wasm: unsupported compound assign".into()),
                        }
                    }
                }
                self.local_set(idx);
            }
            HS::Store { target, expr } => match target {
                StoreT::Field(tb, fi) => {
                    self.expr(tb)?;
                    self.code.push(0xa7);
                    self.expr(expr)?;
                    self.i64_store(*fi * 8);
                }
                StoreT::Index(tb, ti) => {
                    self.expr(tb)?;
                    self.code.push(0xa7);
                    self.i64_load(8);
                    self.expr(ti)?;
                    self.i64_const(8);
                    self.code.push(0x7e);
                    self.code.push(0x7c);
                    self.code.push(0xa7);
                    self.expr(expr)?;
                    self.i64_store(0);
                }
            },
            HS::If { cond, then, els } => {
                self.expr(cond)?;
                self.code.push(0xa7); // i32.wrap_i64
                self.code.extend_from_slice(&[0x04, 0x40]); // if (void)
                self.depth += 1;
                self.scopes.push(HashMap::new());
                for st in then {
                    self.stmt(st)?;
                }
                self.scopes.pop();
                if !els.is_empty() {
                    self.code.push(0x05); // else
                    self.scopes.push(HashMap::new());
                    for st in els {
                        self.stmt(st)?;
                    }
                    self.scopes.pop();
                }
                self.code.push(0x0b); // end
                self.depth -= 1;
            }
            HS::Match { expr, arms } => {
                self.scopes.push(HashMap::new());
                self.expr(expr)?;
                let scr = self.tmp();
                self.local_set(scr);
                // Block for breaking out once an arm matches
                self.code.extend_from_slice(&[0x02, 0x40]); // block
                self.depth += 1;
                let match_block_depth = self.depth;
                for (vi_opt, binds, body) in arms {
                    if let Some(vi) = vi_opt {
                        self.local_get(scr);
                        self.code.push(0xa7);
                        self.i64_load(0);
                        self.i64_const(*vi as i64);
                        self.code.push(0x51); // i64.eq
                        self.code.extend_from_slice(&[0x04, 0x40]); // if
                        self.depth += 1;
                        self.scopes.push(HashMap::new());
                        for (bi, (bname, _)) in binds.iter().enumerate() {
                            let bslot = self.slot(bname);
                            self.local_get(scr);
                            self.code.push(0xa7);
                            self.i64_load(((bi + 1) * 8) as u32);
                            self.local_set(bslot);
                        }
                        for st in body {
                            self.stmt(st)?;
                        }
                        self.scopes.pop();
                        let br_label = self.depth - match_block_depth;
                        self.code.push(0x0c); // br
                        leb_u32(br_label, &mut self.code);
                        self.code.push(0x0b); // end if
                        self.depth -= 1;
                    } else {
                        self.scopes.push(HashMap::new());
                        for st in body {
                            self.stmt(st)?;
                        }
                        self.scopes.pop();
                        let br_label = self.depth - match_block_depth;
                        self.code.push(0x0c); // br
                        leb_u32(br_label, &mut self.code);
                    }
                }
                self.code.push(0x0b); // end block
                self.depth -= 1;
                self.scopes.pop();
            }
            HS::While { cond, body } => {
                self.code.extend_from_slice(&[0x02, 0x40]); // block (break target)
                self.depth += 1;
                let brk_d = self.depth;
                self.code.extend_from_slice(&[0x03, 0x40]); // loop (continue target)
                self.depth += 1;
                let cnt_d = self.depth;
                self.loops.push((brk_d, cnt_d));

                self.expr(cond)?;
                self.code.push(0x50); // i64.eqz
                self.code.push(0x0d); // br_if (break out of outer block)
                leb_u32(self.depth - brk_d, &mut self.code);

                self.scopes.push(HashMap::new());
                for st in body {
                    self.stmt(st)?;
                }
                self.scopes.pop();

                self.code.push(0x0c); // br to loop start
                leb_u32(self.depth - cnt_d, &mut self.code);

                self.loops.pop();
                self.code.push(0x0b); // end loop
                self.depth -= 1;
                self.code.push(0x0b); // end block
                self.depth -= 1;
            }
            HS::For { var, lo, hi, body } => {
                self.scopes.push(HashMap::new());
                self.expr(lo)?;
                let iv = self.slot(var);
                self.local_set(iv);
                self.expr(hi)?;
                let hv = self.tmp();
                self.local_set(hv);

                self.code.extend_from_slice(&[0x02, 0x40]); // outer block (break)
                self.depth += 1;
                let brk_d = self.depth;
                self.code.extend_from_slice(&[0x03, 0x40]); // loop
                self.depth += 1;
                let loop_d = self.depth;

                self.local_get(iv);
                self.local_get(hv);
                self.code.push(0x59); // i64.ge_s
                self.code.push(0x0d); // br_if break
                leb_u32(self.depth - brk_d, &mut self.code);

                // Inner block for `continue` so `continue` jumps to increment
                self.code.extend_from_slice(&[0x02, 0x40]);
                self.depth += 1;
                let cnt_d = self.depth;
                self.loops.push((brk_d, cnt_d));

                for st in body {
                    self.stmt(st)?;
                }

                self.loops.pop();
                self.code.push(0x0b); // end inner continue block
                self.depth -= 1;

                // Increment `iv += 1` and jump to loop head
                self.local_get(iv);
                self.i64_const(1);
                self.code.push(0x7c);
                self.local_set(iv);
                self.code.push(0x0c); // br loop_d
                leb_u32(self.depth - loop_d, &mut self.code);

                self.code.push(0x0b); // end loop
                self.depth -= 1;
                self.code.push(0x0b); // end outer block
                self.depth -= 1;
                self.scopes.pop();
            }
            HS::Break => {
                let (brk_d, _) = *self.loops.last().ok_or("wasm: break outside loop")?;
                self.code.push(0x0c);
                leb_u32(self.depth - brk_d, &mut self.code);
            }
            HS::Continue => {
                let (_, cnt_d) = *self.loops.last().ok_or("wasm: continue outside loop")?;
                self.code.push(0x0c);
                leb_u32(self.depth - cnt_d, &mut self.code);
            }
            HS::Return(e) => {
                match e {
                    Some(t) => self.expr(t)?,
                    None => self.i64_const(0),
                }
                if self.is_reflex {
                    let rloc = self.tmp();
                    self.local_set(rloc);
                    self.local_get(rloc);
                    self.code.push(0xa7);
                    self.i64_load(0);
                    self.code.push(0xa7); // tag != 0
                    self.code.extend_from_slice(&[0x04, 0x40]); // if
                    self.depth += 1;
                    let pfx = self.pool.str_lit(&format!("[reflex:{}] ", self.fn_name))?;
                    let nl = self.pool.str_lit("\n")?;
                    self.i64_const(pfx);
                    self.call(FN_WRITE_STR);
                    self.code.push(0x1a);
                    self.local_get(rloc);
                    let rty = self
                        .h
                        .fns
                        .iter()
                        .find(|f| f.name == self.fn_name)
                        .map(|f| f.ret.clone())
                        .unwrap_or(RTy::Enum(0));
                    self.emit_val_to_str(&rty)?;
                    self.call(FN_WRITE_STR);
                    self.code.push(0x1a);
                    self.i64_const(nl);
                    self.call(FN_WRITE_STR);
                    self.code.push(0x1a);
                    self.code.push(0x0b); // end if
                    self.depth -= 1;
                    self.local_get(rloc);
                }
                self.code.push(0x0f); // return
            }
            HS::Expr(e) => {
                self.expr(e)?;
                self.code.push(0x1a); // drop
            }
        }
        Ok(())
    }
}

/// Emit a complete WebAssembly binary module (`\0asm\x01\0\0\0`) from a
/// checked [`HProgram`], compatible with Bad Apple's `WasmCage`.
pub fn emit_wasm(h: &HProgram) -> Result<Vec<u8>, String> {
    let entry_idx = h.entry.ok_or("wasm: no `host fn main` entry")?;
    let main_fn = &h.fns[entry_idx];
    if main_fn.args.len() > 1 || (main_fn.args.len() == 1 && main_fn.args[0].1 != RTy::Str) {
        return Err("wasm: `host fn main` must take `()` or `(input: str)`".into());
    }

    let mut pool = DataPool::new();
    // Compile all user functions first so `pool` collects all string literals.
    let mut user_bodies: Vec<(u32, Vec<u8>)> = Vec::with_capacity(h.fns.len());
    for f in &h.fns {
        let mut fg = FnGen {
            h,
            pool: &mut pool,
            code: Vec::new(),
            scopes: vec![HashMap::new()],
            nlocals: 0,
            nargs: f.args.len() as u32,
            loops: Vec::new(),
            depth: 0,
            is_reflex: f.is_reflex,
            fn_name: f.name.clone(),
        };
        for (i, (an, _)) in f.args.iter().enumerate() {
            fg.scopes[0].insert(an.clone(), i as u32);
        }
        for s in &f.body {
            fg.stmt(s)?;
        }
        fg.i64_const(0);
        fg.code.push(0x0b); // end function
        user_bodies.push((fg.nlocals, fg.code));
    }

    // ---- Build Type Section (Section 1) ----
    // Type 0: () -> i32               (input_size)
    // Type 1: (i32) -> ()             (input_read)
    // Type 2: (i32) -> i32            (alloc)
    // Type 3: (i32, i32) -> ()        (output_write)
    // Type 4: () -> ()                (exported "run")
    // Type 5: () -> i64               (0-arg user/helper fn)
    // Type 6: (i64) -> i64            (1-arg user/helper fn)
    // Type 7: (i64, i64) -> i64       (2-arg user/helper fn)
    // Additional arities on demand.
    let mut arity_type: HashMap<usize, u32> = HashMap::new();
    arity_type.insert(0, 5);
    arity_type.insert(1, 6);
    arity_type.insert(2, 7);
    let mut extra_arities: Vec<usize> = Vec::new();
    for f in &h.fns {
        let k = f.args.len();
        if !arity_type.contains_key(&k) {
            let tidx = (8 + extra_arities.len()) as u32;
            arity_type.insert(k, tidx);
            extra_arities.push(k);
        }
    }

    let mut sec_type = Vec::new();
    leb_u32((8 + extra_arities.len()) as u32, &mut sec_type);
    // 0: () -> i32
    sec_type.extend_from_slice(&[0x60, 0x00, 0x01, 0x7f]);
    // 1: (i32) -> ()
    sec_type.extend_from_slice(&[0x60, 0x01, 0x7f, 0x00]);
    // 2: (i32) -> i32
    sec_type.extend_from_slice(&[0x60, 0x01, 0x7f, 0x01, 0x7f]);
    // 3: (i32, i32) -> ()
    sec_type.extend_from_slice(&[0x60, 0x02, 0x7f, 0x7f, 0x00]);
    // 4: () -> ()
    sec_type.extend_from_slice(&[0x60, 0x00, 0x00]);
    // 5: () -> i64
    sec_type.extend_from_slice(&[0x60, 0x00, 0x01, 0x7e]);
    // 6: (i64) -> i64
    sec_type.extend_from_slice(&[0x60, 0x01, 0x7e, 0x01, 0x7e]);
    // 7: (i64, i64) -> i64
    sec_type.extend_from_slice(&[0x60, 0x02, 0x7e, 0x7e, 0x01, 0x7e]);
    for &k in &extra_arities {
        sec_type.push(0x60);
        leb_u32(k as u32, &mut sec_type);
        for _ in 0..k {
            sec_type.push(0x7e);
        }
        sec_type.extend_from_slice(&[0x01, 0x7e]);
    }

    // ---- Build Import Section (Section 2) ----
    let mut sec_import = Vec::new();
    leb_u32(4, &mut sec_import);
    for (name, tidx) in [
        ("input_size", 0u32),
        ("input_read", 1u32),
        ("alloc", 2u32),
        ("output_write", 3u32),
    ] {
        push_name("bad_apple", &mut sec_import);
        push_name(name, &mut sec_import);
        sec_import.push(0x00); // func import
        leb_u32(tidx, &mut sec_import);
    }

    // ---- Build Function Section (Section 3) ----
    // 6 built-in helpers (4..=9) + h.fns.len() user fns + 1 exported "run" wrapper
    let total_defined = 6 + h.fns.len() + 1;
    let run_fn_idx = USER_FN_BASE + h.fns.len() as u32;
    let mut sec_func = Vec::new();
    leb_u32(total_defined as u32, &mut sec_func);
    leb_u32(6, &mut sec_func); // 4: __write_str (i64) -> i64
    leb_u32(6, &mut sec_func); // 5: __i64_to_str (i64) -> i64
    leb_u32(7, &mut sec_func); // 6: __str_concat (i64, i64) -> i64
    leb_u32(7, &mut sec_func); // 7: __str_eq (i64, i64) -> i64
    leb_u32(7, &mut sec_func); // 8: __str_contains (i64, i64) -> i64
    leb_u32(5, &mut sec_func); // 9: __read_input () -> i64
    for f in &h.fns {
        leb_u32(arity_type[&f.args.len()], &mut sec_func);
    }
    leb_u32(4, &mut sec_func); // run_fn_idx: () -> ()

    // ---- Build Memory Section (Section 5) ----
    // 1 memory: flags = 1 (has max), initial = 1 page (64 KiB), max = 8 pages (512 KiB)
    let sec_mem = vec![0x01, 0x01, 0x01, 0x08];

    // ---- Build Export Section (Section 7) ----
    let mut sec_export = Vec::new();
    leb_u32(2, &mut sec_export);
    push_name("memory", &mut sec_export);
    sec_export.push(0x02); // memory export
    leb_u32(0, &mut sec_export);
    push_name("run", &mut sec_export);
    sec_export.push(0x00); // func export
    leb_u32(run_fn_idx, &mut sec_export);

    // ---- Build Code Section (Section 10) ----
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(total_defined);
    bodies.push(encode_fn_body(&[], &helper_write_str()));
    bodies.push(encode_fn_body(
        &[(6, 0x7f), (2, 0x7e)],
        &helper_i64_to_str(),
    ));
    bodies.push(encode_fn_body(&[(5, 0x7f)], &helper_str_concat()));
    bodies.push(encode_fn_body(&[(5, 0x7f)], &helper_str_eq()));
    bodies.push(encode_fn_body(&[(6, 0x7f)], &helper_str_contains()));
    bodies.push(encode_fn_body(&[(2, 0x7f)], &helper_read_input()));
    for (nlocals, code) in user_bodies {
        let decls = if nlocals > 0 {
            vec![(nlocals, 0x7e)]
        } else {
            vec![]
        };
        bodies.push(encode_fn_body(&decls, &code));
    }
    // Exported `run` wrapper body
    let mut run_code = Vec::new();
    if main_fn.args.len() == 1 {
        run_code.push(0x10);
        leb_u32(FN_READ_INPUT, &mut run_code);
    }
    run_code.push(0x10);
    leb_u32(USER_FN_BASE + entry_idx as u32, &mut run_code);
    match &main_fn.ret {
        RTy::Unit => {
            run_code.push(0x1a); // drop
        }
        RTy::Str => {
            run_code.push(0x10);
            leb_u32(FN_WRITE_STR, &mut run_code);
            run_code.push(0x1a);
        }
        RTy::I64 | RTy::Bool | RTy::Token => {
            run_code.push(0x10);
            leb_u32(FN_I64_TO_STR, &mut run_code);
            run_code.push(0x10);
            leb_u32(FN_WRITE_STR, &mut run_code);
            run_code.push(0x1a);
        }
        _ => {
            run_code.push(0x1a);
        }
    }
    run_code.push(0x0b); // end
    bodies.push(encode_fn_body(&[], &run_code));

    let mut sec_code = Vec::new();
    leb_u32(bodies.len() as u32, &mut sec_code);
    for b in &bodies {
        leb_u32(b.len() as u32, &mut sec_code);
        sec_code.extend_from_slice(b);
    }

    // ---- Assemble Module ----
    let mut wasm = b"\0asm\x01\0\0\0".to_vec();
    push_section(1, &sec_type, &mut wasm);
    push_section(2, &sec_import, &mut wasm);
    push_section(3, &sec_func, &mut wasm);
    push_section(5, &sec_mem, &mut wasm);
    push_section(7, &sec_export, &mut wasm);
    push_section(10, &sec_code, &mut wasm);
    if !pool.bytes.is_empty() {
        let mut sec_data = Vec::new();
        leb_u32(1, &mut sec_data);
        sec_data.push(0x00); // active segment, memory 0
        sec_data.extend_from_slice(&[0x41, 0x00, 0x0b]); // i32.const 0; end
        leb_u32(pool.bytes.len() as u32, &mut sec_data);
        sec_data.extend_from_slice(&pool.bytes);
        push_section(11, &sec_data, &mut wasm);
    }
    Ok(wasm)
}

fn encode_fn_body(locals: &[(u32, u8)], code: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    leb_u32(locals.len() as u32, &mut body);
    for &(cnt, ty) in locals {
        leb_u32(cnt, &mut body);
        body.push(ty);
    }
    body.extend_from_slice(code);
    body
}

/// Func 4: `__write_str(s: i64) -> i64`
fn helper_write_str() -> Vec<u8> {
    let mut c = Vec::new();
    // output_write(i32.wrap_i64(s), i32.wrap_i64(s >> 32))
    c.extend_from_slice(&[0x20, 0x00, 0xa7]); // local.get 0; i32.wrap_i64
    c.extend_from_slice(&[0x20, 0x00, 0x42, 0x20, 0x88, 0xa7]); // local.get 0; i64.const 32; i64.shr_u; i32.wrap_i64
    c.push(0x10);
    leb_u32(FN_OUTPUT_WRITE, &mut c);
    c.extend_from_slice(&[0x42, 0x00, 0x0b]); // i64.const 0; end
    c
}

/// Func 5: `__i64_to_str(n: i64) -> i64`
/// Locals: 0=n(i64), 1..=6(i32: buf, pos, neg, start, len, d), 7..=8(i64: v, res)
fn helper_i64_to_str() -> Vec<u8> {
    // Allocate 24 bytes at `buf` (local 1), write digits backwards from `pos = buf + 24`,
    // then return `(len << 32) | pos`.
    let mut c = Vec::new();
    // buf = alloc(24)
    c.extend_from_slice(&[0x41, 24, 0x10]);
    leb_u32(FN_ALLOC, &mut c);
    c.extend_from_slice(&[0x21, 0x01]); // local.set 1 (buf)
                                        // pos = buf + 24 (local 2)
    c.extend_from_slice(&[0x20, 0x01, 0x41, 24, 0x6a, 0x21, 0x02]);
    // v = n (local 7)
    c.extend_from_slice(&[0x20, 0x00, 0x21, 0x07]);
    // if n == 0: write '0' at pos - 1 and return (1 << 32) | (pos - 1)
    c.extend_from_slice(&[
        0x20, 0x07, 0x50, // local.get 7; i64.eqz
        0x04, 0x40, // if
        0x20, 0x02, 0x41, 1, 0x6b, 0x21, 0x02, // pos -= 1
        0x20, 0x02, 0x41, 48, 0x3a, 0x00, 0x00, // i32.store8 '0'
        0x42, 0x80, 0x80, 0x80, 0x80, 0x10, // i64.const (1 << 32)
        0x20, 0x02, 0xad, 0x84, // | (pos as u64)
        0x0f, // return
        0x0b, // end if
    ]);
    // if v < 0: neg = 1 (local 3), v = 0 - v
    c.extend_from_slice(&[
        0x20, 0x07, 0x42, 0x00, 0x53, // v < 0
        0x04, 0x40, // if
        0x41, 1, 0x21, 0x03, // neg = 1
        0x42, 0x00, 0x20, 0x07, 0x7d, 0x21, 0x07, // v = 0 - v
        0x0b, // end if
    ]);
    // loop: while v > 0 { pos -= 1; store8(pos, 48 + (v % 10)); v /= 10; }
    c.extend_from_slice(&[
        0x02, 0x40, // block
        0x03, 0x40, // loop
        0x20, 0x07, 0x50, 0x0d, 0x01, // if v == 0 break
        0x20, 0x02, 0x41, 1, 0x6b, 0x21, 0x02, // pos -= 1
        0x20, 0x02, // addr = pos
        0x20, 0x07, 0x42, 10, 0x82, 0xa7, // (v %_u 10) as i32
        0x41, 48, 0x6a, // + '0'
        0x3a, 0x00, 0x00, // i32.store8
        0x20, 0x07, 0x42, 10, 0x80, 0x21, 0x07, // v = v /_u 10
        0x0c, 0x00, // br loop
        0x0b, // end loop
        0x0b, // end block
    ]);
    // if neg == 1: pos -= 1; store8(pos, 45)
    c.extend_from_slice(&[
        0x20, 0x03, 0x04, 0x40, // if neg
        0x20, 0x02, 0x41, 1, 0x6b, 0x21, 0x02, 0x20, 0x02, 0x41, 45, 0x3a, 0x00, 0x00, 0x0b,
    ]);
    // len = (buf + 24) - pos
    c.extend_from_slice(&[
        0x20, 0x01, 0x41, 24, 0x6a, 0x20, 0x02, 0x6b, // (buf + 24) - pos
        0xad, 0x42, 32, 0x86, // (len as u64) << 32
        0x20, 0x02, 0xad, 0x84, // | (pos as u64)
        0x0b, // end
    ]);
    c
}

/// Func 6: `__str_concat(a: i64, b: i64) -> i64`
/// Locals: 0=a(i64), 1=b(i64), 2=pa(i32), 3=la(i32), 4=pb(i32), 5=lb(i32), 6=dst(i32)
fn helper_str_concat() -> Vec<u8> {
    let mut c = Vec::new();
    // pa = i32(a), la = i32(a >> 32), pb = i32(b), lb = i32(b >> 32)
    c.extend_from_slice(&[
        0x20, 0x00, 0xa7, 0x21, 0x02, // pa
        0x20, 0x00, 0x42, 32, 0x88, 0xa7, 0x21, 0x03, // la
        0x20, 0x01, 0xa7, 0x21, 0x04, // pb
        0x20, 0x01, 0x42, 32, 0x88, 0xa7, 0x21, 0x05, // lb
        0x20, 0x03, 0x20, 0x05, 0x6a, // la + lb
        0x45, // i32.eqz
        0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b, // if 0 -> return 0
        0x20, 0x03, 0x20, 0x05, 0x6a, 0x10, // alloc(la + lb)
    ]);
    leb_u32(FN_ALLOC, &mut c);
    c.extend_from_slice(&[
        0x21, 0x06, // dst = local 6
        // memory.copy(dst, pa, la): 0xfc 0x0a 0x00 0x00
        0x20, 0x06, 0x20, 0x02, 0x20, 0x03, 0xfc, 0x0a, 0x00, 0x00,
        // memory.copy(dst + la, pb, lb)
        0x20, 0x06, 0x20, 0x03, 0x6a, 0x20, 0x04, 0x20, 0x05, 0xfc, 0x0a, 0x00, 0x00,
        // return ((la + lb) as u64) << 32 | (dst as u64)
        0x20, 0x03, 0x20, 0x05, 0x6a, 0xad, 0x42, 32, 0x86, 0x20, 0x06, 0xad, 0x84, 0x0b,
    ]);
    c
}

/// Func 7: `__str_eq(a: i64, b: i64) -> i64`
/// Locals: 0=a(i64), 1=b(i64), 2=pa(i32), 3=la(i32), 4=pb(i32), 5=lb(i32), 6=i(i32)
fn helper_str_eq() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&[
        0x20, 0x00, 0xa7, 0x21, 0x02, 0x20, 0x00, 0x42, 32, 0x88, 0xa7, 0x21, 0x03, 0x20, 0x01,
        0xa7, 0x21, 0x04, 0x20, 0x01, 0x42, 32, 0x88, 0xa7, 0x21, 0x05, 0x20, 0x03, 0x20, 0x05,
        0x47, // la != lb
        0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b, // return 0
        0x02, 0x40, // block
        0x03, 0x40, // loop
        0x20, 0x06, 0x20, 0x03, 0x4f, 0x0d, 0x01, // if i >=_u la break
        0x20, 0x02, 0x20, 0x06, 0x6a, 0x2d, 0x00, 0x00, // i32.load8_u(pa + i)
        0x20, 0x04, 0x20, 0x06, 0x6a, 0x2d, 0x00, 0x00, // i32.load8_u(pb + i)
        0x47, // !=
        0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b, // return 0
        0x20, 0x06, 0x41, 1, 0x6a, 0x21, 0x06, // i += 1
        0x0c, 0x00, 0x0b, 0x0b, 0x42, 0x01, 0x0b, // return 1
    ]);
    c
}

/// Func 8: `__str_contains(a: i64, b: i64) -> i64`
/// Locals: 0=a, 1=b, 2=pa, 3=la, 4=pb, 5=lb, 6=i, 7=j
fn helper_str_contains() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&[
        0x20, 0x00, 0xa7, 0x21, 0x02, 0x20, 0x00, 0x42, 32, 0x88, 0xa7, 0x21, 0x03, 0x20, 0x01,
        0xa7, 0x21, 0x04, 0x20, 0x01, 0x42, 32, 0x88, 0xa7, 0x21, 0x05, 0x20, 0x05,
        0x45, // lb == 0 -> return 1
        0x04, 0x40, 0x42, 0x01, 0x0f, 0x0b, 0x20, 0x05, 0x20, 0x03,
        0x4b, // lb >_u la -> return 0
        0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b, // outer loop: while i + lb <= la
        0x02, 0x40, 0x03, 0x40, 0x20, 0x06, 0x20, 0x05, 0x6a, 0x20, 0x03, 0x4b, 0x0d,
        0x01, // if i + lb > la break
        0x41, 0, 0x21, 0x07, // j = 0
        // inner loop: while j < lb
        0x02, 0x40, 0x03, 0x40, 0x20, 0x07, 0x20, 0x05, 0x4f, // j >=_u lb -> found! return 1
        0x04, 0x40, 0x42, 0x01, 0x0f, 0x0b, 0x20, 0x02, 0x20, 0x06, 0x6a, 0x20, 0x07, 0x6a, 0x2d,
        0x00, 0x00, // a[i + j]
        0x20, 0x04, 0x20, 0x07, 0x6a, 0x2d, 0x00, 0x00, // b[j]
        0x47, 0x0d, 0x01, // mismatch -> break inner
        0x20, 0x07, 0x41, 1, 0x6a, 0x21, 0x07, // j += 1
        0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x06, 0x41, 1, 0x6a, 0x21, 0x06, // i += 1
        0x0c, 0x00, 0x0b, 0x0b, 0x42, 0x00, 0x0b,
    ]);
    c
}

/// Func 9: `__read_input() -> i64`
/// Locals: 0=len(i32), 1=ptr(i32)
fn helper_read_input() -> Vec<u8> {
    let mut c = Vec::new();
    c.push(0x10);
    leb_u32(FN_INPUT_SIZE, &mut c);
    c.extend_from_slice(&[
        0x21, 0x00, // len = input_size()
        0x20, 0x00, 0x41, 0, 0x4c, // len <= 0
        0x04, 0x40, 0x42, 0x00, 0x0f, 0x0b, // return 0
        0x20, 0x00, 0x10,
    ]);
    leb_u32(FN_ALLOC, &mut c);
    c.extend_from_slice(&[
        0x21, 0x01, // ptr = alloc(len)
        0x20, 0x01, 0x10,
    ]);
    leb_u32(FN_INPUT_READ, &mut c);
    c.extend_from_slice(&[
        0x20, 0x00, 0xad, 0x42, 32, 0x86, 0x20, 0x01, 0xad, 0x84, 0x0b,
    ]);
    c
}
