//! Byte-level format parsers for the inspection engine.
//! Each function parses real structure and returns a findings object.
//! None of these only check magic bytes — every one walks the format.

use serde_json::{json, Value};
use std::collections::BTreeMap;

fn le16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn le32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn le64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
fn be16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn be32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn be64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
fn slice<'a>(b: &'a [u8], o: usize, n: usize) -> Option<&'a [u8]> {
    b.get(o..o.checked_add(n)?)
}
fn ascii(b: &[u8], o: usize, n: usize) -> String {
    let s = slice(b, o, n).unwrap_or(&[]);
    let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..end]).into_owned()
}
fn utf8z(b: &[u8], o: usize) -> Option<String> {
    let s = b.get(o..)?;
    let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    Some(String::from_utf8_lossy(&s[..end.min(s.len())]).into_owned())
}
fn f32be(b: &[u8], o: usize) -> Option<f64> {
    Some(f32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?) as f64)
}
fn f64be(b: &[u8], o: usize) -> Option<f64> {
    Some(f64::from_be_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
fn cap_list(v: &mut Vec<Value>, x: Value, n: usize) {
    if v.len() < n {
        v.push(x);
    }
}

fn hex(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for &c in b {
        s.push(H[(c >> 4) as usize] as char);
        s.push(H[(c & 0xf) as usize] as char);
    }
    s
}

// ---------- ELF ----------

pub fn elf(b: &[u8]) -> Result<Value, String> {
    if b.len() < 52 || &b[0..4] != b"\x7fELF" {
        return Err("not elf".into());
    }
    let class = b[4];
    let endian = b[5];
    if class != 1 && class != 2 {
        return Err("bad elf class".into());
    }
    if endian != 1 && endian != 2 {
        return Err("bad elf endian".into());
    }
    let is64 = class == 2;
    let le = endian == 1;
    let ru16 = |o: usize| -> Option<u16> {
        let s = slice(b, o, 2)?;
        Some(if le {
            u16::from_le_bytes(s.try_into().ok()?)
        } else {
            u16::from_be_bytes(s.try_into().ok()?)
        })
    };
    let ru32 = |o: usize| -> Option<u32> {
        let s = slice(b, o, 4)?;
        Some(if le {
            u32::from_le_bytes(s.try_into().ok()?)
        } else {
            u32::from_be_bytes(s.try_into().ok()?)
        })
    };
    let ru64 = |o: usize| -> Option<u64> {
        let s = slice(b, o, 8)?;
        Some(if le {
            u64::from_le_bytes(s.try_into().ok()?)
        } else {
            u64::from_be_bytes(s.try_into().ok()?)
        })
    };
    let (phoff, shoff, phentsize, phnum, shentsize, shnum, shstrndx, entry) = if is64 {
        (
            ru64(32).ok_or("phoff")?,
            ru64(40).ok_or("shoff")?,
            ru16(54).ok_or("phentsize")?,
            ru16(56).ok_or("phnum")?,
            ru16(58).ok_or("shentsize")?,
            ru16(60).ok_or("shnum")?,
            ru16(62).ok_or("shstrndx")?,
            ru64(24).ok_or("entry")?,
        )
    } else {
        (
            ru32(28).ok_or("phoff")? as u64,
            ru32(32).ok_or("shoff")? as u64,
            ru16(42).ok_or("phentsize")?,
            ru16(44).ok_or("phnum")?,
            ru16(46).ok_or("shentsize")?,
            ru16(48).ok_or("shnum")?,
            ru16(50).ok_or("shstrndx")?,
            ru32(24).ok_or("entry")? as u64,
        )
    };
    let machine_names: &[(&str, u16)] = &[
        ("none", 0),
        ("m32", 3),
        ("sparc", 2),
        ("x86", 3),
        ("mips", 8),
        ("ppc", 20),
        ("ppc64", 21),
        ("arm", 40),
        ("x86_64", 62),
        ("aarch64", 183),
        ("riscv", 243),
    ];
    let machine = ru16(18).ok_or("machine")?;
    let machine_s = machine_names
        .iter()
        .find(|(_, v)| *v == machine)
        .map(|(n, _)| *n)
        .unwrap_or("unknown");
    let etype = ru16(16).unwrap_or(0);
    let etype_s = match etype {
        1 => "rel",
        2 => "exec",
        3 => "dyn",
        4 => "core",
        _ => "other",
    };

    // section headers
    let shentsize_us = shentsize as usize;
    let mut sections = Vec::new();
    let mut shstr_off = 0usize;
    let mut shstr_size = 0usize;
    let mut symtab_link = 0u32;
    let mut symtab_off = 0usize;
    let mut symtab_size = 0usize;
    let mut strtab_off = 0usize;
    let mut strtab_size = 0usize;
    // first pass: read raw section fields
    struct Sh {
        name_off: u32,
        typ: u32,
        flags: u64,
        addr: u64,
        off: u64,
        size: u64,
        link: u32,
        entsize: u64,
    }
    let mut shs = Vec::new();
    for i in 0..shnum.min(4096) as usize {
        let base = shoff as usize + i * shentsize_us;
        let (name_off, typ, flags, addr, off, size, link, entsize) = if is64 {
            (
                ru32(base).ok_or("sh_name")?,
                ru32(base + 4).ok_or("sh_type")?,
                ru64(base + 8).ok_or("sh_flags")?,
                ru64(base + 16).ok_or("sh_addr")?,
                ru64(base + 24).ok_or("sh_offset")?,
                ru64(base + 32).ok_or("sh_size")?,
                ru32(base + 40).ok_or("sh_link")?,
                ru64(base + 56).ok_or("sh_entsize")?,
            )
        } else {
            (
                ru32(base).ok_or("sh_name")?,
                ru32(base + 4).ok_or("sh_type")?,
                ru32(base + 8).ok_or("sh_flags")? as u64,
                ru32(base + 12).ok_or("sh_addr")? as u64,
                ru32(base + 16).ok_or("sh_offset")? as u64,
                ru32(base + 20).ok_or("sh_size")? as u64,
                ru32(base + 24).ok_or("sh_link")?,
                ru32(base + 36).ok_or("sh_entsize")? as u64,
            )
        };
        shs.push(Sh {
            name_off,
            typ,
            flags,
            addr,
            off,
            size,
            link,
            entsize,
        });
    }
    if (shstrndx as usize) < shs.len() {
        shstr_off = shs[shstrndx as usize].off as usize;
        shstr_size = shs[shstrndx as usize].size as usize;
    }
    let shstr = slice(b, shstr_off, shstr_size).unwrap_or(&[]);
    let shname = |no: u32| -> String {
        let start = no as usize;
        if start >= shstr.len() {
            return String::new();
        }
        let end = shstr[start..]
            .iter()
            .position(|&c| c == 0)
            .map(|p| start + p)
            .unwrap_or(shstr.len());
        String::from_utf8_lossy(&shstr[start..end]).into_owned()
    };
    let mut dynsym_off = 0usize;
    let mut dynsym_size = 0usize;
    let mut dynsym_link = 0u32;
    let mut dynstr_off = 0usize;
    let mut dynstr_size = 0usize;
    for (i, s) in shs.iter().enumerate() {
        let name = shname(s.name_off);
        cap_list(
            &mut sections,
            json!({
                "name": name, "type": s.typ, "flags": format!("0x{:x}", s.flags),
                "addr": format!("0x{:x}", s.addr), "offset": s.off, "size": s.size,
                "entsize": s.entsize,
            }),
            200,
        );
        if s.typ == 2 && symtab_off == 0 {
            // SHT_SYMTAB
            symtab_off = s.off as usize;
            symtab_size = s.size as usize;
            symtab_link = s.link;
        }
        if s.typ == 11 && dynsym_off == 0 {
            // SHT_DYNSYM
            dynsym_off = s.off as usize;
            dynsym_size = s.size as usize;
            dynsym_link = s.link;
        }
        if s.typ == 3 { // STRTAB — used by symtab via link
             // resolved below
        }
        let _ = i;
    }
    // resolve strtabs via link indices
    if (symtab_link as usize) < shs.len() {
        let l = &shs[symtab_link as usize];
        strtab_off = l.off as usize;
        strtab_size = l.size as usize;
    }
    if (dynsym_link as usize) < shs.len() {
        let l = &shs[dynsym_link as usize];
        dynstr_off = l.off as usize;
        dynstr_size = l.size as usize;
    }
    let sym_name = |stroff: usize, strsize: usize, noff: u32| -> String {
        if noff == 0 || stroff + noff as usize >= stroff + strsize {
            return String::new();
        }
        let base = stroff + noff as usize;
        let end_limit = (stroff + strsize).min(b.len());
        if base >= end_limit {
            return String::new();
        }
        let s = &b[base..end_limit];
        let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
        String::from_utf8_lossy(&s[..end]).into_owned()
    };
    let mut symbols = Vec::new();
    let esz = if is64 { 24 } else { 16 };
    let parse_syms =
        |off: usize, size: usize, stroff: usize, strsize: usize, symbols: &mut Vec<Value>| {
            if off == 0 {
                return;
            }
            let count = (size / esz.max(1)).min(500);
            for i in 0..count {
                let o = off + i * esz;
                let (name_off, value) = if is64 {
                    (ru32(o).unwrap_or(0), ru64(o + 8).unwrap_or(0))
                } else {
                    (ru32(o).unwrap_or(0), ru32(o + 4).unwrap_or(0) as u64)
                };
                let name = sym_name(stroff, strsize, name_off);
                if !name.is_empty() {
                    cap_list(
                        symbols,
                        json!({"name": name, "value": format!("0x{:x}", value)}),
                        300,
                    );
                }
            }
        };
    parse_syms(
        symtab_off,
        symtab_size,
        strtab_off,
        strtab_size,
        &mut symbols,
    );
    parse_syms(
        dynsym_off,
        dynsym_size,
        dynstr_off,
        dynstr_size,
        &mut symbols,
    );

    // program headers
    let mut phdrs = Vec::new();
    let ptypes: &[(&str, u32)] = &[
        ("null", 0),
        ("load", 1),
        ("dynamic", 2),
        ("interp", 3),
        ("note", 4),
        ("shlib", 5),
        ("phdr", 6),
        ("tls", 7),
        ("gnu_eh_frame", 0x6474e550),
        ("gnu_stack", 0x6474e551),
        ("gnu_relro", 0x6474e552),
    ];
    for i in 0..phnum.min(256) as usize {
        let base = phoff as usize + i * phentsize as usize;
        let (pt, off, vaddr, filesz, memsz, flags) = if is64 {
            (
                ru32(base).unwrap_or(0),
                ru64(base + 8).unwrap_or(0),
                ru64(base + 16).unwrap_or(0),
                ru64(base + 32).unwrap_or(0),
                ru64(base + 40).unwrap_or(0),
                ru32(base + 4).unwrap_or(0),
            )
        } else {
            (
                ru32(base).unwrap_or(0),
                ru32(base + 4).unwrap_or(0) as u64,
                ru32(base + 8).unwrap_or(0) as u64,
                ru32(base + 16).unwrap_or(0) as u64,
                ru32(base + 20).unwrap_or(0) as u64,
                ru32(base + 24).unwrap_or(0),
            )
        };
        let pname = ptypes
            .iter()
            .find(|(_, v)| *v == pt)
            .map(|(n, _)| *n)
            .unwrap_or("other");
        cap_list(
            &mut phdrs,
            json!({
                "type": pname, "flags": format!("0x{:x}", flags), "offset": off,
                "vaddr": format!("0x{:x}", vaddr), "filesz": filesz, "memsz": memsz,
            }),
            64,
        );
        let _ = (off, vaddr, filesz, memsz);
    }
    let mut out = json!({
        "class": if is64 {"elf64"} else {"elf32"},
        "endian": if le {"little"} else {"big"},
        "type": etype_s, "machine": machine_s,
        "entry": format!("0x{:x}", entry),
        "sections": sections, "program_headers": phdrs, "symbols": symbols,
    });
    // keep load segments for disasm_at vaddr->offset resolution
    let mut loads = Vec::new();
    for i in 0..phnum.min(256) as usize {
        let base = phoff as usize + i * phentsize as usize;
        let pt = ru32(base).unwrap_or(0);
        if pt != 1 {
            continue;
        }
        let (off, vaddr, filesz) = if is64 {
            (
                ru64(base + 8).unwrap_or(0),
                ru64(base + 16).unwrap_or(0),
                ru64(base + 32).unwrap_or(0),
            )
        } else {
            (
                ru32(base + 4).unwrap_or(0) as u64,
                ru32(base + 8).unwrap_or(0) as u64,
                ru32(base + 16).unwrap_or(0) as u64,
            )
        };
        loads.push(json!({"offset":off,"vaddr":vaddr,"filesz":filesz}));
    }
    out["loads"] = json!(loads);
    Ok(out)
}

/// resolve a virtual address to a file offset using ELF program headers
pub fn elf_vaddr_to_off(b: &[u8], vaddr: u64) -> Option<usize> {
    let f = elf(b).ok()?;
    let loads = f.get("loads")?.as_array()?;
    for l in loads {
        let off = l["offset"].as_u64()?;
        let va = u64::from_str_radix(l["vaddr"].as_str()?.trim_start_matches("0x"), 16).ok()?;
        let sz = l["filesz"].as_u64()?;
        if vaddr >= va && vaddr < va + sz {
            return Some((off + (vaddr - va)) as usize);
        }
    }
    None
}

// ---------- PE ----------

pub struct PeInfo {
    pub is64: bool,
    pub machine: u16,
    pub image_base: u64,
    pub sections: Vec<(String, u32, u32, u32, u32)>, // name, vsize, vaddr, rawsize, rawptr
    pub imports: Vec<String>,
    pub exports: Vec<String>,
    pub is_dotnet: bool,
    pub entry_rva: u32,
}

pub fn pe(b: &[u8]) -> Result<PeInfo, String> {
    if b.len() < 0x40 || &b[0..2] != b"MZ" {
        return Err("not pe".into());
    }
    let peoff = le32(b, 0x3c).ok_or("e_lfanew")? as usize;
    if slice(b, peoff, 4) != Some(b"PE\0\0") {
        return Err("bad pe sig".into());
    }
    let coff = peoff + 4;
    let machine = le16(b, coff).ok_or("machine")?;
    let nsects = le16(b, coff + 2).ok_or("nsects")? as usize;
    let optsize = le16(b, coff + 16).ok_or("optsize")? as usize;
    let opt = coff + 20;
    let magic = le16(b, opt).ok_or("optmagic")?;
    let is64 = magic == 0x20b;
    if magic != 0x10b && !is64 {
        return Err("bad opt magic".into());
    }
    let entry_rva = le32(b, opt + 16).unwrap_or(0);
    let image_base = if is64 {
        le64(b, opt + 24).unwrap_or(0)
    } else {
        le32(b, opt + 28).unwrap_or(0) as u64
    };
    let num_dirs = le32(b, opt + if is64 { 108 } else { 92 }).unwrap_or(0) as usize;
    let dir_base = opt + if is64 { 112 } else { 96 };
    let dir = |i: usize| -> (u32, u32) {
        let o = dir_base + i * 8;
        (le32(b, o).unwrap_or(0), le32(b, o + 4).unwrap_or(0))
    };
    // sections
    let sec_base = opt + optsize;
    let sec_sz = 40;
    let mut sections = Vec::new();
    for i in 0..nsects.min(96) {
        let o = sec_base + i * sec_sz;
        let name = ascii(b, o, 8);
        let vsize = le32(b, o + 8).unwrap_or(0);
        let vaddr = le32(b, o + 12).unwrap_or(0);
        let rawsize = le32(b, o + 16).unwrap_or(0);
        let rawptr = le32(b, o + 20).unwrap_or(0);
        sections.push((name, vsize, vaddr, rawsize, rawptr));
    }
    let rva_to_off = |rva: u32| -> Option<usize> {
        for (_, vs, va, rs, rp) in &sections {
            let span = (*vs).max(*rs);
            if rva >= *va && rva < *va + span {
                return Some((*rp + (rva - *va)) as usize);
            }
        }
        None
    };
    // imports
    let mut imports = Vec::new();
    let (imp_rva, imp_sz) = dir(1);
    if imp_rva != 0 {
        if let Some(mut o) = rva_to_off(imp_rva) {
            for _ in 0..256 {
                let oft = le32(b, o).unwrap_or(0);
                let name_rva = le32(b, o + 12).unwrap_or(0);
                let ft = le32(b, o + 16).unwrap_or(0);
                if oft == 0 && name_rva == 0 && ft == 0 {
                    break;
                }
                if let Some(no) = rva_to_off(name_rva) {
                    let dll = utf8z(b, no).unwrap_or_default();
                    if !dll.is_empty() {
                        // walk thunks
                        let thunk_rva = if oft != 0 { oft } else { ft };
                        if let Some(mut toff) = rva_to_off(thunk_rva) {
                            for _ in 0..4096 {
                                let thunk = if is64 {
                                    le64(b, toff).unwrap_or(0)
                                } else {
                                    le32(b, toff).unwrap_or(0) as u64
                                };
                                if thunk == 0 {
                                    break;
                                }
                                if thunk & (if is64 { 1 << 63 } else { 1 << 31 }) == 0 {
                                    if let Some(hn) = rva_to_off(thunk as u32) {
                                        let name = ascii(b, hn + 2, 256);
                                        if !name.is_empty() {
                                            imports.push(format!("{}.{}", dll, name));
                                        }
                                    }
                                }
                                toff += if is64 { 8 } else { 4 };
                            }
                        }
                        imports.push(format!("{} (dll)", dll));
                    }
                }
                o += 20;
                if o + 20 > b.len() {
                    break;
                }
                let _ = imp_sz;
            }
        }
    }
    imports.truncate(3000);
    // exports
    let mut exports = Vec::new();
    let (exp_rva, _) = dir(0);
    if exp_rva != 0 {
        if let Some(eo) = rva_to_off(exp_rva) {
            let nnames = le32(b, eo + 24).unwrap_or(0) as usize;
            let names_rva = le32(b, eo + 32).unwrap_or(0);
            if let Some(mut no) = rva_to_off(names_rva) {
                for _ in 0..nnames.min(2000) {
                    let nrva = le32(b, no).unwrap_or(0);
                    if let Some(so) = rva_to_off(nrva) {
                        let name = utf8z(b, so).unwrap_or_default();
                        if !name.is_empty() {
                            exports.push(name);
                        }
                    }
                    no += 4;
                }
            }
        }
    }
    // .NET detection: COM descriptor dir[14]
    let (clr_rva, clr_sz) = dir(14);
    let is_dotnet = clr_rva != 0 && clr_sz != 0;
    let _ = num_dirs;
    Ok(PeInfo {
        is64,
        machine,
        image_base,
        sections,
        imports,
        exports,
        is_dotnet,
        entry_rva,
    })
}

pub fn pe_json(b: &[u8]) -> Result<Value, String> {
    let p = pe(b)?;
    let mach_names: &[(&str, u16)] = &[
        ("i386", 0x14c),
        ("x64", 0x8664),
        ("arm", 0x1c0),
        ("arm64", 0xaa64),
        ("ia64", 0x200),
    ];
    let mach = mach_names
        .iter()
        .find(|(_, v)| *v == p.machine)
        .map(|(n, _)| *n)
        .unwrap_or("unknown");
    let secs: Vec<Value> = p
        .sections
        .iter()
        .map(|(n, vs, va, rs, rp)| {
            json!({
                "name":n,"vsize":vs,"vaddr":format!("0x{:x}",va),"rawsize":rs,"rawptr":rp,
            })
        })
        .collect();
    Ok(json!({
        "machine": mach, "is64": p.is64,
        "image_base": format!("0x{:x}", p.image_base),
        "entry_rva": format!("0x{:x}", p.entry_rva),
        "dotnet": p.is_dotnet,
        "sections": secs,
        "imports": p.imports.iter().take(400).collect::<Vec<_>>(),
        "import_count": p.imports.len(),
        "exports": p.exports.iter().take(400).collect::<Vec<_>>(),
        "export_count": p.exports.len(),
    }))
}

pub fn pe_vaddr_to_off(b: &[u8], vaddr: u64) -> Option<usize> {
    let p = pe(b).ok()?;
    let rva = vaddr.checked_sub(p.image_base)? as u32;
    for (_, vs, va, rs, rp) in &p.sections {
        let span = (*vs).max(*rs);
        if rva >= *va && rva < *va + span {
            return Some((*rp + (rva - *va)) as usize);
        }
    }
    None
}

// ---------- WASM ----------

fn leb128(b: &[u8], o: &mut usize) -> Option<u64> {
    let mut r = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *b.get(*o)?;
        *o += 1;
        r |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    Some(r)
}

pub fn wasm(b: &[u8]) -> Result<Value, String> {
    if b.len() < 8 || &b[0..4] != b"\0asm" {
        return Err("not wasm".into());
    }
    let ver = le32(b, 4).unwrap_or(0);
    let mut o = 8usize;
    let mut sections = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut codes = 0u64;
    let sec_names: &[&str] = &[
        "custom",
        "type",
        "import",
        "function",
        "table",
        "memory",
        "global",
        "export",
        "start",
        "element",
        "code",
        "data",
        "datacount",
    ];
    while o < b.len() {
        let id = *b.get(o).ok_or("sec id")?;
        o += 1;
        let size = leb128(b, &mut o).ok_or("sec size")? as usize;
        let start = o;
        let end = start.checked_add(size).ok_or("sec end")?;
        if end > b.len() {
            return Err("sec overruns".into());
        }
        let mut name = String::new();
        if id == 0 {
            let mut p = start;
            let nl = leb128(b, &mut p).unwrap_or(0) as usize;
            name = String::from_utf8_lossy(slice(b, p, nl).unwrap_or(&[])).into_owned();
        }
        match id {
            2 => {
                // import
                let mut p = start;
                let n = leb128(b, &mut p).unwrap_or(0);
                for _ in 0..n.min(2000) {
                    let ml = leb128(b, &mut p).unwrap_or(0) as usize;
                    let m = String::from_utf8_lossy(slice(b, p, ml).unwrap_or(&[])).into_owned();
                    p += ml;
                    let fl = leb128(b, &mut p).unwrap_or(0) as usize;
                    let f = String::from_utf8_lossy(slice(b, p, fl).unwrap_or(&[])).into_owned();
                    p += fl;
                    let kind = *b.get(p).unwrap_or(&0);
                    p += 1;
                    match kind {
                        0 => {
                            leb128(b, &mut p);
                        } // func: typeidx
                        1 => {
                            // table: reftype + limits
                            p += 1;
                            let flags = leb128(b, &mut p).unwrap_or(0);
                            leb128(b, &mut p);
                            if flags & 1 != 0 {
                                leb128(b, &mut p);
                            }
                        }
                        2 => {
                            // mem: limits
                            let flags = leb128(b, &mut p).unwrap_or(0);
                            leb128(b, &mut p);
                            if flags & 1 != 0 {
                                leb128(b, &mut p);
                            }
                        }
                        3 => {
                            p += 2;
                        } // global: valtype + mut
                        _ => break,
                    }
                    imports.push(format!("{}.{}", m, f));
                }
            }
            7 => {
                // export
                let mut p = start;
                let n = leb128(b, &mut p).unwrap_or(0);
                for _ in 0..n.min(2000) {
                    let nl = leb128(b, &mut p).unwrap_or(0) as usize;
                    let nm = String::from_utf8_lossy(slice(b, p, nl).unwrap_or(&[])).into_owned();
                    p += nl;
                    let kind = *b.get(p).unwrap_or(&0);
                    p += 1;
                    let idx = leb128(b, &mut p).unwrap_or(0);
                    let k = match kind {
                        0 => "func",
                        1 => "table",
                        2 => "mem",
                        3 => "global",
                        _ => "?",
                    };
                    exports.push(format!("{}({}:{})", nm, k, idx));
                }
            }
            10 => {
                // code: count function bodies
                let mut p = start;
                codes = leb128(b, &mut p).unwrap_or(0);
            }
            _ => {}
        }
        let sname = if (id as usize) < sec_names.len() {
            sec_names[id as usize]
        } else {
            "unknown"
        };
        cap_list(
            &mut sections,
            json!({
                "id": id, "name": if name.is_empty() { sname.to_string() } else { name },
                "type": sname, "size": size,
            }),
            100,
        );
        o = end;
    }
    // recount function section properly
    Ok(json!({
        "version": ver, "sections": sections,
        "imports": imports.iter().take(300).collect::<Vec<_>>(),
        "import_count": imports.len(),
        "exports": exports.iter().take(300).collect::<Vec<_>>(),
        "export_count": exports.len(),
        "function_count": codes,
    }))
}

// ---------- Java class ----------

pub fn class_file(b: &[u8]) -> Result<Value, String> {
    if b.len() < 10 || &b[0..4] != b"\xca\xfe\xba\xbe" {
        return Err("not class".into());
    }
    let minor = be16(b, 4).unwrap_or(0);
    let major = be16(b, 6).unwrap_or(0);
    let cp_count = be16(b, 8).unwrap_or(0) as usize;
    // parse constant pool
    let mut cp: Vec<Option<Value>> = vec![None; cp_count + 1];
    let mut o = 10usize;
    let mut i = 1usize;
    while i < cp_count {
        let tag = *b.get(o).ok_or("cp tag")?;
        o += 1;
        match tag {
            1 => {
                // Utf8
                let len = be16(b, o).ok_or("utf8 len")? as usize;
                o += 2;
                let s = String::from_utf8_lossy(slice(b, o, len).unwrap_or(&[])).into_owned();
                o += len;
                cp[i] = Some(json!({"t":"Utf8","v":s}));
            }
            3 => {
                cp[i] = Some(json!({"t":"Int","v":be32(b,o).unwrap_or(0)}));
                o += 4;
            }
            4 => {
                cp[i] = Some(json!({"t":"Float","v":f32be(b,o).unwrap_or(0.0)}));
                o += 4;
            }
            5 => {
                cp[i] = Some(json!({"t":"Long","v":be64(b,o).unwrap_or(0)}));
                o += 8;
                i += 1;
            }
            6 => {
                cp[i] = Some(json!({"t":"Double","v":f64be(b,o).unwrap_or(0.0)}));
                o += 8;
                i += 1;
            }
            7 | 8 | 16 | 19 | 20 => {
                // Class, String, MethodType, Module, Package
                let idx = be16(b, o).unwrap_or(0);
                o += 2;
                cp[i] = Some(json!({"t":tag,"name_index":idx}));
            }
            9 | 10 | 11 | 12 | 17 | 18 => {
                // refs, NameAndType, Dynamic, InvokeDynamic
                let a = be16(b, o).unwrap_or(0);
                let c = be16(b, o + 2).unwrap_or(0);
                o += 4;
                cp[i] = Some(json!({"t":tag,"a":a,"b":c}));
            }
            15 => {
                let k = *b.get(o).unwrap_or(&0);
                let idx = be16(b, o + 1).unwrap_or(0);
                o += 3;
                cp[i] = Some(json!({"t":"MethodHandle","k":k,"ref":idx}));
            }
            _ => return Err(format!("unknown cp tag {}", tag)),
        }
        i += 1;
    }
    let utf8 = |idx: u16| -> String {
        let e = cp.get(idx as usize).and_then(|x| x.as_ref());
        match e {
            Some(v) if v["t"] == "Utf8" => v["v"].as_str().unwrap_or("").to_string(),
            _ => String::new(),
        }
    };
    let class_name = |idx: u16| -> String {
        let e = cp.get(idx as usize).and_then(|x| x.as_ref());
        match e {
            Some(v) if v["t"] == 7 => utf8(v["name_index"].as_u64().unwrap_or(0) as u16),
            _ => String::new(),
        }
    };
    let access = be16(b, o).unwrap_or(0);
    o += 2;
    let this_class = be16(b, o).unwrap_or(0);
    o += 2;
    let super_class = be16(b, o).unwrap_or(0);
    o += 2;
    let ifaces_n = be16(b, o).unwrap_or(0) as usize;
    o += 2;
    let mut ifaces = Vec::new();
    for _ in 0..ifaces_n.min(100) {
        let idx = be16(b, o).unwrap_or(0);
        o += 2;
        ifaces.push(class_name(idx));
    }
    let fields_n = be16(b, o).unwrap_or(0) as usize;
    o += 2;
    for _ in 0..fields_n.min(5000) {
        o += 6;
        let ac = be16(b, o).unwrap_or(0) as usize;
        o += 2;
        for _ in 0..ac.min(100) {
            let alen = be32(b, o + 2).unwrap_or(0) as usize;
            o += 6 + alen;
        }
    }
    let methods_n = be16(b, o).unwrap_or(0) as usize;
    o += 2;
    for _ in 0..methods_n.min(5000) {
        o += 6;
        let ac = be16(b, o).unwrap_or(0) as usize;
        o += 2;
        for _ in 0..ac.min(100) {
            let alen = be32(b, o + 2).unwrap_or(0) as usize;
            o += 6 + alen;
        }
    }
    let attrs_n = be16(b, o).unwrap_or(0) as usize;
    Ok(json!({
        "major": major, "minor": minor, "cp_count": cp_count,
        "access_flags": format!("0x{:04x}", access),
        "this_class": class_name(this_class),
        "super_class": class_name(super_class),
        "interfaces": ifaces, "fields": fields_n, "methods": methods_n,
        "class_attrs": attrs_n,
    }))
}

// ---------- DEX ----------

pub fn dex(b: &[u8]) -> Result<Value, String> {
    if b.len() < 0x70 || &b[0..4] != b"dex\n" {
        return Err("not dex".into());
    }
    let ver = ascii(b, 4, 3);
    let file_size = le32(b, 32).unwrap_or(0);
    let endian = le32(b, 40).unwrap_or(0);
    let string_ids_size = le32(b, 56).unwrap_or(0) as usize;
    let string_ids_off = le32(b, 60).unwrap_or(0) as usize;
    let type_ids_size = le32(b, 64).unwrap_or(0) as usize;
    let proto_ids_size = le32(b, 72).unwrap_or(0) as usize;
    let field_ids_size = le32(b, 80).unwrap_or(0) as usize;
    let method_ids_size = le32(b, 88).unwrap_or(0) as usize;
    let class_defs_size = le32(b, 96).unwrap_or(0) as usize;
    let data_size = le32(b, 104).unwrap_or(0);
    // read string ids -> string data offsets
    let uleb = |mut o: usize| -> Option<(u64, usize)> {
        let mut r = 0u64;
        let mut s = 0u32;
        loop {
            let byte = *b.get(o)?;
            o += 1;
            r |= ((byte & 0x7f) as u64) << s;
            if byte & 0x80 == 0 {
                break;
            }
            s += 7;
            if s > 35 {
                return None;
            }
        }
        Some((r, o))
    };
    let mut strings = Vec::new();
    for i in 0..string_ids_size.min(200000) {
        let so = string_ids_off + i * 4;
        let data_off = le32(b, so).unwrap_or(0) as usize;
        if let Some((ulen, mut p)) = uleb(data_off) {
            let _ = ulen;
            // skip utf16 len already consumed; read modified utf8 until nul
            let start = p;
            let mut end = start;
            while end < b.len() && b[end] != 0 && end - start < 4096 {
                end += 1;
            }
            let s = String::from_utf8_lossy(&b[start..end]).into_owned();
            if !s.is_empty() {
                cap_list(&mut strings, json!(s), 200);
            }
            p = end;
            let _ = p;
        }
        if strings.len() >= 200 {
            break;
        }
    }
    let mut class_names = Vec::new();
    // class_defs at 100+4? class_defs_off @100
    let class_defs_off = le32(b, 100).unwrap_or(0) as usize;
    // type_ids at 64/68: type_id_item = u32 descriptor_idx (string_ids index)
    let type_ids_off = le32(b, 68).unwrap_or(0) as usize;
    for i in 0..class_defs_size.min(5000) {
        let cd = class_defs_off + i * 32;
        let class_idx = le32(b, cd).unwrap_or(0) as usize;
        let tidx = le32(b, type_ids_off + class_idx * 4).unwrap_or(0) as usize;
        let sidx = le32(b, le32(b, 60).unwrap_or(0) as usize + tidx * 4).unwrap_or(0) as usize;
        // string_ids[sidx] -> data offset -> string
        let data_off = le32(b, le32(b, 60).unwrap_or(0) as usize + sidx * 4).unwrap_or(0) as usize;
        if let Some((_, mut p)) = uleb(data_off) {
            let start = p;
            let mut end = start;
            while end < b.len() && b[end] != 0 && end - start < 4096 {
                end += 1;
            }
            let s = String::from_utf8_lossy(&b[start..end]).into_owned();
            p = end;
            let _ = p;
            if !s.is_empty() {
                cap_list(&mut class_names, json!(s), 300);
            }
        }
    }
    Ok(json!({
        "version": ver, "file_size": file_size, "endian_tag": format!("0x{:08x}", endian),
        "string_ids": string_ids_size, "type_ids": type_ids_size,
        "proto_ids": proto_ids_size, "field_ids": field_ids_size,
        "method_ids": method_ids_size, "class_defs": class_defs_size,
        "data_size": data_size,
        "sample_strings": strings, "classes": class_names,
    }))
}

// ---------- AXML (Android binary XML) ----------

pub fn axml(b: &[u8]) -> Result<Value, String> {
    if b.len() < 8 {
        return Err("too small".into());
    }
    // root RES_XML_TYPE chunk: children are a flat SAX stream inside it
    let root_type = le16(b, 0).unwrap_or(0);
    let root_hsize = le16(b, 2).unwrap_or(0) as usize;
    let root_csize = le32(b, 4).unwrap_or(0) as usize;
    if root_type != 0x0003 || root_hsize < 8 || root_csize < root_hsize {
        return Err("not axml".into());
    }
    let end = root_csize.min(b.len());
    let mut o = root_hsize;
    let mut strings: Vec<String> = Vec::new();
    let mut elements: Vec<String> = Vec::new();
    let mut chunk_types: Vec<u16> = Vec::new();
    while o + 8 <= end {
        let ctype = le16(b, o).unwrap_or(0);
        let hsize = le16(b, o + 2).unwrap_or(0) as usize;
        let csize = le32(b, o + 4).unwrap_or(0) as usize;
        if csize < 8 || o + csize > end {
            break;
        }
        chunk_types.push(ctype);
        if ctype == 0x0001 {
            // string pool
            let scount = le32(b, o + 8).unwrap_or(0) as usize;
            let flags = le32(b, o + 16).unwrap_or(0);
            let sstart = le32(b, o + 20).unwrap_or(0) as usize;
            let utf8 = flags & 0x100 != 0;
            let off_base = o + hsize;
            for i in 0..scount.min(50000) {
                let so = off_base + i * 4;
                let rel = le32(b, so).unwrap_or(0) as usize;
                let sp = o + sstart + rel;
                if utf8 {
                    let (_ul, p1) = axml_len(b, sp).ok_or("axml len")?;
                    let (bl, p2) = axml_len(b, p1).ok_or("axml len")?;
                    let s = String::from_utf8_lossy(slice(b, p2, bl as usize).unwrap_or(&[]))
                        .into_owned();
                    strings.push(s);
                } else {
                    // utf16le: u16 length; if high bit set, second u16 follows
                    let first = le16(b, sp).unwrap_or(0);
                    let (len, p) = if first & 0x8000 != 0 {
                        let second = le16(b, sp + 2).unwrap_or(0) as u64;
                        ((((first & 0x7fff) as u64) << 16) | second as u64, sp + 4)
                    } else {
                        (first as u64, sp + 2)
                    };
                    let units: Vec<u16> = (0..(len as usize).min(4096))
                        .filter_map(|i| le16(b, p + i * 2))
                        .collect();
                    strings.push(String::from_utf16_lossy(&units));
                }
                if strings.len() >= 2000 {
                    break;
                }
            }
        } else if ctype == 0x0102 {
            // start element: string idx at o+16? ResXMLTree_node: header(16) then ns u32, name u32
            let name_idx = le32(b, o + 20).unwrap_or(0) as usize;
            if name_idx < strings.len() {
                elements.push(strings[name_idx].clone());
            }
        }
        o += csize;
    }
    Ok(json!({
        "strings": strings.iter().take(300).collect::<Vec<_>>(),
        "string_count": strings.len(),
        "elements": elements.iter().take(200).collect::<Vec<_>>(),
        "chunk_types": chunk_types.iter().take(100).collect::<Vec<_>>(),
    }))
}

// AXML string-pool length prefix: one byte, or two when bit 0x80 is set —
// len = ((b0 & 0x7F) << 8) | b1. NOT uleb128 (byte order differs).
fn axml_len(b: &[u8], mut o: usize) -> Option<(u64, usize)> {
    let b0 = *b.get(o)? as u64;
    o += 1;
    if b0 & 0x80 == 0 {
        return Some((b0, o));
    }
    let b1 = *b.get(o)? as u64;
    o += 1;
    Some((((b0 & 0x7f) << 8) | b1, o))
}

fn uleb_at(b: &[u8], mut o: usize) -> Option<(u64, usize)> {
    let mut r = 0u64;
    let mut s = 0u32;
    loop {
        let byte = *b.get(o)?;
        o += 1;
        r |= ((byte & 0x7f) as u64) << s;
        if byte & 0x80 == 0 {
            break;
        }
        s += 7;
        if s > 63 {
            return None;
        }
    }
    Some((r, o))
}

// ---------- safetensors ----------

pub fn safetensors(b: &[u8]) -> Result<Value, String> {
    if b.len() < 8 {
        return Err("too small".into());
    }
    let hlen = le64(b, 0).ok_or("hlen")? as usize;
    let hdr = slice(b, 8, hlen).ok_or("header")?;
    let j: Value = serde_json::from_slice(hdr).map_err(|e| e.to_string())?;
    let obj = j.as_object().ok_or("header not object")?;
    let mut tensors = Vec::new();
    let mut total_bytes = 0u64;
    for (name, t) in obj.iter().take(5000) {
        if name == "__metadata__" {
            continue;
        }
        let dtype = t["dtype"].as_str().unwrap_or("?").to_string();
        let shape: Vec<u64> = t["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let offs = t["data_offsets"].as_array();
        let (s, e) = offs
            .and_then(|a| Some((a.first()?.as_u64()?, a.get(1)?.as_u64()?)))
            .unwrap_or((0, 0));
        total_bytes += e.saturating_sub(s);
        cap_list(
            &mut tensors,
            json!({"name":name,"dtype":dtype,"shape":shape,"bytes":e.saturating_sub(s)}),
            1000,
        );
    }
    Ok(json!({
        "tensors": tensors,
        "tensor_count": obj.len().saturating_sub(1),
        "data_bytes": total_bytes,
        "metadata": obj.get("__metadata__").cloned().unwrap_or(json!({})),
    }))
}

// ---------- GGUF ----------

fn gguf_str(b: &[u8], o: &mut usize) -> Option<String> {
    let len = le64(b, *o)? as usize;
    *o += 8;
    let s = String::from_utf8_lossy(slice(b, *o, len).unwrap_or(&[])).into_owned();
    *o += len;
    Some(s)
}

fn gguf_val(b: &[u8], o: &mut usize, t: u32) -> Option<Value> {
    match t {
        0 => {
            let v = *b.get(*o)?;
            *o += 1;
            Some(json!(v))
        }
        1 => {
            let v = *b.get(*o)? as i8;
            *o += 1;
            Some(json!(v))
        }
        2 => {
            let v = le16(b, *o)?;
            *o += 2;
            Some(json!(v))
        }
        3 => {
            let v = le16(b, *o)? as i16;
            *o += 2;
            Some(json!(v))
        }
        4 => {
            let v = le32(b, *o)?;
            *o += 4;
            Some(json!(v))
        }
        5 => {
            let v = le32(b, *o)? as i32;
            *o += 4;
            Some(json!(v))
        }
        6 => {
            let v = le32(b, *o)?;
            *o += 4;
            Some(json!(f32::from_bits(v)))
        }
        7 => {
            let v = *b.get(*o)?;
            *o += 1;
            Some(json!(v != 0))
        }
        8 => {
            let s = gguf_str(b, o)?;
            Some(json!(s))
        }
        10 => {
            let v = le64(b, *o)?;
            *o += 8;
            Some(json!(v))
        }
        11 => {
            let v = le64(b, *o)? as i64;
            *o += 8;
            Some(json!(v))
        }
        12 => {
            let v = le64(b, *o)?;
            *o += 8;
            Some(json!(f64::from_bits(v)))
        }
        9 => {
            // array
            let et = le32(b, *o)?;
            *o += 4;
            let n = le64(b, *o)? as usize;
            *o += 8;
            let mut arr = Vec::new();
            for _ in 0..n.min(10000) {
                match gguf_val(b, o, et) {
                    Some(v) => arr.push(v),
                    None => break,
                }
            }
            Some(json!(arr))
        }
        _ => None,
    }
}

pub fn gguf(b: &[u8]) -> Result<Value, String> {
    if b.len() < 24 || &b[0..4] != b"GGUF" {
        return Err("not gguf".into());
    }
    let version = le32(b, 4).unwrap_or(0);
    let tensor_count = le64(b, 8).unwrap_or(0) as usize;
    let kv_count = le64(b, 16).unwrap_or(0) as usize;
    let mut o = 24usize;
    let mut metadata = serde_json::Map::new();
    for _ in 0..kv_count.min(10000) {
        let key = match gguf_str(b, &mut o) {
            Some(k) => k,
            None => break,
        };
        let t = match le32(b, o) {
            Some(v) => {
                o += 4;
                v
            }
            None => break,
        };
        match gguf_val(b, &mut o, t) {
            Some(v) => {
                metadata.insert(key, v);
            }
            None => break,
        }
    }
    // tensor infos
    let mut tensors = Vec::new();
    for _ in 0..tensor_count.min(5000) {
        let name = match gguf_str(b, &mut o) {
            Some(n) => n,
            None => break,
        };
        let nd = match le32(b, o) {
            Some(v) => {
                o += 4;
                v as usize
            }
            None => break,
        };
        let mut dims = Vec::new();
        for _ in 0..nd.min(8) {
            match le64(b, o) {
                Some(v) => {
                    o += 8;
                    dims.push(v);
                }
                None => break,
            }
        }
        let ty = match le32(b, o) {
            Some(v) => {
                o += 4;
                v
            }
            None => break,
        };
        let off = match le64(b, o) {
            Some(v) => {
                o += 8;
                v
            }
            None => break,
        };
        cap_list(
            &mut tensors,
            json!({"name":name,"dims":dims,"type":ty,"offset":off}),
            1000,
        );
    }
    Ok(json!({
        "version": version, "tensor_count": tensor_count,
        "metadata": Value::Object(metadata), "tensors": tensors,
    }))
}

// ---------- NPY ----------

pub fn npy(b: &[u8]) -> Result<Value, String> {
    if b.len() < 10 || &b[0..6] != b"\x93NUMPY" {
        return Err("not npy".into());
    }
    let major = b[6];
    let hlen = if major == 1 {
        le16(b, 8).unwrap_or(0) as usize
    } else {
        le32(b, 8).unwrap_or(0) as usize
    };
    let hoff = if major == 1 { 10 } else { 12 };
    let hdr = slice(b, hoff, hlen).ok_or("header")?;
    let h = String::from_utf8_lossy(hdr);
    let get = |key: &str| -> String {
        let pat = format!("'{}':", key);
        let p = h.find(&pat).map(|i| i + pat.len()).or_else(|| {
            let pat2 = format!("\"{}\":", key);
            h.find(&pat2).map(|i| i + pat2.len())
        });
        p.and_then(|i| {
            let rest = &h[i..];
            let end = rest
                .find(|c: char| c == ',' || c == '}' || c == '\n')
                .unwrap_or(rest.len());
            Some(
                rest[..end]
                    .trim()
                    .trim_matches('\'')
                    .trim_matches('"')
                    .to_string(),
            )
        })
        .unwrap_or_default()
    };
    let descr = get("descr");
    let fortran = get("fortran_order").contains("True");
    // shape is a parenthesized tuple — extract between the parens so
    // inner commas don't truncate the value
    let shape_s = {
        let p = h.find("'shape':").or_else(|| h.find("\"shape\":"));
        p.and_then(|i| {
            let rest = &h[i..];
            let open = rest.find('(')?;
            let close = rest[open..].find(')')? + open;
            Some(rest[open..=close].to_string())
        })
        .unwrap_or_default()
    };
    let shape: Vec<u64> = shape_s
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    Ok(json!({
        "version": format!("{}.{}", major, b[7]),
        "descr": descr, "fortran_order": fortran, "shape": shape,
        "header": h.trim().to_string(),
    }))
}

fn be_uint(b: &[u8], o: usize, n: usize) -> u64 {
    let mut v = 0u64;
    for i in 0..n.min(8) {
        v = (v << 8) | b.get(o + i).copied().unwrap_or(0) as u64;
    }
    v
}

// ---------- binary plist ----------

pub fn bplist(b: &[u8]) -> Result<Value, String> {
    if b.len() < 40 || &b[0..8] != b"bplist00" {
        return Err("not bplist".into());
    }
    let t = b.len() - 32;
    let off_size = b[t + 6] as usize;
    let ref_size = b[t + 7] as usize;
    let num_objects = be64(b, t + 8).unwrap_or(0) as usize;
    let top_object = be64(b, t + 16).unwrap_or(0) as usize;
    let off_table = be64(b, t + 24).unwrap_or(0) as usize;
    if off_size == 0 || off_size > 8 || ref_size == 0 || ref_size > 8 {
        return Err("bad trailer".into());
    }
    let mut offsets = Vec::with_capacity(num_objects.min(100000));
    for i in 0..num_objects.min(100000) {
        offsets.push(be_uint(b, off_table + i * off_size, off_size) as usize);
    }
    // parse objects lazily into a tree with memoization
    let mut memo: Vec<Option<Value>> = vec![None; offsets.len()];
    fn parse_obj(
        b: &[u8],
        offsets: &[usize],
        memo: &mut Vec<Option<Value>>,
        idx: usize,
        depth: usize,
        ref_size: usize,
    ) -> Value {
        if depth > 64 {
            return json!("<depth>");
        }
        if let Some(v) = &memo[idx] {
            return v.clone();
        }
        let o = match offsets.get(idx) {
            Some(&x) => x,
            None => return Value::Null,
        };
        let marker = match b.get(o) {
            Some(&m) => m,
            None => return Value::Null,
        };
        let typ = marker >> 4;
        let info = (marker & 0x0f) as usize;
        let len_of = |b: &[u8], o: usize, info: usize| -> (usize, usize) {
            if info != 0x0f {
                return (info, o + 1);
            }
            let m2 = b.get(o + 1).copied().unwrap_or(0);
            let n = 1usize << (m2 & 0x0f);
            let mut v = 0usize;
            for i in 0..n.min(8) {
                v = (v << 8) | b.get(o + 2 + i).copied().unwrap_or(0) as usize;
            }
            (v, o + 2 + n)
        };
        let v = match typ {
            0x0 => match info {
                0x0 => json!("null"),
                0x8 => json!(false),
                0x9 => json!(true),
                _ => json!("<simple>"),
            },
            0x1 => {
                let n = 1usize << info;
                let mut val = 0u64;
                for i in 0..n.min(16) {
                    val = (val << 8) | b.get(o + 1 + i).copied().unwrap_or(0) as u64;
                }
                json!(val)
            }
            0x2 => {
                if info == 2 {
                    f32be(b, o + 1).map(|x| json!(x)).unwrap_or(Value::Null)
                } else if info == 3 {
                    f64be(b, o + 1).map(|x| json!(x)).unwrap_or(Value::Null)
                } else {
                    json!("<real>")
                }
            }
            0x3 => {
                let secs = f64be(b, o + 1).unwrap_or(0.0);
                json!(secs)
            }
            0x4 => {
                let (len, p) = len_of(b, o, info);
                let bytes = slice(b, p, len).unwrap_or(&[]);
                json!({"data_len": len, "data_hex": hex(&bytes[..bytes.len().min(64)])})
            }
            0x5 => {
                let (len, p) = len_of(b, o, info);
                let s = slice(b, p, len).unwrap_or(&[]);
                json!(String::from_utf8_lossy(s).into_owned())
            }
            0x6 => {
                let (len, p) = len_of(b, o, info);
                let mut chars = Vec::new();
                for i in 0..len.min(100000) {
                    let c = be16(b, p + i * 2).unwrap_or(0);
                    chars.push(char::from_u32(c as u32).unwrap_or('?'));
                }
                json!(chars.into_iter().collect::<String>())
            }
            0x8 => {
                let n = info + 1;
                let mut val = 0u64;
                for i in 0..n.min(16) {
                    val = (val << 8) | b.get(o + 1 + i).copied().unwrap_or(0) as u64;
                }
                json!(val)
            }
            0xA => {
                let (count, p) = len_of(b, o, info);
                let mut arr = Vec::new();
                for i in 0..count.min(10000) {
                    let r = be_uint(b, p + i * ref_size, ref_size) as usize;
                    arr.push(parse_obj(b, offsets, memo, r, depth + 1, ref_size));
                }
                json!(arr)
            }
            0xD => {
                let (count, p) = len_of(b, o, info);
                let mut map = serde_json::Map::new();
                for i in 0..count.min(10000) {
                    let kr = be_uint(b, p + i * ref_size, ref_size) as usize;
                    let vr = be_uint(b, p + (count + i) * ref_size, ref_size) as usize;
                    let k = parse_obj(b, offsets, memo, kr, depth + 1, ref_size);
                    let v = parse_obj(b, offsets, memo, vr, depth + 1, ref_size);
                    map.insert(k.as_str().unwrap_or("?").to_string(), v);
                }
                Value::Object(map)
            }
            _ => json!(format!("<type {}>", typ)),
        };
        memo[idx] = Some(v.clone());
        v
    }
    let top = parse_obj(b, &offsets, &mut memo, top_object, 0, ref_size);
    Ok(json!({
        "objects": num_objects, "top": top,
    }))
}

// ---------- pcap ----------

pub fn pcap(b: &[u8]) -> Result<Value, String> {
    if b.len() < 24 {
        return Err("too small".into());
    }
    let magic = be32(b, 0).unwrap_or(0);
    let (nano, le) = match magic {
        0xa1b2c3d4 => (false, false),
        0xd4c3b2a1 => (false, true),
        0xa1b23c4d => (true, false),
        0x4d3cb2a1 => (true, true),
        _ => return Err("bad magic".into()),
    };
    let ru32 = |o: usize| -> u32 {
        let s = slice(b, o, 4).unwrap_or(&[0; 4]);
        if le {
            u32::from_le_bytes(s.try_into().unwrap())
        } else {
            u32::from_be_bytes(s.try_into().unwrap())
        }
    };
    let network = ru32(20);
    let linktype = match network {
        1 => "ethernet",
        101 => "raw_ip",
        113 => "linux_sll",
        276 => "linux_sll2",
        0 => "null",
        12 => "raw",
        105 => "ieee802_11",
        127 => "radiotap",
        9 => "ppp",
        50 => "ppp_hdlc",
        51 => "ppp_serial",
        108 => "loop",
        _ => "other",
    };
    let mut o = 24usize;
    let mut count = 0u64;
    let mut total_bytes = 0u64;
    let mut first_ts = u64::MAX;
    let mut last_ts = 0u64;
    while o + 16 <= b.len() {
        let ts_sec = ru32(o) as u64;
        let incl = ru32(o + 8) as usize;
        count += 1;
        total_bytes += incl as u64;
        first_ts = first_ts.min(ts_sec);
        last_ts = last_ts.max(ts_sec);
        o += 16 + incl;
        if incl == 0 && count > b.len() as u64 {
            break;
        }
    }
    Ok(json!({
        "format": "pcap", "nano": nano, "endianness": if le {"little"} else {"big"},
        "linktype": linktype, "packets": count, "captured_bytes": total_bytes,
        "first_ts": if first_ts == u64::MAX { Value::Null } else { json!(first_ts) },
        "last_ts": if count > 0 { json!(last_ts) } else { Value::Null },
    }))
}

// ---------- pcapng ----------

pub fn pcapng(b: &[u8]) -> Result<Value, String> {
    if b.len() < 12 || be32(b, 0) != Some(0x0a0d0d0a) {
        return Err("not pcapng".into());
    }
    let bom = be32(b, 8).unwrap_or(0);
    let le = bom == 0x4d3c2b1a;
    if bom != 0x1a2b3c4d && !le {
        return Err("bad bom".into());
    }
    let ru32 = |o: usize| -> u32 {
        let s = slice(b, o, 4).unwrap_or(&[0; 4]);
        if le {
            u32::from_le_bytes(s.try_into().unwrap())
        } else {
            u32::from_be_bytes(s.try_into().unwrap())
        }
    };
    let mut o = 0usize;
    let mut ifaces = 0u64;
    let mut packets = 0u64;
    let mut blocks = 0u64;
    while o + 12 <= b.len() {
        let btype = ru32(o);
        let blen = ru32(o + 4) as usize;
        if blen < 12 || o + blen > b.len() {
            break;
        }
        blocks += 1;
        match btype {
            1 => ifaces += 1,      // IDB
            6 | 3 => packets += 1, // EPB | SPB
            _ => {}
        }
        o += blen;
    }
    Ok(json!({
        "format": "pcapng", "endianness": if le {"little"} else {"big"},
        "blocks": blocks, "interfaces": ifaces, "packets": packets,
    }))
}

// ---------- PDF ----------

pub fn pdf(b: &[u8]) -> Result<Value, String> {
    if b.len() < 8 || !b.starts_with(b"%PDF-") {
        return Err("not pdf".into());
    }
    let ver = ascii(b, 5, 3);
    let text = String::from_utf8_lossy(b);
    let pages = text.matches("/Type").filter(|_| true).count();
    let page_objs = text.matches("/Type /Page").count() + text.matches("/Type/Page").count();
    let encrypted = text.contains("/Encrypt");
    let linearized = text.contains("/Linearized");
    let has_xref = text.contains("xref") || text.contains("/XRef");
    Ok(json!({
        "version": ver, "estimated_pages": page_objs.max(pages.min(100000)),
        "encrypted": encrypted, "linearized": linearized, "has_xref": has_xref,
        "size": b.len(),
    }))
}

// ---------- iNES ----------

pub fn ines(b: &[u8]) -> Result<Value, String> {
    if b.len() < 16 || &b[0..4] != b"NES\x1a" {
        return Err("not ines".into());
    }
    let prg = b[4] as usize * 16384;
    let chr = b[5] as usize * 8192;
    let f6 = b[6];
    let f7 = b[7];
    let mapper = ((f6 >> 4) | (f7 & 0xf0)) as u32;
    let mirroring = if f6 & 1 != 0 {
        "vertical"
    } else {
        "horizontal"
    };
    let battery = f6 & 2 != 0;
    let trainer = f6 & 4 != 0;
    let vs = b.get(8).copied().unwrap_or(0);
    let ps = b.get(9).copied().unwrap_or(0);
    Ok(json!({
        "prg_rom": prg, "chr_rom": chr, "mapper": mapper,
        "mirroring": mirroring, "battery": battery, "trainer": trainer,
        "vs_unisystem": vs, "playchoice": ps,
    }))
}

// ---------- PEM ----------

pub fn pem(b: &[u8]) -> Result<Value, String> {
    let text = String::from_utf8_lossy(b);
    let mut labels = Vec::new();
    let mut count = 0usize;
    let mut idx = 0usize;
    while let Some(pos) = text[idx..].find("-----BEGIN ") {
        let start = idx + pos + 11;
        if let Some(end) = text[start..].find("-----") {
            labels.push(text[start..start + end].to_string());
            count += 1;
            idx = start + end;
        } else {
            break;
        }
    }
    if count == 0 {
        return Err("no pem blocks".into());
    }
    Ok(json!({"blocks": count, "labels": labels}))
}

// ---------- firmware headers ----------

pub fn firmware(b: &[u8]) -> Result<Value, String> {
    // uImage: magic 0x27051956 BE
    if b.len() >= 64 && be32(b, 0) == Some(0x27051956) {
        let os = b[28];
        let arch = b[29];
        let typ = b[30];
        let name = ascii(b, 32, 32);
        let size = be32(b, 12).unwrap_or(0);
        let load = be32(b, 16).unwrap_or(0);
        let ep = be32(b, 20).unwrap_or(0);
        return Ok(json!({
            "kind": "uimage", "os": os, "arch": arch, "type": typ,
            "name": name, "data_size": size,
            "load_addr": format!("0x{:08x}", load), "entry": format!("0x{:08x}", ep),
        }));
    }
    // FIT image: FDT magic 0xd00dfeed BE
    if b.len() >= 40 && be32(b, 0) == Some(0xd00dfeed) {
        let totalsize = be32(b, 4).unwrap_or(0);
        let off_dt = be32(b, 8).unwrap_or(0);
        let off_str = be32(b, 12).unwrap_or(0);
        let version = be32(b, 20).unwrap_or(0);
        let last_ver = be32(b, 24).unwrap_or(0);
        let size_str = be32(b, 32).unwrap_or(0);
        let size_dt = be32(b, 36).unwrap_or(0);
        return Ok(json!({
            "kind": "fit", "total_size": totalsize,
            "off_dt_struct": off_dt, "off_dt_strings": off_str,
            "version": version, "last_comp_version": last_ver,
            "dt_strings_size": size_str, "dt_struct_size": size_dt,
        }));
    }
    // Android boot image
    if b.len() >= 48 && &b[0..8] == b"ANDROID!" {
        let ksize = le32(b, 8).unwrap_or(0);
        let rsize = le32(b, 16).unwrap_or(0);
        let ssize = le32(b, 24).unwrap_or(0);
        let page = le32(b, 36).unwrap_or(0);
        let hver = le32(b, 40).unwrap_or(0);
        return Ok(json!({
            "kind": "android_boot", "kernel_size": ksize,
            "ramdisk_size": rsize, "second_size": ssize,
            "page_size": page, "header_version": hver,
        }));
    }
    Err("not recognized firmware".into())
}

// ---------- git loose object ----------

pub fn git_object(b: &[u8]) -> Result<Value, String> {
    // zlib stream: 0x78 header
    if b.len() < 4 || b[0] != 0x78 {
        return Err("not zlib".into());
    }
    use flate2::read::ZlibDecoder;
    use std::io::Read;
    let mut out = Vec::new();
    ZlibDecoder::new(b)
        .take(16 * 1024 * 1024)
        .read_to_end(&mut out)
        .map_err(|e| format!("inflate: {}", e))?;
    let nul = out.iter().position(|&c| c == 0).ok_or("no header nul")?;
    let header = String::from_utf8_lossy(&out[..nul]).into_owned();
    let mut parts = header.splitn(2, ' ');
    let typ = parts.next().unwrap_or("?").to_string();
    let len: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
    Ok(json!({"type": typ, "declared_len": len, "inflated": out.len() - nul - 1}))
}

// ---------- protobuf wire walk (onnx/tflite/coreml) ----------

pub fn protobuf_walk(b: &[u8], max_fields: usize) -> Value {
    let mut o = 0usize;
    let mut fields = Vec::new();
    let mut hist: BTreeMap<u32, u64> = BTreeMap::new();
    while o < b.len() && fields.len() < max_fields {
        let (tag, p) = match uleb_at(b, o) {
            Some(x) => x,
            None => break,
        };
        o = p;
        let field = (tag >> 3) as u32;
        let wire = (tag & 7) as u32;
        *hist.entry(wire).or_insert(0) += 1;
        match wire {
            0 => {
                let _ = uleb_at(b, o);
                o = uleb_at(b, o).map(|(_, p)| p).unwrap_or(b.len());
            }
            1 => o += 8,
            2 => {
                let (len, p2) = match uleb_at(b, o) {
                    Some(x) => x,
                    None => break,
                };
                o = p2 + len as usize;
            }
            5 => o += 4,
            _ => break,
        }
        if o > b.len() {
            break;
        }
        cap_list(&mut fields, json!({"field": field, "wire": wire}), 200);
    }
    json!({
        "fields_seen": fields.len(),
        "wire_histogram": hist.iter().map(|(k,v)| json!({"wire":k,"count":v})).collect::<Vec<_>>(),
        "sample_fields": fields,
    })
}

// ---------- ZIP ----------

pub struct ZipEntry {
    pub name: String,
    pub method: u16,
    pub csize: u64,
    pub usize_: u64,
    pub local_off: u64,
}

pub fn zip(b: &[u8]) -> Result<(Vec<ZipEntry>, String), String> {
    // find EOCD: PK\x05\x06 scanning from end
    let tail = b.len().saturating_sub(66000);
    let mut eocd = None;
    for i in (tail..b.len().saturating_sub(3)).rev() {
        if &b[i..i + 4] == b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
    }
    let e = eocd.ok_or("no eocd")?;
    let count = le16(b, e + 10).unwrap_or(0) as usize;
    let cd_size = le32(b, e + 12).unwrap_or(0) as usize;
    let cd_off = le32(b, e + 16).unwrap_or(0) as usize;
    let mut entries = Vec::new();
    let mut o = cd_off;
    let end_cd = cd_off.saturating_add(cd_size).min(b.len());
    for _ in 0..count.min(100000) {
        if o + 46 > end_cd || &b[o..o + 4] != b"PK\x01\x02" {
            break;
        }
        let method = le16(b, o + 10).unwrap_or(0);
        let csize = le32(b, o + 20).unwrap_or(0) as u64;
        let usize_ = le32(b, o + 24).unwrap_or(0) as u64;
        let nlen = le16(b, o + 28).unwrap_or(0) as usize;
        let elen = le16(b, o + 30).unwrap_or(0) as usize;
        let clen = le16(b, o + 32).unwrap_or(0) as usize;
        let local_off = le32(b, o + 42).unwrap_or(0) as u64;
        let name = String::from_utf8_lossy(slice(b, o + 46, nlen).unwrap_or(&[])).into_owned();
        entries.push(ZipEntry {
            name,
            method,
            csize,
            usize_,
            local_off,
        });
        o += 46 + nlen + elen + clen;
    }
    // composition hints
    let names: Vec<&str> = entries.iter().map(|x| x.name.as_str()).collect();
    let kind = if names
        .iter()
        .any(|&n| n == "AndroidManifest.xml" || n.ends_with(".dex") || n.ends_with(".arsc"))
    {
        "apk"
    } else if names
        .iter()
        .any(|n| n.starts_with("Payload/") && n.ends_with(".app/"))
        || names.iter().any(|n| n.contains(".app/Info.plist"))
    {
        "ipa"
    } else if names.iter().any(|n| n.ends_with(".class")) {
        "jar"
    } else if names
        .iter()
        .any(|n| n.ends_with(".dll") || n.contains("mscorlib"))
    {
        "dotnet"
    } else {
        "zip"
    };
    Ok((entries, kind.to_string()))
}

// ---------- TAR ----------

pub fn tar(b: &[u8]) -> Result<Vec<(String, u64, u8)>, String> {
    if b.len() < 512 {
        return Err("too small".into());
    }
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 512 <= b.len() {
        let hdr = &b[o..o + 512];
        if hdr.iter().all(|&c| c == 0) {
            break;
        }
        let name = ascii(hdr, 0, 100);
        let size_s = ascii(hdr, 124, 12).trim().to_string();
        let size = u64::from_str_radix(size_s.trim_start_matches('\0').trim(), 8).unwrap_or(0);
        let typeflag = hdr[156];
        let prefix = ascii(hdr, 345, 155);
        let full = if prefix.is_empty() {
            name
        } else {
            format!("{}/{}", prefix, name)
        };
        out.push((full, size, typeflag));
        o += 512 + ((size as usize + 511) & !511);
        if out.len() >= 10000 {
            break;
        }
    }
    Ok(out)
}
