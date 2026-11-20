//! Universal inspection engine — the organism's sight.
//!
//! `inspect(path)` auto-detects the format and returns an evidence
//! record: sha256 identity, parsed findings, confidence, and honest
//! limitations. Every format listed here is parsed for real — magic
//! bytes only ever *route* to a parser, never substitute for one.
//!
//! Natives-facing API returns `(sha256, payload)` pairs so the VM can
//! witness the target's identity into the hash chain on every call.

use crate::inspect_fmt as fmt;
use crate::lower::sha256_hex;
use capstone::prelude::*;
use serde_json::{json, Value};
use std::path::Path;

const MAX_DEPTH: u32 = 4;
const MAX_MEMBER: usize = 64 * 1024 * 1024;
const MAX_MEMBERS: usize = 200;
const MAX_BUNDLE_FILES: usize = 400;
const MAX_FILE: u64 = 512 * 1024 * 1024;
const DYLD_CACHE: &str = "/System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e";

pub struct Out {
    pub sha: String,
    pub payload: String,
}
type R<T> = Result<T, String>;

fn le16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn le32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn be32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

// ───────────────────────────── detection ─────────────────────────────

/// Returns (format tag, confidence). Extension is only a weak hint for
/// formats with no reliable magic (protobuf-ish model files).
fn detect(b: &[u8], name: &str) -> (&'static str, &'static str) {
    if b.is_empty() {
        return ("empty", "high");
    }
    // gzip
    if b.len() >= 2 && b[0] == 0x1f && b[1] == 0x8b {
        return ("gzip", "high");
    }
    // zip (local header, empty archive EOCD, or spanning marker)
    if b.len() >= 4 && &b[0..2] == b"PK" && matches!(b[2], 0x03 | 0x05 | 0x07) {
        return ("zip", "high");
    }
    // Mach-O magics (thin LE/BE 32/64 + fat + fat64)
    match b.get(0..4) {
        Some([0xcf, 0xfa, 0xed, 0xfe])
        | Some([0xce, 0xfa, 0xed, 0xfe])
        | Some([0xfe, 0xed, 0xfa, 0xcf])
        | Some([0xfe, 0xed, 0xfa, 0xce])
        | Some([0xbe, 0xba, 0xfe, 0xca])
        | Some([0xbf, 0xba, 0xfe, 0xca]) => return ("mach-o", "high"),
        Some([0xca, 0xfe, 0xba, 0xbe]) => {
            // fat Mach-O vs Java class — disambiguate on the u32 at +4:
            // nfat_arch is small (<= ~20); class major version is 45..=70.
            let n = be32(b, 4).unwrap_or(0);
            if (40..=80).contains(&n) {
                return ("java-class", "high");
            }
            return ("mach-o", "high");
        }
        Some([0xca, 0xfe, 0xba, 0xbf]) => return ("mach-o", "high"),
        _ => {}
    }
    if b.len() >= 4 && &b[0..4] == b"\x7fELF" {
        return ("elf", "high");
    }
    if b.len() >= 0x40 && &b[0..2] == b"MZ" {
        let peoff = le32(b, 0x3c).unwrap_or(0) as usize;
        if b.get(peoff..peoff + 4) == Some(b"PE\0\0") {
            return ("pe", "high");
        }
    }
    if b.len() >= 8 && &b[0..4] == b"\0asm" {
        return ("wasm", "high");
    }
    if b.len() >= 8 && &b[0..4] == b"dex\n" {
        return ("dex", "high");
    }
    // Android binary XML: chunk type 0x0003, headerSize 8, then string pool
    if b.len() >= 16
        && le16(b, 0) == Some(0x0003)
        && le16(b, 2) == Some(8)
        && le16(b, 8) == Some(0x0001)
    {
        return ("axml", "high");
    }
    // safetensors: u64 header len then JSON object
    if b.len() >= 10 {
        let hl = u64::from_le_bytes(b[0..8].try_into().unwrap_or([0; 8])) as usize;
        if hl > 0 && hl < b.len() - 8 && b[8] == b'{' {
            if serde_json::from_slice::<Value>(&b[8..8 + hl]).is_ok() {
                return ("safetensors", "high");
            }
        }
    }
    if b.len() >= 4 && &b[0..4] == b"GGUF" {
        return ("gguf", "high");
    }
    if b.len() >= 6 && &b[0..6] == b"\x93NUMPY" {
        return ("npy", "high");
    }
    if b.len() >= 8 && &b[0..8] == b"bplist00" {
        return ("bplist", "high");
    }
    // binary plist XML / JSON text
    let head = &b[..b.len().min(4096)];
    let text_head = String::from_utf8_lossy(head);
    let trimmed = text_head.trim_start();
    if trimmed.starts_with("<?xml") && trimmed.contains("<plist") {
        return ("plist-xml", "high");
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if serde_json::from_slice::<Value>(b).is_ok() {
            return ("json", "high");
        }
    }
    // tar: "ustar" magic at offset 257
    if b.len() >= 262 && &b[257..262] == b"ustar" {
        return ("tar", "high");
    }
    if b.len() >= 16 && &b[0..15] == b"SQLite format 3" {
        return ("sqlite", "high");
    }
    // pcap / pcapng
    if b.len() >= 4 {
        let m = be32(b, 0).unwrap_or(0);
        if matches!(m, 0xa1b2c3d4 | 0xd4c3b2a1 | 0xa1b23c4d | 0x4d3cb2a1) {
            return ("pcap", "high");
        }
        if m == 0x0a0d0d0a {
            return ("pcapng", "high");
        }
    }
    if b.len() >= 5 && b.starts_with(b"%PDF-") {
        return ("pdf", "high");
    }
    if b.len() >= 4 && &b[0..4] == b"NES\x1a" {
        return ("ines", "high");
    }
    if b.len() >= 11 && b.starts_with(b"-----BEGIN ") {
        return ("pem", "high");
    }
    // firmware: uImage 0x27051956 BE, FIT 0xd00dfeed BE, bootimg "ANDROID!"
    if b.len() >= 4 {
        let m = be32(b, 0).unwrap_or(0);
        if m == 0x27051956 || m == 0xd00dfeed {
            return ("firmware", "high");
        }
    }
    if b.len() >= 8 && &b[0..8] == b"ANDROID!" {
        return ("firmware", "high");
    }
    // git loose object: zlib stream whose payload starts with a type tag
    if b.len() >= 4 && b[0] == 0x78 {
        if fmt::git_object(b).is_ok() {
            return ("git-object", "high");
        }
    }
    // extension hints — protobuf-ish model containers have no magic
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "onnx" | "tflite" | "mlmodel" | "pb" | "protobuf" => return ("protobuf", "low"),
        "mlpackage" | "app" | "framework" | "appex" | "xpc" => return ("bundle-dir", "low"),
        _ => {}
    }
    ("unknown", "none")
}

// ───────────────────────────── evidence ─────────────────────────────

fn evidence(
    path: &str,
    bytes: &[u8],
    format: &str,
    confidence: &str,
    findings: Value,
    limitations: Vec<&str>,
) -> Value {
    json!({
        "path": path,
        "sha256": sha256_hex(bytes),
        "size": bytes.len(),
        "format": format,
        "confidence": confidence,
        "findings": findings,
        "limitations": limitations,
    })
}

// ───────────────────────────── zip members ─────────────────────────────

fn zip_member(b: &[u8], e: &fmt::ZipEntry) -> Option<Vec<u8>> {
    let o = e.local_off as usize;
    if o + 30 > b.len() || &b[o..o + 4] != b"PK\x03\x04" {
        return None;
    }
    let method = le16(b, o + 8).unwrap_or(0);
    let nlen = le16(b, o + 26).unwrap_or(0) as usize;
    let elen = le16(b, o + 28).unwrap_or(0) as usize;
    let data = o.checked_add(30)?.checked_add(nlen)?.checked_add(elen)?;
    if e.csize == 0 || e.csize == u64::MAX || e.usize_ > MAX_MEMBER as u64 {
        return None;
    }
    let comp = b.get(data..data.checked_add(e.csize as usize)?)?;
    match method {
        0 => Some(comp.to_vec()),
        8 => {
            use flate2::read::DeflateDecoder;
            use std::io::Read;
            let mut out = Vec::new();
            DeflateDecoder::new(comp)
                .take(MAX_MEMBER as u64)
                .read_to_end(&mut out)
                .ok()?;
            Some(out)
        }
        _ => None,
    }
}

fn member_is_interesting(name: &str) -> bool {
    let n = name.to_lowercase();
    n.ends_with(".dex")
        || n.ends_with(".class")
        || n == "androidmanifest.xml"
        || n.ends_with(".so")
        || n.ends_with(".dylib")
        || n.ends_with(".macho")
        || n.ends_with(".plist")
        || n.ends_with(".onnx")
        || n.ends_with(".safetensors")
        || n.ends_with(".gguf")
        || n.ends_with(".bin")
        || n.contains("macos/")
        || n.contains(".framework/")
}

fn inspect_zip(label: &str, b: &[u8], depth: u32) -> R<Value> {
    let (entries, kind) = fmt::zip(b)?;
    let mut names: Vec<Value> = Vec::new();
    let mut nested = Vec::new();
    let mut total_usize = 0u64;
    for e in entries.iter().take(500) {
        total_usize += e.usize_;
        if names.len() < 200 {
            names.push(json!({"name": e.name, "method": e.method, "size": e.usize_}));
        }
    }
    if depth < MAX_DEPTH {
        for e in entries
            .iter()
            .filter(|e| member_is_interesting(&e.name))
            .take(MAX_MEMBERS)
        {
            if let Some(bytes) = zip_member(b, e) {
                let sub = format!("{}!{}", label, e.name);
                match inspect_bytes(&sub, &bytes, depth + 1) {
                    Ok(v) => {
                        if nested.len() < 60 {
                            nested.push(json!({"member": e.name, "evidence": v}));
                        }
                    }
                    Err(err) => {
                        if nested.len() < 60 {
                            nested.push(json!({"member": e.name, "error": err}));
                        }
                    }
                }
            }
        }
    }
    Ok(evidence(
        label,
        b,
        if kind == "zip" { "zip" } else { "zip" },
        "high",
        json!({
            "kind": kind,
            "entries": entries.len(),
            "total_uncompressed": total_usize,
            "names": names,
            "members": nested,
        }),
        vec!["zip64 archives and unsupported compression methods are skipped"],
    ))
}

// ───────────────────────────── gzip ─────────────────────────────

fn inspect_gzip(label: &str, b: &[u8], depth: u32) -> R<Value> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    let mut out = Vec::new();
    GzDecoder::new(b)
        .take(MAX_MEMBER as u64)
        .read_to_end(&mut out)
        .map_err(|e| format!("gzip decompress: {e}"))?;
    if depth >= MAX_DEPTH {
        return Ok(evidence(
            label,
            b,
            "gzip",
            "high",
            json!({"decompressed_size": out.len()}),
            vec!["recursion depth cap reached — inner payload not inspected"],
        ));
    }
    let inner = inspect_bytes(&format!("{}!inner", label), &out, depth + 1)?;
    Ok(evidence(
        label,
        b,
        "gzip",
        "high",
        json!({"decompressed_size": out.len(), "inner": inner}),
        vec![],
    ))
}

// ───────────────────────────── sqlite ─────────────────────────────

fn inspect_sqlite(label: &str, b: &[u8]) -> R<Value> {
    let path = label;
    let conn = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(|e| format!("sqlite open: {e}"))?;
    let mut st = conn
        .prepare("SELECT type, name, tbl_name FROM sqlite_master ORDER BY type, name")
        .map_err(|e| e.to_string())?;
    let rows = st
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut objects = Vec::new();
    let mut tables = Vec::new();
    for row in rows.flatten().take(2000) {
        let (typ, name, tbl) = row;
        if typ == "table" {
            tables.push((name.clone(), tbl.clone()));
        }
        if objects.len() < 500 {
            objects.push(json!({"type": typ, "name": name, "table": tbl}));
        }
    }
    let mut counts = Vec::new();
    for (name, _) in tables.iter().take(50) {
        let q = format!("SELECT count(*) FROM \"{}\"", name.replace('"', "\"\""));
        let n: i64 = conn.query_row(&q, [], |r| r.get(0)).unwrap_or(-1);
        counts.push(json!({"table": name, "rows": n}));
    }
    Ok(evidence(
        path,
        b,
        "sqlite",
        "high",
        json!({"objects": objects, "object_count": objects.len(), "table_rows": counts}),
        vec!["read-only — no queries executed against user data beyond row counts"],
    ))
}

// ───────────────────────────── directory bundles ─────────────────────────────

fn inspect_dir(path: &str, depth: u32) -> R<Value> {
    if depth > 3 {
        return Ok(json!({"kind": "dir", "note": "depth cap"}));
    }
    let rd = std::fs::read_dir(path).map_err(|e| format!("read_dir({path}): {e}"))?;
    let mut entries = Vec::new();
    let mut count = 0usize;
    for ent in rd.flatten() {
        if count >= MAX_BUNDLE_FILES {
            break;
        }
        let p = ent.path();
        let name = ent.file_name().to_string_lossy().into_owned();
        let meta = match ent.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            count += 1;
            entries.push(json!({"name": name, "kind": "dir"}));
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        count += 1;
        let size = meta.len();
        let fpath = p.to_string_lossy().into_owned();
        if size > MAX_FILE {
            entries.push(
                json!({"name": name, "kind": "file", "size": size, "note": "too large — skipped"}),
            );
            continue;
        }
        match std::fs::read(&p) {
            Ok(bytes) => {
                let (fmt_name, conf) = detect(&bytes, &name);
                let mut rec = json!({
                    "name": name, "kind": "file", "size": size,
                    "sha256": sha256_hex(&bytes), "format": fmt_name, "confidence": conf,
                });
                // full findings only for small interesting files
                if size <= 8 * 1024 * 1024 && fmt_name != "unknown" && fmt_name != "empty" {
                    if let Ok(ev) = inspect_bytes(&fpath, &bytes, depth + 1) {
                        rec["findings"] = ev["findings"].clone();
                    }
                }
                entries.push(rec);
            }
            Err(e) => entries.push(json!({"name": name, "kind": "file", "error": e.to_string()})),
        }
    }
    Ok(json!({
        "kind": "bundle",
        "path": path,
        "entries": entries,
        "entry_count": entries.len(),
    }))
}

// ───────────────────────────── dispatch ─────────────────────────────

fn inspect_bytes(label: &str, b: &[u8], depth: u32) -> R<Value> {
    let (fmt_name, _conf) = detect(b, label);
    let lim: Vec<&str> = vec![];
    match fmt_name {
        "empty" => Ok(evidence(label, b, "empty", "high", json!({}), lim)),
        "gzip" => inspect_gzip(label, b, depth),
        "zip" => inspect_zip(label, b, depth),
        "mach-o" => match haruspex::audit_bytes(b, label) {
            Ok(report) => {
                let v = serde_json::to_value(&report).map_err(|e| e.to_string())?;
                Ok(evidence(
                    label,
                    b,
                    "mach-o",
                    "high",
                    v,
                    vec!["disassembly depth limited to structure; use disasm_at for code"],
                ))
            }
            Err(e) => Ok(evidence(
                label,
                b,
                "mach-o",
                "low",
                json!({"error": e.to_string()}),
                vec!["haruspex parse failed — possible truncated or non-Mach-O CAFEBABE file"],
            )),
        },
        "elf" => match fmt::elf(b) {
            Ok(f) => Ok(evidence(label, b, "elf", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "elf", "low", json!({"error": e}), lim)),
        },
        "pe" => match fmt::pe_json(b) {
            Ok(f) => Ok(evidence(label, b, "pe", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "pe", "low", json!({"error": e}), lim)),
        },
        "wasm" => match fmt::wasm(b) {
            Ok(f) => Ok(evidence(label, b, "wasm", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "wasm", "low", json!({"error": e}), lim)),
        },
        "java-class" => match fmt::class_file(b) {
            Ok(f) => Ok(evidence(label, b, "java-class", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "java-class",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "dex" => match fmt::dex(b) {
            Ok(f) => Ok(evidence(label, b, "dex", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "dex", "low", json!({"error": e}), lim)),
        },
        "axml" => match fmt::axml(b) {
            Ok(f) => Ok(evidence(label, b, "axml", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "axml", "low", json!({"error": e}), lim)),
        },
        "safetensors" => match fmt::safetensors(b) {
            Ok(f) => Ok(evidence(label, b, "safetensors", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "safetensors",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "gguf" => match fmt::gguf(b) {
            Ok(f) => Ok(evidence(label, b, "gguf", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "gguf", "low", json!({"error": e}), lim)),
        },
        "npy" => match fmt::npy(b) {
            Ok(f) => Ok(evidence(label, b, "npy", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "npy", "low", json!({"error": e}), lim)),
        },
        "bplist" => match fmt::bplist(b) {
            Ok(f) => Ok(evidence(label, b, "bplist", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "bplist",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "plist-xml" => {
            let text = String::from_utf8_lossy(b);
            let keys: Vec<String> = text
                .match_indices("<key>")
                .filter_map(|(i, _)| {
                    let rest = &text[i + 5..];
                    let end = rest.find("</key>")?;
                    Some(rest[..end].to_string())
                })
                .take(200)
                .collect();
            Ok(evidence(
                label,
                b,
                "plist-xml",
                "high",
                json!({"keys": keys, "key_count": keys.len()}),
                lim,
            ))
        }
        "json" => {
            let v: Value = serde_json::from_slice(b).map_err(|e| format!("json parse: {e}"))?;
            let summary = match &v {
                Value::Object(m) => {
                    json!({"kind": "object", "keys": m.keys().take(100).collect::<Vec<_>>(), "key_count": m.len()})
                }
                Value::Array(a) => json!({"kind": "array", "len": a.len()}),
                _ => json!({"kind": "scalar"}),
            };
            Ok(evidence(label, b, "json", "high", summary, lim))
        }
        "tar" => match fmt::tar(b) {
            Ok(entries) => {
                let list: Vec<Value> = entries
                    .iter()
                    .take(300)
                    .map(|(n, s, t)| json!({"name": n, "size": s, "typeflag": t}))
                    .collect();
                Ok(evidence(
                    label,
                    b,
                    "tar",
                    "high",
                    json!({"entries": list, "count": entries.len()}),
                    lim,
                ))
            }
            Err(e) => Ok(evidence(label, b, "tar", "low", json!({"error": e}), lim)),
        },
        "sqlite" => {
            // sqlite needs a path, not bytes — handled in inspect_path
            Ok(evidence(
                label,
                b,
                "sqlite",
                "low",
                json!({"note": "use inspect on the file path for schema"}),
                lim,
            ))
        }
        "pcap" => match fmt::pcap(b) {
            Ok(f) => Ok(evidence(label, b, "pcap", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "pcap", "low", json!({"error": e}), lim)),
        },
        "pcapng" => match fmt::pcapng(b) {
            Ok(f) => Ok(evidence(label, b, "pcapng", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "pcapng",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "pdf" => match fmt::pdf(b) {
            Ok(f) => Ok(evidence(
                label,
                b,
                "pdf",
                "high",
                f,
                vec!["page count is an estimate from object markers"],
            )),
            Err(e) => Ok(evidence(label, b, "pdf", "low", json!({"error": e}), lim)),
        },
        "ines" => match fmt::ines(b) {
            Ok(f) => Ok(evidence(label, b, "ines", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "ines", "low", json!({"error": e}), lim)),
        },
        "pem" => match fmt::pem(b) {
            Ok(f) => Ok(evidence(label, b, "pem", "high", f, lim)),
            Err(e) => Ok(evidence(label, b, "pem", "low", json!({"error": e}), lim)),
        },
        "firmware" => match fmt::firmware(b) {
            Ok(f) => Ok(evidence(label, b, "firmware", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "firmware",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "git-object" => match fmt::git_object(b) {
            Ok(f) => Ok(evidence(label, b, "git-object", "high", f, lim)),
            Err(e) => Ok(evidence(
                label,
                b,
                "git-object",
                "low",
                json!({"error": e}),
                lim,
            )),
        },
        "protobuf" => Ok(evidence(
            label,
            b,
            "protobuf",
            "low",
            fmt::protobuf_walk(b, 400),
            vec!["wire-format walk only — no schema; field names unknown without .proto"],
        )),
        _ => Ok(evidence(
            label,
            b,
            "unknown",
            "none",
            json!({"note": "unrecognized format"}),
            vec!["no parser for this format — structure not analyzed"],
        )),
    }
}

// ───────────────────────────── public API ─────────────────────────────

/// Inspect a filesystem path — file or directory bundle.
pub fn inspect_path(path: &str) -> R<Out> {
    let p = Path::new(path);
    let meta = std::fs::metadata(p).map_err(|e| format!("inspect({path}): {e}"))?;
    if meta.is_dir() {
        let findings = inspect_dir(path, 0)?;
        let sha = sha256_hex(path.as_bytes());
        let payload = serde_json::to_string_pretty(&evidence(
            path,
            path.as_bytes(),
            "bundle",
            "high",
            findings,
            vec![
                "directory bundles enumerate members; each member carries its own format evidence",
            ],
        ))
        .map_err(|e| e.to_string())?;
        return Ok(Out { sha, payload });
    }
    if !meta.is_file() {
        return Err(format!("inspect({path}): not a regular file"));
    }
    if meta.len() > MAX_FILE {
        // too big to hold — detect from head bytes only
        use std::io::Read;
        let mut f = std::fs::File::open(p).map_err(|e| e.to_string())?;
        let mut head = vec![0u8; 4 * 1024 * 1024];
        let n = f.read(&mut head).map_err(|e| e.to_string())?;
        head.truncate(n);
        let (fmt_name, conf) = detect(&head, path);
        let payload = serde_json::to_string_pretty(&evidence(
            path,
            &head,
            fmt_name,
            conf,
            json!({"note": format!("file too large ({}) for full parse — head bytes only", meta.len())}),
            vec!["oversized file — only the first 4MiB was read"],
        ))
        .map_err(|e| e.to_string())?;
        return Ok(Out {
            sha: sha256_hex(&head),
            payload,
        });
    }
    let bytes = std::fs::read(p).map_err(|e| format!("inspect({path}): {e}"))?;
    // sqlite needs the path for rusqlite
    let ev = if detect(&bytes, path).0 == "sqlite" {
        inspect_sqlite(path, &bytes)?
    } else {
        inspect_bytes(path, &bytes, 0)?
    };
    let payload = serde_json::to_string_pretty(&ev).map_err(|e| e.to_string())?;
    Ok(Out {
        sha: sha256_hex(&bytes),
        payload,
    })
}

/// Inspect in-memory bytes — URL bodies, carved images, embedded payloads.
/// `label` becomes the evidence `path` field (URL, member name, ...).
pub fn inspect_data(label: &str, bytes: &[u8]) -> R<Out> {
    let ev = inspect_bytes(label, bytes, 0)?;
    let payload = serde_json::to_string_pretty(&ev).map_err(|e| e.to_string())?;
    Ok(Out {
        sha: sha256_hex(bytes),
        payload,
    })
}

/// haruspex Mach-O/dyld audit — deep slice.
pub fn dissect_path(path: &str) -> R<Out> {
    let report =
        haruspex::audit_file(Path::new(path)).map_err(|e| format!("dissect({path}): {e}"))?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let payload = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    Ok(Out {
        sha: sha256_hex(&bytes),
        payload,
    })
}

/// ASCII + UTF-16LE strings with file offsets.
pub fn mach_strings(path: &str, min: usize) -> R<(String, Vec<String>)> {
    let bytes = std::fs::read(path).map_err(|e| format!("mach_strings({path}): {e}"))?;
    let lines: Vec<String> = haruspex::strings(&bytes, min)
        .iter()
        .map(|s| {
            format!(
                "{:>8x} {}{}",
                s.offset,
                s.text,
                if s.wide { " (wide)" } else { "" }
            )
        })
        .collect();
    Ok((sha256_hex(&bytes), lines))
}

/// string → xref → disassembly context, ported from badapple-dissect.
pub fn hunt(path: &str, needle: &str) -> R<(String, Vec<String>)> {
    let bytes = std::fs::read(path).map_err(|e| format!("hunt({path}): {e}"))?;
    let m = haruspex::parse(&bytes).map_err(|e| format!("hunt({path}): {e}"))?;
    let slice = m.slices.first().ok_or("hunt: no slice")?;
    let idx = haruspex::code::index_code(&bytes, slice, |off, len| {
        let (lo, hi) = (off as usize, off as usize + len as usize);
        bytes
            .get(lo..hi)
            .map(|s| s.to_vec())
            .ok_or(haruspex::Error::Malformed("range outside file"))
    })
    .map_err(|e| format!("hunt: index_code: {e}"))?;
    let mut out = Vec::new();
    let mut found = false;
    for os in haruspex::strings(&bytes, 6) {
        if !os.text.contains(needle) {
            continue;
        }
        let Some(va) = slice.vmaddr_of_fileoff(os.offset) else {
            continue;
        };
        let sites: Vec<u64> = idx
            .xrefs
            .iter()
            .filter(|x| x.target == va)
            .map(|x| x.site)
            .collect();
        found = true;
        if sites.is_empty() {
            out.push(format!("{:?} — no code refs", os.text));
            continue;
        }
        out.push(format!("{:?} @ {:#x}", os.text, va));
        for site in sites.iter().take(4) {
            let Some(f) = idx.function_at(*site) else {
                continue;
            };
            out.push(format!(
                "  ref {site:#x} in {} ({:#x}..{:#x})",
                f.label(),
                f.addr,
                f.end
            ));
            let Some(t) = slice.text() else { continue };
            let lo = site.saturating_sub(96).max(f.addr);
            let hi = (site + 24).min(f.end);
            let a = t.fileoff as usize + (lo - t.addr) as usize;
            let bb = t.fileoff as usize + (hi - t.addr) as usize;
            if bb <= bytes.len() && a < bb {
                for line in haruspex::code::disasm_listing(&bytes[a..bb], lo) {
                    let mark = if line.starts_with(&format!("{site:x}:")) {
                        " <=="
                    } else {
                        ""
                    };
                    out.push(format!("      {line}{mark}"));
                }
            }
        }
    }
    if !found {
        out.push(format!("no strings matching {needle:?}"));
    }
    Ok((sha256_hex(&bytes), out))
}

fn dyld_cache_path() -> String {
    std::env::var("COYOTE_DYLD_CACHE").unwrap_or_else(|_| DYLD_CACHE.to_string())
}

/// Image paths inside the dyld shared cache, optionally grep-filtered.
pub fn dyld_list(grep: &str) -> R<Vec<String>> {
    let mut cache = haruspex::cache::SharedCache::open(Path::new(&dyld_cache_path()))
        .map_err(|e| format!("dyld_list: {e}"))?;
    cache
        .enumerate_images()
        .map_err(|e| format!("dyld_list: {e}"))?;
    let out: Vec<String> = cache
        .images
        .iter()
        .map(|i| i.path.clone())
        .filter(|p| grep.is_empty() || p.contains(grep))
        .collect();
    Ok(out)
}

/// Carve a cached image to disk — requires `fs` in addition to `dissect`.
pub fn dyld_carve(name: &str, out: &str) -> R<String> {
    let mut cache = haruspex::cache::SharedCache::open(Path::new(&dyld_cache_path()))
        .map_err(|e| format!("dyld_carve: {e}"))?;
    let (src, bytes) = cache
        .extract(name)
        .map_err(|e| format!("dyld_carve({name}): {e}"))?;
    std::fs::write(out, &bytes).map_err(|e| format!("dyld_carve write({out}): {e}"))?;
    Ok(format!("{src} -> {out} ({} bytes)", bytes.len()))
}

/// Evidence diff between two binaries — haruspex diff for Mach-O,
/// hash/size comparison otherwise.
pub fn bin_diff(old: &str, new: &str) -> R<String> {
    let a_bytes = std::fs::read(old).map_err(|e| format!("bin_diff({old}): {e}"))?;
    let b_bytes = std::fs::read(new).map_err(|e| format!("bin_diff({new}): {e}"))?;
    let a_sha = sha256_hex(&a_bytes);
    let b_sha = sha256_hex(&b_bytes);
    let payload = if detect(&a_bytes, old).0 == "mach-o" && detect(&b_bytes, new).0 == "mach-o" {
        match (
            haruspex::audit_bytes(&a_bytes, old),
            haruspex::audit_bytes(&b_bytes, new),
        ) {
            (Ok(ra), Ok(rb)) => serde_json::to_string_pretty(&haruspex::diff(&ra, &rb))
                .map_err(|e| e.to_string())?,
            _ => json!({
                "old_sha256": a_sha, "new_sha256": b_sha,
                "old_size": a_bytes.len(), "new_size": b_bytes.len(),
                "note": "mach-o parse failed — hash/size comparison only",
            })
            .to_string(),
        }
    } else {
        json!({
            "old_sha256": a_sha, "new_sha256": b_sha,
            "old_size": a_bytes.len(), "new_size": b_bytes.len(),
            "old_format": detect(&a_bytes, old).0,
            "new_format": detect(&b_bytes, new).0,
            "note": "non-Mach-O — structural diff is hash/size only",
        })
        .to_string()
    };
    Ok(payload)
}

// ───────────────────────────── disassembly ─────────────────────────────

fn arch_for_macho(arch: &str) -> Option<&'static str> {
    if arch.contains("arm64") {
        Some("arm64")
    } else if arch.contains("x86_64") {
        Some("x86_64")
    } else if arch.contains("arm64_32") {
        Some("arm64")
    } else {
        None
    }
}

fn macho_vaddr_to_off(slice: &haruspex::Slice, vaddr: u64) -> Option<usize> {
    for s in &slice.sections {
        if vaddr >= s.addr && vaddr < s.addr + s.size {
            return Some((s.fileoff + (vaddr - s.addr)) as usize);
        }
    }
    None
}

/// Linear disassembly at a virtual address — Mach-O/ELF/PE.
/// Not function-aware; resolves vaddr→file offset via section tables.
pub fn disasm_at(path: &str, addr_str: &str) -> R<Vec<String>> {
    let addr = {
        let s = addr_str.strip_prefix("0x").unwrap_or(addr_str);
        u64::from_str_radix(s, 16)
            .or_else(|_| s.parse::<u64>())
            .map_err(|_| format!("disasm_at: bad address {addr_str:?}"))?
    };
    let bytes = std::fs::read(path).map_err(|e| format!("disasm_at({path}): {e}"))?;
    let (fmt_name, _) = detect(&bytes, path);
    let (arch, off) = match fmt_name {
        "mach-o" => {
            let m = haruspex::parse(&bytes).map_err(|e| format!("disasm_at: {e}"))?;
            let slice = m.slices.first().ok_or("disasm_at: no slice")?;
            let arch = arch_for_macho(&slice.arch)
                .ok_or_else(|| format!("disasm_at: unsupported arch {}", slice.arch))?;
            let off = macho_vaddr_to_off(slice, addr)
                .ok_or_else(|| format!("disasm_at: vaddr {addr:#x} not in any section"))?;
            (arch, off)
        }
        "elf" => {
            let f = fmt::elf(&bytes).map_err(|e| format!("disasm_at: {e}"))?;
            let arch = match f["machine"].as_str().unwrap_or("") {
                "x86_64" => "x86_64",
                "aarch64" => "arm64",
                "arm" => "arm",
                "ppc" | "ppc64" => {
                    return Err("disasm_at: ppc not supported by capstone build".into())
                }
                other => return Err(format!("disasm_at: unsupported elf machine {other}")),
            };
            let off = fmt::elf_vaddr_to_off(&bytes, addr)
                .ok_or_else(|| format!("disasm_at: vaddr {addr:#x} not in any PT_LOAD"))?;
            (arch, off)
        }
        "pe" => {
            let p = fmt::pe(&bytes).map_err(|e| format!("disasm_at: {e}"))?;
            let arch = match p.machine {
                0x8664 => "x86_64",
                0x14c => "x86",
                0xaa64 => "arm64",
                m => return Err(format!("disasm_at: unsupported pe machine {m:#x}")),
            };
            let off = fmt::pe_vaddr_to_off(&bytes, addr)
                .ok_or_else(|| format!("disasm_at: vaddr {addr:#x} not in any section"))?;
            (arch, off)
        }
        other => return Err(format!("disasm_at: unsupported format {other}")),
    };
    if off >= bytes.len() {
        return Err("disasm_at: resolved offset past EOF".into());
    }
    let code = &bytes[off..];
    let cs = match arch {
        "arm64" => capstone::Capstone::new()
            .arm64()
            .mode(capstone::arch::arm64::ArchMode::Arm)
            .endian(capstone::Endian::Little)
            .build()
            .map_err(|e| format!("capstone arm64: {e}"))?,
        "x86_64" => capstone::Capstone::new()
            .x86()
            .mode(capstone::arch::x86::ArchMode::Mode64)
            .syntax(capstone::arch::x86::ArchSyntax::Intel)
            .build()
            .map_err(|e| format!("capstone x86_64: {e}"))?,
        "x86" => capstone::Capstone::new()
            .x86()
            .mode(capstone::arch::x86::ArchMode::Mode32)
            .syntax(capstone::arch::x86::ArchSyntax::Intel)
            .build()
            .map_err(|e| format!("capstone x86: {e}"))?,
        "arm" => capstone::Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .endian(capstone::Endian::Little)
            .build()
            .map_err(|e| format!("capstone arm: {e}"))?,
        _ => return Err(format!("disasm_at: no capstone arch for {arch}")),
    };
    let insns = cs
        .disasm_all(code, addr)
        .map_err(|e| format!("capstone disasm: {e}"))?;
    let mut out = Vec::new();
    for i in insns.iter().take(400) {
        out.push(format!(
            "{:x}: {} {}",
            i.address(),
            i.mnemonic().unwrap_or(""),
            i.op_str().unwrap_or("")
        ));
    }
    if out.is_empty() {
        return Err("disasm_at: no instructions decoded".into());
    }
    Ok(out)
}
