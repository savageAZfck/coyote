//! Inspection engine: real-parser fixtures per format, compile-time
//! `cap dissect` enforcement, witness chaining, and VM integration.

use coyote::inspect;
use coyote::lower::sha256_hex;
use coyote::parser;
use coyote::typeck;
use coyote::vm::{Val, Vm};
use std::io::Write;

fn tmpdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("cyinspect{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixture(name: &str, bytes: &[u8]) -> String {
    let p = tmpdir().join(name);
    std::fs::write(&p, bytes).unwrap();
    p.to_string_lossy().into_owned()
}

fn ev(path: &str) -> serde_json::Value {
    let out = inspect::inspect_path(path).unwrap();
    serde_json::from_str(&out.payload).unwrap()
}

// ---------------- fixtures ----------------

fn safetensors_fixture() -> Vec<u8> {
    let hdr =
        br#"{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"__metadata__":{"k":"v"}}"#;
    let mut b = Vec::new();
    b.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    b.extend_from_slice(hdr);
    b.extend_from_slice(&[0u8; 24]);
    b
}

fn gguf_fixture() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&1u64.to_le_bytes()); // kv_count
                                              // kv: key "general.name", type 8 (string), value "test"
    b.extend_from_slice(&12u64.to_le_bytes());
    b.extend_from_slice(b"general.name");
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&4u64.to_le_bytes());
    b.extend_from_slice(b"test");
    // tensor info: name "w", n_dims 2, dims [2,3], type 0 (f32), offset 0
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(b"w");
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&2u64.to_le_bytes());
    b.extend_from_slice(&3u64.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b
}

fn npy_fixture() -> Vec<u8> {
    let hdr = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(1);
    b.push(0);
    b.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
    b.extend_from_slice(hdr);
    b.extend_from_slice(&[0u8; 24]);
    b
}

fn zip_fixture() -> Vec<u8> {
    // one stored (method 0) member "classes.dex" + EOCD → apk composition
    let name = b"classes.dex";
    let data = b"dex\n035\x00";
    let mut b = Vec::new();
    // local file header
    b.extend_from_slice(b"PK\x03\x04");
    b.extend_from_slice(&20u16.to_le_bytes()); // version
    b.extend_from_slice(&0u16.to_le_bytes()); // flags
    b.extend_from_slice(&0u16.to_le_bytes()); // method = stored
    b.extend_from_slice(&0u16.to_le_bytes()); // time
    b.extend_from_slice(&0u16.to_le_bytes()); // date
    b.extend_from_slice(&0u32.to_le_bytes()); // crc (not verified)
    b.extend_from_slice(&(data.len() as u32).to_le_bytes()); // csize
    b.extend_from_slice(&(data.len() as u32).to_le_bytes()); // usize
    b.extend_from_slice(&(name.len() as u16).to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes()); // extra len
    b.extend_from_slice(name);
    b.extend_from_slice(data);
    let cd_off = b.len() as u32;
    // central directory
    b.extend_from_slice(b"PK\x01\x02");
    b.extend_from_slice(&20u16.to_le_bytes());
    b.extend_from_slice(&20u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&(data.len() as u32).to_le_bytes());
    b.extend_from_slice(&(data.len() as u32).to_le_bytes());
    b.extend_from_slice(&(name.len() as u16).to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes()); // extra
    b.extend_from_slice(&0u16.to_le_bytes()); // comment
    b.extend_from_slice(&0u16.to_le_bytes()); // disk
    b.extend_from_slice(&0u16.to_le_bytes()); // int attr
    b.extend_from_slice(&0u32.to_le_bytes()); // ext attr
    b.extend_from_slice(&0u32.to_le_bytes()); // local off
    b.extend_from_slice(name);
    let cd_size = b.len() as u32 - cd_off;
    // EOCD
    b.extend_from_slice(b"PK\x05\x06");
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&cd_size.to_le_bytes());
    b.extend_from_slice(&cd_off.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b
}

fn tar_fixture() -> Vec<u8> {
    let mut b = vec![0u8; 2048];
    // header 0: "hello.txt", size 5, typeflag '0'
    b[0..9].copy_from_slice(b"hello.txt");
    b[124..136].copy_from_slice(b"00000000005\0");
    b[156] = b'0';
    b[257..262].copy_from_slice(b"ustar");
    // data block at 512
    b[512..517].copy_from_slice(b"hello");
    // header 1 at 1024
    let h2 = 1024usize;
    b[h2..h2 + 4].copy_from_slice(b"dir/");
    b[h2 + 156] = b'5';
    b[h2 + 257..h2 + 262].copy_from_slice(b"ustar");
    b
}

fn gzip_fixture(inner: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    let mut e = GzEncoder::new(Vec::new(), Compression::fast());
    e.write_all(inner).unwrap();
    e.finish().unwrap()
}

fn wasm_fixture() -> Vec<u8> {
    let mut b = vec![0u8, b'a', b's', b'm', 1, 0, 0, 0];
    // type section: 1 type () -> ()
    b.extend_from_slice(&[1, 4, 1, 0x60, 0, 0]);
    // function section: 1 func of type 0
    b.extend_from_slice(&[3, 2, 1, 0]);
    // export section: "foo" func 0
    b.extend_from_slice(&[7, 7, 1, 3, b'f', b'o', b'o', 0, 0]);
    // code section: 1 body, size 2, 0 locals, end
    b.extend_from_slice(&[10, 4, 1, 2, 0, 0x0b]);
    b
}

fn class_fixture() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&[0xca, 0xfe, 0xba, 0xbe]);
    b.extend_from_slice(&0u16.to_be_bytes()); // minor
    b.extend_from_slice(&52u16.to_be_bytes()); // major (Java 8)
    b.extend_from_slice(&1u16.to_be_bytes()); // cp_count = 1 (empty pool)
    b.extend_from_slice(&0x0021u16.to_be_bytes()); // access: public super
    b.extend_from_slice(&0u16.to_be_bytes()); // this_class
    b.extend_from_slice(&0u16.to_be_bytes()); // super_class
    b.extend_from_slice(&0u16.to_be_bytes()); // interfaces_count
    b.extend_from_slice(&0u16.to_be_bytes()); // fields_count
    b.extend_from_slice(&0u16.to_be_bytes()); // methods_count
    b.extend_from_slice(&0u16.to_be_bytes()); // attributes_count
    b
}

fn pcap_fixture() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0xa1b2c3d4u32.to_be_bytes()); // magic BE
    b.extend_from_slice(&2u16.to_be_bytes()); // v2.4
    b.extend_from_slice(&4u16.to_be_bytes());
    b.extend_from_slice(&0i32.to_be_bytes()); // tz
    b.extend_from_slice(&0u32.to_be_bytes()); // sigfigs
    b.extend_from_slice(&65535u32.to_be_bytes()); // snaplen
    b.extend_from_slice(&1u32.to_be_bytes()); // ethernet
                                              // one packet: ts, ts_usec, incl, orig + 4 bytes data
    b.extend_from_slice(&1000u32.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes());
    b.extend_from_slice(&4u32.to_be_bytes());
    b.extend_from_slice(&4u32.to_be_bytes());
    b.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    b
}

fn plist_xml_fixture() -> Vec<u8> {
    br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.example.app</string>
    <key>CFBundleVersion</key>
    <string>42</string>
</dict>
</plist>"#
    .to_vec()
}

fn bplist_fixture() -> Vec<u8> {
    // one ASCII-string object "hello", top_object = 0
    let mut b = Vec::new();
    b.extend_from_slice(b"bplist00");
    // object 0 at offset 8: marker 0x55 (ascii string len 5) + "hello"
    b.push(0x55);
    b.extend_from_slice(b"hello");
    let off_table = b.len(); // 14
    b.push(8u8); // offset of object 0
                 // trailer: 32 bytes
    b.extend_from_slice(&[0u8; 6]);
    b.push(1); // offset_int_size
    b.push(1); // object_ref_size
    b.extend_from_slice(&1u64.to_be_bytes()); // num_objects
    b.extend_from_slice(&0u64.to_be_bytes()); // top_object
    b.extend_from_slice(&(off_table as u64).to_be_bytes());
    b
}

fn git_object_fixture() -> Vec<u8> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
    e.write_all(b"blob 5\0hello").unwrap();
    e.finish().unwrap()
}

fn sqlite_fixture() -> String {
    let p = tmpdir().join("test.db");
    let _ = std::fs::remove_file(&p);
    let conn = rusqlite::Connection::open(&p).unwrap();
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", [])
        .unwrap();
    conn.execute("INSERT INTO t (name) VALUES ('a'), ('b'), ('c')", [])
        .unwrap();
    conn.execute("CREATE VIEW v AS SELECT name FROM t", [])
        .unwrap();
    drop(conn);
    p.to_string_lossy().into_owned()
}

fn elf_fixture() -> Vec<u8> {
    // ELF64 LE exec, x86_64, 3 sections: NULL, .text, .shstrtab
    let mut b = vec![0u8; 288];
    b[0..4].copy_from_slice(b"\x7fELF");
    b[4] = 2; // 64-bit
    b[5] = 1; // LE
    b[6] = 1; // version
    b[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    b[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    b[20..24].copy_from_slice(&1u32.to_le_bytes());
    b[24..32].copy_from_slice(&0x401000u64.to_le_bytes()); // entry
    b[40..48].copy_from_slice(&64u64.to_le_bytes()); // shoff
    b[52..54].copy_from_slice(&64u16.to_le_bytes()); // ehsize
    b[54..56].copy_from_slice(&56u16.to_le_bytes()); // phentsize
    b[58..60].copy_from_slice(&64u16.to_le_bytes()); // shentsize
    b[60..62].copy_from_slice(&3u16.to_le_bytes()); // shnum
    b[62..64].copy_from_slice(&2u16.to_le_bytes()); // shstrndx
                                                    // shdr[1] .text @64+64=128
    let s = 128;
    b[s..s + 4].copy_from_slice(&1u32.to_le_bytes()); // name off -> ".text"
    b[s + 4..s + 8].copy_from_slice(&1u32.to_le_bytes()); // SHT_PROGBITS
    b[s + 8..s + 16].copy_from_slice(&6u64.to_le_bytes()); // SHF_ALLOC|EXECINSTR
    b[s + 16..s + 24].copy_from_slice(&0x401000u64.to_le_bytes()); // addr
    b[s + 24..s + 32].copy_from_slice(&0x200u64.to_le_bytes()); // offset
    b[s + 32..s + 40].copy_from_slice(&0x10u64.to_le_bytes()); // size
                                                               // shdr[2] .shstrtab @192
    let s = 192;
    b[s..s + 4].copy_from_slice(&7u32.to_le_bytes()); // name off -> ".shstrtab"
    b[s + 4..s + 8].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
    b[s + 24..s + 32].copy_from_slice(&256u64.to_le_bytes()); // offset
    b[s + 32..s + 40].copy_from_slice(&17u64.to_le_bytes()); // size
                                                             // shstrtab @256: "\0.text\0.shstrtab\0" (17 bytes)
    b[256..273].copy_from_slice(b"\0.text\0.shstrtab\0");
    b
}

fn pe_fixture() -> Vec<u8> {
    // minimal PE32+ with one .text section — enough for the parser
    let mut b = vec![0u8; 0x400];
    b[0..2].copy_from_slice(b"MZ");
    b[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes()); // e_lfanew
    let pe = 0x80;
    b[pe..pe + 4].copy_from_slice(b"PE\0\0");
    let coff = pe + 4;
    b[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes()); // machine x64
    b[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // nsects
    b[coff + 16..coff + 18].copy_from_slice(&0xf0u16.to_le_bytes()); // opt size
    let opt = coff + 20;
    b[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes()); // PE32+
    b[opt + 16..opt + 20].copy_from_slice(&0x1000u32.to_le_bytes()); // entry rva
    b[opt + 24..opt + 32].copy_from_slice(&0x140000000u64.to_le_bytes()); // image base
    b[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // num dirs
                                                                   // section header at opt + 0xf0
    let s = opt + 0xf0;
    b[s..s + 5].copy_from_slice(b".text");
    b[s + 8..s + 12].copy_from_slice(&0x100u32.to_le_bytes()); // vsize
    b[s + 12..s + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // vaddr
    b[s + 16..s + 20].copy_from_slice(&0x200u32.to_le_bytes()); // rawsize
    b[s + 20..s + 24].copy_from_slice(&0x200u32.to_le_bytes()); // rawptr
    b
}

fn dex_fixture() -> Vec<u8> {
    let mut b = vec![0u8; 0x70];
    b[0..8].copy_from_slice(b"dex\n035\0");
    b[32..36].copy_from_slice(&0x70u32.to_le_bytes()); // file_size
    b[36..40].copy_from_slice(&0x70u32.to_le_bytes()); // header_size
    b[40..44].copy_from_slice(&0x12345678u32.to_le_bytes()); // endian tag
                                                             // string_ids_size=0 @56, all counts zero — parser returns header facts
    b
}

fn axml_fixture() -> Vec<u8> {
    // root RES_XML_TYPE chunk (type 0x0003, headerSize 8) containing a
    // UTF-16 string pool + one START_ELEMENT — mirrors real AndroidManifest.xml
    let mut pool_data = Vec::new();
    let mut offsets = Vec::new();
    for s in ["manifest", "com.example.app"] {
        offsets.push(pool_data.len() as u32);
        pool_data.extend_from_slice(&(s.chars().count() as u16).to_le_bytes());
        for c in s.chars() {
            pool_data.extend_from_slice(&(c as u32 as u16).to_le_bytes());
        }
        pool_data.extend_from_slice(&0u16.to_le_bytes());
    }
    let pool_size = 28 + offsets.len() * 4 + pool_data.len();
    let elem_size = 36usize; // node(16) + attrExt(20), 0 attributes
    let total = 8 + pool_size + elem_size;

    let mut b = Vec::new();
    // root chunk
    b.extend_from_slice(&0x0003u16.to_le_bytes());
    b.extend_from_slice(&8u16.to_le_bytes());
    b.extend_from_slice(&(total as u32).to_le_bytes());
    // string pool chunk
    b.extend_from_slice(&0x0001u16.to_le_bytes());
    b.extend_from_slice(&28u16.to_le_bytes());
    b.extend_from_slice(&(pool_size as u32).to_le_bytes());
    b.extend_from_slice(&2u32.to_le_bytes()); // stringCount
    b.extend_from_slice(&0u32.to_le_bytes()); // styleCount
    b.extend_from_slice(&0u32.to_le_bytes()); // flags: UTF-16
    b.extend_from_slice(&((28 + offsets.len() * 4) as u32).to_le_bytes()); // stringsStart
    b.extend_from_slice(&0u32.to_le_bytes()); // stylesStart
    for o in &offsets {
        b.extend_from_slice(&o.to_le_bytes());
    }
    b.extend_from_slice(&pool_data);
    // START_ELEMENT chunk: name = string idx 0 ("manifest")
    b.extend_from_slice(&0x0102u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(&(elem_size as u32).to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes()); // lineNumber
    b.extend_from_slice(&0u32.to_le_bytes()); // comment
    b.extend_from_slice(&0xffff_ffffu32.to_le_bytes()); // ns = none
    b.extend_from_slice(&0u32.to_le_bytes()); // name -> strings[0]
    b.extend_from_slice(&20u16.to_le_bytes()); // attributeStart
    b.extend_from_slice(&20u16.to_le_bytes()); // attributeSize
    b.extend_from_slice(&0u16.to_le_bytes()); // attributeCount
    b.extend_from_slice(&0xffffu16.to_le_bytes()); // idIndex
    b.extend_from_slice(&0xffffu16.to_le_bytes()); // classIndex
    b.extend_from_slice(&0xffffu16.to_le_bytes()); // styleIndex
    b
}

fn firmware_uimage_fixture() -> Vec<u8> {
    let mut b = vec![0u8; 64];
    b[0..4].copy_from_slice(&0x27051956u32.to_be_bytes());
    b[12..16].copy_from_slice(&0x1000u32.to_be_bytes()); // data size
    b[16..20].copy_from_slice(&0x80000000u32.to_be_bytes()); // load addr
    b[20..24].copy_from_slice(&0x80000000u32.to_be_bytes()); // entry
    b[28] = 5; // os = linux
    b[29] = 2; // arch = arm
    b[30] = 2; // type = kernel
    b[32..36].copy_from_slice(b"test");
    b
}

fn ines_fixture() -> Vec<u8> {
    let mut b = vec![0u8; 16];
    b[0..4].copy_from_slice(b"NES\x1a");
    b[4] = 2; // prg banks
    b[5] = 1; // chr banks
    b[6] = 0x02; // battery + mirroring bit0=0 → horizontal
    b
}

fn pem_fixture() -> Vec<u8> {
    b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n-----BEGIN PUBLIC KEY-----\nMFkw\n-----END PUBLIC KEY-----\n".to_vec()
}

fn pcapng_fixture() -> Vec<u8> {
    let mut b = Vec::new();
    // SHB: type, len, bom, major, minor, section_len(u64), trailing len = 28 total
    b.extend_from_slice(&0x0a0d0d0au32.to_le_bytes());
    b.extend_from_slice(&28u32.to_le_bytes());
    b.extend_from_slice(&0x1a2b3c4du32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&28u64.to_le_bytes()); // section len
    b.extend_from_slice(&28u32.to_le_bytes());
    // IDB: type 1, len 20, linktype 1, reserved, snaplen
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&20u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&65535u32.to_le_bytes());
    b.extend_from_slice(&20u32.to_le_bytes());
    // EPB: type 6, len 36, iface 0, tsh, tsl, caplen, origlen, data(4)
    b.extend_from_slice(&6u32.to_le_bytes());
    b.extend_from_slice(&36u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&1000u32.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&[1, 2, 3, 4]);
    b.extend_from_slice(&36u32.to_le_bytes());
    b
}

fn json_fixture() -> Vec<u8> {
    br#"{"name":"coyote","version":1,"tags":["a","b"]}"#.to_vec()
}

fn pdf_fixture() -> Vec<u8> {
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R>>endobj\ntrailer<</Size 4>>\n%%EOF".to_vec()
}

fn onnx_fixture() -> Vec<u8> {
    // minimal protobuf: field 1 varint (ir_version=7), field 2 string "test"
    let mut b = vec![0x08, 0x07]; // field1 wire0 = 7
    b.extend_from_slice(&[0x12, 4]); // field2 wire2 len4
    b.extend_from_slice(b"test");
    b
}

// ---------------- parser tests ----------------

#[test]
fn safetensors_parses() {
    let p = fixture("t.safetensors", &safetensors_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "safetensors");
    assert_eq!(e["findings"]["tensor_count"], 1);
    assert_eq!(e["findings"]["tensors"][0]["name"], "w");
    assert_eq!(e["findings"]["tensors"][0]["dtype"], "F32");
    assert_eq!(e["findings"]["data_bytes"], 24);
    assert!(e["sha256"].as_str().unwrap().len() == 64);
}

#[test]
fn gguf_parses() {
    let p = fixture("m.gguf", &gguf_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "gguf");
    assert_eq!(e["findings"]["tensor_count"], 1);
    assert_eq!(e["findings"]["metadata"]["general.name"], "test");
    assert_eq!(e["findings"]["tensors"][0]["name"], "w");
    assert_eq!(
        e["findings"]["tensors"][0]["dims"],
        serde_json::json!([2, 3])
    );
}

#[test]
fn npy_parses() {
    let p = fixture("a.npy", &npy_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "npy");
    assert_eq!(e["findings"]["descr"], "<f4");
    assert_eq!(e["findings"]["shape"], serde_json::json!([2, 3]));
    assert_eq!(e["findings"]["fortran_order"], false);
}

#[test]
fn zip_composition_and_member_recursion() {
    let p = fixture("app.apk", &zip_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "zip");
    assert_eq!(e["findings"]["kind"], "apk");
    assert_eq!(e["findings"]["entries"], 1);
    // classes.dex member should be recursively inspected → dex format
    let members = e["findings"]["members"].as_array().unwrap();
    assert!(!members.is_empty(), "expected member evidence");
    let dex_ev = &members[0]["evidence"];
    assert_eq!(dex_ev["format"], "dex");
}

#[test]
fn tar_parses() {
    let p = fixture("a.tar", &tar_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "tar");
    let entries = e["findings"]["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["name"], "hello.txt");
    assert_eq!(entries[0]["size"], 5);
    assert_eq!(entries[1]["name"], "dir/");
}

#[test]
fn gzip_recurses() {
    let inner = safetensors_fixture();
    let p = fixture("a.bin.gz", &gzip_fixture(&inner));
    let e = ev(&p);
    assert_eq!(e["format"], "gzip");
    assert_eq!(e["findings"]["inner"]["format"], "safetensors");
    assert_eq!(e["findings"]["inner"]["findings"]["tensor_count"], 1);
}

#[test]
fn wasm_parses() {
    let p = fixture("m.wasm", &wasm_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "wasm");
    assert_eq!(e["findings"]["function_count"], 1);
    assert_eq!(e["findings"]["export_count"], 1);
    assert!(e["findings"]["exports"][0]
        .as_str()
        .unwrap()
        .starts_with("foo("));
}

#[test]
fn java_class_parses() {
    let p = fixture("C.class", &class_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "java-class");
    assert_eq!(e["findings"]["major"], 52);
    assert_eq!(e["findings"]["methods"], 0);
}

#[test]
fn pcap_parses() {
    let p = fixture("cap.pcap", &pcap_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "pcap");
    assert_eq!(e["findings"]["linktype"], "ethernet");
    assert_eq!(e["findings"]["packets"], 1);
}

#[test]
fn pcapng_parses() {
    let p = fixture("cap.pcapng", &pcapng_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "pcapng");
    assert_eq!(e["findings"]["interfaces"], 1);
    assert_eq!(e["findings"]["packets"], 1);
}

#[test]
fn plist_xml_keys() {
    let p = fixture("Info.plist", &plist_xml_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "plist-xml");
    let keys: Vec<&str> = e["findings"]["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(keys.contains(&"CFBundleIdentifier"));
    assert!(keys.contains(&"CFBundleVersion"));
}

#[test]
fn bplist_parses() {
    let p = fixture("b.bin", &bplist_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "bplist");
    assert_eq!(e["findings"]["objects"], 1);
    assert_eq!(e["findings"]["top"], "hello");
}

#[test]
fn git_object_parses() {
    let p = fixture("obj", &git_object_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "git-object");
    assert_eq!(e["findings"]["type"], "blob");
    assert_eq!(e["findings"]["declared_len"], 5);
}

#[test]
fn sqlite_schema() {
    let p = sqlite_fixture();
    let e = ev(&p);
    assert_eq!(e["format"], "sqlite");
    let objs = e["findings"]["objects"].as_array().unwrap();
    assert!(objs
        .iter()
        .any(|o| o["name"] == "t" && o["type"] == "table"));
    assert!(objs.iter().any(|o| o["name"] == "v" && o["type"] == "view"));
    let rows = e["findings"]["table_rows"].as_array().unwrap();
    assert!(rows.iter().any(|r| r["table"] == "t" && r["rows"] == 3));
}

#[test]
fn pe_parses() {
    let p = fixture("m.exe", &pe_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "pe");
    assert_eq!(e["findings"]["machine"], "x64");
    assert_eq!(e["findings"]["is64"], true);
    let secs = e["findings"]["sections"].as_array().unwrap();
    assert_eq!(secs[0]["name"], ".text");
    assert_eq!(secs[0]["vaddr"], "0x1000");
}

#[test]
fn dex_parses() {
    let p = fixture("c.dex", &dex_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "dex");
    assert_eq!(e["findings"]["version"], "035");
    assert_eq!(e["findings"]["endian_tag"], "0x12345678");
}

#[test]
fn axml_parses_utf16_pool_and_elements() {
    let p = fixture("AndroidManifest.xml", &axml_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "axml");
    let strs = e["findings"]["strings"].as_array().unwrap();
    assert!(strs.iter().any(|s| s == "manifest"));
    assert!(strs.iter().any(|s| s == "com.example.app"));
    let elems = e["findings"]["elements"].as_array().unwrap();
    assert_eq!(elems[0], "manifest");
    assert!(e["findings"]["chunk_types"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c == 258)); // 0x0102 START_ELEMENT
}

#[test]
fn firmware_uimage() {
    let p = fixture("u.img", &firmware_uimage_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "firmware");
    assert_eq!(e["findings"]["kind"], "uimage");
    assert_eq!(e["findings"]["data_size"], 0x1000);
    assert_eq!(e["findings"]["name"], "test");
}

#[test]
fn ines_parses() {
    let p = fixture("g.nes", &ines_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "ines");
    assert_eq!(e["findings"]["prg_rom"], 2 * 16384);
    assert_eq!(e["findings"]["chr_rom"], 1 * 8192);
    assert_eq!(e["findings"]["battery"], true);
}

#[test]
fn pem_blocks() {
    let p = fixture("c.pem", &pem_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "pem");
    assert_eq!(e["findings"]["blocks"], 2);
    assert!(e["findings"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l == "CERTIFICATE"));
}

#[test]
fn pdf_facts() {
    let p = fixture("d.pdf", &pdf_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "pdf");
    assert_eq!(e["findings"]["version"], "1.4");
}

#[test]
fn protobuf_walk() {
    let p = fixture("m.onnx", &onnx_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "protobuf");
    assert_eq!(e["findings"]["fields_seen"], 2);
}

#[test]
fn json_summary() {
    let p = fixture("c.json", &json_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "json");
    assert_eq!(e["findings"]["kind"], "object");
    assert_eq!(e["findings"]["key_count"], 3);
}

#[test]
fn unknown_format_honest() {
    let p = fixture(
        "rand.bin",
        &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77],
    );
    let e = ev(&p);
    assert_eq!(e["format"], "unknown");
    assert_eq!(e["confidence"], "none");
    assert!(e["limitations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l.as_str().unwrap().contains("no parser")));
}

// ---------------- real binaries ----------------

#[test]
fn macho_audit_real_binary() {
    // /bin/ls exists on every unix — exercise the haruspex deep path
    let p = "/bin/ls";
    if !std::path::Path::new(p).exists() {
        return;
    }
    let e = ev(p);
    assert_eq!(e["format"], "mach-o");
    assert_eq!(e["confidence"], "high");
    // haruspex Report embeds sha256 + slices
    let f = &e["findings"];
    assert!(f["sha256"].as_str().unwrap().len() == 64);
    assert!(f["slices"].as_array().unwrap().len() >= 1);
}

#[test]
fn elf_parses() {
    let p = fixture("e.elf", &elf_fixture());
    let e = ev(&p);
    assert_eq!(e["format"], "elf");
    assert_eq!(e["findings"]["machine"], "x86_64");
    assert_eq!(e["findings"]["type"], "exec");
    let secs = e["findings"]["sections"].as_array().unwrap();
    assert!(secs.iter().any(|s| s["name"] == ".text"));
    assert!(secs.iter().any(|s| s["name"] == ".shstrtab"));
}

#[test]
fn elf_real_binary() {
    // /usr/bin/env or /bin/ls on macOS is Mach-O, not ELF — find a real ELF if present
    for p in ["/usr/bin/env", "/bin/true"] {
        if !std::path::Path::new(p).exists() {
            continue;
        }
        let bytes = std::fs::read(p).unwrap();
        if &bytes[0..4] == b"\x7fELF" {
            let path = fixture("real.elf", &bytes);
            let e = ev(&path);
            assert_eq!(e["format"], "elf");
            assert!(e["findings"]["sections"].as_array().unwrap().len() > 0);
            return;
        }
    }
    // no ELF on this host — synthetic coverage is in unit tests
}

#[test]
fn dissect_native_real() {
    let p = "/bin/ls";
    if !std::path::Path::new(p).exists() {
        return;
    }
    let out = inspect::dissect_path(p).unwrap();
    let r: serde_json::Value = serde_json::from_str(&out.payload).unwrap();
    assert!(r["sha256"].as_str().unwrap().len() == 64);
    assert!(r["slices"].as_array().is_some());
}

#[test]
fn disasm_at_real_binary() {
    let p = "/bin/ls";
    if !std::path::Path::new(p).exists() {
        return;
    }
    // entry offset from haruspex audit — use first section addr
    let bytes = std::fs::read(p).unwrap();
    let m = haruspex::parse(&bytes).unwrap();
    let slice = &m.slices[0];
    let text = slice.text().expect("__text");
    let lines = inspect::disasm_at(p, &format!("{:#x}", text.addr)).unwrap();
    assert!(!lines.is_empty());
    assert!(lines[0].contains(':'));
}

// ---------------- compile-time capability enforcement ----------------

fn check(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|d| d.to_string())?;
    typeck::check_all(&ast)
        .map(|_| ())
        .map_err(|d| d.to_string())
}

fn rejects(src: &str) -> String {
    match check(src) {
        Err(e) => e,
        Ok(()) => panic!("expected rejection:\n{src}"),
    }
}

#[test]
fn inspect_requires_dissect_cap() {
    let e = rejects("host fn f(p: str) -> str { return inspect(p); }\n");
    assert!(e.contains("dissect"), "diag: {e}");
}

#[test]
fn inspect_with_cap_ok() {
    check("cap dissect\nhost fn f(p: str) -> str { return inspect(p); }\n").unwrap();
}

#[test]
fn all_inspect_natives_gated() {
    for (name, args) in [
        ("dissect", "\"/bin/ls\""),
        ("mach_strings", "\"/bin/ls\", 8"),
        ("hunt", "\"/bin/ls\", \"http\""),
        ("dyld_list", "\"Foundation\""),
        ("dyld_carve", "\"Foundation\", \"/tmp/x\""),
        ("bin_diff", "\"/bin/ls\", \"/bin/ls\""),
        ("disasm_at", "\"/bin/ls\", \"0x1000\""),
    ] {
        let src = format!("host fn f() -> str {{ return str({name}({args})); }}\n");
        let e = rejects(&src);
        assert!(e.contains("capability"), "{name} diag: {e}");
    }
}

#[test]
fn dyld_carve_needs_fs_too() {
    // dissect alone is not enough — carve writes to disk
    let e = rejects("cap dissect\nhost fn f() -> str { return dyld_carve(\"x\", \"/tmp/y\"); }\n");
    assert!(e.contains("fs"), "diag: {e}");
    check("cap dissect, fs\nhost fn f() -> str { return dyld_carve(\"x\", \"/tmp/y\"); }\n")
        .unwrap();
}

#[test]
fn fn_level_cap_bound_enforced() {
    // program declares dissect, but fn bound excludes it
    let e = rejects("cap dissect\nhost fn f(p: str) -> str cap net { return inspect(p); }\n");
    assert!(e.contains("cap"), "diag: {e}");
}

// ---------------- VM integration ----------------

fn vm_of(src: &str) -> Vm {
    let ast = parser::parse(src).expect("parse");
    let checked = typeck::check_all(&ast).expect("check");
    let host = checked.host.expect("host stratum");
    Vm::new(&host, Vec::new(), &sha256_hex(src.as_bytes())).expect("vm")
}

#[test]
fn vm_inspect_returns_evidence() {
    let p = fixture("vmtest.safetensors", &safetensors_fixture());
    let mut vm = vm_of(&format!(
        "cap dissect\nhost fn f() -> str {{ return inspect(\"{p}\"); }}\n"
    ));
    let v = vm.call_fn("f", vec![]).unwrap();
    let Val::S(s) = v else { panic!("expected str") };
    let j: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(j["format"], "safetensors");
    assert_eq!(j["findings"]["tensor_count"], 1);
    // the act of inspection is witnessed
    let wit = vm.witness_log();
    assert!(
        wit.iter().any(|w| w.tag == "inspect"),
        "witness log: {:?}",
        wit.iter().map(|w| &w.tag).collect::<Vec<_>>()
    );
}

#[test]
fn vm_dyld_list_returns_vec() {
    // dyld_list on a nonexistent cache errors — but the type path works;
    // test with empty grep against real cache if present, else skip
    let cache = "/System/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e";
    if !std::path::Path::new(cache).exists() {
        return;
    }
    let mut vm =
        vm_of("cap dissect\nhost fn f() -> i64 { return len(dyld_list(\"Foundation\")); }\n");
    match vm.call_fn("f", vec![]).unwrap() {
        Val::I(n) => assert!(n > 0),
        _ => panic!("expected i64"),
    }
}

#[test]
fn vm_bin_diff_json() {
    let p = fixture("d1.bin", &[1, 2, 3, 4]);
    let q = fixture("d2.bin", &[1, 2, 3, 5]);
    let mut vm = vm_of(&format!(
        "cap dissect\nhost fn f() -> str {{ return bin_diff(\"{p}\", \"{q}\"); }}\n"
    ));
    let Val::S(s) = vm.call_fn("f", vec![]).unwrap() else {
        panic!("str")
    };
    let j: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_ne!(j["old_sha256"], j["new_sha256"]);
    assert_eq!(j["old_size"], 4);
}

// ---------------- fuzz smoke ----------------

#[test]
fn fuzz_truncated_fixtures_never_panic() {
    let fixtures: Vec<Vec<u8>> = vec![
        safetensors_fixture(),
        gguf_fixture(),
        npy_fixture(),
        zip_fixture(),
        tar_fixture(),
        elf_fixture(),
        pe_fixture(),
        wasm_fixture(),
        class_fixture(),
        dex_fixture(),
        axml_fixture(),
        bplist_fixture(),
        pcap_fixture(),
        pcapng_fixture(),
        pdf_fixture(),
        ines_fixture(),
        pem_fixture(),
        firmware_uimage_fixture(),
        onnx_fixture(),
        json_fixture(),
    ];
    // every prefix at stride + every byte-length around chunk boundaries
    for (i, bytes) in fixtures.iter().enumerate() {
        for n in (0..=bytes.len()).step_by(7).chain([bytes.len()]) {
            let p = fixture(&format!("fz{i}_{n}.bin"), &bytes[..n]);
            let _ = inspect::inspect_path(&p); // must not panic
        }
    }
}

#[test]
fn fuzz_magic_plus_random_tail() {
    // deterministic xorshift — no rand dep
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let magics: &[&[u8]] = &[
        b"\x7fELF",
        b"MZ",
        b"\0asm",
        b"dex\n035\0",
        b"GGUF",
        b"\x93NUMPY",
        b"bplist00",
        b"SQLite format 3\0",
        b"%PDF-1.7",
        b"NES\x1a",
        b"-----BEGIN ",
        b"ANDROID!",
    ];
    for (i, magic) in magics.iter().enumerate() {
        for len in [16usize, 64, 256, 1024] {
            let mut b = magic.to_vec();
            b.extend((0..len).map(|_| (next() & 0xff) as u8));
            let p = fixture(&format!("fm{i}_{len}.bin"), &b);
            let _ = inspect::inspect_path(&p);
        }
    }
    // CAFEBABE ambiguity zone — class vs fat macho boundary values
    for n in [0u32, 20, 39, 40, 45, 52, 60, 70, 80, 81, u32::MAX] {
        let mut b = vec![
            0xcafebabeu32.to_be_bytes().to_vec(),
            n.to_be_bytes().to_vec(),
        ]
        .concat();
        b.extend([0u8; 64]);
        let p = fixture(&format!("cafe_{n}.bin"), &b);
        let _ = inspect::inspect_path(&p);
    }
}

#[test]
fn fuzz_gzip_nesting_hits_depth_cap() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    // gzip^8 of a mach-o-ish payload — must terminate at MAX_DEPTH
    let mut payload = vec![0xcf, 0xfa, 0xed, 0xfe];
    payload.extend([0u8; 64]);
    for _ in 0..8 {
        let mut e = GzEncoder::new(Vec::new(), Compression::fast());
        e.write_all(&payload).unwrap();
        payload = e.finish().unwrap();
    }
    let p = fixture("deep.gz", &payload);
    let e = ev(&p);
    assert_eq!(e["format"], "gzip");
    // walk the inner chain — must bottom out at the depth cap, not hang
    let mut cur = e["findings"].clone();
    for _ in 0..10 {
        match cur["inner"]["findings"].clone() {
            serde_json::Value::Object(_) => cur = cur["inner"]["findings"].clone(),
            _ => break,
        }
    }
}

#[test]
fn fuzz_disasm_at_bad_inputs() {
    let p = fixture("d.bin", &[0xde, 0xad, 0xbe, 0xef]);
    // garbage path / bad address / non-macho — all must return Err, not panic
    assert!(inspect::disasm_at("/nonexistent/path", "0x0").is_err());
    assert!(inspect::disasm_at(&p, "not-an-address").is_err());
    assert!(inspect::disasm_at(&p, "0x0").is_err());
    // hunt on tiny file — no strings, no crash
    let _ = inspect::hunt(&p, "needle");
}
