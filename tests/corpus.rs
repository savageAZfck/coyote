//! Corpus gate: every `.cy` file in `examples/` must parse and shape-
//! check. If this test breaks, the language broke a program it shipped.

use coyote::parser;
use coyote::typeck;
use std::path::Path;

#[test]
fn all_examples_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().is_some_and(|x| x == "cy")).then_some(p)
        })
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no .cy examples found in {}",
        dir.display()
    );
    for f in &files {
        let (_src, ast) =
            parser::parse_file(f).unwrap_or_else(|d| panic!("{}: parse failed — {d}", f.display()));
        typeck::check_all(&ast).unwrap_or_else(|d| panic!("{}: check failed — {d}", f.display()));
    }
}
