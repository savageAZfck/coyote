//! Host-stratum soundness: the checker must reject bodies that can't
//! produce the declared return type, cap violations, and sealed access.

use coyote::parser;
use coyote::typeck;

fn check(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|d| d.to_string())?;
    typeck::check_all(&ast)
        .map(|_| ())
        .map_err(|d| d.to_string())
}

fn rejects(src: &str) -> String {
    match check(src) {
        Err(e) => e,
        Ok(()) => panic!("expected this source to be rejected:\n{src}"),
    }
}

// ---- declared return vs body ----

#[test]
fn tail_expr_is_not_a_return() {
    // the footgun that shipped: `-> i64` with a bare trailing expr
    let e = rejects("host fn f(n: i64) -> i64 { n * 2 }\n");
    assert!(e.contains("without returning"), "diag was: {e}");
}

#[test]
fn explicit_return_passes() {
    check("host fn f(n: i64) -> i64 { return n * 2; }\n").unwrap();
}

#[test]
fn if_without_else_falls_through() {
    let e = rejects("host fn f(n: i64) -> i64 { if n > 0 { return n; } }\n");
    assert!(e.contains("without returning"), "diag was: {e}");
}

#[test]
fn if_else_both_return() {
    check("host fn f(n: i64) -> i64 { if n > 0 { return n; } else { return 0; } }\n").unwrap();
}

#[test]
fn early_return_then_dead_code_ok() {
    check("host fn f(n: i64) -> i64 { return n; let x = 99; }\n").unwrap();
}

#[test]
fn match_all_arms_return() {
    check(
        "enum E { A, B }
         host fn f(e: E) -> i64 {
             match e { E::A => { return 1; } E::B => { return 2; } }
         }\n",
    )
    .unwrap();
}

#[test]
fn return_type_mismatch() {
    let e = rejects("host fn f() -> i64 { return \"hi\"; }\n");
    assert!(e.contains("return wants"), "diag was: {e}");
}

#[test]
fn value_return_from_unit_fn() {
    let e = rejects("host fn f() { return 1; }\n");
    assert!(e.contains("unit fn"), "diag was: {e}");
}

#[test]
fn unit_fn_without_return_ok() {
    check("host fn f() { let x = 1; }\n").unwrap();
}

// ---- stratum discipline ----

#[test]
fn tensor_fn_tail_expr_returns() {
    // the tensor stratum *does* tail-return — only host fns are explicit
    check(
        "param p: f16[64, 64]
         io x: in f16[64]
         io y: out f16[64]
         fn f(x: f16[64]) -> f16[64] { relu(x) }
         fn main(x: f16[64]) -> f16[64] { f(x @ p) }\n",
    )
    .unwrap();
}

#[test]
fn sealed_param_unnameable() {
    let e =
        rejects("param secret: sealed f16[64]\nhost fn leak() -> i64 { return len(secret); }\n");
    assert!(e.contains("sealed param"), "diag was: {e}");
}

// ---- capabilities / reflexes ----

#[test]
fn reflex_must_return_reflex() {
    let e = rejects("reflex r() -> i64 { return 1; }\n");
    assert!(e.contains("Reflex"), "diag was: {e}");
}

#[test]
fn reflex_with_reflex_return_ok() {
    check("reflex r() -> Reflex { return Reflex::Pass; }\n").unwrap();
}
