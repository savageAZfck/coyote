//! Host-stratum VM eval: compile a host-only source, drive it through
//! `Vm::call_fn`, assert on `Val`s.

use coyote::lower::sha256_hex;
use coyote::parser;
use coyote::typeck;
use coyote::vm::{Val, Vm};

fn vm_of(src: &str) -> Vm {
    let ast = parser::parse(src).expect("parse");
    let checked = typeck::check_all(&ast).expect("check");
    let host = checked.host.expect("host stratum");
    Vm::new(&host, Vec::new(), &sha256_hex(src.as_bytes())).expect("vm")
}

fn call(vm: &mut Vm, name: &str, args: Vec<Val>) -> Val {
    vm.call_fn(name, args).unwrap()
}

#[test]
fn fib_recursion() {
    let mut vm = vm_of(
        "host fn fib(n: i64) -> i64 {
             if n < 2 { return n; }
             return fib(n - 1) + fib(n - 2);
         }
         host fn main() -> i64 { return 0; }\n",
    );
    match call(&mut vm, "fib", vec![Val::I(10)]) {
        Val::I(v) => assert_eq!(v, 55),
        v => panic!("fib(10) = <non-expected variant>"),
    }
}

#[test]
fn str_builtin_any() {
    let mut vm = vm_of("host fn s() -> str { return str(42); }\n");
    match call(&mut vm, "s", vec![]) {
        Val::S(v) => assert_eq!(&*v, "42"),
        v => panic!("str(42) = <non-expected variant>"),
    }
}

#[test]
fn early_return_skips_dead_code() {
    let mut vm = vm_of("host fn f() -> i64 { return 7; let never = 1 / 0; }\n");
    match call(&mut vm, "f", vec![]) {
        Val::I(v) => assert_eq!(v, 7),
        v => panic!("f() = <non-expected variant>"),
    }
}

#[test]
fn structs_and_fields() {
    let mut vm = vm_of(
        "struct P { x: i64, y: i64 }
         host fn mag(p: P) -> i64 { return p.x * p.x + p.y * p.y; }
         host fn mk() -> i64 {
             let p = P { x: 3, y: 4 };
             return mag(p);
         }\n",
    );
    match call(&mut vm, "mk", vec![]) {
        Val::I(v) => assert_eq!(v, 25),
        v => panic!("mk() = <non-expected variant>"),
    }
}

#[test]
fn vectors_push_len() {
    let mut vm = vm_of(
        "host fn v() -> i64 {
             let xs = [1, 2];
             push(xs, 3);
             return len(xs) + xs[0];
         }\n",
    );
    match call(&mut vm, "v", vec![]) {
        Val::I(v) => assert_eq!(v, 4),
        v => panic!("v() = <non-expected variant>"),
    }
}

#[test]
fn match_enum_dispatch() {
    let mut vm = vm_of(
        "host fn classify(r: Reflex) -> i64 {
             match r {
                 Reflex::Pass => { return 0; }
                 Reflex::Alert(m) => { return 1; }
                 _ => { return 2; }
             }
         }
         host fn go() -> i64 { return classify(Reflex::Alert(\"warn\")); }\n",
    );
    match call(&mut vm, "go", vec![]) {
        Val::I(v) => assert_eq!(v, 1),
        v => panic!("go() = <non-expected variant>"),
    }
}

#[test]
fn run_main_entry() {
    let mut vm = vm_of("host fn main() -> i64 { return 42; }\n");
    match vm.run_main() {
        Ok(Val::I(v)) => assert_eq!(v, 42),
        v => panic!("main = <non-expected variant>"),
    }
}

#[test]
fn stringify_round_trip() {
    let mut vm = vm_of("host fn main() -> i64 { return 0; }\n");
    assert_eq!(
        vm.stringify(&Val::V(std::rc::Rc::new(std::cell::RefCell::new(vec![
            Val::I(1),
            Val::S(std::rc::Rc::from("two")),
        ])))),
        "[1, two]"
    );
}
