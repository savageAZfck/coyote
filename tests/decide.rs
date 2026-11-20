//! `decide` primitive: forward semantics + reverse-mode gradient vs
//! finite differences on the CPU evaluator.

use coyote::eval::Evaluator;
use coyote::parser;
use coyote::typeck;
use std::collections::HashMap;

const FWD: &str = r#"
param wq: trainable f16[2, 3]
param wk: trainable f16[2, 3]
io hq: in f16[2]
io hos: in f16[2, 2]
fn main(hq: f16[2], hos: f16[2,2]) -> f16[2] {
    decide(hq, hos, wq, wk)
}
"#;

const LOSS: &str = r#"
param wq: trainable f16[2, 3]
param wk: trainable f16[2, 3]
io hq: in f16[2]
io hos: in f16[2, 2]
io tgt: in f16[2]
fn main(hq: f16[2], hos: f16[2,2], tgt: f16[2]) -> f32 {
    mse(decide(hq, hos, wq, wk), tgt)
}
"#;

const GRAD: &str = r#"
param wq: trainable f16[2, 3]
param wk: trainable f16[2, 3]
io hq: in f16[2]
io hos: in f16[2, 2]
io tgt: in f16[2]
fn main(hq: f16[2], hos: f16[2,2], tgt: f16[2]) -> f16[6] {
    grad(mse(decide(hq, hos, wq, wk), tgt), wq)
}
"#;

fn inputs() -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    m.insert("hq".into(), vec![0.3, -0.7]);
    m.insert("hos".into(), vec![0.1, 0.2, 0.5, -0.4]);
    m.insert("tgt".into(), vec![0.8, 0.2]);
    m.insert("target".into(), vec![0.8, 0.2]);
    m
}

fn params() -> HashMap<String, Vec<f32>> {
    let mut m = HashMap::new();
    // storage [P,D] = [3,2]: flat[p*2+d] = logical wq[d,p]
    m.insert("wq".into(), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    m.insert("wk".into(), vec![-0.2, 0.1, 0.4, -0.3, 0.6, 0.2]);
    m
}

fn run(src: &str, inputs: &HashMap<String, Vec<f32>>) -> (Vec<f32>, HashMap<String, Vec<f32>>) {
    let ast = parser::parse(src).expect("parse");
    let checked = typeck::check_all(&ast).expect("check");
    let tp = &checked.ane[0].1;
    let mut ev = Evaluator::new(tp);
    ev.set_params(params());
    let out = ev.run(inputs).expect("eval");
    // drain param updates back into the param map for reuse
    let mut p = params();
    for (name, vals) in ev.param_updates() {
        p.insert(name.clone(), vals.clone());
    }
    (out, p)
}

#[test]
fn decide_forward_is_a_distribution() {
    let (out, _) = run(FWD, &inputs());
    assert_eq!(out.len(), 2);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "probs sum to {sum}");
    assert!(out.iter().all(|&v| v > 0.0 && v < 1.0));
}

#[test]
fn decide_grad_matches_finite_difference() {
    // analytic gradient, storage [P,D] flat
    let (dwq, _) = run(GRAD, &inputs());
    assert_eq!(dwq.len(), 6);
    // central differences on the loss
    let eps = 1e-3f32;
    for s in 0..6 {
        let mut pin = params();
        let mut pout = params();
        pin.get_mut("wq").unwrap()[s] -= eps;
        pout.get_mut("wq").unwrap()[s] += eps;
        let ast = parser::parse(LOSS).unwrap();
        let checked = typeck::check_all(&ast).unwrap();
        let mut ev = Evaluator::new(&checked.ane[0].1);
        ev.set_params(pin);
        let l1 = ev.run(&inputs()).unwrap()[0];
        let mut ev = Evaluator::new(&checked.ane[0].1);
        ev.set_params(pout);
        let l2 = ev.run(&inputs()).unwrap()[0];
        let fd = (l2 - l1) / (2.0 * eps);
        assert!(
            (dwq[s] - fd).abs() < 2e-2,
            "dwq[{s}] = {} vs fd {fd}",
            dwq[s]
        );
    }
}

const TRAIN: &str = r#"
param wq: trainable f16[2, 3]
param wk: trainable f16[2, 3]
io hq: in f16[2]
io hos: in f16[2, 2]
io target: in f16[2]
fn decide_loss(hq: f16[2], hos: f16[2,2], target: f16[2]) -> f32 {
    mse(decide(hq, hos, wq, wk), target)
}
fn verdict(hq: f16[2], hos: f16[2,2]) -> f16[2] {
    decide(hq, hos, wq, wk)
}
fn main(hq: f16[2], hos: f16[2,2], target: f16[2]) -> f16[2] {
    let dwq = grad(decide_loss(hq, hos, target), wq);
    let dwk = grad(decide_loss(hq, hos, target), wk);
    wq -= 0.05 * dwq;
    wk -= 0.05 * dwk;
    verdict(hq, hos)
}
"#;

#[test]
fn decide_trains_weights() {
    let ast = parser::parse(TRAIN).unwrap();
    let checked = typeck::check_all(&ast).expect("check");
    let tp = &checked.ane[0].1;
    let mut p = params();
    let mut ev = Evaluator::new(tp);
    let inp = inputs();
    ev.set_params(p.clone());
    let _ = ev.run(&inp).expect("step0");
    let ups = ev.param_updates();
    assert!(!ups.is_empty(), "param updates recorded");
    for (name, vals) in ups {
        p.insert(name.clone(), vals.clone());
        // grads must be nonzero — a frozen update means grad saw nothing
        assert!(
            vals.iter().any(|v| v.abs() > 1e-9),
            "{name} update is all zeros"
        );
    }
    ev.set_params(p.clone());
    let out1 = ev.run(&inp).expect("step1");
    let _ = ev.run(&inp).expect("step2");
    for (name, vals) in ev.param_updates() {
        p.insert(name.clone(), vals.clone());
    }
    ev.set_params(p);
    let out2 = ev.run(&inp).expect("step3");
    let moved: f32 = out1
        .iter()
        .zip(&out2)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(moved > 0.0, "verdict did not change across updates");
}

#[test]
fn decide_rejects_bad_shapes() {
    let bad = r#"
param wq: trainable f16[2, 3]
param wk: trainable f16[2, 3]
io hq: in f16[2]
io hos: in f16[3, 3]
fn main(hq: f16[2], hos: f16[3,3]) -> f16[3] {
    decide(hq, hos, wq, wk)
}
"#;
    let ast = parser::parse(bad).unwrap();
    assert!(typeck::check_all(&ast).is_err());
}
