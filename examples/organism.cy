// organism.cy — the general stratum, end to end.
// host fns run in the VM; ane fns compile to silicon and are
// first-class values — passed, stored, called.

cap print, witness
invariant sealed_never_named

param w: f16[64, 64]
param secret: sealed f16[64]   // naming this in host code is a compile error

struct Frame {
    step: i64,
    tok: i64,
    score: f64,
}

fn affine(x: f16[64]) -> f16[64] {
    relu(x @ w + x)
}

host fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

// a program passed as a value — her "first-class program"
host fn run_step(f: ane(f16[64]) -> f16[64], x: f16[64]) -> f16[64] {
    return f(x);
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    print("organism", me.id, "epoch", me.epoch);

    var total = 0;
    for i in 0..10 {
        total += i;
    }
    var n = 0;
    while n < 4 {
        n += 1;
    }
    let fr = Frame { step: total + n, tok: fib(10), score: rand() };
    print("frame", fr.step, fr.tok, fr.score);

    var xs: [i64] = [];
    push(xs, 3);
    push(xs, 1);
    push(xs, 4);
    print("vec", len(xs), xs[0] + xs[1] + xs[2]);

    // call the program by name, then through a value —
    // both dispatch to silicon
    let x = zeros(64);
    let y1 = affine(x);
    let f = affine;
    let y2 = run_step(f, x);
    print("argmax", argmax(y1), argmax(y2));

    witness("steps", total);
    witness("fib", fr.tok);
    let me2 = self_token();
    print("epoch advanced", me2.epoch - me.epoch);
    witness("done", me2);
    return fr.tok;
}
