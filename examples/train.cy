# coyote — training as a language primitive.
# a two-layer MLP that learns on ANE silicon: forward, backward, and
# the weight update all compile to the same binary. `grad` is a keyword.
#
#   coyote run examples/train.cy -o out/train --steps 20

invariant never_sells_self

provenance {
    author: "savage"
    purpose: "grad — first light for learning"
}

param w1: trainable f16[64, 64]
param w2: trainable f16[64, 64]

io x: in f16[64]
io t: in f16[64]
io y: out f16[64]

fn loss(x: f16[64], t: f16[64], w1: f16[64, 64], w2: f16[64, 64]) -> f32 {
    let h = sigmoid(x @ w1);
    let p = h @ w2;
    mse(p, t)
}

fn predict(x: f16[64], w1: f16[64, 64], w2: f16[64, 64]) -> f16[64] {
    sigmoid(x @ w1) @ w2
}

fn main(x: f16[64], t: f16[64]) -> f16[64] {
    let dw2 = grad(loss(x, t, w1, w2), w2);
    let dw1 = grad(loss(x, t, w1, w2), w1);
    w2 -= 0.1 * dw2;
    w1 -= 0.1 * dw1;
    predict(x, w1, w2)
}
