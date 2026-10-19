// adam_m3b.cy — M3b verification on silicon:
// - backward rules for `rmsnorm`, `relu`, `gelu`, `softmax`
// - `adam(w, grad(loss, w), lr)` optimizer state updates across steps

param w1: trainable f16[64, 64]
param w2: trainable f16[64, 64]
param gamma: frozen f16[64]

io x: in f16[64]
io t: in f16[64]
io y: out f16[64]

fn forward(x: f16[64], w1: f16[64, 64], w2: f16[64, 64]) -> f16[64] {
    let h0 = rmsnorm(x, gamma);
    let h1 = relu(h0 @ w1);
    let h2 = gelu(h1) @ w2;
    softmax(h2) * 4.0 + h1
}

fn loss(x: f16[64], t: f16[64], w1: f16[64, 64], w2: f16[64, 64]) -> f32 {
    let p = forward(x, w1, w2);
    mse(p, t)
}

fn main(x: f16[64], t: f16[64]) -> f16[64] {
    let p = forward(x, w1, w2);
    adam(w1, grad(loss(x, t, w1, w2), w1), 0.03);
    adam(w2, grad(loss(x, t, w1, w2), w2), 0.03);
    p
}
