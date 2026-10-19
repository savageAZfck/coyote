# decode.cy — one transformer decode step with a KV cache.
#
# The cache is `state`: packed [T, D] surfaces that persist across
# invocations and feed back each step. `pos` is a one-hot write
# position; `mask` is the causal mask. `kv.k[pos] = v` is a scatter
# write — it lowers to `kv.k + outer(pos, v)`, a single packed add
# on-chip.

param wq: f16[64, 64]
param wk: f16[64, 64]
param wv: f16[64, 64]
param wo: f16[64, 64]
param wg: f16[64, 64]
param w1: f16[64, 64]
param w2: f16[64, 64]

state kv {
    k: f16[64, 64]
    v: f16[64, 64]
}

io x: in f16[64]
io pos: in f16[64]
io mask: in f16[64]
io y: out f16[64]

fn main(x: f16[64], pos: f16[64], mask: f16[64]) -> f16[64] {
    let q = x @ wq;
    kv.k[pos] = x @ wk;
    kv.v[pos] = x @ wv;
    let s = softmax(q @ kv.k.T * 0.125 + mask);
    let a = s @ kv.v @ wo;
    x + a + (silu(x @ wg) * (x @ w1)) @ w2
}
