// batched_m2.cy — Batched matmul (M > 1), standalone `.T`, and `q4` weights on silicon.

param w1: frozen q4[32, 32]
param w2: frozen f16[2, 2]

io x: in f16[2, 32]
io y: out f16[2, 32]

fn main(x: f16[2, 32]) -> f16[2, 32] {
    let h = gelu(x @ w1);
    let ht = h.T;
    let back = (ht @ w2).T;
    h + back
}
