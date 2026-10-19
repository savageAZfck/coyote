// m2_ops.cy — M2 operator verification on silicon:
// - `q4` weight quantization (`param wq: q4[64, 64]`)
// - `gelu` activation
// - tensor slices `h[0..32]` and `h[32..64]`
// - `concat(lo, hi) -> f16[64]`
// - `gather` / `w[row]` (`w_emb[3]`)
// - `tensor / tensor` elementwise division (`a / b`)

param wq: frozen q4[64, 64]
param w_emb: frozen f16[32, 64]

io x: in f16[64]
io y: out f16[64]

fn main(x: f16[64]) -> f16[64] {
    let h = gelu(x @ wq);
    let lo = h[0..32];
    let hi = h[32..64];
    let swapped = concat(hi, lo);
    let row = w_emb[3];
    let denom = row * row + 1.5;
    (swapped + row) / denom
}
