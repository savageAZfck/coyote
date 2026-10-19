# coyote — first light
# a SwiGLU feed-forward block with RMSNorm, compiled to ANE.
#
#   coyote check examples/ffn.cy
#   coyote build examples/ffn.cy -o out/ffn
#   coyote run   examples/ffn.cy -o out/ffn

invariant never_sells_self
cap hear

provenance {
    author: "savage"
    purpose: "first light"
}

param gam: frozen f16[64]
param wu:  frozen f16[64, 128]
param wg:  frozen f16[64, 128]
param wd:  frozen f16[128, 64]

io x: in f16[64]
io y: out f16[64]

fn ffn<D, F>(x: f16[D], wu: f16[D, F], wg: f16[D, F], wd: f16[F, D]) -> f16[D] {
    let u = x @ wu;
    let g = silu(x @ wg);
    (u * g) @ wd
}

fn main(x: f16[64]) -> f16[64] {
    let h = rmsnorm(x, gam);
    x + ffn(h, wu, wg, wd)
}
