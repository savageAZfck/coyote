// finetune.cy — on-chip adapter fine-tuning.
//
// A frozen base weight plus a factorized residual adapter — the LoRA
// pattern: `x @ w + (relu(x @ a) @ b)`. Forward, backward, and adam
// optimizer updates all compile into the same ANE binary; the adapted
// weights never leave silicon between steps.
//
// Rank note: trainable params arrive as activation surfaces, so
// adapter matmuls dispatch on the activation×activation skeleton
// table — (t,d) ∈ {64,256,1024} × {32,64,128}. Rank 64 is the
// smallest on-table bottleneck; r<64 needs a new skeleton.
//
//   coyote run examples/finetune.cy -o out/finetune --steps 20
//
// To adapt a real checkpoint slice instead of the deterministic init,
// point the frozen param at a safetensors tensor:
//
//   param w: frozen f16[64, 64] @ "models/qwen.safetensors::tensor.name"

invariant never_sells_self

provenance {
    author: "savage"
    purpose: "grad toolkit — residual adapter on ANE"
}

param w: frozen    f16[64, 64]
param a: trainable f16[64, 64]
param b: trainable f16[64, 64]

io x: in f16[64]
io t: in f16[64]
io y: out f16[64]

fn adapt(x: f16[64], w: f16[64, 64], a: f16[64, 64], b: f16[64, 64]) -> f16[64] {
    (x @ w) + (relu(x @ a) @ b)
}

fn loss(x: f16[64], t: f16[64], w: f16[64, 64], a: f16[64, 64], b: f16[64, 64]) -> f32 {
    mse(adapt(x, w, a, b), t)
}

fn main(x: f16[64], t: f16[64]) -> f16[64] {
    adam(a, grad(loss(x, t, w, a, b), a), 0.01);
    adam(b, grad(loss(x, t, w, a, b), b), 0.01);
    adapt(x, w, a, b)
}
