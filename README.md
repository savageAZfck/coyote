# coyote

A programming language for the Apple Neural Engine.

`.cy` source compiles directly to the HWX Mach-O binaries the ANE driver
loads. No CoreML, no `.mlpackage`, no Python, no Apple codegen anywhere in
the pipeline. The model isn't data read by a framework — it's a compiled
program running on silicon you own.

```coyote
param wg:  frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.gate_proj.weight"
param wu:  frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.up_proj.weight"
param wd:  frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.down_proj.weight"
param g0:  frozen f16[1024]       @ "models/qwen3-0.6b/model.safetensors::model.layers.0.input_layernorm.weight"

io x: in f16[1024]
io y: out f16[1024]

fn main(x: f16[1024]) -> f16[1024] {
    let h = rmsnorm(x, g0);
    x + (silu(h @ wg) * (h @ wu)) @ wd
}
```

That's a real Qwen3-0.6B MLP block — the actual checkpoint tensors —
compiled to ANE programs and verified on-chip against the CPU evaluator
at fp16 noise (`max |ane − cpu| = 0.001`).

## what it does today

- **`grad` is a keyword.** `w -= 0.1 * grad(loss(x, t, w1, w2), w1)`
  lowers to forward + backward + an in-place packed weight update on the
  chip. Training is a language primitive, not a framework call.
- **`state` blocks** are persistent on-chip tensors — KV caches that
  scatter-write a row per step and attend over accumulated history.
- **`@ "file.safetensors::tensor"`** loads real weights (BF16/F16/F32),
  with `[r0:r1,c0:c1]` range slices for head-split projections.
- **Shapes are types.** `f16[64, 64] @ f16[128]` fails at compile time
  with a source location, not a runtime crash.
- **`sealed` values cannot reach an io boundary** — the compiler rejects
  the program. Privacy is a compile error, not a policy.
- Every artifact is born with a manifest: source hash, MIL hash,
  invariants, capabilities, weight provenance.

## receipts

`examples/qwenlayer.cy` — a full Qwen3-0.6B decoder layer: 16 GQA
attention heads, RoPE, KV cache (`state`), SwiGLU MLP, residuals. All
105 weights sliced from the real safetensors checkpoint. 635 native ANE
programs, KV accumulating across steps:

```
step   0  |ane−cpu| pred 0.002856  kv 0.179768  |kv| 2309.049
step   3  |ane−cpu| pred 0.002617  kv 0.179768  |kv| 9236.196
✓ 4 steps — weights live on-chip, fed back each step
```

`examples/train.cy` — on-chip training via `grad` (MSE 1.35 → 0.12 over
20 steps, weights fed back each step). `examples/decode.cy` — KV-cache
decode step. `examples/ffn.cy` — first light.

## build + run

```sh
cargo build
./target/debug/coyote check  examples/ffn.cy          # parse + shape-check
./target/debug/coyote build  examples/ffn.cy -o out/ffn
./target/debug/coyote run    examples/ffn.cy -o out/ffn
./target/debug/coyote run    examples/train.cy -o out/train --steps 20
./target/debug/coyote run    examples/qwenlayer.cy -o out/qwenlayer --steps 8
```

The qwen examples want a real checkpoint at `models/qwen3-0.6b/` —
download `Qwen/Qwen3-0.6B` `model.safetensors` (BF16) from Hugging Face.

## the language

`param`/`io`/`state`/`invariant`/`cap`/`provenance` decls, generic-dim
fns (`fn ffn<D,F>(x: f16[D]) -> f16[F]`), `@`/`.T`, `let`/`var`/`op=`,
builtins `relu sigmoid tanh silu softmax rmsnorm mse outer`, `grad`
reverse-mode AD, `trainable`/`frozen`/`sealed` qualifiers. See
[SPEC.md](SPEC.md) for the grammar and semantics, [CHARTER.md](CHARTER.md)
for where it's going (a general-purpose language for AI/AGI/artificial
life — control flow, structs, CPU codegen, and the constitution the
organism lives under, enforced at compile time).

Requires: Apple Silicon, macOS with the ANE. MIT.
