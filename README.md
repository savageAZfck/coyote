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

## the organism stratum

Coyote is two languages that share a type system. `fn` bodies compile to
ANE programs; `host fn` bodies run in a bytecode VM that treats compiled
programs as first-class values:

```coyote
fn layer(x: f16[1024], pos: f16[64], mask: f16[64],
         cos: f16[64], sin: f16[64]) -> f16[2048] { /* GQA + SwiGLU */ }

table emb: f16[151936, 1024] @ "model.safetensors::model.embed_tokens.weight"

host fn main() -> i64 {
    var tok = 151643;                 // BOS
    var x = emb[tok];                 // row gather — host data, never silicon
    var step = 0;
    while step < 8 {
        let lg = layer(x, onehot(step % 64, 64), causal(step % 64, 64),
                       rope_cos(step, 64), rope_sin(step, 64));
        tok = argmax(lg);             // argmax on the host, like Apple's sampler
        witness("tok", tok);          // hash-chained log — the organism's ledger
        x = emb[tok];
        step += 1;
    }
    return tok;
}
```

- **Programs are values.** `let f = layer; f(x, ...)` dispatches to
  silicon (or `--cpu` reference). A program's type is its io signature —
  a shape mismatch across the CPU↔ANE boundary is a compile error.
- **Host types**: `i64 f64 bool str token dfa [T] {K: V} struct enum`,
  control flow `if/else match while for break continue return`,
  `import "file.cy"`, `% && || !`.
- **Constitutional pass at compile time:**
  - **Transitive `cap` checking:** capabilities (`fs`, `patch`, `erase`,
    `cortex`, `net`, ...) propagate across the call graph; using an
    undeclared capability is a compile error (`no cap::X → no X`).
  - **Determinism (`det` / `stoch`):** `host det fn` and `reflex` bodies
    reject any direct or transitive call to stochastic ops (`rand`, `cortex_ask`).
  - **Audited reflexes (`reflex` / `axon`):** `reflex name(...) -> Reflex`
    is statically forced to be `det`, zero-capability, and non-steering
    (reflexes brake, never steer); every non-`Pass` firing (`Alert`,
    `Brake`, `Quarantine`) is automatically hash-chained into `witness.jsonl`.
- **Sovereign organ primitives built into the language:**
  - `dfa_compile` / `dfa_scan` — streaming Aho-Corasick firewall (`macuahuitl`).
  - `quorum` — council deliberation with structurally preserved minority dissent (`tokala`).
  - `propose_patch` — bounded-autonomy marshal with protected-surface & blast-radius caps (`akicita`/`xipe`, requires `cap patch`).
  - `erase` — certified erasure with verified decryption failure (`camazotz`, requires `cap erase`).
  - `read_file` / `write_file` (`cap fs`) and `cortex_ask` (`cap cortex`, SLICKS bridge).
- **`table`** declares host-side data (embedding tables, corpora) — it is
  unnameable inside `fn`, so a 600MB lookup table can never become an ANE
  weight blob.
- **`self_token()`** returns the organism's witnessed identity `{id, epoch}` —
  `id` binds to the build manifest, `epoch` advances with the witness chain.
- **`witness(tag, val)`** appends to a hash-chained log — every token,
  every decision, tamper-evident by construction.
- **`migrate(path)`** freezes the whole organism mid-thought — stack,
  frames, tensors, maps, enums, DFAs, witness chain — into a snapshot.
  `--resume` refuses a body that isn't this organism's own (lineage check
  on the manifest hash) and continues the chain where it froze, even across
  ANE ↔ `--cpu` backends.
- **`die()`** halts with attestation.

The split is deliberate: `cap`/`sealed`/`det`/`reflex`/`invariant` are
compile-time — the constitution is enforced before silicon ever runs.
`die`/`migrate`/`witness` are runtime — identity is produced by living.

## receipts

`examples/qwen28.cy` — a **full 28-layer Qwen3-0.6B decode** driven
entirely from `.cy` source: 29 compiled ANE programs (17,780 program
nodes across `l0`..`l27` + `head`), 2,942 real safetensors weight slices,
28 independent instanced KV caches (`l0.kv`..`l27.kv`) from one `state kv`
spec, surface pooling across ~143k dispatches, every generated token
hash-chained into `witness.jsonl`.

`examples/sovereign.cy` (+ `examples/organs.cy`) — the full sovereign
anatomy in one `.cy` program: `import`, `enum`/`match`, `{K: V}` maps,
`reflex` arc (`axon`), `dfa_compile`/`dfa_scan` (`macuahuitl`),
dissenting `quorum` (`tokala`), `propose_patch` (`akicita`/`xipe`),
certified `erase` (`camazotz`), `cap`-gated file I/O + `cortex_ask`,
and ANE/`--cpu` execution.

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
20 steps, weights fed back each step). `examples/adam_m3b.cy` — on-chip
backprop through `rmsnorm`, `relu`, `gelu`, and `softmax` with `adam`
moment updates. `examples/m2_ops.cy` and `examples/batched_m2.cy` —
batched matmul (`M > 1`), `q4` weights, `gelu`, `concat`, slices
`h[0..32]`, `w[row]` gather, standalone `.T`, and `tensor / tensor`
division verified on silicon against CPU. `examples/decode.cy` — KV-cache
decode step. `examples/ffn.cy` — first light.

`examples/qwenorg.cy` — single-layer Qwen3 organism decode loop.
`examples/organism.cy` — structs, vectors, loops, recursion, and a
program passed by value without needing a checkpoint.

## build + run

```sh
cargo build
./target/debug/coyote check  examples/sovereign.cy    # parse + shape + constitutional check
./target/debug/coyote run    examples/sovereign.cy -o out/sovereign
./target/debug/coyote run    examples/sovereign.cy -o out/sovereign --cpu
./target/debug/coyote run    examples/train.cy -o out/train --steps 20
./target/debug/coyote run    examples/qwen28.cy -o out/qwen28
```

The qwen examples want a real checkpoint at `models/qwen3-0.6b/` —
download `Qwen/Qwen3-0.6B` `model.safetensors` (BF16) from Hugging Face.

## the language

`param`/`io`/`state`/`invariant`/`cap`/`provenance`/`table`/`struct`
decls, generic-dim fns (`fn ffn<D,F>(x: f16[D]) -> f16[F]`), `@`/`.T`,
`let`/`var`/`op=`, builtins `relu sigmoid tanh silu softmax rmsnorm mse
outer`, `grad` reverse-mode AD, `trainable`/`frozen`/`sealed` qualifiers.
`host fn` bodies: full control flow, structs, vectors, first-class
program values, and the organism natives — `argmax onehot causal rope_cos
rope_sin zeros self_token witness die migrate rand`. See
[SPEC.md](SPEC.md) for the grammar and semantics, [CHARTER.md](CHARTER.md)
for where it's going.

Requires: Apple Silicon, macOS with the ANE. MIT.
