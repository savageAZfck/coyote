# coyote — language spec (v1)

bare-metal language for sovereign computation. source → silicon, nothing in
between: `.cy` → lexer → parser → shape-checked TIR (all calls inlined, all
dims concrete) → MIL text → `mil_infer::graph::compile_graph` → HWX binaries
→ ANE.

the load-bearing decisions:

- **tensors are values.** no pointers, no aliasing, no references — memory
  safety is free because the domain never needed the machinery.
- **shapes are in the type.** `f16[64, 128]` is a type; a mismatched `@` is
  a compile error, never a runtime surprise.
- **everything that parses typechecks; everything that typechecks lowers.**
  there is no "works in the checker, fails in codegen" gap — the admitted
  builtin set IS the backend set.
- **constitution is enforced by the compiler.** `sealed` values are tainted
  at lowering; reaching the io boundary is a compile error.

## grammar

```
program    := decl*
decl       := invariant | cap | provenance | param | io | state | fn
invariant  := 'invariant' ident
cap        := 'cap' ident (',' ident)*
provenance := 'provenance' '{' (ident ':' stringlit)* '}'
param      := 'param' ident ':' qual? type ('@' stringlit)?
io         := 'io' ident ':' ('in'|'out') type
state      := 'state' ident '{' (ident ':' type)* '}'      (parsed; M5)
fn         := 'fn' ident ('<' dim (',' dim)* '>')?
              '(' (ident ':' type)* ')' ('->' type)? block
type       := dtype ('[' dim (',' dim)* ']')?
dtype      := f16 | f32 | i32 | q4
dim        := integer | ident        # idents are generic dims
qual       := frozen | trainable | sealed
stmt       := 'let' ident (':' type)? '=' expr ';'
            | 'var' ident (':' type)? '=' expr ';'
            | ident assign_op expr ';'
            | 'return' expr ';'
            | expr ';'
expr       := literal | ident | call | '(' expr ')'
            | expr binop expr | '-' expr | expr '.T'
            | expr '[' index (',' index)* ']'                (parsed; M5)
            | expr '.' ident                                 (state field; M5)
binop      := '@' | '*' | '/' | '+' | '-' | '==' | '!=' | '<' | '>' | '<=' | '>='
```

`@` binds like `*`. named args are `f(x, eps = 1e-5)`.

## v1 semantics

- `fn main` is the entry. its args bind `io in` decls positionally; its
  result must match `io out`.
- generic dims (`fn f<D, F>(x: f16[D]) -> f16[F]`) are bound by unification
  at each callsite and monomorphized — user fns inline into `main`.
- `x @ w`: vector × weight. `w` must be a rank-2 `param` (or `w.T`).
  K ≥ 32, N ≥ 32 (ANE skeleton constraint).
- elementwise `+ - *` on equal shapes or scalar broadcast; `/` by scalar only.
- assignment is SSA rebinding: `var x` reassigned produces a new mil value.
- builtins (all shape-preserving unless noted):
  `relu` (n=64 only — skeleton limit), `sigmoid`, `tanh`, `silu`,
  `softmax`, `rmsnorm(x[C], w[C], eps=…)`, `mse(a, b) → f32`
- `param` blobs init deterministically — xorshift64 seeded by name — so the
  same source always compiles the same artifact. `frozen`/`trainable` are
  recorded in the manifest; `sealed` is enforced: a sealed value reaching
  the output is a compile error.
- `param w: f16[K, N] @ "file.safetensors::tensor"` loads real weights.
  the file order is the *storage* layout: matvec weights are `[N, K]`
  = `[out, in]` (PyTorch order — no transpose), vectors `[C]` or `[1, C]`.
  BF16/F16/F32 decode to f16; shape mismatch is a hard compile error,
  and `src` is recorded in the manifest's birth certificate.
- `"file::tensor[r0:r1]"` / `"file::tensor[r0:r1,c0:c1]"` takes a
  row-major range slice at load time — head-split projections and
  per-head o-column slices without any runtime slicing.
- `sigmoid`/`silu` at n ≤ 1024 hit the LUT kernel; above that they
  decompose into a clamped odd-minimax polynomial (deg-13, |err| ≤ 4e-4)
  over `minimum`/`maximum`/`mul`/`add`, which cover any n. the cpu
  evaluator computes exact σ — reported diffs include the approximation.
- decode-loop io conventions: `pos` (one-hot), `mask` (causal), and
  `cos`/`sin` (rotate-half rope table, θ base 1e6) are generated
  per-step by the driver.
- `invariant`, `cap`, `provenance` are compiled into `coyote_build.json`
  alongside source/mil sha256 — the artifact's birth certificate.

## state — persistent tensors (KV cache)

```
state kv {
    k: f16[64, 64]        # [T, D] packed frames — the cache
    v: f16[64, 64]
}
io pos: in f16[64]      # one-hot write position (host convention)
io mask: in f16[64]     # causal mask (0 for t<=now, -1e4 beyond)

fn main(x: f16[64], pos: f16[64], mask: f16[64]) -> f16[64] {
    kv.k[pos] = x @ wk;                       # scatter write
    kv.v[pos] = x @ wv;
    let s = softmax(q @ kv.k.T * 0.125 + mask);  # masked scores
    let a = s @ kv.v @ wo;                    # value contraction
    x + a
}
```

- `state` fields are runtime packed `[1, T, 1, D]` surfaces — zero-init,
  fed back each step like trainable params. The mil arg is `kv_k`; the
  source name is `kv.k`.
- `kv.k[p] = v` / `+=` — **scatter by one-hot**: desugars to
  `kv.k + outer(p, v)` — one packed add on-chip. Writes are additive;
  fresh slots start zeroed (cache-append semantics). `-=` negates the
  value inside the outer.
- `outer(a[T], b[D]) -> [T, D]` — packed row×col product.
- `q @ kv.k.T` — scores over cache rows (`transpose_y` → avw);
  `s @ kv.v` — value contraction (`transpose_x` → attn·V). The flag is
  chosen by whether the field is `.T`'d — no annotation needed.
- io conventions: `pos` and `mask` io-ins are filled per-step by the
  host (one-hot / causal). Skeleton sizes: `t ∈ {64,256,512,1024}`,
  `d = 64` for scatter; `attn·V` adds `t×{64,128}` at `t=1024`.

## grad — training as a primitive

```
let dw = grad(loss(x, t, w1, w2), w1);   # d loss / d w1 → f16[K, N]
w1 -= lr * grad(loss(x, t, w1, w2), w1); # the SGD step, folded
```

- `grad(call, param)` — tape-based reverse-mode AD over the inlined loss.
  The loss must end in `mse(pred, target)`; the scalar never materializes —
  the seed `2(p−t)/n` is an elementwise chain.
- backward rules emitted: matmul (`transpose_x` matvec for input-grads,
  `transpose_x` outer for weight-grads), add/sub/mul/div-by-scalar,
  `sigmoid`/`tanh`/`silu`/`mse`. `relu`, `softmax`, `rmsnorm`, `.T`
  backwards land in M3b.
- `param w: trainable` — a runtime tensor, not a const: storage is packed
  `[N, K]` (`[1, N, 1, K]` on the wire), so forward hits the avw route and
  backward hits attn·V/outer. `frozen`/`sealed` can't be assigned;
  `trainable` can.
- `w op= s * grad(loss, w)` folds to a `ParamUpdate`: the seed is
  pre-scaled by `±s` before the outer product, so the update is ONE packed
  add — exactly train.mil's `ndy` trick. Updates become extra func
  outputs; the host feeds them back next step.
- io ports feeding a trainable matmul directly are auto-declared
  `[1, 1, 1, K]` (contig) — the avw route can't repack an external
  surface.
- `coyote run --steps N` runs the weight-feedback loop on-chip and prints
  per-step mse + silicon-vs-spec drift. fp16 updates accumulate honest
  quantization drift across steps — that IS hardware training.

## layout contract (coyote ↔ mil_infer)

| coyote        | mil text                          |
|---------------|-----------------------------------|
| `f16[C]`      | `tensor<fp16, [1, C, 1, 1]>`      |
| matvec result | `tensor<fp16, [1, 1, 1, N]>`      |
| `f16[K, N]` w | blob stored `[N, K]` + transpose_y|

elementwise consumers treat flat f16 buffers; the graph compiler inserts
repacks between strided/contiguous currencies automatically.

## driver

```
coyote check f.cy            # parse + shape-check
coyote build f.cy -o dir     # → net.mil + *.bin + coyote_build.json + *.macho
coyote run   f.cy -o dir     # + execute on ANE + diff vs CPU reference
```

## roadmap (the charter's order)

- **M2** batched matmul (M>1 via conv1x1), gather/`embed[]`, concat, slices,
  q4 dequant-at-load, tensor ÷ tensor, gelu, `.T` outside `@`
- **M4-lite** ✓ `state` blocks — packed persistent surfaces, scatter
  writes via one-hot `outer`, masked KV-cache attention on-chip
  (`examples/decode.cy`, verified vs CPU)
- **M3** ✓ `grad(loss, w)` — reverse-mode AD, folded SGD updates, on-chip
  training verified vs CPU. M3b: relu/softmax/rmsnorm/.T backwards,
  adam state
- **M4** `if`/comparisons, masks, transformer block, decode step
- **M5** `state` blocks — fixed-capacity KV cache + runtime occupancy,
  index/slice lowering, `io` streaming
- **M6** `cap` flow-checking (transitive net/mem/sense capabilities),
  `det`/`stoch` declared determinism, signed artifacts, `migrate`, `die`,
  `witness`, `lineage` — the organism stratum

## the organism stratum (implemented)

two strata share one type system. `fn` compiles to an ANE program;
`host fn` compiles to bytecode for the organism VM. every concrete `fn`
becomes its own program artifact (`-o dir/<name>/`).

```
decl       += 'table' ident ':' type ('@' stringlit)?
            | 'struct' ident '{' (ident ':' hty)* '}'
            | 'host' 'fn' ident '(' (ident ':' hty)* ')' ('->' hty)? block
hty        := 'i64' | 'f64' | 'bool' | 'str' | 'token' | 'unit'
            | dtype '[' int (',' int)* ']'          # literal dims only
            | '[' hty ']' | ident                    # struct name
            | 'ane' '(' (hty)* ')' '->' hty          # program value type
hstmt      := 'let'|'var' ident (':' hty)? '=' expr ';'
            | lval assign_op expr ';'
            | 'if' expr block ('else' (block | 'if' ...))?
            | 'while' expr block
            | 'for' ident 'in' expr '..' expr block
            | 'break' ';' | 'continue' ';' | 'return' expr? ';'
expr       += boollit | stringlit | '[' expr* ']' | ident '{' field '=' expr* '}'
            | expr '.' ident                         # struct/token field
            | expr '&&' expr | expr '||' expr | '!' expr | expr '%' expr
```

semantics:

- **programs are values.** an ane fn's name bound in host scope has type
  `ane(argtys) -> retty`. calling it — by name or through a variable —
  marshals tensors into surfaces, runs GraphRuntime, reads the output.
  arg/ret shapes and dtypes must match the signature exactly.
- **tables are host data.** `table emb: f16[V, D] @ "file::tensor"` loads
  in declared order, is indexable (`emb[tok]` → row `f16[D]`), and is
  unnameable inside `fn` — a 300MB lookup table can never become a
  weight blob. rank-2 tensor indexing in host code is a row gather;
  rank-1 indexing yields `f64`.
- **state fields are readable** from host code as `kv.k0lo` — the on-chip
  mirror, written back after every program call.
- **sealed params are unnameable in host code.** the boundary is total:
  not a lint, a missing name.
- **`token`** carries `{id, epoch}`. `id` binds to the build manifest
  hash; `epoch` counts witness entries. `.id`/`.epoch` are the only
  fields.
- **`witness(tag, val)`** appends sha256-chained `{seq, tag, val, hash}`
  entries — the organism's ledger, written to `<out>/witness.jsonl`.
- **`migrate(path)`** serializes stack, frames, tensors, globals, rng and
  the witness chain. `coyote run f.cy --resume snap.bin` verifies the
  snapshot's lineage hash matches this organism's manifest, restores the
  chain, and continues mid-execution — the resumed body's first entry is
  `resume`, chained to the frozen tip.
- **`die()`** witnesses and halts.

natives: `print str len push argmax i64 f64 onehot causal rope_cos
rope_sin zeros self_token witness die migrate rand`.
`rope_cos/sin(pos, d)` is the Qwen3 rotate-half table `θ_i = 1e6^(-2i/2d)`;
`causal(upto, n)` is `0` below `upto`, `-1e4` above; `rand()` is
xorshift64* seeded by lineage — deterministic per organism.

the VM is the point: `die`, `migrate`, `witness` and respawn need
inspectable, serializable execution state — an organism that can freeze
itself mid-thought cannot be native code.

- **M6 (partial, landed)**: `host fn` stratum, program values, `token`,
  `witness` hash chain, `migrate`/`--resume`, `die`, `table`, `struct`,
  full host control flow. remaining: `cap` transitive flow-checking,
  `det`/`stoch`, signed artifacts, `lineage` across machines.
