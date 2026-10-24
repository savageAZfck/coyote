# Coyote — language reference

`.cy` is a multi-stratum systems language for sovereign AI: the
**tensor stratum** compiles to resident binaries — on the neural
accelerator via MIL/HWX (`fn`), or on any stratum a distribution
registers (`gpu fn` and friends) — and the **host stratum** is a
bytecode VM on the CPU that drives those programs as first-class
values, carries governance, identity, and audit, and lowers to wasm.
One file can hold all of it — that file is an organism.

```
.cy ── parse ──► AST ── check_all ──► typed HIR + hostck
                        │
          ┌─────────────┴──────────────┐
     fn  (ane)                  ext fn (registered)
          │                          │
    emit → net.mil             Backend::emit
    compile_graph              Backend::load
    (HWX + Mach-O)             (distribution-provided)
          │                          │
          └────────► resident Vm + program values ◄────────┘
```

## 1. Declaration stratum (file scope)

```ebnf
decl      = import | invariant | cap | provenance | param | io
          | state | table | struct | enum | extern | fn | hostfn | reflex
import    = "import" string                       // file-relative or CYO_PATH
invariant = "invariant" ident                     // constitutional clause
cap       = "cap" ident ("," ident)*
provenance= "provenance" "{" (ident ":" string)* "}"
param     = "param" ident ":" qual? ty ("@" string)?
qual      = "plain" | "frozen" | "trainable" | "sealed"
io        = "io" ident ":" ("in"|"out") ty
state     = "state" ident "{" field+ "}"
table     = "table" ident ":" ty ("@" string)?    // host-side data, never lowered
struct    = "struct" ident "{" field+ "}"
enum      = "enum" ident "{" variant+ "}"
extern    = "extern" ("det"|"stoch")? "fn" ident "(" hargs ")" ("->" hty)? ("cap" idents)? ";"
```

- **`param`** — a weight tensor. Without `@ "file.safetensors::tensor"` it
  inits deterministically from its name (same name, same weights, every
  build). `param` sources support range slices: `tensor[0:2048]`, and
  multi-dim boxes `tensor[a:b,c:d]`.
- **quals** — `frozen` never updates; `trainable` updates via
  `param -= g*lr` or `adam(p, grad(loss,p), lr)`; `sealed` cannot be
  named from the host stratum (compile error) and never crosses the io
  boundary.
- **`state`** — persistent surfaces (KV caches) that live across steps.
- **`invariant`/`cap`/`provenance`** — the constitution: recorded into
  the build manifest and enforced. Reflexes must be `det`, cap-free,
  and return `Reflex`.
- **`import "x"`** — splices another module's declarations;
  file-relative first, then `CYO_PATH` (colon-separated). Modules are
  deduplicated in encounter order; the manifest hash covers all of them.

## 2. Tensor stratum

Typed tensor functions that lower to silicon. Tail-expression returns:
the fn's last expression is its value — no `return` keyword needed.

```ebnf
ty    = dtype "[" dim ("," dim)* "]"            // f16[64], f16[K, N]
dim   = int | ident                             // literals or generics
dtype = "f16" | "f32" | "q4"                    // q4 = 4-bit quantized
fn    = "fn" ident ("<" dims ">")? "(" args ")" ("->" ty)? "{" stmts expr? "}"
expr  = unary | bin | call | index | transpose
bin   = a @ b          // matmul (rhs must be a param or param.T for weight path)
      | a + b | a - b | a * b | a / b           // elementwise / broadcast
builtins: relu sigmoid tanh silu gelu softmax rmsnorm mse outer
        | gather | embed | concat | grad(loss, param)
postfix:  a.T          // transpose view (standalone or in matmul rhs)
```

- A top-level `fn` is a **program** — it compiles to an ANE binary.
  Other compute strata use a target prefix: `gpu fn` typechecks the
  same language but carries a `Gpu` target tag, resolved at emit time
  through `backend::register_backend`. The core toolchain registers
  nothing — a `gpu fn` in a bare build fails at emit with
  "no backend registered". The engine lives in whatever distribution
  ships it; an organism can mix backends in one file and pass either
  as a first-class `ane(...) -> ...` value.
- `grad(loss, p)` — reverse-mode AD: the forward values emit normally;
  backward emits the transpose/accumulate ops in the same binary.
  `adam(p, grad, lr)` folds momentum updates on-chip. (ANE + CPU only.)
- `param.T` — transposed view, used in gradient shapes.

## 3. Host stratum — the organism layer

`host fn`s are bytecode-compiled, run on CPU, and call programs as
first-class `ane(args) -> ret` values. **Explicit `return` is required**
on value-returning paths — a tail expression is a statement here, not
a return. `reflex name(args) -> Reflex { ... }` declares an audited
zero-capability arc.

```ebnf
hostfn  = "host" ("det"|"stoch")? "fn" ident "(" hargs ")" ("->" hty)? "{" hstmts "}"
hty     = i64 | f64 | bool | str | unit | token | dfa | hty tensor
        | "[" hty "]"        // vec      | "{" hty ":" hty "}"  // map
        | structName         | "ane(" htys ") ->" hty
hstmt   = let | var | assign | if | for | while | match | return
        | witness | migrate | call | expr
```

- `let`/`var` — immutable/mutable bindings. `x[i]`, `s.field`, `enum`
  patterns in `match`.
- Programs are values: `let f = affine;` then `f(x)` or passed as
  `ane(f16[64]) -> f16[64]` arguments.
- `self_token()` — the organism's witnessed identity `(id, epoch)`.
- `witness(tag, val)` — appends a hash-chained entry to the log
  (`witness.jsonl` in the build dir).
- `migrate(path)` serializes full VM state; `coyote run --resume`
  restores it.
- `die()` — attested halt.
- `extern` — FFI: `.cy`-declared functions bound at runtime to
  Rust/C/Swift callbacks (`coyote_bind_extern_i64`, `_str`).

### Natives

`print str len push argmax i64 f64 onehot causal rope_cos rope_sin
zeros rand has keys contains split slice_str sha256 parse_num
self_token witness die migrate dfa_compile dfa_scan quorum
propose_patch(cap patch) erase(cap erase) read_file/write_file(cap fs)
cortex_ask(cap cortex, stoch)`

Determinism: `det` fns may not call `stoch` natives/fns; `stoch` is the
annotated escape (sampling, external calls). Reflexes are forced `det`.

## 4. Execution model

- `coyote check` — parse + shape-check + constitutional check.
- `coyote build` — emit `net.mil` + compile graph (warm CYPLAN cache
  keyed on source sha).
- `coyote run` — every output is diffed against the CPU reference
  evaluator (the executable spec): `max |ane − cpu|` printed per step;
  trainable params and state surfaces feed back on-chip between steps.
- `coyote repl` — interactive: live VM for the file's fns/programs, a
  scratch VM for eval lines and incremental declarations.
- Organisms (`host fn main`) run the host VM with programs loaded on
  the selected backend (`--cpu` for pure-CPU execution).

## 5. Semantics notes

- Tensor fns: tail-expression return. Host fns: explicit `return`
  required on all paths — `host fn f() -> i64 { 42 }` is a compile
  error (`may reach its end without returning i64`).
- `sealed` params: naming them in host code is a compile error; they
  never appear in `param_names()` or the runtime surface.
- `witness` entries are `seq/tag/val/sha256-chained` — the audit head
  changes if any entry is tampered.
- Weight init is deterministic (name-seeded); `rand()` is a
  deterministic PRNG typed `stoch` for constitutional bookkeeping.
- `io` inputs on `coyote run` fill deterministically (`io::<name>`
  seeds) — runs are reproducible end to end.
