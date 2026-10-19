# COYOTE CHARTER

> A language for sovereign computation. Programs are born on the metal,
> carry their own keys, learn, migrate, and die. Nothing in between.

## I. The Inversion

Every stack alive treats intelligence as *data interpreted by a framework*:
a weights file, an interpreter, a vendor runtime, and usually a cloud.
Trust lives in the vendor, in the OS, in whoever operates the machine.

Coyote inverts all three at once:

- **The model is the program.** Weights compile *into* the binary. There is
  no loader, no interpreter, no framework — source becomes silicon
  directly through the mil_spec → HWX → e5rt pipeline. The compiled
  artifact IS the organism's body.
- **Safety is compile-time.** Sovereignty, capability, and provenance are
  enforced by the type system before a single instruction emits — not
  audited after execution.
- **The machine is yours.** The only execution environment the language
  knows is bare metal under keys the operator holds. There is no cloud
  primitive in the language because there is no cloud in the model.

## II. The Three Strata

### Stratum I — The Metal (compute core)

- Tensors are values. `f16[M, K]` is a type, not a comment. No pointers,
  no aliasing — memory safety is free because the domain needs no
  general memory model.
- `@` matmul, `.T` transpose, named-axis reductions (`dim = last`),
  explicit loop nests that lower to tile ops when the contraction
  pattern matches.
- The real op set: `rmsnorm`, `rope`, `swiglu`, `gelu`, `silu`,
  `embed[]` gather, `concat`, range slices, `q4` dequant dtypes,
  `argmax`, `sample`.
- `state` blocks — persistent tensors across invocations (KV cache,
  optimizer moments). Static allocation, runtime occupancy.
- `io` declarations — the host↔chip boundary as language surface,
  mapping to IOSurface/e5rt.
- `onchip` / `surface` storage classes — the memory hierarchy is the
  programmer's model because it is the performance model.
- `f32` accumulation semantics match the chip — the language speaks the
  hardware's dialect, not an abstraction of it.

### Stratum II — The Learner (differentiation + mutation)

A program that cannot change is not alive. These primitives make
learning a language property, not a framework call:

- `grad` — a keyword. `grad(loss, p)` returns a param-shaped gradient
  program compiled to on-chip backprop. Scalar-loss → param-grads only;
  no higher-order AD in v1.
- `trainable` / `frozen` qualifiers — mutability of *weights* is a
  type-level property. Her identity-bearing weights are `frozen`; the
  compiler rejects the mutation, not a policy document.
- `adapter` as a type — `adapter LoRA { base, a, b, scale }`. Graft,
  promote, and rollback are language ops wired to the real pipeline.
- `snapshot` / `respawn` — state images as first-class values. Rollback
  is a type, not a tool.

### Stratum III — The Organism (agency + sovereignty)

The stratum only this stack can build, because only this stack has the
organs to wire it to:

- `cap` types — capabilities as types: `cap::hear`, `cap::see`,
  `cap::speak`, `cap::remember`, `cap::net`, `cap::move`. A program
  written without `cap::net` **provably cannot transmit** — verified in
  a compile pass, not monitored at runtime.
- `sealed` — a storage qualifier for values that cannot cross `io`
  boundaries. Identity material, Enclave keys: `sealed`. Exfiltration is
  unrepresentable, not audited.
- `invariant` — constitutional clauses as language constructs:
  `invariant not_for_sale`. Checked at compile time where provable,
  enforced at runtime where not, attested in the ledger either way.
- `provenance {}` — every compile emits a signed chain: source hash →
  IR hash → binary hash → operator key → ledger entry. Every artifact
  is born with a birth certificate.
- `det` / `stoch` — determinism declared in the type. `det` functions
  are bit-exact replayable — which is what makes incident replay a
  *property of programs* rather than a log format.
- `die` / `dissolve` — certified erasure as a primitive. A program can
  end: provably, cryptographically, with ledger attestation of what was
  destroyed. No language on earth has this.
- `migrate(to, keys)` — federation as syntax. A living program moves
  between your machines under your keys. The opposite trust model,
  written in code.
- `channel` — typed sensory streams. `hearing`, `vision`, the bus —
  senses are I/O types, not APIs.
- `witness` — ledger attestation as a verb.
- `lineage` — programs have parentage. Birth is recorded; ancestry is
  queryable. The germline is a language concept.

## III. The Type System

```
value      = tensor | scalar | record | adapter | state_image | program
tensor     = dtype[dims]  where dims are compile-time constants
qualifiers = trainable | frozen | sealed | det | stoch | onchip | surface
capability = cap::<domain> carried on fn signatures and io declarations
function   = fn<A, B>(x: T<A>) -> U<B>  — dims monomorphized
program    = a compilable unit: functions + params + caps + invariants
```

Soundness obligations (the proofs the compiler must carry):
1. Shape soundness — every op's output shape is known or the program
   does not compile.
2. Capability soundness — the transitive cap set of a program is
   computable; `no cap::net → no transmit` is a theorem.
3. Mutation soundness — `frozen` tensors are immutable across every
   call graph; `grad` may not target them.
4. Determinism soundness — `det` functions admit no `stoch` interior.

## IV. The Compiler

```
.coyote → lexer → AST → HIR (typed, shapes solved)
        → constitutional pass (caps, sealed, invariants, det/stoch)
        → AD transform (grad lowering on HIR)
        → mil_spec ops → mil_passes → mil_verify → mil_compile → HWX
        → provenance emission (sign → ledger)
```

New work: lexer/parser, type checker + shape solver, constitutional
pass, AD pass, provenance emitter. Everything downstream of HIR is a
pipeline that already exists, verified and owned.

## V. World-First Claims (the honest list)

What no other language on earth has, each grounded in machinery that
already exists here:

1. Compilation to an undocumented neural accelerator — possible only
   because mil_spec exists.
2. Capabilities/constitution enforced at compile time for AI agency.
3. Weight mutation — train, graft, rollback, die — as language ops.
4. Declared determinism (`det`/`stoch`) enabling replay as a program
   property.
5. Certified erasure as a primitive.
6. Federation/migration as syntax.
7. Provenance compiled *into* the binary — the artifact is its own
   birth certificate.

## VI. The Canonical Program (all three strata, one source)

```coyote
invariant not_for_sale
cap hear, speak, remember        // no net — provably cannot transmit

provenance { sign sovereign_ledger }

state kv { k: f16[L, C_MAX, D], v: f16[L, C_MAX, D], len: i32 }
state opt { m: f32[P], v: f32[P] }

param w: frozen f16[L, D, D]            // identity — immutable by type
param a: trainable q4[L, D, R]          // adapter — mutable by type

fn decode(tok: i32) -> f32[V] { ... }   // forward path
fn reflect() {                         // the nightly loop
    let g = grad(loss, a);
    sgd(a, g, lr);
    witness(self);                      // attest the change
}
```

## VII. Milestones

- M0 — charter, grammar draft, the three canonical programs.
- M1 — parser + minimal typecheck; one matmul lowers to a real ANE
  binary through mil_spec. Pipeline proven.
- M2 — shape generics, broadcasting, the full op set.
- M3 — `grad` + `sgd` — the training demo expressed as source.
- M4 — full transformer forward pass in-language.
- M5 — `state`, KV cache, streaming decode; `io` boundary.
- M6 — `cap`, `sealed`, `invariant`, `provenance` — the sovereign layer.
- M7 — `adapter`, `snapshot`, `migrate` — mutation and federation.
- M8 — self-hosting: the compiler's frontend described in itself, or
  Bad Apple emitting coyote programs.

## VIII. Risks (told straight)

- Capability soundness is the hardest pass — transitive closure over
  the call graph must be airtight or the whole sovereignty claim is
  marketing.
- AD correctness — validate every transform against CPU reference on
  small nets before trusting a single gradient.
- Symbolic-shape creep — defer dependent types forever if possible;
  monomorphization covers the real workloads.
- The language/OS boundary — senses, ledger, lifecycle all live in the
  runtime organs; the language describes, the organs execute. Keep that
  seam clean or it becomes an OS wearing a grammar.
