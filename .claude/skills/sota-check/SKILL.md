---
name: sota-check
description: One repeatable pass that stops fak from re-inventing known kernel art - before writing or optimizing a compute kernel (a quantized GEMM, a fused attention, a KV-cache reuse, a MoE dispatch, a Metal/CUDA kernel), it checks the SOTA prior-art matrix for the production reference (llama.cpp / Marlin / CUTLASS / FlashInfer / vLLM / SGLang / a named paper), decides the route (borrow / bind / stay-minimal), holds the result to the named oracle, and records what was consulted in a Prior-art trailer. Runs `fak sota <op|file>` to surface the reference, reads it, routes deliberately, and (when the matrix has a blind spot) adds the missing row so the next person inherits the map. The inward kernel-engineering counterpart of industry-score (the outward field map). Use before any kernel-optimization commit, when the PRIOR_ART advisory gate fires, when onboarding a new compute operation, or on a /loop cadence to keep the matrix honest against the tree. Preserve useful cohort-specific alternatives as bounded optional modules or recipes when they remain supportable, while excluding genuinely superseded approaches.
---

# sota-check - the prior-art-before-scratch pass

## Why this skill exists

This repo has a documented failure mode: an agent reaches for "implement the Mac Q6_K fused
MLP from scratch" or "hand-roll the amd64 kquant SIMD" without first checking that
llama.cpp's GGML kernels, Marlin, CUTLASS, FlashInfer, or a named paper already solved the
same contraction - and re-derives, badly, what is known art. Prior art WAS being researched
(idea_scout, the docs/notes/RESEARCH-* corpus, the SOTA matrix) but the research was inert:
nothing on the kernel-commit path forced an agent to consult it.

The law, in one line: **the default answer to "should I write this kernel from scratch?" is
NO - find the production reference first, route deliberately (borrow / bind / stay-minimal),
prove against an oracle, and record what you read.**

The single source of truth is `internal/sotamatrix` (a flat in-binary literal, same discipline
as `internal/benchcatalog`). The human front door is
[`docs/sota/README.md`](../../../docs/sota/README.md). Three surfaces read the matrix: the
`fak sota` command (lookup), the `PRIOR_ART` advisory gate (commit-time nudge), and
`tools/sota_coverage_scorecard.py` (keeps the matrix complete against the tree).

## The pass

1. **Look up and capture the reference - BEFORE writing code.** Run the installed
   first-class binary and save its read-only machine payload:
   ```bash
   fak sota <slug> --json > sota-<slug>.json
   fak sota <the-file-you-are-about-to-edit> --json > sota-path.json
   ```
   It reports the SOTA stack, `PrimaryLink`, chosen route, oracle, and named papers. `fak sota
   list` enumerates every operation. If the operation is NOT in the matrix, that is itself the
   finding - go to step 5. Persist a generated study note only through the separate explicit
   `fak sota <slug> --write` gate; JSON capture alone must not mutate the repository.
2. **Read the primary reference.** Open the `PrimaryLink`. Record the research date,
   canonical URL, pinned revision/version, and license beside the captured JSON. The point is not
   to copy bytes (the licenses and the language differ) - it is to learn the technique (the
   fused dequant tile, the online-softmax recurrence, the radix-tree reuse) so the fak version is an informed
   implementation, not a naive one. A web search for the named paper is fair game here, but it
   supplements the captured local catalog result rather than replacing it.
3. **Route honestly.** Pick `stay-minimal` (the bit-exact contract is fak's value, not raw
   throughput - most rows), `bind` (use the production library/format directly: cuBLAS fp16,
   the GGUF format), or `borrow` (adapt the reference technique). The hard rule: **borrow a
   kernel only after a witness for the current path exists**, so the choice is evidence-based
   and not a premature bet.
4. **Prove against the oracle.** The matrix row names the witness (almost always `cpuref` f32
   with a cosine floor, or bit-identity). A kernel with no recorded oracle floor is not done -
   it can read green on a CPU box while being wrong on the device.
5. **If the matrix had a blind spot, add the row.** When `fak sota` did not know the operation,
   or the sota-coverage scorecard flags an uncovered kernel file, add one `Op` to
   `internal/sotamatrix/sotamatrix.go`: the FileGlobs that should trigger the advisory, the
   verified fak path, the SOTA stack, a primary http(s) link, the route, and the oracle. This
   is the additive-leaf discipline - the next person inherits the reference you found.
6. **Stamp the commit.** End the kernel commit with a `Prior-art:` trailer naming what you
   consulted - e.g. `Prior-art: Marlin fused dequant-MMA (IST-DASLab/marlin); cosine >= 0.995
   vs HF AWQ`. This silences the advisory gate AND leaves a durable, greppable record.

## Portfolio output — best default plus bounded coverage

Do not collapse the matrix to one universal winner. Produce both (a) the best net-true default
for fak's common supported case and (b) a **bounded capability superset** view of useful
alternatives for named cohorts. Route each viable technique through `/field-borrow` as
`DEFAULT`, `OPTIONAL-MODULE`, `RECIPE`, `WATCH`, or `EXCLUDE`. A globally second-place approach
may still deserve an isolated module or recipe for a different accelerator, provider, privacy
constraint, deployment topology, compatibility requirement, cost envelope, or operator
preference. It must not displace the default merely to increase feature count.

Bound the exercise: require dated evidence, a supported cohort, a modular seam, incremental
support cost, and a disconfirming witness. Mark genuinely dominated, unsafe,
license-incompatible, out-of-scope, and support-uneconomic approaches `EXCLUDE`; mark volatile
or immature moment-in-time findings `WATCH` with a review trigger. This prevents both ego-driven
"not invented here" rejection and boil-the-ocean accumulation.

## What "done" proves

- `fak sota <op>` resolves the operation and you read its `PrimaryLink` before writing code.
- The kernel is held to the row's oracle (a recorded cosine floor / bit-identity), not just a
  "looks right" eyeball.
- The commit carries a `Prior-art:` trailer.
- If you added a matrix row, `python tools/sota_coverage_scorecard.py` reads clean (the new row
  has an existing fak-path file, a primary link, and an oracle) and the tree coverage is full.

## The anti-gaming rule (specific to this surface)

**Silence the PRIOR_ART advisory by actually checking prior art, never by stamping
`Prior-art: n/a` reflexively or muting the gate.** The trailer is a record of a reference you
read - if you genuinely could find no prior art for a truly novel contraction, say *that* in
the trailer (`Prior-art: none found - novel <X>; oracle cpuref cosine >= <f>`) so the claim is
falsifiable and the next person can disprove it by naming the art you missed. An empty `n/a`
on a GEMM that llama.cpp ships is the exact failure this skill exists to catch.

## Adding an operation to the matrix

When fak gains a new compute operation, add one `Op` row to
`internal/sotamatrix/sotamatrix.go` (keep the slice sorted by `Slug`). The sota-coverage
scorecard then holds it to the same discipline - the fak-path file must exist, the primary
link must be a real URL, the oracle must be named - and the PRIOR_ART gate starts surfacing
its reference whenever the matching kernel files are touched.

## ASCII gotcha

Keep the scorecard and Go source ASCII-friendly in comments and help strings (use ` - ` for a
dash, straight quotes). The matrix DATA legitimately carries UTF-8 (cosine symbols, deltas) in
its string fields; that is output content, not source scaffolding, and is fine.
