# Metal default-flip graduation gate (#62, epic #300, track #263)

When does the Apple-Silicon Metal path stop being an opt-in and become the default
*runtime* path? #62 ("auto-enable Metal on darwin/arm64, retire the `-tags fakmetal`
opt-in") is the **last** step of the GPU decode-parity playbook (#977 step 5): the
opt-in retires for a backend only once that backend reaches **measured parity** *and*
the #64 perf gate is **green** and has earned trust. This note pins the gate and records
the current verdict so the flip is a checkable event, not a judgment call.

## What "retire the opt-in" decomposes into

#62 has two independent halves. The pure-Go artifact (`CGO_ENABLED=0 go build`) must stay
byte-unchanged either way (DIRECTION.md rule 1).

| Half | What it means | State |
|---|---|---|
| **Build-tag opt-in** | drop the extra `-tags fakmetal`; gate Metal on `darwin && arm64 && cgo` | **RETIRED + committed** (`881b7daf`) |
| **Runtime auto-select** | on a Metal-capable build, pick the GPU without `--metal`/`FAK_METAL` | **coded, uncommitted, blocked** (see below) |

### Build-tag half — done

`881b7daf` flipped every Metal source file's constraint from
`//go:build darwin && cgo && fakmetal` to `//go:build darwin && arm64 && cgo`
(`internal/metalgemm/{metalgemm,decode,q4k,q8}.go`,
`internal/model/metal_{decode,prefill,prefill_hybrid,q4k}*.go`,
`internal/compute/metal*.go`). Effect: `CGO_ENABLED=1 go build` on Apple Silicon now
links Metal with **no special tag**, while `CGO_ENABLED=0` falls to the deterministic
stubs (`*_off.go` / `metalgemm_stub.go`) — the pure-Go artifact is unchanged. The
`-tags fakmetal` opt-in is gone from the build.

### Runtime auto-select half — coded but not cleanly committable

`cmd/fak/serve.go:resolveServeMetal` is rewritten so Metal **auto-selects whenever
`metalgemm.Available()`** (a linked backend + a usable device), and `--metal`/`FAK_METAL=1`
no longer *enable* Metal — they only change the *unavailable* case from silent CPU
fallback to fail-loud. `cmd/fak/serve_test.go` (`TestResolveServeMetal`,
`TestServeMetalFlagDefaultsFalse`) and `cmd/fak/servewiring.go` (the `metal` row →
`verdictWired`) accompany it. The resolver tests are pure-Go (they exercise the
`metalgemm` stub), so they gate on non-Metal CI too.

**Blocker (why it is not shipped here):** these edits sit uncommitted in a shared
multi-session tree, and the working-tree copies are *entangled* with unrelated in-flight
work — `serve.go` also carries a `#35` admission-controller hunk
(`SetAdmissionController(NewAdmissionController(...))`) and `cmd/modelbench/main.go`
carries an unrelated metal-residency-bench feature. A whole-file `git commit -- <path>`
would sweep that sibling WIP under a `#62` subject, which the trunk discipline forbids;
partial-hunk staging is not safe on this tree. The de-entangled commit is the smallest
next step (below).

## Graduation verdict — per backend (the parity half)

The auto-select wiring is *correct* (it picks Metal when present; a model the GPU path
cannot help self-declines to CPU), but the owner ties the **full** retirement to measured
SOTA parity. Current state on the M3 Pro verify node:

| backend / phase | fak-Metal | llama.cpp-Metal bar (observed-external) | ratio | gate |
|---|---:|---:|---:|---|
| dense Qwen2.5-**7B Q8** decode (#67) | ~0.99× of bar (cos 0.999991) | — | **~0.99×** | parity candidate |
| Qwen3.6-**27B q4_k** decode (#64/#68/#70) | 1.2 tok/s | 7.29 tok/s | **0.16×** | **#64 FAIL** (fail-closed, expected) |
| Qwen3.6-27B q4_k prefill (#63/#71) | warm ≪ bar (pp ~0.6 @P29) | 51.55 tok/s @pp22 | ≪0.5× | not asserted |

- The **dense 7B Q8 decode** lane (#67) is the only backend at parity — the first
  candidate the owner names to graduate.
- The headline **27B q4_k** decode/prefill the SOTA-gap table tracks is **far under** the
  0.5× bar; #64 stays RED by design until the GPU-resident decode + device-GEMM work
  (#67/#69/#70/#71) closes the launch-bound wall.

So the auto-select is safe to ship (it never *worsens* a run — Metal already beats fak's
own CPU lane, and unhelpable models decline), but #62's "make Metal the default once
parity holds" is, today, **true only for dense 7B Q8 decode**; the 27B q4_k story it cites
is `not yet`.

Bar provenance: the llama.cpp-Metal numbers (7.29 decode / 51.55 prefill) are
**observed-external**, provenance-caveated per #459 (build/flags) and #452 (conditions) —
relayed, not a fak witness. See `metal-perf-gate-arm-2026-06-26.md` and
`qwen36-perf-gate-metal-20260626.md`.

## Smallest next step to close #62

1. Land the entangled siblings first (`#35` admission wiring in `serve.go`; the
   residency-bench feature in `modelbench/main.go`) so the #62 hunks stand alone.
2. Commit the de-entangled trio by explicit path —
   `cmd/fak/serve.go` (resolveServeMetal auto-select), `cmd/fak/serve_test.go`,
   `cmd/fak/servewiring.go` — as `feat(serve): auto-enable Metal on darwin/arm64 (#62) (fak cmd)`.
   The leaf is `cmd` (the code lives under `cmd/`), not `experiments`.
3. Gate: `go test ./cmd/fak -run 'ResolveServeMetal|ServeMetalFlag' -count=1` (pure-Go,
   passes on non-Metal CI); on the M3 Pro node, `CGO_ENABLED=1 go test ./cmd/fak` plus a
   `fak serve --gguf <dense-7B-Q8>` smoke confirming the GPU engages with no `--metal`.

The parity-gate half (q4_k) remains hardware-gated on the Metal verify node and is tracked
by #64/#67/#69/#70/#71 — it does not block the dense-Q8 auto-select wiring, only the claim
that Metal is at SOTA parity for the 27B q4_k headline.
