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
| **Runtime auto-select** | on a Metal-capable build, pick the GPU without `--metal`/`FAK_METAL` | **SHIPPED** (`dfe9de9b`, de-entangled from the #35 sibling) |

### Build-tag half — done

`881b7daf` flipped every Metal source file's constraint from
`//go:build darwin && cgo && fakmetal` to `//go:build darwin && arm64 && cgo`
(`internal/metalgemm/{metalgemm,decode,q4k,q8}.go`,
`internal/model/metal_{decode,prefill,prefill_hybrid,q4k}*.go`,
`internal/compute/metal*.go`). Effect: `CGO_ENABLED=1 go build` on Apple Silicon now
links Metal with **no special tag**, while `CGO_ENABLED=0` falls to the deterministic
stubs (`*_off.go` / `metalgemm_stub.go`) — the pure-Go artifact is unchanged. The
`-tags fakmetal` opt-in is gone from the build.

### Runtime auto-select half — shipped (`dfe9de9b`)

`cmd/fak/serve.go:resolveServeMetal` is rewritten so Metal **auto-selects whenever
`metalgemm.Available()`** (a linked backend + a usable device), and `--metal`/`FAK_METAL=1`
no longer *enable* Metal — they only change the *unavailable* case from silent CPU
fallback to fail-loud. `cmd/fak/serve_test.go` (`TestResolveServeMetal`,
`TestServeMetalFlagDefaultsFalse`) and `cmd/fak/servewiring.go` (the `metal` row →
`verdictWired`) accompany it. The resolver tests are pure-Go (they exercise the
`metalgemm` stub), so they gate on non-Metal CI too.

**Shipped de-entangled (2026-06-29, `dfe9de9b`).** The trio landed by staging only the
three `#62` hunks of `serve.go` via `git apply --cached --recount` — the sibling `#35`
admission-controller hunk (`SetAdmissionController(NewAdmissionController(...))`) and the
`cmd/modelbench/main.go` residency-bench feature were left uncommitted in the worktree, so
no sibling WIP was swept under the `#62` subject (the "land siblings first" step turned out
unnecessary). Verified GPU-free on the win32 box: `go build`/`go vet ./cmd/fak` clean,
`go test ./cmd/fak -run 'ResolveServeMetal|ServeMetalFlag'` both PASS.

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

## Status (2026-06-29) and the remaining residual

Both halves of #62's **wiring** are now on trunk: the build-tag flip in `881b7daf` and the
runtime auto-select in `dfe9de9b` (de-entangled via partial-hunk staging, so the `#35`
admission and `modelbench` siblings were *not* swept). What remains before the opt-in is
fully retired is the **parity gate**, not more code:

1. On the M3 Pro node, record the on-device witness — `CGO_ENABLED=1 go test ./cmd/fak`
   plus a `fak serve --gguf <dense-7B-Q8>` smoke confirming the GPU engages with no
   `--metal` (the host-independent `ResolveServeMetal|ServeMetalFlag` already PASS GPU-free).
2. Close the parity half (the 27B q4_k decode/prefill SOTA gap) via #64/#67/#69/#70/#71;
   until that gate is green, Metal-as-default is honestly claimed only for dense 7B Q8
   decode (#67, ~0.99× of bar). The q4_k parity remains hardware-gated on the Metal verify
   node — it does not block the auto-select wiring (now shipped), only the claim that Metal
   is at SOTA parity for the 27B q4_k headline.
