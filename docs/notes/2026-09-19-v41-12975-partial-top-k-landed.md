# 2026-09-19 — V4.1 native track: promoted the O(E log k) partial top-k into the V4 router seam (#12975), landed and closed

Lane: `model` — `fak-flow lease live --lane model --repo-root ../fak` => `held=false total=0 live=0
reapable=0`. No durable lease was persisted (the arbitration was a pure decision); nothing to release.
Landed: **`1bb14db849da6acf160b89942673082e1c723190`**, pushed to `origin/main`.
Parent epic: #12640 (OPEN — dispatch parent). V4 Flash 0731 control (@ `7872f01b`, epic #12635) untouched.

This is the ledger counterpart to the private `docs/tickets/deepseek-native/` receipts. It is
recorded in the public tree because the work is public engine code and this session owns the
public checkout; the private ledger write is blocked by the same lease-boundary guard documented
in the `W-CROSSREPO` receipts.

## What shipped

The opt-in `SetV4BitonicTopK` seam delegated to `compute.PersistentBitonicTopK`, which is **not** an
O(E log k) partial selection: it is a full bitonic sorting network that pads E to the next power of
two and sorts every slot. At the admitted V4.1 router geometry (E=384, top-6) it padded 384 -> 512
and ran ~11520 compare-exchanges against the reference stable sort's ~3300, measuring **~1.85x
slower** than the reference it was meant to beat. No production path carried a genuine partial
selection.

`internal/model/v4_topk_partial.go` promotes `v4PartialTopKIndices`, a real O(E log k) bounded
min-heap top-k, out of the benchmark-only scratch. `v4TopKIndices` routes its flagged arm through
it, keeping the reference stable sort as the **default** path (byte-identical when the flag is
undeclared) and as the **fail-closed fallback** on any malformed kernel result. The pinned
tie-break (descending score, lower expert index first) is preserved exactly, and degenerate k
(`k<=0` or `k>len`) fails closed to the full reference width. The duplicate benchmark-only
implementation was retired to a thin alias of the production kernel so the cost benchmark cannot
drift from what the router selects.

## Witness

Deterministic RED -> GREEN (`TestV4PartialTopKFlagSelectsBoundedKernel`), measured by allocation
bytes via `testing.Benchmark`'s `AllocedBytesPerOp` so host noise cannot flip it:

```text
RED (old bitonic delegation):  flagged top-k allocates 4192 B/op, want < 1000  => FAIL
GREEN (bounded heap):          flagged top-k allocates 152 B/op                => PASS
```

The kernels are semantically equivalent (both return the correct top-k), so only their WORK
distinguishes them; that is why the witness pins the allocation footprint rather than a value.

Speedup at E=384, k=6 (`BenchmarkV4TopKIndicesE384` / `BenchmarkV41RouterTopKCostE384`):

```text
reference_stable_sort          ~19785 ns/op
partial_select_kernel           ~569 ns/op   (~35x faster than the reference)
bitonic_full_sort_retired     ~36570 ns/op   (~1.85x slower than the reference)
```

```text
go build ./internal/model/                                     => exit 0
go test ./internal/model/ -count=1                             => ok (169.675s)
go vet ./internal/model                                        => exit 0
gofmt -l <4 files>                                             => clean
fak-boundary check --staged                                    => PASS (0 violations)
```

Provenance (declared single-provider vertical pass; nested subagents forbidden by the lane):

```text
Separation-Verdict: SINGLE_PROVIDER
Single-Provider: model
```

## Landing and lifecycle

- `fak-flow land 12975 --repo-root ../fak --verify "go test ./internal/model/ -run V4PartialTopKFlagSelectsBoundedKernel -count=1"`
  => `{"ok":true,"code":"success","applied":true,"committed":true}`, commit `1bb14db849da`.
- `fak sync push` => `pushed main -> origin/main`; `origin/main == 1bb14db849da` verified.
- #12975 CLOSED (completed) with the landed SHA and witness comment.
- Worktree `ticket-12975` removed and branch `fak/ticket-12975` deleted after verifying the content
  was identical to trunk (`git diff origin/main -- internal/model/v4_topk_partial.go` empty).

## Session flag filed: the Strix trigger over-fires on pure-CPU model paths

`fak-flow land --verify go-build` was refused at the scoped ownership check over this pure-CPU
model leaf:

```text
fak validate --mine internal/model/v4_topk_partial.go ... --json
  => ok=false; phases: gofmt ok, build ok, vet ok, test ok, strix_validation failed
  => failures: strix-controller-authority: UNSUPPORTED_VALIDATOR_BINARY
```

`internal/model` is listed wholesale in `gpuRoots` (`cmd/fak/validate_strix.go:20`), so any Go
change in the tree's largest package demands physical Strix silicon on a host that cannot provide
a stamped WSL validator binary. The land fell back to the documented custom-verify route
(`--verify "<test cmd>"`), which ran the issue's own witness command green before landing; no gate
was weakened. Filed as **#13293**, including the companion `fak-flow` defect:
`scopedValidatePhaseIsWitness` lists `"strix"` while the emitted phase is `"strix_validation"`, and
`classifyScopedValidateOutput` fails a land closed on the host-capability phase even when
build/vet/test are green. (The misplaced private copy, fak-private#2162, was closed as such.)

## Next frontier

- **Software (bounded):** the V4.1-native bounded software set is drained again at trunk
  `1bb14db84`. #13225 stays over bound.
- **Physical (not a bounded code leaf):** #13288 / #13216 residues are `[HW-WITNESSED]` physical
  boxes on the strix3 native serve warmup under the pinned vcruz Q2_K `58d8ac86` — the
  `mission-native` hardware lane.

No native V4.1 generation or physical parity claimed. #12640 stays OPEN. No `[HW-WITNESSED]`
criterion touched.
