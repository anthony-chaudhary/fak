# Path-portable Go verification builds: zero cross-root compiles experimentally, link remains

Observed at `2026-08-28T04:02:09Z` for [#9661](https://github.com/anthony-chaudhary/fak/issues/9661), with the cgo follow-up observed at `2026-08-28T04:40:23Z` for [#9671](https://github.com/anthony-chaudhary/fak/issues/9671) and the Make link census at `2026-08-28T05:03:26Z` for [#9672](https://github.com/anthony-chaudhary/fak/issues/9672). The first source event is detached worker commit `ee6f393f419042c130a1b4b08a69883a962bc3f8`; the cgo follow-up was first measured as an experimental diff over `0f36db3068126dc8c70d4a6956fc71639e82d912` and landed at `e1df7c2cc`; the Make split was measured over `b262395b602bfe67d6bd9183cf405457b91bf866`, which contains that cgo fix. The runtime is Go 1.26.6 on Darwin/arm64 with cgo enabled. Refresh this note when the Go toolchain changes, an isolated verification seam changes its build flags, or the matched Windows/amd64 benchmark is rerun.

## Verdict

FAK's isolated verification commands materialize identical source under a different absolute directory on each run. Without `-trimpath`, Go 1.26.6 hashes that package directory into the compile action ID. A shared `GOCACHE` therefore cannot reuse most repository packages across those roots. Adding `-trimpath` at the verification-only `buildcheck`, `ci-preflight`, and `validate` build/run/vet/test seams made the second fresh-root build **2.58× faster** (`23.59s` → `9.14s`) and reduced executed compile actions from **590 to 1**. That is 589 of 590 recompiles avoided.

This is a useful measured improvement, not the requested 10× result. The original one surviving compile was `internal/compute`'s cgo/Metal package. The #9671 experiment below removes that action without disabling Metal; every targeted sample still links the 85 MB runtime. The #9672 Make census then separates the named developer artifacts from 147 unrelated whole-tree command links. These mechanism falsifiers do not satisfy #9661's required 5+5 matched Windows/amd64 completion matrix.

The implementation deliberately does not change `scripts/build.sh`, the debuggable `make build` profile flags, CI `GOFLAGS`, or released artifacts. Debuggable builds retain host paths; only disposable verification roots opt into path normalization. The #9672 split changes which target owns the whole-tree sweep, while `make ci` retains both `build` and `build-all` as direct prerequisites.

## Alternating fresh-root witness

The complete machine-readable receipt is [`docs/_witnesses/build10x/path-cache-cross-root-2026-08-27.json`](../_witnesses/build10x/path-cache-cross-root-2026-08-27.json). Each sample extracted the same committed `git archive HEAD` into a never-before-used root. Control and treatment used separate, initially empty cache directories; each cache was shared only with the second same-arm root. `go build -x` supplied the executed compile/link counts.

| Order | Arm | Cache state | Real | Compile | Link |
|---:|---|---|---:|---:|---:|
| 1 | control, no `-trimpath` | empty | 26.00s | 840 | 1 |
| 2 | treatment, `-trimpath` | empty | 27.31s | 840 | 1 |
| 3 | control, new root | control cache populated | 23.59s | 590 | 1 |
| 4 | treatment, new root | treatment cache populated | 9.14s | 1 | 1 |

The cold pair is intentionally flat: path normalization cannot create a hit before an artifact exists. The second-root pair is the mechanism test. The control still compiled 590 actions because their absolute source directories moved; treatment reused all but the cgo/Metal action. This action-count reversal is stronger evidence than wall time alone and makes the residual explicit.

Both arms then produced an artifact and ran:

```text
<artifact> preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args '{}'
verdict=DENY reason=POLICY_BLOCK by=monitor
```

The behavior outputs were byte-identical. Binary hashes were intentionally different because `-trimpath` changes debug/build metadata; the receipt records both sizes and SHA-256 digests rather than pretending byte identity is the behavior contract.

## Cgo follow-up: the last compile action

The machine-readable #9671 receipt is [`docs/_witnesses/build10x/cgo-path-cache-cross-root-2026-08-28.json`](../_witnesses/build10x/cgo-path-cache-cross-root-2026-08-28.json). It compares committed `0f36db306` with the same tree plus one production-line deletion: the redundant `#cgo CFLAGS: -I${SRCDIR}` in `internal/compute/metal.go`. Go's cgo documentation guarantees that the source directory is already on the compiler include path. The explicit directive changed no header lookup, but `${SRCDIR}` expanded to each checkout's absolute path and entered the build-action hash.

The action-key trace made the cause direct:

| Arm | Root A compute action | Root B compute action | Source-declared CFLAGS |
|---|---|---|---|
| committed baseline | `d2886bd3…` | `8bc74e1f…` | `-O2 -g -I<absolute-root>/internal/compute` |
| #9671 treatment | `cc8fc799…` | `cc8fc799…` | `-O2 -g` |

With separate initially empty caches for baseline and treatment, each second root reused only its own first root:

| Sample | Cache state | Real | Compile | Link |
|---|---|---:|---:|---:|
| baseline root A | empty | 27.34s | 840 | 1 |
| baseline root B | populated across roots | 12.15s | 1 | 1 |
| treatment root A | empty | 25.17s | 840 | 1 |
| treatment root B | populated across roots | 6.43s | 0 | 1 |

The treatment removed the last cross-root compile and cut this second-root sample by **1.89×** (`12.15s` → `6.43s`). A third treatment root changed one comment in `internal/compute/metal_shim.m`; its compute action changed to `44b330b1…` and 11 package compiles executed, proving native-source changes still invalidate `internal/compute` and its dependent chain.

Both treatment roots produced byte-identical 85,260,210-byte artifacts (`sha256:2dc0d6fb…`). `go tool nm` retained `_fmetal_init` and `_mg_init`, and both artifacts returned `verdict=DENY reason=POLICY_BLOCK by=monitor` for the policy smoke. The Darwin/arm64/cgo tags, Objective-C source, Metal frameworks, shared `metalgemm` device seam, and runtime selection remain unchanged.

The receipt remains labeled **experimental evidence** because it was captured from an uncommitted diff over `0f36db306`. The one-line treatment subsequently landed at `e1df7c2cc`; the #9672 link census below runs on pinned `b262395b6`, which contains it. The cgo receipt's dedicated committed-tip test reconciliation remains part of #9671's closure record rather than being silently rewritten as pre-landing evidence.

## Make follow-up: named artifacts versus 147 command links

The machine-readable #9672 receipt is [`docs/_witnesses/build10x/make-build-link-census-2026-08-28.json`](../_witnesses/build10x/make-build-link-census-2026-08-28.json). It compares the former `make build` graph with the same pinned source after moving only `go build ./...` to an explicit `build-all` target. Both arms emitted the same four real outputs: debuggable `fak`, native `repoguard`, Windows `repoguard.exe`, and native `dispatchworker`. Each arm had a separate initially empty `GOCACHE`; its warm sample retained the exact named output paths so Go could validate their embedded build IDs.

| Sample | Cache / output state | Real | Compile | Cgo | Link |
|---|---|---:|---:|---:|---:|
| former build graph, cold | empty / absent | 53.62s | 1,330 | 14 | 151 |
| split build graph, cold | empty / absent | 31.14s | 947 | 14 | 4 |
| former build graph, warm | populated / retained | 15.83s | 0 | 0 | 147 |
| split build graph, warm | populated / retained | 1.26s | 0 | 0 | 0 |

The cold sample removed 147 links and ran **1.72× faster**. The warm sample removed the same 147 links and ran **12.56× faster**. This is not a linker optimization: `go build ./...` links every main package to a disposable output on every invocation, whereas Go can validate and reuse the named artifacts. Baseline and treatment produced byte-identical copies of all four outputs. The native guard self-test passed 41/41, dispatchworker emitted a guarded dry-run packet, the Windows hook remained a Windows/amd64 PE, and `fak` retained the policy verdict `DENY reason=POLICY_BLOCK by=monitor`.

The contract stays explicit. `make build` emits the runtime and hook artifacts. `make build-all` owns `go build ./...`. `make ci` names both as direct prerequisites, so the full tree still compiles and all commands still link before a green-bar claim. The existing `tools/build_entrypoint_test.py` locks this graph beside the release/profile anti-drift tests.

These timings are a Darwin/arm64 diagnostic with one cold and one warm sample per arm. They do not replace #9673's five-cold/five-warm Windows/amd64 completion matrix, and the 1.72× cold result is not itself #9661's 10× result.

## Official source ledger

### Go 1.26.6 — shipped mechanism

- **Source state:** released as `go1.26.6`, commit [`1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e`](https://github.com/golang/go/tree/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e), source event `2026-08-13T17:25:07Z`; observed `2026-08-28T04:02:09Z`.
- **Action identity:** [`src/cmd/go/internal/work/exec.go:260-310@1ea5a71`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/src/cmd/go/internal/work/exec.go#L260-L310) hashes `dir <absolute package directory>` when `-trimpath` is absent and records the stable `trimpath` mode when it is present. This is the exact cause exercised here.
- **Cgo source directory:** [`src/cmd/cgo/doc.go:87-116@1ea5a71`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/src/cmd/cgo/doc.go#L87-L116) defines `${SRCDIR}` as the absolute source directory and separately guarantees that the C compiler already receives that directory as an implied include path. This makes `#cgo CFLAGS: -I${SRCDIR}` redundant for the colocated `metal_backend.h` while explaining its path-bearing action input.
- **Compiler rewrite:** [`src/cmd/go/internal/work/gc.go:136-176@1ea5a71`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/src/cmd/go/internal/work/gc.go#L136-L176) passes the computed rewrite to the compiler while compiling absolute file inputs.
- **Cache semantics:** [`src/cmd/go/internal/work/buildid.go:26-65@1ea5a71`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/src/cmd/go/internal/work/buildid.go#L26-L65) defines action ID as a hash of inputs and content ID as a hash of output; [`buildid.go:424-482`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/src/cmd/go/internal/work/buildid.go#L424-L482) uses that action ID to query the artifact cache.
- **License:** Go 1.26.6's exact-revision [`LICENSE`](https://github.com/golang/go/blob/1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e/LICENSE) is BSD-3-Clause. Disposition: **ADAPT** — fak invokes the shipped flag and adapts its path-stability contract at existing command seams; no Go source or comments were copied.
- **Refresh trigger:** a Go toolchain update or any change to `buildActionID`/`Action.trimpath`.

### Bazel 9.2.0 — diagnostic doctrine

- **Source state:** released as `9.2.0`, commit [`8220c6198837d5c13d53fea211cf3282aa12408a`](https://github.com/bazelbuild/bazel/tree/8220c6198837d5c13d53fea211cf3282aa12408a), source event `2026-07-13T18:10:42Z`; observed `2026-08-28T04:02:09Z`.
- **Cross-machine cache test:** [`site/en/remote/cache-remote.md:37-147@8220c61`](https://github.com/bazelbuild/bazel/blob/8220c6198837d5c13d53fea211cf3282aa12408a/site/en/remote/cache-remote.md#L37-L147) says repeated actions need identical keys, prescribes comparing execution logs, and names host-environment leakage as a cross-machine miss cause. FAK adapted that doctrine into alternating roots plus `go build -x` action counts.
- **Cache model:** [`site/en/remote/caching.md:42-64@8220c61`](https://github.com/bazelbuild/bazel/blob/8220c6198837d5c13d53fea211cf3282aa12408a/site/en/remote/caching.md#L42-L64) describes declared action inputs, cache lookup, execution on miss, and upload after execution.
- **License:** Bazel 9.2.0's exact-revision [`LICENSE`](https://github.com/bazelbuild/bazel/blob/8220c6198837d5c13d53fea211cf3282aa12408a/LICENSE) is Apache-2.0. Disposition: **INSPIRE-ONLY** — only the diagnostic principle was used; no Bazel code, test, prose, or configuration was copied.
- **Refresh trigger:** a Bazel remote-cache documentation or execution-log protocol change.

No upstream implementation was ported, so neither source adds attribution text to fak binaries. The sources establish why the experiment is sound; the local action trace establishes that it applies here.

## FAK self-query and seam

Durable study search returned no prior record for `Go build trimpath shared cache absolute source path Bazel`. Three capability queries (`Go build trimpath shared cache`, `cross-root Go build cache reuse`, and `path portable isolated verification build`) returned only prompt/context caching or session persistence, not build artifacts. The dev index was unavailable through the installed runtime because `index` has moved to `fak-dev`; raw symbol search found the existing isolated command seams and no path-portable cache contract.

Disposition: **PARTIAL**. FAK already had the right isolated verification roots and a shared Go cache, but their argv admitted absolute-root identity. The exact seams are:

- `internal/devcmd/buildcheck.go` — `buildCheckArgs` for isolated build/vet;
- `internal/devcmd/ci_preflight.go` — committed-tip `go run` generated check and `go build ./...`;
- `cmd/fak/validate.go` — affected build/vet/test inside a freshly materialized checkout.

Portfolio disposition: **DEFAULT** for these disposable verification seams; **EXCLUDE** for debuggable developer builds because source-path fidelity is their declared job. Negative-control tests inspect exact argv for build, run, vet, and test, so a global environment flag cannot silently widen the scope.

## Next falsifier

Keep #9661 open. Verify #9672 at its landed commit, then run #9673's five-cold/five-warm matched Windows/amd64 matrix on top of landed #9671 and #9672. If cross-root compile actions remain zero but elapsed time stays above 6,555 ms, the named executable links or host artifact scan are the evidenced next targets. If compile actions return, diff the action inputs before changing more code.

Companions: [#9661](https://github.com/anthony-chaudhary/fak/issues/9661) · [#9671](https://github.com/anthony-chaudhary/fak/issues/9671) · [#9672](https://github.com/anthony-chaudhary/fak/issues/9672) · [runtime/dev split #6019](https://github.com/anthony-chaudhary/fak/issues/6019) · [`docs/dev-tooling.md`](../dev-tooling.md)
