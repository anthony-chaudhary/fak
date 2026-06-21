# Race detector (`go test -race`) — requirements & race-free guarantee

> Tracking issue: [E-001 · Go Race Detector Support](https://github.com/anthony-chaudhary/fak/issues/12).

The `fak` module is concurrent in its hot path — the context-MMU, the radix KV
cache, the gateway, and the decode loop all run under multiple goroutines. Go's
[data-race detector](https://go.dev/doc/articles/race_detector) is the only thing
that turns a benign-looking concurrent read/write into a hard, reproducible
failure, so it is a first-class part of this module's test story.

## The guarantee

**The full `fak` suite passes under `-race` with zero data races**, and CI
enforces it on every push and pull request (the `race detector · go test -race
./...` job in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)). A
change that introduces a data race in any package — hot path or not — fails that
required check.

Packages with explicit concurrency/leak coverage that the `-race` run exercises:

| Package | What the race run guards |
|---|---|
| `internal/ctxmmu` | context-MMU page table + ledger under concurrent map/unmap (`leak_test.go`) |
| `internal/radixkv` | radix KV prefix cache under concurrent insert/lookup/evict (`leak_test.go`) |
| `internal/gateway` | request fan-in / streaming under concurrent sessions (`leak_test.go`) |
| `internal/model` | decode-loop allocation + KV access (`decode_alloc_test.go`) |

The whole module is run under `-race`, not just these — they are simply the
packages where concurrency is the point of the test.

## Requirement: cgo + a C compiler

The race detector is implemented in C (ThreadSanitizer) and **requires cgo**
(`CGO_ENABLED=1`) plus a working C compiler (`gcc`/`clang`). Without them, `go
test -race` cannot build an instrumented binary.

> ⚠️ **Important Windows caveat.** The canonical Windows dev host runs with
> `CGO_ENABLED=0` and **no `gcc`/`clang` installed**, so `-race` cannot run
> natively there. Worse, a `-race` build on a cgo-less toolchain does not error
> in an obvious way — it just fails to produce an instrumented binary. Run the
> race detector in one of the cgo-capable environments below instead.

## Running it locally

> **The one-command path:** [`tools/race_test.sh`](../../tools/race_test.sh)
> wraps the run with a **cgo preflight** — it proves a C compiler is present and
> forces `CGO_ENABLED=1` *before* building, then delegates to `fak/test.sh` with
> the CI-matching flags (`-race -count=1 -timeout=25m ./...`). If cgo is missing
> it refuses with a clear message (exit `2`) instead of silently building a
> race-*blind* binary and a false green. Use `tools/race_test.sh --check` to run
> only the preflight. The manual `go test -race` forms below are equivalent on a
> cgo-capable host.

### Linux / macOS (cgo on by default)

```bash
go test -race ./...          # whole module
go test -race ./internal/ctxmmu/   # one package
```

macOS uses the Xcode `clang`; most Linux distros ship `gcc`. Nothing else is
needed — `CGO_ENABLED` defaults to `1` on these platforms.

### Windows via WSL (the canonical local path on the dev box)

The repo's normal Go test runner already shells into WSL (see
[`fak/test.sh`](../../fak/test.sh)) because native Windows test execution is
blocked by an Application Control policy. The same WSL distro provides the cgo
toolchain the race detector needs:

```bash
# from a WSL Ubuntu shell with gcc installed (apt install build-essential)
cd /mnt/c/.../fleet-public/fak
CGO_ENABLED=1 GOTOOLCHAIN=auto go test -race -count=1 ./...
```

`GOTOOLCHAIN=auto` lets the distro's older `go` fetch the version pinned in
`fak/go.mod`. The source lives on the `/mnt/c` 9p mount, so the first run pays an
enumerate/compile tax; `FAK_FAST=1 bash ./test.sh -race ./...` mirrors the
sources onto ext4 for a faster inner loop.

### CI (the durable witness)

The `race-detector` job in `.github/workflows/ci.yml` runs on `ubuntu-latest`,
which ships `gcc` and defaults `CGO_ENABLED=1`. It runs `go test -race -count=1
-timeout=25m ./...` — `-count=1` forces a real, uncached run so a green check is
always a *fresh* read-back of the race-free claim, never a stale cache hit.
`CGO_ENABLED` is set explicitly in the job so it fails loudly if it ever lands on
a cgo-less image rather than silently building a race-blind binary.

> **Why `-timeout=25m`.** ThreadSanitizer adds ~5-10x CPU overhead, which pushes
> compute-bound packages (notably `internal/turnbench`'s grid/sweep tests) past
> the default 10m per-package binary timeout. The longer timeout only grants
> wall-clock — it does not change which tests run or relax any assertion. Use the
> same `-timeout` when running `-race` locally.

## Maintaining the guarantee

- New concurrent code: add a test that actually exercises the concurrency (spawn
  goroutines, hit the shared state from several at once). The detector only sees
  races on code paths a test drives.
- A flaky race finding is still a real finding — the detector reports on
  *observed* interleavings, so re-run to reproduce, but never paper over it with
  a retry.
- If `cc --version` fails in the CI job, the runner image lost its C toolchain —
  fix the toolchain, do not drop `-race`.
