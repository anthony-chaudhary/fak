---
title: "Developer Tooling — debug, profile, and test fak"
description: "The hands-on developer-tooling guide for fak: build and run, the test runner (make + WSL), debugging with fak debug and fak doctor, profiling and benchmarking, and the commit-and-ship dev loop."
---

# Developer tooling: debug, profile, test

This is the hands-on guide to the CLI tools you use while *working on* fak —
debugging, profiling, and testing — plus the dev loop they sit inside. It is the
practitioner companion to the navigational [Work map](WORK-MAP.md) (which routes a
task to the right front door) and the verb-by-verb [CLI reference](cli-reference.md)
(which lists every `fak` verb). Read [`AGENTS.md`](../AGENTS.md) first for the build
commands and the hard rules; this page is the "now I'm in the loop, what do I run?"
layer.

> **Honest scope.** `fak test` ships (a host-aware runner over `go test`); a
> dedicated `fak profile` verb does not yet. The capabilities each wraps already
> exist as the Go toolchain plus a few `fak`/`make` verbs, and this guide shows the
> real commands. The
> [What ships vs. what's planned](#what-ships-vs-whats-planned) table at the end is
> explicit about which is which, so you never reach for a verb that isn't there.

## Build and run

The Go module is the repository root, so every `go` command runs from the clone root.

```bash
go build -o fak ./cmd/fak     # -> ./fak  (fak.exe on Windows); ~30-60s cold, instant warm
./fak --help                  # every verb
./fak doctor --help           # the read-only diagnostic (below)
```

The 60-second, no-key/no-model/no-GPU proof is the canonical first run — see
[`AGENTS.md`](../AGENTS.md) and the full [repro packet](repro-packet.md).

## The test runner

`fak test` is the host-aware runner: it resolves the right `go test` invocation for
the tier you ask for and, on Windows, routes it through `test.ps1` (WSL) automatically
so you never hit the OS-policy block below. The `make` target set is the authoritative
gate it sits over; `fak test --list` prints the tiers, and `fak test -n <tier>` prints
the resolved command without running it. It sits over the `make` target set — the
authoritative gates — with one host caveat that bites on Windows.

| Command | What it runs | When |
|---|---|---|
| `fak test [fast\|full\|race\|<pkg>]` | the host-aware wrapper over `go test` (default tier `fast`); on Windows routes to WSL via `test.ps1`; `fak test fast -- -run TestX` passes flags through | the one-verb inner loop over the targets below |
| `make test-fast` | `build` + `vet` + `go test -short ./...` (~2s smoke tier; skips the weight-backed model witnesses) | the pre-commit / pre-push floor — ~95% of logic regressions in seconds |
| `make test` | `go test ./...` (full suite incl. the ~538 MB f32/safetensors model oracle) | the authoritative gate before you trust a model-touching change |
| `make test-affected` | `fak affected` → `go test` for only the packages your working-tree change can reach (changed + transitive importers, test imports included) | the fast inner loop on the REAL oracle (no `-short`) for a one-leaf edit |
| `make test-race` | `CGO_ENABLED=1 go test -short -race ./...`, cgo-preflighted (refuses on a compiler-less box rather than building a race-blind false green) | catch a data race locally instead of minutes later in CI — see [testing/race-detector.md](testing/race-detector.md) |
| `make ci` | the full gate: `build` + `gofmt-check` + `vet` + `test` + `claims-lint` + the doc/scorecard gates | the green-bar definition the guards expect before you ship |

For a single package, `go test ./internal/<pkg>/... -count=1` is the direct form
(`-count=1` defeats the test cache when you want a clean re-run).

> **Windows host caveat.** Native `go build` / `go vet` / `go run` work, but native
> `go test` is blocked by an OS Application-Control policy on the freshly-compiled
> test binaries. Run the suite under WSL with `./test.ps1` from the repo root (it
> shells the same `go test` inside WSL and defaults to the ext4 mirror fast path,
> `FAK_FAST=1`, so test source enumeration does not run from slow `/mnt/c` drvfs).
> This is an OS quirk, not a code failure; `fak affected` and every `make test*`
> target above inherit the same "run under WSL on this box" contract. See
> [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md).

## Debugging

Two read-only diagnostics ship today, plus the integration-level "why was my call
denied?" guide.

### `fak debug` — the context debugger

`fak debug` attaches to a *finished* session as if to a core dump and answers a
follow-up by demand-paging only the working set the question touches, instead of
replaying the whole transcript. It is a context/session debugger, not a source-level
step debugger.

```bash
fak debug --list                                  # discover real Claude Code transcripts on this box; prints the command to attach each
fak debug --session <path/to/session.jsonl>       # ingest a real transcript as a core image
fak debug --cmd report --query "what did X do?"    # demand-page the working set for one follow-up, emit cdb-report.json
fak debug                                          # no --session: hermetic demo over the committed synthetic fixture
```

Sub-commands (`--cmd`): `report` · `html` · `info` · `bt` · `x` · `ws` · `grep` ·
`tombstone` · `context-query` · `context-diff`. With no `--session` it runs the
committed demo fixture and says so on stderr. The measured behaviour (an 18 KB page
table over a 1.2 MB swap device, follow-ups paging in ~1.8–6.2% of the resident
image) is written up in [benchmarks/CDB-RESULTS.md](benchmarks/CDB-RESULTS.md).

### `fak doctor` — the answer-shape diagnostic

`fak doctor` is a read-only operator diagnostic: it runs the degeneration/verbosity
witness over a candidate answer and cross-checks the real kernel admit verdict the
context-MMU would reach on the same bytes, then prints the recommended action per
finding. Exit `0` = healthy, `1` = at least one finding, `2` = usage error, so it
also composes as a CI gate over a captured answer.

### Debugging a denied tool call

When the kernel denies, repairs, or quarantines a call and you need to know why, the
integration guide [integrations/debugging.md](integrations/debugging.md) walks the
verdict surface and the audit log.

## Profiling and benchmarking

There is no `fak profile` verb. Profiling fak is standard Go profiling plus the
benchmark verbs the repo already ships.

### Go pprof (CPU, memory, blocking)

The kernel is a Go binary, so the Go toolchain's profilers apply directly. Profile a
hot package through its benchmarks:

```bash
# CPU + allocation profile for one package's benchmarks (run under WSL on Windows)
go test -run=NONE -bench=. -benchmem \
        -cpuprofile cpu.out -memprofile mem.out ./internal/<pkg>/...

go tool pprof -top cpu.out          # hottest functions
go tool pprof -http=:0 cpu.out      # interactive flame graph in a browser
```

`-benchmem` reports allocations/op, the number to drive toward zero on a hot-path
change (the screening gates and the decode meter are held at a green allocation
budget by their tests). `go tool pprof` also reads a `--cpuprofile` captured from a
live `fak serve` if you wire `net/http/pprof` for an ops investigation.

### The benchmark verbs

| Command | What it does |
|---|---|
| `fak benchmarks list [--offline] [--json]` | the single discoverable index of every benchmark fak ships — what each measures and its cold-start cost (`--offline` = zero-asset only) |
| `fak benchmarks describe <name>` | one benchmark's purpose, run command, key flags, and doc |
| `fak benchmarks run <name> [-- extra args]` | run it (prints the resolved command; runs the `cmd/*bench` benches via `go run`) |
| `fak bench --suite <suite> --out report.json` | run a benchmark suite directly (`make bench` runs the `tau2-smoke` suite) |
| `fak ablate` | the self-ablation sweep — turn one feature off and measure the delta, to prove a gain is net-true |

Every perf number is held to the [net-true-value standard](../EXTENDING.md): measured
against the real (tuned, not naive) alternative, net of its own cost, scope stated,
provenance-labeled, reproducible. A profile that isn't reproducible is `not yet`, not
a result.

## The dev loop (commit and ship)

The tooling above feeds one loop: build -> test -> commit-by-path -> ship. The rules
below are enforced *below* the agent layer (git hooks refuse a violation), so they
are verbs, not etiquette. A dirty shared tree is not a reason to leave finished work
loose: inspect it with `fak sweep`, then land the coherent, green slice by explicit
path.

```bash
fak sweep                                        # group the dirty tree by lane; --json for a loop
make test-fast                                   # green the smoke tier first
fak commit --preview -m "<subject>" --path <p>   # lint the first subject/stamp before git is touched
fak commit --path <p> -m "<subject>"             # preferred commit path for a narrow change
# or:
fak sweep --apply --lane <lane> -m "<subject>"   # preferred commit path for a whole lane group
# subject: Conventional-Commits, verb-led, with a (fak <leaf>) trailer, e.g.
#   fix(gateway): treat same-tick ready as positive (fak gateway)
```

`fak commit --path <p> -m "<msg>"` mechanizes the whole rule: it stages only the
named paths under a lock, runs the real hooks, and asserts the committed file set
equals what you asked for (refusing `PATHSPEC_RACE` if a peer swept extra files in).
Preview the message without touching git with `fak commit --preview -m "<subj>"
--path <p>` — it catches a noun-led subject, a missing `(fak <leaf>)` trailer, or a
stamp/lane mismatch up front, which is the only place you can fix them on a shared
trunk. `fak sweep --apply --lane <lane> -m "<subj>"` is the layer above it for a
dirty tree: it reuses the same lane resolver, appends the `(fak <lane>)` trailer when
needed, and commits exactly that lane's dirty paths through the safe-commit path.
Raw `git commit -s -- <explicit paths>` remains the fallback when the binary is not
available; do not use `git add -A`. Work directly on `main`; the trunk guard refuses
an off-trunk commit (`OFF_TRUNK`). Default is to ship: once `make ci` is green,
commit and push.

Full contributor contract: [`CONTRIBUTING.md`](../CONTRIBUTING.md). How a *feature*
attaches as a leaf behind a `Register*` seam: [`EXTENDING.md`](../EXTENDING.md). A
broader catalog of verbs, runners, and demo scripts:
[fak/related-items.md](fak/related-items.md).

## What ships vs. what's planned

So you never reach for a verb that isn't there:

| Capability | Today | Dedicated verb |
|---|---|---|
| Enhanced debugging | `fak debug` (context/session core-dump debugger) + `fak doctor` (answer-shape diagnostic) + [integrations/debugging.md](integrations/debugging.md) | shipped |
| Built-in profiling | Go pprof (`go test -cpuprofile/-memprofile -bench`) + `fak benchmarks` / `fak bench` / `fak ablate` | a `fak profile` wrapper is **planned**, not shipped |
| Test runner | `fak test` (host-aware runner: routes `go test` to WSL on Windows), over `make test-fast` / `make test` / `make test-affected` / `make test-race` / `make ci`, `fak affected`, `./test.ps1` (WSL) | shipped |
| Dev workflow guide | this page, plus [`AGENTS.md`](../AGENTS.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md), [Work map](WORK-MAP.md) | shipped |

`fak test` is the first of these convenience verbs to land — it encodes the host
knowledge this guide carries (routing `go test` to WSL on Windows automatically) over
the same `make`/`go test` gates. The remaining planned wrapper, `fak profile`, would
be a thin verb over Go pprof; it is tracked under the developer-tooling epic, and until
it lands the profiling commands above are the supported path.
