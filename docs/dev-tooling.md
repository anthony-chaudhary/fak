---
title: "Developer Tooling — debug, profile, and test fak"
description: "The hands-on developer-tooling guide for fak: build and run, the test runner (make + WSL), debugging with fak debug and fak doctor, profiling and benchmarking, and the commit-and-ship dev loop."
---

# Developer tooling by task and platform

> **Primary audience:** contributors who already have a fak checkout and need to choose
> the correct build, test, debug, profile, or committed-tip witness for one development
> question. Start with [`CONTRIBUTING.md`](../CONTRIBUTING.md) for initial setup and the
> repository landing contract.

This page maps a question to one command family, then explains each family in depth.
The [Work map](WORK-MAP.md) chooses a project front door; the
[CLI reference](cli-reference.md) lists the runtime `fak` verbs, while `fak-dev help`
lists repository-development commands. `AGENTS.md` remains the machine-oriented
authority for shared-tree safety and recovery.

Repository-only checks use the separately built `fak-dev` artifact. Install it from
the checkout with `go install ./cmd/fak-dev`; the temporary `fak dev ...` compatibility
spelling only delegates to a sibling or `PATH`-installed `fak-dev` process. After that
one-time install, `fak self-update --root <checkout>` also detects a side-by-side
`fak-dev`, builds and smoke-checks both artifacts from the same verified `origin/main`,
and converges them in one cycle. A host without `fak-dev` remains product-only.

### Keep `fak-dev` current with the runtime binary

From the repository root, run:

```bash
fak self-update --root .
```

The update gate builds, vets, and smokes `fak` before swapping it. When `fak-dev`
already exists beside any converged `fak` copy, the same cycle builds and smokes the
companion before either artifact is accepted. An absent companion is not an install
request. `fak self-update --check --root .` is the read-only freshness path.

**Value frame and problem checklist:** for maintainers on a shared checkout, inherited
repository build scratch could disappear during a concurrent cleanup; the manual
alternative was setting `GOTMPDIR` for each update or installing both binaries
separately; the bounded default is better because updater-owned Go work uses OS temp and
the installed pair advances through one fail-closed gate. This is stewardship work: P1
context is preserved, P2 removes an observed failed multi-minute build and retry without
claiming a broader speedup, P3 stays reversible at one child-process seam, and P4 is
witnessed by the installed-path update and stamp read-back.

**Default-admission record:** indication — an updater-owned Go child, with `fak-dev`
already installed beside a converged `fak`; comparator — inherit the caller's
`GOTMPDIR`, with a manual per-run override as the strongest practical alternative;
benefit — one guarded command converges both artifacts; harms/interactions — a custom
Go temp location is ignored for updater children and OS-temp disk use lasts for the
build, with no authority, network, or telemetry expansion; uncertainty — low-space or
constrained OS temp directories, reviewed by 2026-09-20 if gate-failure telemetry shows
disk or permission errors; contraindication — an unavailable or unwritable OS temp
directory, which fails closed before swap; dose/safeguards — only `go` children launched
by `internal/selfinstall.RealRunner`, while non-Go children inherit the environment and
build/vet/smoke remain mandatory; consent/control — `--check` is read-only, `--target`
bounds an explicit install, and output names targets and failures; surveillance/movement
— watch `outcome=gate-failed`, OS-temp errors, and post-swap stamp mismatches, disabling
the override if it causes a repeat; rollback is removal of the one `GOTMPDIR`
assignment. **Verdict: CONDITIONAL DEFAULT** — narrow, reversible, and fail-closed.

The 2026-08-20 live witness first failed without the isolation:

```text
build fak-dev companion: open C:\work\fak\_scratch\go-tmp\go-build...\_pkg_.a: The system cannot find the path specified.
self-update: outcome=gate-failed
```

With the guarded path, both installed stamps matched the fetched tip:

```text
origin/main=284beb4bd8b43330656cfe214d731195e6069973
fak:     build: 284beb4bd8b4
fak-dev: build: 284beb4bd8b43330656cfe214d731195e6069973
```

`TestRealRunnerKeepsGoWorkOutsideRepoScratch` pins the temp boundary.
`TestSelfUpdateProbeReadsOwnPathAfterSwap` pins Windows post-swap read-back: the old
process remains mapped, so the final census invokes the deployed path and observes the
new file instead of reporting the old in-process stamp as a false divergence.

### Age out inherited repository `GOTMPDIR` sessions

Removing a persistent environment value does not rewrite an already-running agent or shell's
environment. Those inherited sessions may therefore continue creating compiler work below
`_scratch/go-tmp` until they exit. Audit that exact root with `fak tree-doctor --go-tmp --json`;
apply the same plan with `fak tree-doctor --go-tmp --apply --json` after reviewing it. The default
20-minute grace is configurable with `--go-tmp-grace`, and `--go-tmp-root` overrides the inherited
value while remaining constrained below the repository `_scratch` directory.

The maintenance pass enumerates host processes once, preserves any canonical child referenced by
the snapshot or touched inside the grace window, and fails closed on ambiguous enumeration. Apply
uses unique OS-temp quarantine, a second source/destination reference check, and exact bottom-up
deletion. As old sessions finish, their unreferenced work naturally crosses the grace and becomes
reclaimable; no peer process is terminated. See [Generated-output defaults](generated-output-defaults.md#repository-go-compiler-scratch)
for the safety contract and why broad `--sweep-scratch` is not the operational path.

**Next action:** from the repository root, run `fak test --list` to inventory the
host-aware test and non-test witness tiers without executing them. This is a one-time
discovery step, not a proof and not a command that every later task must repeat; after it
returns, choose the task-specific witness in the table below.

## Query the documentation index before surveying the tree

Start an unfamiliar repository task with one bounded lookup:

```bash
fak-dev index "shared-tree commit"          # default: ranked search of the curated INDEX.md map
fak-dev index docs "shared-tree commit"     # equivalent explicit spelling
fak-dev index graph --json                  # HEAD-only census of every tracked Markdown document
```

The shorthand is intentionally the default because documentation lookup is the common
orientation path. Named index operations still win when the first argument is a known
subcommand (`lane`, `refs`, `graph`, and the others listed by `fak-dev index`); use
`index docs <query>` when the query itself starts with one of those reserved words. The
query reads the curated map in the checkout. The graph census reads only committed
`HEAD`, so peer-dirty tracked bodies and untracked Markdown cannot change its evidence.

**Value frame:** for contributors and coding agents, the problem is repeated broad tree
surveys before the owning authority is known; today the fallback is manual grep plus a
large index read; the default is better because one read-only query narrows the route;
the witness is shorthand/explicit output equality plus a non-zero, HEAD-pinned corpus.

**Default-admission record:** indication — orient before a broad survey; comparator —
manual grep or reading the full map; expected benefit — less context and fewer repeated
reads; harms — a reserved first word can select a subcommand and ranking can miss an
uncurated page; uncertainty — relevance is metadata search, not full-text retrieval;
contraindication — use `rg` for exact symbols or non-Markdown files; dose/control — one
query, bounded with `--limit` and inspectable with `--json`; surveillance — the CLI tests,
`fak-dev index graph --json`, and the reciprocal index-sync gate; rollback — restore the
required `docs` subcommand and prior tree reader. **Verdict: DEFAULT**: the path is
read-only, reversible, locally witnessed, and keeps explicit controls.

## Match proof to activation depth

A green source tree is not proof that the binary currently serving an operator contains that source. Use the earliest row that reaches the failure boundary; every row deliberately names what it cannot prove.

| Proof depth | Command / witness | A pass proves | It does **not** prove |
|---|---|---|---|
| Working overlay | `fak-dev buildcheck --vet --mine <path>...` for compile feedback; `fak validate --mine <path>...` for the owned prospective tree | `HEAD` plus exactly the named working paths compiles (and, for `validate`, passes its affected build/vet/test gate) while peer WIP is hidden. | No commit exists; native code outside the named package graph, installed binaries, and running processes are unchanged. |
| Prospective native link | `fak validate --mine <native-or-Go-path>...` before commit; `fak commit --path <path>...` reruns the prospective admission automatically | The host-native package owning changed Go, cgo, Objective-C, header, or assembly paths links and its changed-package tests pass against the exact proposed tree. This is the row that catches a source/native-link regression such as #9224 before history changes. | It does not prove committed `HEAD`, another OS/accelerator envelope, or that an installed/running copy advanced. |
| Committed tip | `fak-dev ci-preflight` after the explicit-path commit | An archived checkout of the literal `HEAD` is gofmt-clean and buildable without peer WIP. | It does not replace pre-commit tests, inspect an installed copy, or restart a mapped process. |
| Installed copy | Invoke the candidate by its resolved absolute path and run `<absolute-fak> version`; for a source install, run `<absolute-fak> self-update --check --root . --json` | The selected on-disk executable identifies itself, and the self-update receipt compares that executable with the checkout tip. Repeat for every resolved copy when `PATH` contains more than one. | A matching file on disk does not prove an already-running gateway, hook host, or agent mapped the new bytes. |
| Running activation | Restart or hand off the owned process, then capture that process's own startup/health receipt and executable/build identity (the generic zero-model canary is tracked by #8915) | The process handling new work reports the expected executable identity after restart/handoff. This is the row for a stale-harness case. | A shell-side `version`, install receipt, or source test cannot substitute for process-owned identity; where a surface emits no activation receipt, activation remains **unproven**. |

Cold-reader rule: a missing native symbol in the proposed source belongs to **Prospective native link**; a long-lived harness still serving old bytes belongs to **Running activation**. Do not move either claim upward to an earlier, easier row.
## Choose the witness for the question

| Question | Start with | What a pass proves |
|---|---|---|
| Which current document owns this task? | `fak-dev index "<task terms>" --limit 5` | The ranked entries come from the curated repository map; this narrows the first read but does not replace code or test authority. |
| Does my working-tree change compile without peers' untracked Go files or an in-tree binary affecting the answer? | `fak-dev buildcheck --vet` | Inner-loop compile feedback: the isolated overlay compiles and vets the requested packages. It is not the pre-commit or committed-tip gate. |
| Which changed packages and importers need focused tests? | `fak test affected --list` | Task-specific planning after the global tier inventory: it prints the bounded changed-package set; run `fak test affected` to execute that set for behavior proof. |
| Does the fast repository smoke gate pass? | `make test-fast` | Build, vet, and short tests pass; weight-backed witnesses remain outside this tier. |
| Is the complete pre-commit tree green? | `make ci` | Build, formatting, vet, full tests, claims lint, and repository gates pass in the supported test environment. |
| Is the committed tip clean and buildable despite a peer-dirty checkout? | `fak-dev ci-preflight` | An archived checkout of `HEAD` is gofmt-clean and builds. **This post-commit proof never replaces the full pre-commit test gate.** |
| Why did a tool call fail or a completed session behave unexpectedly? | [`fak debug`](#fak-debug--the-context-debugger), [`fak doctor`](#fak-doctor--the-answer-shape-diagnostic), or the [denial guide](#debugging-a-denied-tool-call) | The selected diagnostic reports its scoped finding; it is not a repository green bar. |
| Is a performance change net-true? | The [benchmark verbs](#the-benchmark-verbs) and the repository's benchmark authority | Execute a matched baseline and candidate to produce the task proof; then run the normal repository gates separately. Listing a benchmark command or passing CI is not performance evidence. |

`fak debug`, `fak profile`, and `fak test` are shipped CLI surfaces. `fak profile` and
`fak test` are host-aware wrappers over the Go toolchain and repository gates. `fak
`test --list` only discovers all runner tiers; `fak test -n <tier>` dry-runs one
resolved tier; `fak test <tier-or-package>` executes one. `fak test affected --list`
performs a different, task-specific discovery: changed packages and their importers.
`make ci` is the authoritative pre-commit green bar.

## Choose the platform path

| Platform | Build and vet | Test execution | Required full gate |
|---|---|---|---|
| Linux or macOS | Native Go toolchain and `fak-dev buildcheck` | Native `fak test` / `make test*` | `make ci` |
| This native Windows control host | Native Go build, vet, run, and `fak-dev buildcheck` | The selected WSL distro needs Go 1.26+; the Windows-host Go install is not visible inside WSL, and `GOTOOLCHAIN=auto` cannot bootstrap without a base `go` command. Prefer host-aware `fak test <tier-or-package>`; it routes to WSL. Use `./test.ps1 [package]` as the direct WSL suite entry. Do not run native `go test`. | Run `./test.ps1` for the full test suite plus the native non-test gates named by the change; CI is the complete `make ci` witness. Then run `fak-dev ci-preflight` for committed `HEAD`. |
| CI or a sanctioned compute node | Use the runner's declared toolchain | Use the checked-in gate or workload command | The workflow or node witness named by the task; do not substitute local hardware absence for that witness |

## Route context

| Dimension | Contract in force |
|---|---|
| Mode | Where the commands apply: source development in a repository checkout; installed-binary operation belongs to the operator route. |
| Generation | Which guidance applies: this `gen/now` task-and-platform route. Historical notes and scorecards do not override this route. |
| Lifecycle | When each gate applies: choose the question → run the narrow witness for feedback → run `make ci` before commit to test the complete working-tree change → run `fak-dev ci-preflight` after commit to prove archived `HEAD` without peer WIP → push. |
| Support | Build, test, debug, profile, and ship-loop selection are covered. Product use starts at [`README.md`](../README.md), deployment recovery at [`docs/operator/`](operator/), and research hypotheses at the [research-agent playbook](playbooks/research-agent.md). |

For the ranked improvement program behind these tools, see the
[testing/linting infrastructure scorecard](TESTING-LINTING-INFRA-SCORECARD.md).

## Build and run

The Go module is the repository root, so every `go` command runs from the clone root.
Choose the command by intent:

```bash
fak-dev buildcheck --vet              # repository compile/vet check; no in-tree binary
make build                            # produce a debuggable ./fak for local use
make build-all                        # compile every package and link every command
./fak --help                          # every verb on the produced binary
./fak doctor --help                   # the read-only diagnostic described below
```

A raw `go build -o fak ./cmd/fak` deliberately produces the runnable in-tree binary; it
is not a collision-safe verification command on the shared Windows trunk. The 60-second,
no-key/no-model/no-GPU proof uses that produced binary — see `AGENTS.md` and the full
[repro packet](repro-packet.md). The build-profile table below applies to that runtime
artifact; `fak-dev` is the separately built contributor command.

`make build` emits the artifacts an operator asked for: the debuggable `./fak` plus the
native/Windows repository-guard hooks and dispatch worker. It does not also link every
demo and benchmark command. Use the explicit `make build-all` target when the question is
whether the complete package tree builds. `make ci` names both targets as direct
prerequisites, so separating the fast artifact build does not narrow the green-bar gate.

### Runtime, developer tooling, and local outputs

`fak` and `fak-dev` are two executables from the same Go module, not two
repositories and not two independently versioned source packages:

- `cmd/fak` is the adopter-facing runtime and guard. Its dependency graph stays
  free of repository-maintenance code.
- `cmd/fak-dev` links the tracked implementation in `internal/devcmd`. Keeping
  that graph behind a process boundary prevents release/runtime installs from
  carrying CI, worktree, issue, and maintainer-only machinery.
- `tools/` is tracked legacy repository automation. New durable tooling belongs
  in a Go leaf/verb; it is not installed by copying `tools/`.
- Root names beginning with `_` are ignored scratch by `/_*` in `.gitignore`.
  For example, `_tooling.json` is a regenerable tooling-quality scorecard
  output. It is neither a package manifest nor input to new-machine setup.
  Compiled `*.exe` files are ignored for the same reason: source is authoritative.

A clean clone therefore has no local binary or `_tooling.json`, by design. For a
maintainer setup with the checkout's exact source revision:

```bash
go install ./cmd/fak
go install ./cmd/fak-dev
fak dev availability --json
```

`go install` writes both executables to `GOBIN` (or `GOPATH/bin`); put that
directory on `PATH`. The availability probe reports whether `fak-dev` resolves
as a sibling of `fak`, elsewhere on `PATH`, or is missing. From outside a source
checkout, install a released matching pair explicitly:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
go install github.com/anthony-chaudhary/fak/cmd/fak-dev@latest
```

Runtime-only users install only `fak`. Commands under the compatibility spelling
`fak dev ...` deliberately hand off to `fak-dev`; they do not silently relink,
download, or regenerate developer tooling. Release packaging for the companion
artifact is tracked separately in issue #6024.

## Build profiles

Every shipping build routes through **one** entrypoint, `scripts/build.sh`, so the
`-trimpath`/`-ldflags`/version-stamp flags live in a single place and cannot drift apart
(both Dockerfiles and `release-artifacts.yml` call it; the guard `tools/build_entrypoint_test.py`
reds if any of them re-inlines the recipe). You select a **profile** with `$PROFILE`; the
`make` targets pick one for you. Every profile stamps `internal/appversion.BuildVersion`, so
`fak version` is honest no matter how you built — only the strip/trim/DWARF posture differs:

| Profile   | `make` target     | `-trimpath` | strip `-s -w` | DWARF   | `CGO_ENABLED` | `-race` | Use |
|-----------|-------------------|:-----------:|:-------------:|:-------:|:-------------:|:-------:|-----|
| `release` | `make release`    | ✓           | ✓             | stripped| `0` (static)  | —       | the **shipped** binary — stripped, reproducible-ready, stamped |
| `dev`     | `make build`      | —           | —             | kept    | `0` (static)  | —       | fast local artifact + hook build; `dlv` (the Go debugger) can set a breakpoint and step |
| `race`    | `make build-race` | —           | —             | kept    | `1` (cgo)     | ✓       | opt-in race-detector variant; **not** the static pure-Go binary |

The `dev` profile keeps DWARF/symbols and host paths so a debugger works out of the box; add
`GCFLAGS='all=-N -l' make build` for pristine stepping (disables inlining/optimization). The
`race` profile's `-race` **requires** cgo, so `make build-race` sets `CGO_ENABLED=1` and
preflights a C compiler, refusing rather than building a race-blind binary on a cgo-less box
(the same contract as `make test-race`). `TAGS=cuda` (the GPU image's delta from `release`)
and future `dev`/`race` build tags layer on top of the profile, not beside it.

## Feature-tag variant taxonomy

The codebase maintains a strict taxonomy of supported build-tag combinations, distinguishing
pure-Go / GPU-free reproducible artifacts from cgo / hardware-accelerated backends and in-flight
work-in-progress fences (#3712, epic #3708). The single source of truth is declared in
[`docs/build-variants.json`](build-variants.json) (mirrored in `.github/build-variants.json`).

The matrix is split into two primary classes:
- **Pure-Go / GPU-free rows**: compile with `CGO_ENABLED=0` across all release targets without
  any external C compiler or GPU toolchain. These form the default shipped binaries and advisory
  WIP fences.
- **cgo / toolchain-linked rows**: compile with `CGO_ENABLED=1` and require external headers, C/C++
  compilers, and platform/accelerator SDKs (CUDA, NCCL, Vulkan, Metal).

| Variant | Build tags | `CGO_ENABLED` | Default inclusion | Target GOOS/GOARCH | Toolchain requirements | Gate / Class |
|---|---|:---:|---|---|---|---|
| `default` | `""` | `0` | default | `all` (`linux/*`, `darwin/*`, `windows/*`) | Go 1.26+ (pure Go, CGO_ENABLED=0) | `matrix` (trunk-gating) |
| `darwin-arm64` | `""` | `0` | default | `darwin/arm64` | Go 1.26+ cross-compiler (pure Go) | `matrix` (trunk-gating) |
| `linux-amd64` | `""` | `0` | default | `linux/amd64` | Go 1.26+ (pure Go) | `matrix` (trunk-gating) |
| `windows-amd64` | `""` | `0` | default | `windows/amd64` | Go 1.26+ cross-compiler (pure Go) | `matrix` (trunk-gating) |
| `wip_sessionfleet` | `wip_sessionfleet` | `0` | opt-in | `all` | Go 1.26+ (pure Go) | `advisory` (continue-on-error) |
| `cuda` | `cuda` | `1` | opt-in | `linux/amd64` | Go 1.26+, C compiler, CUDA Toolkit (nvcc, cudart, cublas) | `external` (`cuda-build.yml`) |
| `cuda-nccl` | `cuda nccl` | `1` | opt-in | `linux/amd64` | Go 1.26+, CUDA Toolkit, NVIDIA NCCL headers/libs | `none` (unwitnessed) |
| `cuda-windows` | `cuda` | `1` | opt-in | `windows/amd64` | Go 1.26+, MSVC C++, CUDA Toolkit for Windows | `manual` (`windows-cuda.yml`) |
| `vulkan-windows` | `vulkan` | `1` | opt-in | `windows/amd64` | Go 1.26+, C compiler, Vulkan SDK, AMD driver | `none` (unwitnessed) |
| `metal` | `""` | `1` | opt-in | `darwin/arm64` | macOS 14+, Apple Clang, Metal / MPS frameworks | `none` (unwitnessed) |

### Validating the variant matrix

Validate the build matrix locally with `fak-dev`:

```bash
fak-dev build-matrix                      # test pure-Go variants for release targets
fak-dev build-matrix --target linux/amd64 # narrow to a single target
fak-dev build-matrix --dry-run            # inspect planned compile matrix commands
fak-dev build-matrix --json               # machine-readable fak.build_matrix_report.v1 envelope
```

In CI, `.github/workflows/build-matrix.yml` validates all declared pure-Go variants across all release
targets on every push and PR.

## The test runner

`fak test` is the host-aware runner: it resolves the right `go test` invocation for
the tier you ask for and, on Windows, routes it through `test.ps1` (WSL) automatically
so you never hit the OS-policy block below. The `make` target set is the authoritative
gate it sits over; `fak test --list` prints the tiers, and `fak test -n <tier>` prints
the resolved command without running it. It sits over the `make` target set — the
authoritative gates — with one host caveat that bites on Windows.
For machine consumers, put `--json` before the tier: `fak test --json -n race`
emits `fak.test_repair_packet.v1` with the resolved command, and `fak test --json
<tier-or-package>` captures the command result into the same envelope with
`findings`, `exit_code`, output tails, and a `next_action`. The same packet shape
also covers the first non-test gates: `fak test --json build`, `vet`, `gofmt`,
`codelint <path>`, `ruff <path>`, and `scorecard`. When ruff is not installed, the
packet is an explicit `SKIP`, not a clean lint signal. The scorecard packet wraps
the native `fak scorecard control-pane --check --json` ratchet and carries
scorecard-regression or unmeasured-scorecard diagnostics; `fak test scorecard`
stays read-only and refuses `--pin`.
For changed-package work, use `fak test affected ...`: it delegates to `fak affected`
so the agent-facing front door still gets the affected planner's `--json`, `--list`,
`--file`, `--budget`, report, and pass-through `go test` flags.
For slow-test triage, run `make test-durations` to execute the fast tier through
`go test -json` and write the stable `fak.test_duration_ledger.v1` package/test
ledger with ranked budget findings. Use `fak test durations --run <tier-or-package>`
for a custom target, or `--input`/stdin to fold an existing stream.
To turn that ledger into parallel work, run `fak test shards`: it emits
`fak.test_shard_plan.v1` with balanced `go test` commands, using measured package
elapsed time instead of hand-ordered package lists.

| Command | What it runs | When |
|---|---|---|
| `fak test [fast\|full\|race\|<pkg>]` | the host-aware wrapper over `go test` (default tier `fast`); on Windows routes to WSL via `test.ps1`; `fak test fast -- -run TestX` passes flags through | the one-verb inner loop over the targets below |
| `fak test [build\|vet\|gofmt]` | runs `go build ./...`, `go vet ./...`, or `gofmt -l .`; `gofmt` exits non-zero when files are listed | the fast non-test gates from `make ci`, exposed through the same runner |
| `fak test codelint <path...>` | delegates to `fak codelint` over the same language packs | lint agent-written code through the same agent-facing test front door |
| `fak test ruff [path...]` | runs `ruff check` when ruff is on PATH; otherwise reports an explicit skip | make Python lint availability visible instead of silently treating a missing tool as clean |
| `fak test scorecard [--timeout N\|--baseline FILE]` | runs `fak scorecard control-pane --check`; refuses `--pin` so the test surface cannot rewrite the ratchet baseline | scorecard-ratchet proof through the same test front door |
| `fak test --json [-n] [build\|vet\|gofmt\|codelint\|ruff\|scorecard\|fast\|full\|race\|<pkg>]` | emits `fak.test_repair_packet.v1` for the resolved command or finished run, with normalized finding class, diagnostics, exit code, output tails, and next action | agent-facing repair packets instead of raw terminal logs |
| `fak test affected [--json\|--list\|--file P\|--budget DUR] [-- go test args]` | delegates to `fak affected`, selecting changed packages plus transitive importers, with JSON/list dry-runs and budget/report flags preserved | the default agent loop before paying for the full suite |
| `make test-durations` | runs `fak test durations --run fast --out .fak/test-duration-ledger.json` with package/test budgets | find the slowest next package or test before widening CI shards |
| `fak test shards --input .fak/test-duration-ledger.json --shards 4 --go-arg -short` | reads a duration ledger and emits deterministic balanced `go test` shard commands as `fak.test_shard_plan.v1` | local proof/planning step before wiring CI to consume measured shards |
| `make test-fast` | `build` + `vet` + `go test -short ./...` (~2s smoke tier; skips the weight-backed model witnesses) | the pre-commit / pre-push floor — ~95% of logic regressions in seconds |
| `make test` | `go test ./...` (full suite incl. the ~538 MB f32/safetensors model oracle) | the authoritative gate before you trust a model-touching change |
| `make test-affected` | `fak affected` → `go test` for only the packages your working-tree change can reach (changed + transitive importers, test imports included) | the fast inner loop on the REAL oracle (no `-short`) for a one-leaf edit |
| `make test-race` | `CGO_ENABLED=1 go test -short -race ./...`, cgo-preflighted (refuses on a compiler-less box rather than building a race-blind false green) | catch a data race locally instead of minutes later in CI — see [testing/race-detector.md](https://github.com/anthony-chaudhary/fak/blob/main/docs/testing/race-detector.md) |
| `make build-all` | `go build ./...`: compile every package and link every command | explicit whole-tree build proof; also a direct `make ci` prerequisite |
| `make ci` | the full gate: `build` + `build-all` + `gofmt-check` + `vet` + `test` + `claims-lint` + the doc/scorecard gates | the green-bar definition the guards expect before you ship |

For a single package, `go test ./internal/<pkg>/... -count=1` is the direct form
(`-count=1` defeats the test cache when you want a clean re-run).

> **Windows host caveat.** The selected WSL distro needs Go 1.26+; the Windows-host
> Go install is not visible inside WSL, and `GOTOOLCHAIN=auto` cannot bootstrap without a
> base `go` command. Native `go build` / `go vet` / `go run` work,
> but native `go test` is blocked by an OS Application-Control policy on the freshly-compiled
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

`fak profile` is the host-aware profiler: it resolves the `go test -bench`
invocation for a package, captures CPU and allocation profiles, routes through WSL
on Windows, and points at `go tool pprof` for inspection. It is a convenience layer
over standard Go profiling; the benchmark verbs below remain the curated perf
surfaces.

```bash
fak profile ./internal/ctxmmu/                         # CPU + allocation profiles for all package benchmarks
fak profile ./internal/recall/ --bench BenchmarkDigest # narrow to one benchmark regexp
fak profile ./internal/ctxmmu/ --benchtime 2s --top    # profile, then print pprof -top
fak profile ./internal/ctxmmu/ -n                      # print the resolved command without running it
```

### Go pprof (CPU, memory, blocking)

The kernel is a Go binary, so the Go toolchain's profilers apply directly. Profile a
hot package through its benchmarks:

```bash
# CPU + allocation profile for one package's benchmarks (run under WSL on Windows)
go test -run=^$ -bench=. -benchmem \
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

Every perf number is held to the [net-true-value standard](https://github.com/anthony-chaudhary/fak/blob/main/EXTENDING.md): measured
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
fak sweep --clean-junk                           # optional: remove only classified junk files you own
make test-fast                                   # green the smoke tier first
fak commit --preview -m "<subject>" --path <p>   # lint the first subject/stamp before git is touched
fak commit --path <p> -m "<subject>"             # preferred commit path for a narrow change
# or:
fak sweep --apply --lane <lane> -m "<subject>"   # preferred commit path for a whole lane group
fak sync push                                    # safe publish path; retries moving-trunk races
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
When the sweep report starts with junk, `fak sweep --clean-junk` removes only
freshly classified junk files and refuses directories or unsafe paths; rerun
`fak sweep --json` afterward before committing lane work.
Publish finished work with `fak sync push`; it is the push-side sibling of
`fak sync check/apply`, so moving-trunk races get retried without pull/stash/reset,
and the output still reports the remaining dirty-tree next action.
Raw `git commit -s -- <explicit paths>` remains the fallback when the binary is not
available; do not use `git add -A`. Work directly on `main`; the trunk guard refuses
an off-trunk commit (`OFF_TRUNK`). Default is to ship: once `make ci` is green,
commit and push through `fak sync push`.

Full contributor contract: [`CONTRIBUTING.md`](https://github.com/anthony-chaudhary/fak/blob/main/CONTRIBUTING.md). How a *feature*
attaches as a leaf behind a `Register*` seam: [`EXTENDING.md`](https://github.com/anthony-chaudhary/fak/blob/main/EXTENDING.md). A
broader catalog of verbs, runners, and demo scripts:
[fak/related-items.md](fak/related-items.md).

## What ships vs. what's planned

So you never reach for a verb that isn't there:

| Capability | Today | Dedicated verb |
|---|---|---|
| Enhanced debugging | `fak debug` (context/session core-dump debugger) + `fak doctor` (answer-shape diagnostic) + [integrations/debugging.md](integrations/debugging.md) | shipped |
| Built-in profiling | `fak profile` (host-aware wrapper over `go test -bench -cpuprofile -memprofile`) + Go pprof + `fak benchmarks` / `fak bench` / `fak ablate` | shipped |
| Test runner | `fak test` (host-aware runner: routes `go test` to WSL on Windows), `fak test affected` (the affected-package agent loop), over `make test-fast` / `make test` / `make test-affected` / `make test-race` / `make ci`, `fak affected`, `./test.ps1` (WSL) | shipped |
| Dev workflow guide | this page, plus [`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md), [`CONTRIBUTING.md`](https://github.com/anthony-chaudhary/fak/blob/main/CONTRIBUTING.md), [Work map](WORK-MAP.md) | shipped |

`fak test` and `fak profile` encode the host knowledge this guide carries (routing
`go test` to WSL on Windows automatically) over the same `make`/`go test` gates.
They are the developer-experience layer, not a replacement for the repo's
authoritative CI gates.
