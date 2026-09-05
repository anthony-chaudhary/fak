# Makefile — portable build/test entrypoints (unit 12). On Windows without make,
# use scripts/ci.ps1, which this mirrors.
.PHONY: ci build build-all cross-build-harnessres clean vet architest-gate test test-fast test-integration test-airgap smoke-build test-fast-build-regression test-affected test-durations test-race bench mac-perf status status-check release-staleness release-staleness-check release-readiness garden garden-check dogfood-recent dogfood-test performance-rsi-health vcache-gate cache-default-readiness gitdaily-score claims-lint cache-headline-lint cachedoc-numbers-lint salience dos-lint index-sync model logvault-drill gofmt-check hygiene demo-audit demo-tool-tests demo-scorecards scorecard-ratchet demo-smoke demo-headless-smoke demo-live-status demo-https-status demo-published-status demo-published-check demo-readiness-status gated-tests cuda-check cuda-build cuda-test cuda-accept cuda-occupancy

VERIFY_LOOP_BUDGET ?= 30s
SMOKE_BUILD_BUDGET ?= 2m
SMOKE_BUILD_KILL_AFTER ?= 5s
GO ?= go
TEST_DURATION_LEDGER ?= .fak/test-duration-ledger.json
TEST_PACKAGE_BUDGET ?= 30s
TEST_TEST_BUDGET ?= 5s
ARCHITEST_GATE_RE ?= ^(TestEveryPackageDeclaresTier|TestNoUpwardImports|TestRootImportsNothingInternal|TestSingleOpenAIChatClient)$$

# ci is THE local green gate (AGENTS.md: "Green = make ci"). It must stay aligned with
# .github/workflows/ci.yml's HARD steps so a pre-push `make ci` fails on the same things
# GH does, instead of a developer only discovering a gofmt/hygiene break after the push.
# gofmt-check + hygiene are the deterministic, no-network CI gates that were previously
# CI-only — wired in here to close that local↔CI drift. (Network/range gates — leak-scan,
# dos-review — stay CI/githook-only; the release-substrate suite stays CI-only by weight.)
# demo-audit is the one-command local demo health gate: static docs/metadata/link
# contracts, demo-audit tool unit tests, quality/robustness scorecards, plus browser and headless dynamic witnesses.
# demo-live-status / demo-https-status / demo-published-status / demo-readiness-status
# are optional network status views. demo-published-check is the optional network check for the HTTPS GitHub Pages copy and remote share image.
# demo-smoke builds and serves every browser demo under a reverse-proxy-style base
# path, proving the local pages and lightweight APIs still come up. demo-headless-smoke
# runs the model-free terminal witnesses from run-the-demos.md.
# cuda-check is the GPU-free CUDA ABI/header preflight — deterministic, no CUDA toolkit,
# so it joins the local gate the same way (the cuda-build.yml `static` job is its CI mirror).
ci: build build-all cross-build-harnessres gofmt-check disambiguation-generated-check vet test claims-lint cache-headline-lint cachedoc-numbers-lint cache-default-readiness gitdaily-score salience dos-lint index-sync hygiene demo-tool-tests demo-scorecards scorecard-ratchet cache-proving demo-smoke demo-headless-smoke gated-tests cuda-check
	@echo "CI OK"

build:
	# Emit a debuggable ./fak through the ONE build entrypoint (dev profile, #3710):
	# DWARF/symbols kept + host paths intact so Delve can step, BuildVersion stamped so
	# `fak version` is honest. Set GCFLAGS='all=-N -l' for pristine stepping.
	# The shipped stripped/reproducible binary is `make release`; the race variant is
	# `make build-race`; see docs/dev-tooling.md for the profile flag-delta table.
	OUT=fak PROFILE=dev sh scripts/build.sh
	# Drop the repo-guard PreToolUse hook binary into the (gitignored) tools/.bin so
	# the Claude Code hook runs the Go guard instead of spawning Python per tool call
	# (.claude/settings.json prefers this binary, falling back to tools/repo_guard.py
	# until it exists). Part of `build` so every green gate self-propagates it.
	go build -o tools/.bin/repoguard ./cmd/repoguard
	# Also cross-build a fresh Windows repoguard.exe. The native-Windows Claude Code
	# PreToolUse hook can ONLY exec a .exe (CreateProcess refuses the bare-named PE),
	# yet green cycles often run under Linux/WSL where the line above emits only the
	# bare name -- so a stale .exe from an old build silently shadows the fix forever
	# (the shim picks the freshest binary, but Windows can't run the fresh bare one).
	# Cross-building keeps BOTH hook binaries fresh every green cycle, any build host (#3400).
	GOOS=windows GOARCH=amd64 go build -o tools/.bin/repoguard.exe ./cmd/repoguard
	# Likewise the Go DOS dispatch-worker launcher — the interpreter-free cutover
	# target for tools/dispatch_worker.py (parity-tested; see dos.toml [supervise]).
	go build -o tools/.bin/dispatchworker ./cmd/dispatchworker

# Compile every package and link every command. Keep this explicit: `make build` emits
# the developer binary and hook binaries operators asked for, while this whole-tree gate
# also links the repository's large demo/benchmark command fleet. `ci` names both as
# direct prerequisites so the complete pre-commit contract is unchanged.
build-all:
	go build ./...

.PHONY: release
# The release binary, built through the ONE canonical recipe every shipping consumer
# uses (scripts/build.sh, #3709) — so `make release` locally exercises the exact
# -trimpath/-ldflags/stamp invocation the Dockerfiles and the release-artifacts
# workflow ship. Plain `build:` above stays the fast dev build.
release:
	sh scripts/build.sh

# build-race (#3710): the opt-in race-instrumented dev binary, built through the ONE
# entrypoint (scripts/build.sh, PROFILE=race). -race REQUIRES cgo (CGO_ENABLED=1), so
# this is NOT the static pure-Go binary `make build` / `make release` produce. Same cc
# preflight as `test-race`: REFUSE (exit 2) on a cgo-less toolchain rather than silently
# building a race-BLIND binary. See docs/dev-tooling.md for the profile flag-delta table.
.PHONY: build-race
build-race:
	@command -v cc >/dev/null 2>&1 && cc --version >/dev/null 2>&1 || { \
		echo "build-race: BLOCKED -- no working C compiler (cc); the Go race detector needs cgo (CGO_ENABLED=1)." >&2; \
		echo "  Building -race without it silently produces a race-BLIND binary. Run on a cgo-capable box (WSL/Linux/macOS)." >&2; \
		exit 2; }
	CGO_ENABLED=1 OUT=fak PROFILE=race sh scripts/build.sh
	@echo "build-race OK (race-instrumented ./fak via scripts/build.sh PROFILE=race; NOT the static binary)"

# clean: prune the stray go-build binaries that pile up at the module root across
# bench/demo builds (hundreds of MB of gitignored *.exe / bare cmd/<name> outputs).
# The build-artifact twin of `git-maint`, safe by construction: it only ever removes
# files git ignores and keeps the live `fak` binary. Run it from a loop/CI (or by hand)
# to keep the tree from bloating; `make clean -- --dry-run`-style previews are available
# via `go run ./cmd/fak clean-bins --dry-run`.
clean:
	go run ./cmd/fak clean-bins


# Compile every harness-resource platform reader even when CI runs on one GOOS.
cross-build-harnessres:
	@for os in linux darwin windows; do \
		echo "cross-building internal/harnessres for $$os"; \
		CGO_ENABLED=0 GOOS=$$os go build ./internal/harnessres/... || exit $$?; \
	done

vet:
	go vet ./...

architest-gate:
	go test -count=1 ./internal/architest/ -run '$(ARCHITEST_GATE_RE)'

# The suite boundary strips provider credentials using envconfiglint's secret-name
# registry before any test process starts. Tests that need synthetic credentials set
# them explicitly inside their own process; inherited developer credentials never cross.
test: architest-gate
	go run ./cmd/testenv -- go test ./...

# test-fast: the 2s smoke tier — the synthetic + architest invariants only.
# `-short` skips the weight-backed model witnesses (the ~538MB f32/safetensors
# loads that are the slow + OOM-prone part of the WSL suite — see fak/test.sh and
# the model-test OOM note). That is ~95% of logic regressions in seconds, so it is
# the right floor for a pre-commit / pre-push gate. Pair `build vet` with it so a
# commit that doesn't compile or vet-clean is caught at the same gate. The full
# `test` target (no -short) still runs the real oracle locally + in CI.
smoke-build:
	@status=0; timeout --kill-after=$(SMOKE_BUILD_KILL_AFTER) $(SMOKE_BUILD_BUDGET) $(GO) build -buildvcs=false ./... || status=$$?; \
	if [ $$status -eq 124 ] || [ $$status -eq 137 ]; then \
		echo "test-fast build: TIMEOUT after $(SMOKE_BUILD_BUDGET); WSL /mnt/c remedy: retry from the Linux filesystem or raise SMOKE_BUILD_BUDGET." >&2; \
		exit 124; \
	fi; \
	exit $$status

test-fast-build-regression:
	sh scripts/test-fast-build_test.sh

test-fast: smoke-build test-fast-build-regression vet architest-gate
	go test -short ./...
	@echo "test-fast OK (smoke tier; run 'make test' for the weight-backed witnesses)"

# test-affected: the fast INNER loop. `fak affected` runs `go test` for only the
# packages your working-tree change can affect (the changed packages + every package that
# transitively imports one, test imports included) instead of the whole `go test ./...`.
# For a one-leaf edit that is seconds, not minutes — so you can verify the REAL oracle
# (not -short) on every change. The full `test` target stays the authoritative gate; this
# never DROPS coverage on what you changed, it only skips packages your change can't reach.
test-affected: build
	go run ./cmd/fak affected --budget $(VERIFY_LOOP_BUDGET) --report .fak/verify-loop-affected.json
	@echo "test-affected OK (affected packages only, budget $(VERIFY_LOOP_BUDGET); run 'make test' for the full oracle)"

test-durations:
	go run ./cmd/fak test durations --run fast --out $(TEST_DURATION_LEDGER) --package-budget $(TEST_PACKAGE_BUDGET) --test-budget $(TEST_TEST_BUDGET)
	@echo "test-durations OK (ledger: $(TEST_DURATION_LEDGER); budgets package=$(TEST_PACKAGE_BUDGET) test=$(TEST_TEST_BUDGET))"

# test-race (#1311): the fast LOCAL correctness gate for the inner loop — the
# data-race signal CI has (the separate `race detector` job in .github/workflows/ci.yml)
# but the local `make test-fast` / `make ci` chain LACKS (both run `go test` WITHOUT
# -race; CLAIMS.md notes -race needs cgo+gcc, absent natively on the Windows dev box).
# Without this the inner loop verifies only build+vet+test locally and learns of a data
# race minutes later, out-of-band, in CI — the gap #1311 names against #1148's
# "loop velocity x witnessed error-correction" goal. ONE command: `go test -short -race
# ./...`, run directly like the `test` / `test-fast` targets above (so it inherits their
# "run under WSL — native `go test` is OS-blocked on this host" contract; see AGENTS.md
# and the AVOID-TESTING note). It mirrors CI's race job: forces CGO_ENABLED=1 and
# cgo-PREFLIGHTS a working C compiler first, REFUSING (exit 2) rather than letting -race
# silently build a race-BLIND binary + a false green on a cgo-less toolchain. `-short`
# skips the slow weight-backed model oracle so it stays a seconds-to-minutes pre-commit
# gate; the full no-`-short` `go test -race ./...` stays authoritative in CI (and
# tools/race_test.sh wraps the full run on a nested-module checkout). Kept a SEPARATE
# target (not folded into `make ci`) to mirror CI's own separate-job architecture — the
# ~2-10x -race slowdown never gates the fast build/vet/test feedback, and a cgo-less box
# is refused, not wedged. Pair with `make test-affected` for the REAL (no -short) oracle
# on the packages your change can reach: race everywhere (short) + oracle on what you
# touched, without the CI round-trip. See docs/testing/race-detector.md.
test-race:
	@command -v cc >/dev/null 2>&1 && cc --version >/dev/null 2>&1 || { \
		echo "test-race: BLOCKED -- no working C compiler (cc); the Go race detector needs cgo (CGO_ENABLED=1)." >&2; \
		echo "  Running -race without it silently builds a race-BLIND binary + a false green. Run on a cgo-capable" >&2; \
		echo "  box (WSL/Linux/macOS) -- see docs/testing/race-detector.md." >&2; \
		exit 2; }
	CGO_ENABLED=1 go test -short -race -count=1 -timeout=25m ./...
	@echo "test-race OK (-short -race ./...; full no--short -race stays authoritative in CI -- pair with 'make test-affected' for the oracle)"

.PHONY: test-airgap
test-airgap:
	go test -v ./internal/airgaptest -run 'TestAirGapBootstrap_ZeroEgress'

bench:
	go build -o fak ./cmd/fak && ./fak bench --suite tau2-smoke --out report.json

# mac-perf: on-device shift-left performance gate for Apple Silicon Metal & Mac inference.
# Benchmarks tok/s decode and prefill throughput on the native Metal engine and validates
# the 3-way Mac comparison packet.
mac-perf: build
	@echo "== Mac Shift-Left Performance Verification =="
	@go test -v ./internal/macbench -run '^TestValidateComparisonPacketNodeMacOSA$$'
	@go test -v ./internal/model -run '^$$' -bench '^BenchmarkMetalQ2KGemv$$'
	@go test -v ./internal/model -run '^$$' -bench '^BenchmarkMetalQ2KGemmSteady$$'
	@go test -v ./internal/model -run '^$$' -bench '^BenchmarkMetalQ4KGemv$$'
	@go test -v ./internal/model -run '^$$' -bench '^BenchmarkMetalQ4KGemmSteady$$'
	@./fak macbench validate-comparison --input experiments/benchmark/runs/by-machine/node-macos-a/20260903T050000Z-macbench-threeway/packet.json --json
	@echo "mac-perf OK (Apple Silicon Metal tok/s and prefill performance verified)"

# status: the cross-domain "where do we stand right now?" rollup — folds git +
# benchmarks + work + industry into ONE control-pane view (the sibling of
# scorecard_control_pane one level up: that folds the scorecard FAMILY, this folds
# the four top-level domains). Read-only, pure-stdlib, no network by default; the
# git pane is the crystal-clear center, the benchmark pane rolls up catalog.json
# with per-run measured|modeled|unknown provenance (unknown surfaced loudly). NOT
# in the hard `ci` chain — it folds optional/slow sub-tools (industry scorecard);
# its hermetic *test* is gated automatically by `gated-tests`. status-check is the
# advisory gate (non-zero only on a HARD pane failure or stale benchmarks).
status:
	@python3 tools/fresh_status.py

status-check:
	@python3 tools/fresh_status.py --check

# release-staleness: the PUBLISH-freshness gate — is the version
# `go install ...@latest` resolves actually current, or has the trunk moved far past
# it? `fak release-staleness` answers it as a gateable number (commits + days behind
# the latest tag) with a control-pane envelope. Wiring it into a target makes the
# VERY_STALE signal LIVE instead of a verb nobody runs (#1367 / epic #1354): a loop or
# operator runs `make release-staleness` to see the lag, `make release-staleness-check`
# to gate on it. Read-only. The release-readiness scorecard credits this wiring.
release-staleness:
	@go run ./cmd/fak release-staleness --json

release-staleness-check:
	@go run ./cmd/fak release-staleness --check

# release-readiness: the deterministic release-debt scorecard — can fak cut, validate,
# publish, and roll back a release at agentic speed? Re-derived from git + the tracked
# tree + live release signals; folds into the scorecard control pane. Pure-stdlib.
release-readiness:
	@go run ./cmd/fak release readiness

# garden: the default-on gardening bundle — ONE read-only fold over the repo's
# self-maintenance passes (the scorecard control pane + fresh status), so "run the
# gardening" is one command instead of three. This is the fold over the folds:
# scorecard_control_pane folds the scorecard FAMILY, fresh_status folds the four
# domains, and THIS folds those into one verdict. Read-only — it measures and
# reports, never fixes (auto-fix is a later witness-gated rung). NOT in the hard `ci`
# chain: like `status` it folds slow members, and the scorecard ratchet already gates
# in ci.yml. Skipped when FAK_GARDEN is off (the env-side governor brake). Add --deep
# for the slower fleet loop-audit member. garden-check is the gate (non-zero only when
# a gating member regressed or a pass failed to run). Full design:
# docs/notes/GARDENING-BUNDLE-DEFAULT-ON-2026-06-25.md.
garden:
	@go run ./cmd/fak garden

garden-check:
	@go run ./cmd/fak garden --check

dogfood-recent:
	@python3 tools/recent_feature_dogfood.py

# dogfood-test (#10821): hermetic, fast test verifying the dogfood launcher's
# bounded-wait loops, dead-PID fail-fast logic, and provider graduation gates.
dogfood-test:
	@bash scripts/dogfood-claude_test.sh
	@echo "dogfood-test OK"

# test-integration (#10822): multi-process local loopback integration tests verifying
# real HTTP/SSE gateway proxying, tool call adjudication, and child supervision.
test-integration: build
	@echo "running integration test suite..."
	go test -count=1 -run "Integration|E2E" ./internal/gateway ./cmd/fak
	@echo "test-integration OK"

# performance-rsi-health: deterministic live-repository loop-health grade and
# named debt evidence from the issue-9768 dogfood receipt. This grades the
# measurement/improvement loop; it does not claim the parent 100x target.
performance-rsi-health:
	@go run ./cmd/fak performance-rsi-scorecard --input docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json --json

# vcache-gate: the repeatable vCache scorecard dogfood gate (#791). The recent-feature
# dogfood packet RUNS the score for daily visibility; this is the dedicated, fast,
# deterministic GATE that asserts the 2x scorecard floor on both paths (default is
# 2x-ready; an unreachable threshold fails) and exits non-zero on a regression. No
# network, no telemetry -- the planned star-anchor proof is deterministic.
vcache-gate:
	@python3 tools/vcache_scorecard_gate_test.py
	@python3 tools/vcache_scorecard_gate.py

# cache-default-readiness (#1568, cache-frontier Next-50 item 50 "Release gate"): the
# named release gate that blocks default-on cache claims. vcache-gate above asserts the
# 2x economics floor; this asserts the stronger "useful by default" bar — the
# vcachescore.DefaultReadiness fold fails CLOSED on the three regression classes item 50
# names: per-plane scoring (a provider-only, low default-usefulness verdict), cold-path
# correctness (a hit-dependent request path), and provenance labels (a plane whose
# WITNESSED/OBSERVED/FORECAST label collapses). The regression contract is pinned by
# internal/vcachescore/readiness_test.go — a passing witnessed-planes report stays
# default_ready, while each regression flips the verdict to "blocked". These tests already
# run inside `go test ./...`; naming them as a dedicated, deterministic (no network, no
# telemetry) gate makes "cache-default readiness" a discoverable release gate rather than
# an anonymous unit test. Exits non-zero if any regression stops blocking.
cache-default-readiness:
	@go test -count=1 ./internal/vcachescore/ -run '^TestDefaultReadiness'
	@echo "cache-default-readiness OK"

# gitdaily-score (#5587): the named, deterministic health gate for the daily lock-aware
# git-hygiene spine (#5577). `fak git-daily --score` grades the job's OWN `fak-git-daily/1`
# ledger -- adoption, outcome health, fold drift -- into an A-F grade with named evidence,
# so "is this job still good?" is a command rather than an operator diffing rows by eye.
# This target runs that same card against the CAPTURED REAL LEDGER pinned in
# internal/metrics/git_daily_health_test.go: TestGradeGitDailyHealthOnRealLedger asserts the
# graded output, and TestRealLedgerCaptureIsDerivedFromTheRawRows is the anti-self-report
# check that the pinned tally is what the raw rows actually fold to -- so the witness can
# never drift from the evidence it claims to summarize. Like cache-default-readiness above,
# these already run inside `go test ./...`; NAMING them is what makes the daily job's health
# a discoverable gate instead of an anonymous unit test. Deterministic: no network, no
# clock, no git. For this clone's own live grade, run `fak git-daily --score`.
gitdaily-score:
	@go test -count=1 ./internal/metrics/ -run 'RealLedger'
	@echo "gitdaily-score OK"

# model: export the real SmolLM2-135M weights the in-kernel engine (--engine inkernel)
# loads from FAK_MODEL_DIR. One-time; needs Python. See GETTING-STARTED.md §4b.
model:
	./scripts/fetch-model.sh

# logvault-drill: the restore-path cadence drill (#2453). Restores one captured
# source into a temp dir and re-verifies it (re-hash against the manifest chain +
# chained-journal verifiers), hermetic and self-contained, so `fak logvault`'s
# recoverability promise stays a witnessed capability instead of an untested
# hypothesis. Run on a cadence (nightly / a green gate); exits non-zero on any
# restore mismatch or refusal.
logvault-drill:
	./scripts/logvault-restore-drill.sh

# claims-lint (#6218): every "- [" line in CLAIMS.md carries exactly one honesty tag AND a
# machine-readable EXPOSURE state -- on by default, or gated WITH a stated reason. That is Q6
# of docs/standards/net-true-value.md, enforced by reusing internal/claimcheck's own Realized
# type + gradeRealized rule rather than a second vocabulary. The awk tag-counter this replaced
# could only count tags, so a [SHIPPED] line reading "reproducible now" could silently mean
# "reproducible now, once you set the right env var"; the go test also EMITS the default-off
# capability count instead of leaving it to a grep over prose.
claims-lint:
	@go test -count=1 -v -run 'TestCLAIMSLedger' ./internal/claimcheck/

# cache-headline-lint (#1564, cache-frontier Next-50 item 46 "Remove legacy"): a cache
# "win" headline -- "99% cache", "cache win" -- that omits which PLANE
# (provider/kernel/context/forecast) and provenance the number belongs to fails review.
# This is the standing guard behind epic #1490's honest per-mechanism attribution: once
# the provider-vs-kernel-vs-context split lands, an un-linted blended "99% cache" re-
# introduces exactly the legacy the epic removes. Narrow + false-positive-free by design
# (a fixed set of known legacy-headline shapes, cleared by ANY co-located plane/provenance
# label -- same discipline as check_provenance_labels). The unit test pins the pass/fail
# contract (unlabeled headline FAILS, labeled headline PASSES); the tree audit proves the
# committed corpus is clean. Pure Python, no dos/Go dependency, so it runs unconditionally.
cache-headline-lint:
	@python3 tools/check_cache_headlines_test.py
	@python3 tools/check_cache_headlines.py --audit-tree

# cachedoc-numbers-lint: the checking layer for RECENT-OPERATIONAL cachevalue docs
# (e.g. docs/integrations/fable5-more-usage-for-free.md). BENCHMARK-AUTHORITY.md is the
# SoT for COMMITTED benchmark claims and readme_freshness_audit guards the README, but a
# doc that reports this-week's `fak cachevalue report` fleet/dev telemetry is too live-
# moving to earn an authority row -- yet is exactly where a number rots or an arithmetic
# slip hides (the real one: "60 discovered -- 15 priced, 36 held out" read as 60=15+36).
# Each guarded doc gets a manifest under tools/docnumbers/ binding every rendered number
# to a trimmed FROZEN snapshot field + the arithmetic invariants the doc asserts, so the
# audit is fully hermetic: doc render == manifest expected == snapshot field, and the
# sums/formulas close. False-positive-free by design -- open-ended/live windows are
# checked against the frozen snapshot ONLY, never re-derived to equality (see the window
# taxonomy in the tool). The unit test pins the pass/fail contract; the tree audit proves
# the committed corpus is clean. Pure Python, no dos/Go dependency -- runs unconditionally.
cachedoc-numbers-lint:
	@python3 tools/cachedoc_numbers_audit_test.py
	@python3 tools/cachedoc_numbers_audit.py

# salience (dos-kernel docs/391): the first WIRED consumer of the `dos salience` verdict
# (it was built-but-latent — nothing routed on it; see the usefulness audit in
# docs/notes/DOS-SALIENCE-USEFULNESS-AUDIT-2026-06-24.md). It routes every CLAIMS.md claim
# through dos.salience.partition — [SHIPPED]→LIVE, [SIMULATED]/[STUB]→PARKED — asserts the
# no-loss invariant (nothing dropped ledger→fold) and cross-checks live/parked counts
# against the ledger, so a true-but-parked claim is never silently lost. Also a cross-repo
# regression sentinel: it pins the kernel's park-declared contract fak depends on. A real
# gate where the dos kernel is importable (this trunk, the fleet hosts); an advisory SKIP
# (exit 0) where it is not — so it joins `make ci` without breaking a box that lacks dos.
# Intentionally NOT a HARD ci.yml step (CI is hermetic — no dos kernel); the pure logic is
# gated there by tools/claims_salience_register_test.py.
salience:
	@python3 tools/claims_salience_register.py --check

# dos-lint (#1194): fak dogfoods DOS on itself via dos.toml, but nothing validated that
# file -- so lane-taxonomy defects accreted silently (a lane in [lanes].concurrent but not
# [lanes.trees] can't be arbitrated; a lane listed twice makes the roster spawn-order-
# sensitive; an autopick lane absent from concurrent is silently skipped). modelreg,
# nightrun, dojo and loopdrive each shipped half-wired and were caught only by a human
# reading a 500-line TOML. This wires `dos lint` into the green gate so the next one reds
# the build instead. Uses DEFAULT `dos lint` (gates on warn+error), NOT `--strict` (errors
# only): the recurring class is the CONCURRENT_LANES_OVERLAP *warning*, which --strict
# exempts. Real gate where the `dos` CLI is present (this trunk, the fleet hosts); an
# advisory SKIP (exit 0) where it is not (hermetic CI runners) -- the same dos-availability
# contract as the `salience` target above.
dos-lint:
	@if command -v dos >/dev/null 2>&1; then \
		dos lint; \
	else \
		echo "dos-lint: SKIP (dos CLI not on PATH)"; \
	fi

# index-sync (#511): the curated INDEX.md / llms.txt must not drift from the tree.
# Two gates: the reciprocal orphan gate (dangling links + unlisted dated notes) and
# the llms-full.txt check-mode drift gate. The orphan gate has a parity-proven Go twin
# (hooks.gateIndexSyncTree), so it runs via `fak hygiene --gates INDEX_SYNC` (no Python
# interpreter spawn) and falls back to the Python checker only on exit 2 (could-not-run).
# The grandfathered renderer is driven through fak llms-full so checks use committed bytes.
index-sync:
	@go build -o tools/.bin/fak ./cmd/fak
	@tools/.bin/fak hygiene --gates INDEX_SYNC; \
	rc=$$?; \
	if [ $$rc -eq 2 ]; then \
		echo "fak hygiene INDEX_SYNC could not run; falling back to Python"; \
		python3 tools/check_index_sync.py --audit-tree; \
	elif [ $$rc -ne 0 ]; then \
		exit $$rc; \
	fi
	@tools/.bin/fak llms-full --check
	@echo "index-sync OK"

# gofmt-check: every committed .go file is gofmt-formatted — the local mirror of ci.yml's
# HARD gofmt gate (G-001). Linux/WSL ONLY: a native-Windows checkout under
# core.autocrlf=true rewrites .go to CRLF, which `gofmt -l` would flag as a false positive
# (.gitattributes pins only *.sh/*.golden to LF), so scripts/ci.ps1 deliberately omits this
# and relies on the WSL `make ci` / CI for the canonical LF check. Fix with `gofmt -w .`.
#
# The scan is whole-tree; the REPORT and the exit condition are scoped (#6490). Declare the
# change under test in GOFMT_OWNED_PATHS (whitespace-separated repo-relative files or
# directories) and the findings split into two labelled groups: the files this change owns,
# which FAIL the gate, and pre-existing tree debt from the other sessions sharing this
# checkout, which is reported as a visible but non-fatal notice. With no scope declared the
# whole tree is the owned set — the pre-split behavior, preserved as the fallback. Scoping
# the STYLE check is safe because `build:` / `vet:` stay whole-tree: scoping can hide a
# stale format, never a real break (the same pairing `fak validate` uses).
#
#   make gofmt-check GOFMT_OWNED_PATHS="internal/foo cmd/bar/baz.go"
#
# The body lives in scripts/gofmt-check.sh so the gate is runnable and testable on its own
# (witnessed by internal/hooks/gofmt_check_script_test.go), the way `build:` delegates to
# scripts/build.sh.
disambiguation-generated-check:
	go run ./cmd/fak disambiguation generate --check --json

gofmt-check:
	@sh scripts/gofmt-check.sh

# hygiene: the deterministic, no-network repo-hygiene gates ci.yml runs HARD — doc
# placement, links, hosted-demo URLs, demo-command refs, browser-demo metadata, file admission, secret shapes —
# mirrored into the local gate so a pre-push `make ci` catches them. The --audit-tree
# checkers with a parity-proven Go twin (DOC_PLACEMENT, BROKEN_LINK, FILE_ADMISSION,
# SECRET_SHAPE, BRAND_CONSISTENCY, PROVENANCE_LABEL, HARDWARE_TELL, DEMO_COMMAND, BROWSER_CONTRACT,
# DEMO_LIVE_LINKS, GUARD_MCP_STATUS) now run in ONE process via `fak hygiene` — no per-checker Python
# interpreter spawn (~15-20s of process-create + Defender tax saved on Windows). Exit 2 = could-not-run,
# so we fall back to the Python checkers; exit 1 = a gate fired (HARD fail).
hygiene:
	@go build -o tools/.bin/fak ./cmd/fak
	@tools/.bin/fak hygiene --gates DOC_PLACEMENT,BROKEN_LINK,FILE_ADMISSION,SECRET_SHAPE,BRAND_CONSISTENCY,PROVENANCE_LABEL,HARDWARE_TELL,DEMO_COMMAND,BROWSER_CONTRACT,DEMO_LIVE_LINKS,GUARD_MCP_STATUS; \
	rc=$$?; \
	if [ $$rc -eq 2 ]; then \
		echo "fak hygiene could not run; falling back to Python gates"; \
		python3 tools/check_doc_placement.py --audit-tree && \
		python3 tools/check_links.py --audit-tree && \
		python3 tools/check_committed_files.py --audit-tree && \
		python3 tools/check_secret_shapes.py --audit-tree && \
		python3 tools/scrub_hardware_names.py --check && \
		python3 tools/demo_command_audit.py && \
		python3 tools/demo_browser_contract.py && \
		python3 tools/demo_live_links.py && \
		python3 tools/guard_mcp_status_audit.py; \
	elif [ $$rc -ne 0 ]; then \
		exit $$rc; \
	fi
	@go test ./internal/pythongate -run TestNoNewPythonTools
	@go test ./internal/windowgate -run TestTrackedTreeHasNoPopups
	@go test ./internal/benchlineagegate -run TestEveryBenchEmitterStampsLineage
	@go test ./internal/godfileceiling -run TestLiveTreeUnderCeiling
	@echo "hygiene OK"

# demo-audit: full local demo health gate. Static checks are network-free; dynamic checks
# bind only loopback and temp files. The optional published GitHub Pages drift check is
# separate because it depends on external deployment state.
demo-audit:
	@$(MAKE) --no-print-directory demo-tool-tests
	@python3 tools/demo_live_links.py
	@python3 tools/demo_command_audit.py
	@python3 tools/demo_browser_contract.py
	@$(MAKE) --no-print-directory demo-scorecards
	@python3 tools/demo_http_smoke.py --timeout 60
	@python3 tools/demo_headless_smoke.py --timeout 120
	@echo "demo-audit OK"

# demo-tool-tests: unit tests for the demo audit harnesses outside the scorecards.
# The scorecard unit tests stay in demo-scorecards so that target remains standalone.
demo-tool-tests:
	@python3 tools/demo_registry_test.py
	@python3 tools/demo_live_links_test.py
	@python3 tools/demo_command_audit_test.py
	@python3 tools/demo_browser_contract_test.py
	@python3 tools/demo_http_smoke_test.py
	@python3 tools/demo_headless_smoke_test.py
	@python3 tools/guard_mcp_status_audit_test.py
	@python3 tools/openai_live_prereq_audit_test.py
	@python3 tools/openai_hosted_live_pilot_test.py
	@python3 tools/claude_historical_guard_audit_test.py
	@echo "demo-tool-tests OK"

# demo-scorecards: per-card regression sentinels for every grade-A scorecard (#1423
# Part 1, the "stays Excellent" half of epic #1414). Each scorecard's bare run exits 1
# the moment it picks up even one defect, so a regression on a clean surface reds
# `make ci` here — the per-card complement to the portfolio grade ratchet
# (scorecard-ratchet), which catches a letter-slip across the whole family. Read-only
# and deterministic: no model, no network, no build. A card is wired once it is BOTH
# grade A AND debt 0; an indebted grade-A card (code-slop below; also agent-readiness,
# product, cuda-dev, popularization under the B1-B6 burndown) reds the gate on day 1, so
# it is added the moment its debt reaches 0 — never before (the trap #1268 #E names).
demo-scorecards:
	@python3 tools/demo_quality_scorecard_test.py
	@python3 tools/demo_quality_scorecard.py
	@python3 tools/demo_quality_scorecard.py --check-doc
	@python3 tools/demo_robustness_scorecard_test.py
	@python3 tools/demo_robustness_scorecard.py
	@python3 tools/demo_robustness_scorecard.py --check-doc
	@python3 tools/code_slop_scorecard_test.py
	@python3 tools/steerability_scorecard_test.py
	@python3 tools/steerability_scorecard.py >/dev/null
	@python3 tools/bench_dx_scorecard_test.py
	@python3 tools/bench_dx_scorecard.py >/dev/null
	@python3 tools/bench_dx_scorecard.py --check-doc
	@python3 tools/benchmark_authority_test.py
	@python3 tools/benchmark_authority.py --check docs/benchmarks/AUTHORITY-GENERATED-SAMPLE.md
	@python3 tools/intent_literal_scorecard_test.py
	@python3 tools/intent_literal_scorecard.py >/dev/null
	@python3 tools/stability_scorecard_test.py
	@python3 tools/stability_scorecard.py >/dev/null
	@python3 tools/observability_scorecard_test.py
	@python3 tools/observability_scorecard.py >/dev/null
	@python3 tools/rsi_maturity_scorecard_test.py
	@python3 tools/rsi_maturity_scorecard.py >/dev/null
	@python3 tools/doc_appeal_scorecard_test.py
	@python3 tools/doc_appeal_scorecard.py >/dev/null
	@python3 tools/persona_readiness_scorecard_test.py
	@python3 tools/persona_readiness_scorecard.py >/dev/null
	@python3 tools/persona_fit_scorecard_test.py
	@python3 tools/persona_fit_scorecard.py >/dev/null
	@python3 tools/concept_disambiguation_scorecard_test.py
	@python3 tools/concept_disambiguation_scorecard.py >/dev/null
	@python3 tools/docs_scorecard_test.py
	@python3 tools/docs_scorecard.py >/dev/null
	@python3 tools/sota_coverage_scorecard_test.py
	@python3 tools/sota_coverage_scorecard.py >/dev/null
	@echo "demo-scorecards OK"
# code-slop scorecard: only the unit-test line gates here for now. The bare run +
# --check-doc both exit 1 while slop-debt > 0 (the scorecard reports honestly), so
# wiring them would red-gate `make ci` immediately. Add them once slop-debt is driven
# to 0; until then the scorecard is run on demand / via the control pane (the slop epic).

# scorecard-ratchet: the portfolio floor, not a standalone README-only red gate.
# The README freshness card rides scorecard_control_pane.py --check with
# readme_debt pinned at 0, so a front-page regression reds through the existing
# green ratchet (#779/#893): debt may hold or fall, never rise.
#
# The "stays Excellent" gate (#1423, epic #1414): --check now enforces the GRADE
# axis, not only the raw-unit sum. The pinned baseline (tools/scorecard_baseline.json)
# carries per-metric grade_weights, so a scorecard slipping a letter (A->B) reds the
# build EVEN WHEN total_debt held flat — the regression a flat raw total would hide.
# HARD by default; FAK_SCORECARD_GRADE_RATCHET=0 demotes it to advisory for a
# deliberate one-off pin on a known-dirty tree. Mirror of the milestone climb
# ratchet (#1442). Re-pin both axes with `--pin` after a real drop.
scorecard-ratchet:
	@python3 tools/readme_freshness_audit_test.py
	@python3 tools/readme_freshness_audit.py --json >/dev/null
	@python3 tools/scorecard_control_pane_test.py
	@python3 tools/scorecard_control_pane.py --check
	@echo "scorecard-ratchet OK"

# cache-proving: the real-session regression floor for the managed-cache levers
# (epic #1844). Validates the durable nightrun ledgers row-by-row (schema, closed
# mechanism vocabulary, the 0.9r-0.25c savings identity), folds every cache concept
# to a rung on the evidence ladder, and ratchets against the pinned baseline
# (tools/managed_cache_proving_ground.data/baseline.json): counts may only grow,
# rungs may only climb, violations may only shrink. Read-only and deterministic —
# concurrent sessions appending ledger rows keep it green; a ledger rewrite, schema
# drift, or a lever losing its durable witness turns it red. Re-pin after an
# intentional rung climb with `--write-baseline`.
cache-proving:
	@python3 tools/managed_cache_proving_ground_test.py
	@python3 tools/managed_cache_proving_ground.py --check
	@echo "cache-proving OK"

# demo-smoke: dynamic but hermetic. It builds the browser demos into a temp dir,
# starts them on loopback, mounts each behind its documented base path, verifies
# the page and a lightweight JSON API, then tears the process down.
demo-smoke:
	@python3 tools/demo_http_smoke.py --timeout 60
	@echo "demo-smoke OK"

# demo-headless-smoke: dynamic but hermetic. It runs the documented model-free
# terminal witnesses and checks each emits its pinned invariant.
demo-headless-smoke:
	@python3 tools/demo_headless_smoke.py --timeout 120
	@echo "demo-headless-smoke OK"

# demo-live-status: optional network witness for the live VM, rendered as a compact
# hosted/local-only HTTP/API/HTTPS table.
demo-live-status:
	@python3 tools/demo_live_links.py --live --timeout 8 --status

# demo-https-status: optional strict network witness for launch surfaces that need
# hosted demos embeddable from HTTPS pages. It exits non-zero until TLS termination exists.
demo-https-status:
	@python3 tools/demo_live_links.py --live --timeout 8 --require-https --status

# demo-published-status: optional network witness for the public HTTPS Pages copy,
# rendered as the same compact table. It exits non-zero while Pages is stale.
demo-published-status:
	@python3 tools/demo_live_links.py --published --timeout 12 --status

# demo-published-check: optional network witness for the public HTTPS Pages copy. It
# fails while Pages is stale or the remote share image/hosted links drift.
demo-published-check:
	@python3 tools/demo_live_links.py --published --timeout 12

# demo-readiness-status: optional all-up deployment view. It runs static, live VM,
# strict HTTPS, and published Pages checks and exits non-zero until every view is clean.
demo-readiness-status:
	@python3 tools/demo_live_links.py --readiness --timeout 8

# gated-tests: the tool-test no-blackhole runner — the systemic backstop under the
# hand-enumerated tool-test steps in ci.yml. It discovers EVERY tools/*_test.py, runs the
# hermetic ones the enumerated steps don't (so a new test can never silently go un-run),
# and asserts the quarantine manifest is internally consistent. Mirrors ci.yml's HARD step
# so a pre-push `make ci` catches a tool-test regression locally. Runs its own unit tests
# first, then --check (pure fs), then --run (the ~88 hermetic tests). Linux/WSL like the
# other python gates — a few tests are platform-divergent, so scripts/ci.ps1 runs --check only.
gated-tests:
	@python3 tools/gated_tool_tests_test.py
	@python3 tools/gated_tool_tests.py --check
	@python3 tools/gated_tool_tests.py --run

# ---- CUDA dev loop (see docs/cuda-dev.md) ------------------------------------------------
# cuda-check: the GPU-FREE local CUDA gate. ABI parity plus standalone cuda_backend.h
# portability — no nvcc, no GPU, no cgo — so it runs anywhere with bash/python; when a
# host C compiler exists, build_cuda.sh check also runs a strict header-only parse. The
# no-toolchain Windows host keeps the python mirror in scripts/ci.ps1, and this remains
# the local twin of cuda-build.yml's `static` job. A GPU host adds `go vet -tags cuda`
# for the cgo type-check.
cuda-check:
	@bash internal/compute/build_cuda.sh check
	@echo "cuda-check OK (ABI/header preflight; run 'make cuda-build' on a CUDA host for the nvcc build)"

# cuda-build / cuda-test: the REAL -tags cuda build + Approx witness — REQUIRES a CUDA
# toolchain (nvcc + cuBLAS). Delegates to internal/compute/build_cuda.sh; Linux/WSL/GPU node
# only (the win32 dev host has none — see docs/cuda-dev.md for the no-sudo WSL setup).
cuda-build:
	bash internal/compute/build_cuda.sh build

# First-class single-arch Blackwell developer builds (#4182). These keep the
# fast one-architecture path; the multi-arch distributable is owned by #4183.
cuda-build-sm100:
	FAK_CUDA_ARCH=sm_100 bash internal/compute/build_cuda.sh build

cuda-build-sm120:
	FAK_CUDA_ARCH=sm_120 bash internal/compute/build_cuda.sh build

cuda-test:
	bash internal/compute/build_cuda.sh test

# cuda-accept: run EVERY on-GPU acceptance witness with one verdict (SKIP-is-not-PASS) via
# tools/cuda_acceptance.sh. GPU node only; a no-GPU host exits non-zero (never a false green).
cuda-accept:
	bash tools/cuda_acceptance.sh

# cuda-occupancy (#4188): print the decode-shaped per-kernel occupancy + HBM-traffic witness
# (internal/compute/decode_occupancy.go) — the file-after-measurement gate for the KernelWiki
# Blackwell decode shortlist (Cluster G, docs/notes/2026-07-10-kernelwiki-study.md). The verdict
# arm is EXACT (grid-block vs SM counts, operand-byte counts; no timer), so it prints real numbers
# on ANY host — nothing skips, so there is no skip-as-pass to launder. The DEVICE corroboration
# (ncu achieved-occupancy / DRAM-%) is the separate GPU-node harness
# tools/dgx_decode_occupancy_ncu.sh, and the report's last line states it was NOT run here rather
# than claiming it (same honesty discipline as cuda-accept). Baseline + per-candidate verdicts:
# docs/notes/2026-07-11-decode-occupancy-witness-measurement.md.
cuda-occupancy:
	go test ./internal/compute -run TestDecodeOccupancyWitnessReport -count=1 -v

# negframe-ratchet (#3545): the diff-scoped negation-framing gate — the CI twin of the
# `fak score negframe` scorecard. `--since $(BASE)` scans the agent-steer prose corpus
# (AGENTS.md, CLAUDE.md, the skills) and exits 1 ONLY when the working tree vs BASE
# introduces a NEW mechanical (confidently reframable) negative — pre-existing debt never
# reds here (that is the scorecard's job), so a change is gated only on the negativity it
# itself adds. BASE defaults to origin/main; override per-invocation, e.g.
# `make negframe-ratchet BASE=HEAD~1` (CI passes the PR base SHA — see
# .github/workflows/negframe-ratchet.yml, which checks out with fetch-depth 0 so the
# ratchet's `git show <base>:<path>` can read the prior copy of each steer-prose doc).
BASE ?= origin/main
.PHONY: negframe-ratchet
negframe-ratchet:
	@go run ./cmd/fak score negframe --since $(BASE)
