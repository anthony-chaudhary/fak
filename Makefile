# Makefile — portable build/test entrypoints (unit 12). On Windows without make,
# use scripts/ci.ps1, which this mirrors.
.PHONY: ci build vet architest-gate test test-fast test-affected test-durations test-race bench status status-check release-staleness release-staleness-check release-readiness garden garden-check dogfood-recent vcache-gate claims-lint salience dos-lint index-sync model gofmt-check hygiene demo-audit demo-tool-tests demo-scorecards scorecard-ratchet demo-smoke demo-headless-smoke demo-live-status demo-https-status demo-published-status demo-published-check demo-readiness-status gated-tests cuda-check cuda-build cuda-test cuda-accept

VERIFY_LOOP_BUDGET ?= 30s
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
ci: build gofmt-check vet test claims-lint salience dos-lint index-sync hygiene demo-tool-tests demo-scorecards scorecard-ratchet demo-smoke demo-headless-smoke gated-tests cuda-check
	@echo "CI OK"

build:
	go build ./...
	# Drop the repo-guard PreToolUse hook binary into the (gitignored) tools/.bin so
	# the Claude Code hook runs the Go guard instead of spawning Python per tool call
	# (.claude/settings.json prefers this binary, falling back to tools/repo_guard.py
	# until it exists). Part of `build` so every green gate self-propagates it.
	go build -o tools/.bin/repoguard ./cmd/repoguard
	# Likewise the Go DOS dispatch-worker launcher — the interpreter-free cutover
	# target for tools/dispatch_worker.py (parity-tested; see dos.toml [supervise]).
	go build -o tools/.bin/dispatchworker ./cmd/dispatchworker

vet:
	go vet ./...

architest-gate:
	go test -count=1 ./internal/architest/ -run '$(ARCHITEST_GATE_RE)'

test: architest-gate
	go test ./...

# test-fast: the 2s smoke tier — the synthetic + architest invariants only.
# `-short` skips the weight-backed model witnesses (the ~538MB f32/safetensors
# loads that are the slow + OOM-prone part of the WSL suite — see fak/test.sh and
# the model-test OOM note). That is ~95% of logic regressions in seconds, so it is
# the right floor for a pre-commit / pre-push gate. Pair `build vet` with it so a
# commit that doesn't compile or vet-clean is caught at the same gate. The full
# `test` target (no -short) still runs the real oracle locally + in CI.
test-fast: build vet architest-gate
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

bench:
	go build -o fak ./cmd/fak && ./fak bench --suite tau2-smoke --out report.json

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
	@python3 tools/release_readiness_scorecard.py

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

# vcache-gate: the repeatable vCache scorecard dogfood gate (#791). The recent-feature
# dogfood packet RUNS the score for daily visibility; this is the dedicated, fast,
# deterministic GATE that asserts the 2x scorecard floor on both paths (default is
# 2x-ready; an unreachable threshold fails) and exits non-zero on a regression. No
# network, no telemetry -- the planned star-anchor proof is deterministic.
vcache-gate:
	@python3 tools/vcache_scorecard_gate_test.py
	@python3 tools/vcache_scorecard_gate.py

# model: export the real SmolLM2-135M weights the in-kernel engine (--engine inkernel)
# loads from FAK_MODEL_DIR. One-time; needs Python. See GETTING-STARTED.md §4b.
model:
	./scripts/fetch-model.sh

# claims-lint: every "- [" line in CLAIMS.md carries exactly one tag.
claims-lint:
	@awk '/^- \[/{n=0; if(index($$0,"[SHIPPED]"))n++; if(index($$0,"[SIMULATED]"))n++; if(index($$0,"[STUB]"))n++; c++; if(n!=1){print "VIOLATION:",$$0; bad++}} END{printf "claims-lint: %d lines, %d violations\n",c,bad; if(bad>0||c==0)exit 1}' CLAIMS.md

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
# gen_llms_full.py is not yet ported (#928), so it stays on Python.
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
	@python3 tools/gen_llms_full.py --check
	@echo "index-sync OK"

# gofmt-check: every committed .go file is gofmt-formatted — the local mirror of ci.yml's
# HARD gofmt gate (G-001). Linux/WSL ONLY: a native-Windows checkout under
# core.autocrlf=true rewrites .go to CRLF, which `gofmt -l` would flag as a false positive
# (.gitattributes pins only *.sh/*.golden to LF), so scripts/ci.ps1 deliberately omits this
# and relies on the WSL `make ci` / CI for the canonical LF check. Fix with `gofmt -w .`.
gofmt-check:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt: not formatted (run 'gofmt -w .' from the repo root):"; \
		echo "$$unformatted"; exit 1; \
	fi; \
	echo "gofmt: clean"

# hygiene: the deterministic, no-network repo-hygiene gates ci.yml runs HARD — doc
# placement, links, hosted-demo URLs, demo-command refs, browser-demo metadata, file admission, secret shapes —
# mirrored into the local gate so a pre-push `make ci` catches them. The seven --audit-tree
# checkers with a parity-proven Go twin (DOC_PLACEMENT, BROKEN_LINK, FILE_ADMISSION,
# SECRET_SHAPE, BRAND_CONSISTENCY, PROVENANCE_LABEL, HARDWARE_TELL) now run in ONE process via
# `fak hygiene` — no per-checker Python interpreter spawn (~15-20s of process-create + Defender
# tax saved on Windows). Exit 2 = could-not-run, so we fall back to the Python checkers;
# exit 1 = a gate fired (HARD fail). The remaining checkers (demo_* x3, guard_mcp_status_audit)
# are not yet ported (#928) and stay on Python.
hygiene:
	@go build -o tools/.bin/fak ./cmd/fak
	@tools/.bin/fak hygiene --gates DOC_PLACEMENT,BROKEN_LINK,FILE_ADMISSION,SECRET_SHAPE,BRAND_CONSISTENCY,PROVENANCE_LABEL,HARDWARE_TELL; \
	rc=$$?; \
	if [ $$rc -eq 2 ]; then \
		echo "fak hygiene could not run; falling back to Python gates"; \
		python3 tools/check_doc_placement.py --audit-tree && \
		python3 tools/check_links.py --audit-tree && \
		python3 tools/check_committed_files.py --audit-tree && \
		python3 tools/check_secret_shapes.py --audit-tree && \
		python3 tools/check_brand_consistency.py --audit-tree && \
		python3 tools/check_provenance_labels.py --audit-tree && \
		python3 tools/scrub_hardware_names.py --check; \
	elif [ $$rc -ne 0 ]; then \
		exit $$rc; \
	fi
	@python3 tools/demo_live_links.py
	@python3 tools/demo_command_audit.py
	@python3 tools/demo_browser_contract.py
	@python3 tools/guard_mcp_status_audit.py
	@go test ./internal/pythongate -run TestNoNewPythonTools
	@go test ./internal/windowgate -run TestTrackedTreeHasNoPopups
	@go test ./internal/benchlineagegate -run TestEveryBenchEmitterStampsLineage
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
	@python3 tools/check_provenance_labels_test.py
	@python3 tools/guard_mcp_status_audit_test.py
	@python3 tools/openai_live_prereq_audit_test.py
	@python3 tools/openai_hosted_live_pilot_test.py
	@python3 tools/claude_historical_guard_audit_test.py
	@echo "demo-tool-tests OK"

# demo-scorecards: content-only demo quality and robustness scorecards. These are
# read-only and deterministic: no model, no network, no build.
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
	@python3 tools/intent_literal_scorecard_test.py
	@python3 tools/intent_literal_scorecard.py >/dev/null
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
	@python3 tools/scorecard_control_pane_test.py
	@python3 tools/scorecard_control_pane.py --check
	@echo "scorecard-ratchet OK"

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

cuda-test:
	bash internal/compute/build_cuda.sh test

# cuda-accept: run EVERY on-GPU acceptance witness with one verdict (SKIP-is-not-PASS) via
# tools/cuda_acceptance.sh. GPU node only; a no-GPU host exits non-zero (never a false green).
cuda-accept:
	bash tools/cuda_acceptance.sh
