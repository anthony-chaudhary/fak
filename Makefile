# Makefile — portable build/test entrypoints (unit 12). On Windows without make,
# use scripts/ci.ps1, which this mirrors.
.PHONY: ci build vet test test-fast bench claims-lint salience index-sync model gofmt-check hygiene demo-audit demo-tool-tests demo-scorecards demo-smoke demo-headless-smoke demo-live-status demo-https-status demo-published-status demo-published-check demo-readiness-status gated-tests cuda-check cuda-build cuda-test cuda-accept

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
# cuda-check is the GPU-free CUDA ABI parity gate — deterministic, no toolchain, so it joins
# the local gate the same way (the cuda-build.yml `static` job is its CI mirror).
ci: build gofmt-check vet test claims-lint salience index-sync hygiene demo-tool-tests demo-scorecards demo-smoke demo-headless-smoke gated-tests cuda-check
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

test:
	go test ./...

# test-fast: the 2s smoke tier — the synthetic + architest invariants only.
# `-short` skips the weight-backed model witnesses (the ~538MB f32/safetensors
# loads that are the slow + OOM-prone part of the WSL suite — see fak/test.sh and
# the model-test OOM note). That is ~95% of logic regressions in seconds, so it is
# the right floor for a pre-commit / pre-push gate. Pair `build vet` with it so a
# commit that doesn't compile or vet-clean is caught at the same gate. The full
# `test` target (no -short) still runs the real oracle locally + in CI.
test-fast: build vet
	go test -short ./...
	@echo "test-fast OK (smoke tier; run 'make test' for the weight-backed witnesses)"

bench:
	go build -o fak ./cmd/fak && ./fak bench --suite tau2-smoke --out report.json

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

# index-sync (#511): the curated INDEX.md / llms.txt must not drift from the tree.
# Two gates: the reciprocal orphan gate (dangling links + unlisted dated notes) and
# the llms-full.txt check-mode drift gate. Python (like claims-lint), so it sits here.
index-sync:
	@python3 tools/check_index_sync.py --audit-tree
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
# mirrored into the local gate so a pre-push `make ci` catches them. Pure-stdlib python
# (like claims-lint / index-sync), fast, and audits the git-tracked tree (so peer
# working-tree WIP doesn't affect it).
hygiene:
	@python3 tools/check_doc_placement.py --audit-tree
	@python3 tools/check_links.py --audit-tree
	@python3 tools/demo_live_links.py
	@python3 tools/demo_command_audit.py
	@python3 tools/demo_browser_contract.py
	@python3 tools/check_committed_files.py --audit-tree
	@python3 tools/check_secret_shapes.py --audit-tree
	@python3 tools/check_brand_consistency.py --audit-tree
	@python3 tools/scrub_hardware_names.py --check
	@python3 tools/check_provenance_labels.py --audit-tree
	@python3 tools/guard_mcp_status_audit.py
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
	@echo "demo-scorecards OK"

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
# cuda-check: the GPU-FREE local CUDA gate. Pure-text ABI parity (header <-> kernels <-> cgo
# binding) — no nvcc, no GPU, no cgo — so it runs ANYWHERE, including the no-toolchain Windows
# host (scripts/ci.ps1 mirrors it) and inside `make ci`. The local twin of cuda-build.yml's
# `static` job. A GPU host adds `go vet -tags cuda` for the cgo type-check.
cuda-check:
	@python3 tools/cuda_abi_parity.py --check
	@echo "cuda-check OK (ABI parity; run 'make cuda-build' on a CUDA host for the nvcc build)"

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
