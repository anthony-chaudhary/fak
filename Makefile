# Makefile — portable build/test entrypoints (unit 12). On Windows without make,
# use scripts/ci.ps1, which this mirrors.
.PHONY: ci build vet test test-fast bench claims-lint index-sync model gofmt-check hygiene

# ci is THE local green gate (AGENTS.md: "Green = make ci"). It must stay aligned with
# .github/workflows/ci.yml's HARD steps so a pre-push `make ci` fails on the same things
# GH does, instead of a developer only discovering a gofmt/hygiene break after the push.
# gofmt-check + hygiene are the deterministic, no-network CI gates that were previously
# CI-only — wired in here to close that local↔CI drift. (Network/range gates — leak-scan,
# dos-review — stay CI/githook-only; the release-substrate suite stays CI-only by weight.)
ci: build gofmt-check vet test claims-lint index-sync hygiene
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
# placement, links, file admission, secret shapes — mirrored into the local gate so a
# pre-push `make ci` catches them. Pure-stdlib python (like claims-lint / index-sync),
# fast, and audits the git-tracked tree (so peer working-tree WIP doesn't affect it).
hygiene:
	@python3 tools/check_doc_placement.py --audit-tree
	@python3 tools/check_links.py --audit-tree
	@python3 tools/check_committed_files.py --audit-tree
	@python3 tools/check_secret_shapes.py --audit-tree
	@python3 tools/scrub_hardware_names.py --check
	@echo "hygiene OK"
