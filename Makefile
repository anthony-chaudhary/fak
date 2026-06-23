# Makefile — portable build/test entrypoints (unit 12). On Windows without make,
# use scripts/ci.ps1, which this mirrors.
.PHONY: ci build vet test test-fast bench claims-lint index-sync model

ci: build vet test claims-lint index-sync
	@echo "CI OK"

build:
	go build ./...

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
