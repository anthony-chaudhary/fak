package main

import (
	"os"
	"path/filepath"
)

// nightrunLedgerPath anchors a docs/nightrun-relative ledger constant (e.g.
// harnessres.DefaultLedgerRel = "docs/nightrun/harness-resources.jsonl") to the
// repo root so a telemetry append lands in the real docs/nightrun regardless of the
// process cwd. Without this, a guard/serve run whose cwd is a subdir — most visibly
// `go test ./cmd/fak`, whose cwd is the cmd/fak package dir — resolves the relative
// path against that cwd and forks a SHADOW tree (cmd/fak/docs/nightrun/*.jsonl) that
// then sits tracked-but-perpetually-dirty and re-appends on every test run.
//
// It reuses repoRoot() (a go.mod upward walk, no git subprocess — the same anchor
// knownBadLedgerPath and program.go already use for this ledger class). From the repo
// root the join is a byte-for-byte no-op; from any subdir inside the module it now
// resolves the true root; outside any module repoRoot() falls back to cwd, so a run
// with no repo (a hermetic t.TempDir() cwd) writes under that temp dir and never
// touches the repo. The parent dir is created because the gatewayusageledger /
// cachevalueledger Append helpers do not MkdirAll themselves.
func nightrunLedgerPath(rel string) string {
	p := filepath.Join(repoRoot(), filepath.FromSlash(rel))
	if dir := filepath.Dir(p); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	return p
}
