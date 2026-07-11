package main

import (
	"os"
	"path/filepath"
)

// nightrunLedgerPath anchors a repo-relative live-ledger constant (normally under
// .fak/nightrun) to the repo root regardless of process cwd. Without the anchor, a
// guard/serve run from a subdir would fork a shadow state tree under that subdir.
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
