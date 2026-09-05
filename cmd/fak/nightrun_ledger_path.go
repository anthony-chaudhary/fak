package main

import (
	"os"
	"path/filepath"
	"strings"
)

const ledgerRootEnv = "FAK_LEDGER_ROOT"

// nightrunLedgerPath anchors a repo-relative live-ledger constant to the primary
// checkout, even when fak is running from a linked worker worktree. Linked
// worktrees share the primary checkout's .git directory; resolving that common
// directory prevents each worker from forking an append-only ledger that would
// later collide at the land seam (#3208).
//
// FAK_LEDGER_ROOT is the explicit control-plane override. Without it, the linked
// worktree's .git/commondir metadata identifies the primary checkout without a
// subprocess on this hot append path. If neither is available we retain the
// historical repoRoot() behavior, including hermetic executions outside Git.
func nightrunLedgerPath(rel string) string {
	p := filepath.Join(nightrunLedgerRoot(repoRoot()), filepath.FromSlash(rel))
	if dir := filepath.Dir(p); dir != "" && dir != "." {
		_ = os.MkdirAll(dir, 0o755)
	}
	return p
}

func nightrunLedgerRoot(localRoot string) string {
	if explicit := strings.TrimSpace(os.Getenv(ledgerRootEnv)); explicit != "" && filepath.IsAbs(explicit) {
		return filepath.Clean(explicit)
	}
	if ws := strings.TrimSpace(os.Getenv("FAK_WORKSPACE_ROOT")); ws != "" && filepath.IsAbs(ws) {
		if _, err := os.Stat(filepath.Join(ws, "go.mod")); err == nil {
			return filepath.Clean(ws)
		}
	}
	localRoot = filepath.Clean(localRoot)
	gitMarker := filepath.Join(localRoot, ".git")
	if info, err := os.Stat(gitMarker); err == nil && info.IsDir() {
		return localRoot // the primary checkout already owns the common directory
	}
	marker, err := os.ReadFile(gitMarker)
	if err != nil {
		return localRoot
	}
	gitDirText := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(string(marker)), "gitdir:"))
	if gitDirText == "" {
		return localRoot
	}
	gitDir := gitDirText
	if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(localRoot, gitDir)
	}
	commonText, err := os.ReadFile(filepath.Join(filepath.Clean(gitDir), "commondir"))
	if err != nil {
		return localRoot
	}
	common := strings.TrimSpace(string(commonText))
	if !filepath.IsAbs(common) {
		common = filepath.Join(gitDir, common)
	}
	common = filepath.Clean(common)
	if filepath.Base(common) != ".git" {
		return localRoot
	}
	return filepath.Dir(common)
}
