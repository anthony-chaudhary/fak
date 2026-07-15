package main

import (
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

var hygienePushDelta = committedPushPaths

// scopeHygieneFindingsToPush mirrors architest's #2088 push-delta rule for the
// whole-tree hygiene path. A peer's undeclared leaf stays visible but advisory;
// only a leaf delivered by this push can hard-refuse it. If git/trunk read-back is
// unavailable, findings remain blocking (safe standalone/archive fallback).
func scopeHygieneFindingsToPush(root string, findings []hooks.Finding) []hooks.Finding {
	paths, scoped := hygienePushDelta(root)
	return hooks.ScopeTierDeclaredFindings(findings, paths, scoped)
}

func committedPushPaths(root string) ([]string, bool) {
	mergeBase, err := exec.Command("git", "-C", root, "merge-base", "origin/main", "HEAD").Output()
	if err != nil || strings.TrimSpace(string(mergeBase)) == "" {
		return nil, false
	}
	base := strings.TrimSpace(string(mergeBase))
	out, err := exec.Command("git", "-C", root, "diff", "--name-only", base+"..HEAD", "--", "internal/").Output()
	if err != nil {
		return nil, false
	}
	var paths []string
	for _, line := range strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			paths = append(paths, line)
		}
	}
	return paths, true
}
