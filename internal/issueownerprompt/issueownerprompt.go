// Package issueownerprompt validates that issue resolver goal prompts faithfully
// compose the canonical issue owner lifecycle without drift or private copies.
//
// Invariant: issue owner prompt resolution is fail-closed and deterministic.
// Precondition: the target directory must exist and contain the canonical lifecycle file.
// Guard: missing lifecycle files, unapproved resolvers, or drifted invariants fail validation immediately.
package issueownerprompt

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// LifecycleFile is the canonical filename defining the binding issue owner lifecycle.
	LifecycleFile = "issue-owner-lifecycle.md"
	includeLine   = "Read `.claude/goal-prompts/issue-owner-lifecycle.md` now and obey it as the binding `ISSUE_OWNER` / `LEAF_CHILD` lifecycle."
)

var lifecycleInvariants = []string{
	"implement the root",
	"`BOUNDED`, `BROAD`, or `BLOCKED_EXTERNAL`",
	"`BOUNDED` owners do not launch children; they begin root edits",
	"only one delegation level is allowed",
	"the next action is root implementation, not another launch mechanism",
	"Witness effects independently",
	"Refusal and park contract",
	"Orphan child closeout remains mandatory",
	"every owned child, lease, intent, process, and effect",
}

var deltaInvariants = []string{
	"root implementation",
	"executable work-shape classification",
	"one-level admitted delegation",
	"independent witness",
	"park semantics",
	"exhaustive child/lease/intent closeout",
}

var obsoleteLifecycle = []string{
	"claim ONE menu item",
	"resolve it, ship it WITNESSED, then stop",
	"one witnessed menu item is a complete",
	"take ONE lane",
	"top ready leaf",
}

// ValidateDir verifies that every tracked issue resolver composes the canonical
// owner lifecycle instead of carrying a private copy that can drift.
func ValidateDir(dir string) error {
	canonical, err := os.ReadFile(filepath.Join(dir, LifecycleFile))
	if err != nil {
		return fmt.Errorf("read canonical lifecycle: %w", err)
	}
	for _, invariant := range lifecycleInvariants {
		if !strings.Contains(string(canonical), invariant) {
			return fmt.Errorf("canonical lifecycle missing %q", invariant)
		}
	}

	matches, err := filepath.Glob(filepath.Join(dir, "resolve-*-issue-*.md"))
	if err != nil {
		return err
	}
	sort.Strings(matches)
	if len(matches) == 0 {
		return fmt.Errorf("no tracked issue resolvers in %s", dir)
	}
	for _, path := range matches {
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		body := string(raw)
		if firstLine(body) != includeLine {
			return fmt.Errorf("%s must compose %s on its first line", filepath.Base(path), LifecycleFile)
		}
		if !strings.Contains(body, "## Domain delta:") {
			return fmt.Errorf("%s must contain a named domain delta after the lifecycle include", filepath.Base(path))
		}
		for _, invariant := range deltaInvariants {
			if !strings.Contains(body, invariant) {
				return fmt.Errorf("%s missing lifecycle drift assertion %q", filepath.Base(path), invariant)
			}
		}
		for _, stale := range obsoleteLifecycle {
			if strings.Contains(body, stale) {
				return fmt.Errorf("%s copies obsolete lifecycle %q", filepath.Base(path), stale)
			}
		}
	}
	return nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSuffix(s[:i], "\r")
	}
	return strings.TrimSuffix(s, "\r")
}
