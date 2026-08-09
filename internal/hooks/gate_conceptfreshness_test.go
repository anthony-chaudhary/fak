package hooks

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

// TestConceptFreshnessDetailNamesTheTreeScopedCure pins the half of #5829 that lives in
// the refusal text: this gate scores a git tree, so the command it prints has to be the
// one that regenerates from that same tree.
//
// The pre-#5829 detail interpolated res.Regenerate into "run `%s`", and for a CheckGitTree
// result that constant was the WORKTREE command - so the refusal instructed the operator to
// regenerate from a tree the gate had never looked at, and the retry refused again with the
// same finding. The negative assertion below is the actual regression pin: it is what fails
// if someone re-points the printed cure at the worktree generator.
func TestConceptFreshnessDetailNamesTheTreeScopedCure(t *testing.T) {
	const scored = "8ba5d4305066bd10407690b763013e846d914d93"
	res := conceptcatalog.FreshnessResult{
		StalePaths: []string{conceptcatalog.GeneratedIndex},
		Regenerate: conceptcatalog.RegenerateStagedCommandFor(scored),
	}
	got := conceptFreshnessDetail(res)

	if !strings.Contains(got, conceptcatalog.RegenerateStagedCommand) {
		t.Fatalf("detail must print the tree-scoped cure %q, got: %s", conceptcatalog.RegenerateStagedCommand, got)
	}
	// The SHA is the load-bearing part. Without it the operator retypes the command into a
	// shell whose index is not the tree that refused, and the retry refuses again (#5829).
	if !strings.Contains(got, scored) {
		t.Fatalf("detail must name the tree that was scored, or the cure is not portable out of the hook: %s", got)
	}
	// The worktree command may be NAMED (the detail calls it out as the wrong one), but it
	// must never be the thing the operator is told to run.
	if strings.Contains(got, "run `"+conceptcatalog.RegenerateCommand+"`") {
		t.Fatalf("detail prescribes the worktree regeneration, which cannot clear a git-tree refusal (#5829): %s", got)
	}
	if !strings.Contains(got, "STAGED GIT TREE") {
		t.Fatalf("detail must say which tree was scored, or the operator cannot tell why the worktree regen failed to clear it: %s", got)
	}
	if !strings.Contains(got, conceptcatalog.GeneratedIndex) {
		t.Fatalf("detail must name the artifact that actually drifted: %s", got)
	}
}

// TestConceptFreshnessDetailDistinguishesTheTwoCommands guards the property that makes the
// message readable: the two commands are different strings. If a future refactor collapses
// RegenerateStagedCommand into RegenerateCommand, the detail would say "run X ... X answers
// a different tree" - self-contradictory, and the gate would be unclearable again.
func TestConceptFreshnessDetailDistinguishesTheTwoCommands(t *testing.T) {
	if conceptcatalog.RegenerateStagedCommand == conceptcatalog.RegenerateCommand {
		t.Fatal("the tree-scoped and worktree cures must stay distinct commands (#5829)")
	}
}
