package gitgate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The trunk law bans rewriting shared history, and `git rebase` was refused
// CATEGORICALLY to enforce it — including `git rebase --abort`, which is the UNDO
// of a rebase, not one. That made the law self-refuting: a checkout already
// mid-rebase (a human, another tool, or a `pull.rebase=true` config can all put it
// there, and this argv-only rung sees none of them) had no sanctioned exit, so the
// shared tree stayed parked in a conflicted detached-HEAD state that fails every
// peer's commit. Refusing the repair REACHED the fatal outcome the law exists to
// prevent. Witnessed live in .dispatch-runs/guard-audit as the NEVER_AMEND_SHARED
// deny class (17 of 91 POLICY_BLOCKs) on resolver/rebase-carrier lanes.
func TestRebaseStateControlIsNotAHistoryRewrite(t *testing.T) {
	g := New()
	for _, cmd := range []string{
		"git rebase --abort",
		"git rebase --quit",
		"git rebase --show-current-patch",
		"git rebase --show-current-patch=diff",
		"git -C /work/fak rebase --abort",
	} {
		t.Run(cmd, func(t *testing.T) {
			if law, denied := g.Classify(cmd); denied {
				t.Fatalf("Classify(%q) denied with law=%q — a rebase UNDO/read is not a shared-history rewrite", cmd, law)
			}
			if v := g.Adjudicate(context.Background(), cmdCall("Bash", "command", cmd)); v.Kind != abi.VerdictDefer {
				t.Fatalf("Adjudicate(%q) Kind=%v, want Defer", cmd, v.Kind)
			}
		})
	}
}

// Everything that ADVANCES the rewrite stays refused. `--continue` and `--skip`
// apply further commits, so they are the act the law bans; because `--abort` is
// always available as the safe exit, keeping them refused traps nothing.
func TestRebaseAdvancingFormsStayRefused(t *testing.T) {
	g := New()
	for _, cmd := range []string{
		"git rebase origin/main",
		"git rebase -i HEAD~3",
		"git rebase --interactive origin/main",
		"git rebase --autostash origin/main",
		"git rebase --continue",
		"git rebase --skip",
		"git rebase --edit-todo",
		"git rebase --onto main feature",
		// The exemption is whole-argv, so a state-control flag cannot launder a
		// real rebase onto the same line.
		"git rebase --abort origin/main",
		"git rebase --abort --continue",
		"git rebase --quit --onto main",
	} {
		t.Run(cmd, func(t *testing.T) {
			law, denied := g.Classify(cmd)
			if !denied {
				t.Fatalf("Classify(%q) deferred, want %s refusal", cmd, neverAmendSharedReason)
			}
			if !strings.Contains(law, neverAmendSharedReason) {
				t.Fatalf("Classify(%q) law=%q, want token %s", cmd, law, neverAmendSharedReason)
			}
		})
	}
}

// A refusal that leaves no exit is the failure mode this change fixes, so the
// rebase law must NAME the sanctioned exit — and name one that is actually
// allowed, not another refusal (the self-refuting-remedy loop).
func TestRebaseRefusalNamesTheAllowedExit(t *testing.T) {
	g := New()
	law, denied := g.Classify("git rebase origin/main")
	if !denied {
		t.Fatal("Classify(git rebase origin/main) deferred, want a refusal")
	}
	if !strings.Contains(law, "--abort") {
		t.Fatalf("rebase law does not name `git rebase --abort` as the exit; law=%q", law)
	}
	if _, denied := g.Classify("git rebase --abort"); denied {
		t.Fatal("the rebase law recommends `git rebase --abort` but the rung refuses it")
	}
}
