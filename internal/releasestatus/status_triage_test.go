package releasestatus

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TestTriageNextActionDispositions pins every gating next_action kind to its expected
// choicetriage disposition. This is the lock against silent drift: if a Signal's prose is
// edited so a knowable-engineering move starts matching an authority token (or a real release
// authority stops matching one), the pin fails loudly instead of quietly re-paging a person
// or quietly dropping a genuine human decision onto the fleet.
func TestTriageNextActionDispositions(t *testing.T) {
	cases := map[string]choicetriage.Disposition{
		// genuine PUBLISH/RELEASE/BUDGET authority -> waits on a person
		"cut_release":            choicetriage.HumanResidual,
		"cut_release_hot_tree":   choicetriage.HumanResidual,
		"promote_stable":         choicetriage.HumanResidual,
		"promote_release_branch": choicetriage.HumanResidual,
		"fix_ci_billing":         choicetriage.HumanResidual,
		// knowable engineering -> the fleet drives it in a fresh context
		"fix_ci":                 choicetriage.FreshContext,
		"confirm_ci":             choicetriage.FreshContext,
		"fix_workflow":           choicetriage.FreshContext,
		"fix_version_topology":   choicetriage.FreshContext,
		"repair_stable_evidence": choicetriage.FreshContext,
		"clean_worktree":         choicetriage.FreshContext,
		"hold":                   choicetriage.FreshContext,
		"consider_stable":        choicetriage.FreshContext,
	}
	for kind, want := range cases {
		v, ok := TriageNextAction(NextAction{Kind: kind})
		if !ok {
			t.Errorf("kind %q: expected a gating triage verdict, got none", kind)
			continue
		}
		if v.Disposition != want {
			t.Errorf("kind %q: disposition = %s, want %s (reason: %s)", kind, v.Disposition, want, v.Reason)
		}
		if got := NextActionNeedsHuman(NextAction{Kind: kind}); got != (want == choicetriage.HumanResidual) {
			t.Errorf("kind %q: NeedsHuman = %v, want %v", kind, got, want == choicetriage.HumanResidual)
		}
	}
}

// TestTriageCoversEveryGatingKind fails if actionNextActions grows a kind with no Signal — the
// fold must classify every gating action, never silently default a real ACTION to the fleet.
func TestTriageCoversEveryGatingKind(t *testing.T) {
	for kind := range actionNextActions {
		if _, ok := nextActionSignals[kind]; !ok {
			t.Errorf("gating kind %q has no triage Signal", kind)
		}
	}
	// And no Signal names a kind that is not actually gating (a dead entry that would never
	// be reached and could rot).
	for kind := range nextActionSignals {
		if !actionNextActions[kind] {
			t.Errorf("triage Signal %q is not a gating next_action kind", kind)
		}
	}
}

// TestTriageNonGating confirms the advisory kinds that already read OK surface no split.
func TestTriageNonGating(t *testing.T) {
	for _, kind := range []string{"wait", "pause_auto_release", "unknown", ""} {
		if _, ok := TriageNextAction(NextAction{Kind: kind}); ok {
			t.Errorf("non-gating kind %q must not be triaged", kind)
		}
		if line := AttentionTriageLine(Status{NextAction: NextAction{Kind: kind}}); line != "" {
			t.Errorf("non-gating kind %q must surface no triage line, got %q", kind, line)
		}
	}
}

// TestAttentionTriageLine checks the one-line readout names the side and disposition.
func TestAttentionTriageLine(t *testing.T) {
	human := AttentionTriageLine(Status{NextAction: NextAction{Kind: "cut_release"}})
	if !strings.Contains(human, "needs-you") || !strings.Contains(human, "cut_release") {
		t.Errorf("cut_release line = %q, want it to name needs-you + the kind", human)
	}
	fleet := AttentionTriageLine(Status{NextAction: NextAction{Kind: "fix_ci"}})
	if !strings.Contains(fleet, "fleet-drives") || !strings.Contains(fleet, "fix_ci") {
		t.Errorf("fix_ci line = %q, want it to name fleet-drives + the kind", fleet)
	}
}

// TestTriageSelfcheck is the deterministic proof surfaced at the CLI.
func TestTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}

// TestTriagedGatingKinds confirms the split helper partitions all gating kinds.
func TestTriagedGatingKinds(t *testing.T) {
	needHuman, fleet := TriagedGatingKinds()
	if len(needHuman)+len(fleet) != len(nextActionSignals) {
		t.Fatalf("split covers %d+%d kinds, want %d", len(needHuman), len(fleet), len(nextActionSignals))
	}
	if len(needHuman) != 5 {
		t.Errorf("expected 5 human-residual kinds, got %d: %v", len(needHuman), needHuman)
	}
	if len(fleet) != 8 {
		t.Errorf("expected 8 fleet-driven kinds, got %d: %v", len(fleet), fleet)
	}
}
