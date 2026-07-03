package gateway

import (
	"strings"
	"testing"
)

// reversibilityRefusal builds a ToolAdjudication shaped exactly as the wire
// produces one for the reversibility rung: renderVerdict maps the adjudicator's
// RequireWitness to Kind=REQUIRE_WITNESS with an empty Reason, Disposition
// ESCALATE, and the full preview envelope JSON in Detail["claim"].
func reversibilityRefusal() ToolAdjudication {
	claim := `{"class":"outward-facing","preview":"outward-facing command: git push origin main","confirm_token":"fak-0011223344556677","dry_run_hint":"try git push --dry-run first"}`
	return ToolAdjudication{
		Tool:     "PowerShell",
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "REQUIRE_WITNESS",
			By:          "monitor/reversibility",
			Disposition: "ESCALATE",
			Detail:      map[string]string{"claim": claim},
		},
	}
}

// The in-band note must carry the whole confirm recipe — preview, token, and
// dry-run hint — because a content-channel-only client (Claude Code on the
// Anthropic wire) has no other way to learn how to complete the gate's
// two-phase confirm. Before this note, the refusal rendered as the dead-end
// "PowerShell (/ESCALATE)".
func TestAdjudicationNoteCarriesReversibilityConfirmRecipe(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{reversibilityRefusal()})
	for _, want := range []string{
		"PowerShell (REQUIRE_WITNESS/ESCALATE)",
		"outward-facing command: git push origin main",
		`"_fak_confirm":"fak-0011223344556677"`,
		"try git push --dry-run first",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("adjudicationNote missing %q:\n%s", want, note)
		}
	}
	if strings.Contains(note, "(/ESCALATE)") {
		t.Fatalf("empty-reason refusal still renders the malformed (/ESCALATE) form:\n%s", note)
	}
}

// The deny-all summary path (every proposed call refused) must carry the same
// recipe — a deny-all turn is exactly where the agent is otherwise stuck.
func TestDenySummaryCarriesReversibilityConfirmRecipe(t *testing.T) {
	sum := denySummary([]ToolAdjudication{reversibilityRefusal()})
	for _, want := range []string{
		"REQUIRE_WITNESS",
		`"_fak_confirm":"fak-0011223344556677"`,
	} {
		if !strings.Contains(sum, want) {
			t.Fatalf("denySummary missing %q:\n%s", want, sum)
		}
	}
}

// A REQUIRE_WITNESS from any other gate (a plan-CFI approval, a registered
// restrictive kind) must NOT grow a confirm recipe: the recipe is only valid
// where the adjudicator's reversibility rung will actually verify the echoed
// token.
func TestReversibilityNoteScopedToReversibilityGate(t *testing.T) {
	adj := reversibilityRefusal()
	adj.Verdict.By = "monitor/plan-cfi"
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("non-reversibility REQUIRE_WITNESS grew a confirm recipe: %q", got)
	}

	adj = reversibilityRefusal()
	adj.Verdict.Detail = nil
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("missing claim payload still produced a recipe: %q", got)
	}

	adj = reversibilityRefusal()
	adj.Verdict.Detail = map[string]string{"claim": `{"class":"outward-facing"}`}
	if got := reversibilityGateNote(adj); got != "" {
		t.Fatalf("claim without a confirm token still produced a recipe: %q", got)
	}
}

// Refusals that carry a closed-vocabulary Reason keep their exact historical
// rendering — the pinned fixtures elsewhere (messages_test.go, sessionobs)
// depend on "Tool (REASON/DISPOSITION)".
func TestAdjudicationNoteKeepsReasonRenderingForPlainDenies(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "DEFAULT_DENY", Disposition: "TERMINAL"},
	}})
	if !strings.Contains(note, "Write (DEFAULT_DENY/TERMINAL)") {
		t.Fatalf("plain deny rendering changed:\n%s", note)
	}
}
