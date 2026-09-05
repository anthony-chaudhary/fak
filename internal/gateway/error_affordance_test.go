package gateway

import (
	"strings"
	"testing"
)

func TestErrorAffordanceKnownReasons(t *testing.T) {
	cases := map[string]string{
		"OFF_TRUNK":                "commit on main with fak commit --path <owned-path> -m <message>",
		"OUT_OF_TREE_WRITE":        "write inside the workspace or place scratch data in the OS temp directory",
		"POLICY_BLOCK":             "choose a tool and arguments admitted by the active policy",
		"OVERHEAD_BUDGET_EXCEEDED": "measure against the declared budget, then reduce the overhead or update the witnessed envelope",
		"INVALID_TOOL_ARGUMENTS":   "correct the tool arguments to match the declared schema and retry",
		"SELF_MODIFY":              "route an authorized core-lock edit through fak commit --core-lock-maintenance-witness, or send an outward message with fak slack send",
		"TRUST_VIOLATION":          "Action violates trust or Information Flow Control (IFC) boundary. Check taint source, avoid outbound egress of untrusted data, or request policy authorization.",
	}
	for reason, want := range cases {
		if got := errorAffordance(reason); got != want {
			t.Errorf("errorAffordance(%q) = %q, want %q", reason, got, want)
		}
	}
}

func TestErrorAffordanceUnknownPassesThroughVerbatim(t *testing.T) {
	unknown := "VENDOR_X: do not alter `RAW_LITERAL`\x00"
	if got := errorAffordance(unknown); got != unknown {
		t.Fatalf("unknown reason changed: got %q want %q", got, unknown)
	}
}

func TestErrorAffordanceIdempotent(t *testing.T) {
	for reason := range errorAffordances {
		once := errorAffordance(reason)
		if twice := errorAffordance(once); twice != once {
			t.Fatalf("%s is not idempotent: once=%q twice=%q", reason, once, twice)
		}
	}
}

func TestErrorAffordanceAppearsBeforeReasonInRefusalNote(t *testing.T) {
	adj := ToolAdjudication{Tool: "shell", Verdict: WireVerdict{Kind: "DENY", Reason: "OFF_TRUNK", Disposition: "TERMINAL"}}
	got := adjudicationNote([]ToolAdjudication{adj})
	action := errorAffordance("OFF_TRUNK")
	if actionIndex, reasonIndex := strings.Index(got, action), strings.Index(got, "OFF_TRUNK"); actionIndex < 0 || reasonIndex < 0 || actionIndex >= reasonIndex {
		t.Fatalf("action must lead reason: action=%d reason=%d note=%q", actionIndex, reasonIndex, got)
	}
}

// TestSelfModifyRefusalCarriesActionableAffordance fences #4988: a SELF_MODIFY
// refusal that renders no specific remedy previously fell through to the bare
// reason token, which a governed agent read as an end_turn stop. The affordance
// now supplies a sanctioned next step, so the note names a recovery command.
func TestSelfModifyRefusalCarriesActionableAffordance(t *testing.T) {
	adj := ToolAdjudication{Tool: "shell", Verdict: WireVerdict{Kind: "DENY", Reason: "SELF_MODIFY", Disposition: "TERMINAL"}}
	got := adjudicationNote([]ToolAdjudication{adj})
	action := errorAffordance("SELF_MODIFY")
	if !strings.Contains(got, action) {
		t.Fatalf("SELF_MODIFY note lacks its actionable affordance: note=%q want-substring=%q", got, action)
	}
	if !strings.Contains(got, "fak commit --core-lock-maintenance-witness") {
		t.Fatalf("SELF_MODIFY affordance must name a sanctioned recovery command: note=%q", got)
	}
}
