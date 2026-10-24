package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// A refusal carrying an operator approval id renders the id for the human, keeps
// the runnable command out of agent-visible text, and hands the exact command to
// the out-of-band operator remedy channel.
func TestOperatorApprovalIDReachesNoteAndRemedy(t *testing.T) {
	const id = "ap-0123456789ab"
	w := renderVerdict(abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor",
		Meta: map[string]string{detailOperatorApprovalID: id}}, nil)
	if w.Detail[detailOperatorApprovalID] != id {
		t.Fatalf("wire detail = %v, want %s=%s", w.Detail, detailOperatorApprovalID, id)
	}
	a := ToolAdjudication{Tool: "Bash", Verdict: w}

	notes, _ := renderRefusalNotes(a)
	if !strings.Contains(notes, id) {
		t.Fatalf("in-band note lacks the approval id: %q", notes)
	}
	if strings.Contains(notes, "fak guard") {
		t.Fatalf("in-band note carries a runnable fak guard command: %q", notes)
	}
	if got := OperatorRemedyCommand(a); got != "fak guard allow --approve "+id {
		t.Fatalf("OperatorRemedyCommand = %q, want the approve command", got)
	}

	a.Admitted = true
	if note := operatorApprovalNote(a); note != "" {
		t.Fatalf("an admitted call must not offer an approval: %q", note)
	}
}
