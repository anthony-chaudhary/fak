package adjudicator_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// The kernel->observer wiring itself is witnessed by internal/kernel's
// TestReapNotifiesPostExecutionObserverOnlyOnCommittedSuccess; this test pins
// the adjudicator half: a committed-success observation creates the read-back
// receipt.
func TestObserveResultCreatesWriteReceipt(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "zz_receipt_kernel_boundary.tmp")
	t.Cleanup(func() { _ = os.Remove(target) })

	adj := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{"write_file": true}})
	call := &abi.ToolCall{
		Tool: "write_file", TraceID: "trace-integration",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"` + filepath.ToSlash(target) + `"}`)},
	}
	adj.ObserveResult(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted})
	if op, ok := adj.AuthoredPath("trace-integration", target); !ok || op != call.SeqNo {
		t.Fatalf("post-execution receipt = %d,%v; seq=%d", op, ok, call.SeqNo)
	}
}
