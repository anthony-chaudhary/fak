package adjudicator_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestKernelCompletionCreatesWriteReceipt(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "zz_receipt_kernel_boundary.tmp")
	t.Cleanup(func() { _ = os.Remove(target) })

	const engineID = "write-receipt-integration"
	adj := adjudicator.New(adjudicator.Policy{Allow: map[string]bool{"write_file": true}})
	k := kernel.New(engineID, kernel.WithAdjudicators([]abi.Adjudicator{adj}))
	call := &abi.ToolCall{
		Tool: "write_file", TraceID: "trace-integration",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"` + filepath.ToSlash(target) + `"}`)},
	}
	abi.RegisterEngine(engineID, receiptStructuredWriteEngine{target: target})
	h, verdict := k.Submit(context.Background(), call)
	if verdict.Kind != abi.VerdictAllow {
		t.Fatalf("submit = %v", verdict.Kind)
	}
	if _, err := k.Reap(context.Background(), h); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(target); err != nil {
		t.Fatalf("write effect absent: %v", err)
	}
	if op, ok := adj.AuthoredPath("trace-integration", target); !ok || op != call.SeqNo {
		t.Fatalf("post-execution receipt = %d,%v; seq=%d", op, ok, call.SeqNo)
	}
}

type receiptStructuredWriteEngine struct{ target string }

func (e receiptStructuredWriteEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	if err := os.WriteFile(e.target, []byte("written"), 0o600); err != nil {
		return &abi.Result{Call: c, Status: abi.StatusError, Outcome: abi.OutcomeCommitted}, nil
	}
	return &abi.Result{Call: c, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted}, nil
}
func (receiptStructuredWriteEngine) Caps() []abi.Capability { return nil }
