package microagent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestFoldVerifiedReceiptRoutesRPCCompletionBeforeRootFold(t *testing.T) {
	newReceipt := func(t *testing.T) microagent.CompletionReceipt {
		t.Helper()
		exec := scriptExec(t, []string{"lookup"}, "private child transcript payload")
		subagent, err := microagent.NewRPCSubagent("child-1", exec, 4096, nil)
		if err != nil {
			t.Fatalf("NewRPCSubagent: %v", err)
		}
		subagent.WithSummarizer(func(string, []microagent.RPCStep) string { return "verified result" })
		result := subagent.RunScript(context.Background(), []microagent.ToolAction{{Tool: "lookup"}})
		return result.Receipt("child-1")
	}

	t.Run("accept", func(t *testing.T) {
		root := microagent.NewContext(4096)
		verifier := microagent.ReceiptVerifierFunc(func(_ context.Context, got microagent.CompletionReceipt) microagent.ReceiptReview {
			if got.Child != "child-1" || got.Summary != "verified result" || got.Allowed != 1 {
				t.Fatalf("receipt = %+v", got)
			}
			return microagent.AcceptReceipt()
		})
		if err := microagent.FoldVerifiedReceipt(context.Background(), root, newReceipt(t), verifier, nil); err != nil {
			t.Fatalf("FoldVerifiedReceipt: %v", err)
		}
		if got := root.Messages()[0].Content; !strings.Contains(got, "verified result") || strings.Contains(got, "private child transcript payload") {
			t.Fatalf("root context crossed receipt boundary: %q", got)
		}
	})

	t.Run("reject", func(t *testing.T) {
		root := microagent.NewContext(4096)
		err := microagent.FoldVerifiedReceipt(context.Background(), root, newReceipt(t), microagent.ReceiptVerifierFunc(func(context.Context, microagent.CompletionReceipt) microagent.ReceiptReview {
			return microagent.RejectReceipt("missing witness")
		}), nil)
		if !errors.Is(err, microagent.ErrReceiptRejected) {
			t.Fatalf("error = %v, want ErrReceiptRejected", err)
		}
		if got := root.Len(); got != 0 {
			t.Fatalf("rejected receipt folded %d messages into root", got)
		}
	})

	t.Run("one bounded correction", func(t *testing.T) {
		root := microagent.NewContext(4096)
		checks := 0
		verifier := microagent.ReceiptVerifierFunc(func(_ context.Context, receipt microagent.CompletionReceipt) microagent.ReceiptReview {
			checks++
			if receipt.Summary == "corrected result" {
				return microagent.AcceptReceipt()
			}
			return microagent.CorrectReceipt("clarify outcome")
		})
		corrector := microagent.ReceiptCorrectorFunc(func(_ context.Context, receipt microagent.CompletionReceipt, request string) (microagent.CompletionReceipt, error) {
			if request != "clarify outcome" {
				t.Fatalf("correction request = %q", request)
			}
			receipt.Summary = "corrected result"
			return receipt, nil
		})
		if err := microagent.FoldVerifiedReceipt(context.Background(), root, newReceipt(t), verifier, corrector); err != nil {
			t.Fatalf("FoldVerifiedReceipt: %v", err)
		}
		if checks != 2 || !strings.Contains(root.Messages()[0].Content, "corrected result") {
			t.Fatalf("checks=%d root=%v", checks, root.Messages())
		}
		alwaysCorrect := microagent.ReceiptVerifierFunc(func(context.Context, microagent.CompletionReceipt) microagent.ReceiptReview {
			return microagent.CorrectReceipt("again")
		})
		passThrough := microagent.ReceiptCorrectorFunc(func(_ context.Context, receipt microagent.CompletionReceipt, _ string) (microagent.CompletionReceipt, error) {
			return receipt, nil
		})
		err := microagent.FoldVerifiedReceipt(context.Background(), microagent.NewContext(4096), newReceipt(t), alwaysCorrect, passThrough)
		if !errors.Is(err, microagent.ErrReceiptCorrectionLimit) {
			t.Fatalf("error = %v, want ErrReceiptCorrectionLimit", err)
		}
	})
}
