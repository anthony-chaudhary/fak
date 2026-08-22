package microagent_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestRPCResultCompactsChildTranscriptsIntoBoundedEvidenceReceipts(t *testing.T) {
	root := microagent.NewContext(16384)
	verified := 0
	for _, goalClass := range []string{"research", "implementation", "operations"} {
		t.Run(goalClass, func(t *testing.T) {
			transcriptTail := "END-OF-" + strings.ToUpper(goalClass) + "-CHILD-TRANSCRIPT"
			transcript := fmt.Sprintf("%s child transcript\n%s\n%s", goalClass, strings.Repeat("private intermediate evidence ", 400), transcriptTail)
			exec := scriptExec(t, []string{"lookup"}, transcript)
			decisionJournal, path := openJournal(t)
			child, err := microagent.NewRPCSubagent(goalClass, exec, 16384, decisionJournal)
			if err != nil {
				t.Fatalf("NewRPCSubagent: %v", err)
			}
			child.WithSummarizer(func(_ string, steps []microagent.RPCStep) string {
				return string(steps[0].Stdout)
			})

			result := child.RunScript(context.Background(), []microagent.ToolAction{{Tool: "lookup"}})
			receipt, err := result.Receipt(goalClass)
			if err != nil {
				t.Fatalf("Receipt: %v", err)
			}
			if !receipt.SummaryTruncated {
				t.Fatal("oversized child transcript was not compacted")
			}
			encoded, err := receipt.Encode()
			if err != nil {
				t.Fatalf("Encode receipt: %v", err)
			}
			if len(encoded) > microagent.MaxCompletionReceiptBytes {
				t.Fatalf("receipt bytes = %d, want <= %d", len(encoded), microagent.MaxCompletionReceiptBytes)
			}
			if strings.Contains(string(encoded), transcriptTail) {
				t.Fatalf("full child transcript crossed receipt boundary: %s", encoded)
			}

			rows, err := journal.ReadRows(path)
			if err != nil {
				t.Fatalf("ReadRows: %v", err)
			}
			if checked, err := journal.Verify(path); err != nil || checked != 1 {
				t.Fatalf("Verify journal = (%d, %v), want (1, nil)", checked, err)
			}
			if len(rows) != 1 || len(receipt.Provenance) != 1 {
				t.Fatalf("journal rows=%d provenance=%v, want one stable reference", len(rows), receipt.Provenance)
			}
			if rows[0].ResultDigest == "" {
				t.Fatal("provenance row does not bind the child result bytes")
			}
			if got, want := receipt.Provenance[0].Ref, "sha256:"+rows[0].Hash; got != want {
				t.Fatalf("provenance ref = %q, want %q", got, want)
			}

			verifier := microagent.ReceiptVerifierFunc(func(_ context.Context, got microagent.CompletionReceipt) microagent.ReceiptReview {
				if len(got.Provenance) != 1 || got.Provenance[0].Ref != receipt.Provenance[0].Ref {
					t.Fatalf("verifier received unstable provenance: %+v", got.Provenance)
				}
				verified++
				return microagent.AcceptReceipt()
			})
			before := root.Len()
			if err := microagent.FoldVerifiedReceipt(context.Background(), root, receipt, verifier, nil); err != nil {
				t.Fatalf("FoldVerifiedReceipt: %v", err)
			}
			messages := root.Messages()
			if len(messages) != before+1 || messages[len(messages)-1].Content != string(encoded) {
				t.Fatalf("root did not retain only the encoded receipt: %+v", messages)
			}
		})
	}
	if verified != 3 {
		t.Fatalf("verified receipts = %d, want all three harness goal classes", verified)
	}

	t.Run("refuses missing durable provenance", func(t *testing.T) {
		exec := scriptExec(t, []string{"lookup"}, "unwitnessed child output")
		child, err := microagent.NewRPCSubagent("unwitnessed", exec, 4096, nil)
		if err != nil {
			t.Fatalf("NewRPCSubagent: %v", err)
		}
		result := child.RunScript(context.Background(), []microagent.ToolAction{{Tool: "lookup"}})
		if _, err := result.Receipt("unwitnessed"); !errors.Is(err, microagent.ErrReceiptProvenance) {
			t.Fatalf("Receipt without durable provenance = %v, want ErrReceiptProvenance", err)
		}
	})
}
