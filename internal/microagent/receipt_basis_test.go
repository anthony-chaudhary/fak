package microagent_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func validTestReceipt() microagent.CompletionReceipt {
	return microagent.CompletionReceipt{
		Schema:  microagent.CompletionReceiptSchema,
		Child:   "child-worker-1",
		Summary: "child task completed successfully",
		Provenance: []microagent.EvidenceRef{
			{
				Kind: "journal-row",
				Ref:  "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			},
		},
		Allowed: 1,
		Denied:  0,
		Errored: 0,
	}
}

func TestReceiptBasisBeforeFold(t *testing.T) {
	receipt := validTestReceipt()
	acceptVerifier := microagent.ReceiptVerifierFunc(func(_ context.Context, got microagent.CompletionReceipt) microagent.ReceiptReview {
		return microagent.AcceptReceipt()
	})

	const (
		inputKey = "src/kernel/engine.go"
		digestA  = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
		digestB  = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	)

	basisA := microagent.InputBasis{
		Version: microagent.InputBasisVersion,
		Inputs: map[string]string{
			inputKey: digestA,
		},
	}

	snapshotA := map[string]string{
		inputKey:            digestA,
		"docs/reference.md": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
	}

	snapshotB := map[string]string{
		inputKey: digestB,
	}

	t.Run("receipt based on input A meets snapshot B refuses with root unchanged", func(t *testing.T) {
		root := microagent.NewContext(4096)
		root.Append("user", "existing root goal")
		beforeBytes, err := root.Encode()
		if err != nil {
			t.Fatalf("root.Encode: %v", err)
		}
		beforeLen := root.Len()
		beforeTokens := root.Tokens()

		err = microagent.FoldVerifiedReceiptWithSnapshot(
			context.Background(),
			root,
			receipt,
			basisA,
			snapshotB,
			acceptVerifier,
			nil,
		)
		if !errors.Is(err, microagent.ErrReceiptBasisMismatch) {
			t.Fatalf("err = %v, want ErrReceiptBasisMismatch", err)
		}

		afterBytes, err := root.Encode()
		if err != nil {
			t.Fatalf("root.Encode after: %v", err)
		}
		if !bytes.Equal(beforeBytes, afterBytes) {
			t.Fatalf("root bytes modified on mismatch refusal: before=%q after=%q", string(beforeBytes), string(afterBytes))
		}
		if root.Len() != beforeLen {
			t.Fatalf("root.Len() = %d, want %d", root.Len(), beforeLen)
		}
		if root.Tokens() != beforeTokens {
			t.Fatalf("root.Tokens() = %d, want %d", root.Tokens(), beforeTokens)
		}
	})

	t.Run("snapshot A matching basis passes verifier and appends to root", func(t *testing.T) {
		root := microagent.NewContext(4096)
		root.Append("user", "existing root goal")
		beforeLen := root.Len()

		err := microagent.FoldVerifiedReceiptWithSnapshot(
			context.Background(),
			root,
			receipt,
			basisA,
			snapshotA,
			acceptVerifier,
			nil,
		)
		if err != nil {
			t.Fatalf("FoldVerifiedReceiptWithSnapshot: %v", err)
		}

		if root.Len() != beforeLen+1 {
			t.Fatalf("root.Len() = %d, want %d", root.Len(), beforeLen+1)
		}
		lastMsg := root.Messages()[root.Len()-1]
		if lastMsg.Role != "tool" {
			t.Fatalf("lastMsg.Role = %q, want 'tool'", lastMsg.Role)
		}
		if !strings.Contains(lastMsg.Content, "child task completed successfully") {
			t.Fatalf("lastMsg.Content does not contain summary: %q", lastMsg.Content)
		}
	})

	t.Run("missing or empty basis refuses with root unchanged", func(t *testing.T) {
		tests := []struct {
			name  string
			basis microagent.InputBasis
		}{
			{
				name:  "zero-value basis",
				basis: microagent.InputBasis{},
			},
			{
				name: "empty inputs map",
				basis: microagent.InputBasis{
					Version: microagent.InputBasisVersion,
					Inputs:  map[string]string{},
				},
			},
			{
				name: "nil inputs map with version",
				basis: microagent.InputBasis{
					Version: microagent.InputBasisVersion,
					Inputs:  nil,
				},
			},
			{
				name: "missing version with populated inputs",
				basis: microagent.InputBasis{
					Version: "",
					Inputs: map[string]string{
						inputKey: digestA,
					},
				},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				root := microagent.NewContext(4096)
				root.Append("user", "root prompt")
				beforeBytes, err := root.Encode()
				if err != nil {
					t.Fatalf("root.Encode: %v", err)
				}
				beforeLen := root.Len()

				err = microagent.FoldVerifiedReceiptWithSnapshot(
					context.Background(),
					root,
					receipt,
					tt.basis,
					snapshotA,
					acceptVerifier,
					nil,
				)
				if !errors.Is(err, microagent.ErrMissingReceiptBasis) {
					t.Fatalf("err = %v, want ErrMissingReceiptBasis", err)
				}

				afterBytes, err := root.Encode()
				if err != nil {
					t.Fatalf("root.Encode: %v", err)
				}
				if !bytes.Equal(beforeBytes, afterBytes) {
					t.Fatalf("root bytes modified on refusal: before=%q after=%q", string(beforeBytes), string(afterBytes))
				}
				if root.Len() != beforeLen {
					t.Fatalf("root.Len() = %d, want %d", root.Len(), beforeLen)
				}
			})
		}
	})

	t.Run("missing key in snapshot refuses with root unchanged", func(t *testing.T) {
		root := microagent.NewContext(4096)
		beforeBytes, _ := root.Encode()

		err := microagent.FoldVerifiedReceiptWithSnapshot(
			context.Background(),
			root,
			receipt,
			basisA,
			map[string]string{"unrelated/path.go": digestA},
			acceptVerifier,
			nil,
		)
		if !errors.Is(err, microagent.ErrReceiptBasisMismatch) {
			t.Fatalf("err = %v, want ErrReceiptBasisMismatch", err)
		}
		afterBytes, _ := root.Encode()
		if !bytes.Equal(beforeBytes, afterBytes) {
			t.Fatalf("root modified")
		}
	})

	t.Run("receipt basis helper FoldVerifiedReceiptWithBasis", func(t *testing.T) {
		root := microagent.NewContext(4096)
		rb := receipt.WithBasis(basisA)

		// Mismatched snapshot refuses
		err := microagent.FoldVerifiedReceiptWithBasis(
			context.Background(),
			root,
			rb,
			snapshotB,
			acceptVerifier,
			nil,
		)
		if !errors.Is(err, microagent.ErrReceiptBasisMismatch) {
			t.Fatalf("err = %v, want ErrReceiptBasisMismatch", err)
		}
		if root.Len() != 0 {
			t.Fatalf("root.Len() = %d, want 0", root.Len())
		}

		// Matching snapshot appends
		err = microagent.FoldVerifiedReceiptWithBasis(
			context.Background(),
			root,
			rb,
			snapshotA,
			acceptVerifier,
			nil,
		)
		if err != nil {
			t.Fatalf("FoldVerifiedReceiptWithBasis: %v", err)
		}
		if root.Len() != 1 {
			t.Fatalf("root.Len() = %d, want 1", root.Len())
		}
	})

	t.Run("snapshot resolver interface and validate basis", func(t *testing.T) {
		snap := microagent.Snapshot(snapshotA)
		if err := snap.ValidateBasis(basisA); err != nil {
			t.Fatalf("snap.ValidateBasis: %v", err)
		}

		snapB := microagent.Snapshot(snapshotB)
		if err := snapB.ValidateBasis(basisA); !errors.Is(err, microagent.ErrReceiptBasisMismatch) {
			t.Fatalf("err = %v, want ErrReceiptBasisMismatch", err)
		}

		resolver := microagent.SnapshotResolverFunc(func(key string) (string, bool) {
			if key == inputKey {
				return digestA, true
			}
			return "", false
		})
		root := microagent.NewContext(4096)
		err := microagent.FoldVerifiedReceiptWithResolver(
			context.Background(),
			root,
			receipt,
			basisA,
			resolver,
			acceptVerifier,
			nil,
		)
		if err != nil {
			t.Fatalf("FoldVerifiedReceiptWithResolver: %v", err)
		}
		if root.Len() != 1 {
			t.Fatalf("root.Len() = %d, want 1", root.Len())
		}
	})

	t.Run("real subagent receipt fold witness", func(t *testing.T) {
		exec := scriptExec(t, []string{"lookup"}, "child worker output")
		journal, _ := openJournal(t)
		subagent, err := microagent.NewRPCSubagent("child-real", exec, 4096, journal)
		if err != nil {
			t.Fatalf("NewRPCSubagent: %v", err)
		}
		subagent.WithSummarizer(func(string, []microagent.RPCStep) string { return "real summarized result" })
		result := subagent.RunScript(context.Background(), []microagent.ToolAction{{Tool: "lookup"}})
		realReceipt, err := result.Receipt("child-real")
		if err != nil {
			t.Fatalf("Receipt: %v", err)
		}

		root := microagent.NewContext(4096)
		beforeBytes, _ := root.Encode()

		// Refuses against snapshotB
		err = microagent.FoldVerifiedReceiptWithSnapshot(
			context.Background(),
			root,
			realReceipt,
			basisA,
			snapshotB,
			acceptVerifier,
			nil,
		)
		if !errors.Is(err, microagent.ErrReceiptBasisMismatch) {
			t.Fatalf("err = %v, want ErrReceiptBasisMismatch", err)
		}
		afterBytes, _ := root.Encode()
		if !bytes.Equal(beforeBytes, afterBytes) {
			t.Fatalf("root modified")
		}

		// Accepts against snapshotA
		err = microagent.FoldVerifiedReceiptWithSnapshot(
			context.Background(),
			root,
			realReceipt,
			basisA,
			snapshotA,
			acceptVerifier,
			nil,
		)
		if err != nil {
			t.Fatalf("FoldVerifiedReceiptWithSnapshot: %v", err)
		}
		if root.Len() != 1 || !strings.Contains(root.Messages()[0].Content, "real summarized result") {
			t.Fatalf("unexpected root messages: %v", root.Messages())
		}
	})
}
