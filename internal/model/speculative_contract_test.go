package model

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// oneHotLogits creates a float32 logits row where winner has the highest logit.
func oneHotLogits(vocabSize, winner int) []float32 {
	row := make([]float32, vocabSize)
	for i := range row {
		row[i] = -10.0
	}
	if winner >= 0 && winner < vocabSize {
		row[winner] = 10.0
	}
	return row
}

// TestSpeculativeContract_DraftProposer tests DraftProposer with mock and Ngram drafters.
func TestSpeculativeContract_DraftProposer(t *testing.T) {
	ctx := context.Background()

	t.Run("MockDraftProposer", func(t *testing.T) {
		var mock DraftProposer = &MockDraftProposer{
			ProposeFunc: func(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
				return NewLinearProposal([]int{101, 102}, map[string]any{"source": "mock"}), nil
			},
		}

		proposal, err := mock.Propose(ctx, []int{1, 2, 3}, 5)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(proposal.Tokens) != 2 || proposal.Tokens[0] != 101 || proposal.Tokens[1] != 102 {
			t.Fatalf("unexpected tokens: %v", proposal.Tokens)
		}
		if proposal.Metadata["source"] != "mock" {
			t.Fatalf("unexpected metadata: %v", proposal.Metadata)
		}

		// Context cancellation
		cancelCtx, cancel := context.WithCancel(ctx)
		cancel()
		mockWithCtx := &MockDraftProposer{
			ProposeFunc: func(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
				if err := ctx.Err(); err != nil {
					return DraftProposal{}, err
				}
				return DraftProposal{}, nil
			},
		}
		if _, err := mockWithCtx.Propose(cancelCtx, []int{1}, 2); err == nil {
			t.Fatal("expected context error on canceled context")
		}
	})

	t.Run("NgramDraftProposer_Linear", func(t *testing.T) {
		drafter := NgramDrafter{
			Enabled:  true,
			MinMatch: 2,
			MaxMatch: 4,
			MaxDraft: 3,
		}
		proposer := NewNgramDraftProposer(drafter)

		// Committed history with repeating n-gram [20, 30] followed by [40, 50, 60]
		committed := []int{10, 20, 30, 40, 50, 60, 99, 20, 30}
		proposal, err := proposer.Propose(ctx, committed, 3)
		if err != nil {
			t.Fatalf("unexpected error from NgramDraftProposer: %v", err)
		}

		want := []int{40, 50, 60}
		if len(proposal.Tokens) != len(want) {
			t.Fatalf("got tokens %v, want %v", proposal.Tokens, want)
		}
		for i := range want {
			if proposal.Tokens[i] != want[i] {
				t.Fatalf("tokens[%d] = %d, want %d", i, proposal.Tokens[i], want[i])
			}
		}

		if proposal.Metadata["proposer"] != "ngram" {
			t.Errorf("expected proposer metadata 'ngram', got %v", proposal.Metadata["proposer"])
		}

		// Disabled drafter yields empty proposal without error
		disabledProposer := NewNgramDraftProposer(NgramDrafter{Enabled: false})
		emptyProp, err := disabledProposer.Propose(ctx, committed, 3)
		if err != nil {
			t.Fatalf("unexpected error from disabled drafter: %v", err)
		}
		if len(emptyProp.Tokens) != 0 {
			t.Fatalf("expected empty proposal when disabled, got %v", emptyProp.Tokens)
		}
	})

	t.Run("NgramDraftProposer_Tree", func(t *testing.T) {
		// History with multiple continuation branches for n-gram [1, 2, 3]
		committed := []int{
			1, 2, 3, 10, 11,
			99,
			1, 2, 3, 20, 21,
			98,
			1, 2, 3,
		}
		drafter := NgramDrafter{
			Enabled:  true,
			MinMatch: 3,
			MaxMatch: 3,
			MaxDraft: 2,
		}
		proposer := NewNgramDraftProposer(drafter).WithTree(2)
		proposal, err := proposer.Propose(ctx, committed, 2)
		if err != nil {
			t.Fatalf("unexpected error proposing tree: %v", err)
		}

		if proposal.Tree == nil {
			t.Fatal("expected non-nil CandidateTree in tree mode")
		}
		if len(proposal.Tree.Nodes) == 0 {
			t.Fatal("expected tree nodes to be populated")
		}
		if len(proposal.Mask) != len(proposal.Tree.Nodes) {
			t.Fatalf("mask size %d != tree size %d", len(proposal.Mask), len(proposal.Tree.Nodes))
		}
	})
}

// TestSpeculativeContract_DraftVerifier_MockSession tests DraftVerifier with mock session
// producing full acceptance with bonus correction token, partial acceptance, and edge cases.
func TestSpeculativeContract_DraftVerifier_MockSession(t *testing.T) {
	ctx := context.Background()
	vocabSize := 256

	t.Run("FullAcceptance_BonusCorrectionToken", func(t *testing.T) {
		// Target predicts token 10 at committed boundary
		boundaryLogits := oneHotLogits(vocabSize, 10)

		// Draft proposes [10, 20, 30]
		draft := []int{10, 20, 30}

		// VerifyForward returns:
		// pos 0 (after 10): predicts 20 (matches draft[1])
		// pos 1 (after 20): predicts 30 (matches draft[2])
		// pos 2 (after 30): predicts 40 (bonus token!)
		verifyLogits := [][]float32{
			oneHotLogits(vocabSize, 20),
			oneHotLogits(vocabSize, 30),
			oneHotLogits(vocabSize, 40),
		}

		evictCalled := 0
		stepCalled := 0
		verifier := NewMockSessionVerifier(
			boundaryLogits,
			func(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
				return verifyLogits
			},
			func(tok int) []float32 {
				stepCalled++
				return oneHotLogits(vocabSize, 0)
			},
			func(from, n int) int {
				evictCalled += n
				return n
			},
		)

		proposal := NewLinearProposal(draft, nil)
		res, err := verifier.Verify(ctx, []int{1, 2, 3}, proposal)
		if err != nil {
			t.Fatalf("unexpected verify error: %v", err)
		}

		if res.NumAccepted != 3 {
			t.Fatalf("expected 3 accepted tokens, got %d", res.NumAccepted)
		}
		if len(res.AcceptedTokens) != 3 || res.AcceptedTokens[0] != 10 || res.AcceptedTokens[1] != 20 || res.AcceptedTokens[2] != 30 {
			t.Fatalf("unexpected accepted tokens: %v", res.AcceptedTokens)
		}
		if res.CorrectionToken != 40 {
			t.Fatalf("expected bonus correction token 40, got %d", res.CorrectionToken)
		}
		if res.RollbackKVCount != 0 {
			t.Fatalf("expected 0 rollback count on full acceptance, got %d", res.RollbackKVCount)
		}
	})

	t.Run("PartialAcceptance_CorrectionTokenAtDivergence", func(t *testing.T) {
		// Boundary predicts token 10
		boundaryLogits := oneHotLogits(vocabSize, 10)

		// Draft proposes [10, 20, 999]
		draft := []int{10, 20, 999}

		// VerifyForward returns:
		// pos 0: predicts 20 (matches draft[1])
		// pos 1: predicts 35 (diverges with draft[2]=999!)
		// pos 2: predicts 50
		verifyLogits := [][]float32{
			oneHotLogits(vocabSize, 20),
			oneHotLogits(vocabSize, 35),
			oneHotLogits(vocabSize, 50),
		}

		evictedCount := 0
		verifier := NewMockSessionVerifier(
			boundaryLogits,
			func(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
				return verifyLogits
			},
			nil,
			func(from, n int) int {
				evictedCount += n
				return n
			},
		)

		proposal := NewLinearProposal(draft, nil)
		res, err := verifier.Verify(ctx, []int{1, 2, 3}, proposal)
		if err != nil {
			t.Fatalf("unexpected verify error: %v", err)
		}

		if res.NumAccepted != 2 {
			t.Fatalf("expected 2 accepted tokens, got %d", res.NumAccepted)
		}
		if len(res.AcceptedTokens) != 2 || res.AcceptedTokens[0] != 10 || res.AcceptedTokens[1] != 20 {
			t.Fatalf("unexpected accepted tokens: %v", res.AcceptedTokens)
		}
		if res.CorrectionToken != 35 {
			t.Fatalf("expected correction token 35 at divergence, got %d", res.CorrectionToken)
		}
		if res.RollbackKVCount != 1 {
			t.Fatalf("expected 1 rollback count, got %d", res.RollbackKVCount)
		}
		if evictedCount != 1 {
			t.Fatalf("expected 1 evicted position, got %d", evictedCount)
		}
	})

	t.Run("ZeroAcceptance_ImmediateCorrection", func(t *testing.T) {
		boundaryLogits := oneHotLogits(vocabSize, 10)
		draft := []int{888, 999}

		verifyLogits := [][]float32{
			oneHotLogits(vocabSize, 50),
			oneHotLogits(vocabSize, 60),
		}

		evictedCount := 0
		verifier := NewMockSessionVerifier(
			boundaryLogits,
			func(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
				return verifyLogits
			},
			nil,
			func(from, n int) int {
				evictedCount += n
				return n
			},
		)

		proposal := NewLinearProposal(draft, nil)
		res, err := verifier.Verify(ctx, []int{1, 2, 3}, proposal)
		if err != nil {
			t.Fatalf("unexpected verify error: %v", err)
		}

		if res.NumAccepted != 0 {
			t.Fatalf("expected 0 accepted tokens, got %d", res.NumAccepted)
		}
		if len(res.AcceptedTokens) != 0 {
			t.Fatalf("expected nil or empty accepted tokens, got %v", res.AcceptedTokens)
		}
		if res.CorrectionToken != 10 {
			t.Fatalf("expected correction token 10 from boundary, got %d", res.CorrectionToken)
		}
		if res.RollbackKVCount != 2 {
			t.Fatalf("expected 2 rollback count, got %d", res.RollbackKVCount)
		}
		if evictedCount != 2 {
			t.Fatalf("expected 2 evicted positions, got %d", evictedCount)
		}
	})

	t.Run("EmptyProposal_Bypass", func(t *testing.T) {
		boundaryLogits := oneHotLogits(vocabSize, 77)
		verifier := NewMockSessionVerifier(boundaryLogits, nil, nil, nil)

		res, err := verifier.Verify(ctx, []int{1, 2, 3}, DraftProposal{})
		if err != nil {
			t.Fatalf("unexpected verify error on empty proposal: %v", err)
		}

		if res.NumAccepted != 0 || res.CorrectionToken != 77 || res.RollbackKVCount != 0 {
			t.Fatalf("unexpected result on empty proposal: %+v", res)
		}
	})
}

// TestSpeculativeContract_TreeProposalAndCausalMask tests CandidateTree construction,
// causal mask derivation, cycle detection, and tree verification.
func TestSpeculativeContract_TreeProposalAndCausalMask(t *testing.T) {
	ctx := context.Background()
	vocabSize := 256

	t.Run("TreeConstructionAndCausalMask", func(t *testing.T) {
		// Branches:
		// Branch 0: [10, 20, 30]
		// Branch 1: [10, 20, 31]
		// Branch 2: [10, 25]
		branches := [][]int{
			{10, 20, 30},
			{10, 20, 31},
			{10, 25},
		}

		tree, err := BuildCandidateTreeFromBranches(branches)
		if err != nil {
			t.Fatalf("unexpected error building tree: %v", err)
		}

		// Expected 5 nodes:
		// Node 0: Token 10, Parent -1, Depth 0
		// Node 1: Token 20, Parent 0, Depth 1
		// Node 2: Token 30, Parent 1, Depth 2
		// Node 3: Token 31, Parent 1, Depth 2
		// Node 4: Token 25, Parent 0, Depth 1
		if len(tree.Nodes) != 5 {
			t.Fatalf("expected 5 nodes, got %d", len(tree.Nodes))
		}

		if tree.Nodes[0].Token != 10 || tree.Nodes[0].Parent != -1 || tree.Nodes[0].Depth != 0 {
			t.Errorf("node 0 mismatch: %+v", tree.Nodes[0])
		}
		if tree.Nodes[1].Token != 20 || tree.Nodes[1].Parent != 0 || tree.Nodes[1].Depth != 1 {
			t.Errorf("node 1 mismatch: %+v", tree.Nodes[1])
		}
		if tree.Nodes[2].Token != 30 || tree.Nodes[2].Parent != 1 || tree.Nodes[2].Depth != 2 {
			t.Errorf("node 2 mismatch: %+v", tree.Nodes[2])
		}
		if tree.Nodes[3].Token != 31 || tree.Nodes[3].Parent != 1 || tree.Nodes[3].Depth != 2 {
			t.Errorf("node 3 mismatch: %+v", tree.Nodes[3])
		}
		if tree.Nodes[4].Token != 25 || tree.Nodes[4].Parent != 0 || tree.Nodes[4].Depth != 1 {
			t.Errorf("node 4 mismatch: %+v", tree.Nodes[4])
		}

		// Derive and verify causal attention mask
		mask, err := tree.DeriveMask()
		if err != nil {
			t.Fatalf("unexpected error deriving mask: %v", err)
		}

		// Node 0 (root): attends only to 0
		if !mask[0][0] || mask[0][1] || mask[0][2] || mask[0][3] || mask[0][4] {
			t.Errorf("mask[0] violation: %v", mask[0])
		}
		// Node 1: attends to 0, 1
		if !mask[1][0] || !mask[1][1] || mask[1][2] || mask[1][3] || mask[1][4] {
			t.Errorf("mask[1] violation: %v", mask[1])
		}
		// Node 2 (token 30): attends to 0, 1, 2; NOT 3 or 4
		if !mask[2][0] || !mask[2][1] || !mask[2][2] || mask[2][3] || mask[2][4] {
			t.Errorf("mask[2] violation: %v", mask[2])
		}
		// Node 3 (token 31): attends to 0, 1, 3; NOT 2 or 4
		if !mask[3][0] || !mask[3][1] || mask[3][2] || !mask[3][3] || mask[3][4] {
			t.Errorf("mask[3] violation: %v", mask[3])
		}
		// Node 4 (token 25): attends to 0, 4; NOT 1, 2, or 3
		if !mask[4][0] || mask[4][1] || mask[4][2] || mask[4][3] || !mask[4][4] {
			t.Errorf("mask[4] violation: %v", mask[4])
		}

		proposal, err := NewTreeProposal(tree, map[string]any{"engine": "tree-test"})
		if err != nil {
			t.Fatalf("unexpected error creating tree proposal: %v", err)
		}
		if len(proposal.Tokens) != 5 {
			t.Fatalf("expected 5 proposal tokens, got %d", len(proposal.Tokens))
		}
	})

	t.Run("CycleAndOutOfBoundsDetection", func(t *testing.T) {
		// Cycle: Node 0 -> Node 1 -> Node 0
		cyclicTree := &CandidateTree{
			Nodes: []CandidateNode{
				{Token: 1, Parent: 1, Depth: 0},
				{Token: 2, Parent: 0, Depth: 1},
			},
		}
		if _, err := cyclicTree.DeriveMask(); err == nil {
			t.Fatal("expected error on cyclic candidate tree")
		}

		// Out of bounds parent
		badTree := &CandidateTree{
			Nodes: []CandidateNode{
				{Token: 1, Parent: 99, Depth: 0},
			},
		}
		if _, err := badTree.DeriveMask(); err == nil {
			t.Fatal("expected error on out-of-bounds parent")
		}
	})

	t.Run("TreeVerification_BranchAcceptanceWithBonusToken", func(t *testing.T) {
		// Tree with 2 branches:
		// Root: Node 0 (Token 10)
		// Child A: Node 1 (Token 20, child of 0) -> Child A1: Node 2 (Token 30, child of 1)
		// Child B: Node 3 (Token 21, child of 0)
		tree := &CandidateTree{
			Nodes: []CandidateNode{
				{Token: 10, Parent: -1, Children: []int{1, 3}, Depth: 0},
				{Token: 20, Parent: 0, Children: []int{2}, Depth: 1},
				{Token: 30, Parent: 1, Children: nil, Depth: 2},
				{Token: 21, Parent: 0, Children: nil, Depth: 1},
			},
		}

		// Target boundary predicts 10 (matches root)
		boundaryLogits := oneHotLogits(vocabSize, 10)

		// Target predictions per tree node:
		// Node 0 (10): predicts 20 (selects child A=Node 1!)
		// Node 1 (20): predicts 30 (selects child A1=Node 2!)
		// Node 2 (30): predicts 42 (bonus token!)
		// Node 3 (21): predicts 99
		treeLogits := [][]float32{
			oneHotLogits(vocabSize, 20),
			oneHotLogits(vocabSize, 30),
			oneHotLogits(vocabSize, 42),
			oneHotLogits(vocabSize, 99),
		}

		stepTokens := make([]int, 0)
		evictCalled := 0
		verifier := NewMockSessionVerifier(
			boundaryLogits,
			func(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
				return treeLogits
			},
			func(tok int) []float32 {
				stepTokens = append(stepTokens, tok)
				return oneHotLogits(vocabSize, 0)
			},
			func(from, n int) int {
				evictCalled += n
				return n
			},
		)

		proposal, err := NewTreeProposal(tree, nil)
		if err != nil {
			t.Fatalf("failed to create tree proposal: %v", err)
		}

		res, err := verifier.Verify(ctx, []int{1, 2, 3}, proposal)
		if err != nil {
			t.Fatalf("unexpected tree verify error: %v", err)
		}

		// Accepted path should be Node 0 -> Node 1 -> Node 2: tokens [10, 20, 30]
		if res.NumAccepted != 3 {
			t.Fatalf("expected 3 accepted tokens, got %d", res.NumAccepted)
		}
		wantTokens := []int{10, 20, 30}
		for i, tok := range wantTokens {
			if res.AcceptedTokens[i] != tok {
				t.Fatalf("accepted[%d] = %d, want %d", i, res.AcceptedTokens[i], tok)
			}
		}

		if res.CorrectionToken != 42 {
			t.Fatalf("expected bonus correction token 42, got %d", res.CorrectionToken)
		}

		// 4 speculative tree nodes verified, 3 accepted -> 1 rejected
		if res.RollbackKVCount != 1 {
			t.Fatalf("expected RollbackKVCount=1, got %d", res.RollbackKVCount)
		}

		// Cache replayed the 3 accepted tokens sequentially
		if len(stepTokens) != 3 || stepTokens[0] != 10 || stepTokens[1] != 20 || stepTokens[2] != 30 {
			t.Fatalf("expected replayed step tokens [10, 20, 30], got %v", stepTokens)
		}
	})
}

// TestSpeculativeContract_ProposerIsolation verifies that neither proposer touches
// KV cache or attention internals directly.
func TestSpeculativeContract_ProposerIsolation(t *testing.T) {
	ctx := context.Background()

	// 1. Inspect structural isolation via reflection:
	// Verify that DraftProposer implementations and types have no fields pointing to
	// KVCache, Session, Model, Activations, or attention buffer types.
	forbiddenSubstrings := []string{
		"KVCache",
		"kvcache",
		"Session",
		"session",
		"Model",
		"Activations",
		"attention",
		"attn",
	}

	inspectType := func(val any, typeName string) {
		typ := reflect.TypeOf(val)
		if typ.Kind() == reflect.Ptr {
			typ = typ.Elem()
		}
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			fieldTypeStr := field.Type.String()
			fieldNameStr := field.Name
			for _, forbidden := range forbiddenSubstrings {
				if strings.Contains(fieldTypeStr, forbidden) || strings.Contains(fieldNameStr, forbidden) {
					t.Errorf("Type %s field %s of type %s violates KV/attention buffer isolation contract (matched %q)",
						typeName, fieldNameStr, fieldTypeStr, forbidden)
				}
			}
		}
	}

	inspectType(&NgramDraftProposer{}, "NgramDraftProposer")
	inspectType(&MockDraftProposer{}, "MockDraftProposer")
	inspectType(&DraftProposal{}, "DraftProposal")
	inspectType(&CandidateTree{}, "CandidateTree")
	inspectType(&CandidateNode{}, "CandidateNode")

	// 2. Behavioral isolation:
	// Calling Propose receives only (ctx, committed, maxDraft) and produces a DraftProposal.
	// It operates purely on token IDs with no external side effects on KV memory.
	committed := []int{5, 6, 7, 8, 5, 6, 7}
	proposer := NewNgramDraftProposer(NgramDrafter{
		Enabled:  true,
		MinMatch: 2,
		MaxMatch: 4,
		MaxDraft: 2,
	})

	// Mutating the input slice after proposal generation does not mutate the proposal
	proposal, err := proposer.Propose(ctx, committed, 2)
	if err != nil {
		t.Fatalf("propose error: %v", err)
	}

	origToken := proposal.Tokens[0]
	committed[4] = 9999
	if proposal.Tokens[0] != origToken {
		t.Fatal("proposal tokens mutated when committed slice changed: memory isolation violated")
	}
}

// TestSpeculativeContract_SyntheticSessionIntegration runs SessionDraftVerifier with
// a live synthetic Session to verify end-to-end full and partial acceptance against
// deterministic serial Step execution.
func TestSpeculativeContract_SyntheticSessionIntegration(t *testing.T) {
	ctx := context.Background()

	// Small synthetic PreNorm model
	cfg := cfgV(48, 2, 4, 2, 16, 96)
	m := NewSynthetic(cfg)
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// 1. Establish ground truth serial steps
	ref := m.NewSession()
	tl := ref.Prefill(prompt)
	tok0 := argmaxF32(tl)
	l0 := ref.Step(tok0)
	tok1 := argmaxF32(l0)
	l1 := ref.Step(tok1)
	tok2 := argmaxF32(l1)

	t.Logf("Ground truth serial continuation: [%d, %d] -> bonus %d", tok0, tok1, tok2)

	// 2. Full acceptance via SessionDraftVerifier
	testSession := m.NewSession()
	verifier := NewSessionDraftVerifier(testSession)

	proposal := NewLinearProposal([]int{tok0, tok1}, map[string]any{"mode": "full-test"})
	res, err := verifier.Verify(ctx, prompt, proposal)
	if err != nil {
		t.Fatalf("verify error: %v", err)
	}

	if res.NumAccepted != 2 {
		t.Fatalf("expected 2 accepted tokens, got %d", res.NumAccepted)
	}
	if res.AcceptedTokens[0] != tok0 || res.AcceptedTokens[1] != tok1 {
		t.Fatalf("expected accepted tokens [%d, %d], got %v", tok0, tok1, res.AcceptedTokens)
	}
	if res.CorrectionToken != tok2 {
		t.Fatalf("expected bonus correction token %d, got %d", tok2, res.CorrectionToken)
	}
	if res.RollbackKVCount != 0 {
		t.Fatalf("expected 0 rollback count on full acceptance, got %d", res.RollbackKVCount)
	}

	// 3. Partial acceptance via SessionDraftVerifier
	testSession2 := m.NewSession()
	verifier2 := NewSessionDraftVerifier(testSession2)

	// Propose [tok0, badToken]
	badToken := (tok1 + 17) % cfg.VocabSize
	if badToken == tok1 {
		badToken = (tok1 + 31) % cfg.VocabSize
	}

	proposal2 := NewLinearProposal([]int{tok0, badToken}, map[string]any{"mode": "partial-test"})
	res2, err := verifier2.Verify(ctx, prompt, proposal2)
	if err != nil {
		t.Fatalf("verify error: %v", err)
	}

	if res2.NumAccepted != 1 {
		t.Fatalf("expected 1 accepted token, got %d", res2.NumAccepted)
	}
	if res2.AcceptedTokens[0] != tok0 {
		t.Fatalf("expected accepted token %d, got %v", tok0, res2.AcceptedTokens)
	}
	if res2.CorrectionToken != tok1 {
		t.Fatalf("expected correction token %d at divergence, got %d", tok1, res2.CorrectionToken)
	}
	if res2.RollbackKVCount != 1 {
		t.Fatalf("expected 1 rollback count on partial acceptance, got %d", res2.RollbackKVCount)
	}
}
