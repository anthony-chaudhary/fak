package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

// TestSpeculativeDecodingSubstrate is the unified witness verifying the multi-architecture
// speculative decoding engine across MTP heads, sidecar draft models, and prompt n-grams.
func TestSpeculativeDecodingSubstrate(t *testing.T) {
	t.Run("MultiArchitectureProposalGenerators", func(t *testing.T) {
		ctx := context.Background()
		prompt := []int{1, 2, 3, 4, 1, 2, 3}

		// 1. NGram proposal generator
		drafter := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 3}
		ngramGen := NewNGramProposalGenerator(drafter)
		if ngramGen.Name() != "ngram" {
			t.Fatalf("ngramGen.Name() = %q, want ngram", ngramGen.Name())
		}
		propNgram, err := ngramGen.Propose(ctx, prompt, 3)
		if err != nil {
			t.Fatalf("ngram propose: %v", err)
		}
		if len(propNgram.Tokens) == 0 {
			t.Fatalf("ngram proposal empty, want continuation of suffix [1, 2, 3]")
		}
		if propNgram.Tokens[0] != 4 {
			t.Fatalf("ngram proposal first token = %d, want 4", propNgram.Tokens[0])
		}

		// 2. NGram proposal generator with tree
		treeGen := NewNGramProposalGenerator(drafter).WithTree(2)
		propTree, err := treeGen.Propose(ctx, prompt, 3)
		if err != nil {
			t.Fatalf("ngram tree propose: %v", err)
		}
		if propTree.Tree == nil || len(propTree.Tree.Nodes) == 0 {
			t.Fatalf("ngram tree proposal empty, want candidate tree")
		}

		// 3. Draft model proposal generator (mock/fn)
		mockDraftFn := func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
			draft := make([]int, maxDraft)
			for i := 0; i < maxDraft; i++ {
				draft[i] = committed[len(committed)-1] + i + 1
			}
			return draft, nil
		}
		draftModelGen := NewDraftModelProposalGeneratorWithFn(mockDraftFn)
		if draftModelGen.Name() != "draft_model" {
			t.Fatalf("draftModelGen.Name() = %q, want draft_model", draftModelGen.Name())
		}
		propDraft, err := draftModelGen.Propose(ctx, prompt, 4)
		if err != nil {
			t.Fatalf("draft model propose: %v", err)
		}
		if len(propDraft.Tokens) != 4 {
			t.Fatalf("draft model proposal len = %d, want 4", len(propDraft.Tokens))
		}

		// 4. MTP proposal generator (mock/fn)
		mockMTPFn := func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
			return []int{42, 43, 44}, nil
		}
		mtpGen := NewMTPProposalGeneratorWithFn(mockMTPFn)
		if mtpGen.Name() != "mtp" {
			t.Fatalf("mtpGen.Name() = %q, want mtp", mtpGen.Name())
		}
		propMTP, err := mtpGen.Propose(ctx, prompt, 3)
		if err != nil {
			t.Fatalf("mtp propose: %v", err)
		}
		if !reflect.DeepEqual(propMTP.Tokens, []int{42, 43, 44}) {
			t.Fatalf("mtp proposal tokens = %v, want [42, 43, 44]", propMTP.Tokens)
		}

		// Verify coordinator handles all registered generators
		engine := NewSpeculativeEngine(nil, ngramGen, DefaultSpeculativeEngineConfig())
		engine.RegisterGenerator(draftModelGen)
		engine.RegisterGenerator(mtpGen)

		if engine.PrimaryGenerator().Name() != "ngram" {
			t.Fatalf("primary generator = %q, want ngram", engine.PrimaryGenerator().Name())
		}
		if err := engine.SetPrimaryGenerator("mtp"); err != nil {
			t.Fatalf("SetPrimaryGenerator(mtp): %v", err)
		}
		if engine.PrimaryGenerator().Name() != "mtp" {
			t.Fatalf("primary generator = %q, want mtp", engine.PrimaryGenerator().Name())
		}
	})

	t.Run("SinglePassParallelVerificationAndRollback", func(t *testing.T) {
		ctx := context.Background()
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()

		prompt := []int{1, 2, 3, 4}
		lastLogits := target.Prefill(prompt)

		// Compute true next token
		trueNext0 := argmaxF32(lastLogits)

		// Create proposal where token 0 matches trueNext0, token 1 is intentionally divergent
		wrongTok := (trueNext0 + 1) % cfg.VocabSize
		proposal := NewLinearProposal([]int{trueNext0, wrongTok, wrongTok}, nil)

		initLen := target.Cache.Len()
		res, err := ParallelVerifyKernel(ctx, target, prompt, proposal, lastLogits, nil, nil)
		if err != nil {
			t.Fatalf("parallel verify: %v", err)
		}

		// Token 0 should be accepted, Token 1 rejected -> 1 accepted token, 2 rolled back
		if res.NumAccepted != 1 {
			t.Fatalf("res.NumAccepted = %d, want 1", res.NumAccepted)
		}
		if len(res.AcceptedTokens) != 1 || res.AcceptedTokens[0] != trueNext0 {
			t.Fatalf("res.AcceptedTokens = %v, want [%d]", res.AcceptedTokens, trueNext0)
		}
		if res.RollbackKVCount != 2 {
			t.Fatalf("res.RollbackKVCount = %d, want 2", res.RollbackKVCount)
		}

		// Target cache len should now be initLen + 1 (only the accepted token retained)
		if target.Cache.Len() != initLen+1 {
			t.Fatalf("target.Cache.Len() = %d, want %d", target.Cache.Len(), initLen+1)
		}
	})

	t.Run("TemperatureZeroZeroOutputDivergence", func(t *testing.T) {
		ctx := context.Background()
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)

		prompt := []int{5, 12, 18, 24, 30}
		maxNew := 12

		// 1. Reference autoregressive greedy decode
		refSession := m.NewSession()
		refLogits := refSession.Prefill(prompt)
		refCommitted := append([]int(nil), prompt...)
		refGenerated := make([]int, 0, maxNew)
		curLogits := refLogits
		for len(refGenerated) < maxNew {
			nextTok := argmaxF32(curLogits)
			refGenerated = append(refGenerated, nextTok)
			refCommitted = append(refCommitted, nextTok)
			curLogits = refSession.Step(nextTok)
		}

		// 2. Speculative decoding with NGramProposalGenerator
		specSession := m.NewSession()
		drafter := NgramDrafter{Enabled: true, MinMatch: 2, MaxMatch: 4, MaxDraft: 3}
		ngramGen := NewNGramProposalGenerator(drafter)
		engine := NewSpeculativeEngine(specSession, ngramGen, SpeculativeEngineConfig{
			MaxDraft:           3,
			Temperature:        0.0,
			TripwireStrict:     true,
			SanitizeRepetition: true,
		})

		specGenerated, err := engine.Generate(ctx, prompt, maxNew)
		if err != nil {
			t.Fatalf("engine.Generate: %v", err)
		}

		// Output MUST be bit-exact identical to autoregressive baseline
		if !reflect.DeepEqual(specGenerated, refGenerated) {
			t.Fatalf("zero-divergence violation:\n want: %v\n  got: %v", refGenerated, specGenerated)
		}

		// 3. Speculative decoding with sidecar draft model
		draftModelSession := m.NewSession()
		sidecarGen := NewDraftModelProposalGenerator(draftModelSession, 3)
		specSession2 := m.NewSession()
		engine2 := NewSpeculativeEngine(specSession2, sidecarGen, SpeculativeEngineConfig{
			MaxDraft:       3,
			Temperature:    0.0,
			TripwireStrict: true,
		})

		specGenerated2, err2 := engine2.Generate(ctx, prompt, maxNew)
		if err2 != nil {
			t.Fatalf("engine2.Generate: %v", err2)
		}

		if !reflect.DeepEqual(specGenerated2, refGenerated) {
			t.Fatalf("sidecar draft zero-divergence violation:\n want: %v\n  got: %v", refGenerated, specGenerated2)
		}
	})

	t.Run("TripwireRejectsDivergenceAndNonFiniteLogits", func(t *testing.T) {
		// Test NaN logits handling
		lastLogits := []float32{1.0, float32(math.NaN()), 3.0}
		targetLogits := [][]float32{{1.0, 2.0, 3.0}}
		_, _, err := TripwireVerify([]int{0}, []int{0, 2}, lastLogits, targetLogits)
		if !errors.Is(err, ErrSpeculativeNonFiniteLogits) {
			t.Fatalf("expected ErrSpeculativeNonFiniteLogits, got %v", err)
		}

		// Test Inf logits handling
		lastLogitsInf := []float32{1.0, float32(math.Inf(1)), 3.0}
		_, _, errInf := TripwireVerify([]int{0}, []int{0, 2}, lastLogitsInf, targetLogits)
		if !errors.Is(errInf, ErrSpeculativeNonFiniteLogits) {
			t.Fatalf("expected ErrSpeculativeNonFiniteLogits for +Inf, got %v", errInf)
		}

		// Test strict argmax tripwire
		cleanLogits := []float32{1.0, 5.0, 2.0} // argmax is 1
		targetRows := [][]float32{{3.0, 1.0, 4.0}} // argmax is 2
		targetArgmax := []int{1, 2}

		// Draft proposes token 0 (not matching argmax 1) -> 0 accepted, bonus is 1
		accepted, bonus, err := TripwireVerify([]int{0}, targetArgmax, cleanLogits, targetRows)
		if err != nil {
			t.Fatalf("tripwire error: %v", err)
		}
		if len(accepted) != 0 {
			t.Fatalf("accepted tokens = %v, want []", accepted)
		}
		if bonus != 1 {
			t.Fatalf("bonus token = %d, want 1", bonus)
		}

		// Draft proposes token 1 (matches argmax 1) -> 1 accepted, bonus is 2
		accepted2, bonus2, err2 := TripwireVerify([]int{1}, targetArgmax, cleanLogits, targetRows)
		if err2 != nil {
			t.Fatalf("tripwire error: %v", err2)
		}
		if len(accepted2) != 1 || accepted2[0] != 1 {
			t.Fatalf("accepted tokens = %v, want [1]", accepted2)
		}
		if bonus2 != 2 {
			t.Fatalf("bonus token = %d, want 2", bonus2)
		}
	})

	t.Run("RepetitionPenaltyLoopSanitizer", func(t *testing.T) {
		sanitizer := NewRepetitionPenaltySanitizer(1.0, 0.5)

		// 1. Test cyclic loop detection
		historyWithLoop := []int{1, 2, 3, 10, 20, 10, 20, 10, 20}
		hasLoop, period := sanitizer.DetectDegenerateLoop(historyWithLoop, 1, 4, 3)
		if !hasLoop || period != 2 {
			t.Fatalf("DetectDegenerateLoop = (%v, %d), want (true, 2)", hasLoop, period)
		}

		// 2. Test proposal sanitization on cyclic loop
		cyclicProposal := NewLinearProposal([]int{10, 20}, nil)
		sanitized := sanitizer.SanitizeProposal(cyclicProposal, historyWithLoop)
		if len(sanitized.Tokens) != 0 {
			t.Fatalf("SanitizeProposal should have suppressed cyclic proposal tokens, got: %v", sanitized.Tokens)
		}
		if sanitized.Metadata["sanitized_degenerate_loop"] != true {
			t.Fatalf("expected metadata sanitized_degenerate_loop = true")
		}

		// 3. Test penalty application to logits
		logits := []float32{10.0, 10.0, 10.0}
		counts := []int32{0, 2, 0}
		penalized := sanitizer.ApplyPenalty(logits, counts)

		// Token 1 has count 2: penalty = 1.0*2 + 0.5 = 2.5
		if penalized[1] != 7.5 {
			t.Fatalf("penalized[1] = %v, want 7.5", penalized[1])
		}
		if penalized[0] != 10.0 || penalized[2] != 10.0 {
			t.Fatalf("penalized unrepeated tokens modified: %v", penalized)
		}

		// 4. Test counts sanitization on rollback
		countsBuf := make([]int32, 10)
		committed := []int{1, 2, 3}
		accepted := []int{4}
		sanitizer.SanitizeCounts(countsBuf, committed, accepted)

		if countsBuf[1] != 1 || countsBuf[2] != 1 || countsBuf[3] != 1 || countsBuf[4] != 1 {
			t.Fatalf("SanitizeCounts failed to populate committed and accepted: %v", countsBuf)
		}
		if countsBuf[5] != 0 {
			t.Fatalf("unaccepted token has non-zero count: %d", countsBuf[5])
		}
	})

	t.Run("CandidateTreeParallelVerification", func(t *testing.T) {
		ctx := context.Background()
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()

		prompt := []int{1, 2, 3}
		lastLogits := target.Prefill(prompt)
		rootTok := argmaxF32(lastLogits)

		// Build branch candidates
		branch1 := []int{rootTok, 10, 20}
		branch2 := []int{rootTok, 11, 21}
		tree, err := BuildCandidateTreeFromBranches([][]int{branch1, branch2})
		if err != nil {
			t.Fatalf("BuildCandidateTreeFromBranches: %v", err)
		}

		proposal, err := NewTreeProposal(tree, nil)
		if err != nil {
			t.Fatalf("NewTreeProposal: %v", err)
		}

		res, err := ParallelVerifyKernel(ctx, target, prompt, proposal, lastLogits, nil, nil)
		if err != nil {
			t.Fatalf("ParallelVerifyKernel with tree: %v", err)
		}

		// At minimum the root token must be accepted
		if res.NumAccepted < 1 {
			t.Fatalf("res.NumAccepted = %d, want at least 1 (root token)", res.NumAccepted)
		}
		if res.AcceptedTokens[0] != rootTok {
			t.Fatalf("first accepted token = %d, want rootTok %d", res.AcceptedTokens[0], rootTok)
		}
	})
}
