package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
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
		cleanLogits := []float32{1.0, 5.0, 2.0}    // argmax is 1
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

// speculativeParityOracle is the serial (autoregressive greedy) reference decoder.
// It returns the emitted continuation and the exact committed history the target
// session must end at, so a speculative path can be compared against it on FULLY
// COMMITTED STATE (tokens, KV length, and token lineage), not emitted tokens alone.
func speculativeParityOracle(s *Session, prompt []int, maxNew int) (emitted []int, committed []int) {
	logits := s.Prefill(prompt)
	committed = append([]int(nil), prompt...)
	emitted = make([]int, 0, maxNew)
	for len(emitted) < maxNew {
		next := argmaxF32(logits)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		emitted = append(emitted, next)
		committed = append(committed, next)
		logits = s.Step(next)
	}
	return emitted, committed
}

// assertCommittedStateParity verifies a speculative target session's COMPLETE
// committed state equals the serial oracle's: resident KV length, per-position
// token lineage, and (when provided) the boundary logits after the last commit.
// Emitted-token parity alone is deliberately NOT sufficient here (#12422): a
// speculative path can emit the right prefix while retaining a phantom KV suffix.
func assertCommittedStateParity(t *testing.T, label string, target *Session, wantCommitted []int) {
	t.Helper()
	gotLen := target.Cache.Len()
	if gotLen != len(wantCommitted) {
		t.Fatalf("%s: committed KV length = %d, want %d (serial parity)\n want committed: %v",
			label, gotLen, len(wantCommitted), wantCommitted)
	}
	report, err := target.VerifyTokenLineage(wantCommitted)
	if err != nil {
		t.Fatalf("%s: committed token lineage diverged from serial oracle: %v (want %v)",
			label, err, wantCommitted)
	}
	if report.Positions != len(wantCommitted) {
		t.Fatalf("%s: lineage positions = %d, want %d", label, report.Positions, len(wantCommitted))
	}
}

// TestSpeculativeTransitionStateParity is the #12422 standalone-engine witness:
// for empty, fully-rejected, partially-accepted, and fully-accepted proposals, the
// engine's returned transition and the target session's committed state must agree
// with a serial target-only decode of the same returned tokens. A correction token
// is a proposed NEXT token, not a committed one, so it must never advance resident
// state until the caller explicitly steps it.
func TestSpeculativeTransitionStateParity(t *testing.T) {
	ctx := context.Background()

	t.Run("FullyRejected", func(t *testing.T) {
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()
		prompt := []int{1, 2, 3, 4}
		lastLogits := target.Prefill(prompt)
		trueNext := argmaxF32(lastLogits)

		// Draft whose first token disagrees with the target argmax -> zero accepted.
		wrong := (trueNext + 1) % cfg.VocabSize
		gen := NewDraftModelProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return []int{wrong, wrong, wrong}, nil
		})
		engine := NewSpeculativeEngine(target, gen, DefaultSpeculativeEngineConfig())

		accepted, bonus, _, err := engine.Step(ctx, prompt, lastLogits, nil)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(accepted) != 0 {
			t.Fatalf("accepted = %v, want none (fully rejected)", accepted)
		}
		if bonus != trueNext {
			t.Fatalf("correction token = %d, want target argmax %d", bonus, trueNext)
		}
		// Rejected draft must leave the target exactly at the prompt boundary.
		assertCommittedStateParity(t, "fully-rejected", target, prompt)
	})

	t.Run("PartiallyAccepted", func(t *testing.T) {
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()
		prompt := []int{7, 8, 9}
		lastLogits := target.Prefill(prompt)
		first := argmaxF32(lastLogits)
		// One correct head, then a divergent tail.
		wrong := (first + 1) % cfg.VocabSize
		gen := NewDraftModelProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return []int{first, wrong, wrong}, nil
		})
		engine := NewSpeculativeEngine(target, gen, DefaultSpeculativeEngineConfig())

		accepted, _, _, err := engine.Step(ctx, prompt, lastLogits, nil)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(accepted) != 1 || accepted[0] != first {
			t.Fatalf("accepted = %v, want [%d]", accepted, first)
		}
		// Only the accepted token may remain resident; the rejected tail is rolled back.
		wantCommitted := append(append([]int(nil), prompt...), accepted...)
		assertCommittedStateParity(t, "partially-accepted", target, wantCommitted)
	})

	t.Run("EmptyProposal", func(t *testing.T) {
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()
		prompt := []int{2, 4, 6}
		lastLogits := target.Prefill(prompt)
		gen := NewDraftModelProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return nil, nil
		})
		engine := NewSpeculativeEngine(target, gen, DefaultSpeculativeEngineConfig())

		accepted, bonus, _, err := engine.Step(ctx, prompt, lastLogits, nil)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if len(accepted) != 0 {
			t.Fatalf("accepted = %v, want none (empty proposal)", accepted)
		}
		if bonus != argmaxF32(lastLogits) {
			t.Fatalf("correction = %d, want argmax %d", bonus, argmaxF32(lastLogits))
		}
		assertCommittedStateParity(t, "empty-proposal", target, prompt)
	})

	t.Run("FullyAccepted", func(t *testing.T) {
		cfg := syntheticDecodeCfg()
		m := NewSynthetic(cfg)
		target := m.NewSession()
		prompt := []int{5, 10, 15}
		lastLogits := target.Prefill(prompt)

		// A target-consistent linear draft: every token equals the target's own
		// next greedy token, so the whole run is accepted.
		draftSession := m.NewSession()
		draft, _ := speculativeParityOracle(draftSession, prompt, 3)
		if len(draft) == 0 {
			t.Fatal("oracle produced an empty draft")
		}
		gen := NewDraftModelProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
			return append([]int(nil), draft...), nil
		})
		engine := NewSpeculativeEngine(target, gen, SpeculativeEngineConfig{
			MaxDraft:       len(draft),
			Temperature:    0.0,
			TripwireStrict: true,
		})

		accepted, bonus, _, err := engine.Step(ctx, prompt, lastLogits, nil)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if !reflect.DeepEqual(accepted, draft) {
			t.Fatalf("accepted = %v, want full draft %v", accepted, draft)
		}
		_ = bonus // the correction/bonus token is a proposal for the NEXT slot, not committed
		assertCommittedStateParity(t, "fully-accepted", target,
			append(append([]int(nil), prompt...), accepted...))
	})
}

// TestSpeculativeBudgetStateParity is the #12422 output-budget witness: when an
// accepted speculative draft is longer than the remaining maxNew budget, the engine
// must commit EXACTLY the tokens it returns and roll back every verified suffix that
// the budget excluded. The target session's committed history must equal
// prompt + returned tokens, never carry a phantom suffix the caller never received.
func TestSpeculativeBudgetStateParity(t *testing.T) {
	ctx := context.Background()
	cfg := syntheticDecodeCfg()
	m := NewSynthetic(cfg)

	prompt := []int{3, 6, 9, 12, 15}
	const draftLen = 6
	maxNew := 2 // strictly smaller than the fully-accepted draft, forcing a truncating budget

	// Oracle: serial greedy continuation of the SAME length the engine may emit.
	refSession := m.NewSession()
	refEmitted, refCommitted := speculativeParityOracle(refSession, prompt, maxNew)

	// Build a target-consistent draft by serially decoding the same model for
	// draftLen tokens. Feeding the target its own continuation guarantees every
	// draft token verifies, so Step accepts the WHOLE run and retains it in KV —
	// the exact precondition for the budget-truncation defect.
	draftSession := m.NewSession()
	targetConsistentDraft, _ := speculativeParityOracle(draftSession, prompt, draftLen)
	if len(targetConsistentDraft) < draftLen {
		t.Fatalf("oracle produced only %d draft tokens, need %d for the budget witness",
			len(targetConsistentDraft), draftLen)
	}

	specSession := m.NewSession()
	gen := NewDraftModelProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) {
		return append([]int(nil), targetConsistentDraft...), nil
	})
	engine := NewSpeculativeEngine(specSession, gen, SpeculativeEngineConfig{
		MaxDraft:       draftLen,
		Temperature:    0.0,
		TripwireStrict: true,
	})

	got, err := engine.Generate(ctx, prompt, maxNew)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !reflect.DeepEqual(got, refEmitted) {
		t.Fatalf("budget-truncated output diverged from serial oracle:\n want: %v\n  got: %v", refEmitted, got)
	}

	// The committed state must match the returned tokens exactly — no phantom suffix.
	assertCommittedStateParity(t, "budget-truncated", specSession, refCommitted)

	// And the engine must refuse/agree on a second Generate call from the same
	// session boundary: serial re-decode of the committed history must reproduce it.
	if _, err := specSession.VerifyTokenLineage(refCommitted); err != nil {
		t.Fatalf("post-budget lineage re-verify: %v", err)
	}
}

// TestTripwireVerifyRejectsShortArgmaxVector is the #12422 malformed-vector
// witness: TripwireVerify must return a typed error (never panic or index out of
// range) when the supplied target argmax vector is shorter than the draft.
func TestTripwireVerifyRejectsShortArgmaxVector(t *testing.T) {
	draft := []int{1, 2, 3, 4}
	shortArgmax := []int{1, 2} // only 2 slots for a 4-token draft
	lastLogits := []float32{1.0, 2.0, 3.0}
	targetLogits := [][]float32{{1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("TripwireVerify panicked on short argmax vector: %v", r)
		}
	}()
	_, _, err := TripwireVerify(draft, shortArgmax, lastLogits, targetLogits)
	if err == nil {
		t.Fatalf("TripwireVerify accepted a short target argmax vector (%d slots for %d draft tokens); want typed error",
			len(shortArgmax), len(draft))
	}
	if !errors.Is(err, ErrSpeculativeShortArgmaxVector) {
		t.Fatalf("short argmax error = %v, want ErrSpeculativeShortArgmaxVector", err)
	}
}

// TestRollbackResidentSuffixUsesResidentAuthority is the #12422 store-discipline
// witness: the commit/rollback seam must measure and mutate the SAME resident
// store. sessionCommittedPrefixTokens is that authority (halKV when a backend
// session is resident, else Cache); RollbackResidentSuffix must leave exactly the
// requested prefix resident and never report a host-cache length that disagrees
// with the device store. This test exercises the host arm directly and pins the
// authority relationship the device arm shares through evictKV.
func TestRollbackResidentSuffixUsesResidentAuthority(t *testing.T) {
	cfg := syntheticDecodeCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	defer s.Close()

	prompt := []int{1, 2, 3, 4, 5}
	s.Prefill(prompt)

	// Extend the resident store past the prompt with real steps.
	extra := 3
	for i := 0; i < extra; i++ {
		s.Step((i + 1) % cfg.VocabSize)
	}
	full := s.speculativeResidentLen()
	if full != len(prompt)+extra {
		t.Fatalf("resident length before rollback = %d, want %d", full, len(prompt)+extra)
	}

	// Roll back the speculative suffix; the seam must read and mutate the same store.
	s.RollbackResidentSuffix(extra)
	if got := s.speculativeResidentLen(); got != len(prompt) {
		t.Fatalf("resident length after rollback = %d, want %d (authority must be consistent)", got, len(prompt))
	}
	if s.Cache.Len() != len(prompt) {
		t.Fatalf("host cache length after rollback = %d, want %d", s.Cache.Len(), len(prompt))
	}

	// Over-rollback must clamp at zero, never go negative.
	s.RollbackResidentSuffix(1000)
	if got := s.speculativeResidentLen(); got != 0 {
		t.Fatalf("resident length after over-rollback = %d, want 0", got)
	}

	// A non-positive request is a no-op.
	s.RollbackResidentSuffix(0)
	s.RollbackResidentSuffix(-5)
	if got := s.speculativeResidentLen(); got != 0 {
		t.Fatalf("non-positive rollback changed state: %d, want 0", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
// TestSpeculativePenaltyScratchReuseMatchesClone pins the per-round penalty scratch row:
// ApplyPenaltyInto is bit-identical to the ApplyPenalty clone, the chain verify accepts the
// same tokens as a clone-per-row penalized greedy reference, and the penalty costs one row
// allocation per round instead of one per draft row.
func TestSpeculativePenaltyScratchReuseMatchesClone(t *testing.T) {
	sanitizer := NewRepetitionPenaltySanitizer(1.0, 0.5)
	cfg := syntheticDecodeCfg()
	m := NewSynthetic(cfg)
	prompt := []int{1, 2, 3, 4}
	counts := make([]int32, cfg.VocabSize)
	for _, tok := range prompt {
		counts[tok]++
	}

	src := []float32{4, 3, 2, 1, 0}
	srcCounts := []int32{0, 2, 1, 0, 3}
	dst := []float32{9, 9, 9, 9, 9, 9, 9, 9}
	into := sanitizer.ApplyPenaltyInto(dst, src, srcCounts)
	clone := sanitizer.ApplyPenalty(src, srcCounts)
	if len(into) != len(clone) {
		t.Fatalf("ApplyPenaltyInto len %d, want %d", len(into), len(clone))
	}
	for i := range clone {
		if math.Float32bits(into[i]) != math.Float32bits(clone[i]) {
			t.Fatalf("ApplyPenaltyInto[%d] = %v, want %v", i, into[i], clone[i])
		}
	}
	if &into[0] != &dst[0] {
		t.Fatalf("ApplyPenaltyInto did not reuse the scratch backing array")
	}

	ref := m.NewSession()
	last := ref.Prefill(prompt)
	sim := append([]int32(nil), counts...)
	want := make([]int, 4)
	want[0] = argmaxF32(sanitizer.ApplyPenalty(last, sim))
	for i := 1; i < len(want); i++ {
		sim[want[i-1]]++
		want[i] = argmaxF32(sanitizer.ApplyPenalty(ref.Step(want[i-1]), sim))
	}
	draft := []int{want[0], want[1], want[2], (want[3] + 1) % cfg.VocabSize}

	target := m.NewSession()
	lastLogits := target.Prefill(prompt)
	res, err := ParallelVerifyKernel(context.Background(), target, prompt, NewLinearProposal(draft, nil), lastLogits, sanitizer, counts)
	if err != nil {
		t.Fatalf("ParallelVerifyKernel: %v", err)
	}
	if !reflect.DeepEqual(res.AcceptedTokens, want[:3]) || res.CorrectionToken != want[3] {
		t.Fatalf("verify = accepted %v correction %d, want %v correction %d", res.AcceptedTokens, res.CorrectionToken, want[:3], want[3])
	}

	// A first-token rejection skips verify rows entirely. Match the boundary in
	// both arms, then reject at token two so row work and rollback are identical.
	allocationRef := m.NewSession()
	allocationLogits := allocationRef.Prefill(prompt)
	boundary := argmaxF32(allocationLogits)
	allocationCounts := make([]int32, cfg.VocabSize)
	allocationCounts[(boundary+1)%cfg.VocabSize] = 1 // penalize only a nonwinner
	if argmaxF32(sanitizer.ApplyPenalty(allocationLogits, allocationCounts)) != boundary {
		t.Fatal("allocation fixture changed the boundary winner")
	}
	nextLogits := allocationRef.Step(boundary)
	nextCounts := append([]int32(nil), allocationCounts...)
	nextCounts[boundary]++
	plainNext := argmaxF32(nextLogits)
	penalizedNext := argmaxF32(sanitizer.ApplyPenalty(nextLogits, nextCounts))
	allocationRef.Close()
	rejected := 0
	for rejected == plainNext || rejected == penalizedNext {
		rejected++
	}
	const k = 6
	verifyRows := make([]int, k)
	for i := range verifyRows {
		verifyRows[i] = rejected
	}
	verifyRows[0] = boundary
	run := func(s *RepetitionPenaltySanitizer) float64 {
		return testing.AllocsPerRun(5, func() {
			sess := m.NewSession()
			ll := sess.Prefill(prompt)
			got, err := ParallelVerifyKernel(context.Background(), sess, prompt, NewLinearProposal(verifyRows, nil), ll, s, allocationCounts)
			if err != nil {
				t.Fatalf("ParallelVerifyKernel: %v", err)
			}
			if len(got.TargetLogits) != k || got.NumAccepted != 1 {
				t.Fatalf("allocation fixture verified %d rows and accepted %d tokens, want %d and 1", len(got.TargetLogits), got.NumAccepted, k)
			}
		})
	}
	plain, penalized := run(nil), run(sanitizer)
	if extra := penalized - plain; extra > 1 {
		t.Fatalf("penalty allocs/round = %v over plain, want at most 1 scratch row (k=%d)", extra, k)
	}
}

type speculativeBoundaryOwnerBackend struct {
	compute.Backend
	owner   sync.Mutex
	queries int
}

func (b *speculativeBoundaryOwnerBackend) ForwardExecutionOwner() *sync.Mutex {
	b.queries++
	return &b.owner
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=100ms
func TestParallelVerifyKnownBoundaryReject(t *testing.T) {
	ctx := context.Background()
	cfg := syntheticDecodeCfg()
	prompt := []int{3, 7, 11}
	makeBoundary := func() []float32 {
		row := make([]float32, cfg.VocabSize)
		for i := range row {
			row[i] = -10
		}
		row[7], row[8] = 9, 8
		return row
	}
	sameBits := func(a, b []float32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				return false
			}
		}
		return true
	}
	for _, tc := range []struct {
		name    string
		tie     bool
		penalty bool
		want    int
		draft   int
	}{
		{name: "raw", want: 7, draft: 9},
		{name: "lowest_index_tie", tie: true, want: 7, draft: 8},
		{name: "penalty_changes_correction", penalty: true, want: 8, draft: 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSynthetic(cfg)
			target, oracle := m.NewSession(), m.NewSession()
			t.Cleanup(target.Close)
			t.Cleanup(oracle.Close)
			target.Prefill(prompt)
			oracle.Prefill(prompt)
			boundary := makeBoundary()
			if tc.tie {
				boundary[8] = 9
			}
			counts := make([]int32, cfg.VocabSize)
			var sanitizer *RepetitionPenaltySanitizer
			if tc.penalty {
				counts[7] = 2
				sanitizer = NewRepetitionPenaltySanitizer(1, 0.5)
			}
			beforeBoundary := append([]float32(nil), boundary...)
			beforeCounts := append([]int32(nil), counts...)
			beforeKV := target.Cache.Clone()
			beforeLogits := append([]float32(nil), target.LastLogits()...)
			draft := []int{tc.draft, 9, 10}
			beforeDraft := append([]int(nil), draft...)
			res, err := ParallelVerifyKernel(ctx, target, prompt, NewLinearProposal(draft, nil), boundary, sanitizer, counts)
			if err != nil {
				t.Fatal(err)
			}
			if res.NumAccepted != 0 || len(res.AcceptedTokens) != 0 || res.CorrectionToken != tc.want ||
				res.RollbackKVCount != 0 || res.TargetLogits != nil || res.LastAcceptedLogits != nil {
				t.Fatal("known boundary rejection did not return only the independently scored correction")
			}
			if !sameBits(boundary, beforeBoundary) || !reflect.DeepEqual(counts, beforeCounts) || !reflect.DeepEqual(draft, beforeDraft) {
				t.Fatal("verification mutated caller-owned input")
			}
			if target.Cache.Len() != beforeKV.Len() || !reflect.DeepEqual(target.Cache.K, beforeKV.K) ||
				!reflect.DeepEqual(target.Cache.V, beforeKV.V) || !sameBits(target.LastLogits(), beforeLogits) {
				t.Fatal("known rejection changed target state")
			}
			if !reflect.DeepEqual(target.Step(tc.want), oracle.Step(tc.want)) {
				t.Fatal("correction continuation differs from untouched target")
			}
		})
	}
	t.Run("nonfinite_boundary", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			raw       float32
			frequency float64
			penalty   bool
		}{
			{name: "raw_nan", raw: float32(math.NaN())},
			{name: "raw_positive_infinity", raw: float32(math.Inf(1))},
			{name: "raw_negative_infinity", raw: float32(math.Inf(-1))},
			{name: "effective_nan", raw: -10, frequency: math.NaN(), penalty: true},
			{name: "effective_float32_overflow", raw: -10, frequency: math.MaxFloat64, penalty: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				target := NewSynthetic(cfg).NewSession()
				t.Cleanup(target.Close)
				target.Prefill(prompt)
				boundary := makeBoundary()
				boundary[len(boundary)-1] = tc.raw
				counts := make([]int32, cfg.VocabSize)
				var sanitizer *RepetitionPenaltySanitizer
				if tc.penalty {
					counts[len(counts)-1] = 1
					sanitizer = NewRepetitionPenaltySanitizer(tc.frequency, 0)
				}
				before := append([]float32(nil), boundary...)
				prior := target.Cache.Len()
				_, err := ParallelVerifyKernel(ctx, target, prompt, NewLinearProposal([]int{9, 10}, nil), boundary, sanitizer, counts)
				if !errors.Is(err, ErrSpeculativeNonFiniteLogits) {
					t.Fatalf("nonfinite boundary error = %v", err)
				}
				if target.Cache.Len() != prior || !sameBits(boundary, before) {
					t.Fatal("invalid boundary mutated state or input")
				}
			})
		}
	})
	t.Run("matching_first_still_verifies", func(t *testing.T) {
		m := NewSynthetic(cfg)
		target, oracle := m.NewSession(), m.NewSession()
		t.Cleanup(target.Close)
		t.Cleanup(oracle.Close)
		boundary := append([]float32(nil), target.Prefill(prompt)...)
		refBoundary := oracle.Prefill(prompt)
		first := argmaxF32(refBoundary)
		next := argmaxF32(oracle.Step(first))
		res, err := ParallelVerifyKernel(ctx, target, prompt,
			NewLinearProposal([]int{first, (next + 1) % cfg.VocabSize}, nil), boundary, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.NumAccepted != 1 || !reflect.DeepEqual(res.AcceptedTokens, []int{first}) ||
			res.CorrectionToken != next || len(res.TargetLogits) != 2 || res.RollbackKVCount != 1 {
			t.Fatal("matching first token bypassed target verification or changed partial acceptance")
		}
		assertCommittedStateParity(t, "matching-first", target, append(append([]int(nil), prompt...), first))
	})
	t.Run("malformed_input_does_not_shortcut", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			short       bool
			invalidTail bool
		}{
			{name: "wrong_boundary_width", short: true},
			{name: "invalid_later_token", invalidTail: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				target := NewSynthetic(cfg).NewSession()
				t.Cleanup(target.Close)
				target.Prefill(prompt)
				boundary, draft := makeBoundary(), []int{9, 10}
				if tc.short {
					boundary = boundary[:len(boundary)-1]
				}
				if tc.invalidTail {
					draft[1] = cfg.VocabSize
				}
				var res VerificationResult
				var err error
				var fault any
				func() {
					defer func() { fault = recover() }()
					res, err = ParallelVerifyKernel(ctx, target, prompt, NewLinearProposal(draft, nil), boundary, nil, nil)
				}()
				if fault == nil && err == nil && res.TargetLogits == nil {
					t.Fatal("malformed input received a successful no-work rejection")
				}
			})
		}
	})
	t.Run("preserves_session_admission", func(t *testing.T) {
		base, ok := compute.Lookup("cpu-ref")
		if !ok {
			t.Fatal("cpu-ref backend unavailable")
		}
		sticky := errors.New("test backend failure")
		for _, name := range []string{"closed", "sticky", "device_only", "compressed_v4"} {
			t.Run(name, func(t *testing.T) {
				var target *Session
				var want error
				probe := &speculativeBoundaryOwnerBackend{Backend: base}
				if name == "compressed_v4" {
					target = NewSynthetic(v4FlashTracerConfig([]int{0, 4})).NewSession()
					want = ErrV4FlashAttentionUnimplemented
				} else {
					var err error
					target, err = NewSynthetic(cfg).NewBackendSessionChecked(probe)
					if err != nil {
						t.Fatal(err)
					}
					if name == "device_only" {
						target.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
					} else {
						target.Close()
						if name == "sticky" {
							target.halFailure = sticky
							want = sticky
						}
					}
				}
				t.Cleanup(target.Close)
				boundary := make([]float32, target.M.Cfg.VocabSize)
				boundary[7] = 1
				beforeQueries := probe.queries
				var got error
				func() {
					defer func() {
						if r := recover(); r != nil {
							var ok bool
							got, ok = r.(error)
							if !ok {
								t.Fatalf("non-error admission panic: %T", r)
							}
						}
					}()
					_, got = ParallelVerifyKernel(ctx, target, nil, NewLinearProposal([]int{9}, nil), boundary, nil, nil)
				}()
				if name != "compressed_v4" {
					if probe.queries <= beforeQueries {
						t.Fatal("admission did not enter the backend owner domain")
					}
					if !probe.owner.TryLock() {
						probe.owner.Unlock()
						t.Fatal("admission failure leaked the backend owner")
					}
					probe.owner.Unlock()
				}
				if got == nil {
					t.Fatal("known rejection bypassed session admission")
				}
				if want != nil && !errors.Is(got, want) {
					t.Fatalf("admission error = %v, want closed sentinel", got)
				}
				if name == "device_only" {
					var typed *BackendForwardOperationError
					if !errors.As(got, &typed) || typed.Path != "device-only" || !target.BackendSessionClosed() || !target.HostFallbackObserved() {
						t.Fatal("device-only refusal lost typed fail-closed state")
					}
				}
				if (name == "closed" || name == "sticky") && !target.BackendSessionClosed() {
					t.Fatal("closed target reopened")
				}
				if target.Cache.Len() != 0 {
					t.Fatal("refused target advanced its cache")
				}
			})
		}
	})

	t.Run("success_releases_backend_owner", func(t *testing.T) {
		base, ok := compute.Lookup("cpu-ref")
		if !ok {
			t.Fatal("cpu-ref backend unavailable")
		}
		probe := &speculativeBoundaryOwnerBackend{Backend: base}
		target, err := NewSynthetic(cfg).NewBackendSessionChecked(probe)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(target.Close)
		boundary := append([]float32(nil), target.Prefill(prompt)...)
		beforeQueries := probe.queries
		wrong := (argmaxF32(boundary) + 1) % cfg.VocabSize
		res, err := ParallelVerifyKernel(ctx, target, prompt, NewLinearProposal([]int{wrong}, nil), boundary, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if probe.queries <= beforeQueries {
			t.Fatal("known rejection did not enter the backend owner domain")
		}
		if !probe.owner.TryLock() {
			probe.owner.Unlock()
			t.Fatal("known rejection leaked the backend owner")
		}
		probe.owner.Unlock()
		if res.NumAccepted != 0 || res.TargetLogits != nil || res.RollbackKVCount != 0 {
			t.Fatal("owner-scoped known rejection performed speculative work")
		}
	})
}
