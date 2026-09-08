package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestQwen35MTPDepthNDraftSequentialFeedbackAndCacheCoherence(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{2, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)

	type call struct {
		pos      int
		prior    []float32
		feedback []float32
	}
	var calls []call
	realStep := d.step
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		feedback, logits, err := realStep(f, pos, prior, embedding)
		calls = append(calls, call{
			pos:      pos,
			prior:    append([]float32(nil), prior...),
			feedback: append([]float32(nil), feedback...),
		})
		return feedback, logits, err
	}

	first := d.Propose(prompt)
	if err := d.Err(); err != nil {
		t.Fatalf("first depth-N proposal: %v", err)
	}
	if len(first) != 4 {
		t.Fatalf("first draft = %v, want depth 4", first)
	}
	if got := d.forward.draft.Cache.Len(); got != len(prompt)+3 {
		t.Fatalf("speculative MTP cache length = %d, want committed %d + speculative 3", got, len(prompt))
	}
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("drafting mutated target cache length to %d, want %d", target.Cache.Len(), len(prompt))
	}
	if d.pending == nil {
		t.Fatal("successful multi-step proposal retained no rollback checkpoint")
	}
	if len(calls) != len(prompt)+3 {
		t.Fatalf("forward calls = %d, want %d committed catch-up + speculative steps", len(calls), len(prompt)+3)
	}
	for i, call := range calls {
		if call.pos != i {
			t.Fatalf("call %d position = %d, want %d", i, call.pos, i)
		}
	}
	if got := calls[0].prior; !reflect.DeepEqual(got, make([]float32, m.Cfg.HiddenSize)) {
		t.Fatalf("first committed prior hidden = %v, want zero boundary", got)
	}
	targetHidden0, err := target.TargetHiddenAt(0)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "shifted committed target hidden", targetHidden0, calls[1].prior)
	targetHidden1, err := target.TargetHiddenAt(1)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "first speculative target seed", targetHidden1, calls[2].prior)
	assertFloat32BitsEqual(t, "draft feedback step 2", calls[2].feedback, calls[3].prior)
	assertFloat32BitsEqual(t, "draft feedback step 3", calls[3].feedback, calls[4].prior)

	target.Step(first[0])
	extended := append(append([]int(nil), prompt...), first[0])
	beforeSecond := len(calls)
	second := d.Propose(extended)
	if err := d.Err(); err != nil {
		t.Fatalf("second depth-N proposal: %v", err)
	}
	if len(second) != 4 {
		t.Fatalf("second draft = %v, want depth 4", second)
	}
	secondCalls := calls[beforeSecond:]
	if len(secondCalls) != 4 {
		t.Fatalf("second proposal calls = %d, want one committed catch-up + three speculative", len(secondCalls))
	}
	if secondCalls[0].pos != len(prompt) {
		t.Fatalf("committed rebase position = %d, want %d", secondCalls[0].pos, len(prompt))
	}
	assertFloat32BitsEqual(t, "committed rebase uses target hidden", targetHidden1, secondCalls[0].prior)
	if got := d.forward.draft.Cache.Len(); got != len(extended)+3 {
		t.Fatalf("second speculative MTP cache length = %d, want %d", got, len(extended)+3)
	}
	if !reflect.DeepEqual(d.processed, extended) {
		t.Fatalf("processed committed tokens = %v, want %v", d.processed, extended)
	}
}

func TestQwen35MTPDepthNPartialFailureRollsBackDraftCache(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)

	injected := errors.New("injected depth-N draft failure")
	realStep := d.step
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		feedback, logits, err := realStep(f, pos, prior, embedding)
		if err == nil && pos == len(prompt)+1 {
			return nil, nil, injected
		}
		return feedback, logits, err
	}

	if got := d.Propose(prompt); got != nil {
		t.Fatalf("proposal after partial failure = %v, want nil", got)
	}
	if !errors.Is(d.Err(), injected) {
		t.Fatalf("runtime error = %v, want injected failure", d.Err())
	}
	if d.pending != nil {
		t.Fatal("failed proposal retained an active rollback checkpoint")
	}
	if got := d.forward.draft.Cache.Len(); got != len(prompt) {
		t.Fatalf("draft cache after failure = %d, want committed boundary %d", got, len(prompt))
	}
	if got := d.forward.lastPos; got != len(prompt)-1 {
		t.Fatalf("draft last position after failure = %d, want %d", got, len(prompt)-1)
	}
	if !reflect.DeepEqual(d.processed, prompt) {
		t.Fatalf("processed tokens after failure = %v, want %v", d.processed, prompt)
	}
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("draft failure mutated target cache length to %d", target.Cache.Len())
	}
}

func TestSpecDecodeGreedyQwen35MTPDepthAdmission(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	n := 12

	for depth := 1; depth <= Qwen35MTPMaxDraftDepth; depth++ {
		t.Run(fmt.Sprintf("depth-%d", depth), func(t *testing.T) {
			want := m.NewSession()
			want.captureTargetHidden = true
			t.Cleanup(want.Close)
			wantOutput := want.Generate(prompt, n)

			target := m.NewSession()
			t.Cleanup(target.Close)
			run, err := SpecDecodeGreedyQwen35MTPDepthN(target, prompt, n, depth)
			if err != nil {
				t.Fatalf("native depth-%d MTP decode: %v", depth, err)
			}
			if !reflect.DeepEqual(run.Output, wantOutput) {
				t.Fatalf("depth-%d output = %v, want target greedy %v", depth, run.Output, wantOutput)
			}
			if run.DraftedTokens == 0 || run.Rounds == 0 {
				t.Fatalf("depth-%d run did no drafting: %+v", depth, run)
			}
			if run.AcceptedDrafts+run.EvictKV != run.DraftedTokens {
				t.Fatalf("depth-%d accounting accepted %d + rejected %d != drafted %d", depth, run.AcceptedDrafts, run.EvictKV, run.DraftedTokens)
			}
			assertQwen35MTPTargetStateEqual(t, target, want)
		})
	}

	for _, tc := range []struct {
		name   string
		depth  int
		mutate func(*Session)
		reason string
	}{
		{name: "zero", depth: 0},
		{name: "above bound", depth: Qwen35MTPMaxDraftDepth + 1, reason: "exceeds the witnessed native bound"},
		{name: "quant format", depth: 2, mutate: func(s *Session) { s.Quant = true }, reason: "#9985"},
		{name: "malformed shape", depth: 2, mutate: func(s *Session) {
			meta := s.M.manifest["mtp.fc.weight"]
			meta.Shape = []int{1, 1}
			s.M.manifest["mtp.fc.weight"] = meta
		}, reason: "weight shape"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := qwen35MTPEnabledSyntheticModel(t)
			target := model.NewSession()
			t.Cleanup(target.Close)
			if tc.mutate != nil {
				tc.mutate(target)
			}
			_, err := SpecDecodeGreedyQwen35MTPDepthN(target, prompt, 2, tc.depth)
			if tc.depth == 0 {
				if !errors.Is(err, ErrQwen35MTPInvalidDraftLength) {
					t.Fatalf("zero-depth error = %v, want ErrQwen35MTPInvalidDraftLength", err)
				}
			} else {
				unsupported := assertQwen35MTPUnsupported(t, err)
				if tc.reason != "" && !strings.Contains(unsupported.Reason, tc.reason) {
					t.Fatalf("unsupported reason = %q, want substring %q", unsupported.Reason, tc.reason)
				}
			}
			if target.Cache.Len() != 0 {
				t.Fatalf("typed refusal mutated target cache length to %d", target.Cache.Len())
			}
		})
	}
}

func TestSpecDecodeGreedyQwen35MTPDepthNDeterministicAcceptance(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3
	wantSession := m.NewSession()
	wantSession.captureTargetHidden = true
	t.Cleanup(wantSession.Close)
	wantOutput := wantSession.Generate(prompt, depth+1)
	history := append(append([]int(nil), prompt...), wantOutput...)

	builder := func(target *Session, depth int) (*Qwen35MTPDraftSession, error) {
		d, err := NewQwen35MTPDraftSession(target, depth)
		if err != nil {
			return nil, err
		}
		realStep := d.step
		d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
			feedback, logits, err := realStep(f, pos, prior, embedding)
			if err != nil || pos+1 >= len(history) {
				return feedback, logits, err
			}
			for i := range logits {
				logits[i] = 0
			}
			logits[history[pos+1]] = 1
			return feedback, logits, nil
		}
		return d, nil
	}

	target := m.NewSession()
	t.Cleanup(target.Close)
	run, err := specDecodeGreedyQwen35MTPDepthN(target, prompt, depth+1, depth, builder)
	if err != nil {
		t.Fatalf("full-acceptance native depth-N decode: %v", err)
	}
	if !reflect.DeepEqual(run.Output, wantOutput) {
		t.Fatalf("full-acceptance output = %v, want %v", run.Output, wantOutput)
	}
	if run.Rounds != 1 || run.DraftedTokens != depth || run.AcceptedDrafts != depth || run.EvictKV != 0 {
		t.Fatalf("full-acceptance accounting = %+v, want one round and %d/%d/0 drafted/accepted/rejected", run, depth, depth)
	}
	assertQwen35MTPTargetStateEqual(t, target, wantSession)

	partialBuilder := func(target *Session, depth int) (*Qwen35MTPDraftSession, error) {
		d, err := builder(target, depth)
		if err != nil {
			return nil, err
		}
		perfectStep := d.step
		d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
			feedback, logits, err := perfectStep(f, pos, prior, embedding)
			if err == nil && pos == len(prompt) {
				for i := range logits {
					logits[i] = 0
				}
				logits[(history[pos+1]+1)%len(logits)] = 1
			}
			return feedback, logits, err
		}
		return d, nil
	}

	partialWant := m.NewSession()
	partialWant.captureTargetHidden = true
	t.Cleanup(partialWant.Close)
	partialOutput := partialWant.Generate(prompt, 2)
	partialTarget := m.NewSession()
	t.Cleanup(partialTarget.Close)
	partial, err := specDecodeGreedyQwen35MTPDepthN(partialTarget, prompt, 2, depth, partialBuilder)
	if err != nil {
		t.Fatalf("partial-acceptance native depth-N decode: %v", err)
	}
	if !reflect.DeepEqual(partial.Output, partialOutput) {
		t.Fatalf("partial-acceptance output = %v, want %v", partial.Output, partialOutput)
	}
	if partial.Rounds != 1 || partial.DraftedTokens != depth || partial.AcceptedDrafts != 1 || partial.EvictKV != depth-1 {
		t.Fatalf("partial-acceptance accounting = %+v, want one accepted and %d rejected", partial, depth-1)
	}
	assertQwen35MTPTargetStateEqual(t, partialTarget, partialWant)
}

func TestDraftVocabTruncation_CorrectTokenIDs(t *testing.T) {
	// 1. Verify DraftVocabFilter standalone structure and indexing.
	subset := []int{5, 12, 100, 2048, 40000, 240000}
	filter := NewDraftVocabFilter(subset)
	if filter == nil {
		t.Fatal("NewDraftVocabFilter returned nil for valid subset")
	}
	if filter.Len() != len(subset) {
		t.Fatalf("filter.Len() = %d, want %d", filter.Len(), len(subset))
	}
	for i, id := range subset {
		if got := filter.RemapIndex(i); got != id {
			t.Fatalf("RemapIndex(%d) = %d, want %d", i, got, id)
		}
		if !filter.Contains(id) {
			t.Fatalf("filter.Contains(%d) = false, want true", id)
		}
		idx, ok := filter.Index(id)
		if !ok || idx != i {
			t.Fatalf("filter.Index(%d) = (%d, %v), want (%d, true)", id, idx, ok, i)
		}
	}
	if filter.Contains(999) {
		t.Fatal("filter.Contains(999) = true for absent token")
	}
	if got := filter.RemapIndex(-1); got != -1 {
		t.Fatalf("RemapIndex(-1) = %d, want -1", got)
	}
	if got := filter.RemapIndex(len(subset)); got != -1 {
		t.Fatalf("RemapIndex(out_of_bounds) = %d, want -1", got)
	}

	// 2. Verify argmax within subset remapping to full vocabulary token ID.
	subsetLogits := []float32{-10.0, 5.2, 0.1, 8.4, 2.0, -1.0}
	// Max logit is at index 3 (value 8.4), corresponding to token 2048.
	wantTokenID := 2048
	gotTokenID := filter.Argmax(subsetLogits)
	if gotTokenID != wantTokenID {
		t.Fatalf("filter.Argmax = %d, want %d", gotTokenID, wantTokenID)
	}

	// 3. Verify forward projection and draft steps with synthetic model.
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 2)
	if err != nil {
		t.Fatalf("NewQwen35MTPDraftSession: %v", err)
	}
	t.Cleanup(d.Close)

	// Synthetic model VocabSize is 3: token IDs are 0, 1, 2.
	// Truncate to subset {0, 2}
	d.WithDraftVocabFilter([]int{0, 2})
	if d.DraftVocabFilter() == nil || d.DraftVocabFilter().Len() != 2 {
		t.Fatalf("session DraftVocabFilter length = %v, want 2", d.DraftVocabFilter())
	}

	draft := d.Propose(prompt)
	if err := d.Err(); err != nil {
		t.Fatalf("proposal with draft vocab filter error: %v", err)
	}
	if len(draft) == 0 {
		t.Fatal("propose with covered tokens returned empty draft")
	}
	for _, tok := range draft {
		if tok != 0 && tok != 2 {
			t.Fatalf("proposed draft token %d not in covered subset {0, 2}", tok)
		}
	}

	// 4. Verify end-to-end speculative decode with Qwen35MTPDraftConfig.
	target2 := m.NewSession()
	t.Cleanup(target2.Close)
	cfg := Qwen35MTPDraftConfig{
		Depth:            2,
		DraftVocabFilter: NewDraftVocabFilter([]int{0, 1, 2}),
	}
	run, err := SpecDecodeGreedyQwen35MTPConfig(target2, prompt, 4, cfg)
	if err != nil {
		t.Fatalf("SpecDecodeGreedyQwen35MTPConfig: %v", err)
	}
	if len(run.Output) != 4 {
		t.Fatalf("run.Output length = %d, want 4", len(run.Output))
	}
}

func TestDraftVocabTruncation_FallbackAndBoundaries(t *testing.T) {
	// 1. Test coverage threshold logic.
	filter := NewDraftVocabFilterWithThreshold([]int{10, 20, 30}, 0.8)
	if filter.CoverageThreshold != 0.8 {
		t.Fatalf("CoverageThreshold = %f, want 0.8", filter.CoverageThreshold)
	}

	// High-confidence distribution: token 20 clearly dominates.
	highConf := []float32{1.0, 10.0, 1.0}
	tok, prob, ok := filter.ArgmaxWithProb(highConf)
	if !ok || tok != 20 || prob < 0.8 {
		t.Fatalf("high confidence ArgmaxWithProb = (%d, %f, %v), want (20, >=0.8, true)", tok, prob, ok)
	}

	// Low-confidence distribution: probability spread evenly, max prob < 0.8.
	lowConf := []float32{2.0, 2.1, 2.0}
	tok, prob, ok = filter.ArgmaxWithProb(lowConf)
	if ok || tok != 20 || prob >= 0.8 {
		t.Fatalf("low confidence ArgmaxWithProb = (%d, %f, %v), want (20, <0.8, false)", tok, prob, ok)
	}

	// 2. Test fallback to standard decode when probability is below threshold in session.
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 2)
	if err != nil {
		t.Fatalf("NewQwen35MTPDraftSession: %v", err)
	}
	t.Cleanup(d.Close)

	// Set impossible coverage threshold -> forces clean fallback/rejection (empty draft).
	d.WithDraftVocabFilterThreshold([]int{0, 1, 2}, 0.99999)
	draft := d.Propose(prompt)
	if draft != nil {
		t.Fatalf("propose with below-threshold coverage returned %v, want nil (fallback)", draft)
	}
	if err := d.Err(); err != nil {
		t.Fatalf("fallback latched error unexpectedly: %v", err)
	}
	// Target cache must not be mutated by fallback.
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("target cache length = %d, want %d", target.Cache.Len(), len(prompt))
	}

	// 3. Test filter bypassed: when candidate token is outside truncated subset.
	d2, err := NewQwen35MTPDraftSession(target, 2)
	if err != nil {
		t.Fatalf("NewQwen35MTPDraftSession: %v", err)
	}
	t.Cleanup(d2.Close)
	// Filter covers only token 0
	d2.WithDraftVocabFilter([]int{0})
	// Full logits where token 2 wins -> filter bypassed with un-covered token
	fullLogits := []float32{0.0, 0.0, 10.0}
	cand, ok := d2.selectCandidate(fullLogits)
	if ok {
		t.Fatalf("selectCandidate on bypassed uncovered token returned ok=true, want false")
	}
	if cand != 2 {
		t.Fatalf("selectCandidate cand = %d, want 2", cand)
	}

	// 4. Test boundary conditions.
	if nilFilter := NewDraftVocabFilter(nil); nilFilter != nil {
		t.Fatalf("NewDraftVocabFilter(nil) = %v, want nil", nilFilter)
	}
	if emptyFilter := NewDraftVocabFilter([]int{}); emptyFilter != nil {
		t.Fatalf("NewDraftVocabFilter([]) = %v, want nil", emptyFilter)
	}

	// Out of bounds token handling in parMatRowsSubset:
	w := []float32{1, 2, 3, 4} // 2 tokens, hidden size 2
	x := []float32{1, 1}
	oobSubset := []int{-1, 0, 1, 5}
	oobLogits := parMatRowsSubset(w, x, oobSubset, 2)
	if len(oobLogits) != 4 {
		t.Fatalf("len(oobLogits) = %d, want 4", len(oobLogits))
	}
	if !math.IsInf(float64(oobLogits[0]), -1) {
		t.Fatalf("oobLogits[0] = %f, want -Inf", oobLogits[0])
	}
	if oobLogits[1] != 3.0 { // 1*1 + 2*1
		t.Fatalf("oobLogits[1] = %f, want 3.0", oobLogits[1])
	}
	if oobLogits[2] != 7.0 { // 3*1 + 4*1
		t.Fatalf("oobLogits[2] = %f, want 7.0", oobLogits[2])
	}
	if !math.IsInf(float64(oobLogits[3]), -1) {
		t.Fatalf("oobLogits[3] = %f, want -Inf", oobLogits[3])
	}
}

func TestDraftVocabTruncation_Speedup(t *testing.T) {
	// Micro-benchmark matrix projection: 40k subset vs full 248k vocabulary.
	const fullVocab = 248000
	const subsetVocab = 40000
	const hiddenSize = 64

	// 1. Verify arithmetic FLOPs reduction: 248k vs 40k is 6.2x lower FLOPs.
	fullFLOPs := int64(2) * fullVocab * hiddenSize
	subsetFLOPs := int64(2) * subsetVocab * hiddenSize
	flopRatio := float64(fullFLOPs) / float64(subsetFLOPs)
	if flopRatio < 2.5 {
		t.Fatalf("FLOPs reduction = %.2fx, want >= 2.5x", flopRatio)
	}

	w := make([]float32, fullVocab*hiddenSize)
	for i := range w {
		w[i] = float32((i%100)+1) * 0.001
	}
	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = float32(i+1) * 0.01
	}
	subset := make([]int, subsetVocab)
	for i := range subset {
		subset[i] = i
	}

	// Warm-up both paths.
	_ = parMatRows(w, x, fullVocab, hiddenSize)
	_ = parMatRowsSubset(w, x, subset, hiddenSize)

	// Measure full 248k projection and 40k subset projection over multiple iterations.
	const iters = 8
	var bestSpeedup float64
	var bestFull, bestSub time.Duration
	for trial := 0; trial < 3; trial++ {
		t0 := time.Now()
		for iter := 0; iter < iters; iter++ {
			_ = parMatRows(w, x, fullVocab, hiddenSize)
		}
		trialFull := time.Since(t0)

		t1 := time.Now()
		for iter := 0; iter < iters; iter++ {
			_ = parMatRowsSubset(w, x, subset, hiddenSize)
		}
		trialSub := time.Since(t1)

		if trialSub > 0 {
			speedup := float64(trialFull) / float64(trialSub)
			if speedup > bestSpeedup {
				bestSpeedup = speedup
				bestFull = trialFull
				bestSub = trialSub
			}
		}
	}

	t.Logf("248k full projection (%d iters): %v, 40k subset projection (%d iters): %v, speedup: %.2fx (FLOP reduction: %.2fx)",
		iters, bestFull, iters, bestSub, bestSpeedup, flopRatio)

	if bestSpeedup < 2.5 {
		t.Fatalf("projection speedup = %.2fx, want >= 2.5x", bestSpeedup)
	}
}

func BenchmarkDraftVocabTruncation_40k_vs_248k(b *testing.B) {
	const fullVocab = 248000
	const subsetVocab = 40000
	const hiddenSize = 128

	w := make([]float32, fullVocab*hiddenSize)
	for i := range w {
		w[i] = float32((i%100)+1) * 0.001
	}
	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = float32(i+1) * 0.01
	}
	subset := make([]int, subsetVocab)
	for i := range subset {
		subset[i] = i
	}

	b.Run("full_248k", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = parMatRows(w, x, fullVocab, hiddenSize)
		}
	})

	b.Run("truncated_40k", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = parMatRowsSubset(w, x, subset, hiddenSize)
		}
	})
}
