package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func qwen35MTPSeedSelectingCoordinate(t *testing.T, logits []float32, wantNot int) (Qwen38SamplerConfig, []float32, int) {
	t.Helper()
	for seed := uint64(0); seed < 100; seed++ {
		cfg := Qwen38SamplerConfig{Temperature: 1, TopP: 1, Seed: seed}
		sampler, err := NewQwen38SampledSpeculativeVerifier(cfg)
		if err != nil {
			t.Fatal(err)
		}
		q, err := sampler.ApplySamplerToLogits(logits)
		if err != nil {
			t.Fatal(err)
		}
		coordinate := sampler.SampleFromDistribution(q, sampler.PRNG().NextFloat32())
		if coordinate != wantNot {
			return cfg, q, coordinate
		}
	}
	t.Fatal("no deterministic non-argmax sampler seed found")
	return Qwen38SamplerConfig{}, nil, -1
}

func qwen35MTPSetFixedDraftLogits(d *Qwen35MTPDraftSession, logits []float32) {
	d.step = func(_ *Qwen35MTPForward, _ int, _, _ []float32) ([]float32, []float32, error) {
		return make([]float32, d.target.M.Cfg.HiddenSize), append([]float32(nil), logits...), nil
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftSampledProcessedQSelectsNonArgmax(t *testing.T) {
	logits := []float32{0, 0, 0}
	cfg, wantQ, wantCoordinate := qwen35MTPSeedSelectingCoordinate(t, logits, 0)
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)
	d, err := NewQwen35MTPDraftSessionWithConfig(target, Qwen35MTPDraftConfig{Depth: 1, SamplerConfig: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	qwen35MTPSetFixedDraftLogits(d, logits)

	draft, evidence := d.ProposeWithEvidence(prompt)
	if !reflect.DeepEqual(draft, []int{wantCoordinate}) || wantCoordinate == argmaxF32(logits) {
		t.Fatalf("sampled draft = %v, want non-argmax coordinate %d", draft, wantCoordinate)
	}
	if len(evidence) != 1 || evidence[0].Provenance != DraftEvidenceSampledProcessedQ || evidence[0].Token != wantCoordinate {
		t.Fatalf("sampled evidence = %+v", evidence)
	}
	for i, probability := range wantQ {
		want := float32(math.Log(float64(probability)))
		if evidence[0].LogProbs[i] != want {
			t.Fatalf("log q[%d] = %g, want exact %g", i, evidence[0].LogProbs[i], want)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftSampledCompactCoordinates(t *testing.T) {
	logits := []float32{0, 0}
	cfg, wantQ, wantCoordinate := qwen35MTPSeedSelectingCoordinate(t, logits, 0)
	subset := []int{2, 0}
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)
	d, err := NewQwen35MTPDraftSessionWithConfig(target, Qwen35MTPDraftConfig{
		Depth:            1,
		DraftVocabFilter: NewDraftVocabFilter(subset),
		SamplerConfig:    &cfg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	qwen35MTPSetFixedDraftLogits(d, logits)

	draft, evidence := d.ProposeWithEvidence(prompt)
	wantToken := subset[wantCoordinate]
	if !reflect.DeepEqual(draft, []int{wantToken}) || len(evidence) != 1 {
		t.Fatalf("compact sampled proposal = %v/%+v, want token %d", draft, evidence, wantToken)
	}
	if !reflect.DeepEqual(evidence[0].TokenIDs, subset) || evidence[0].Token != wantToken {
		t.Fatalf("compact coordinate evidence = %+v", evidence[0])
	}
	for i, probability := range wantQ {
		if want := float32(math.Log(float64(probability))); evidence[0].LogProbs[i] != want {
			t.Fatalf("compact log q[%d] = %g, want exact %g", i, evidence[0].LogProbs[i], want)
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftSampledThresholdRejectsSelectedLowProbabilityAndRollsBackRNG(t *testing.T) {
	logits := []float32{1, 0}
	cfg, q, selected := qwen35MTPSeedSelectingCoordinate(t, logits, 0)
	if q[selected] >= 0.5 {
		t.Fatalf("fixture selected q[%d]=%g, want below threshold", selected, q[selected])
	}
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)
	d, err := NewQwen35MTPDraftSessionWithConfig(target, Qwen35MTPDraftConfig{
		Depth:             1,
		CoverageThreshold: 0.5,
		SamplerConfig:     &cfg,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	qwen35MTPSetFixedDraftLogits(d, logits)
	before := d.sampler.CheckpointRNG()
	draft, evidence := d.ProposeWithEvidence(prompt)
	after := d.sampler.CheckpointRNG()
	if draft != nil || evidence != nil {
		t.Fatalf("low-probability selected token escaped threshold: %v/%+v", draft, evidence)
	}
	if after != before {
		t.Fatalf("refused proposal advanced RNG: before=%+v after=%+v", before, after)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftInvalidSamplerConfigRefused(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)
	cfg := Qwen38SamplerConfig{Temperature: 1, TopP: 1, TopK: 4}
	d, err := NewQwen35MTPDraftSessionWithConfig(target, Qwen35MTPDraftConfig{Depth: 1, SamplerConfig: &cfg})
	if d != nil || !errors.Is(err, ErrSamplingModeUnsupported) {
		t.Fatalf("invalid sampler construction = (%v, %v), want nil/%v", d, err, ErrSamplingModeUnsupported)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftNilAndGreedySamplerPreserveLegacyProposal(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	newTarget := func() *Session {
		target := m.NewSession()
		target.captureTargetHidden = true
		t.Cleanup(target.Close)
		target.Prefill(prompt)
		return target
	}
	legacy, err := NewQwen35MTPDraftSessionWithConfig(newTarget(), Qwen35MTPDraftConfig{Depth: 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(legacy.Close)
	greedyCfg := Qwen38SamplerConfig{Temperature: 0, Greedy: true, Seed: 99}
	greedy, err := NewQwen35MTPDraftSessionWithConfig(newTarget(), Qwen35MTPDraftConfig{Depth: 2, SamplerConfig: &greedyCfg})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(greedy.Close)

	legacyDraft, legacyEvidence := legacy.ProposeWithEvidence(prompt)
	greedyDraft, greedyEvidence := greedy.ProposeWithEvidence(prompt)
	if greedy.sampler != nil || !reflect.DeepEqual(greedyDraft, legacyDraft) || !reflect.DeepEqual(greedyEvidence, legacyEvidence) {
		t.Fatalf("greedy config changed legacy proposal: legacy=%v/%+v greedy=%v/%+v", legacyDraft, legacyEvidence, greedyDraft, greedyEvidence)
	}
	for i, row := range greedyEvidence {
		if row.Provenance != DraftEvidenceModelSoftmaxGreedy {
			t.Fatalf("row %d provenance = %q, want legacy %q", i, row.Provenance, DraftEvidenceModelSoftmaxGreedy)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestQwen35MTPDraftSampledConfigRefusedByGreedySpecDecode(t *testing.T) {
	cfg := Qwen38SamplerConfig{Temperature: 1, TopP: 1, Seed: 7}
	run, err := SpecDecodeGreedyQwen35MTPConfig(nil, nil, 1, Qwen35MTPDraftConfig{Depth: 1, SamplerConfig: &cfg})
	var unsupported *Qwen35MTPSpecDecodeUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("sampled config error = %v, want typed Qwen35 speculative-decode refusal", err)
	}
	if len(run.Output) != 0 || run.Rounds != 0 || run.DraftedTokens != 0 || run.AcceptedDrafts != 0 {
		t.Fatalf("sampled config returned run before refusal: %+v", run)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftGreedySamplerConfigAllowedByGreedySpecDecode(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	legacyTarget := m.NewSession()
	t.Cleanup(legacyTarget.Close)
	legacy, err := SpecDecodeGreedyQwen35MTPConfig(legacyTarget, prompt, 3, Qwen35MTPDraftConfig{Depth: 1})
	if err != nil {
		t.Fatal(err)
	}
	greedyTarget := m.NewSession()
	t.Cleanup(greedyTarget.Close)
	greedyCfg := Qwen38SamplerConfig{Temperature: 0, Greedy: true, Seed: 7}
	greedy, err := SpecDecodeGreedyQwen35MTPConfig(greedyTarget, prompt, 3, Qwen35MTPDraftConfig{Depth: 1, SamplerConfig: &greedyCfg})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(greedy, legacy) {
		t.Fatalf("greedy sampler config changed legacy greedy decode: legacy=%+v greedy=%+v", legacy, greedy)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPDraftLogprobEvidence(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	d.WithDraftVocabFilter([]int{0, 1})

	draft, evidence := d.ProposeWithEvidence(prompt)
	if len(draft) != 3 || len(evidence) != len(draft) {
		t.Fatalf("draft/evidence lengths = %d/%d, want 3/3", len(draft), len(evidence))
	}
	for i, row := range evidence {
		if row.Token != draft[i] {
			t.Fatalf("row %d token = %d, want %d", i, row.Token, draft[i])
		}
		if !reflect.DeepEqual(row.TokenIDs, []int{0, 1}) {
			t.Fatalf("row %d token IDs = %v, want compact coordinates [0 1]", i, row.TokenIDs)
		}
		if len(row.LogProbs) != 2 {
			t.Fatalf("row %d log-probability width = %d, want 2", i, len(row.LogProbs))
		}
		var probabilityMass float64
		finite := 0
		for _, logp := range row.LogProbs {
			if !math.IsInf(float64(logp), -1) {
				finite++
				probabilityMass += math.Exp(float64(logp))
			}
		}
		if finite < 2 {
			t.Fatalf("row %d is one-hot evidence: %v", i, row.LogProbs)
		}
		if math.Abs(probabilityMass-1) > 1e-5 {
			t.Fatalf("row %d probability mass = %.8f, want 1", i, probabilityMass)
		}
	}

	evidence[0].LogProbs[0] = 99
	evidence[0].TokenIDs[0] = 99
	_, again := d.ProposeWithEvidence(prompt)
	if len(again) != 3 || again[0].LogProbs[0] == 99 || again[0].TokenIDs[0] == 99 {
		t.Fatalf("caller mutation leaked into subsequent evidence: %+v", again)
	}

	d.SetDraftVocabFilter(nil)
	realStep := d.step
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		feedback, _, err := realStep(f, pos, prior, embedding)
		return feedback, []float32{float32(math.Inf(1)), 0, 0, 0}, err
	}
	invalidDraft, invalidEvidence := d.ProposeWithEvidence(prompt)
	if len(invalidDraft) < 2 {
		t.Fatalf("non-finite later row did not exercise a selected draft token: %v", invalidDraft)
	}
	if invalidEvidence != nil {
		t.Fatalf("partial evidence escaped after a non-finite row: %+v", invalidEvidence)
	}

	d.SetCoverageThreshold(2)
	refused, stale := d.ProposeWithEvidence(prompt)
	if refused != nil || stale != nil {
		t.Fatalf("clean refusal returned proposal/evidence = %v/%+v", refused, stale)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestMTPProposalGeneratorCarriesDraftEvidence(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	d.WithDraftVocabFilter([]int{0, 1})

	proposal, err := NewMTPProposalGenerator(d).Propose(context.Background(), prompt, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(proposal.Tokens) != 2 || len(proposal.DraftEvidence) != 2 {
		t.Fatalf("proposal token/evidence lengths = %d/%d, want 2/2", len(proposal.Tokens), len(proposal.DraftEvidence))
	}
	for i := range proposal.Tokens {
		if proposal.DraftEvidence[i].Token != proposal.Tokens[i] {
			t.Fatalf("proposal row %d token = %d, want %d", i, proposal.DraftEvidence[i].Token, proposal.Tokens[i])
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestDraftProposalStochasticEvidenceAdmission(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)

	d, err := NewQwen35MTPDraftSession(target, 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	d.WithDraftVocabFilter([]int{0, 1})

	proposal, err := NewMTPProposalGenerator(d).Propose(context.Background(), prompt, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(proposal.DraftEvidence) != 2 {
		t.Fatalf("draft evidence rows = %d, want 2", len(proposal.DraftEvidence))
	}
	for i, row := range proposal.DraftEvidence {
		if row.Provenance != DraftEvidenceModelSoftmaxGreedy {
			t.Fatalf("row %d provenance = %q, want %q", i, row.Provenance, DraftEvidenceModelSoftmaxGreedy)
		}
		finite := 0
		for _, logp := range row.LogProbs {
			if !math.IsInf(float64(logp), -1) {
				finite++
			}
		}
		if finite < 2 {
			t.Fatalf("row %d did not exercise nondegenerate model evidence: %v", i, row.LogProbs)
		}
	}
	if admitted, err := proposal.StochasticDraftEvidence(); admitted != nil || !errors.Is(err, ErrDraftEvidenceNotProcessedSamplingQ) {
		t.Fatalf("greedy model evidence admission = (%+v, %v), want nil/%v", admitted, err, ErrDraftEvidenceNotProcessedSamplingQ)
	}

	impossibleSelection := DraftProposal{
		Tokens: []int{2},
		DraftEvidence: []DraftTokenEvidence{{
			Token:      2,
			LogProbs:   []float32{float32(math.Log(0.4)), float32(math.Log(0.6)), float32(math.Inf(-1))},
			Provenance: DraftEvidenceSampledProcessedQ,
		}},
	}
	if admitted, err := impossibleSelection.StochasticDraftEvidence(); admitted != nil || !errors.Is(err, ErrDraftEvidenceIncomplete) {
		t.Fatalf("zero-q selected token admission = (%+v, %v), want nil/%v", admitted, err, ErrDraftEvidenceIncomplete)
	}
}

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
		{name: "quant format", depth: 2, mutate: func(s *Session) { s.Quant = true }, reason: "this quant format remains unsupported"},
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

	// Wall-clock speedup on a shared host is load, not a verdict; BenchmarkDraftVocabTruncation_40k_vs_248k
	// measures it. The subset projection must equal the full projection on the kept rows.
	full := parMatRows(w, x, fullVocab, hiddenSize)
	sub := parMatRowsSubset(w, x, subset, hiddenSize)
	if len(sub) != len(subset) {
		t.Fatalf("subset projection has %d rows, want %d", len(sub), len(subset))
	}
	for i, row := range subset {
		if sub[i] != full[row] {
			t.Fatalf("subset row %d = %g, want the full projection's row %d = %g", i, sub[i], row, full[row])
		}
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

// fak-test:runtime fast est=1ms lane=default
func TestDraftVocabAllMaskedRefusalAndLegacyControls(t *testing.T) {
	negInf := float32(math.Inf(-1))
	for _, tc := range []struct {
		name      string
		subset    []int
		logits    []float32
		threshold float32
		token     int
		prob      float32
		ok        bool
	}{
		{"empty-filter", nil, []float32{1}, 0, -1, 0, false},
		{"empty-logits", []int{4}, nil, 0, -1, 0, false},
		{"one-masked", []int{4}, []float32{negInf}, 0, -1, 0, false},
		{"all-masked", []int{-1, 99}, []float32{negInf, negInf}, 0, -1, 0, false},
		{"all-masked-threshold", []int{4, 5}, []float32{negInf, negInf}, .5, -1, 0, false},
		{"ignored-extra-logit", []int{4}, []float32{negInf, 1}, 0, -1, 0, false},
		{"ignored-extra-nan", []int{4}, []float32{negInf, float32(math.NaN())}, 0, -1, 0, false},
		{"ignored-extra-positive-infinity", []int{4}, []float32{negInf, float32(math.Inf(1))}, 0, -1, 0, false},
		{"shorter-masked-logits", []int{4, 5}, []float32{negInf}, 0, -1, 0, false},
		{"finite-after-mask", []int{4, 5}, []float32{negInf, -3}, 0, 5, 1, true},
		{"finite-tie-first", []int{5, 4}, []float32{2, 2}, .5, 5, .5, true},
		{"finite-tie-threshold", []int{5, 4}, []float32{2, 2}, .75, 5, .5, false},
		// These are compatibility controls, not endorsements of a wider IEEE policy.
		{"legacy-nan-first", []int{4, 5}, []float32{float32(math.NaN()), 1}, 0, 4, 0, true},
		{"legacy-nan-after-mask", []int{4, 5}, []float32{negInf, float32(math.NaN())}, 0, 4, 0, true},
		{"legacy-positive-infinity", []int{4, 5}, []float32{1, float32(math.Inf(1))}, 0, 5, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := NewDraftVocabFilterWithThreshold(tc.subset, tc.threshold)
			token, prob, ok := f.ArgmaxWithProb(tc.logits)
			if token != tc.token || prob != tc.prob || ok != tc.ok {
				t.Fatalf("selection = (%d,%g,%v), want (%d,%g,%v)", token, prob, ok, tc.token, tc.prob, tc.ok)
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPAllMaskedProjectionRefusesDepthOne(t *testing.T) {
	for _, tc := range []struct {
		name       string
		subset     []int
		suppressed []int
		wantCount  int
	}{
		{"invalid-subset", []int{-1, 99}, nil, 0},
		{"suppressed-subset", []int{0, 1}, []int{0, 1}, 0},
		{"finite-subset", []int{0}, nil, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := qwen35MTPEnabledSyntheticModel(t)
			target := m.NewSession()
			target.captureTargetHidden = true
			t.Cleanup(target.Close)
			prompt := []int{0, 1}
			target.Prefill(prompt)
			d, err := NewQwen35MTPDraftSession(target, 1)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(d.Close)
			d.WithDraftVocabFilter(tc.subset)
			d.forward.draft.M.Cfg.SuppressTokens = append([]int(nil), tc.suppressed...)
			hidden := make([]float32, m.Cfg.HiddenSize)
			for i := range hidden {
				hidden[i] = 1
			}
			logits := d.forward.ProjectFiltered(hidden, tc.subset)
			if len(logits) != len(tc.subset) {
				t.Fatalf("projection length %d", len(logits))
			}
			if tc.wantCount == 0 {
				for i, v := range logits {
					if !math.IsInf(float64(v), -1) {
						t.Fatalf("masked row %d = %g", i, v)
					}
				}
				if token, ok := d.selectCandidate(logits); ok || token != -1 {
					t.Fatalf("masked projection selected (%d,%v)", token, ok)
				}
			}
			draft := d.Propose(prompt)
			if len(draft) != tc.wantCount {
				t.Fatalf("draft = %v, want length %d", draft, tc.wantCount)
			}
			if tc.wantCount == 0 && draft != nil {
				t.Fatalf("refusal must be nil: %v", draft)
			}
			for _, token := range draft {
				if token != 0 {
					t.Fatalf("unexpected candidate %d", token)
				}
			}
			if d.Err() != nil {
				t.Fatalf("clean refusal latched error: %v", d.Err())
			}
			if target.Cache.Len() != len(prompt) {
				t.Fatalf("target cache mutated: %d", target.Cache.Len())
			}
			if d.pending != nil {
				t.Fatal("depth-one selection created a speculative checkpoint")
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPLaterMaskedRowStopsAndFiniteRetryContinues(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	prompt := []int{0, 1}
	target.Prefill(prompt)
	d, err := NewQwen35MTPDraftSession(target, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	d.WithDraftVocabFilter([]int{0, 1})
	realStep := d.step
	maskLater := true
	calls := 0
	d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
		calls++
		feedback, logits, err := realStep(f, pos, prior, embedding)
		if err == nil && maskLater && pos >= len(prompt) {
			logits = []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}
		}
		return feedback, logits, err
	}
	draft := d.Propose(prompt)
	if len(draft) != 1 || draft[0] < 0 || draft[0] > 1 {
		t.Fatalf("partial draft = %v", draft)
	}
	if calls != len(prompt)+1 {
		t.Fatalf("forward calls = %d, want %d", calls, len(prompt)+1)
	}
	if d.Err() != nil {
		t.Fatal(d.Err())
	}
	maskLater = false
	draft = d.Propose(prompt)
	if len(draft) != 3 {
		t.Fatalf("finite retry = %v", draft)
	}
	for _, token := range draft {
		if token < 0 || token > 1 {
			t.Fatalf("invalid candidate %d", token)
		}
	}
	if d.Err() != nil {
		t.Fatal(d.Err())
	}
	if target.Cache.Len() != len(prompt) {
		t.Fatalf("target cache mutated: %d", target.Cache.Len())
	}
}

// fak-test:runtime medium est=2s lane=default
func TestQwen35MTPFilterUpdateReplaysProjectionCoordinates(t *testing.T) {
	for _, depth := range []int{1, 3} {
		for _, update := range []string{"reorder", "resize", "remove", "empty"} {
			t.Run(fmt.Sprintf("depth-%d/%s", depth, update), func(t *testing.T) {
				m := qwen35MTPEnabledSyntheticModel(t)
				prompt := []int{0, 1}
				target := m.NewSession()
				target.captureTargetHidden = true
				t.Cleanup(target.Close)
				target.Prefill(prompt)
				wantTarget := m.NewSession()
				wantTarget.captureTargetHidden = true
				t.Cleanup(wantTarget.Close)
				wantTarget.Prefill(prompt)
				d, err := NewQwen35MTPDraftSession(target, depth)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(d.Close)
				d.WithDraftVocabFilter([]int{0, 2})
				if d.filterDirty {
					t.Fatal("pre-proposal filter dirtied an empty projection cache")
				}
				var positions []int
				realStep := d.step
				d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
					positions = append(positions, pos)
					return realStep(f, pos, prior, embedding)
				}
				if got := d.Propose(prompt); len(got) != depth || d.Err() != nil {
					t.Fatalf("initial proposal %v: %v", got, d.Err())
				}
				if len(positions) != len(prompt)+depth-1 {
					t.Fatalf("initial calls %v", positions)
				}
				old := d.forward
				pending := d.pending
				if depth > 1 && pending == nil {
					t.Fatal("multi-step fixture has no pending snapshot")
				}
				var filter *DraftVocabFilter
				switch update {
				case "reorder":
					filter = NewDraftVocabFilter([]int{2, 0})
				case "resize":
					filter = NewDraftVocabFilter([]int{1})
				case "empty":
					filter = &DraftVocabFilter{}
				}
				d.SetDraftVocabFilter(filter)
				if !d.filterDirty || old.closed || d.pending != pending {
					t.Fatal("setter eagerly replaced or failed to invalidate live draft state")
				}
				positions = nil
				got := d.Propose(prompt)
				if d.Err() != nil {
					t.Fatal(d.Err())
				}
				if d.filterDirty || d.forward == old || !old.closed {
					t.Fatal("successful replay did not replace and clean the draft owner")
				}
				if old.draft.Cache.Len() != len(prompt) {
					t.Fatalf("old cache was not restored before close: %d", old.draft.Cache.Len())
				}
				if pending != nil && pending.snapshot.Cache != nil {
					t.Fatal("old proposal snapshot was not released")
				}
				if len(positions) != len(prompt)+depth-1 {
					t.Fatalf("update replay calls %v", positions)
				}
				for i, pos := range positions {
					if pos != i {
						t.Fatalf("replay position %d = %d", i, pos)
					}
				}
				fresh, err := NewQwen35MTPDraftSession(wantTarget, depth)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(fresh.Close)
				fresh.SetDraftVocabFilter(filter)
				want := fresh.Propose(prompt)
				if fresh.Err() != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("updated %v versus fresh %v: %v", got, want, fresh.Err())
				}
				assertFloat32BitsEqual(t, "updated committed projection", d.lastLogits, fresh.lastLogits)
				assertQwen35MTPTargetStateEqual(t, target, wantTarget)
				owner := d.forward
				positions = nil
				if again := d.Propose(prompt); !reflect.DeepEqual(again, want) || d.Err() != nil {
					t.Fatalf("fixed-filter repeat %v: %v", again, d.Err())
				}
				if d.forward != owner || len(positions) != depth-1 {
					t.Fatalf("unchanged filter replayed committed work: %v", positions)
				}
			})
		}
	}

}

// fak-test:runtime medium est=1s lane=default
func TestQwen35MTPFilterUpdateFailuresKeepExistingLatch(t *testing.T) {
	for _, stage := range []string{"restore", "recreate", "replay"} {
		t.Run(stage, func(t *testing.T) {
			m := qwen35MTPEnabledSyntheticModel(t)
			prompt := []int{0, 1}
			target := m.NewSession()
			target.captureTargetHidden = true
			t.Cleanup(target.Close)
			target.Prefill(prompt)
			wantTarget := m.NewSession()
			wantTarget.captureTargetHidden = true
			t.Cleanup(wantTarget.Close)
			wantTarget.Prefill(prompt)
			d, err := NewQwen35MTPDraftSession(target, 3)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(d.Close)
			d.WithDraftVocabFilter([]int{0, 2})
			if got := d.Propose(prompt); len(got) != 3 || d.Err() != nil {
				t.Fatalf("initial %v: %v", got, d.Err())
			}
			old, pending := d.forward, d.pending
			d.WithDraftVocabFilter([]int{2, 0})
			injected := errors.New("filter-update replay failure")
			calls := 0
			realStep := d.step
			d.step = func(f *Qwen35MTPForward, pos int, prior, embedding []float32) ([]float32, []float32, error) {
				calls++
				feedback, logits, err := realStep(f, pos, prior, embedding)
				if err == nil && stage == "replay" {
					return nil, nil, injected
				}
				return feedback, logits, err
			}
			meta := m.manifest["mtp.fc.weight"]
			if stage == "restore" {
				pending.snapshot.Close()
			}
			if stage == "recreate" {
				delete(m.manifest, "mtp.fc.weight")
			}
			got := d.Propose(prompt)
			m.manifest["mtp.fc.weight"] = meta
			if got != nil || d.Err() == nil || !d.filterDirty {
				t.Fatalf("failed update = %v, error %v, dirty %v", got, d.Err(), d.filterDirty)
			}
			var failure *Qwen35MTPDrafterError
			if !errors.As(d.Err(), &failure) {
				t.Fatalf("untyped failure %v", d.Err())
			}
			wantStage := "committed catch-up"
			if stage == "restore" {
				wantStage = "rollback"
			}
			if failure.Stage != wantStage {
				t.Fatalf("failure stage %q, want %q", failure.Stage, wantStage)
			}
			if d.pending != nil || pending.snapshot.Cache != nil {
				t.Fatal("failure retained old proposal snapshot")
			}
			switch stage {
			case "restore":
				if old.closed || d.forward != old || calls != 0 {
					t.Fatal("restore failure continued to replace draft owner")
				}
			case "recreate":
				if !old.closed || d.forward != nil || calls != 0 {
					t.Fatal("recreate failure retained owner or executed replay")
				}
			case "replay":
				if !errors.Is(d.Err(), injected) || !old.closed || d.forward == old || calls != 1 {
					t.Fatalf("replay failure ownership/cause: %v calls=%d", d.Err(), calls)
				}
				if d.forward.draft.Cache.Len() != 0 || d.forward.lastPos != -1 || len(d.processed) != 0 || len(d.lastLogits) != 0 {
					t.Fatal("failed replay did not restore its new empty draft boundary")
				}
			}
			latched, before := d.Err(), calls
			d.SetDraftVocabFilter(nil)
			if again := d.Propose(prompt); again != nil || d.Err() != latched || calls != before {
				t.Fatal("filter update bypassed existing runtime-error latch")
			}
			assertQwen35MTPTargetStateEqual(t, target, wantTarget)
		})
	}
}
