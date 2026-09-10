package agent

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type speculativeRequestProbe struct {
	calls     int
	maxDrafts []int
	tokens    []int
	err       error
	after     func()
}

type prefixReuseQwenBackend struct{ compute.Backend }

type boundaryRejectQwenBackend struct{ *prefixReuseQwenBackend }

type speculativeStepCountBackend struct {
	compute.Backend
	matMulCalls  int
	readCalls    int
	cancelOnRead int
	cancel       func()
}

func (b *speculativeStepCountBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matMulCalls++
	return b.Backend.MatMul(w, x)
}

func (b *speculativeStepCountBackend) Read(t compute.Tensor) []float32 {
	b.readCalls++
	out := b.Backend.Read(t)
	if b.cancel != nil && b.readCalls == b.cancelOnRead {
		b.cancel()
	}
	return out
}

func (*prefixReuseQwenBackend) Name() string          { return "prefix-reuse-device" }
func (*prefixReuseQwenBackend) Qwen35GDNPath() string { return model.Qwen35GDNVulkanPath }

func (*boundaryRejectQwenBackend) Qwen35SequenceAllLogitsPath() string {
	return compute.Qwen35SequenceAllLogitsPath
}

func (b *prefixReuseQwenBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	return b.Backend.(compute.TensorCloner).CloneTensor(t)
}

func (b *prefixReuseQwenBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	output := compute.NewF32(b.Backend, append([]int(nil), normalizedInput.Shape...), append([]float32(nil), b.Read(normalizedInput)...))
	return output, convState, recurrentState, nil
}

func (*speculativeRequestProbe) Name() string { return "request-probe" }

func (p *speculativeRequestProbe) Propose(ctx context.Context, _ []int, maxDraft int) (model.DraftProposal, error) {
	p.calls++
	p.maxDrafts = append(p.maxDrafts, maxDraft)
	if err := ctx.Err(); err != nil {
		return model.DraftProposal{}, err
	}
	if p.err != nil {
		if p.after != nil {
			p.after()
		}
		return model.DraftProposal{}, p.err
	}
	tokens := append([]int(nil), p.tokens...)
	if p.after != nil {
		p.after()
	}
	return model.NewLinearProposal(tokens, nil), nil
}

func speculativeRequestPlanner(t *testing.T, probe *speculativeRequestProbe) (*InKernelPlanner, []int) {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, nil, "speculative-request", false, nil, false)
	p.EnableSpeculativeDecoding(probe, 4)
	return p, []int{10, 20, 30, 40, 10, 20, 30}
}

func speculativeHybridConfig() model.Config {
	return model.Config{
		ModelType:             "qwen3_5_text",
		HiddenSize:            32,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      8,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    8,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

func speculativeAcceptedDraft(t *testing.T, m *model.Model, prompt []int, n int) []int {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	logits := s.Prefill(prompt)
	draft := make([]int, 0, n)
	for range n {
		token := model.ArgmaxF32(logits)
		draft = append(draft, token)
		logits = s.Step(token)
	}
	return draft
}

func speculativeBackendPlanner(t *testing.T, probe *speculativeRequestProbe) (*InKernelPlanner, []int, *speculativeStepCountBackend) {
	t.Helper()
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	backend := &speculativeStepCountBackend{Backend: base}
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	p := NewInKernelPlanner(m, nil, "speculative-request-lifetime", false, backend, false)
	p.EnableSpeculativeDecoding(probe, 4)
	return p, []int{10, 20, 30, 40, 10, 20, 30}, backend
}

func assertSpeculativeRequestStateReleased(t *testing.T, p *InKernelPlanner, prompt []int, probe *speculativeRequestProbe) {
	t.Helper()
	eng := p.SpeculativeEngine()
	if got := eng.LastLogits(); len(got) != 0 {
		t.Fatalf("shared speculative engine retained %d request logits", len(got))
	}
	probe.err = nil
	eng.SetLastLogits(make([]float32, p.m.Cfg.VocabSize))
	_, _, _, err := eng.Step(context.Background(), prompt, nil, nil)
	if !errors.Is(err, model.ErrSpeculativeNilTarget) {
		t.Fatalf("shared speculative engine retained request target: Step error=%v, want ErrSpeculativeNilTarget", err)
	}
	eng.SetLastLogits(nil)
}

func TestGreedySpeculativeRequestEligibilityExcludesScoreTransforms(t *testing.T) {
	p, _ := speculativeRequestPlanner(t, &speculativeRequestProbe{})
	tests := []struct {
		name                string
		temperature         float64
		bias                model.LogitBias
		frequency, presence float64
		want                bool
	}{
		{name: "raw greedy", want: true},
		{name: "negative zero remains greedy", temperature: -0.0, want: true},
		{name: "temperature sampling", temperature: 0.01},
		{name: "logit bias", bias: model.LogitBias{1: 0.25}},
		{name: "frequency penalty", frequency: 0.1},
		{name: "presence penalty", presence: 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.greedySpeculativeRequestEligible(tt.temperature, tt.bias, tt.frequency, tt.presence)
			if got != tt.want {
				t.Fatalf("eligibility = %v, want %v", got, tt.want)
			}
		})
	}
	p.DisableSpeculativeDecoding()
	if p.greedySpeculativeRequestEligible(0, nil, 0, 0) {
		t.Fatal("planner without an engine remained eligible")
	}
}

func TestVerifyGreedySpeculativeRoundBoundaryRejectReportsNoRollback(t *testing.T) {
	cfg := speculativeHybridConfig()
	m := model.NewSynthetic(cfg)
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref backend unavailable")
	}
	backend := &boundaryRejectQwenBackend{prefixReuseQwenBackend: &prefixReuseQwenBackend{Backend: base}}
	target, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()

	boundary := make([]float32, cfg.VocabSize)
	boundary[17] = 4
	proposal := model.NewLinearProposal([]int{18, 19, 20, 21}, nil)
	result, err := verifyGreedySpeculativeRound(context.Background(), target, nil, proposal, boundary, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.AcceptedTokens) != 0 || result.CorrectionToken != 17 {
		t.Fatalf("accepted/correction=%v/%d, want none/17", result.AcceptedTokens, result.CorrectionToken)
	}
	if result.RollbackKVCount != 0 {
		t.Fatalf("rollback=%d, want 0 for a boundary rejection with no target operation", result.RollbackKVCount)
	}
}

func TestSpeculativeColdSessionRetainsConfiguredBackend(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	backend := compute.Default()
	devicePlanner := NewInKernelPlanner(m, nil, "device", false, backend, false)
	deviceSession := devicePlanner.newSpeculativeSession()
	defer deviceSession.Close()
	if deviceSession.Backend != backend {
		t.Fatalf("cold speculative session backend = %T, want configured %T", deviceSession.Backend, backend)
	}

	hostPlanner := NewInKernelPlanner(m, nil, "host", false, nil, false)
	hostSession := hostPlanner.newSpeculativeSession()
	defer hostSession.Close()
	if hostSession.Backend != nil {
		t.Fatalf("host speculative session unexpectedly retained backend %T", hostSession.Backend)
	}
}

func TestSpeculativeDevicePrefixSnapshotRestoresRepeatedPrompt(t *testing.T) {
	cfg := speculativeHybridConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	backend := &prefixReuseQwenBackend{Backend: compute.Default()}
	prompt := []int{10, 20, 30, 40, 10, 20, 30}

	coldProbe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	cold := NewInKernelPlanner(m, nil, "cold-device", false, backend, false)
	cold.EnableSpeculativeDecoding(coldProbe, 4)
	var want []int
	if _, err := cold.generateReusedRecovering(context.Background(), prompt, 5, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		want = append(want, token)
		return false
	}); err != nil {
		t.Fatalf("cold device decode: %v", err)
	}

	reuseProbe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	reuse := NewInKernelPlanner(m, nil, "reused-device", false, backend, false)
	reuse.EnableSpeculativeDecoding(reuseProbe, 4)
	if reuse.tree == nil {
		t.Fatal("eligible hybrid backend did not initialize prefix reuse")
	}
	var first []int
	firstResult, err := reuse.generateReusedRecovering(context.Background(), prompt, 5, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		first = append(first, token)
		return false
	})
	if err != nil {
		t.Fatalf("first reusable device decode: %v", err)
	}
	if firstResult.matched != 0 || !reflect.DeepEqual(first, want) {
		t.Fatalf("first reusable result matched=%d output=%v, want cold output=%v", firstResult.matched, first, want)
	}
	var second []int
	secondResult, err := reuse.generateReusedRecovering(context.Background(), prompt, 5, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		second = append(second, token)
		return false
	})
	if err != nil {
		t.Fatalf("restored device decode: %v", err)
	}
	if secondResult.cacheable != len(prompt) || secondResult.matched != len(prompt) {
		t.Fatalf("restored cacheable/matched=%d/%d, want %d/%d", secondResult.cacheable, secondResult.matched, len(prompt), len(prompt))
	}
	if !reflect.DeepEqual(second, want) {
		t.Fatalf("restored output=%v, want cold output=%v", second, want)
	}
	if session := reuse.newSpeculativeSession(); session.Backend != backend {
		session.Close()
		t.Fatalf("reused planner constructed backend %T, want configured %T", session.Backend, backend)
	} else {
		session.Close()
	}
}

func TestNonGreedyRequestsBypassSpeculativeGenerator(t *testing.T) {
	tests := []struct {
		name                string
		temperature         float64
		bias                model.LogitBias
		frequency, presence float64
	}{
		{name: "temperature sampling", temperature: 0.7},
		{name: "logit bias", bias: model.LogitBias{1: 2}},
		{name: "frequency penalty", frequency: 0.5},
		{name: "presence penalty", presence: 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := &speculativeRequestProbe{tokens: []int{1, 2, 3, 4}}
			p, prompt := speculativeRequestPlanner(t, probe)
			var emitted []int
			res, err := p.generateReusedRecovering(context.Background(), prompt, 2, tt.temperature, 0.9, 8, tt.bias, tt.frequency, tt.presence, nil, func(token int) bool {
				emitted = append(emitted, token)
				return false
			})
			if err != nil {
				t.Fatalf("ordinary decode: %v", err)
			}
			if probe.calls != 0 {
				t.Fatalf("speculative generator calls = %d, want 0", probe.calls)
			}
			if res.gen != 2 || len(emitted) != 2 {
				t.Fatalf("ordinary generated/emitted = %d/%d, want 2/2", res.gen, len(emitted))
			}
		})
	}
}

func TestGreedySpeculativeLoopPreservesBudgetEmitCancellationAndProposalFallback(t *testing.T) {
	t.Run("clips every proposal to remaining output budget", func(t *testing.T) {
		probe := &speculativeRequestProbe{tokens: []int{0, 1, 2, 3, 4, 5}}
		p, prompt := speculativeRequestPlanner(t, probe)
		res, err := p.generateReusedRecovering(context.Background(), prompt, 3, 0, 0, 0, nil, 0, 0, nil, nil)
		if err != nil {
			t.Fatalf("speculative decode: %v", err)
		}
		if res.gen != 3 {
			t.Fatalf("generated = %d, want exact budget 3", res.gen)
		}
		if len(probe.maxDrafts) == 0 || probe.maxDrafts[0] != 3 {
			t.Fatalf("proposal budgets = %v, want first budget 3", probe.maxDrafts)
		}
		for _, budget := range probe.maxDrafts {
			if budget < 1 || budget > 3 {
				t.Fatalf("proposal budget %d escaped remaining output budget", budget)
			}
		}
	})

	t.Run("emit stop ends after counted token", func(t *testing.T) {
		probe := &speculativeRequestProbe{tokens: []int{0}}
		p, prompt := speculativeRequestPlanner(t, probe)
		var emitted []int
		res, err := p.generateReusedRecovering(context.Background(), prompt, 4, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			emitted = append(emitted, token)
			return true
		})
		if err != nil {
			t.Fatalf("speculative emit stop: %v", err)
		}
		if !res.stopped || res.gen != 1 || len(emitted) != 1 {
			t.Fatalf("stopped/generated/emitted = %v/%d/%v, want true/1/one token", res.stopped, res.gen, emitted)
		}
	})

	t.Run("cancellation remains an error rather than a stop", func(t *testing.T) {
		probe := &speculativeRequestProbe{tokens: []int{0}}
		p, prompt := speculativeRequestPlanner(t, probe)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		emitted := 0
		res, err := p.generateReusedRecovering(ctx, prompt, 4, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
			emitted++
			cancel()
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error = %v, want context.Canceled", err)
		}
		if res.stopped {
			t.Fatal("cancellation was reported as a model stop")
		}
		if res.gen != emitted || emitted == 0 {
			t.Fatalf("generated/emitted = %d/%d, want equal nonzero counts", res.gen, emitted)
		}
	})

	t.Run("proposal error falls back to ordinary target token", func(t *testing.T) {
		probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
		p, prompt := speculativeRequestPlanner(t, probe)
		var emitted []int
		res, err := p.generateReusedRecovering(context.Background(), prompt, 2, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			emitted = append(emitted, token)
			return false
		})
		if err != nil {
			t.Fatalf("proposal fallback: %v", err)
		}
		if probe.calls != 2 || res.gen != 2 || len(emitted) != 2 {
			t.Fatalf("calls/generated/emitted = %d/%d/%v, want 2/2/two tokens", probe.calls, res.gen, emitted)
		}
	})
}

func TestGreedySpeculativeCancellationStopsBeforeFurtherEmissionOrTargetStep(t *testing.T) {
	t.Run("accepted token cancellation suppresses remaining accepted and bonus", func(t *testing.T) {
		probe := &speculativeRequestProbe{}
		p, prompt, backend := speculativeBackendPlanner(t, probe)
		probe.tokens = speculativeAcceptedDraft(t, p.m, prompt, 3)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		emitted := make([]int, 0, 1)
		matMulsAtEmit := -1
		res, err := p.generateReusedRecovering(ctx, prompt, 4, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			emitted = append(emitted, token)
			matMulsAtEmit = backend.matMulCalls
			cancel()
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error=%v, want context.Canceled", err)
		}
		if res.gen != 1 || !reflect.DeepEqual(emitted, probe.tokens[:1]) {
			t.Fatalf("generated/emitted=%d/%v, want one accepted token %v", res.gen, emitted, probe.tokens[:1])
		}
		if backend.matMulCalls != matMulsAtEmit {
			t.Fatalf("target operations advanced after cancellation: matmuls=%d at_emit=%d", backend.matMulCalls, matMulsAtEmit)
		}
		stats := p.SpeculativeEngine().Stats()
		if stats.VerificationRounds != 1 || stats.BonusTokensEmitted != 0 {
			t.Fatalf("rounds/bonuses=%d/%d, want 1/0", stats.VerificationRounds, stats.BonusTokensEmitted)
		}
		assertSpeculativeRequestStateReleased(t, p, prompt, probe)
	})

	t.Run("fallback token cancellation suppresses target step", func(t *testing.T) {
		probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
		p, prompt, backend := speculativeBackendPlanner(t, probe)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		emitted := 0
		matMulsAtEmit := -1
		res, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
			emitted++
			matMulsAtEmit = backend.matMulCalls
			cancel()
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error=%v, want context.Canceled", err)
		}
		if res.gen != 1 || emitted != 1 {
			t.Fatalf("generated/emitted=%d/%d, want 1/1", res.gen, emitted)
		}
		if backend.matMulCalls != matMulsAtEmit {
			t.Fatalf("fallback target advanced after cancellation: matmuls=%d at_emit=%d", backend.matMulCalls, matMulsAtEmit)
		}
		if stats := p.SpeculativeEngine().Stats(); stats.VerificationRounds != 0 || stats.BonusTokensEmitted != 0 {
			t.Fatalf("fallback rounds/bonuses=%d/%d, want 0/0", stats.VerificationRounds, stats.BonusTokensEmitted)
		}
	})

	t.Run("emitted correction counts once but cancellation suppresses target step", func(t *testing.T) {
		probe := &speculativeRequestProbe{}
		p, prompt, backend := speculativeBackendPlanner(t, probe)
		correct := speculativeAcceptedDraft(t, p.m, prompt, 1)[0]
		probe.tokens = []int{(correct + 1) % p.m.Cfg.VocabSize}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		emitted := make([]int, 0, 1)
		matMulsAtEmit := -1
		res, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
			emitted = append(emitted, token)
			matMulsAtEmit = backend.matMulCalls
			cancel()
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation error=%v, want context.Canceled", err)
		}
		if res.gen != 1 || !reflect.DeepEqual(emitted, []int{correct}) {
			t.Fatalf("generated/emitted=%d/%v, want correction %d", res.gen, emitted, correct)
		}
		if backend.matMulCalls != matMulsAtEmit {
			t.Fatalf("correction target advanced after cancellation: matmuls=%d at_emit=%d", backend.matMulCalls, matMulsAtEmit)
		}
		stats := p.SpeculativeEngine().Stats()
		if stats.VerificationRounds != 1 || stats.BonusTokensEmitted != 1 {
			t.Fatalf("rounds/bonuses=%d/%d, want 1/1", stats.VerificationRounds, stats.BonusTokensEmitted)
		}
	})
}

func TestGreedySpeculativeCancellationBeforeFirstRoundEmission(t *testing.T) {
	t.Run("proposal failure observes cancellation before fallback", func(t *testing.T) {
		probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
		p, prompt, backend := speculativeBackendPlanner(t, probe)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		matMulsAtCancel := -1
		probe.after = func() {
			matMulsAtCancel = backend.matMulCalls
			cancel()
		}
		emitted := 0
		res, err := p.generateReusedRecovering(ctx, prompt, 2, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
			emitted++
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v, want context.Canceled", err)
		}
		if res.gen != 0 || emitted != 0 {
			t.Fatalf("generated/emitted=%d/%d, want 0/0", res.gen, emitted)
		}
		if backend.matMulCalls != matMulsAtCancel {
			t.Fatalf("target advanced after proposer cancellation: matmuls=%d at_cancel=%d", backend.matMulCalls, matMulsAtCancel)
		}
	})

	t.Run("verification completion observes cancellation before accepted or correction", func(t *testing.T) {
		probe := &speculativeRequestProbe{}
		p, prompt, backend := speculativeBackendPlanner(t, probe)
		probe.tokens = speculativeAcceptedDraft(t, p.m, prompt, 2)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		backend.cancel = cancel
		// Prompt prefill reads one logits row; serial verification reads one row
		// for each of the two draft tokens. Cancel on the latter's completion.
		backend.cancelOnRead = 1 + len(probe.tokens)
		emitted := 0
		res, err := p.generateReusedRecovering(ctx, prompt, 3, 0, 0, 0, nil, 0, 0, nil, func(int) bool {
			emitted++
			return false
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v, want context.Canceled", err)
		}
		if res.gen != 0 || emitted != 0 {
			t.Fatalf("generated/emitted=%d/%d, want 0/0 after verifier cancellation", res.gen, emitted)
		}
		if backend.readCalls != backend.cancelOnRead {
			t.Fatalf("logits reads=%d, want cancellation on verifier completion read %d", backend.readCalls, backend.cancelOnRead)
		}
	})
}

func TestSpeculativePrefixEligibilityLatchesOnlyAfterCompletedOuterTurn(t *testing.T) {
	tok := loadProbeTok(t)
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	p := NewInKernelPlanner(m, tok, "speculative-prefix-eligibility", false, nil, false)
	p.EnableSpeculativeDecoding(probe, 4)
	messages := []Message{{Role: RoleUser, Content: "repeat this prompt"}}
	promptTokens := encodedInKernelPromptTokens(t, tok, messages, cfg)

	if got := p.kvPrefixEligiblePromptTokens(promptTokens); got != 0 {
		t.Fatalf("first speculative turn eligible=%d, want 0 before any completed admission", got)
	}
	if _, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(1)); err != nil {
		t.Fatal(err)
	}
	if probe.calls == 0 {
		t.Fatal("first turn did not execute the speculative route")
	}
	if got := p.kvPrefixEligiblePromptTokens(promptTokens); got != promptTokens {
		t.Fatalf("second speculative turn eligible=%d, want prompt length %d after outer completion", got, promptTokens)
	}
	if _, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(1)); err != nil {
		t.Fatal(err)
	}
}

func TestGreedySpeculativeBonusStatsReflectActualEmission(t *testing.T) {
	t.Run("output budget suppresses bonus", func(t *testing.T) {
		probe := &speculativeRequestProbe{}
		p, prompt := speculativeRequestPlanner(t, probe)
		probe.tokens = speculativeAcceptedDraft(t, p.m, prompt, 3)
		res, err := p.generateReusedRecovering(context.Background(), prompt, 3, 0, 0, 0, nil, 0, 0, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		stats := p.SpeculativeEngine().Stats()
		if res.gen != 3 || stats.VerificationRounds != 1 || stats.BonusTokensEmitted != 0 {
			t.Fatalf("generated/rounds/bonuses=%d/%d/%d, want 3/1/0", res.gen, stats.VerificationRounds, stats.BonusTokensEmitted)
		}
	})

	t.Run("stop token suppresses correction", func(t *testing.T) {
		probe := &speculativeRequestProbe{}
		p, prompt := speculativeRequestPlanner(t, probe)
		correct := speculativeAcceptedDraft(t, p.m, prompt, 1)[0]
		probe.tokens = []int{(correct + 1) % p.m.Cfg.VocabSize}
		res, err := p.generateReusedRecovering(context.Background(), prompt, 3, 0, 0, 0, nil, 0, 0, map[int]bool{correct: true}, nil)
		if err != nil {
			t.Fatal(err)
		}
		stats := p.SpeculativeEngine().Stats()
		if !res.stopped || res.gen != 0 || stats.VerificationRounds != 1 || stats.BonusTokensEmitted != 0 {
			t.Fatalf("stopped/generated/rounds/bonuses=%v/%d/%d/%d, want true/0/1/0", res.stopped, res.gen, stats.VerificationRounds, stats.BonusTokensEmitted)
		}
	})
}

func TestGreedySpeculativeMeasurementMatchesEmittedTokens(t *testing.T) {
	probe := &speculativeRequestProbe{}
	p, prompt := speculativeRequestPlanner(t, probe)
	accepted := speculativeAcceptedDraft(t, p.m, prompt, 2)
	probe.tokens = []int{accepted[0], (accepted[1] + 1) % p.m.Cfg.VocabSize}

	base := time.Unix(1_000, 0)
	tick := 0
	measurement := &nativeInferenceMeasurement{
		startedAt:             time.Now().Add(-time.Second),
		decodeTokenIDsEnabled: true,
		traceNow: func() time.Time {
			tick++
			return base.Add(time.Duration(tick) * time.Nanosecond)
		},
	}
	var emitted []int
	res, err := p.generateReusedRecovering(context.Background(), prompt, 2, 0, 0, 0, nil, 0, 0, nil, func(token int) bool {
		emitted = append(emitted, token)
		return false
	}, measurement)
	if err != nil {
		t.Fatal(err)
	}
	if res.gen != len(emitted) || !reflect.DeepEqual(measurement.tokenIDs, emitted) || !reflect.DeepEqual(measurement.decodeTokenIDs, emitted) {
		t.Fatalf("generated/emitted/receipt_ids/trace_ids=%d/%v/%v/%v, want identical", res.gen, emitted, measurement.tokenIDs, measurement.decodeTokenIDs)
	}
	if len(measurement.logprobs) != len(emitted) || measurement.ttftS <= 0 {
		t.Fatalf("logprobs/ttft=%d/%g, want %d/positive", len(measurement.logprobs), measurement.ttftS, len(emitted))
	}
	for i, logprob := range measurement.logprobs {
		if math.IsNaN(logprob) || math.IsInf(logprob, 0) {
			t.Fatalf("logprob[%d]=%v is not finite", i, logprob)
		}
	}
	if len(measurement.traceEvents) != len(emitted) {
		t.Fatalf("trace events=%d, want emitted count %d", len(measurement.traceEvents), len(emitted))
	}
	for i, event := range measurement.traceEvents {
		if event.TokenIndex != i+1 || event.ElapsedNS <= 0 {
			t.Fatalf("trace[%d]=%+v, want one-based token index and positive elapsed time", i, event)
		}
	}
}

func TestSpeculativeOptInDoesNotBypassContextLimit(t *testing.T) {
	tok := loadProbeTok(t)
	messages := []Message{{Role: RoleUser, Content: "oversize"}}
	cfg := tinyConcurrencyConfig()
	promptTokens := encodedInKernelPromptTokens(t, tok, messages, cfg)
	cfg.MaxPositionEmbeddings = promptTokens

	probe := &speculativeRequestProbe{tokens: []int{0}}
	p := NewInKernelPlanner(&model.Model{Cfg: cfg}, tok, "speculative-context-limit", false, nil, false)
	p.EnableSpeculativeDecoding(probe, 4)
	comp, err := p.Complete(context.Background(), messages, nil, WithMaxTokens(1))
	if comp != nil {
		t.Fatalf("oversize speculative request returned completion: %+v", comp)
	}
	var contextErr *InKernelContextLengthError
	if !errors.As(err, &contextErr) {
		t.Fatalf("oversize error = %T (%v), want *InKernelContextLengthError", err, err)
	}
	if probe.calls != 0 {
		t.Fatalf("proposal generator ran %d times before context refusal", probe.calls)
	}
	want := &InKernelContextLengthError{PromptTokens: promptTokens, MaxNewTokens: 1, MaxContext: promptTokens}
	if !reflect.DeepEqual(contextErr, want) {
		t.Fatalf("context error = %+v, want %+v", contextErr, want)
	}
}
