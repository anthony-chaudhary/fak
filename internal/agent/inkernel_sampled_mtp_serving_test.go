package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func sampledServingPlanner(t *testing.T, m *model.Model) *InKernelPlanner {
	t.Helper()
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	p := NewInKernelPlanner(m, nil, "synthetic-qwen38-sampled-mtp", false, nil, false)
	p.quant = false
	if err := p.EnableVulkanMTP(2); err != nil {
		t.Fatal(err)
	}
	return p
}

// This oracle runs the real native producer, target transaction and rejection
// planner directly. The request-path witness must reach the same decisions;
// neither side substitutes a scripted proposal or a copied sampler.
func sampledServingOracle(t *testing.T, m *model.Model, prompt []int, maxNew int, seed int64, temp, topP float64, topK int, minP float64) ([]int, []model.SpecRejectionPlan) {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	cfg := model.Qwen38SamplerConfig{Temperature: float32(temp), TopP: 1, Seed: uint64(seed)}
	d, err := model.NewQwen35MTPDraftSessionWithConfig(s, model.Qwen35MTPDraftConfig{Depth: 2, SamplerConfig: &cfg})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	logits := s.Prefill(prompt)
	committed := slices.Clone(prompt)
	key := model.SplitMix64(uint64(seed))
	if key == 0 {
		key = 1
	}
	var ws samplerWorkspace
	var out []int
	var plans []model.SpecRejectionPlan
	offset := 0
	for len(out) < maxNew {
		r, err := d.BeginSampledRoundWithTargetProbabilities(context.Background(), committed, logits, key, offset, func(row []float32) (model.SpecProbabilityRow, error) {
			return sampleLogitsWithPenaltyDistributionWS(row, temp, topP, topK, minP, nil, 0, 0, nil, &ws)
		})
		if err != nil {
			t.Fatalf("real oracle round: %v", err)
		}
		plan := r.Plan()
		plans = append(plans, plan)
		keep := min(len(plan.AcceptedTokens), maxNew-len(out))
		out = append(out, plan.AcceptedTokens[:keep]...)
		committed = append(committed, plan.AcceptedTokens[:keep]...)
		logits, err = r.Commit(keep)
		if err != nil {
			t.Fatal(err)
		}
		offset = plan.NextStep
		if len(out) == maxNew {
			break
		}
		out = append(out, plan.CorrectionToken)
		committed = append(committed, plan.CorrectionToken)
		logits = s.Step(plan.CorrectionToken)
	}
	return out, plans
}

func assertSampledServingResidentPrefix(t *testing.T, m *model.Model, s *model.Session, prefix []int) {
	t.Helper()
	if s.Cache.Len() != len(prefix) {
		t.Fatalf("resident prefix=%d, published prefix=%d", s.Cache.Len(), len(prefix))
	}
	serial := m.NewSession()
	defer serial.Close()
	serial.Prefill(prefix)
	// A following real step witnesses both attention and recurrent state, not
	// merely a cache-length counter. The hook runs immediately before Close.
	want, got := serial.Step(1), s.Step(1)
	if len(got) != len(want) {
		t.Fatalf("continuation width=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if math.IsNaN(float64(want[i])) || math.IsNaN(float64(got[i])) || math.IsInf(float64(want[i]), 0) || math.IsInf(float64(got[i]), 0) || math.Abs(float64(want[i]-got[i])) > 1e-5 {
			t.Fatalf("resident continuation logit %d=%g, serial=%g", i, got[i], want[i])
		}
	}
}

// fak-test:runtime medium est=1s lane=default
func TestInKernelSampledMTPServingStepsTerminalCorrection(t *testing.T) {
	m := model.NewSyntheticQwen38MTP()
	prompt := []int{3, 5, 11}
	var seed int64
	var correction int
	for candidate := int64(1); candidate <= 16; candidate++ {
		out, plans := sampledServingOracle(t, m, prompt, 1, candidate, 0.8, 0, 1, 0)
		if len(plans[0].AcceptedTokens) == 0 {
			seed, correction = candidate, out[0]
			break
		}
	}
	if seed == 0 {
		t.Fatal("fixture did not witness a correction-first round")
	}
	for _, mode := range []string{"emit-stop", "cancel", "trace-error"} {
		t.Run(mode, func(t *testing.T) {
			p := sampledServingPlanner(t, m)
			ctx, cancel := context.WithCancel(withNativeSeedlessSampleSeed(context.Background(), seed))
			defer cancel()
			var got []int
			closed := false
			p.specSessionCloseHook = func(s *model.Session) {
				closed = true
				assertSampledServingResidentPrefix(t, m, s, append(slices.Clone(prompt), got...))
			}
			var measurement *nativeInferenceMeasurement
			if mode == "trace-error" {
				now := time.Unix(10, 0)
				measurement = &nativeInferenceMeasurement{traceNow: func() time.Time {
					now = now.Add(-time.Second)
					return now
				}}
			}
			retried := false
			res, err := p.generateReusedWithOOMRetry(ctx, prompt, 5, 0.8, 0, 1, nil, 0, 0, nil, func(token int) bool {
				got = append(got, token)
				if mode == "cancel" {
					cancel()
				}
				return mode == "emit-stop"
			}, func() { retried = true }, measurement)
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel error=%v", err)
			}
			if mode == "trace-error" && (err == nil || !strings.Contains(err.Error(), "clock moved backwards")) {
				t.Fatalf("trace error=%v", err)
			}
			if mode == "emit-stop" && err != nil {
				t.Fatal(err)
			}
			if !closed || retried || !slices.Equal(got, []int{correction}) || res.vulkanMTP == nil || len(res.vulkanMTP.RoundDiagnostics) != 1 || !res.vulkanMTP.RoundDiagnostics[0].BonusEmitted {
				t.Fatalf("terminal correction state: got=%v closed=%v retried=%v receipt=%+v", got, closed, retried, res.vulkanMTP)
			}
		})
	}
}

type sampledServingRetryProbe struct {
	compute.Backend
	trimmedLarge int
}

func (b *sampledServingRetryProbe) TrimLarge(int) { b.trimmedLarge++ }

// fak-test:runtime medium est=1s lane=default
func TestInKernelSampledMTPServingRefusesTypedOOMReplay(t *testing.T) {
	// This is a serving retry-boundary software fault witness, not a kernel
	// allocation failure or hardware OOM measurement. Real CPU F32 MTP and
	// target work runs until the first correction is actually emitted.
	m := model.NewSyntheticQwen38MTP()
	prompt := []int{3, 5, 11}
	var seed int64
	var correction int
	for candidate := int64(1); candidate <= 16; candidate++ {
		out, plans := sampledServingOracle(t, m, prompt, 1, candidate, 0.8, 0, 1, 0)
		if len(plans[0].AcceptedTokens) == 0 {
			seed, correction = candidate, out[0]
			break
		}
	}
	if seed == 0 {
		t.Fatal("fixture did not witness a correction-first round")
	}
	p := sampledServingPlanner(t, m)
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend unavailable")
	}
	probe := &sampledServingRetryProbe{Backend: backend}
	closed := 0
	armed, injected, retried := false, false, false
	p.specSessionCloseHook = func(s *model.Session) {
		closed++
		if s.Backend != nil {
			t.Fatal("fault witness replaced the real CPU session backend")
		}
		// Install only the outer retry-boundary trim probe while retiring the
		// real session. It never supplies target logits or draft computation.
		p.backend = probe
	}
	const site = "sampled-serving-post-emission-retry-witness"
	measurement := &nativeInferenceMeasurement{traceNow: func() time.Time {
		if armed && !injected {
			injected = true
			panic(&compute.DeviceAllocError{Bytes: 1 << 20, Class: compute.MemoryScratchpad, Site: site})
		}
		return time.Unix(10, 0)
	}}
	var got []int
	res, err := p.generateReusedWithOOMRetry(withNativeSeedlessSampleSeed(context.Background(), seed), prompt, 1, 0.8, 0, 1, nil, 0, 0, nil, func(token int) bool {
		got = append(got, token)
		armed = true
		return false
	}, func() {
		// Negative control: without the production guard, idle pools could
		// trim and this callback would allow a second genuine CPU request.
		retried = true
		p.backend = nil
		armed = false
	}, measurement)
	var oom *InKernelOOMError
	if !errors.As(err, &oom) || oom.Bytes != 1<<20 || oom.Class != compute.MemoryScratchpad || oom.Site != site {
		t.Fatalf("typed allocation failure was replaced: %v", err)
	}
	if closed != 1 || !injected || retried || probe.trimmedLarge != 0 || len(p.InKernelOOMRetryStats().Rows) != 0 {
		t.Fatalf("sampled failure reached retry admission: closed=%v injected=%v retried=%v trim=%+v stats=%+v", closed, injected, retried, probe, p.InKernelOOMRetryStats())
	}
	if !slices.Equal(got, []int{correction}) || res.gen != 1 || res.vulkanMTP == nil || !res.vulkanMTP.Used || res.vulkanMTP.ProposalRounds != 1 || len(res.vulkanMTP.RoundDiagnostics) != 1 || !res.vulkanMTP.RoundDiagnostics[0].BonusEmitted {
		t.Fatalf("typed failure lost published prefix or receipt: tokens=%v gen=%d receipt=%+v", got, res.gen, res.vulkanMTP)
	}
	// Positive control: the same typed error really would enter pool trimming
	// without the sampled guard; a non-trimmable probe cannot pass vacuously.
	if !p.prepareDeviceOOMRetry(err) || probe.trimmedLarge != 1 {
		t.Fatal("retry-boundary probe cannot witness the guard's absence")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestInKernelSampledMTPServingUsesCanonicalTarget(t *testing.T) {
	for _, tc := range []struct {
		name string
		topP float64
		topK int
		minP float64
	}{
		{"full", 0, 0, 0},
		{"nucleus", 0.73, 0, 0},
		{"minp-topk-nucleus", 0.8, 4, 0.15},
		{"correction-heavy", 0, 1, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := model.NewSyntheticQwen38MTP()
			p := sampledServingPlanner(t, m)
			prompt := []int{3, 5, 11}
			const seed, maxNew = int64(37), 9
			want, plans := sampledServingOracle(t, m, prompt, maxNew, seed, 0.8, tc.topP, tc.topK, tc.minP)
			var got []int
			closed := false
			p.specSessionCloseHook = func(s *model.Session) {
				closed = true
				assertSampledServingResidentPrefix(t, m, s, append(slices.Clone(prompt), got...))
			}
			ctx := withNativeMinP(withNativeSeedlessSampleSeed(context.Background(), seed), &tc.minP)
			measurement := &nativeInferenceMeasurement{decodeTokenIDsEnabled: true}
			res, err := p.generateReusedRecovering(ctx, prompt, maxNew, 0.8, tc.topP, tc.topK, nil, 0, 0, nil, func(token int) bool {
				got = append(got, token)
				return false
			}, measurement)
			if err != nil || !closed || !slices.Equal(got, want) {
				t.Fatalf("actual request: tokens=%v want=%v closed=%v err=%v", got, want, closed, err)
			}
			if !slices.Equal(measurement.tokenIDs, got) || !slices.Equal(measurement.decodeTokenIDs, got) {
				t.Fatalf("measurement lost actual emitted tokens: %+v", measurement.tokenIDs)
			}
			if res.vulkanMTP == nil || !res.vulkanMTP.Used || res.vulkanMTP.ProposalRounds != len(plans) || res.vulkanMTP.DowngradeReason != "" {
				t.Fatalf("real sampled path was not reached: %+v", res.vulkanMTP)
			}
			if res.specCost == nil || res.specCost.Completed.Attempts != len(plans) || res.vulkanMTP.ClosedRoundTotals == nil {
				t.Fatalf("sampled cost omitted real rounds: %+v", res.specCost)
			}
			for _, diagnostic := range res.vulkanMTP.RoundDiagnostics {
				if !diagnostic.RoundClosed || diagnostic.Target == nil || diagnostic.Target.Engine != "fak-native" || diagnostic.DraftElapsed < 0 || diagnostic.InclusiveElapsed < diagnostic.DraftElapsed {
					t.Fatalf("invalid committed native receipt: %+v", diagnostic)
				}
			}
			if tc.topK == 1 && res.vulkanMTP.RejectedTokens == 0 {
				t.Fatal("fixture failed to exercise correction and reserved NextStep coordinates")
			}
			// The same request-local seed starts a fresh draft stream even when
			// the planner's prior request recorded acceptance/rejection history.
			p.specSessionCloseHook = nil
			var replay []int
			_, err = p.generateReusedRecovering(ctx, prompt, maxNew, 0.8, tc.topP, tc.topK, nil, 0, 0, nil, func(token int) bool {
				replay = append(replay, token)
				return false
			})
			if err != nil || !slices.Equal(replay, got) {
				t.Fatalf("request-owned replay=%v first=%v err=%v", replay, got, err)
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestInKernelSampledMTPServingCommitsOnlyEmittedPrefix(t *testing.T) {
	m := model.NewSyntheticQwen38MTP()
	prompt := []int{3, 5, 11}
	// Find a witnessed first round with an accepted suffix that the emitter
	// will not publish. This is real MTP sampling, not a manufactured proposal.
	var seed int64
	var first []int
	for candidate := int64(1); candidate <= 16; candidate++ {
		out, plans := sampledServingOracle(t, m, prompt, 2, candidate, 10, 0, 0, 0)
		if len(plans[0].AcceptedTokens) == 2 {
			seed, first = candidate, out
			break
		}
	}
	if seed == 0 {
		t.Fatal("fixture did not witness a two-token accepted prefix")
	}
	for _, mode := range []string{"emit-stop", "cancel-after-emission", "max-new", "token-stop-before-publication"} {
		t.Run(mode, func(t *testing.T) {
			p := sampledServingPlanner(t, m)
			ctx, cancel := context.WithCancel(withNativeSeedlessSampleSeed(context.Background(), seed))
			defer cancel()
			var got []int
			closed := false
			p.specSessionCloseHook = func(s *model.Session) {
				closed = true
				assertSampledServingResidentPrefix(t, m, s, append(slices.Clone(prompt), got...))
			}
			maxNew := 5
			if mode == "max-new" {
				maxNew = 1
			}
			var stops map[int]bool
			if mode == "token-stop-before-publication" {
				stops = map[int]bool{first[0]: true}
			}
			res, err := p.generateReusedRecovering(ctx, prompt, maxNew, 10, 0, 0, nil, 0, 0, stops, func(token int) bool {
				got = append(got, token)
				if mode == "cancel-after-emission" {
					cancel()
				}
				return mode == "emit-stop"
			})
			if mode == "cancel-after-emission" {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel error=%v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			want := first[:1]
			if mode == "token-stop-before-publication" {
				want = nil
				if len(res.vulkanMTP.RoundDiagnostics) != 0 {
					t.Fatal("aborted round published a successful target receipt")
				}
			} else if len(res.vulkanMTP.RoundDiagnostics) != 1 || res.vulkanMTP.RoundDiagnostics[0].Target.AcceptedTokens != 1 {
				t.Fatalf("receipt did not describe the actual retained prefix: %+v", res.vulkanMTP)
			}
			if !closed || !slices.Equal(got, want) {
				t.Fatalf("published=%v want=%v close=%v", got, want, closed)
			}
		})
	}
}

// sampledServingFaultModel clones read-only exported weight views before changing
// one named tensor. All target and MTP work still uses the real F32 implementation.
func sampledServingFaultModel(t *testing.T, fault string) *model.Model {
	t.Helper()
	source := model.NewSyntheticQwen38MTP()
	names := []string{"model.embed_tokens.weight", "model.norm.weight", "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"}
	common := []string{"input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
	attention := []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"}
	linear := []string{"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight", "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm.weight", "linear_attn.out_proj.weight"}
	for layer, kind := range source.Cfg.LayerTypes {
		for _, suffix := range common {
			names = append(names, fmt.Sprintf("model.layers.%d.%s", layer, suffix))
		}
		selected := attention
		if kind == "linear_attention" {
			selected = linear
		}
		for _, suffix := range selected {
			names = append(names, fmt.Sprintf("model.layers.%d.%s", layer, suffix))
		}
	}
	for _, suffix := range append(append(slices.Clone(common), attention...), "self_attn.q_norm.weight", "self_attn.k_norm.weight") {
		names = append(names, "mtp.layers.0."+suffix)
	}
	var tensors []model.NamedTensorF32
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("CPU reference backend unavailable")
	}
	for _, name := range names {
		weight, err := source.WeightSource(backend).Weight(name, compute.F32)
		if err != nil {
			t.Fatal(err)
		}
		data, ok := weight.Backend().Host(weight)
		if !ok {
			t.Fatal("synthetic F32 weight is not host-resident")
		}
		data = slices.Clone(data)
		if name == fault {
			for i := range data {
				data[i] = float32(math.NaN())
			}
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: weight.Shape, Data: data})
	}
	original := model.RetainMTP
	model.RetainMTP = true
	t.Cleanup(func() { model.RetainMTP = original })
	m, err := model.NewFromF32Tensors(source.Cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// fak-test:runtime medium est=1s lane=default
func TestInKernelSampledMTPServingFallbackAndFatalTarget(t *testing.T) {
	prompt := []int{3, 5, 11}
	t.Run("unsupported-first-draft-row", func(t *testing.T) {
		m := sampledServingFaultModel(t, "mtp.fc.weight")
		p := sampledServingPlanner(t, m)
		ctx := withNativeSeedlessSampleSeed(context.Background(), 37)
		var got, want []int
		res, err := p.generateReusedRecovering(ctx, prompt, 3, 0.8, 0.9, 0, nil, 0, 0, nil, func(token int) bool { got = append(got, token); return false })
		if err != nil || res.vulkanMTP == nil || res.vulkanMTP.Used || res.vulkanMTP.DowngradeReason != vulkanMTPDowngradeEmptyProposal {
			t.Fatalf("safe fallback result=%+v err=%v", res.vulkanMTP, err)
		}
		p.DisableSpeculativeDecoding()
		_, err = p.generateReusedRecovering(ctx, prompt, 3, 0.8, 0.9, 0, nil, 0, 0, nil, func(token int) bool { want = append(want, token); return false })
		if err != nil || !slices.Equal(got, want) {
			t.Fatalf("first-row refusal changed ordinary RNG: got=%v want=%v err=%v", got, want, err)
		}
	})
	t.Run("invalid-target-is-not-fallback", func(t *testing.T) {
		m := sampledServingFaultModel(t, "model.norm.weight")
		p := sampledServingPlanner(t, m)
		emitted := false
		res, err := p.generateReusedRecovering(withNativeSeedlessSampleSeed(context.Background(), 37), prompt, 3, 0.8, 0, 0, nil, 0, 0, nil, func(int) bool { emitted = true; return false })
		if err == nil || emitted || res.vulkanMTP == nil || res.vulkanMTP.DowngradeReason == vulkanMTPDowngradeEmptyProposal {
			t.Fatalf("invalid target restarted serially: emitted=%v receipt=%+v err=%v", emitted, res.vulkanMTP, err)
		}
	})
}

// fak-test:runtime medium est=1s lane=default
func TestInKernelSampledMTPServingAdmissionFences(t *testing.T) {
	for _, tc := range []struct {
		name           string
		seed           bool
		bias           model.LogitBias
		freq, presence float64
	}{
		{name: "explicit-seed", seed: true},
		{name: "bias", bias: model.LogitBias{1: 0.2}},
		{name: "frequency", freq: 0.1},
		{name: "presence", presence: 0.1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := sampledServingPlanner(t, model.NewSyntheticQwen38MTP())
			ctx := withNativeSeedlessSampleSeed(context.Background(), 37)
			if tc.seed {
				seed := int64(7)
				ctx = withNativeTargetSampleSeed(ctx, &seed)
			}
			p.specSessionCloseHook = func(*model.Session) { t.Fatal("fenced request entered speculative serving") }
			res, err := p.generateReusedRecovering(ctx, []int{3, 5, 11}, 2, 0.8, 0.9, 0, tc.bias, tc.freq, tc.presence, nil, nil)
			if err != nil || res.vulkanMTP != nil {
				t.Fatalf("target-only fence receipt=%+v err=%v", res.vulkanMTP, err)
			}
		})
	}
	p := sampledServingPlanner(t, model.NewSyntheticQwen38MTP())
	if p.sampledSpeculativeRequestEligible(0, nil, 0, 0) || !p.greedySpeculativeRequestEligible(0, nil, 0, 0) {
		t.Fatal("greedy route changed")
	}
	for _, temp := range []float64{math.NaN(), math.Inf(1), math.SmallestNonzeroFloat64, math.MaxFloat64} {
		if p.sampledSpeculativeRequestEligible(temp, nil, 0, 0) {
			t.Fatalf("unsupported draft temperature %g admitted", temp)
		}
	}
}
