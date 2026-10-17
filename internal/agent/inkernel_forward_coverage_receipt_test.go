package agent

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime medium est=6s lane=default
// fak-test:usefulness protects=v41-exact-prefix-receipt-first-token-provenance
func TestV41PrefixHitReceiptPreservesForwardCoverage(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	ref := compute.Default()
	if ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("receipt witness requires the qualified CPU reference backend")
	}
	backend := &v41CompleteBackend{Backend: ref, available: true}
	m := v41CompleteModel(t)
	defer m.CloseWeights()
	p := v41CompletePlanner(t, m, backend, false)
	tree := radixkv.New(0)
	p.tree = tree
	prompt := []int{1, 3, 5}
	defer tree.EvictPrefix(prompt)

	run := func(ids []int, maxNew, wantMatched int) (*model.NativeInferenceReceipt, int) {
		t.Helper()
		measurement := &nativeInferenceMeasurement{startedAt: time.Now()}
		var emitted []int
		headsBefore := len(backend.headOutputs)
		gen, promptTok, cacheable, matched, tier, prefillS, decodeS, stopped, err := p.generateReusedContextWithBias(
			t.Context(), ids, maxNew, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				emitted = append(emitted, id)
				return false
			}, measurement,
		)
		if err != nil || stopped || gen != maxNew || matched != wantMatched {
			t.Fatalf("generated/stopped/matched/error = %d/%t/%d/%v, want %d/false/%d/nil", gen, stopped, matched, err, maxNew, wantMatched)
		}
		accounting := p.nativeCacheAccountingFor(nativeCacheAccountingFacts{
			promptTokens: promptTok, cacheableTokens: cacheable, matchedTokens: matched,
			sourceTier: tier, prefillSecs: prefillS, generated: gen,
		})
		receipt := p.buildNativeInferenceReceipt(measurement, prefillS, decodeS, accounting)
		if err := validateNativeInferenceReceiptForwardCoverage(m.Cfg, receipt); err != nil {
			t.Fatalf("prefix receipt matched=%d/%d generated=%d: %v", matched, promptTok, gen, err)
		}
		if err := accounting.Validate(); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(receipt.TokenIDs, emitted) || len(receipt.ForwardCoverage) != len(emitted) ||
			accounting.RestoredTokens != wantMatched || accounting.ComputedTokens != len(ids)-wantMatched {
			t.Fatalf("receipt changed token or prompt-work accounting: %+v", receipt)
		}
		return receipt, len(backend.headOutputs) - headsBefore
	}

	cold, _ := run(prompt, 64, 0)
	// Before the snapshot correction, this exact replay is rejected with the
	// observed 63 coverage items for 64 generated outputs.
	exact, heads := run(prompt, 64, len(prompt))
	if heads != 63 || exact.PrefillSeconds != 0 || !reflect.DeepEqual(exact.TokenIDs, cold.TokenIDs) ||
		!reflect.DeepEqual(exact.ForwardCoverage, cold.ForwardCoverage) {
		t.Fatalf("exact 64-token hit must use cached first logits and 63 new heads: heads=%d receipt=%+v", heads, exact)
	}
	if exact.ForwardCoverage[0].LogitsSource != model.NativeForwardLogitsSourceV41Full {
		t.Fatal("cached first-token logits lost their original prefill source")
	}
	for i, coverage := range exact.ForwardCoverage[1:] {
		if coverage.LogitsSource != model.NativeForwardLogitsSourceV41Incremental {
			t.Fatalf("output %d did not retain its actual incremental source", i+1)
		}
	}
	one, heads := run(prompt, 1, len(prompt))
	if heads != 0 || one.PrefillSeconds != 0 || !reflect.DeepEqual(one.TokenIDs, cold.TokenIDs[:1]) ||
		one.ForwardCoverage[0] != cold.ForwardCoverage[0] {
		t.Fatalf("exact one-token hit recomputed or lost cached-logit provenance: heads=%d receipt=%+v", heads, one)
	}

	// Diverge from the generated continuation so only the original prompt is a
	// hit. The suffix must produce fresh logits before its first output is sampled.
	suffix := append(append([]int(nil), prompt...), (cold.TokenIDs[0]+1)%m.Cfg.VocabSize)
	partial, heads := run(suffix, 1, len(prompt))
	if heads == 0 {
		t.Fatal("partial hit sampled ancestor logits without forwarding the suffix")
	}
	p.tree = nil
	partialCold, _ := run(suffix, 1, 0)
	if !reflect.DeepEqual(partial.TokenIDs, partialCold.TokenIDs) {
		t.Fatalf("partial-hit output %v differs from cold suffix control %v", partial.TokenIDs, partialCold.TokenIDs)
	}
}

type sequenceNativeForwardCoverageSource struct {
	coverage []model.NativeForwardCoverage
	calls    int
}

func (s *sequenceNativeForwardCoverageSource) NativeForwardCoverage() *model.NativeForwardCoverage {
	if s.calls >= len(s.coverage) {
		s.calls++
		return nil
	}
	coverage := s.coverage[s.calls]
	s.calls++
	return &coverage
}

// fak-test:runtime fast est=20ms lane=default
// fak-test:usefulness protects=native-receipt-token-bound-full-forward-coverage
func TestNativeInferenceReceiptBindsPerTokenForwardCoverage(t *testing.T) {
	cfg := model.Config{ModelType: "deepseek41", NumLayers: 40, HeadDim: 512}
	cold := model.NativeForwardCoverage{
		Schema:           model.NativeForwardCoverageSchemaV1,
		ModelType:        cfg.ModelType,
		ConfiguredLayers: cfg.NumLayers,
		HeadDim:          cfg.HeadDim,
		ExecutedLayers:   cfg.NumLayers,
		FullDepth:        true,
		Complete:         true,
		LogitsSource:     model.NativeForwardLogitsSourceV41Full,
	}
	incremental := cold
	incremental.LogitsSource = model.NativeForwardLogitsSourceV41Incremental
	coverageSource := &sequenceNativeForwardCoverageSource{
		coverage: []model.NativeForwardCoverage{cold, incremental},
	}
	measurement := &nativeInferenceMeasurement{}
	emitted := make([]int, 0, 2)
	lane := &decodeLane{
		logits:                []float32{0, 4, 1},
		stops:                 map[int]bool{},
		maxNew:                2,
		measurement:           measurement,
		forwardCoverageSource: coverageSource,
		emit: func(token int) bool {
			emitted = append(emitted, token)
			return false
		},
	}
	if next, advance := lane.decodeOne(t.Context()); next != 1 || !advance {
		t.Fatalf("first decode next/advance = %d/%t, want 1/true", next, advance)
	}
	lane.logits = []float32{0, 1, 5}
	if next, advance := lane.decodeOne(t.Context()); next != 0 || advance {
		t.Fatalf("terminal decode next/advance = %d/%t, want 0/false", next, advance)
	}
	if !reflect.DeepEqual(emitted, []int{1, 2}) || !reflect.DeepEqual(measurement.tokenIDs, emitted) ||
		!reflect.DeepEqual(measurement.forwardCoverage, []model.NativeForwardCoverage{cold, incremental}) || coverageSource.calls != 2 {
		t.Fatalf("decode token/coverage pairing = emitted %v ids %v coverage %+v calls %d", emitted, measurement.tokenIDs, measurement.forwardCoverage, coverageSource.calls)
	}
	planner := &InKernelPlanner{m: &model.Model{Cfg: cfg}, modelID: "deepseek-v4.1-flash"}
	receipt := planner.buildNativeInferenceReceipt(measurement, 0.1, 0)
	if err := validateNativeInferenceReceiptForwardCoverage(cfg, receipt); err != nil {
		t.Fatalf("valid cold-prefill coverage refused: %v", err)
	}
	if !reflect.DeepEqual(receipt.TokenIDs, []int{1, 2}) || !reflect.DeepEqual(receipt.ForwardCoverage, []model.NativeForwardCoverage{cold, incremental}) {
		t.Fatalf("receipt token/coverage projection = %v/%+v", receipt.TokenIDs, receipt.ForwardCoverage)
	}
	measurement.forwardCoverage[0].ExecutedLayers = 0
	if receipt.ForwardCoverage[0].ExecutedLayers != cfg.NumLayers {
		t.Fatal("receipt coverage aliases request-local measurement storage")
	}

	oneCoverageSource := &sequenceNativeForwardCoverageSource{coverage: []model.NativeForwardCoverage{cold, incremental}}
	oneMeasurement := &nativeInferenceMeasurement{}
	oneLane := &decodeLane{
		logits:                []float32{0, 4},
		stops:                 map[int]bool{},
		maxNew:                1,
		measurement:           oneMeasurement,
		forwardCoverageSource: oneCoverageSource,
	}
	if _, advance := oneLane.decodeOne(t.Context()); advance || len(oneMeasurement.tokenIDs) != 1 || len(oneMeasurement.forwardCoverage) != 1 || oneCoverageSource.calls != 1 {
		t.Fatalf("max_tokens=1 token/coverage/calls/advance = %d/%d/%d/%t, want 1/1/1/false", len(oneMeasurement.tokenIDs), len(oneMeasurement.forwardCoverage), oneCoverageSource.calls, advance)
	}

	stopCoverageSource := &sequenceNativeForwardCoverageSource{coverage: []model.NativeForwardCoverage{cold}}
	stopMeasurement := &nativeInferenceMeasurement{}
	stopLane := &decodeLane{
		logits:                []float32{0, 4},
		stops:                 map[int]bool{1: true},
		maxNew:                1,
		measurement:           stopMeasurement,
		forwardCoverageSource: stopCoverageSource,
	}
	if _, advance := stopLane.decodeOne(t.Context()); advance || !stopLane.stopped || len(stopMeasurement.tokenIDs) != 0 || len(stopMeasurement.forwardCoverage) != 0 || stopCoverageSource.calls != 0 {
		t.Fatalf("stop token leaked into receipt: advance=%t stopped=%t ids=%v coverage=%+v calls=%d", advance, stopLane.stopped, stopMeasurement.tokenIDs, stopMeasurement.forwardCoverage, stopCoverageSource.calls)
	}

	for _, tc := range []struct {
		name    string
		receipt model.NativeInferenceReceipt
	}{
		{
			name:    "missing coverage",
			receipt: model.NativeInferenceReceipt{TokenIDs: []int{52}},
		},
		{
			name: "incomplete traversal",
			receipt: model.NativeInferenceReceipt{
				TokenIDs:        []int{52},
				ForwardCoverage: []model.NativeForwardCoverage{func() model.NativeForwardCoverage { c := cold; c.Complete = false; return c }()},
			},
		},
		{
			name: "executed layer mismatch",
			receipt: model.NativeInferenceReceipt{
				TokenIDs:        []int{52},
				ForwardCoverage: []model.NativeForwardCoverage{func() model.NativeForwardCoverage { c := cold; c.ExecutedLayers--; c.FullDepth = false; return c }()},
			},
		},
		{
			name: "token cardinality mismatch",
			receipt: model.NativeInferenceReceipt{
				TokenIDs:        []int{52, 53},
				ForwardCoverage: []model.NativeForwardCoverage{cold},
			},
		},
		{
			name: "live model type mismatch",
			receipt: model.NativeInferenceReceipt{
				TokenIDs: []int{52},
				ForwardCoverage: []model.NativeForwardCoverage{func() model.NativeForwardCoverage {
					c := cold
					c.ModelType = "deepseek_v41"
					return c
				}()},
			},
		},
		{
			name: "live configured layers mismatch",
			receipt: model.NativeInferenceReceipt{
				TokenIDs: []int{52},
				ForwardCoverage: []model.NativeForwardCoverage{func() model.NativeForwardCoverage {
					c := cold
					c.ConfiguredLayers--
					c.ExecutedLayers--
					return c
				}()},
			},
		},
		{
			name: "live head dimension mismatch",
			receipt: model.NativeInferenceReceipt{
				TokenIDs: []int{52},
				ForwardCoverage: []model.NativeForwardCoverage{func() model.NativeForwardCoverage {
					c := cold
					c.HeadDim--
					return c
				}()},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateNativeInferenceReceiptForwardCoverage(cfg, &tc.receipt); !errors.Is(err, model.ErrNativeForwardCoverageInvalid) {
				t.Fatalf("validation error = %v, want %v", err, model.ErrNativeForwardCoverageInvalid)
			}
		})
	}

	legacy := &model.NativeInferenceReceipt{TokenIDs: []int{7}}
	if err := validateNativeInferenceReceiptForwardCoverage(model.Config{ModelType: "qwen2"}, legacy); err != nil {
		t.Fatalf("non-V4.1 receipt semantics changed: %v", err)
	}
}
