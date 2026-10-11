package agent

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

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
