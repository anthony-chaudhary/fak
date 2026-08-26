package agent

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeInferenceMeasurementMatchesRawModelLogSoftmax(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	ids := synthIDs(cfg.VocabSize, 12, 9070)
	const generated = 4

	// Compute the oracle independently from the same raw Session logits. This is
	// the mathematical quantity the receipt promises, not decoded-text recovery.
	s := m.NewSession()
	defer s.Close()
	logits := s.Prefill(ids)
	wantIDs := make([]int, 0, generated)
	wantLP := make([]float64, 0, generated)
	for i := 0; i < generated; i++ {
		tok := 0
		for j := 1; j < len(logits); j++ {
			if logits[j] > logits[tok] {
				tok = j
			}
		}
		maxLogit := float64(logits[0])
		for _, raw := range logits[1:] {
			maxLogit = math.Max(maxLogit, float64(raw))
		}
		var denom float64
		for _, raw := range logits {
			denom += math.Exp(float64(raw) - maxLogit)
		}
		wantIDs = append(wantIDs, tok)
		wantLP = append(wantLP, float64(logits[tok])-maxLogit-math.Log(denom))
		if i != generated-1 {
			logits = s.Step(tok)
		}
	}

	p := &InKernelPlanner{m: m, modelID: "synthetic-receipt", quant: false}
	measurement := &nativeInferenceMeasurement{startedAt: time.Now()}
	gen, _, _, _, _, prefillS, decodeS, _, err := p.generateReusedContextWithBias(
		context.Background(), ids, generated, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil, measurement,
	)
	if err != nil {
		t.Fatal(err)
	}
	if gen != generated || len(measurement.tokenIDs) != gen || len(measurement.logprobs) != gen {
		t.Fatalf("generated/ids/logprobs = %d/%d/%d, want %d equal entries", gen, len(measurement.tokenIDs), len(measurement.logprobs), generated)
	}
	for i := range wantIDs {
		if measurement.tokenIDs[i] != wantIDs[i] {
			t.Fatalf("token[%d] = %d, want raw-logit argmax %d", i, measurement.tokenIDs[i], wantIDs[i])
		}
		if math.Abs(measurement.logprobs[i]-wantLP[i]) > 1e-12 || math.IsNaN(measurement.logprobs[i]) || math.IsInf(measurement.logprobs[i], 0) {
			t.Fatalf("logprob[%d] = %.17g, want finite raw-logit log_softmax %.17g", i, measurement.logprobs[i], wantLP[i])
		}
	}
	for name, seconds := range map[string]float64{"prefill": prefillS, "ttft": measurement.ttftS, "decode": decodeS} {
		if seconds < 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
			t.Fatalf("%s seconds = %v, want finite non-negative", name, seconds)
		}
	}
}

type receiptOOMRetryBackend struct {
	compute.Backend
	measurement *nativeInferenceMeasurement
	failed      bool
	recycled    int
	trimmed     int
	trimLarge   int
}

func (b *receiptOOMRetryBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if !b.failed && len(b.measurement.tokenIDs) == 1 {
		b.failed = true
		panic(&compute.DeviceAllocError{Bytes: 4096, Site: "receipt-test-after-first-token", Class: compute.MemoryScratchpad})
	}
	return b.Backend.MatMul(w, x)
}

func (b *receiptOOMRetryBackend) Recycle() { b.recycled++ }
func (b *receiptOOMRetryBackend) Trim()    { b.trimmed++ }
func (b *receiptOOMRetryBackend) TrimLarge(int) {
	b.trimLarge++
}

func TestNativeInferenceMeasurementOOMRetryDropsPartialAttempt(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	measurement := &nativeInferenceMeasurement{startedAt: time.Now()}
	be := &receiptOOMRetryBackend{Backend: compute.Default(), measurement: measurement}
	p := &InKernelPlanner{m: m, modelID: "synthetic-retry", backend: be}
	ids := synthIDs(cfg.VocabSize, 8, 9071)
	var emitted []int
	res, err := p.generateReusedWithOOMRetry(context.Background(), ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(tok int) bool {
		emitted = append(emitted, tok)
		return false
	}, func() {
		emitted = emitted[:0]
		measurement.reset()
	}, measurement)
	if err != nil {
		t.Fatal(err)
	}
	if !be.failed || be.recycled == 0 || be.trimmed == 0 || be.trimLarge != 1 {
		t.Fatalf("injected post-token OOM retry = failed %v recycled %d trimmed %d trimLarge %d, want true/positive/positive/1", be.failed, be.recycled, be.trimmed, be.trimLarge)
	}
	if res.gen != 3 || len(emitted) != 3 || len(measurement.tokenIDs) != 3 || len(measurement.logprobs) != 3 {
		t.Fatalf("retry retained partial attempt: gen/emitted/ids/logprobs = %d/%d/%d/%d, want 3/3/3/3", res.gen, len(emitted), len(measurement.tokenIDs), len(measurement.logprobs))
	}
	for i := range emitted {
		if emitted[i] != measurement.tokenIDs[i] {
			t.Fatalf("retry emitted token[%d]=%d receipt=%d", i, emitted[i], measurement.tokenIDs[i])
		}
	}
}

func TestNativeInferenceReceiptRejectsModifiedLogits(t *testing.T) {
	p := NewInKernelPlanner(model.NewSynthetic(tinyConcurrencyConfig()), loadProbeTok(t), "synthetic-strict", false, nil, false)
	positive := 0.5
	topK := 2
	cases := map[string][]SampleOpt{
		"temperature":       {WithTemperature(&positive)},
		"top-p":             {WithTopP(&positive)},
		"top-k":             {WithTopK(&topK)},
		"logit-bias":        {WithLogitBias(map[int]float64{1: 1})},
		"frequency-penalty": {WithFrequencyPenalty(&positive)},
		"presence-penalty":  {WithPresencePenalty(&positive)},
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			opts = append(opts, WithNativeInferenceReceipt(true))
			_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hello"}}, nil, opts...)
			var unsupported *NativeInferenceReceiptUnsupportedError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error = %T %v, want typed receipt refusal", err, err)
			}
		})
	}
}
