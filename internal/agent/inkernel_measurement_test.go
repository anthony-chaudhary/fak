package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
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
		identity := qwen35MetalStateIdentityMeasurementFixture(999, model.Qwen35MetalStateAuthoritySequence)
		b.measurement.qwen35MetalStateIdentity = &identity
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
	if measurement.qwen35MetalStateIdentity != nil {
		t.Fatalf("retry retained partial Qwen Metal state identity: %+v", measurement.qwen35MetalStateIdentity)
	}
	for i := range emitted {
		if emitted[i] != measurement.tokenIDs[i] {
			t.Fatalf("retry emitted token[%d]=%d receipt=%d", i, emitted[i], measurement.tokenIDs[i])
		}
	}
}

func qwen35MetalStateIdentityMeasurementFixture(index int, authority string) model.Qwen35MetalStateIdentityReceipt {
	digest := func(offset int) string { return fmt.Sprintf("%064x", index*16+offset) }
	receipt := model.Qwen35MetalStateIdentityReceipt{
		Schema:              model.Qwen35MetalStateIdentitySchema,
		Available:           true,
		Authority:           authority,
		OwnerGeneration:     digest(1),
		Tokens:              32,
		TokenLineageSHA256:  digest(2),
		FullAttentionLayers: 1,
		GDNLayers:           1,
		StateCount:          5,
		States: []model.Qwen35MetalStateDigest{
			{Layer: 0, Role: model.Qwen35MetalStateRoleKRaw, Elements: 8 + index, SHA256: digest(3)},
			{Layer: 0, Role: model.Qwen35MetalStateRoleKPost, Elements: 8 + index, SHA256: digest(4)},
			{Layer: 0, Role: model.Qwen35MetalStateRoleV, Elements: 8 + index, SHA256: digest(5)},
			{Layer: 1, Role: model.Qwen35MetalStateRoleGDNConv, Elements: 12 + index, SHA256: digest(6)},
			{Layer: 1, Role: model.Qwen35MetalStateRoleGDNRecurrent, Elements: 16 + index, SHA256: digest(7)},
		},
		DigestOperations:  7,
		DigestInputBytes:  4096 + uint64(index),
		DigestNanoseconds: int64(1000 + index),
		BindingSHA256:     digest(8),
	}
	if authority == model.Qwen35MetalStateAuthoritySequence {
		receipt.GDNSnapshotOps = 1
		receipt.GDNSeedOps = 1
		receipt.GDNStateD2HBytes = uint64(112 + 8*index)
		receipt.GDNStateH2DBytes = uint64(112 + 8*index)
	}
	return receipt
}

func TestNativeInferenceReceiptCarriesRequestLocalQwen35MetalStateIdentity(t *testing.T) {
	p := &InKernelPlanner{modelID: "qwen35-metal-state-identity", metal: true, q4k: true}
	const requests = 32
	type result struct {
		index   int
		receipt *NativeInferenceReceipt
		want    model.Qwen35MetalStateIdentityReceipt
		source  *model.Qwen35MetalStateIdentityReceipt
		reset   *model.Qwen35MetalStateIdentityReceipt
	}
	results := make(chan result, requests)
	start := make(chan struct{})
	for i := 0; i < requests; i++ {
		go func(index int) {
			measurement := &nativeInferenceMeasurement{}
			authority := model.Qwen35MetalStateAuthorityControl
			if index%4 == 2 {
				authority = model.Qwen35MetalStateAuthoritySequence
			}
			identity := qwen35MetalStateIdentityMeasurementFixture(index, authority)
			if index%2 != 0 {
				// Poison an unavailable value so omission proves the Available gate.
				identity.Available = false
			}
			measurement.qwen35MetalStateIdentity = &identity
			<-start
			got := p.buildNativeInferenceReceipt(measurement, 0, 0)
			measurement.reset()
			results <- result{index: index, receipt: got, want: identity, source: &identity, reset: measurement.qwen35MetalStateIdentity}
		}(i)
	}
	close(start)
	for range requests {
		got := <-results
		if got.reset != nil {
			t.Fatalf("request %d reset retained Metal state identity: %+v", got.index, got.reset)
		}
		identity := got.receipt.Qwen35MetalStateIdentity
		if got.index%2 != 0 {
			if identity != nil {
				t.Fatalf("request %d received unavailable or foreign state identity: %+v", got.index, identity)
			}
			continue
		}
		if identity == nil || !reflect.DeepEqual(*identity, got.want) {
			t.Fatalf("request %d state identity = %+v, want exact request-local %+v", got.index, identity, got.want)
		}
		originalDigest := got.source.States[0].SHA256
		identity.States[0].SHA256 = "caller mutation"
		if got.source.States[0].SHA256 != originalDigest {
			t.Fatalf("request %d receipt aliases measurement state slice", got.index)
		}
	}
}

func TestQwen35MetalStateIdentityEnablementIsExactPublicFreshP32Metal(t *testing.T) {
	newPlanner := func() *InKernelPlanner {
		return &InKernelPlanner{
			m:     &model.Model{Cfg: model.Config{LayerTypes: []string{"linear_attention"}}},
			metal: true,
			q4k:   true,
		}
	}
	measurement := &nativeInferenceMeasurement{}
	ids := make([]int, 32)
	if !shouldEnableQwen35MetalStateIdentity(newPlanner(), measurement, ids, 0, nil) {
		t.Fatal("exact public fresh-P32 Metal request did not enable state identity")
	}

	for name, mutate := range map[string]func(*InKernelPlanner, **nativeInferenceMeasurement, *[]int, *int, *[]float32){
		"no-public-receipt": func(_ *InKernelPlanner, m **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) { *m = nil },
		"receipt-disabled": func(_ *InKernelPlanner, m **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) {
			(*m).inferenceDisabled = true
		},
		"cpu": func(p *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) {
			p.metal = false
		},
		"not-q4k": func(p *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) {
			p.q4k = false
		},
		"device-backend": func(p *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) {
			p.backend = compute.Default()
		},
		"non-hybrid": func(p *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, _ *int, _ *[]float32) {
			p.m = &model.Model{Cfg: model.Config{}}
		},
		"non-P32": func(_ *InKernelPlanner, _ **nativeInferenceMeasurement, prompt *[]int, _ *int, _ *[]float32) {
			*prompt = (*prompt)[:31]
		},
		"cached-prefix": func(_ *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, matched *int, _ *[]float32) {
			*matched = 1
		},
		"cached-logits": func(_ *InKernelPlanner, _ **nativeInferenceMeasurement, _ *[]int, _ *int, logits *[]float32) {
			*logits = []float32{1}
		},
	} {
		t.Run(name, func(t *testing.T) {
			planner := newPlanner()
			m := &nativeInferenceMeasurement{}
			prompt := append([]int(nil), ids...)
			matched := 0
			var logits []float32
			mutate(planner, &m, &prompt, &matched, &logits)
			if shouldEnableQwen35MetalStateIdentity(planner, m, prompt, matched, logits) {
				t.Fatalf("%s request acquired fresh-P32 state identity", name)
			}
		})
	}
}

type qwen35MetalStateIdentitySessionStub struct {
	finalized bool
	available bool
	receipt   model.Qwen35MetalStateIdentityReceipt
	err       error
}

func (s qwen35MetalStateIdentitySessionStub) FinalizeQwen35MetalStateIdentityReceipt() (bool, error) {
	return s.finalized, s.err
}

func (s qwen35MetalStateIdentitySessionStub) Qwen35MetalStateIdentityReceipt() (model.Qwen35MetalStateIdentityReceipt, bool) {
	return s.receipt, s.available
}

func TestFinalizeAndCaptureQwen35MetalStateIdentityBothArmsFailClosed(t *testing.T) {
	for _, authority := range []string{model.Qwen35MetalStateAuthorityControl, model.Qwen35MetalStateAuthoritySequence} {
		t.Run(authority, func(t *testing.T) {
			want := qwen35MetalStateIdentityMeasurementFixture(41, authority)
			measurement := &nativeInferenceMeasurement{}
			if err := finalizeAndCaptureQwen35MetalStateIdentity(qwen35MetalStateIdentitySessionStub{finalized: true, available: true, receipt: want}, measurement); err != nil {
				t.Fatal(err)
			}
			if measurement.qwen35MetalStateIdentity == nil || !reflect.DeepEqual(*measurement.qwen35MetalStateIdentity, want) {
				t.Fatalf("captured identity = %+v, want exact %s arm %+v", measurement.qwen35MetalStateIdentity, authority, want)
			}
			want.States[0].SHA256 = "source mutation"
			if measurement.qwen35MetalStateIdentity.States[0].SHA256 == "source mutation" {
				t.Fatal("common capture path retained a model receipt slice alias")
			}
		})
	}

	base := qwen35MetalStateIdentityMeasurementFixture(42, model.Qwen35MetalStateAuthorityControl)
	for name, stub := range map[string]qwen35MetalStateIdentitySessionStub{
		"not-finalized":       {available: true, receipt: base},
		"not-available":       {finalized: true, receipt: base},
		"unavailable-receipt": {finalized: true, available: true, receipt: func() model.Qwen35MetalStateIdentityReceipt { r := base; r.Available = false; return r }()},
		"finalize-error":      {err: errors.New("snapshot boundary failed")},
	} {
		t.Run(name, func(t *testing.T) {
			measurement := &nativeInferenceMeasurement{}
			if err := finalizeAndCaptureQwen35MetalStateIdentity(stub, measurement); err == nil {
				t.Fatalf("%s state identity published without a complete model receipt", name)
			}
			if measurement.qwen35MetalStateIdentity != nil {
				t.Fatalf("%s failure published state identity: %+v", name, measurement.qwen35MetalStateIdentity)
			}
		})
	}
}

func TestNativeInferenceReceiptCarriesRequestLocalQwen35MetalForwardSequence(t *testing.T) {
	p := &InKernelPlanner{modelID: "qwen35-metal-receipt", metal: true, q4k: true}
	const requests = 32
	type result struct {
		index   int
		receipt *NativeInferenceReceipt
		reset   model.Qwen35MetalForwardSequenceReceipt
	}
	results := make(chan result, requests)
	start := make(chan struct{})
	for i := 0; i < requests; i++ {
		go func(index int) {
			measurement := &nativeInferenceMeasurement{}
			uploadBytes := uint64(64<<10) + uint64(index)
			readbackBytes := uint64(16<<10) + uint64(3*index)
			if index%2 == 0 {
				measurement.qwen35MetalForwardSequence = model.Qwen35MetalForwardSequenceReceipt{
					Path:              model.Qwen35MetalGDNSequenceForwardPath,
					Available:         true,
					Tokens:            index + 1,
					CommandBuffers:    1,
					Encoders:          7,
					TerminalWaits:     1,
					TerminalReadbacks: 1,
					HostUploadBytes:   uploadBytes,
					HostReadbackBytes: readbackBytes,
					Committed:         true,
					CompletedWait:     true,
					TimingAvailable:   true,
					GPUMilliseconds:   2.5,
					WaitMilliseconds:  3.5,
				}
			} else {
				// Poison the unavailable value so omission proves the Available gate,
				// rather than passing because its transfer counters happened to be zero.
				measurement.qwen35MetalForwardSequence = model.Qwen35MetalForwardSequenceReceipt{
					HostUploadBytes:   uploadBytes,
					HostReadbackBytes: readbackBytes,
				}
			}
			<-start
			got := p.buildNativeInferenceReceipt(measurement, 0, 0)
			measurement.reset()
			results <- result{index: index, receipt: got, reset: measurement.qwen35MetalForwardSequence}
		}(i)
	}
	close(start)
	for range requests {
		got := <-results
		if got.reset != (model.Qwen35MetalForwardSequenceReceipt{}) {
			t.Fatalf("request %d reset retained Metal sequence evidence: %+v", got.index, got.reset)
		}
		sequence := got.receipt.Qwen35MetalForwardSequence
		if got.index%2 != 0 {
			if sequence != nil {
				t.Fatalf("request %d received another request's Metal sequence receipt: %+v", got.index, sequence)
			}
			continue
		}
		if sequence == nil || !sequence.Available || sequence.Tokens != got.index+1 {
			t.Fatalf("request %d sequence receipt = %+v, want available request-local T%d receipt", got.index, sequence, got.index+1)
		}
		wantUpload := uint64(64<<10) + uint64(got.index)
		wantReadback := uint64(16<<10) + uint64(3*got.index)
		if sequence.HostUploadBytes != wantUpload || sequence.HostReadbackBytes != wantReadback {
			t.Fatalf("request %d transfer bytes = %d/%d, want request-local %d/%d", got.index, sequence.HostUploadBytes, sequence.HostReadbackBytes, wantUpload, wantReadback)
		}
		if sequence.Path != model.Qwen35MetalGDNSequenceForwardPath || sequence.CommandBuffers != 1 || sequence.Encoders != 7 || sequence.TerminalWaits != 1 || sequence.TerminalReadbacks != 1 || !sequence.Committed || !sequence.CompletedWait || !sequence.TimingAvailable || sequence.GPUMilliseconds != 2.5 || sequence.WaitMilliseconds != 3.5 {
			t.Fatalf("request %d sequence receipt lost typed model witnesses: %+v", got.index, sequence)
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
