package main

import (
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestWholeTokenWitnessUsesPromotedP1AndPreservesProfilerMode(t *testing.T) {
	if !metalgemm.Available() {
		t.Fatal("physical Metal device required")
	}
	cfg := model.Config{
		HiddenSize: 256, NumLayers: 4, NumHeads: 4, NumKVHeads: 2,
		HeadDim: 64, IntermediateSize: 256, VocabSize: 512,
		RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1,
		LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim: 3, LinearKeyHeadDim: 64, LinearNumKeyHeads: 2,
		LinearValueHeadDim: 64, LinearNumValueHeads: 4, AttnOutputGate: true,
		FullAttentionInterval: 4, NormGain1p: true, QKNorm: true,
	}
	m := model.NewSynthetic(cfg)
	var closeOnce sync.Once
	closeCalls := 0
	closeWeights := func() (err error) {
		closeOnce.Do(func() { closeCalls++; err = m.CloseWeights() })
		return err
	}
	t.Cleanup(func() {
		if err := closeWeights(); err != nil {
			t.Error(err)
		}
	})
	m.Quantize()
	newSession := func() *model.Session {
		s := m.NewSession()
		s.Q4K, s.Metal, s.MetalQ4K = true, true, true
		return s
	}
	// Existing profile behavior still opts into per-layer phase observation.
	profileSession := newSession()
	profiler, err := configureNativeProfileSession(profileSession, map[string]string{
		nativeProfileSequenceSelector:     nativeProfileSelectorOn,
		nativeProfileDecodeHandoffControl: model.Qwen35DecodeHandoffAuto.String(),
	})
	profileSession.Close()
	if err != nil || profiler == nil || profileSession.PhaseProfiler != profiler {
		t.Fatalf("existing profiler mode changed: profiler=%v err=%v", profiler, err)
	}
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}
	report, err := runWholeToken(newSession(), prompt, 2, "whole-token", time.Now(), closeWeights)
	if err != nil {
		t.Fatal(err)
	}
	if closeCalls != 1 || report.Prefill.Tokens != 32 || len(report.Decode) != 2 || len(report.Tokens) != 2 || len(report.GreedyNext) != 2 || report.Handoff.BlockAcceptedCalls != 2 {
		t.Fatalf("runner omitted lifecycle/execution proof: %+v closes=%d", report, closeCalls)
	}
	var totalPhases float64
	for _, phase := range []string{"load_setup", "session_setup", "prefill", "finalize", "decode_with_head", "host_selection_and_receipt_bookkeeping", "cleanup"} {
		if report.Phases[phase] <= 0 {
			t.Fatalf("missing measured %s phase", phase)
		}
		totalPhases += report.Phases[phase]
	}
	if report.TotalSeconds < totalPhases || report.PeakRSSBytes == 0 {
		t.Fatalf("incomplete total/memory: %+v", report)
	}
	for _, receipt := range report.Decode {
		if receipt.Tokens != 1 || !receipt.CompletedWait || receipt.IntermediateReadbacks != 0 {
			t.Fatalf("missing physical P1: %+v", receipt)
		}
	}
	t.Logf("device=%s phases=%v P1=%+v", report.Device, report.Phases, report.Decode)
}

func TestWholeTokenSerializedReceiptRejectsStaleAndTamperedEvidence(t *testing.T) {
	op := wholeTokenOperation{WholeSequenceOperation: model.WholeSequenceOperation{CacheBefore: 32, CacheAfter: 33}}
	op.Before.Tokens = 32
	// A real serialized report carries the runtime's concrete capability path, which
	// the report also records at build time; readback must pin to that recorded token.
	const sequencePath = "metal/qwen35-gdn-preprojected-sequence-v1"
	op.After = model.WholeSequenceReceipt{Tokens: 1, Committed: true, CompletedWait: true, CommandBuffers: 1, TerminalWaits: 1, TerminalReadbacks: 1, Path: sequencePath, EvidenceState: model.WholeSequenceEvidenceExecuted}
	op.CountsAfter.BlockAcceptedCalls = 1
	report := wholeTokenReport{ExpectedRoute: "whole-token", SequencePath: sequencePath, Tokens: []int{7}, Operations: []wholeTokenOperation{op}}
	report.BindingSHA256, _ = wholeTokenBinding(report)
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded wholeTokenReport
	if err = json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if err = validateWholeTokenReport(decoded); err != nil {
		t.Fatal(err)
	}
	decoded.Tokens[0]++
	if validateWholeTokenReport(decoded) == nil {
		t.Fatal("accepted tampered serialized report")
	}
	decoded.Operations[0].Before = decoded.Operations[0].After
	decoded.BindingSHA256, _ = wholeTokenBinding(decoded)
	if validateWholeTokenReport(decoded) == nil {
		t.Fatal("accepted stale receipt with a recomputed binding")
	}
	// A fabricated capability path must not revalidate even with a recomputed
	// binding: readback pins the receipt to the token the report recorded.
	fabricated := report
	fabricated.Operations = append([]wholeTokenOperation(nil), op)
	fabricated.Operations[0].After.Path = "fabricated/other-backend-v1"
	fabricated.Operations[0].Before.Path = "fabricated/other-backend-v1"
	fabricated.BindingSHA256, _ = wholeTokenBinding(fabricated)
	if validateWholeTokenReport(fabricated) == nil {
		t.Fatal("accepted a receipt with a fabricated capability path and recomputed binding")
	}
	op.After = op.Before
	op.CountsAfter.ResidentAcceptedCalls = 0
	op.CountsAfter.BlockAcceptedCalls = 2
	if validateWholeTokenOperation(op, "per-layer", nil, sequencePath) == nil {
		t.Fatal("accepted two block calls for one Step")
	}
	op.CountsAfter.BlockAcceptedCalls = 1
	if err := validateWholeTokenOperation(op, "per-layer", nil, sequencePath); err != nil {
		t.Fatal(err)
	}
}

// deviceIdentitySeamBackend is a fake compute.Backend whose Name/Tier are fixture
// values, so the whole-token device-identity seam can be exercised with no Metal
// device and no registration side effects.
type deviceIdentitySeamBackend struct {
	compute.Backend
	name, tier string
}

func (b deviceIdentitySeamBackend) Name() string { return b.name }
func (b deviceIdentitySeamBackend) Tier() string { return b.tier }

func TestWholeTokenDeviceIdentitySeam(t *testing.T) {
	fallback := metalgemm.DeviceName()

	if got := wholeTokenDeviceIdentity(&model.Session{
		Backend: deviceIdentitySeamBackend{Backend: compute.Default(), name: "fixture-dev", tier: "fixture-tier"},
	}); got != "fixture-dev/fixture-tier" {
		t.Fatalf("backend name/tier identity = %q, want %q", got, "fixture-dev/fixture-tier")
	}

	if got := wholeTokenDeviceIdentity(&model.Session{
		Backend: deviceIdentitySeamBackend{Backend: compute.Default(), name: "fixture-dev", tier: ""},
	}); got != "fixture-dev" {
		t.Fatalf("backend name-only identity = %q, want %q", got, "fixture-dev")
	}

	if got := wholeTokenDeviceIdentity(&model.Session{}); got != fallback {
		t.Fatalf("nil backend identity = %q, want legacy fallback %q", got, fallback)
	}

	if got := wholeTokenDeviceIdentity(nil); got != fallback {
		t.Fatalf("nil session identity = %q, want legacy fallback %q", got, fallback)
	}
}
