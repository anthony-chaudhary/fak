package main

import (
	"encoding/json"
	"sync"
	"testing"
	"time"

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
	op := wholeTokenOperation{CacheBefore: 32, CacheAfter: 33}
	op.Before.Tokens = 32
	op.After = model.Qwen35MetalForwardSequenceReceipt{Tokens: 1, Committed: true, CompletedWait: true, CommandBuffers: 1, TerminalWaits: 1, TerminalReadbacks: 1, Path: model.Qwen35MetalGDNSequenceForwardPath, EvidenceState: model.Qwen35MetalSequenceEvidenceExecuted}
	op.CountsAfter.BlockAcceptedCalls = 1
	report := wholeTokenReport{ExpectedRoute: "whole-token", Tokens: []int{7}, Operations: []wholeTokenOperation{op}}
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
	op.After = op.Before
	op.CountsAfter.ResidentGDNAcceptedCalls = 0
	op.CountsAfter.BlockAcceptedCalls = 2
	if validateWholeTokenOperation(op, "per-layer") == nil {
		t.Fatal("accepted two block calls for one Step")
	}
	op.CountsAfter.BlockAcceptedCalls = 1
	if err := validateWholeTokenOperation(op, "per-layer"); err != nil {
		t.Fatal(err)
	}
}
