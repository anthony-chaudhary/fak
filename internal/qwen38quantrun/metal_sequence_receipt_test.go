package qwen38quantrun

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestValidateQwenMetalSequencePairAcceptsOnlyCoherentPair(t *testing.T) {
	control, candidate := validQwenMetalSequencePair()
	got := ValidateQwenMetalSequencePair(control, candidate)
	if got.Verdict != QwenMetalSequencePASS || len(got.Findings) != 0 {
		t.Fatalf("coherent pair = %+v, want PASS", got)
	}
}

func TestValidateQwenMetalSequencePairAdversarialMatrix(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*QwenMetalSequenceArm, *QwenMetalSequenceArm)
		arm    string
		reason QwenMetalSequenceHoldReason
	}{
		{"control receipt absent", func(control, _ *QwenMetalSequenceArm) { control.Receipt = nil }, QwenMetalSequenceControl, HoldQwenMetalReceiptMissing},
		{"control sequence unexpected", func(control, _ *QwenMetalSequenceArm) {
			control.Receipt.Qwen35MetalForwardSequence = validQwenMetalSequenceReceipt()
			control.Receipt.Qwen35MetalForwardSequence.SelectorState = model.Qwen35MetalSequenceSelectorOff
			control.Receipt.Qwen35MetalForwardSequence.EvidenceState = model.Qwen35MetalSequenceEvidenceNotSelected
		}, QwenMetalSequenceControl, HoldQwenMetalSequenceUnexpected},
		{"candidate receipt absent", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt = nil }, QwenMetalSequenceCandidate, HoldQwenMetalReceiptMissing},
		{"candidate sequence absent", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Qwen35MetalForwardSequence = nil }, QwenMetalSequenceCandidate, HoldQwenMetalSequenceMissing},
		{"sequence unavailable", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.Available = false
		}, QwenMetalSequenceCandidate, HoldQwenMetalSequenceUnavailable},
		{"wrong sequence path", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.Path = "metal/other"
		}, QwenMetalSequenceCandidate, HoldQwenMetalSequencePath},
		{"selector off", func(_, candidate *QwenMetalSequenceArm) { candidate.SelectorEnabled = false }, QwenMetalSequenceCandidate, HoldQwenMetalSelectorOff},
		{"selector path disagreement", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.ForwardPath = qwenMetalControlForwardPath }, QwenMetalSequenceCandidate, HoldQwenMetalForwardPath},
		{"selector receipt disagreement", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.SelectorState = model.Qwen35MetalSequenceSelectorOff
		}, QwenMetalSequenceCandidate, HoldQwenMetalSelectorReceipt},
		{"execution evidence disagreement", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.EvidenceState = model.Qwen35MetalSequenceEvidenceUnsupported
		}, QwenMetalSequenceCandidate, HoldQwenMetalEvidenceState},
		{"intermediate wait", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.IntermediateWaits = 1
		}, QwenMetalSequenceCandidate, HoldQwenMetalIntermediateWaits},
		{"intermediate readback", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.IntermediateReadbacks = 1
		}, QwenMetalSequenceCandidate, HoldQwenMetalIntermediateReads},
		{"zero tokens", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Qwen35MetalForwardSequence.Tokens = 0 }, QwenMetalSequenceCandidate, HoldQwenMetalSequenceTokens},
		{"negative tokens", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Qwen35MetalForwardSequence.Tokens = -1 }, QwenMetalSequenceCandidate, HoldQwenMetalSequenceTokens},
		{"zero command buffers", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.CommandBuffers = 0
		}, QwenMetalSequenceCandidate, HoldQwenMetalCommandBuffers},
		{"impossible command buffers", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.CommandBuffers = 2
		}, QwenMetalSequenceCandidate, HoldQwenMetalCommandBuffers},
		{"zero encoders", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Qwen35MetalForwardSequence.Encoders = 0 }, QwenMetalSequenceCandidate, HoldQwenMetalEncoders},
		{"negative encoders", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Qwen35MetalForwardSequence.Encoders = -1 }, QwenMetalSequenceCandidate, HoldQwenMetalEncoders},
		{"missing terminal wait", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.TerminalWaits = 0
		}, QwenMetalSequenceCandidate, HoldQwenMetalTerminalWaits},
		{"duplicate terminal wait", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.TerminalWaits = 2
		}, QwenMetalSequenceCandidate, HoldQwenMetalTerminalWaits},
		{"missing terminal readback", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.TerminalReadbacks = 0
		}, QwenMetalSequenceCandidate, HoldQwenMetalTerminalReadbacks},
		{"unexpected intermediate readback", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.TerminalReadbacks = 2
		}, QwenMetalSequenceCandidate, HoldQwenMetalTerminalReadbacks},
		{"uncommitted", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.Committed = false
		}, QwenMetalSequenceCandidate, HoldQwenMetalUncommitted},
		{"incomplete", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.CompletedWait = false
		}, QwenMetalSequenceCandidate, HoldQwenMetalIncomplete},
		{"missing upload bytes", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.HostUploadBytes = 0
		}, QwenMetalSequenceCandidate, HoldQwenMetalUploadBytes},
		{"wrong upload bytes", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.HostUploadBytes++
		}, QwenMetalSequenceCandidate, HoldQwenMetalUploadBytes},
		{"missing download bytes", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.HostReadbackBytes = 0
		}, QwenMetalSequenceCandidate, HoldQwenMetalReadbackBytes},
		{"wrong download bytes", func(_, candidate *QwenMetalSequenceArm) {
			candidate.Receipt.Qwen35MetalForwardSequence.HostReadbackBytes++
		}, QwenMetalSequenceCandidate, HoldQwenMetalReadbackBytes},
		{"fallback active", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.FallbackActive = true }, QwenMetalSequenceCandidate, HoldQwenMetalFallback},
		{"external runtime", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Engine = qwen38quant.EngineLlamaCpp }, QwenMetalSequenceCandidate, HoldQwenMetalEngine},
		{"wrong backend", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Backend = "cpu-ref" }, QwenMetalSequenceCandidate, HoldQwenMetalBackend},
		{"not q4k", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Q4K = false }, QwenMetalSequenceCandidate, HoldQwenMetalQuantization},
		{"receipt model mismatch", func(_, candidate *QwenMetalSequenceArm) { candidate.Receipt.Model = "other" }, QwenMetalSequenceCandidate, HoldQwenMetalArtifactModel},
		{"artifact incomplete", func(_, candidate *QwenMetalSequenceArm) { candidate.Artifact.ArtifactSHA256 = "" }, QwenMetalSequenceCandidate, HoldQwenMetalArtifactInvalid},
		{"artifact mismatch", func(_, candidate *QwenMetalSequenceArm) { candidate.Artifact.ArtifactSHA256 = strings.Repeat("f", 64) }, QwenMetalSequencePair, HoldQwenMetalArtifactMismatch},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			control, candidate := validQwenMetalSequencePair()
			test.mutate(&control, &candidate)
			got := ValidateQwenMetalSequencePair(control, candidate)
			want := []QwenMetalSequenceFinding{{Arm: test.arm, Reason: test.reason}}
			if got.Verdict != QwenMetalSequenceHOLD || !reflect.DeepEqual(got.Findings, want) {
				t.Fatalf("validation = %+v, want HOLD %+v", got, want)
			}
		})
	}
}

func TestValidateQwenMetalSequencePairReasonOrderIgnoresMutationOrder(t *testing.T) {
	mutations := []func(*QwenMetalSequenceArm){
		func(arm *QwenMetalSequenceArm) { arm.Receipt.Backend = "cpu-ref" },
		func(arm *QwenMetalSequenceArm) { arm.SelectorEnabled = false },
		func(arm *QwenMetalSequenceArm) { arm.Receipt.Qwen35MetalForwardSequence.CommandBuffers = 0 },
		func(arm *QwenMetalSequenceArm) { arm.Receipt.Qwen35MetalForwardSequence.CompletedWait = false },
	}
	validate := func(order []int) QwenMetalSequenceValidation {
		control, candidate := validQwenMetalSequencePair()
		for _, index := range order {
			mutations[index](&candidate)
		}
		return ValidateQwenMetalSequencePair(control, candidate)
	}
	forward := validate([]int{0, 1, 2, 3})
	reverse := validate([]int{3, 2, 1, 0})
	if !reflect.DeepEqual(forward, reverse) {
		t.Fatalf("finding order depends on mutation order: forward=%+v reverse=%+v", forward, reverse)
	}
	want := []QwenMetalSequenceFinding{
		{Arm: QwenMetalSequenceCandidate, Reason: HoldQwenMetalBackend},
		{Arm: QwenMetalSequenceCandidate, Reason: HoldQwenMetalSelectorOff},
		{Arm: QwenMetalSequenceCandidate, Reason: HoldQwenMetalCommandBuffers},
		{Arm: QwenMetalSequenceCandidate, Reason: HoldQwenMetalIncomplete},
	}
	if !reflect.DeepEqual(forward.Findings, want) {
		t.Fatalf("stable findings = %+v, want %+v", forward.Findings, want)
	}
}

func TestValidateQwenMetalSequencePairValidatesArmsBeforePairComparison(t *testing.T) {
	control, candidate := validQwenMetalSequencePair()
	control.Receipt = nil
	candidate.Artifact.ArtifactSHA256 = strings.Repeat("f", 64)
	got := ValidateQwenMetalSequencePair(control, candidate)
	want := []QwenMetalSequenceFinding{{Arm: QwenMetalSequenceControl, Reason: HoldQwenMetalReceiptMissing}}
	if !reflect.DeepEqual(got.Findings, want) {
		t.Fatalf("invalid arm reached pair comparison: %+v", got)
	}
}

func validQwenMetalSequencePair() (QwenMetalSequenceArm, QwenMetalSequenceArm) {
	hash := strings.Repeat("a", 64)
	artifact := qwen38quant.Identity{
		Model: "Qwen/Qwen3.8-27B-Q4_K_M", CheckpointSHA256: hash, ArtifactSHA256: hash,
		TokenizerSHA256: hash, TemplateSHA256: hash, QuantizerRevision: "q4-k-r1",
		RuntimeRevision: strings.Repeat("b", 40), FakModuleRev: "internal/model@r1+gabcdef0",
	}
	control := QwenMetalSequenceArm{
		Artifact: artifact,
		Receipt: &model.NativeInferenceReceipt{
			Model: artifact.Model, Engine: qwen38quant.EngineFakNative, Backend: qwenMetalBackend,
			ForwardPath: qwenMetalControlForwardPath, Q4K: true,
			Qwen35MetalForwardSequence: &model.Qwen35MetalForwardSequenceReceipt{
				SelectorState: model.Qwen35MetalSequenceSelectorOff,
				EvidenceState: model.Qwen35MetalSequenceEvidenceNotSelected,
			},
		},
	}
	candidate := QwenMetalSequenceArm{
		SelectorEnabled:           true,
		Artifact:                  artifact,
		ExpectedHostUploadBytes:   65536,
		ExpectedHostReadbackBytes: 16384,
		Receipt: &model.NativeInferenceReceipt{
			Model: artifact.Model, Engine: qwen38quant.EngineFakNative, Backend: qwenMetalBackend,
			ForwardPath: model.Qwen35MetalGDNSequenceForwardPath, Q4K: true,
			Qwen35MetalForwardSequence: validQwenMetalSequenceReceipt(),
		},
	}
	return control, candidate
}

func validQwenMetalSequenceReceipt() *model.Qwen35MetalForwardSequenceReceipt {
	return &model.Qwen35MetalForwardSequenceReceipt{
		Path: model.Qwen35MetalGDNSequenceForwardPath, Available: true, Tokens: 32,
		SelectorState: model.Qwen35MetalSequenceSelectorOn, EvidenceState: model.Qwen35MetalSequenceEvidenceExecuted,
		CommandBuffers: 1, Encoders: 7, TerminalWaits: 1, TerminalReadbacks: 1,
		HostUploadBytes: 65536, HostReadbackBytes: 16384, Committed: true, CompletedWait: true,
	}
}
