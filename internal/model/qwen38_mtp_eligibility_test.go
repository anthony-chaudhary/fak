package model

import (
	"encoding/json"
	"testing"
)

func TestEvaluateQwen38MTPEligibility(t *testing.T) {
	eligible := Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true, MTPBackendReady: true, F32: true, Greedy: true,
		Depth: 3, FreshSession: true, MemoryHeadroomOK: true, OperatorEnabled: true,
	}
	tests := []struct {
		name   string
		mutate func(*Qwen38MTPEligibilityInput)
		reason Qwen38MTPDowngradeReason
	}{
		{"eligible", func(*Qwen38MTPEligibilityInput) {}, Qwen38MTPEligible},
		{"model or artifact", func(in *Qwen38MTPEligibilityInput) { in.Qwen38MTPArtifact = false }, Qwen38MTPModelUnsupported},
		{"backend", func(in *Qwen38MTPEligibilityInput) { in.MTPBackendReady = false }, Qwen38MTPBackendUnsupported},
		{"precision", func(in *Qwen38MTPEligibilityInput) { in.F32 = false }, Qwen38MTPPrecisionUnsupported},
		{"sampling", func(in *Qwen38MTPEligibilityInput) { in.Greedy = false }, Qwen38MTPSamplingUnsupported},
		{"depth", func(in *Qwen38MTPEligibilityInput) { in.Depth = Qwen35MTPMaxDraftDepth + 1 }, Qwen38MTPDepthUnsupported},
		{"session", func(in *Qwen38MTPEligibilityInput) { in.FreshSession = false }, Qwen38MTPSessionNotFresh},
		{"memory", func(in *Qwen38MTPEligibilityInput) { in.MemoryHeadroomOK = false }, Qwen38MTPMemoryUnsafe},
		{"policy", func(in *Qwen38MTPEligibilityInput) { in.OperatorEnabled = false }, Qwen38MTPDisabledByPolicy},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := eligible
			tt.mutate(&in)
			got := EvaluateQwen38MTPEligibility(in)
			if got.DowngradeReason != tt.reason {
				t.Fatalf("downgrade reason=%q want %q", got.DowngradeReason, tt.reason)
			}
			wantEngine := Qwen38EngineMTP
			wantEligible := true
			if tt.reason != Qwen38MTPEligible {
				wantEngine = Qwen38EngineTargetDecode
				wantEligible = false
			}
			if got.Engine != wantEngine || got.Eligible != wantEligible {
				t.Fatalf("receipt=%+v want engine=%q eligible=%v", got, wantEngine, wantEligible)
			}
			if got.RequestedDepth != in.Depth {
				t.Fatalf("requested_depth=%d want %d", got.RequestedDepth, in.Depth)
			}
			if got.Engine == "llama.cpp" {
				t.Fatal("eligibility silently selected llama.cpp")
			}

			receiptJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal receipt: %v", err)
			}
			var receiptFields map[string]any
			if err := json.Unmarshal(receiptJSON, &receiptFields); err != nil {
				t.Fatalf("unmarshal receipt: %v", err)
			}
			if receiptFields["engine"] != string(wantEngine) {
				t.Fatalf("receipt engine=%v want %q; json=%s", receiptFields["engine"], wantEngine, receiptJSON)
			}
			if tt.reason == Qwen38MTPEligible {
				if _, present := receiptFields["downgrade_reason"]; present {
					t.Fatalf("eligible receipt unexpectedly has downgrade_reason: %s", receiptJSON)
				}
			} else if receiptFields["downgrade_reason"] != string(tt.reason) {
				t.Fatalf("receipt downgrade_reason=%v want %q; json=%s", receiptFields["downgrade_reason"], tt.reason, receiptJSON)
			}
		})
	}
}

func TestEvaluateQwen38MTPEligibilityUsesActualRetainedTensorTypes(t *testing.T) {
	q4, ref := qwen38MTPQ4KTestModels(t)
	input := Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		Backend:           Qwen38MTPBackendMetal,
		Model:             q4,
		Greedy:            true,
		Depth:             3,
		FreshSession:      true,
		MemoryHeadroomOK:  true,
		OperatorEnabled:   true,
	}
	got := EvaluateQwen38MTPEligibility(input)
	if !got.Eligible ||
		got.Engine != Qwen38EngineMTP ||
		got.Backend != Qwen38MTPBackendMetal ||
		got.MTPTensorFormat != Qwen38MTPFormatQ4K ||
		got.RequestedDepth != 3 ||
		!got.TargetEquivalent {
		t.Fatalf("eligible Q4_K receipt=%+v", got)
	}

	// Artifact names are not evidence: one retained F32 projection beside the
	// Q4_K set makes the actual layout mixed and must downgrade explicitly.
	name := qwen38MTPMatrixTensors[0]
	q4.manifest[name] = ref.manifest[name]
	mixed := EvaluateQwen38MTPEligibility(input)
	if mixed.Eligible ||
		mixed.Engine != Qwen38EngineTargetDecode ||
		mixed.DowngradeReason != Qwen38MTPPrecisionUnsupported {
		t.Fatalf("mixed-layout receipt=%+v, want ordinary native precision downgrade", mixed)
	}
}
