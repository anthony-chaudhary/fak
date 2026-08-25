package model

import (
	"strings"
	"testing"
)

func TestVerifySpeculatorVerifierReceiptRejectsFinalNormAliasDrift(t *testing.T) {
	manifest := []string{
		"model.llm.norm.weight",
		"norm.weight",                // unrelated shorter lookalike
		"vision.encoder.norm.weight", // unrelated longer lookalike
		"lm_head.weight",
	}
	receipt := SpeculatorVerifierReceipt{VerifierRoles: map[string]string{
		SpeculatorFinalNormRole: "norm.weight",
		SpeculatorLMHeadRole:    "lm_head.weight",
	}}

	proof, err := VerifySpeculatorVerifierReceipt(manifest, receipt)
	if err == nil {
		t.Fatal("wrong final-norm source produced an accepted speculative receipt")
	}
	if len(proof) != 1 || proof[0].Canonical != SpeculatorFinalNormRole || proof[0].ResolvedSource != "model.llm.norm.weight" {
		t.Fatalf("provenance = %+v, want canonical and resolved final-norm source", proof)
	}
	for _, want := range []string{SpeculatorFinalNormRole, "model.llm.norm.weight", "norm.weight", "candidate aliases"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not report %q", err, want)
		}
	}
}

func TestVerifySpeculatorVerifierReceiptManifestCases(t *testing.T) {
	tests := []struct {
		name     string
		manifest []string
		receipt  map[string]string
		wantErr  string
	}{
		{
			name:     "canonical",
			manifest: []string{"model.norm.weight", "lm_head.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.norm.weight", SpeculatorLMHeadRole: "lm_head.weight"},
		},
		{
			name:     "multimodal llm aliases",
			manifest: []string{"model.llm.norm.weight", "model.llm.lm_head.weight", "norm.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.llm.norm.weight", SpeculatorLMHeadRole: "model.llm.lm_head.weight"},
		},
		{
			name:     "missing final norm alias",
			manifest: []string{"vision.norm.weight", "lm_head.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.norm.weight"},
			wantErr:  "has no live source",
		},
		{
			name:     "tied output head",
			manifest: []string{"model.norm.weight", "model.embed_tokens.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.norm.weight", SpeculatorLMHeadRole: "model.embed_tokens.weight"},
		},
		{
			name:     "optional head omitted",
			manifest: []string{"model.norm.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.norm.weight"},
		},
		{
			name:     "native mtp role",
			manifest: []string{"model.norm.weight", "mtp.proj.weight"},
			receipt:  map[string]string{SpeculatorFinalNormRole: "model.norm.weight", "mtp.proj.weight": "mtp.proj.weight"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proof, err := VerifySpeculatorVerifierReceipt(tt.manifest, SpeculatorVerifierReceipt{VerifierRoles: tt.receipt})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(proof) != len(tt.receipt) {
				t.Fatalf("proof roles = %d, want %d: %+v", len(proof), len(tt.receipt), proof)
			}
		})
	}
}
