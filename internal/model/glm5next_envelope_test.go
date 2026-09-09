package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// GLM5NextOperatingEnvelopeReceipt schema for Issue #9438.
type GLM5NextOperatingEnvelopeReceipt struct {
	Schema                  string                 `json:"schema"`
	Issue                   int                    `json:"issue"`
	Role                    string                 `json:"role"`
	Engine                  string                 `json:"engine"`
	Model                   string                 `json:"model"`
	TotalParameters         int64                  `json:"total_parameters"`
	ActiveParameters        int64                  `json:"active_parameters"`
	LayersTotal             int                    `json:"layers_total"`
	KDARecurrentLayers      int                    `json:"kda_recurrent_layers"`
	DSASparseLayers         int                    `json:"dsa_sparse_layers"`
	FixedKDABallastBytes    int64                  `json:"fixed_kda_ballast_bytes"`
	EvaluatedStages         []GLM5NextContextStage `json:"evaluated_stages"`
	AdmissionChecksVerified bool                   `json:"admission_checks_verified"`
	ComparisonBaseline      string                 `json:"comparison_baseline"`
}

func TestGLM5NextOperatingEnvelope(t *testing.T) {
	env := DefaultGLM5NextOperatingEnvelope()
	if env.LayersTotal != 45 {
		t.Fatalf("expected 45 layers, got %d", env.LayersTotal)
	}
	if env.KDARecurrentLayers != 34 || env.DSASparseLayers != 11 {
		t.Fatalf("expected 34 KDA and 11 DSA layers, got %d and %d",
			env.KDARecurrentLayers, env.DSASparseLayers)
	}
	if len(env.Stages) != 5 {
		t.Fatalf("expected 5 staged context tiers, got %d", len(env.Stages))
	}

	// 1. Stage checks
	shortStage := env.Stages[0]
	if shortStage.ContextTokens != 4096 {
		t.Fatalf("stage 0 context tokens: %d, want 4096", shortStage.ContextTokens)
	}
	// Fixed KDA ballast is ~142.6 MB
	if shortStage.KDABallastBytes != 142606336 {
		t.Fatalf("unexpected KDA ballast bytes: %d", shortStage.KDABallastBytes)
	}
	// 1M stage check
	oneMillionStage := env.Stages[4]
	if oneMillionStage.ContextTokens != 1048576 {
		t.Fatalf("stage 4 context tokens: %d, want 1048576", oneMillionStage.ContextTokens)
	}
	// DSA KV at 1M: 11 * 512 * 2 * 1048576 = 11,811,160,064 bytes (~11.81 GB)
	if oneMillionStage.DSAKVBytes != 11811160064 {
		t.Fatalf("unexpected 1M DSA KV bytes: %d", oneMillionStage.DSAKVBytes)
	}
	// Total at 1M: ~18.2GB weights + 11.81GB KV + 0.14GB KDA = ~30.15 GB, feasible on 96GB/128GB
	if !oneMillionStage.FeasibleOn96GB || !oneMillionStage.FeasibleOn128GB {
		t.Fatalf("1M context with active experts should be feasible on 96GB/128GB")
	}

	// 2. Admission checks
	// On 24 GB VRAM: 4k fits, 1M exceeds
	vram24GB := int64(24 * 1024 * 1024 * 1024)
	if err := env.AdmitGLM5NextContext(4096, vram24GB); err != nil {
		t.Fatalf("expected 4k context to admit on 24GB: %v", err)
	}
	if err := env.AdmitGLM5NextContext(1048576, vram24GB); err == nil {
		t.Fatalf("expected 1M context to refuse on 24GB")
	}

	// 1M context admits on 96 GB
	vram96GB := int64(96 * 1024 * 1024 * 1024)
	if err := env.AdmitGLM5NextContext(1048576, vram96GB); err != nil {
		t.Fatalf("expected 1M context to admit on 96GB: %v", err)
	}

	// Beyond 1M refuses
	if err := env.AdmitGLM5NextContext(2000000, vram96GB); err == nil {
		t.Fatalf("expected >1M context to refuse")
	}

	// 3. Emit witness receipt
	receipt := GLM5NextOperatingEnvelopeReceipt{
		Schema:                  "fak-native-operating-envelope/v1",
		Issue:                   9438,
		Role:                    "candidate",
		Engine:                  "fak-native",
		Model:                   env.Model,
		TotalParameters:         env.TotalParameters,
		ActiveParameters:        env.ActiveParameters,
		LayersTotal:             env.LayersTotal,
		KDARecurrentLayers:      env.KDARecurrentLayers,
		DSASparseLayers:         env.DSASparseLayers,
		FixedKDABallastBytes:    shortStage.KDABallastBytes,
		EvaluatedStages:         env.Stages,
		AdmissionChecksVerified: true,
		ComparisonBaseline:      "SGLang / vLLM (explicitly selected and engine-labelled)",
	}

	receiptDir := filepath.Join("..", "..", "docs", "_witnesses", "issue-9438-glm53-operating-envelope")
	if err := os.MkdirAll(receiptDir, 0755); err != nil {
		t.Logf("note: could not create witness dir: %v", err)
		return
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	receiptPath := filepath.Join(receiptDir, "receipt.json")
	if err := os.WriteFile(receiptPath, data, 0644); err != nil {
		t.Fatalf("failed to write receipt: %v", err)
	}
}
