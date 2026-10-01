package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// LocalCollectiveEPReceipt describes a synthetic host operator check, not NCCL qualification.
type LocalCollectiveEPReceipt struct {
	Schema                   string  `json:"schema"`
	Issue                    int     `json:"issue"`
	Role                     string  `json:"role"`
	Engine                   string  `json:"engine"`
	Fixture                  string  `json:"fixture"`
	EvidenceScope            string  `json:"evidence_scope"`
	HardwareQualified        bool    `json:"hardware_qualified"`
	RanksTested              []int   `json:"ranks_tested"`
	CollectiveOperation      string  `json:"collective_operation"`
	DeltaCosineVsSingleRank  float64 `json:"delta_cosine_vs_single_rank"`
	MaxAbsoluteDelta         float64 `json:"max_absolute_delta"`
	SyntheticShardingChecked bool    `json:"synthetic_sharding_checked"`
}

// fak-test:runtime fast est=10ms lane=default
func TestLocalCollectiveExpertParallelDeltaReceipt(t *testing.T) {
	// Build synthetic 8-expert MoE model
	m := epGenMoeModel(8, 4)

	// Rank 1 reference (host expert delta)
	plan1, err := ExpertParallelPlan(8, 1)
	if err != nil {
		t.Fatalf("ExpertParallelPlan(8, 1) failed: %v", err)
	}
	picks := []routePick{
		{expert: 0, weight: 0.4},
		{expert: 2, weight: 0.3},
		{expert: 5, weight: 0.2},
		{expert: 7, weight: 0.1},
	}
	x := make([]float32, m.Cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%3-1) * 0.5
	}
	mat := f32Kernel{m}

	deltaSingle, err := m.ExpertParallelDelta(0, x, mat, picks, plan1, LocalCollective{})
	if err != nil {
		t.Fatalf("single-rank EP delta failed: %v", err)
	}

	// Host-only EP partial reduction (ranks = 2, 4, 8)
	ranksList := []int{2, 4, 8}
	var maxAbsDiff float64
	var minCosine float64 = 1.0

	for _, ranks := range ranksList {
		planN, err := ExpertParallelPlan(8, ranks)
		if err != nil {
			t.Fatalf("ExpertParallelPlan(8, %d) failed: %v", ranks, err)
		}
		deltaMulti, err := m.ExpertParallelDelta(0, x, mat, picks, planN, LocalCollective{})
		if err != nil {
			t.Fatalf("multi-rank EP delta (ranks=%d) failed: %v", ranks, err)
		}
		cos := epCosine(deltaSingle, deltaMulti)
		if cos < minCosine {
			minCosine = cos
		}
		diff := epMaxAbs(deltaSingle, deltaMulti)
		if diff > maxAbsDiff {
			maxAbsDiff = diff
		}
		if cos < 0.999999 {
			t.Fatalf("ranks=%d: cosine %g < 0.999999 vs single-rank", ranks, cos)
		}
	}

	// Emit witness receipt
	receipt := LocalCollectiveEPReceipt{
		Schema:                   "fak-local-ep-software-check/v2",
		Issue:                    10946,
		Role:                     "software-check",
		Engine:                   "fak-native",
		Fixture:                  "synthetic 8-expert MoE",
		EvidenceScope:            "host ExpertParallelDelta; no NCCL, GPU, model artifacts, or logits",
		HardwareQualified:        false,
		RanksTested:              ranksList,
		CollectiveOperation:      "LocalCollective.AllReduceSum (host float32)",
		DeltaCosineVsSingleRank:  minCosine,
		MaxAbsoluteDelta:         maxAbsDiff,
		SyntheticShardingChecked: true,
	}

	receiptDir := t.TempDir()
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	t.Logf("software receipt: %s", data)
	receiptPath := filepath.Join(receiptDir, "receipt.json")
	if err := os.WriteFile(receiptPath, data, 0644); err != nil {
		t.Fatalf("failed to write receipt: %v", err)
	}
}
