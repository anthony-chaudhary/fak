package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRoutedExpertActiveSet_HeaderArithmetic builds a minimal glm_moe_dsa GGUF with a known
// batched routed-expert band and asserts RoutedExpertActiveSet derives the Lane F (#3074) active
// set — routed-expert resident band, per-expert bytes, and K×per-expert active bytes/token — from
// the header alone, reading no tensor payload. The fixture's one F32 [E,I,H] routed blob is
// E*I*H*4 bytes across E experts, and glmMoeDsaExpertGGUF sets expert_used_count (K) = 2.
func TestRoutedExpertActiveSet_HeaderArithmetic(t *testing.T) {
	const E, I, H = 4, 3, 2 // 4 experts, expert FFN len 3, hidden 2
	gate := make([]float32, E*I*H)
	for i := range gate {
		gate[i] = float32(i)
	}
	raw := glmMoeDsaExpertGGUF(E, I, H, gate)
	p := filepath.Join(t.TempDir(), "glm.gguf")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	as, ok, err := ws.RoutedExpertActiveSet()
	if err != nil || !ok {
		t.Fatalf("RoutedExpertActiveSet ok=%v err=%v (want ok=true)", ok, err)
	}
	wantBand := int64(E * I * H * 4) // one F32 batched routed tensor payload, all experts
	wantPer := wantBand / int64(E)
	if as.NumExperts != E {
		t.Errorf("NumExperts=%d, want %d", as.NumExperts, E)
	}
	if as.ExpertsUsed != 2 {
		t.Errorf("ExpertsUsed(K)=%d, want 2", as.ExpertsUsed)
	}
	if as.RoutedResident != wantBand {
		t.Errorf("RoutedResident=%d, want %d", as.RoutedResident, wantBand)
	}
	if as.PerExpert != wantPer {
		t.Errorf("PerExpert=%d, want %d", as.PerExpert, wantPer)
	}
	if as.ActivePerToken != wantPer*2 {
		t.Errorf("ActivePerToken=%d, want %d (K=2 × per-expert)", as.ActivePerToken, wantPer*2)
	}
}
