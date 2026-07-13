package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

// TestQuantSweepHeaderAccounting is the Lane F (#3074) quant-sweep golden: it drives the SAME
// glm_moe_dsa architecture through a set of quant "arms" — one per expert-tensor quant — and asserts
// the two header-derived table columns (resident bytes and active-bytes/token) come PURELY from the
// header, tracking the quant with no tensor read. It is the fixture proof behind
// docs/notes/GLM52-QUANT-SWEEP-HEADER-TABLE.md: the real GLM-5.2 arms (UD-Q4_K_M / UD-Q4_K_S /
// Q4_K-pure / Q3) differ only in their per-tensor quant recipe, so the accounting that fills those
// two columns is exactly EstimateLoadBytes (resident) + RoutedExpertActiveSet.ActiveBytesPerToken.
func TestQuantSweepHeaderAccounting(t *testing.T) {
	// 256-elem routed band (1 K-quant super-block, multiple of 32 for Q8_0) + a fixed Q8_0 non-expert
	// (embedding) tensor held high-precision across every arm, exactly as UD recipes keep embeddings.
	const E, I, H, K = 2, 2, 64, 2
	const nonElems = 256
	nonType := TensorQ8_0

	arms := []struct {
		name       string
		expertType TensorType
	}{
		{"f32-ref", TensorF32},
		{"q8_0", TensorQ8_0},
		{"q6_k", TensorQ6_K},
		{"q5_k", TensorQ5_K},
		{"q4_k", TensorQ4_K}, // the shipped GLM-5.2 UD-Q4_K_M expert quant
		{"q3_k", TensorQ3_K}, // the leaner Q3 arm
	}

	nonBytes := int64(mustPayloadBytes(t, "token_embd.weight", []uint64{nonElems}, nonType))
	var prevResident int64
	for i, arm := range arms {
		raw := glmQuantArmGGUF(t, E, I, H, K, arm.expertType, nonType, nonElems)
		p := filepath.Join(t.TempDir(), arm.name+".gguf")
		if err := os.WriteFile(p, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		ws, err := OpenWeights(p)
		if err != nil {
			t.Fatalf("[%s] OpenWeights: %v", arm.name, err)
		}

		// Column 1 — resident bytes, straight off EstimateLoadBytes (header-only).
		resident, err := ws.EstimateLoadBytes()
		if err != nil {
			ws.Close()
			t.Fatalf("[%s] EstimateLoadBytes: %v", arm.name, err)
		}
		expertBytes := int64(mustPayloadBytes(t, "blk.0.ffn_gate_exps.weight", []uint64{H, I, E}, arm.expertType))
		if want := expertBytes + nonBytes; resident != want {
			t.Errorf("[%s] resident=%d, want %d (expert %d + non-expert %d)", arm.name, resident, want, expertBytes, nonBytes)
		}

		// Column 2 — active-bytes/token, from the header active set.
		as, ok, err := ws.RoutedExpertActiveSet()
		ws.Close()
		if err != nil || !ok {
			t.Fatalf("[%s] RoutedExpertActiveSet ok=%v err=%v", arm.name, ok, err)
		}
		wantActive := (expertBytes/int64(E))*int64(K) + nonBytes
		if as.ActiveBytesPerToken != wantActive {
			t.Errorf("[%s] ActiveBytesPerToken=%d, want %d (K×per-expert + non-expert)", arm.name, as.ActiveBytesPerToken, wantActive)
		}
		// With K == E the routed band is fully active, so active-bytes/token == resident here.
		if as.ActiveBytesPerToken != resident {
			t.Errorf("[%s] ActiveBytesPerToken=%d != resident=%d (K==E fixture)", arm.name, as.ActiveBytesPerToken, resident)
		}

		// The sweep must shrink resident monotonically as the expert quant narrows.
		if i > 0 && resident >= prevResident {
			t.Errorf("[%s] resident=%d not < previous arm %d — quant did not shrink the header estimate", arm.name, resident, prevResident)
		}
		prevResident = resident
	}
}
