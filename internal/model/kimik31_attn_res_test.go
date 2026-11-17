package model

// kimik31_attn_res_test.go — parity + contract tests for the K3.1
// attention-residual bank (ticket 1611-kimi-k31, port leaf sglang-kimi-k31-2;
// device-port plan Stage 3). The reference semantics are the file's own eager
// transcription, aggregate_stream_torch in SGLang
// python/sglang/srt/layers/attn_residual.py at a7d9f81ee2; these tests pin the
// Go host forms against an independent float64 transcription of that function
// plus the write-cursor / layer-0 short-circuit contract.
//
// fak-test:runtime fast est=10ms lane=default

import (
	"math"
	"math/rand"
	"testing"
)

// naiveAttnResMix is an independent float64 transcription of
// aggregate_stream_torch: rows = bank[0..nvb) ++ prefix; normed = RMSNorm(rows);
// scores = normed . scoreProjW; probs = softmax(scores); mixed = sum probs*rows.
func naiveAttnResMix(bank [][]float64, prefix []float64, scoreProjW, scoreNormW []float64, eps float64) []float64 {
	H := len(prefix)
	rows := make([][]float64, 0, len(bank)+1)
	rows = append(rows, bank...)
	rows = append(rows, prefix)
	scores := make([]float64, len(rows))
	for r, row := range rows {
		var ss float64
		for _, v := range row {
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(H)+eps)
		var s float64
		for h := 0; h < H; h++ {
			s += row[h] * inv * scoreNormW[h] * scoreProjW[h]
		}
		scores[r] = s
	}
	m := scores[0]
	for _, v := range scores[1:] {
		if v > m {
			m = v
		}
	}
	var sum float64
	probs := make([]float64, len(scores))
	for i, v := range scores {
		probs[i] = math.Exp(v - m)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	out := make([]float64, H)
	for r, row := range rows {
		for h := 0; h < H; h++ {
			out[h] += probs[r] * row[h]
		}
	}
	return out
}

// TestAttnResScoreMatchesReference pins the host mixture against the float64
// transcription for a 2-token bank with 2 live snapshots.
func TestAttnResScoreMatchesReference(t *testing.T) {
	const H = 6
	rng := rand.New(rand.NewSource(5))
	pick := func() []float64 {
		out := make([]float64, H)
		for i := range out {
			out[i] = rng.Float64()*2 - 1
		}
		return out
	}
	scoreProjW := pick()
	scoreNormW := pick()
	_ = scoreNormW

	bank := NewKimiK31AttnResBank(2, H, 3)
	prefix := make([]float32, 2*H)
	for i := range prefix {
		prefix[i] = float32(rng.Float64()*2 - 1)
	}
	// Write two snapshots.
	for w := 0; w < 2; w++ {
		snap := make([]float32, 2*H)
		for i := range snap {
			snap[i] = float32(rng.Float64()*2 - 1)
		}
		bank.Write(snap)
	}

	got := KimiK31AttnResScore(prefix, bank, f32s(scoreProjW), f32s(scoreNormW), 1e-5)

	for tok := 0; tok < 2; tok++ {
		bankRows := [][]float64{
			f64s(bank.row(tok, 0)), f64s(bank.row(tok, 1)),
		}
		want := naiveAttnResMix(bankRows, f64s(prefix[tok*H:(tok+1)*H]), scoreProjW, scoreNormW, 1e-5)
		for h := 0; h < H; h++ {
			if math.Abs(float64(got[tok*H+h])-want[h]) > 1e-5 {
				t.Errorf("token %d dim %d: got %v want %v", tok, h, got[tok*H+h], want[h])
			}
		}
	}
}

// TestAttnResZeroBankIsIdentity pins the reference's nvb==0 early return: with
// nothing banked the mixture is the prefix unchanged.
func TestAttnResZeroBankIsIdentity(t *testing.T) {
	const H = 4
	bank := NewKimiK31AttnResBank(1, H, 3)
	prefix := []float32{1, -2, 3, -4}
	got := KimiK31AttnResScore(prefix, bank, make([]float32, H), make([]float32, H), 1e-5)
	for h := range prefix {
		if got[h] != prefix[h] {
			t.Fatalf("nvb=0 dim %d: got %v want the prefix %v", h, got[h], prefix[h])
		}
	}
}

// TestAttnResWriteAdvancesCursorAndBankRow pins the write contract: Write stores
// the pre-attention prefix into row nvb and advances the cursor.
func TestAttnResWriteAdvancesCursorAndBankRow(t *testing.T) {
	const H = 3
	bank := NewKimiK31AttnResBank(1, H, 2)
	if bank.NumValidBlocksFor() != 0 {
		t.Fatalf("fresh bank cursor = %d, want 0", bank.NumValidBlocksFor())
	}
	first := []float32{1, 2, 3}
	bank.Write(first)
	if bank.NumValidBlocksFor() != 1 {
		t.Fatalf("cursor after 1 write = %d, want 1", bank.NumValidBlocksFor())
	}
	for h, v := range first {
		if got := bank.row(0, 0)[h]; got != v {
			t.Fatalf("bank row 0 dim %d = %v, want %v", h, got, v)
		}
	}
	second := []float32{4, 5, 6}
	bank.Write(second)
	for h, v := range second {
		if got := bank.row(0, 1)[h]; got != v {
			t.Fatalf("bank row 1 dim %d = %v, want %v", h, got, v)
		}
	}
}

// TestAttnResAggregateAppliesOutNorm pins the normed aggregation point:
// aggregate = RMSNorm(mix, outNormW), and the nvb==0 short circuit is
// out_norm(prefix) (NOT the identity).
func TestAttnResAggregateAppliesOutNorm(t *testing.T) {
	const H = 4
	bank := NewKimiK31AttnResBank(1, H, 2)
	outNormW := []float32{1, 1, 1, 1}
	scoreProjW := []float32{1, 0, 0, 0}
	scoreNormW := []float32{1, 1, 1, 1}
	prefix := []float32{3, 3, 3, 3}

	got := KimiK31AttnResAggregate(prefix, bank, scoreProjW, scoreNormW, outNormW, 1e-5)
	// nvb=0 -> out_norm(prefix); with a constant vector of 3, rms = 3, so out = 1 each.
	for h := range got {
		if math.Abs(float64(got[h])-1.0) > 1e-5 {
			t.Fatalf("nvb=0 aggregate dim %d = %v, want 1 (out_norm of the constant prefix)", h, got[h])
		}
	}
}

// TestAttnResBankFullPanics proves the bank fails closed rather than silently
// dropping a snapshot.
func TestAttnResBankFullPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic when writing past the bank capacity")
		}
	}()
	bank := NewKimiK31AttnResBank(1, 2, 1)
	bank.Write([]float32{1, 2})
	bank.Write([]float32{3, 4}) // capacity is 1
}
