package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestFormatConfigLineEmitsActiveSetAxis witnesses that the -plan-only config line surfaces the
// full MoE active-set axis Lane F (#3074) needs — experts_used (K) and expert_ffn_len — not just
// the total expert count. Regression guard for the emit-K increment: the loader already reads
// these into cfg, so the only failure mode is the printer dropping them again.
func TestFormatConfigLineEmitsActiveSetAxis(t *testing.T) {
	cfg := model.Config{
		ModelType:           "glm_moe_dsa",
		NumLayers:           79,
		HiddenSize:          6144,
		NumExperts:          256,
		NumExpertsPerTok:    8,
		MoEIntermediateSize: 1536,
	}
	got := formatConfigLine(cfg)
	for _, want := range []string{
		"model_type=glm_moe_dsa",
		"layers=79",
		"hidden=6144",
		"experts=256",
		"experts_used=8",      // K — the highest-leverage unread scalar (Lane F)
		"expert_ffn_len=1536", // expert FFN length, read while the header is open
	} {
		if !strings.Contains(got, want) {
			t.Errorf("config line missing %q\n  got: %s", want, got)
		}
	}
}

// TestFormatActiveSetLineDivisors witnesses that the -plan-only active-set line emits BOTH roofline
// divisors Lane F (#3074) derives — active-bytes/token (K×per-expert + non-expert stream) and
// active-params/token — when K is known, and flags them PENDING(K) rather than guessing when it is
// not. The field names are the ceiling-doc / quant-sweep table's contract, so a rename must break
// this test.
func TestFormatActiveSetLineDivisors(t *testing.T) {
	known := formatActiveSetLine(ggufload.RoutedExpertActiveSet{
		NumExperts: 256, ExpertsUsed: 8,
		RoutedResident: 1 << 30, PerExpert: 1 << 20, NonExpertResident: 2 << 30,
		ActivePerToken: 4 << 20, ActiveBytesPerToken: 6 << 20, ActiveParamsPerToken: 42_000_000_000,
	})
	for _, want := range []string{"non_expert_resident=", "active_bytes_per_tok=", "active_params_per_tok=42.00B", "DERIVED"} {
		if !strings.Contains(known, want) {
			t.Errorf("K-known active-set line missing %q\n  got: %s", want, known)
		}
	}
	pending := formatActiveSetLine(ggufload.RoutedExpertActiveSet{
		NumExperts: 256, ExpertsUsed: 0, RoutedResident: 1 << 30, PerExpert: 1 << 20, NonExpertResident: 2 << 30,
	})
	for _, want := range []string{"active_bytes_per_tok=PENDING(K)", "active_params_per_tok=PENDING(K)"} {
		if !strings.Contains(pending, want) {
			t.Errorf("K-unread active-set line missing %q\n  got: %s", want, pending)
		}
	}
}

// TestArgmaxLocal witnesses the alloc-free argmax the -decode loop uses to pick the next
// token: it must return the index of the single max logit, ties resolving to the first.
func TestArgmaxLocal(t *testing.T) {
	cases := []struct {
		v    []float32
		want int
	}{
		{[]float32{0.1, 0.9, 0.3}, 1},
		{[]float32{-5, -1, -9}, 1},
		{[]float32{2, 2, 2}, 0},   // tie → first
		{[]float32{7}, 0},         // singleton
		{[]float32{-1, -1, 0}, 2}, // max at tail
	}
	for _, c := range cases {
		if got := argmaxLocal(c.v); got != c.want {
			t.Errorf("argmaxLocal(%v)=%d want %d", c.v, got, c.want)
		}
	}
}

// TestDecodeTokS witnesses the throughput math and its guards: steps÷wall, and a zero
// (never a NaN/Inf) for the degenerate steps≤0 / wall≤0 inputs the timer can hand it.
func TestDecodeTokS(t *testing.T) {
	if got := decodeTokS(64, 2*time.Second); got != 32 {
		t.Errorf("decodeTokS(64,2s)=%v want 32", got)
	}
	if got := decodeTokS(0, time.Second); got != 0 {
		t.Errorf("decodeTokS(0,1s)=%v want 0", got)
	}
	if got := decodeTokS(10, 0); got != 0 {
		t.Errorf("decodeTokS(10,0)=%v want 0 (no div-by-zero)", got)
	}
}

// TestFormatDecodeResultEchoesKnobs witnesses that the machine-parseable RESULT line the
// sweep runner greps carries the decoded tok/s, the first-token id (the C1 argmax witness),
// and the parse-stable field keys. A rename of these keys silently breaks the sweep parser,
// so this pins the contract.
func TestFormatDecodeResultEchoesKnobs(t *testing.T) {
	got := formatDecodeResult(64, 3, 2*time.Second, 248068, 0, 0, false)
	for _, want := range []string{
		"RESULT ",
		"decode_tok_s=32.0000",
		"first_token_id=248068",
		"steps=64",
		"warmup=3",
		"gomaxprocs=",
		"fak_workers=",
		"fak_kq_int8=",
		"fak_q4k=",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("decode RESULT line missing %q\n  got: %s", want, got)
		}
	}
}

func TestFormatDecodeResultIncludesFailClosedRoofline(t *testing.T) {
	got := formatDecodeResult(40, 2, 25*time.Second, 248068, 15_000_000_000, 90, true)
	for _, want := range []string{
		"decode_tok_s=1.6000",
		"bytes_per_token=15000000000",
		"stream_peak_gbps=90.0000",
		"achieved_gbps=24.0000",
		"decode_bw_util_%=26.67",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatDecodeResult() missing %q in %q", want, got)
		}
	}
}

func TestDecodeBandwidthRejectsZeroPeak(t *testing.T) {
	if achieved, util := decodeBandwidth(15_000_000_000, 1.6, 0); achieved != 0 || util != 0 {
		t.Fatalf("decodeBandwidth(zero peak)=(%v,%v), want (0,0)", achieved, util)
	}
}
