package main

import (
	"errors"
	"os"
	"os/exec"
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
	got := formatDecodeResult(64, 3, 2*time.Second, 248068, 0, 0, false, "interleave=applied(reason=eligible,nodes=0-7,regions=339)")
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
		`numa="interleave=applied(reason=eligible,nodes=0-7,regions=339)"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("decode RESULT line missing %q\n  got: %s", want, got)
		}
	}
}

func TestFormatDecodeResultIncludesFailClosedRoofline(t *testing.T) {
	got := formatDecodeResult(40, 2, 25*time.Second, 248068, 15_000_000_000, 90, true, "interleave=skipped(reason=single_node)")
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

// TestRequireRooflineFailsClosedBeforeLoad witnesses the process-level acceptance guard #4626
// calls out ("hard-fail if STREAM measures 0 rather than reporting a bogus 100% util"): with
// -require-roofline set but no positive -membw, main() must refuse with a non-zero exit and the
// roofline message BEFORE touching the model. TestDecodeBandwidthRejectsZeroPeak only pins the
// soft helper (returns 0,0); if the CLI guard regressed, decodeBandwidth would still emit a
// bogus "decode_bw_util_%=0.00 stream_peak_gbps=0.0000" row instead of failing closed. This
// re-execs the test binary so the real main() exit path is exercised, not a stubbed copy.
func TestRequireRooflineFailsClosedBeforeLoad(t *testing.T) {
	if os.Getenv("Q4KDIAG_FAILCLOSED_CHILD") == "1" {
		// -gguf is non-empty so the usage guard passes; the ONLY exit-2 path left with these
		// args is the roofline guard. A missing model file can never be reached.
		os.Args = []string{"q4kdiag", "-gguf", "/nonexistent/model.gguf", "-require-roofline"}
		main()
		return // unreachable: main() must os.Exit before here.
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestRequireRooflineFailsClosedBeforeLoad$", "-test.v")
	cmd.Env = append(os.Environ(), "Q4KDIAG_FAILCLOSED_CHILD=1")
	out, err := cmd.CombinedOutput()

	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		t.Fatalf("child did not exit non-zero; err=%v\noutput:\n%s", err, out)
	}
	if ee.ExitCode() != 2 {
		t.Fatalf("fail-closed exit code = %d, want 2\noutput:\n%s", ee.ExitCode(), out)
	}
	if !strings.Contains(string(out), "-require-roofline requires -membw > 0") {
		t.Fatalf("missing fail-closed reason in child stderr\noutput:\n%s", out)
	}
	// The bogus-row regression must NOT appear: a fail-closed refusal never prints a decode row.
	if strings.Contains(string(out), "decode_bw_util_%=") {
		t.Fatalf("guard leaked a decode_bw_util_%% row instead of failing closed\noutput:\n%s", out)
	}
}
