package h2hbench

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/timingspan"
)

func TestPrefillTPSUsesServerPrefillSpanNotClientTTFT(t *testing.T) {
	r := Row{PromptTokens: 100, CachedTokens: 50, TTFTMS: 2000, E2EMS: 9000, CompletionTokens: 40,
		Server: &ServerTimings{CacheN: 50, PromptN: 50, PromptMS: 500}}
	deriveRates(&r)
	if r.PrefillTPS != 100 {
		t.Fatalf("PrefillTPS = %v, want 100 (50 uncached tokens over 0.5 s server prefill)", r.PrefillTPS)
	}
	if r.PrefillSpan != "prefill_start..prefill_end" {
		t.Fatalf("PrefillSpan = %q", r.PrefillSpan)
	}
}

func TestPrefillTPSRefusesClientTTFTWhenNoPrefillClock(t *testing.T) {
	r := Row{PromptTokens: 100, CachedTokens: 0, TTFTMS: 2000, E2EMS: 9000, CompletionTokens: 40}
	deriveRates(&r)
	if r.PrefillUnmeasured != "no_server_prefill_clock" {
		t.Fatalf("PrefillUnmeasured = %q; the dropped rate needs a reason", r.PrefillUnmeasured)
	}
	if r.PrefillTPS != 0 || r.PrefillSpan != "" {
		t.Fatalf("PrefillTPS = %v span %q; client TTFT must not stand in for prefill", r.PrefillTPS, r.PrefillSpan)
	}
}

func TestDecodeTPSUsesServerDecodeSpanNotClientClock(t *testing.T) {
	r := Row{CompletionTokens: 40, TTFTMS: 2000, E2EMS: 9000,
		Server: &ServerTimings{PredictedN: 40, PredictedMS: 2000}}
	deriveRates(&r)
	if r.DecodeTPS != 19.5 {
		t.Fatalf("DecodeTPS = %v, want 19.5 (39 tokens over 2 s server decode)", r.DecodeTPS)
	}
	if r.DecodeSpan != "prefill_end..last_token" {
		t.Fatalf("DecodeSpan = %q", r.DecodeSpan)
	}
}

func TestDecodeTPSRefusesClientTTFTWhenNoServerClock(t *testing.T) {
	r := Row{CompletionTokens: 40, TTFTMS: 2000, E2EMS: 9000}
	deriveRates(&r)
	if r.DecodeTPS != 0 || r.DecodeSpan != "" {
		t.Fatalf("DecodeTPS = %v span %q; client TTFT must not stand in for decode", r.DecodeTPS, r.DecodeSpan)
	}
	if r.DecodeUnmeasured != "no_server_decode_clock" {
		t.Fatalf("DecodeUnmeasured = %q; the dropped rate needs a reason", r.DecodeUnmeasured)
	}
}

func TestCompareRefusesDecodeAcrossSpans(t *testing.T) {
	fak := Cell{Arm: "fak", Role: RoleFak, Scenario: ScenarioDecode, Size: 256, N: 1, DecodeP50: 50, DecodeSpan: "prefill_end..last_token"}
	rival := Cell{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioDecode, Size: 256, N: 1, DecodeP50: 20, DecodeSpan: "request_received..first_byte_sent"}
	losses, wins, refusals := compareCells([]Cell{fak, rival})
	if len(losses)+len(wins) != 0 || len(refusals) != 1 {
		t.Fatalf("losses=%v wins=%v refusals=%v; decode across spans must be refused", losses, wins, refusals)
	}
}

func TestRatioOfRefusesDifferentSpans(t *testing.T) {
	a := Metric{Value: 5000, Span: timingspan.PromptSpan}
	b := Metric{Value: 3000, Span: timingspan.FirstEmitSpan}
	_, err := RatioOf(a, b)
	if !errors.Is(err, timingspan.ErrSpanMismatch) {
		t.Fatalf("RatioOf across spans err = %v, want SPAN_MISMATCH", err)
	}
	got, err := RatioOf(a, Metric{Value: 2500, Span: timingspan.PromptSpan})
	if err != nil || got != 2 {
		t.Fatalf("RatioOf same span = %v, %v; want 2", got, err)
	}
}

func TestPrefillTPSCountsRowsWithoutServerClock(t *testing.T) {
	timed := Row{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioCold, Size: 100, OK: true, PromptTokens: 100,
		Server: &ServerTimings{PromptN: 100, PromptMS: 500}}
	untimed := Row{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioCold, Size: 100, OK: true, PromptTokens: 100, TTFTMS: 2000}
	deriveRates(&timed)
	deriveRates(&untimed)
	if untimed.PrefillUnmeasured != "no_server_prefill_clock" {
		t.Fatalf("PrefillUnmeasured = %q, want the no-clock reason", untimed.PrefillUnmeasured)
	}
	cells := Summarize([]Row{timed, untimed}).Cells
	if len(cells) != 1 {
		t.Fatalf("cells = %d, want 1", len(cells))
	}
	if cells[0].PrefillDropped != 1 || cells[0].PrefillSpan != "prefill_start..prefill_end" {
		t.Fatalf("cell dropped=%d span=%q; the unclocked row must be counted, not silently skipped", cells[0].PrefillDropped, cells[0].PrefillSpan)
	}
}

func TestCompareRefusesPrefillAcrossSpans(t *testing.T) {
	fak := Cell{Arm: "fak", Role: RoleFak, Scenario: ScenarioCold, Size: 100, N: 1, PrefillP50: 50, PrefillSpan: "prefill_start..prefill_end"}
	rival := Cell{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioCold, Size: 100, N: 1, PrefillP50: 200, PrefillSpan: "request_received..first_byte_sent"}
	losses, wins, refusals := compareCells([]Cell{fak, rival})
	for _, l := range append(losses, wins...) {
		if l.Metric == "prefill_tps_p50" {
			t.Fatalf("prefill compared across spans: %+v", l)
		}
	}
	if len(refusals) != 1 {
		t.Fatalf("refusals = %v, want one SPAN_MISMATCH refusal", refusals)
	}
}
