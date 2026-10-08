package perfledger

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestNativeTimingSummaryPreservesMeasuredPhases(t *testing.T) {
	recs := []Record{
		{PromptTokens: 100, CompletionTokens: 4, CachedTokens: 2, NativeTiming: NewNativeTiming(10, 4, 10, 8)},
		{PromptTokens: 200, CompletionTokens: 6, CachedTokens: 3, NativeTiming: NewNativeTiming(20, 6, 20, 0)},
		{PromptTokens: 300, CompletionTokens: 8, CachedTokens: 4, NativeTiming: NewNativeTiming(0, 8, 0, 2)},
		{PromptTokens: 40, CompletionTokens: 10, CachedTokens: 5}, // proxy or an older row
	}
	// A persisted rate is derived data: replay must recompute it from the phase's
	// tokens and duration instead of trusting a stale or forged saved value.
	recs[0].NativeTiming.PrefillTPS = 999999
	recs[0].NativeTiming.DecodeTPS = 999999

	rep := BuildReport(recs, 4, false, 0)
	n := rep.Summary.NativeTiming
	if n == nil {
		t.Fatal("native timing summary is absent")
	}
	if n.Count != 3 || n.PrefillMeasured != 2 || n.DecodeMeasured != 2 {
		t.Fatalf("native timing coverage = %+v, want count=3 prefill=2 decode=2", n)
	}
	if n.PrefillP50MS != 10 || n.DecodeP50MS != 2 {
		t.Errorf("phase latency p50 = prefill %.3fms decode %.3fms, want 10ms/2ms", n.PrefillP50MS, n.DecodeP50MS)
	}
	if n.PrefillTPSP50 != 1000 || n.DecodeTPSP50 != 500 {
		t.Errorf("phase rate p50 = prefill %.2f decode %.2f, want 1000/500", n.PrefillTPSP50, n.DecodeTPSP50)
	}
	if n.PrefillTPSWeighted != 1000 || n.DecodeTPSWeighted != 1200 {
		t.Errorf("weighted phase rates = prefill %.2f decode %.2f, want 1000/1200", n.PrefillTPSWeighted, n.DecodeTPSWeighted)
	}

	line := RenderCompact(rep)
	for _, want := range []string{"native phases=3", "measured 2/3", "prefill p50=10ms/1000.0 tok/s", "decode p50=2ms/500.0 tok/s"} {
		if !strings.Contains(line, want) {
			t.Errorf("compact report %q does not contain %q", line, want)
		}
	}
	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"native_timing"`, `"prefill_measured":2`, `"decode_tps_weighted":1200`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("report JSON %s does not contain %s", raw, want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeTimingRejectsInvalidAndMissingMeasurements(t *testing.T) {
	for name, got := range map[string]*NativeTiming{
		"missing":  NewNativeTiming(9, 7, 0, 0),
		"negative": NewNativeTiming(9, 7, -1, -2),
		"nan":      NewNativeTiming(9, 7, math.NaN(), math.NaN()),
		"infinite": NewNativeTiming(9, 7, math.Inf(1), math.Inf(-1)),
	} {
		if got != nil {
			t.Errorf("%s timing = %+v, want nil", name, got)
		}
	}

	partial := NewNativeTiming(9, 7, 9, math.Inf(1))
	if partial == nil || partial.PrefillMS != 9 || partial.PrefillTokens != 9 || partial.PrefillTPS != 1000 {
		t.Fatalf("partial valid timing = %+v, want measured prefill", partial)
	}
	if partial.DecodeMS != 0 || partial.DecodeTokens != 0 || partial.DecodeTPS != 0 {
		t.Fatalf("invalid decode leaked into partial timing: %+v", partial)
	}

	s := Summarize([]Record{{PromptTokens: 3}, {PromptTokens: 9, NativeTiming: partial}})
	if s.NativeTiming == nil || s.NativeTiming.Count != 1 || s.NativeTiming.PrefillMeasured != 1 || s.NativeTiming.DecodeMeasured != 0 {
		t.Fatalf("missing/partial fold = %+v, want one native row with prefill only", s.NativeTiming)
	}
	raw, err := json.Marshal(Record{PromptTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "native_timing") {
		t.Fatalf("older/proxy row invented native timing: %s", raw)
	}
}
