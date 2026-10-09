package harnesskit

import (
	"encoding/json"
	"errors"
	"testing"
)

func mustEnvelope(t *testing.T, served int) ContextEnvelope {
	t.Helper()
	env, err := DeriveContextEnvelope(ContextEnvelopeInput{ServedWindow: served, Source: WindowServed})
	if err != nil {
		t.Fatalf("DeriveContextEnvelope(%d): %v", served, err)
	}
	return env
}

func TestDeriveContextEnvelopeTable(t *testing.T) {
	cases := []struct {
		served int
		want   ContextEnvelope
	}{
		{32768, ContextEnvelope{ServedWindow: 32768, ContextWindow: 32768, OutputTokens: 4096, ReserveTokens: 7168, CompactTrigger: 25600, KeepRecentTokens: 4096, SummaryTokens: 4096, PostCompactTokens: 24576, ReclaimTokens: 1024, Viable: false, Reason: ReasonWindowTooSmallForHarness, Provenance: ContextEnvelopeProvenance}},
		{131072, ContextEnvelope{ServedWindow: 131072, ContextWindow: 131072, OutputTokens: 16384, ReserveTokens: 22528, CompactTrigger: 108544, KeepRecentTokens: 20000, SummaryTokens: 16384, PostCompactTokens: 52768, ReclaimTokens: 55776, Viable: true, Provenance: ContextEnvelopeProvenance}},
		{262144, ContextEnvelope{ServedWindow: 262144, ContextWindow: 163840, QualityCapped: true, OutputTokens: 20480, ReserveTokens: 27648, CompactTrigger: 136192, KeepRecentTokens: 20000, SummaryTokens: 20480, PostCompactTokens: 56864, ReclaimTokens: 79328, Viable: true, Provenance: ContextEnvelopeProvenance}},
		{1048576, ContextEnvelope{ServedWindow: 1048576, ContextWindow: 163840, QualityCapped: true, OutputTokens: 20480, ReserveTokens: 27648, CompactTrigger: 136192, KeepRecentTokens: 20000, SummaryTokens: 20480, PostCompactTokens: 56864, ReclaimTokens: 79328, Viable: true, Provenance: ContextEnvelopeProvenance}},
	}
	for _, tc := range cases {
		got := mustEnvelope(t, tc.served)
		if got != tc.want {
			t.Errorf("served %d:\n got  %+v\n want %+v", tc.served, got, tc.want)
		}
		if got.CompactTrigger+SummaryPromptOverheadTokens+got.SummaryTokens > got.ServedWindow {
			t.Errorf("served %d: summary request does not fit the served window", tc.served)
		}
	}
}

func TestDeriveContextEnvelopeRefusals(t *testing.T) {
	if _, err := DeriveContextEnvelope(ContextEnvelopeInput{ServedWindow: 65536, Source: WindowDerived}); !errors.Is(err, ErrDerivedWindow) {
		t.Fatalf("derived source: err = %v, want ErrDerivedWindow", err)
	}
	if _, err := DeriveContextEnvelope(ContextEnvelopeInput{ServedWindow: 65536}); !errors.Is(err, ErrDerivedWindow) {
		t.Fatalf("unset source: err = %v, want ErrDerivedWindow", err)
	}
	if _, err := DeriveContextEnvelope(ContextEnvelopeInput{ServedWindow: 0, Source: WindowServed}); !errors.Is(err, ErrNoServedWindow) {
		t.Fatalf("zero window: err = %v, want ErrNoServedWindow", err)
	}
}

// TestDeriveContextEnvelopeFixedPoint witnesses that the derivation cannot stack: feeding
// a result's ContextWindow back in as the served window never shrinks it again.
func TestDeriveContextEnvelopeFixedPoint(t *testing.T) {
	for _, served := range []int{131072, 1048576} {
		first := mustEnvelope(t, served)
		second := mustEnvelope(t, first.ContextWindow)
		if second.ContextWindow != first.ContextWindow {
			t.Fatalf("served %d: re-derivation moved ContextWindow %d -> %d", served, first.ContextWindow, second.ContextWindow)
		}
		if second.ReserveTokens != first.ReserveTokens || second.KeepRecentTokens != first.KeepRecentTokens {
			t.Fatalf("served %d: re-derivation moved compaction numbers", served)
		}
	}
}

func TestDeriveContextEnvelopeMonotone(t *testing.T) {
	prev := 0
	for _, w := range []int{16384, 32768, 65536, 131072, 163840, 262144, 1048576} {
		got := mustEnvelope(t, w).ContextWindow
		if got < prev {
			t.Fatalf("ContextWindow(%d) = %d < previous %d", w, got, prev)
		}
		if got > w || got > QualityCapTokens {
			t.Fatalf("ContextWindow(%d) = %d exceeds served window or quality cap", w, got)
		}
		prev = got
	}
}

func TestDeriveContextEnvelopeOverrides(t *testing.T) {
	env, err := DeriveContextEnvelope(ContextEnvelopeInput{ServedWindow: 131072, Source: WindowServed, MaxOutputTokens: 8192, FixedPromptTokens: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if env.OutputTokens != 8192 || env.SummaryTokens != 8192 || !env.Viable {
		t.Fatalf("override envelope = %+v", env)
	}
	if env.PostCompactTokens != 4096+env.SummaryTokens+env.KeepRecentTokens {
		t.Fatalf("PostCompactTokens ignores FixedPromptTokens override: %+v", env)
	}
}

func TestContextEnvelopeJSONShape(t *testing.T) {
	raw, err := json.Marshal(mustEnvelope(t, 32768))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"served_window", "context_window", "quality_capped", "output_tokens", "reserve_tokens", "compact_trigger", "keep_recent_tokens", "summary_tokens", "post_compact_tokens", "reclaim_tokens", "viable", "reason", "provenance"} {
		if _, ok := m[k]; !ok {
			t.Errorf("JSON missing %q: %s", k, raw)
		}
	}
	if m["reason"] != ReasonWindowTooSmallForHarness {
		t.Errorf("reason = %v", m["reason"])
	}
}
