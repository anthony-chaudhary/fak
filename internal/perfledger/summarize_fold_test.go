package perfledger

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeTimeWeightedThroughput(t *testing.T) {
	// Two identical served turns 900ms apart: each prefills 500 tokens in
	// 50ms (10000 tok/s) and decodes 100 tokens in 50ms (2000 tok/s). The
	// window spans [first start, last end] = 1100ms, of which 100ms is
	// prefill and 100ms is active decode.
	recs := []Record{
		NewRecord(time.UnixMilli(1000), "stop", "local", 500, 101, 0, 100*time.Millisecond, 50*time.Millisecond),
		NewRecord(time.UnixMilli(2000), "stop", "local", 500, 101, 0, 100*time.Millisecond, 50*time.Millisecond),
	}
	s := Summarize(recs)
	if s.PrefillTPSTimeWeighted != 909.09 {
		t.Errorf("prefill_tps_time_weighted = %v, want 909.09 (10000 tok/s held 100ms of a 1100ms window)", s.PrefillTPSTimeWeighted)
	}
	if s.DecodeTPSTimeWeighted != 181.82 {
		t.Errorf("decode_tps_time_weighted = %v, want 181.82 (2000 tok/s held 100ms of a 1100ms window)", s.DecodeTPSTimeWeighted)
	}
	if s.DecodeTPSActive != 2000 {
		t.Errorf("decode_tps_active = %v, want 2000 (idle gaps excluded)", s.DecodeTPSActive)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeTimeWeightedThroughputSingleRecord(t *testing.T) {
	// One turn: the window is its own [start, end] = 100ms, so the
	// time-weighted means are each phase rate held for half the window.
	rec := NewRecord(time.UnixMilli(1000), "stop", "local", 500, 101, 0, 100*time.Millisecond, 50*time.Millisecond)
	s := Summarize([]Record{rec})
	if s.PrefillTPSTimeWeighted != 5000 {
		t.Errorf("prefill_tps_time_weighted = %v, want 5000", s.PrefillTPSTimeWeighted)
	}
	if s.DecodeTPSTimeWeighted != 1000 {
		t.Errorf("decode_tps_time_weighted = %v, want 1000", s.DecodeTPSTimeWeighted)
	}
	if s.DecodeTPSActive != 2000 {
		t.Errorf("decode_tps_active = %v, want 2000", s.DecodeTPSActive)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeTimeWeightedThroughputDegenerate(t *testing.T) {
	// A row without a first-token boundary and a synthetic probe contribute
	// nothing: every time-weighted field stays absent.
	recs := []Record{
		NewRecord(time.UnixMilli(1000), "stop", "local", 500, 101, 0, 100*time.Millisecond, 0),
		{Schema: Schema, UnixMS: 2000, E2EMS: 100, TTFTMS: 50, PromptTokens: 500, CompletionTokens: 101, Synthetic: true},
	}
	s := Summarize(recs)
	if s.PrefillTPSTimeWeighted != 0 || s.DecodeTPSTimeWeighted != 0 || s.DecodeTPSActive != 0 {
		t.Errorf("degenerate window = prefill %v decode %v active %v, want all absent",
			s.PrefillTPSTimeWeighted, s.DecodeTPSTimeWeighted, s.DecodeTPSActive)
	}
}
