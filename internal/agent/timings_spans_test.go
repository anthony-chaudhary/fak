package agent

import (
	"encoding/json"
	"testing"
)

func spansOf(t *testing.T, tm *Timings) map[string]string {
	t.Helper()
	raw, err := json.Marshal(tm)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Spans map[string]string `json:"spans"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	return got.Spans
}

func TestNewTimingsDeclaresSpanOfEachClockField(t *testing.T) {
	spans := spansOf(t, NewTimings(100, 0, 10, 1, 2))
	if spans["prompt_ms"] != "prefill_start..prefill_end" {
		t.Fatalf("prompt_ms span = %q", spans["prompt_ms"])
	}
	if spans["predicted_ms"] != "prefill_end..last_token" {
		t.Fatalf("predicted_ms span = %q", spans["predicted_ms"])
	}
}

func TestNewTimingsOmitsSpanForAbsentClock(t *testing.T) {
	if spans := spansOf(t, NewTimings(100, 0, 0, 0, 0)); spans != nil {
		t.Fatalf("spans declared with no measured clock: %v", spans)
	}
}

func TestFirstEmitSpanDeclaredWhenMeasured(t *testing.T) {
	tm := NewTimings(100, 0, 10, 1, 2)
	tm.FirstEmitMS = 600
	spans := spansOf(t, tm)
	if spans["first_emit_ms"] != "request_received..first_byte_sent" {
		t.Fatalf("first_emit_ms span = %q", spans["first_emit_ms"])
	}
}

func TestInkling2LoopTimingsCarryFirstEmit(t *testing.T) {
	tm := inkling2LoopTimings(100, Inkling2LoopGeneration{Generated: 3, PrefillSeconds: 0.5, DecodeSeconds: 0.2, FirstEmitSeconds: 0.6})
	if tm.FirstEmitMS != 600 {
		t.Fatalf("FirstEmitMS = %v, want 600", tm.FirstEmitMS)
	}
	if spansOf(t, tm)["first_emit_ms"] == "" {
		t.Fatal("first_emit_ms measured but no span declared")
	}
}
