package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func TestRunVCacheContextJoinAttributesResetVsProviderMiss(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:20Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500}`,
		// a reset lands at 00:00:25Z; the next turn re-warms the whole prefix.
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:30Z","input_tokens":100,"cache_creation_input_tokens":42000}`,
		// a second family with no lifecycle event anywhere nearby its own spike.
		`{"session_id":"beta","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":30000}`,
		`{"session_id":"beta","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_read_input_tokens":30000,"cache_creation_input_tokens":400}`,
		`{"session_id":"beta","captured_utc":"2026-06-26T00:00:20Z","input_tokens":50,"cache_read_input_tokens":30000,"cache_creation_input_tokens":400}`,
		`{"session_id":"beta","captured_utc":"2026-06-26T00:20:00Z","input_tokens":100,"cache_creation_input_tokens":31000}`,
	)
	events := writeLines(t, "events.jsonl",
		`{"kind":"context_reset","family":"alpha","unix_millis":1782432025000,"outcome":"reset","detail":"hidden restart re-entered cold"}`,
	)

	var out, errb bytes.Buffer
	code := runVCacheContextJoin(&out, &errb, []string{"--telemetry", tel, "--events", events, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	var rep vcacheobserve.JoinReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if rep.Schema != vcacheobserve.JoinSchema {
		t.Fatalf("schema: got %q want %q", rep.Schema, vcacheobserve.JoinSchema)
	}
	if rep.Summary.PlanningAttributed == 0 {
		t.Fatalf("expected at least one context_planning attribution, got summary=%+v changes=%+v", rep.Summary, rep.Changes)
	}
	if rep.Summary.ProviderAttributed == 0 {
		t.Fatalf("expected at least one provider_cache_behavior attribution, got summary=%+v changes=%+v", rep.Summary, rep.Changes)
	}
	for _, c := range rep.Changes {
		if c.Change != vcacheobserve.ChangeCacheCreateSpike {
			continue
		}
		switch c.Family {
		case "alpha":
			if c.Cause != vcacheobserve.CausePlanning {
				t.Fatalf("alpha spike should be context_planning, got %s", c.Cause)
			}
		case "beta":
			if c.Cause != vcacheobserve.CauseProviderBehavior {
				t.Fatalf("beta spike should be provider_cache_behavior, got %s", c.Cause)
			}
		}
	}
}

func TestRunVCacheContextJoinHumanTable(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:20Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:30Z","input_tokens":100,"cache_creation_input_tokens":42000}`,
	)
	events := writeLines(t, "events.jsonl",
		`{"kind":"context_reset","family":"alpha","unix_millis":1782432025000,"outcome":"reset","detail":"hidden restart"}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheContextJoin(&out, &errb, []string{"--telemetry", tel, "--events", events})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"context-join", "attribution:", "context_planning", "family"} {
		if !strings.Contains(got, want) {
			t.Fatalf("output missing %q:\n%s", want, got)
		}
	}
}

func TestRunVCacheContextJoinNeedsEvents(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000}`,
	)
	var out, errb bytes.Buffer
	if code := runVCacheContextJoin(&out, &errb, []string{"--telemetry", tel}); code != 2 {
		t.Fatalf("want usage exit 2 with no --events, got %d, stderr=%s", code, errb.String())
	}
}

func TestRunVCacheContextJoinNeedsSource(t *testing.T) {
	events := writeLines(t, "events.jsonl",
		`{"kind":"context_reset","family":"alpha","unix_millis":1000,"outcome":"reset"}`,
	)
	var out, errb bytes.Buffer
	if code := runVCacheContextJoin(&out, &errb, []string{"--events", events}); code != 2 {
		t.Fatalf("want usage exit 2 with no --transcript/--telemetry, got %d", code)
	}
}

func TestRunVCacheContextJoinUnknownEventKindSkipped(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000}`,
		`{"session_id":"alpha","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500}`,
	)
	events := writeLines(t, "events.jsonl",
		`{"kind":"not_a_real_kind","family":"alpha","unix_millis":1000,"outcome":"bogus"}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheContextJoin(&out, &errb, []string{"--telemetry", tel, "--events", events, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	var rep vcacheobserve.JoinReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v", err)
	}
	if rep.Events != 0 {
		t.Fatalf("expected the unknown-kind event to be skipped, got Events=%d", rep.Events)
	}
}
