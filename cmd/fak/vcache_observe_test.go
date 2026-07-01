package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// writeLines writes JSONL content to a temp file and returns its path.
func writeLines(t *testing.T, name string, lines ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRunVCacheObserveTelemetry(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000,"cache_read_input_tokens":0,"ephemeral_1h_input_tokens":40000}`,
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_creation_input_tokens":500,"cache_read_input_tokens":40000,"ephemeral_1h_input_tokens":500}`,
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:20Z","input_tokens":50,"cache_creation_input_tokens":500,"cache_read_input_tokens":40000,"ephemeral_1h_input_tokens":500}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheObserve(&out, &errb, []string{"--telemetry", tel})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"owner split", "provider", "prompt_cache", "fak", "NOT_OBSERVED", "base provider cache", "M5 governor", "OBSERVED", "DECISION", "false-warm"} {
		if !strings.Contains(got, want) {
			t.Fatalf("output missing %q:\n%s", want, got)
		}
	}
}

func TestRunVCacheObserveFlagsReorderedTelemetry(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:20Z","input_tokens":50,"cache_creation_input_tokens":500,"cache_read_input_tokens":40000,"ephemeral_1h_input_tokens":500}`,
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000,"cache_read_input_tokens":0,"ephemeral_1h_input_tokens":40000}`,
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_creation_input_tokens":500,"cache_read_input_tokens":40000,"ephemeral_1h_input_tokens":500}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheObserve(&out, &errb, []string{"--telemetry", tel})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	if got := out.String(); !strings.Contains(got, "input order: reordered 2 same-family turn(s)") {
		t.Fatalf("output missing reorder warning:\n%s", got)
	}
}

func TestRunVCacheObserveJSONReconciles(t *testing.T) {
	tel := writeLines(t, "tel.jsonl",
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:00Z","input_tokens":100,"cache_creation_input_tokens":40000,"ephemeral_1h_input_tokens":40000}`,
		`{"session_id":"s1","captured_utc":"2026-06-26T00:00:10Z","input_tokens":50,"cache_read_input_tokens":40000,"cache_creation_input_tokens":500,"ephemeral_1h_input_tokens":500}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheObserve(&out, &errb, []string{"--telemetry", tel, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	var rep vcacheobserve.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if rep.Schema != vcacheobserve.Schema {
		t.Fatalf("schema: %q", rep.Schema)
	}
	if rep.Turns != 2 || rep.FamilyCount != 1 {
		t.Fatalf("turns=%d families=%d", rep.Turns, rep.FamilyCount)
	}
	if len(rep.Panels) != 9 {
		t.Fatalf("want 9 panels, got %d", len(rep.Panels))
	}
	if len(rep.OwnerSlices) != 2 {
		t.Fatalf("want provider/fak owner slices, got %d", len(rep.OwnerSlices))
	}
	if rep.OwnerSlices[0].Owner != "provider" || rep.OwnerSlices[0].Mechanism != "prompt_cache" || rep.OwnerSlices[0].Provenance != vcacheobserve.Observed {
		t.Fatalf("provider owner slice not explicit: %+v", rep.OwnerSlices[0])
	}
	if rep.OwnerSlices[1].Owner != "fak" || rep.OwnerSlices[1].Provenance != vcacheobserve.NotObserved || rep.OwnerSlices[1].SavedTokenEquiv != 0 {
		t.Fatalf("fak owner slice should be unobserved zero, got %+v", rep.OwnerSlices[1])
	}
}

func TestRunVCacheObserveTranscript(t *testing.T) {
	// A minimal real-shaped Claude transcript: a user line, then two assistant turns
	// with nested usage (the second reads the cache the first wrote).
	tr := writeLines(t, "9ab.jsonl",
		`{"type":"user","timestamp":"2026-06-26T00:00:00Z","message":{"role":"user"}}`,
		`{"type":"assistant","timestamp":"2026-06-26T00:00:05Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":2,"cache_creation_input_tokens":40000,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":40000,"ephemeral_5m_input_tokens":0}}}}`,
		`{"type":"assistant","timestamp":"2026-06-26T00:00:15Z","message":{"role":"assistant","model":"claude-opus-4-8","usage":{"input_tokens":2,"cache_creation_input_tokens":500,"cache_read_input_tokens":40000,"cache_creation":{"ephemeral_1h_input_tokens":500,"ephemeral_5m_input_tokens":0}}}}`,
	)
	var out, errb bytes.Buffer
	code := runVCacheObserve(&out, &errb, []string{"--transcript", tr, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, errb.String())
	}
	var rep vcacheobserve.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v", err)
	}
	if rep.Turns != 2 {
		t.Fatalf("want 2 cache-bearing turns, got %d", rep.Turns)
	}
	if rep.Families[0].Key != "9ab" {
		t.Fatalf("family key from filename: got %q want 9ab", rep.Families[0].Key)
	}
	if rep.Aggregate.CacheReadTokens != 40000 {
		t.Fatalf("cache read tokens: got %.0f want 40000", rep.Aggregate.CacheReadTokens)
	}
}

func TestRunVCacheObserveNeedsSource(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runVCacheObserve(&out, &errb, nil); code != 2 {
		t.Fatalf("want usage exit 2 with no source, got %d", code)
	}
}
