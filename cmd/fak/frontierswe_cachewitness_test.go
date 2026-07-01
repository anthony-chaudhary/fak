package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// metricsBody renders a minimal fak gateway /metrics body carrying the KV-prefix
// cache family cachewitness.Parse folds — the same series a real fak serve emits.
func metricsBody(turns, prompt, reused, provider uint64) string {
	return fmt.Sprintf(`# HELP fak_gateway_kv_prefix_turns_total turns
fak_gateway_kv_prefix_turns_total %d
fak_gateway_kv_prefix_prompt_tokens_total %d
fak_gateway_kv_prefix_reused_tokens_total %d
fak_gateway_inference_cached_prompt_tokens_total %d
`, turns, prompt, reused, provider)
}

// TestFrontiersweCacheWitnessOfflineFoldsSeries is the C8 acceptance on the offline
// path: a directory of captured /metrics scrapes folds into the per-turn reuse
// series + the realized reuse rate (the C14 input), with WITNESSED/OBSERVED
// provenance kept separate, and a JSONL trace is written. No gateway, no model.
func TestFrontiersweCacheWitnessOfflineFoldsSeries(t *testing.T) {
	dir := t.TempDir()
	// Four scrapes with cumulative counters growing and reuse compounding (turn 1
	// cold, turns 2..4 reuse the resident prefix). Zero-padded names = scrape order.
	scrapes := []struct{ turns, prompt, reused, provider uint64 }{
		{1, 1000, 0, 0},
		{2, 2100, 1000, 0},
		{3, 3300, 2100, 0},
		{4, 4600, 3300, 0},
	}
	for i, s := range scrapes {
		name := filepath.Join(dir, fmt.Sprintf("%04d.metrics", i))
		if err := os.WriteFile(name, []byte(metricsBody(s.turns, s.prompt, s.reused, s.provider)), 0o644); err != nil {
			t.Fatalf("write scrape %d: %v", i, err)
		}
	}
	trace := filepath.Join(dir, "trace.jsonl")

	var stdout, stderr bytes.Buffer
	code := runFrontiersweCacheWitness(&stdout, &stderr, []string{"--metrics-dir", dir, "--out", trace})
	if code != 0 {
		t.Fatalf("cache-witness exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}

	var series frontierswe.CacheWitnessSeries
	if err := json.Unmarshal(stdout.Bytes(), &series); err != nil {
		t.Fatalf("stdout is not the series JSON: %v\nstdout:\n%s", err, stdout.String())
	}
	if series.Schema != frontierswe.CacheWitnessSchema {
		t.Errorf("schema = %q, want %q", series.Schema, frontierswe.CacheWitnessSchema)
	}
	if len(series.Points) != 4 {
		t.Fatalf("points = %d, want 4", len(series.Points))
	}
	if want := 3300.0 / 4600.0; series.RealizedReuseRate < want-1e-9 || series.RealizedReuseRate > want+1e-9 {
		t.Errorf("realized reuse rate = %v, want %v", series.RealizedReuseRate, want)
	}
	if !series.CacheBit {
		t.Errorf("CacheBit = false, want true")
	}
	if series.Provenance["cum_reused_tokens"] != "WITNESSED" || series.Provenance["provider_cache_read_tokens"] != "OBSERVED" {
		t.Errorf("provenance labels wrong: %v", series.Provenance)
	}

	// The JSONL trace has one line per point plus a summary line, each self-describing.
	b, err := os.ReadFile(trace)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 5 {
		t.Fatalf("trace lines = %d, want 5 (4 points + summary)\n%s", len(lines), b)
	}
	var summary struct {
		Kind              string  `json:"kind"`
		RealizedReuseRate float64 `json:"realized_reuse_rate"`
	}
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &summary); err != nil {
		t.Fatalf("summary line not JSON: %v", err)
	}
	if summary.Kind != "summary" {
		t.Errorf("last line kind = %q, want summary", summary.Kind)
	}
	// The stderr summary keeps the provenance split explicit.
	if !strings.Contains(stderr.String(), "WITNESSED") || !strings.Contains(stderr.String(), "OBSERVED") {
		t.Errorf("stderr summary missing provenance split:\n%s", stderr.String())
	}
}

// TestFrontiersweCacheWitnessNoInputIsUsageError: no mode selected is a usage error
// (exit 2), not a silent empty success.
func TestFrontiersweCacheWitnessNoInputIsUsageError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runFrontiersweCacheWitness(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("no-input exit = %d, want 2\nstderr:\n%s", code, stderr.String())
	}
}

// TestFrontiersweCacheWitnessMutualExclusion: two input modes at once is rejected.
func TestFrontiersweCacheWitnessMutualExclusion(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFrontiersweCacheWitness(&stdout, &stderr, []string{"--metrics-dir", "x", "--gateway", "127.0.0.1:8080"})
	if code != 2 {
		t.Fatalf("dual-mode exit = %d, want 2\nstderr:\n%s", code, stderr.String())
	}
}
