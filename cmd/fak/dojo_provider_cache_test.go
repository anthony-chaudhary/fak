package main

// Tests for the provider-cache/cache_read_share cross-provider cell (#4504):
// the pure per-provider fold, the catalog lockstep witness, and the live-arm
// per-provider render, mirroring the provider-turns (#4505) witnesses.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestDojoProviderCacheEpisodesFromSessions(t *testing.T) {
	sessions := []sessionaudit.Session{
		// Two claude sessions summing to input 200 + cache_read 1700 +
		// cache_creation 100 = 2000 total input tokens -> share 0.85 over 10 turns.
		{PerModel: map[string]sessionaudit.ModelCounts{
			"claude-fable-5": {Turns: 6, Input: 100, CacheRead: 850, CacheCreate: 50},
		}},
		{PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 4, Input: 100, CacheRead: 850, CacheCreate: 50},
			// A synthetic (non-billed) row never folds into any provider.
			"<synthetic>": {Turns: 3, Input: 999, CacheRead: 999},
		}},
		// A gpt session with cache fields present: share 100/400 = 0.25.
		{PerModel: map[string]sessionaudit.ModelCounts{
			"gpt-5-codex": {Turns: 4, Input: 300, CacheRead: 100},
		}},
		// A kimi session that bills input but NO cache fields at all: the share
		// must be UNMEASURED, never a fabricated 0.0 (#4490's honesty rule).
		{PerModel: map[string]sessionaudit.ModelCounts{
			"kimi-k2": {Turns: 2, Input: 500},
		}},
		// An unreadable session never folds.
		{Error: "unreadable", PerModel: map[string]sessionaudit.ModelCounts{
			"claude-fable-5": {Turns: 9, Input: 9999, CacheRead: 1},
		}},
	}
	ins := providerCacheEpisodesFromSessions(sessions)
	if len(ins) != 3 {
		t.Fatalf("expected one episode per provider (claude, gpt, kimi), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order so ticks stay comparable.
	claude, gpt, kimi := ins[0], ins[1], ins[2]
	if claude.Prediction.Lever != "provider-cache" || claude.Prediction.Metric != "cache_read_share" {
		t.Fatalf("episode cell wrong: %s/%s", claude.Prediction.Lever, claude.Prediction.Metric)
	}
	// The pinned claim literal (#4504): a seeded genuine estimate, not a floor.
	if claude.Prediction.Claimed != 0.8 || claude.Prediction.IntentionalFloor || claude.Prediction.LowerIsBetter {
		t.Fatalf("pinned provider-cache claim drifted: %+v", claude.Prediction)
	}
	if !claude.Outcome.Measured || claude.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a provider with cache billing must measure (OBSERVED): %+v", claude.Outcome)
	}
	if claude.Outcome.Realized != 0.85 || claude.Outcome.Sample != 10 || !strings.Contains(claude.Outcome.Source, "claude") {
		t.Fatalf("claude fold wrong: %+v, want share 0.85 over 10 turns", claude.Outcome)
	}
	if !gpt.Outcome.Measured || gpt.Outcome.Realized != 0.25 || gpt.Outcome.Sample != 4 || !strings.Contains(gpt.Outcome.Source, "gpt") {
		t.Fatalf("gpt fold wrong: %+v, want share 0.25 over 4 turns", gpt.Outcome)
	}
	if kimi.Outcome.Measured || !strings.Contains(kimi.Outcome.Source, "kimi") || !strings.Contains(kimi.Outcome.Source, "no cache_read/cache_creation") {
		t.Fatalf("a provider billing no cache fields must be UNMEASURED with the reason named: %+v", kimi.Outcome)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED,
	// never a fabricated zero share.
	empty := providerCacheEpisodesFromSessions(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesProviderCacheEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the provider-cache lever emits.
	emitted := map[string]bool{}
	for _, in := range providerCacheEpisodesFromSessions([]sessionaudit.Session{
		{PerModel: map[string]sessionaudit.ModelCounts{"claude-fable-5": {Turns: 1, Input: 1, CacheRead: 1}}},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-cache" {
			continue
		}
		for _, m := range lv.Metrics {
			if !emitted[m.Name] {
				t.Fatalf("catalog advertises metric %q the lever never emits", m.Name)
			}
		}
		if len(lv.Metrics) != len(emitted) {
			t.Fatalf("catalog lists %d metrics but the lever emits %d", len(lv.Metrics), len(emitted))
		}
	}
}

// TestRunDojoLiveFoldsProviderCachePerProvider pins the cross-provider cell
// (#4504): `fak dojo run --live` folds provider-cache/cache_read_share from the
// session corpus with ONE episode PER PROVIDER — each carrying the same
// registered claim and that provider's own billed cache-read share — and marks
// a provider relaying no cache fields UNMEASURED instead of fabricating 0.0.
func TestRunDojoLiveFoldsProviderCachePerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with two providers: a claude session billing
	// cache reads (share 800/1000 = 0.8) and a glm session billing input only
	// (no cache fields -> UNMEASURED), dropped where sessionaudit.Discover (via
	// CLAUDE_CONFIG_DIR) finds them without touching the real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-fable-5","usage":{"input_tokens":100,"output_tokens":5,"cache_read_input_tokens":800,"cache_creation_input_tokens":100}}}` + "\n"
	if err := os.WriteFile(filepath.Join(nsDir, "session-claude.jsonl"), []byte(claude), 0o600); err != nil {
		t.Fatal(err)
	}
	glm := `{"type":"assistant","message":{"id":"g1","model":"glm-5.2","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n"
	if err := os.WriteFile(filepath.Join(nsDir, "session-glm.jsonl"), []byte(glm), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should measure the provider-cache fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	shares := map[string]dojo.Episode{}
	for _, e := range got.Report.Episodes {
		if e.Lever != "provider-cache" {
			continue
		}
		switch {
		case strings.Contains(e.Source, "provider claude"):
			shares["claude"] = e
		case strings.Contains(e.Source, "provider glm"):
			shares["glm"] = e
		default:
			t.Fatalf("provider-cache episode with an unexpected provider source: %+v", e)
		}
	}
	if len(shares) != 2 {
		t.Fatalf("live report should carry one provider-cache episode per provider (claude, glm), got %d: %s", len(shares), out.String())
	}
	claudeEp := shares["claude"]
	if claudeEp.Metric != "cache_read_share" || claudeEp.Claimed != 0.8 {
		t.Fatalf("claude episode lost the registered claim: %+v", claudeEp)
	}
	if claudeEp.Verdict != dojo.VerdictCalibrated || claudeEp.Realized != 0.8 || claudeEp.Sample != 1 {
		t.Fatalf("claude episode wrong: %+v, want measured share 0.8 over 1 turn", claudeEp)
	}
	glmEp := shares["glm"]
	if glmEp.Verdict != dojo.VerdictUnmeasured || glmEp.Claimed != 0.8 {
		t.Fatalf("glm episode must be UNMEASURED (no cache fields billed) with the claim intact: %+v", glmEp)
	}
}
