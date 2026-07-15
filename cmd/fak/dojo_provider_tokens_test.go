package main

// Tests for the provider-tokens/tokens_per_completed_issue cross-provider cell
// (#4503): the pure per-provider fold, the catalog lockstep witness, and the
// live-arm per-provider render, mirroring the provider-cost (#4488) witnesses.
// This is TOTAL billed tokens (input + output + cache_read + cache_creation) per
// completed session keyed by provider — NOT USD cost, NOT output-only tokens; a
// provider whose corpus rows carry no billed tokens folds UNMEASURED, never a
// fabricated 0 (#4490's honesty rule).

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

func TestDojoProviderTokensEpisodesFromSessions(t *testing.T) {
	sessions := []sessionaudit.Session{
		// Two completed claude sessions, each totalling 1,000,000 billed tokens
		// (input + output + cache_read + cache_creation) -> mean 1,000,000 per
		// completed session over 2 sessions.
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 2, Input: 200000, Output: 300000, CacheRead: 400000, CacheCreate: 100000},
		}},
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 1, Input: 1000000},
			// A synthetic (non-billed) row never keys or totals a provider.
			"<synthetic>": {Turns: 5, Input: 999},
		}},
		// A glm-dominant session: glm carries it (5 turns > 1) but its rows bill NO
		// tokens, so its total is UNMEASURED, never a fabricated 0. The stray claude
		// turn does NOT fabricate a glm total — only glm's own models total into
		// glm's column.
		{AssistantTurns: 6, PerModel: map[string]sessionaudit.ModelCounts{
			"glm-5.2":         {Turns: 5},
			"claude-opus-4-8": {Turns: 1, Input: 100000},
		}},
		// Not completed tasks: unreadable, turn-less, and synthetic-only sessions
		// must not fold.
		{Error: "unreadable", AssistantTurns: 9, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 9, Input: 999999}}},
		{AssistantTurns: 0},
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{"<synthetic>": {Turns: 2}}},
	}
	ins := providerTokensEpisodesFromSessions(sessions)
	if len(ins) != 2 {
		t.Fatalf("expected one episode per provider (claude measured, glm unmeasured), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order so ticks stay comparable.
	claude, glm := ins[0], ins[1]
	if claude.Prediction.Lever != "provider-tokens" || claude.Prediction.Metric != "tokens_per_completed_issue" {
		t.Fatalf("episode cell wrong: %s/%s", claude.Prediction.Lever, claude.Prediction.Metric)
	}
	// The pinned claim literal (#4503): a seeded genuine estimate, not a floor.
	if claude.Prediction.Claimed != 1000000.0 || claude.Prediction.IntentionalFloor || claude.Prediction.LowerIsBetter {
		t.Fatalf("pinned provider-tokens claim drifted: %+v", claude.Prediction)
	}
	if !claude.Outcome.Measured || claude.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a provider with billed tokens must measure (OBSERVED): %+v", claude.Outcome)
	}
	if claude.Outcome.Realized != 1000000.0 || claude.Outcome.Sample != 2 || !strings.Contains(claude.Outcome.Source, "claude") {
		t.Fatalf("claude fold wrong: %+v, want 1,000,000 tokens over 2 completed sessions", claude.Outcome)
	}
	if glm.Outcome.Measured || !strings.Contains(glm.Outcome.Source, "glm") || !strings.Contains(glm.Outcome.Source, "no billed tokens") {
		t.Fatalf("a token-free provider must be UNMEASURED with the reason named, never 0: %+v", glm.Outcome)
	}
	if glm.Outcome.Sample != 1 {
		t.Fatalf("the glm episode should carry its completed-session count (1), got %+v", glm.Outcome)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED,
	// never a fabricated zero.
	empty := providerTokensEpisodesFromSessions(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesProviderTokensEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the provider-tokens lever emits.
	emitted := map[string]bool{}
	for _, in := range providerTokensEpisodesFromSessions([]sessionaudit.Session{
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 1, Input: 200000}}},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-tokens" {
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

// TestRunDojoLiveFoldsProviderTokensPerProvider pins the cross-provider cell
// (#4503): `fak dojo run --live` folds provider-tokens/tokens_per_completed_issue
// from the session corpus with ONE episode PER PROVIDER — carrying the same
// registered claim and that provider's own total billed tokens per completed
// session.
func TestRunDojoLiveFoldsProviderTokensPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with a claude session billing 1,000,000 total
	// tokens (200000 input + 300000 output + 400000 cache_read + 100000
	// cache_creation), dropped where sessionaudit.Discover (via CLAUDE_CONFIG_DIR)
	// finds it without touching the real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-opus-4-8","usage":{"input_tokens":200000,"output_tokens":300000,"cache_read_input_tokens":400000,"cache_creation_input_tokens":100000}}}` + "\n"
	if err := os.WriteFile(filepath.Join(nsDir, "session-claude.jsonl"), []byte(claude), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should measure the provider-tokens fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	var claudeEp dojo.Episode
	found := false
	for _, e := range got.Report.Episodes {
		if e.Lever != "provider-tokens" {
			continue
		}
		if strings.Contains(e.Source, "provider claude") {
			claudeEp = e
			found = true
		}
	}
	if !found {
		t.Fatalf("live report should carry a provider-tokens episode for claude: %s", out.String())
	}
	if claudeEp.Metric != "tokens_per_completed_issue" || claudeEp.Claimed != 1000000.0 {
		t.Fatalf("claude episode lost the registered claim: %+v", claudeEp)
	}
	if claudeEp.Realized != 1000000.0 || claudeEp.Sample != 1 {
		t.Fatalf("claude episode wrong: %+v, want measured 1,000,000 tokens over 1 completed session", claudeEp)
	}
}
