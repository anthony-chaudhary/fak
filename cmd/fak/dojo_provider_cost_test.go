package main

// Tests for the provider-cost/cost_per_completed_issue cross-provider cell
// (#4488): the pure per-provider fold, the catalog lockstep witness, and the
// live-arm per-provider render, mirroring the provider-cache (#4504) witnesses.
// The dollars come from the EXISTING sessionaudit per-model price table (opus/
// sonnet/haiku/fable); a provider whose models that table does not price folds
// UNMEASURED, never a fabricated $0.00 (#4490's honesty rule).

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

func TestDojoProviderCostEpisodesFromSessions(t *testing.T) {
	sessions := []sessionaudit.Session{
		// Two completed claude sessions, each billing $3.00 at opus rates
		// (200000 input tokens x $15/MTok) -> mean $3.00 per completed session
		// over 2 sessions.
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 2, Input: 200000},
		}},
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 1, Input: 200000},
			// A synthetic (non-billed) row never keys or prices a provider.
			"<synthetic>": {Turns: 5, Input: 999},
		}},
		// A glm-dominant session: glm carries it (5 turns > 1) but is UNPRICED in
		// the sessionaudit table, so its cost is UNMEASURED, never a fabricated
		// $0.00. The stray claude turn does NOT fabricate a glm cost — only glm's
		// own models bill into glm's column.
		{AssistantTurns: 6, PerModel: map[string]sessionaudit.ModelCounts{
			"glm-5.2":         {Turns: 5, Input: 500000},
			"claude-opus-4-8": {Turns: 1, Input: 100000},
		}},
		// Not completed tasks: unreadable, turn-less, and synthetic-only sessions
		// must not fold.
		{Error: "unreadable", AssistantTurns: 9, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 9, Input: 999999}}},
		{AssistantTurns: 0},
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{"<synthetic>": {Turns: 2}}},
	}
	ins := providerCostEpisodesFromSessions(sessions)
	if len(ins) != 2 {
		t.Fatalf("expected one episode per provider (claude measured, glm unmeasured), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order so ticks stay comparable.
	claude, glm := ins[0], ins[1]
	if claude.Prediction.Lever != "provider-cost" || claude.Prediction.Metric != "cost_per_completed_issue" {
		t.Fatalf("episode cell wrong: %s/%s", claude.Prediction.Lever, claude.Prediction.Metric)
	}
	// The pinned claim literal (#4488): a seeded genuine estimate, not a floor.
	if claude.Prediction.Claimed != 3.0 || claude.Prediction.IntentionalFloor || claude.Prediction.LowerIsBetter {
		t.Fatalf("pinned provider-cost claim drifted: %+v", claude.Prediction)
	}
	if !claude.Outcome.Measured || claude.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a provider with priced billing must measure (OBSERVED): %+v", claude.Outcome)
	}
	if claude.Outcome.Realized != 3.0 || claude.Outcome.Sample != 2 || !strings.Contains(claude.Outcome.Source, "claude") {
		t.Fatalf("claude fold wrong: %+v, want $3.00 over 2 completed sessions", claude.Outcome)
	}
	if glm.Outcome.Measured || !strings.Contains(glm.Outcome.Source, "glm") || !strings.Contains(glm.Outcome.Source, "no priced billing") {
		t.Fatalf("an unpriced provider must be UNMEASURED with the reason named, never $0.00: %+v", glm.Outcome)
	}
	if glm.Outcome.Sample != 1 {
		t.Fatalf("the glm episode should carry its completed-session count (1), got %+v", glm.Outcome)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED,
	// never a fabricated zero cost.
	empty := providerCostEpisodesFromSessions(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesProviderCostEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the provider-cost lever emits.
	emitted := map[string]bool{}
	for _, in := range providerCostEpisodesFromSessions([]sessionaudit.Session{
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 1, Input: 200000}}},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-cost" {
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

// TestRunDojoLiveFoldsProviderCostPerProvider pins the cross-provider cell
// (#4488): `fak dojo run --live` folds provider-cost/cost_per_completed_issue
// from the session corpus with ONE episode PER PROVIDER — each carrying the same
// registered claim and that provider's own billed USD per completed session — and
// marks a provider the price table cannot price UNMEASURED instead of $0.00.
func TestRunDojoLiveFoldsProviderCostPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with two providers: a claude session billing
	// 200000 input tokens at opus rates ($15/MTok -> $3.00) and a glm session the
	// sessionaudit table cannot price (-> UNMEASURED), dropped where
	// sessionaudit.Discover (via CLAUDE_CONFIG_DIR) finds them without touching
	// the real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-opus-4-8","usage":{"input_tokens":200000,"output_tokens":0}}}` + "\n"
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
		t.Fatalf("dojo run --live should measure the provider-cost fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	costs := map[string]dojo.Episode{}
	for _, e := range got.Report.Episodes {
		if e.Lever != "provider-cost" {
			continue
		}
		switch {
		case strings.Contains(e.Source, "provider claude"):
			costs["claude"] = e
		case strings.Contains(e.Source, "provider glm"):
			costs["glm"] = e
		default:
			t.Fatalf("provider-cost episode with an unexpected provider source: %+v", e)
		}
	}
	if len(costs) != 2 {
		t.Fatalf("live report should carry one provider-cost episode per provider (claude, glm), got %d: %s", len(costs), out.String())
	}
	claudeEp := costs["claude"]
	if claudeEp.Metric != "cost_per_completed_issue" || claudeEp.Claimed != 3.0 {
		t.Fatalf("claude episode lost the registered claim: %+v", claudeEp)
	}
	if claudeEp.Verdict != dojo.VerdictCalibrated || claudeEp.Realized != 3.0 || claudeEp.Sample != 1 {
		t.Fatalf("claude episode wrong: %+v, want measured $3.00 over 1 completed session", claudeEp)
	}
	glmEp := costs["glm"]
	if glmEp.Verdict != dojo.VerdictUnmeasured || glmEp.Claimed != 3.0 {
		t.Fatalf("glm episode must be UNMEASURED (unpriced provider) with the claim intact: %+v", glmEp)
	}
}
