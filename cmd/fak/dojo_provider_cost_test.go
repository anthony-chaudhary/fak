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
	"math"
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
		// A glm-dominant session: glm carries it (5 turns > 1) and is now PRICED in
		// the sessionaudit table (#4823 — published glm rate), so its cost MEASURES:
		// 500000 input tokens x $1.4/MTok = $0.70. The stray claude turn does NOT
		// contaminate glm's cost — only glm's own models bill into glm's column.
		{AssistantTurns: 6, PerModel: map[string]sessionaudit.ModelCounts{
			"glm-5.2":         {Turns: 5, Input: 500000},
			"claude-opus-4-8": {Turns: 1, Input: 100000},
		}},
		// A gpt-dominant session: gpt carries it (5 turns > 1) but stays UNPRICED in
		// the sessionaudit table, so its cost is UNMEASURED, never a fabricated
		// $0.00 (#4490's honesty rule). The stray claude turn does NOT fabricate a
		// gpt cost — only gpt's own models bill into gpt's column.
		{AssistantTurns: 6, PerModel: map[string]sessionaudit.ModelCounts{
			"gpt-5":           {Turns: 5, Input: 500000},
			"claude-opus-4-8": {Turns: 1, Input: 100000},
		}},
		// Not completed tasks: unreadable, turn-less, and synthetic-only sessions
		// must not fold.
		{Error: "unreadable", AssistantTurns: 9, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 9, Input: 999999}}},
		{AssistantTurns: 0},
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{"<synthetic>": {Turns: 2}}},
	}
	ins := providerCostEpisodesFromSessions(sessions)
	if len(ins) != 3 {
		t.Fatalf("expected one episode per provider (claude+glm measured, gpt unmeasured), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order (claude, glm, gpt) so ticks stay comparable.
	claude, glm, gpt := ins[0], ins[1], ins[2]
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
	// glm is now priced (#4823): $0.70 over 1 completed session, MEASURED/OBSERVED.
	if !glm.Outcome.Measured || glm.Outcome.Provenance != dojo.Observed || !strings.Contains(glm.Outcome.Source, "glm") {
		t.Fatalf("a now-priced provider (glm) must MEASURE (OBSERVED): %+v", glm.Outcome)
	}
	if math.Abs(glm.Outcome.Realized-0.70) > 1e-9 || glm.Outcome.Sample != 1 {
		t.Fatalf("glm fold wrong: %+v, want $0.70 over 1 completed session", glm.Outcome)
	}
	if gpt.Outcome.Measured || !strings.Contains(gpt.Outcome.Source, "gpt") || !strings.Contains(gpt.Outcome.Source, "no priced billing") {
		t.Fatalf("an unpriced provider must be UNMEASURED with the reason named, never $0.00: %+v", gpt.Outcome)
	}
	if gpt.Outcome.Sample != 1 {
		t.Fatalf("the gpt episode should carry its completed-session count (1), got %+v", gpt.Outcome)
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
// (#4488 / #4823): `fak dojo run --live` folds provider-cost/cost_per_completed_issue
// from the session corpus with ONE episode PER PROVIDER — each carrying the same
// registered claim and that provider's own billed USD per completed session. It is
// the Done-condition witness for #4823: the now-priced non-Claude providers
// (deepseek, glm, kimi) fold MEASURED, while a provider the table still cannot
// price (gpt) stays UNMEASURED instead of a fabricated $0.00.
func TestRunDojoLiveFoldsProviderCostPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus, one session per provider, dropped where
	// sessionaudit.Discover (via CLAUDE_CONFIG_DIR) finds them without touching the
	// real host corpus. claude bills 200000 input at opus rates ($15/MTok -> $3.00);
	// deepseek/glm/kimi each bill 1 MTok input at their published rates (#4823) so
	// they MEASURE; gpt stays unpriced so it folds UNMEASURED.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fixtures := map[string]string{
		"session-claude.jsonl":   `{"type":"assistant","message":{"id":"m1","model":"claude-opus-4-8","usage":{"input_tokens":200000,"output_tokens":0}}}`,
		"session-deepseek.jsonl": `{"type":"assistant","message":{"id":"d1","model":"deepseek-v4-pro","usage":{"input_tokens":1000000,"output_tokens":0}}}`,
		"session-glm.jsonl":      `{"type":"assistant","message":{"id":"g1","model":"glm-5.2","usage":{"input_tokens":1000000,"output_tokens":0}}}`,
		"session-kimi.jsonl":     `{"type":"assistant","message":{"id":"k1","model":"kimi-k2.6","usage":{"input_tokens":1000000,"output_tokens":0}}}`,
		"session-gpt.jsonl":      `{"type":"assistant","message":{"id":"p1","model":"gpt-5","usage":{"input_tokens":1000000,"output_tokens":0}}}`,
	}
	for name, line := range fixtures {
		if err := os.WriteFile(filepath.Join(nsDir, name), []byte(line+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
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
		for _, p := range []string{"claude", "deepseek", "glm", "kimi", "gpt"} {
			if strings.Contains(e.Source, "provider "+p) {
				costs[p] = e
			}
		}
	}
	claudeEp := costs["claude"]
	if claudeEp.Metric != "cost_per_completed_issue" || claudeEp.Claimed != 3.0 {
		t.Fatalf("claude episode lost the registered claim: %+v", claudeEp)
	}
	if claudeEp.Verdict != dojo.VerdictCalibrated || claudeEp.Realized != 3.0 || claudeEp.Sample != 1 {
		t.Fatalf("claude episode wrong: %+v, want measured $3.00 over 1 completed session", claudeEp)
	}
	// #4823 Done condition: deepseek, glm, and kimi now fold MEASURED (not UNMEASURED).
	for _, p := range []string{"deepseek", "glm", "kimi"} {
		ep, ok := costs[p]
		if !ok {
			t.Fatalf("live report missing a provider-cost episode for %q: %s", p, out.String())
		}
		if ep.Verdict == dojo.VerdictUnmeasured || ep.Realized <= 0 || ep.Sample != 1 || ep.Claimed != 3.0 {
			t.Fatalf("%s episode must MEASURE per #4823 (priced provider), got %+v", p, ep)
		}
	}
	// gpt stays unpriced -> UNMEASURED (the #4490 honesty rule still fires).
	gptEp, ok := costs["gpt"]
	if !ok || gptEp.Verdict != dojo.VerdictUnmeasured || gptEp.Claimed != 3.0 {
		t.Fatalf("gpt episode must be UNMEASURED (still-unpriced provider) with the claim intact: %+v", gptEp)
	}
}
