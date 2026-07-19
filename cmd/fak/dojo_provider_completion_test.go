package main

// Tests for the provider-completion/verified_completion_rate cross-provider cell
// (#4506): the pure per-provider fold, the catalog lockstep witness, and the
// live-arm per-provider render, mirroring the provider-tokens (#4503) witnesses.
// This is verified closes / dispatched keyed by provider — the per-provider
// analog of the fak-aggregate dispatch-yield cell (#4497). A verified close is a
// completed, non-interrupted session (the corpus proxy for a closed task); an
// interrupted session is a dispatched attempt that did not verify-close. A
// provider whose every dispatched session was interrupted measures a witnessed
// 0.0 (never UNMEASURED); only a corpus with no dispatched session at all folds
// UNMEASURED, never a fabricated rate.

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

func TestDojoProviderCompletionEpisodesFromSessions(t *testing.T) {
	sessions := []sessionaudit.Session{
		// Three dispatched claude sessions; one was interrupted, so 2/3 reconcile as
		// a verified close -> rate 0.6667 over 3 dispatched.
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 2, Input: 100},
		}},
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 2, Input: 100},
		}},
		{AssistantTurns: 1, Interrupted: 1, PerModel: map[string]sessionaudit.ModelCounts{
			// An interrupted turn is still billed, so the session still keys to claude
			// as a dispatched attempt — it just does not count as a verified close.
			"claude-opus-4-8": {Turns: 1, Input: 100},
		}},
		// A single glm session, interrupted: dispatched 1, closed 0 -> a WITNESSED
		// 0.0 completion rate, measured (not UNMEASURED), never a fabricated rate.
		{AssistantTurns: 1, Interrupted: 1, PerModel: map[string]sessionaudit.ModelCounts{
			"glm-5.2": {Turns: 1, Input: 100},
		}},
		// Not dispatched task attempts: an unreadable session (no billed provider to
		// key by) and a synthetic-only session (no billable provider) must not fold.
		{Error: "unreadable", AssistantTurns: 9, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 9, Input: 999}}},
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{"<synthetic>": {Turns: 2}}},
	}
	ins := providerCompletionEpisodesFromSessions(sessions)
	if len(ins) != 2 {
		t.Fatalf("expected one episode per provider (claude, glm), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order so ticks stay comparable.
	claude, glm := ins[0], ins[1]
	if claude.Prediction.Lever != "provider-completion" || claude.Prediction.Metric != "verified_completion_rate" {
		t.Fatalf("episode cell wrong: %s/%s", claude.Prediction.Lever, claude.Prediction.Metric)
	}
	// The pinned claim literal (#4506): a seeded genuine estimate, higher-is-better,
	// not a floor.
	if claude.Prediction.Claimed != 0.5 || claude.Prediction.IntentionalFloor || claude.Prediction.LowerIsBetter {
		t.Fatalf("pinned provider-completion claim drifted: %+v", claude.Prediction)
	}
	if !claude.Outcome.Measured || claude.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a provider with dispatched sessions must measure (OBSERVED): %+v", claude.Outcome)
	}
	if math.Abs(claude.Outcome.Realized-2.0/3.0) > 1e-9 || claude.Outcome.Sample != 3 || !strings.Contains(claude.Outcome.Source, "claude") {
		t.Fatalf("claude fold wrong: %+v, want 2/3 closed over 3 dispatched", claude.Outcome)
	}
	// The key honesty case: an all-interrupted provider is a WITNESSED 0.0, measured
	// — never dropped to UNMEASURED and never a fabricated non-zero.
	if !glm.Outcome.Measured || glm.Outcome.Provenance != dojo.Observed {
		t.Fatalf("an all-interrupted provider must MEASURE a witnessed 0.0, not go UNMEASURED: %+v", glm.Outcome)
	}
	if glm.Outcome.Realized != 0.0 || glm.Outcome.Sample != 1 || !strings.Contains(glm.Outcome.Source, "glm") {
		t.Fatalf("glm fold wrong: %+v, want 0/1 closed over 1 dispatched", glm.Outcome)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED,
	// never a fabricated rate.
	empty := providerCompletionEpisodesFromSessions(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesProviderCompletionEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the provider-completion lever emits.
	emitted := map[string]bool{}
	for _, in := range providerCompletionEpisodesFromSessions([]sessionaudit.Session{
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Turns: 1, Input: 100}}},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-completion" {
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

// TestRunDojoLiveFoldsProviderCompletionPerProvider pins the cross-provider cell
// (#4506): `fak dojo run --live` folds provider-completion/verified_completion_rate
// from the session corpus with ONE episode PER PROVIDER — carrying the same
// registered claim and that provider's own verified-close / dispatched rate.
func TestRunDojoLiveFoldsProviderCompletionPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with a single, non-interrupted claude session (two
	// assistant turns) -> 1 dispatched, 1 verified close -> rate 1.0. Dropped where
	// sessionaudit.Discover (via CLAUDE_CONFIG_DIR) finds it without touching the
	// real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n" +
		`{"type":"assistant","message":{"id":"m2","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n"
	if err := os.WriteFile(filepath.Join(nsDir, "session-claude.jsonl"), []byte(claude), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should measure the provider-completion fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	var claudeEp dojo.Episode
	found := false
	for _, e := range got.Report.Episodes {
		if e.Lever != "provider-completion" {
			continue
		}
		if strings.Contains(e.Source, "provider claude") {
			claudeEp = e
			found = true
		}
	}
	if !found {
		t.Fatalf("live report should carry a provider-completion episode for claude: %s", out.String())
	}
	if claudeEp.Metric != "verified_completion_rate" || claudeEp.Claimed != 0.5 {
		t.Fatalf("claude episode lost the registered claim: %+v", claudeEp)
	}
	if claudeEp.Realized != 1.0 || claudeEp.Sample != 1 {
		t.Fatalf("claude episode wrong: %+v, want measured rate 1.0 over 1 dispatched session", claudeEp)
	}
}
