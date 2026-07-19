package main

// Tests for the provider-toolcall/tool_call_success_rate cross-provider cell
// (#4507): the pure per-provider fold, the catalog lockstep witness, and the
// live-arm per-provider render, mirroring the provider-cache (#4504) witnesses.

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

func TestDojoproviderToolcallEpisodesFromCorpus(t *testing.T) {
	claude := map[string]sessionaudit.ModelCounts{"claude-fable-5": {Turns: 6, Input: 100}}
	sessions := []sessionaudit.Session{
		// Two claude sessions summing to 30 tool results with 3 errored ->
		// first-try success rate 0.9 over sample 30.
		{PerModel: claude, NToolResult: 20,
			Behavior: sessionaudit.Behavior{ToolErrors: map[string]int64{"Bash": 2}}},
		{PerModel: map[string]sessionaudit.ModelCounts{
			"claude-opus-4-8": {Turns: 4, Input: 100},
			// A synthetic (non-billed) row never decides the provider key.
			"<synthetic>": {Turns: 3, Input: 999},
		}, NToolResult: 10,
			Behavior: sessionaudit.Behavior{ToolErrors: map[string]int64{"Edit": 1}}},
		// A gpt session: 4 results, 3 errored (one from a result the transcript
		// never attributed to a tool_use) -> rate 0.25.
		{PerModel: map[string]sessionaudit.ModelCounts{"gpt-5-codex": {Turns: 4, Input: 300}}, NToolResult: 4,
			Behavior: sessionaudit.Behavior{ToolErrors: map[string]int64{"Bash": 2, "?": 1}}},
		// A kimi session that made NO tool calls contributes nothing: with no
		// adjudicated call there is no rate to fold, so kimi gets no episode.
		{PerModel: map[string]sessionaudit.ModelCounts{"kimi-k2": {Turns: 2, Input: 500}}},
		// A session with only harness-synthetic turns has no billed provider to
		// key by, whatever tools it called.
		{PerModel: map[string]sessionaudit.ModelCounts{"<synthetic>": {Turns: 5}}, NToolResult: 7},
		// An unreadable session never folds.
		{Error: "unreadable", PerModel: claude, NToolResult: 50,
			Behavior: sessionaudit.Behavior{ToolErrors: map[string]int64{"Bash": 50}}},
	}
	ins := providerToolcallEpisodesFromCorpus(sessions)
	if len(ins) != 2 {
		t.Fatalf("expected one episode per tool-calling provider (claude, gpt), got %d: %+v", len(ins), ins)
	}
	// Providers render in stable sorted order so ticks stay comparable.
	claudeEp, gptEp := ins[0], ins[1]
	if claudeEp.Prediction.Lever != "provider-toolcall" || claudeEp.Prediction.Metric != "tool_call_success_rate" {
		t.Fatalf("episode cell wrong: %s/%s", claudeEp.Prediction.Lever, claudeEp.Prediction.Metric)
	}
	// The pinned claim literal (#4507): a seeded genuine estimate, not a floor.
	if claudeEp.Prediction.Claimed != 0.9 || claudeEp.Prediction.IntentionalFloor || claudeEp.Prediction.LowerIsBetter {
		t.Fatalf("pinned provider-toolcall claim drifted: %+v", claudeEp.Prediction)
	}
	if !claudeEp.Outcome.Measured || claudeEp.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a tool-calling provider must measure (OBSERVED): %+v", claudeEp.Outcome)
	}
	if claudeEp.Outcome.Realized != 0.9 || claudeEp.Outcome.Sample != 30 || !strings.Contains(claudeEp.Outcome.Source, "claude") {
		t.Fatalf("claude fold wrong: %+v, want rate 0.9 over 30 results", claudeEp.Outcome)
	}
	if !gptEp.Outcome.Measured || gptEp.Outcome.Realized != 0.25 || gptEp.Outcome.Sample != 4 || !strings.Contains(gptEp.Outcome.Source, "gpt") {
		t.Fatalf("gpt fold wrong: %+v, want rate 0.25 over 4 results", gptEp.Outcome)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED,
	// never a fabricated rate.
	empty := providerToolcallEpisodesFromCorpus(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesProviderToolcallEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the provider-toolcall lever emits.
	emitted := map[string]bool{}
	for _, in := range providerToolcallEpisodesFromCorpus([]sessionaudit.Session{
		{PerModel: map[string]sessionaudit.ModelCounts{"claude-fable-5": {Turns: 1, Input: 1}}, NToolResult: 1},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-toolcall" {
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

// TestRunDojoLiveFoldsProviderToolcallPerProvider pins the cross-provider cell
// (#4507): `fak dojo run --live` folds provider-toolcall/tool_call_success_rate
// from the session corpus with ONE episode PER PROVIDER — each carrying the same
// registered claim and that provider's own first-try success rate — and emits no
// episode for a provider whose sessions never called a tool.
func TestRunDojoLiveFoldsProviderToolcallPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with two providers: a claude session with ten
	// adjudicated tool calls, one errored (rate 9/10 = 0.9, exactly the claim),
	// and a glm session that never called a tool (no episode), dropped where
	// sessionaudit.Discover (via CLAUDE_CONFIG_DIR) finds them without touching
	// the real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var uses, results []string
	for i := 0; i < 10; i++ {
		id := string(rune('a' + i))
		uses = append(uses, `{"type":"tool_use","id":"tu_`+id+`","name":"Bash","input":{"command":"ls"}}`)
		isErr := "false"
		if i == 0 {
			isErr = "true"
		}
		results = append(results, `{"type":"tool_result","tool_use_id":"tu_`+id+`","is_error":`+isErr+`,"content":"out"}`)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-fable-5","usage":{"input_tokens":100,"output_tokens":5},"content":[` + strings.Join(uses, ",") + `]}}` + "\n" +
		`{"type":"user","message":{"content":[` + strings.Join(results, ",") + `]}}` + "\n"
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
		t.Fatalf("dojo run --live should measure the provider-toolcall fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	var eps []dojo.Episode
	for _, e := range got.Report.Episodes {
		if e.Lever == "provider-toolcall" {
			eps = append(eps, e)
		}
	}
	if len(eps) != 1 {
		t.Fatalf("live report should carry exactly one provider-toolcall episode (claude tool-called, glm did not), got %d: %s", len(eps), out.String())
	}
	ep := eps[0]
	if !strings.Contains(ep.Source, "provider claude") {
		t.Fatalf("episode should fold the claude sessions: %+v", ep)
	}
	if ep.Metric != "tool_call_success_rate" || ep.Claimed != 0.9 {
		t.Fatalf("claude episode lost the registered claim: %+v", ep)
	}
	if ep.Verdict != dojo.VerdictCalibrated || ep.Realized != 0.9 || ep.Sample != 10 {
		t.Fatalf("claude episode wrong: %+v, want measured rate 0.9 over 10 results", ep)
	}
}
