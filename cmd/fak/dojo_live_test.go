package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestLogDojoEpisodeFileWritesScorableInputs(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	input := dojo.ScoredInput{
		Prediction: dojo.Prediction{
			Lever:   "vcache-warmth",
			Metric:  "warm_recall",
			Claimed: 1.0,
			Unit:    "fraction",
			Basis:   "test",
		},
		Outcome: dojo.Outcome{
			Realized:   1.0,
			Provenance: dojo.Observed,
			Source:     "test provider cache window",
			Measured:   true,
			Sample:     4,
		},
	}
	if err := logDojoEpisodeFile("guard", []dojo.ScoredInput{input}); err != nil {
		t.Fatalf("logDojoEpisodeFile: %v", err)
	}

	lc, err := dojo.ReadLiveCorpus(filepath.Join(root, filepath.FromSlash(dojo.LiveEpisodesRel)))
	if err != nil {
		t.Fatalf("ReadLiveCorpus: %v", err)
	}
	if lc.Found != 1 || lc.Scorable != 1 {
		t.Fatalf("live corpus found/scorable = %d/%d, want 1/1 (%+v)", lc.Found, lc.Scorable, lc)
	}
	inputs := dojo.ScorableLiveEpisodes(lc)
	if len(inputs) != 1 || inputs[0].Prediction.Metric != "warm_recall" || !inputs[0].Outcome.Measured {
		t.Fatalf("scorable inputs = %+v, want measured warm_recall", inputs)
	}
}

func TestRunDojoLiveFoldsScoredRows(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	// Isolate the provider-turns session-corpus fold (#4505) from the host's
	// real ~/.claude corpus; an empty corpus folds one honest UNMEASURED episode.
	t.Setenv("CLAUDE_CONFIG_DIR", t.TempDir())
	input := dojo.ScoredInput{
		Prediction: dojo.Prediction{
			Lever:   "vcache-warmth",
			Metric:  "warm_recall",
			Claimed: 1.0,
			Unit:    "fraction",
			Basis:   "test",
		},
		Outcome: dojo.Outcome{
			Realized:   1.0,
			Provenance: dojo.Observed,
			Source:     "test provider cache window",
			Measured:   true,
			Sample:     4,
		},
	}
	if err := logDojoEpisodeFile("serve", []dojo.ScoredInput{input}); err != nil {
		t.Fatalf("logDojoEpisodeFile: %v", err)
	}

	// Seed a loop ledger so the live arm's dispatch-yield fold (#4497) measures:
	// 2 dispatched workers, 1 diff-witnessed close -> verified_ship_rate 0.5.
	t.Setenv("FAK_LOOP_LEDGER", "")
	ledger := filepath.Join(root, ".fak", "loops.jsonl")
	for _, ev := range []loopmgr.Event{
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED"},
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED"},
		{LoopID: "issue-resolve-progress", Kind: loopmgr.EventEnd, Reason: "OK", Metrics: map[string]int64{"closed_now": 1}},
	} {
		if _, err := loopmgr.Append(ledger, ev); err != nil {
			t.Fatalf("seed loop ledger: %v", err)
		}
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should fold the scored row, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	if got.Live.Found != 1 || got.Live.Scorable != 1 {
		t.Fatalf("live corpus found/scorable = %d/%d, want 1/1", got.Live.Found, got.Live.Scorable)
	}
	if got.Report.Measured != 2 || got.Report.Unmeasured != 6 {
		t.Fatalf("live report measured/unmeasured = %d/%d, want 2/6 (scored live row + dispatch-yield measured; provider-turns, provider-cache, cache-read-share, provider-cost, provider-tokens, and provider-completion each honestly unmeasured on the empty session corpus)", got.Report.Measured, got.Report.Unmeasured)
	}
	var yield *dojo.Episode
	for i := range got.Report.Episodes {
		if got.Report.Episodes[i].Lever == "dispatch-yield" {
			yield = &got.Report.Episodes[i]
		}
	}
	if yield == nil {
		t.Fatalf("live report is missing the dispatch-yield episode: %s", out.String())
	}
	if yield.Metric != "verified_ship_rate" || yield.Claimed != 0.5 || yield.Realized != 0.5 {
		t.Fatalf("dispatch-yield episode wrong: %+v", yield)
	}
}

// TestRunDojoLiveFoldsProviderTurnsPerProvider pins the cross-provider cell
// (#4505): `fak dojo run --live` folds provider-turns/turns_per_task from the
// session corpus with ONE episode PER PROVIDER — each carrying the same
// registered claim and that provider's own measured median — so the report is
// the leaderboard the issue asks for, not a single blended number.
func TestRunDojoLiveFoldsProviderTurnsPerProvider(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	// A fixture session corpus with two providers: a 3-turn claude task and a
	// 1-turn glm task, dropped where sessionaudit.Discover (via
	// CLAUDE_CONFIG_DIR) finds them without touching the real host corpus.
	home := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	nsDir := filepath.Join(home, "projects", "fixture-ns")
	if err := os.MkdirAll(nsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claude := `{"type":"assistant","message":{"id":"m1","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n" +
		`{"type":"assistant","message":{"id":"m2","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n" +
		`{"type":"assistant","message":{"id":"m3","model":"claude-opus-4-8","usage":{"input_tokens":10,"output_tokens":5}}}` + "\n"
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
		t.Fatalf("dojo run --live should measure the provider-turns fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	medians := map[string]dojo.Episode{}
	for _, e := range got.Report.Episodes {
		if e.Lever != "provider-turns" {
			continue
		}
		switch {
		case strings.Contains(e.Source, "provider claude"):
			medians["claude"] = e
		case strings.Contains(e.Source, "provider glm"):
			medians["glm"] = e
		default:
			t.Fatalf("provider-turns episode with an unexpected provider source: %+v", e)
		}
	}
	if len(medians) != 2 {
		t.Fatalf("live report should carry one provider-turns episode per provider (claude, glm), got %d: %s", len(medians), out.String())
	}
	for provider, wantMedian := range map[string]float64{"claude": 3, "glm": 1} {
		e := medians[provider]
		if e.Metric != "turns_per_task" || e.Claimed != 20.0 {
			t.Fatalf("%s episode lost the registered claim: %+v", provider, e)
		}
		if e.Verdict == dojo.VerdictUnmeasured || e.Realized != wantMedian || e.Sample != 1 {
			t.Fatalf("%s episode wrong: %+v, want measured median %g over 1 session", provider, e, wantMedian)
		}
	}
}
