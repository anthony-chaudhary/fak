package main

// Tests for the provider-firsttry/first_try_green_rate cross-provider cell
// (#4494): the attempt-ledger adapter, the catalog lockstep witness, and the
// live-arm fold, mirroring the context-restore (#4486) and provider-toolcall
// (#4493) witnesses. The pure per-provider fold's own semantics are pinned in
// internal/dojo/claim_provider_firsttry_test.go; here the shell proves the
// ledger adapts honestly and the cell surfaces in `fak dojo run --live`.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestDojoLoadDispatchFirstTryAttemptsAdaptsSpawnedRows pins the attempt-ledger
// adapter: one FirstTryAttempt per SPAWNED start row, keyed by the shared
// leaderboard provider key, with Attempts 0 because the ledger records the
// dispatch but no per-worker acceptance-gate attempt yet — so the pure fold
// over the adapted rows is ONE honest UNMEASURED episode, never a fabricated
// 0.0 rate.
func TestDojoLoadDispatchFirstTryAttemptsAdaptsSpawnedRows(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FAK_LOOP_LEDGER", "")
	ledger := filepath.Join(root, ".fak", "loops.jsonl")
	for _, ev := range []loopmgr.Event{
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"},
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "codex"},
		// Not a dispatched worker: a start row with another reason, and an end row.
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "RESTARTED", Principal: "claude"},
		{LoopID: "issue-resolve-progress", Kind: loopmgr.EventEnd, Reason: "OK", Metrics: map[string]int64{"closed_now": 1}},
	} {
		if _, err := loopmgr.Append(ledger, ev); err != nil {
			t.Fatalf("seed loop ledger: %v", err)
		}
	}

	got := loadDispatchFirstTryAttempts(root)
	if len(got) != 2 {
		t.Fatalf("want one attempt per SPAWNED row, got %d: %+v", len(got), got)
	}
	// Providers map through the shared leaderboard keying (#4505): the claude
	// backend keys claude, the codex backend keys gpt.
	if got[0].Provider != "claude" || got[1].Provider != "gpt" {
		t.Fatalf("providers must map through the shared leaderboard keying, got %+v", got)
	}
	for i, a := range got {
		if a.Attempts != 0 || a.Green {
			t.Fatalf("row %d: the ledger records no per-worker gate attempt yet, want Attempts 0 / not Green, got %+v", i, a)
		}
	}

	// The honesty rule end-to-end: dispatched-but-ungated rows fold to ONE
	// honest UNMEASURED episode carrying the registered claim.
	eps := dojo.ProviderFirstTryEpisodes(got)
	if len(eps) != 1 || eps[0].Outcome.Measured {
		t.Fatalf("ungated dispatches must fold one honest UNMEASURED episode, got %+v", eps)
	}
	if eps[0].Prediction.Lever != "provider-firsttry" || eps[0].Prediction.Metric != "first_try_green_rate" || eps[0].Prediction.Claimed != 0.5 {
		t.Fatalf("UNMEASURED episode lost the registered claim: %+v", eps[0].Prediction)
	}

	// Fail-open: a root with no ledger at all adapts to nil and the lever still
	// folds the same honest UNMEASURED episode.
	if got := loadDispatchFirstTryAttempts(t.TempDir()); got != nil {
		t.Fatalf("a missing ledger must adapt to nil attempts, got %+v", got)
	}
	ins, err := providerFirstTryLever{root: t.TempDir()}.Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatalf("Episodes on a missing ledger must fail open, got %v", err)
	}
	if len(ins) != 1 || ins[0].Outcome.Measured {
		t.Fatalf("a missing ledger must fold one honest UNMEASURED episode, got %+v", ins)
	}
}

// TestDojoCatalogMatchesProviderFirstTryEmittedMetrics keeps the registered
// catalog row in lockstep with the metrics the lever actually emits.
func TestDojoCatalogMatchesProviderFirstTryEmittedMetrics(t *testing.T) {
	t.Setenv("FAK_LOOP_LEDGER", "")
	emitted := map[string]bool{}
	ins, err := providerFirstTryLever{root: t.TempDir()}.Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatalf("Episodes: %v", err)
	}
	for _, in := range ins {
		emitted[in.Prediction.Metric] = true
	}
	found := false
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-firsttry" {
			continue
		}
		found = true
		for _, m := range lv.Metrics {
			if !emitted[m.Name] {
				t.Fatalf("catalog advertises metric %q the lever never emits", m.Name)
			}
		}
		if len(lv.Metrics) != len(emitted) {
			t.Fatalf("catalog lists %d metrics but the lever emits %d", len(lv.Metrics), len(emitted))
		}
	}
	if !found {
		t.Fatal("dojoLeverCatalog does not carry the registered provider-firsttry row")
	}
}

// TestRunDojoLiveFoldsProviderFirstTry pins the Done condition (#4494):
// `fak dojo run --live` reports provider-firsttry/first_try_green_rate — here
// honestly UNMEASURED, because the attempt ledger records dispatches but no
// per-worker acceptance-gate attempts yet — carrying the registered claim,
// never a fabricated rate.
func TestRunDojoLiveFoldsProviderFirstTry(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	// Isolate the session-corpus folds from the host's real ~/.claude corpus.
	t.Setenv("CLAUDE_CONFIG_DIR", t.TempDir())

	// A live corpus with one scored row so the live arm folds a report, plus a
	// loop ledger with one dispatched worker (the attempt-ledger population).
	input := dojo.ScoredInput{
		Prediction: dojo.Prediction{Lever: "vcache-warmth", Metric: "warm_recall", Claimed: 1.0, Unit: "fraction", Basis: "test"},
		Outcome:    dojo.Outcome{Realized: 1.0, Provenance: dojo.Observed, Source: "test provider cache window", Measured: true, Sample: 4},
	}
	if err := logDojoEpisodeFile("serve", []dojo.ScoredInput{input}); err != nil {
		t.Fatalf("logDojoEpisodeFile: %v", err)
	}
	t.Setenv("FAK_LOOP_LEDGER", "")
	ledger := filepath.Join(root, ".fak", "loops.jsonl")
	if _, err := loopmgr.Append(ledger, loopmgr.Event{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"}); err != nil {
		t.Fatalf("seed loop ledger: %v", err)
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should fold, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	var eps []dojo.Episode
	for _, e := range got.Report.Episodes {
		if e.Lever == "provider-firsttry" {
			eps = append(eps, e)
		}
	}
	if len(eps) != 1 {
		t.Fatalf("live report should carry exactly one provider-firsttry episode, got %d: %s", len(eps), out.String())
	}
	ep := eps[0]
	if ep.Metric != "first_try_green_rate" || ep.Claimed != 0.5 {
		t.Fatalf("provider-firsttry episode lost the registered claim: %+v", ep)
	}
	if ep.Verdict != dojo.VerdictUnmeasured {
		t.Fatalf("with no per-worker gate attempts on the ledger the cell must stay honestly %s, got %+v", dojo.VerdictUnmeasured, ep)
	}
}
