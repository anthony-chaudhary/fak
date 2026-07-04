package main

import (
	"bytes"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// economicsFixtureEvents is a small mixed ledger: a dispatch-tick loop that refuses
// one spawn (weekly-capped) and lands one witnessed run, plus a progress loop that
// carries the dispatch baseline/open/closed metrics. It exercises every witnessed
// figure the fold derives.
func economicsFixtureEvents() []loopmgr.Event {
	return []loopmgr.Event{
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventFire, TSUnixNano: 1000,
			Metrics: map[string]int64{"live": 2, "max_workers": 3}},
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusRefused,
			Reason: "WEEKLY_CAPPED", TSUnixNano: 2000, Metrics: map[string]int64{"live": 2, "max_workers": 3}},
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventFire, TSUnixNano: 3000,
			Metrics: map[string]int64{"live": 1, "max_workers": 3}},
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusAdmitted, TSUnixNano: 4000},
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventStart, RunID: "run-1", Status: loopmgr.StatusRunning, TSUnixNano: 5000},
		{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventEnd, RunID: "run-1", Status: loopmgr.StatusClaimedDone,
			TSUnixNano: 10_000, Metrics: map[string]int64{"duration_ms": 5000}},
		{LoopID: "issue-resolve-progress", Kind: loopmgr.EventWitness, Status: loopmgr.StatusWitnessedDone,
			TSUnixNano: 11_000, Metrics: map[string]int64{
				"baseline_open": 483, "open_now": 400, "closed_by_loop_total": 805, "witnessed_open": 1}},
	}
}

func TestFoldLoopEconomicsWitnessedFromLedger(t *testing.T) {
	rep := foldLoopEconomics(economicsFixtureEvents(), loopEconomicsOpts{LedgerPath: "x"}, time.Unix(0, 20_000).UTC())

	if rep.Schema != schemaLoopEconomics {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Loops != 2 {
		t.Fatalf("loops = %d, want 2", rep.Loops)
	}
	w := rep.Witnessed
	if w.Fires != 2 || w.Admitted != 1 || w.Refused != 1 || w.Started != 1 || w.Ended != 1 || w.WitnessedDone != 1 {
		t.Fatalf("lifecycle counts = %+v", w)
	}
	if w.DuplicateAttemptsAvoided != 1 {
		t.Fatalf("duplicate attempts avoided = %d, want 1 (the refused spawn)", w.DuplicateAttemptsAvoided)
	}
	if w.BaselineOpen != 483 || w.ObservedOpen != 400 || w.IssuesClosedByLoop != 805 || w.WitnessedOpen != 1 {
		t.Fatalf("progress figures = baseline=%d open=%d closed=%d wopen=%d",
			w.BaselineOpen, w.ObservedOpen, w.IssuesClosedByLoop, w.WitnessedOpen)
	}
	if w.EffectiveWorkers != 2 || w.WorkerCap != 3 {
		t.Fatalf("workers = %d of %d, want 2 of 3", w.EffectiveWorkers, w.WorkerCap)
	}
	if w.WallTimeSource != "run_durations" || math.Abs(w.WallTimeSeconds-5.0) > 1e-9 {
		t.Fatalf("wall time = %.3fs src=%s, want 5.0s run_durations", w.WallTimeSeconds, w.WallTimeSource)
	}
	if w.RetryRate == nil || math.Abs(*w.RetryRate-0.5) > 1e-9 {
		t.Fatalf("retry rate = %v, want 0.5", w.RetryRate)
	}
	if w.CloseRate == nil || math.Abs(*w.CloseRate-805.0/1205.0) > 1e-9 {
		t.Fatalf("close rate = %v, want 805/1205", w.CloseRate)
	}
}

func TestFoldLoopEconomicsTokenAccountsDefaultNotYet(t *testing.T) {
	rep := foldLoopEconomics(economicsFixtureEvents(), loopEconomicsOpts{LedgerPath: "x"}, time.Unix(0, 20_000).UTC())

	if rep.ProviderCache.Status != "not_yet" || rep.ProviderCache.TokenEquivalentSaved != 0 {
		t.Fatalf("provider account = %+v, want not_yet/0", rep.ProviderCache)
	}
	if rep.FakAuthored.Status != "not_yet" || rep.FakAuthored.TokenEquivalentSaved != 0 {
		t.Fatalf("fak account = %+v, want not_yet/0", rep.FakAuthored)
	}
	if rep.Modeled.Status != "not_yet" || rep.Modeled.TokenEquivalentSaved != 0 {
		t.Fatalf("modeled account = %+v, want not_yet/0", rep.Modeled)
	}
	// The three token accounts must be named as not-yet so a real 0 is never read as a win.
	want := map[string]bool{
		"provider_cache.token_equivalent_saved": true,
		"fak_authored.token_equivalent_saved":   true,
		"modeled.token_equivalent_saved":        true,
	}
	for _, f := range rep.NotYet {
		delete(want, f)
	}
	if len(want) != 0 {
		t.Fatalf("not_yet = %v, missing %v", rep.NotYet, want)
	}
}

func TestFoldLoopEconomicsOperatorInputsPopulateAccounts(t *testing.T) {
	providerTokens, fakTokens, perAvoided := int64(12_000), int64(3_400), int64(50_000)
	rep := foldLoopEconomics(economicsFixtureEvents(), loopEconomicsOpts{
		LedgerPath:              "x",
		ProviderCacheTokens:     &providerTokens,
		FakAuthoredTokens:       &fakTokens,
		ModeledTokensPerAvoided: &perAvoided,
	}, time.Unix(0, 20_000).UTC())

	if rep.ProviderCache.Status != "observed" || rep.ProviderCache.TokenEquivalentSaved != 12_000 {
		t.Fatalf("provider account = %+v", rep.ProviderCache)
	}
	if rep.FakAuthored.Status != "witnessed" || rep.FakAuthored.TokenEquivalentSaved != 3_400 {
		t.Fatalf("fak account = %+v", rep.FakAuthored)
	}
	// modeled = duplicate_attempts_avoided (1) * tokens_per_avoided (50k).
	if rep.Modeled.Status != "modeled" || rep.Modeled.TokenEquivalentSaved != 50_000 {
		t.Fatalf("modeled account = %+v, want 50000", rep.Modeled)
	}
	if len(rep.NotYet) != 0 {
		t.Fatalf("not_yet = %v, want empty once every account is supplied", rep.NotYet)
	}
}

func TestFoldLoopEconomicsLoopFilter(t *testing.T) {
	rep := foldLoopEconomics(economicsFixtureEvents(),
		loopEconomicsOpts{LedgerPath: "x", LoopID: "issue-resolve-progress"}, time.Unix(0, 20_000).UTC())

	if rep.Loops != 1 || rep.LoopFilter != "issue-resolve-progress" {
		t.Fatalf("filter = %q loops=%d", rep.LoopFilter, rep.Loops)
	}
	// The progress loop carries no fire/refuse events, so those rates are not witnessed.
	if rep.Witnessed.Fires != 0 || rep.Witnessed.RetryRate != nil {
		t.Fatalf("filtered witnessed = %+v, want no fires and nil retry rate", rep.Witnessed)
	}
	if rep.Witnessed.IssuesClosedByLoop != 805 {
		t.Fatalf("filtered closed = %d, want 805", rep.Witnessed.IssuesClosedByLoop)
	}
}

// TestLoopEconomicsCommandJSON is the command-level witness: a real hash-chained
// ledger folded through `fak loop economics --json`.
func TestLoopEconomicsCommandJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopTestEventAt(t, path, loopmgr.Event{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventFire}, 1000)
	appendLoopTestEventAt(t, path, loopmgr.Event{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventAdmit,
		Status: loopmgr.StatusRefused, Reason: "WEEKLY_CAPPED"}, 2000)
	appendLoopTestEventAt(t, path, loopmgr.Event{LoopID: "issue-resolve-progress", Kind: loopmgr.EventWitness,
		Status:  loopmgr.StatusWitnessedDone,
		Metrics: map[string]int64{"baseline_open": 483, "open_now": 400, "closed_by_loop_total": 805}}, 3000)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"economics", "--ledger", path, "--json"})
	if code != 0 {
		t.Fatalf("runLoop economics code=%d stderr=%s", code, stderr.String())
	}
	var rep loopEconomicsReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, stdout.String())
	}
	if rep.Schema != schemaLoopEconomics {
		t.Fatalf("schema = %q", rep.Schema)
	}
	if rep.Witnessed.DuplicateAttemptsAvoided != 1 || rep.Witnessed.IssuesClosedByLoop != 805 {
		t.Fatalf("witnessed = %+v", rep.Witnessed)
	}
	if rep.ProviderCache.Status != "not_yet" {
		t.Fatalf("provider account should default not_yet, got %q", rep.ProviderCache.Status)
	}
}

// TestLoopEconomicsCommandTable exercises the human render path end to end.
func TestLoopEconomicsCommandTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopTestEventAt(t, path, loopmgr.Event{LoopID: "issue-dispatch/claude", Kind: loopmgr.EventFire}, 1000)

	var stdout, stderr bytes.Buffer
	code := runLoop(&stdout, &stderr, []string{"economics", "--ledger", path})
	if code != 0 {
		t.Fatalf("runLoop economics code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"fak loop economics", "WITNESSED", "duplicate attempts avoided", "not_yet"} {
		if !bytes.Contains([]byte(out), []byte(want)) {
			t.Fatalf("table output missing %q:\n%s", want, out)
		}
	}
}
