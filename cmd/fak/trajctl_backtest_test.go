package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// trajctl_backtest_test.go — issue #2573. The end-to-end CLI witness for the qualification
// gate: `fak trajctl backtest --scorer X --corpus Y` emits a calibration report, and a
// corpus on which the scorer reads history BACKWARDS fails qualification with a distinct
// exit code. Every corpus below is written through trajctl.Append, so the fixtures are real
// recorded ledgers and not hand-built structs that skip validation.

// writeBacktestCorpus writes rows to a temp ledger and returns its path.
func writeBacktestCorpus(t *testing.T, rows []trajctl.Row) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	for _, r := range rows {
		if err := trajctl.Append(path, r); err != nil {
			t.Fatalf("seed corpus: %v", err)
		}
	}
	return path
}

// alignedCorpus records history the commit scorer reads CORRECTLY: the phase whose commit
// was witnessed belongs to the objective that ended up met.
func alignedCorpus(t *testing.T) string {
	t.Helper()
	plan := []trajctl.PlanPhase{{ID: "p1", Title: "the only phase"}}
	return writeBacktestCorpus(t, []trajctl.Row{
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "met", Statement: "the met objective", Plan: plan, Status: trajctl.StatusMet}),
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "dropped", Statement: "the abandoned objective", Plan: plan, Status: trajctl.StatusAbandoned}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "met", Value: 1.0, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, UnixMillis: 10,
			Evidence: []trajctl.EvidenceRef{{Kind: "commit", Ref: "aaa1", Detail: "p1"}}}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "dropped", Value: 0.0, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, UnixMillis: 10}),
	})
}

// invertedCorpus records history the commit scorer reads BACKWARDS: the objective carrying
// the witnessed commit is the one that was abandoned, and the one that was met has no
// recorded commit at all. Replaying the scorer over it anti-correlates, so this is the
// known-bad shape the done condition requires to FAIL qualification through the real CLI.
func invertedCorpus(t *testing.T) string {
	t.Helper()
	plan := []trajctl.PlanPhase{{ID: "p1", Title: "the only phase"}}
	return writeBacktestCorpus(t, []trajctl.Row{
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "met", Statement: "met with no witnessed commit", Plan: plan, Status: trajctl.StatusMet}),
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "dropped", Statement: "abandoned despite a witnessed commit", Plan: plan, Status: trajctl.StatusAbandoned}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "met", Value: 1.0, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, UnixMillis: 10}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "dropped", Value: 0.0, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, UnixMillis: 10,
			Evidence: []trajctl.EvidenceRef{{Kind: "commit", Ref: "bbb1", Detail: "p1"}}}),
	})
}

// runBacktest is the CLI call under test, returning exit code and both streams.
func runBacktest(argv ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runTrajctlBacktest(&out, &errb, argv)
	return code, out.String(), errb.String()
}

// TestTrajctlBacktestEmitsACalibrationReportAndFailsAKnownBadCorpus is the done-condition
// witness at the CLI: `--scorer X --corpus Y` emits the pinned calibration report, and the
// same scorer on history it reads backwards is REFUSED — with an exit code distinct from
// both success and a tool error.
func TestTrajctlBacktestEmitsACalibrationReportAndFailsAKnownBadCorpus(t *testing.T) {
	code, out, errb := runBacktest("--scorer", "commit", "--corpus", alignedCorpus(t), "--json")
	if code != 0 {
		t.Fatalf("aligned corpus exit=%d, want 0\nstdout=%s\nstderr=%s", code, out, errb)
	}
	var good trajctl.BacktestReport
	if err := json.Unmarshal([]byte(out), &good); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if good.Schema != trajctl.BacktestSchema {
		t.Errorf("schema = %q, want %q", good.Schema, trajctl.BacktestSchema)
	}
	if good.Verdict != trajctl.BacktestQualified || good.Reason != trajctl.BacktestTracksOutcome {
		t.Fatalf("aligned corpus = %s/%s, want %s/%s: %s", good.Verdict, good.Reason, trajctl.BacktestQualified, trajctl.BacktestTracksOutcome, good.Detail)
	}
	if good.Method != trajctl.CommitScorerMethod {
		t.Errorf("the alias must resolve to the stable method id, got %q", good.Method)
	}
	// A CALIBRATION report, not a bare verdict: the per-objective replay must be there.
	if len(good.Cases) != 2 {
		t.Fatalf("want one case per replayed objective, got %d", len(good.Cases))
	}

	code, out, errb = runBacktest("--scorer", "commit", "--corpus", invertedCorpus(t), "--json")
	if code != trajctlBacktestExitRefused {
		t.Fatalf("known-bad corpus exit=%d, want %d (ran-and-refused)\nstdout=%s\nstderr=%s",
			code, trajctlBacktestExitRefused, out, errb)
	}
	var bad trajctl.BacktestReport
	if err := json.Unmarshal([]byte(out), &bad); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if bad.Verdict != trajctl.BacktestNotQualified || bad.Reason != trajctl.BacktestAntiCorrelated {
		t.Fatalf("known-bad corpus = %s/%s, want %s/%s: %s", bad.Verdict, bad.Reason, trajctl.BacktestNotQualified, trajctl.BacktestAntiCorrelated, bad.Detail)
	}
	if bad.Qualified() {
		t.Error("a refused backtest must never report Qualified()")
	}
}

// TestTrajctlBacktestToolErrorsExitApartFromRefusals pins the confusion risk at the shell
// boundary: a backtest that could not RUN exits 1 with a BACKTEST_ERROR report, never the
// refusal code and never 0. A CI step branching on the exit code can therefore tell a broken
// invocation from a scorer that lost.
func TestTrajctlBacktestToolErrorsExitApartFromRefusals(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "no-such-corpus.jsonl")
	code, out, _ := runBacktest("--scorer", "commit", "--corpus", missing, "--json")
	if code != trajctlBacktestExitErrored {
		t.Fatalf("missing corpus exit=%d, want %d\n%s", code, trajctlBacktestExitErrored, out)
	}
	var rep trajctl.BacktestReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("a tool error must still emit a report: %v\n%s", err, out)
	}
	if rep.Verdict != trajctl.BacktestErrored || rep.Reason != trajctl.BacktestNoCorpus {
		t.Errorf("missing corpus = %s/%s, want %s/%s", rep.Verdict, rep.Reason, trajctl.BacktestErrored, trajctl.BacktestNoCorpus)
	}

	// A scorer that cannot run offline is a typed refusal to run, not a failed scorer.
	code, out, _ = runBacktest("--scorer", "judge", "--corpus", alignedCorpus(t), "--json")
	if code != trajctlBacktestExitErrored {
		t.Fatalf("non-replayable scorer exit=%d, want %d\n%s", code, trajctlBacktestExitErrored, out)
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if rep.Reason != trajctl.BacktestNotReplayable {
		t.Errorf("judge scorer reason = %s, want %s", rep.Reason, trajctl.BacktestNotReplayable)
	}

	// An unknown scorer name likewise.
	code, out, _ = runBacktest("--scorer", "no-such-scorer", "--corpus", alignedCorpus(t), "--json")
	if code != trajctlBacktestExitErrored {
		t.Fatalf("unknown scorer exit=%d, want %d\n%s", code, trajctlBacktestExitErrored, out)
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if rep.Reason != trajctl.BacktestUnknownScorer {
		t.Errorf("unknown scorer reason = %s, want %s", rep.Reason, trajctl.BacktestUnknownScorer)
	}
}

// TestTrajctlBacktestRequiresAnExplicitCorpus guards the footgun the flag comment names: a
// backtest must never silently grade a scorer against the live ledger it is meant to
// protect, so --corpus has no default and its absence is a usage error.
func TestTrajctlBacktestRequiresAnExplicitCorpus(t *testing.T) {
	if code, _, errb := runBacktest("--scorer", "commit", "--json"); code != 2 {
		t.Errorf("missing --corpus exit=%d, want 2 (usage): %s", code, errb)
	}
	if code, _, errb := runBacktest("--corpus", alignedCorpus(t), "--json"); code != 2 {
		t.Errorf("missing --scorer exit=%d, want 2 (usage): %s", code, errb)
	}
}

// TestTrajctlBacktestRendersTheHumanReport covers the non-JSON path: the verdict and its
// reason code land on the last line, and every replayed objective is listed.
func TestTrajctlBacktestRendersTheHumanReport(t *testing.T) {
	code, out, errb := runBacktest("--scorer", "commit", "--corpus", alignedCorpus(t),
		"--incumbent", trajctl.ActivityDivergenceScorerMethod)
	// The incumbent emits nothing on a ledger corpus, so it is unmeasurable and the
	// regression check cannot fire — the candidate still qualifies on its own reading.
	if code != 0 {
		t.Fatalf("exit=%d, want 0\nstdout=%s\nstderr=%s", code, out, errb)
	}
	for _, want := range []string{
		string(trajctl.BacktestQualified),
		string(trajctl.BacktestTracksOutcome),
		"candidate  " + trajctl.CommitScorerMethod,
		"incumbent  " + trajctl.ActivityDivergenceScorerMethod,
		"met", "dropped",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q, got:\n%s", want, out)
		}
	}
	// An unmeasurable incumbent must read as n/a, never as a coefficient of zero.
	if !strings.Contains(out, "n/a") {
		t.Errorf("an unmeasurable incumbent must render n/a, got:\n%s", out)
	}
}

// TestTrajctlBacktestIsReachableFromTheVerb pins the dispatch wiring and the documented
// qualification gate: `fak trajctl backtest` routes to the subcommand, and the verb's usage
// names the gate so an operator can find it without reading the source.
func TestTrajctlBacktestIsReachableFromTheVerb(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{"backtest", "--scorer", "commit", "--corpus", alignedCorpus(t), "--json"}); code != 0 {
		t.Fatalf("dispatch exit=%d\nstdout=%s\nstderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), trajctl.BacktestSchema) {
		t.Errorf("the dispatched subcommand must emit the pinned report, got:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	if code := runTrajctl(&out, &errb, []string{"--help"}); code != 0 {
		t.Fatalf("help exit=%d", code)
	}
	help := out.String()
	for _, want := range []string{"backtest --scorer X --corpus Y", "a scorer ships with its backtest", "INCONCLUSIVE"} {
		if !strings.Contains(help, want) {
			t.Errorf("usage must document the qualification gate (%q missing), got:\n%s", want, help)
		}
	}
}
