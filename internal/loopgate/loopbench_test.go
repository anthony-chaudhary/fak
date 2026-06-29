package loopgate

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// updateGolden regenerates the committed bench artifact:
//
//	go test ./internal/loopgate/ -run TestVerifiedVsNaiveLoopBench -update
var updateGolden = flag.Bool("update", false, "regenerate the bench report golden artifact")

const benchGoldenPath = "testdata/verified-vs-naive-loop.report.json"

// TestVerifiedVsNaiveLoopBench is the re-runnable command behind issue #1190:
// it produces the JSON report and proves the gated loop's false-done rate is
// lower than the naive loop's on the corpus.
func TestVerifiedVsNaiveLoopBench(t *testing.T) {
	rep := CompareLoops(DefaultBenchCorpus())

	// Headline (issue #1190 acceptance): the gate lowers the false-done rate.
	if !(rep.Gated.FalseDoneRate < rep.Naive.FalseDoneRate) {
		t.Fatalf("gated false-done rate %.3f must be below naive %.3f",
			rep.Gated.FalseDoneRate, rep.Naive.FalseDoneRate)
	}
	if rep.Naive.FalseDone == 0 {
		t.Fatalf("corpus does not exercise a false done; naive false-done count is 0")
	}
	if rep.Gated.FalseDone != 0 {
		t.Fatalf("the gate accepts only witnessed dones; gated false-done = %d, want 0", rep.Gated.FalseDone)
	}
	// Every solvable episode must reach a witnessed done under the gate.
	if rep.Gated.WitnessedDoneReached != rep.Gated.Episodes {
		t.Fatalf("gated loop reached witnessed done on %d/%d episodes",
			rep.Gated.WitnessedDoneReached, rep.Gated.Episodes)
	}
	// Slop the gate removes must be non-negative; the corpus is built so a
	// false done ships heavier slop than its eventual witnessed commit.
	if rep.SlopShippedDelta <= 0 {
		t.Fatalf("gate removed no slop; delta = %d", rep.SlopShippedDelta)
	}
	// Net-true (docs/standards/net-true-value.md): the win must survive the
	// gate's own adjudication cost.
	if !rep.NetTrue.GateEarnsKeep {
		t.Fatalf("gate does not earn its keep net of cost: net = %d tokens", rep.NetTrue.NetTokens)
	}

	got, err := rep.JSON()
	if err != nil {
		t.Fatalf("render report: %v", err)
	}
	got = append(got, '\n')

	if *updateGolden {
		if err := os.MkdirAll(filepath.Dir(benchGoldenPath), 0o755); err != nil {
			t.Fatalf("mkdir testdata: %v", err)
		}
		if err := os.WriteFile(benchGoldenPath, got, 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		return
	}

	want, err := os.ReadFile(benchGoldenPath)
	if err != nil {
		t.Fatalf("read golden %s (regenerate with -update): %v", benchGoldenPath, err)
	}
	if !bytes.Equal(bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n")), got) {
		t.Fatalf("report drifted from %s; regenerate with: go test ./internal/loopgate/ -run %s -update",
			benchGoldenPath, t.Name())
	}
}

// TestCompareLoopsTooEasyCorpusIsHonest checks the honest "corpus too easy to
// separate them" path the issue allows: a corpus with no trap yields a
// zero false-done delta and a finding that says so rather than a gate win.
func TestCompareLoopsTooEasyCorpusIsHonest(t *testing.T) {
	easy := []Episode{
		{ID: "honest-a", Turns: []BenchTurn{{ClaimedDone: true, Witnessed: true, GateCostTokens: 100}}},
		{ID: "honest-b", Turns: []BenchTurn{{ClaimedDone: true, Witnessed: true, GateCostTokens: 100}}},
	}
	rep := CompareLoops(easy)
	if rep.Naive.FalseDone != 0 || rep.FalseDoneRateDelta != 0 {
		t.Fatalf("easy corpus should not separate the loops; got false-done %d delta %.3f",
			rep.Naive.FalseDone, rep.FalseDoneRateDelta)
	}
	if rep.NetTrue.GateEarnsKeep {
		t.Fatalf("no rework avoided on an easy corpus, so the gate cannot earn its keep here")
	}
	if got := rep.Finding; got == "" || !contains(got, "too easy") {
		t.Fatalf("finding should report the corpus is too easy, got %q", got)
	}
}

func contains(s, sub string) bool {
	return bytes.Contains([]byte(s), []byte(sub))
}
