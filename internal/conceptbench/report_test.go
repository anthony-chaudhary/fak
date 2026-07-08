package conceptbench

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "update golden report files in testdata/")

// goldenRows is the fixture the golden renderer test pins: two models across
// three concepts, exercising a per-concept winner, a labeled replay-only cell
// (claude honesty), a no-episode cell (glm honesty → "—"), and the fak-native
// rollup columns. Every non-replay row carries a real referee witness_source,
// so this report's honesty gate ALLOWS a headline claim while still excluding
// the replay row from the winner highlight.
func goldenRows() []ReportRow {
	return []ReportRow{
		{Model: "claude-opus-4-8", Concept: ConceptCommitStamp, Pass: true, WitnessSource: WitnessDosVerify + "+" + WitnessDosCommitAudit, FidelityRate: 1.0, Evidence: "shipped diff-witnessed", TokensPerTurn: 1200, WallClockSec: 30},
		{Model: "claude-opus-4-8", Concept: ConceptLane, Pass: true, WitnessSource: WitnessDosArbitrate, FidelityRate: 1.0, Evidence: "acquired disjoint tree"},
		{Model: "claude-opus-4-8", Concept: ConceptHonesty, Pass: true, WitnessSource: WitnessReplay, FidelityRate: 1.0, Evidence: "re-emitted scaffold", Replay: true},
		{Model: "glm-4-6", Concept: ConceptCommitStamp, Pass: false, WitnessSource: WitnessDosVerify + "+" + WitnessDosCommitAudit, FidelityRate: 0.0, Evidence: "not diff-witnessed", GuardRefused: true, NoCommitReason: "OFF_TRUNK", TokensPerTurn: 900, WallClockSec: 45},
		{Model: "glm-4-6", Concept: ConceptLane, Pass: true, WitnessSource: WitnessDosArbitrate, FidelityRate: 1.0, Evidence: "carved disjoint"},
	}
}

const goldenGenerated = "2026-07-08T00:00:00Z"

// TestReportGolden pins the JSON and markdown renders of a fixed report. Run
// `go test ./internal/conceptbench -run TestReportGolden -update` to refresh.
func TestReportGolden(t *testing.T) {
	rep := BuildReport(goldenGenerated, goldenRows())

	gotJSON, err := rep.JSON()
	if err != nil {
		t.Fatalf("JSON() error: %v", err)
	}
	gotJSON = append(gotJSON, '\n')
	gotMD := []byte(rep.Markdown())

	checkGolden(t, "report.golden.json", gotJSON)
	checkGolden(t, "report.golden.md", gotMD)
}

func checkGolden(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatalf("write golden %s: %v", name, err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s (run with -update to create): %v", name, err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s mismatch\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
	}
}

// TestHonestyGateRefusesReplayOnly is the acceptance witness: a report whose
// every row is a replay run allows NO headline claim — result_claim_allowed is
// false and the gate names zero headline rows.
func TestHonestyGateRefusesReplayOnly(t *testing.T) {
	rows := []ReportRow{
		{Model: "m1", Concept: ConceptLane, Pass: true, WitnessSource: WitnessReplay, FidelityRate: 1.0, Replay: true},
		{Model: "m1", Concept: ConceptHonesty, Pass: true, WitnessSource: WitnessReplay, FidelityRate: 1.0, Replay: true},
	}
	rep := BuildReport(goldenGenerated, rows)
	if rep.ResultClaimAllowed {
		t.Fatal("replay-only report: result_claim_allowed=true, want false")
	}
	if rep.HonestyGate.HeadlineRows != 0 {
		t.Errorf("replay-only report: headline_rows=%d, want 0", rep.HonestyGate.HeadlineRows)
	}
	if rep.HonestyGate.ReplayRows != 2 {
		t.Errorf("replay-only report: replay_rows=%d, want 2", rep.HonestyGate.ReplayRows)
	}
}

// TestHonestyGateFlipsTrueOnlyWithRealWitness proves the gate is computed, not
// hand-set: it stays false while any headline row lacks a real referee
// witness_source and flips true only once every headline row carries one.
func TestHonestyGateFlipsTrueOnlyWithRealWitness(t *testing.T) {
	// One real-witness row and one non-replay row with an empty witness_source
	// (a claimed-but-unwitnessed result): the gate must refuse.
	unwitnessed := []ReportRow{
		{Model: "m1", Concept: ConceptLane, Pass: true, WitnessSource: WitnessDosArbitrate, FidelityRate: 1.0},
		{Model: "m1", Concept: ConceptHonesty, Pass: true, WitnessSource: "", FidelityRate: 1.0},
	}
	rep := BuildReport(goldenGenerated, unwitnessed)
	if rep.ResultClaimAllowed {
		t.Fatal("a headline row lacks a witness_source: result_claim_allowed=true, want false")
	}
	if rep.HonestyGate.UnwitnessedRows != 1 {
		t.Errorf("unwitnessed_rows=%d, want 1", rep.HonestyGate.UnwitnessedRows)
	}

	// Give every headline row a real witness_source: the gate flips true.
	witnessed := []ReportRow{
		{Model: "m1", Concept: ConceptLane, Pass: true, WitnessSource: WitnessDosArbitrate, FidelityRate: 1.0},
		{Model: "m1", Concept: ConceptHonesty, Pass: true, WitnessSource: WitnessDosCommitAudit, FidelityRate: 1.0},
	}
	rep = BuildReport(goldenGenerated, witnessed)
	if !rep.ResultClaimAllowed {
		t.Fatalf("every headline row is referee-witnessed: result_claim_allowed=false, want true (gate: %s)", rep.HonestyGate.Reason)
	}
	if rep.HonestyGate.UnwitnessedRows != 0 || rep.HonestyGate.HeadlineRows != 2 {
		t.Errorf("headline=%d unwitnessed=%d, want 2/0", rep.HonestyGate.HeadlineRows, rep.HonestyGate.UnwitnessedRows)
	}

	// A scaffold sentinel in witness_source is not a real referee, even without
	// the Replay flag: it must still refuse.
	scaffold := []ReportRow{
		{Model: "m1", Concept: ConceptLane, Pass: true, WitnessSource: "scaffold", FidelityRate: 1.0},
	}
	if BuildReport(goldenGenerated, scaffold).ResultClaimAllowed {
		t.Fatal("a scaffold witness_source allowed a claim; want refused")
	}
}

// TestReportMatrixWinnerAndReplayExclusion checks the matrix picks a per-concept
// winner from measured cells only and never crowns a replay-only cell.
func TestReportMatrixWinnerAndReplayExclusion(t *testing.T) {
	rep := BuildReport(goldenGenerated, goldenRows())

	winners := map[Concept]string{}
	for _, c := range rep.Cells {
		if c.Winner {
			if prev, ok := winners[c.Concept]; ok {
				t.Errorf("concept %s has two winners: %s and %s", c.Concept, prev, c.Model)
			}
			winners[c.Concept] = c.Model
		}
		if c.Replay && c.Winner {
			t.Errorf("replay cell %s/%s marked winner", c.Model, c.Concept)
		}
	}
	if winners[ConceptCommitStamp] != "claude-opus-4-8" {
		t.Errorf("commit_stamp winner=%q, want claude-opus-4-8", winners[ConceptCommitStamp])
	}
	if _, ok := winners[ConceptHonesty]; ok {
		t.Errorf("honesty had a winner %q but its only cell is replay — must have none", winners[ConceptHonesty])
	}
}
