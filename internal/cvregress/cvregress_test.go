package cvregress

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// row builds a minimal multi-turn ledger row for the fold.
func row(date string, turns, prompt, reused uint64, unixMillis int64) cachevalueledger.Row {
	return cachevalueledger.Row{
		Schema:       cachevalueledger.Schema,
		Date:         date,
		SessionType:  "guard",
		Provider:     "fak",
		Mechanism:    "kv_prefix_reuse",
		Context:      "test",
		Turns:        turns,
		PromptTokens: prompt,
		ReusedTokens: reused,
		UnixMillis:   unixMillis,
	}
}

// healthyCorpus mirrors the real docs/nightrun/cache-value.jsonl multi-turn sessions
// (44.9%–80.8% hit, write-amp 0.24–1.23) that the DefaultBaseline must keep green.
func healthyCorpus() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		row("2026-07-01", 5, 413, 260, 1),
		row("2026-07-02", 6, 433, 305, 2),
		row("2026-07-03", 10, 1356, 1096, 3),
		row("2026-07-04", 9, 1293, 1022, 4),
		row("2026-07-05", 8, 998, 758, 5),
		row("2026-07-06", 28, 1370, 615, 6), // churny but still 44.9% hit / 1.23 write-amp
	}
}

func TestFold_HealthyCorpusClears(t *testing.T) {
	rep := Fold(healthyCorpus(), DefaultBaseline())
	if rep.Verdict != "OK" || !rep.OK {
		t.Fatalf("healthy corpus should be OK/green, got %s ok=%v: %s", rep.Verdict, rep.OK, rep.Finding)
	}
	if rep.Scored != 6 {
		t.Fatalf("all 6 multi-turn sessions should score, got %d", rep.Scored)
	}
	if len(rep.Regressions) != 0 {
		t.Fatalf("healthy corpus should have no regressions, got %d: %+v", len(rep.Regressions), rep.Regressions)
	}
}

func TestFold_HitRateFloorFires(t *testing.T) {
	rows := append(healthyCorpus(), row("2026-07-07", 12, 2000, 400, 7)) // 20% hit — below floor
	rep := Fold(rows, DefaultBaseline())
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("a 20%% hit session must regress, got %s ok=%v: %s", rep.Verdict, rep.OK, rep.Finding)
	}
	if len(rep.Regressions) != 1 {
		t.Fatalf("exactly the one bad session should flag, got %d", len(rep.Regressions))
	}
	if got := rep.Regressions[0]; !strings.Contains(got.Reason, "hit-rate") {
		t.Fatalf("reason should name the hit-rate tripwire, got %q", got.Reason)
	}
}

func TestFold_WriteAmpCeilingFiresIndependently(t *testing.T) {
	// A baseline whose write-amp ceiling is the binding constraint while hit-rate passes,
	// isolating the write-amp path from the hit-rate path.
	base := Baseline{HitRatePctFloor: 10, WriteAmpCeiling: 0.5, MinPromptTokens: 100}
	// 50% hit -> write-amp (1000-500)/500 = 1.0 > 0.5 ceiling, but 50% > 10% floor.
	rep := Fold([]cachevalueledger.Row{row("2026-07-08", 4, 1000, 500, 8)}, base)
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("write-amp 1.0 over a 0.5 ceiling must regress, got %s: %s", rep.Verdict, rep.Finding)
	}
	got := rep.Regressions[0]
	if strings.Contains(got.Reason, "hit-rate") {
		t.Fatalf("hit-rate 50%% is above the 10%% floor and must NOT be named, got %q", got.Reason)
	}
	if !strings.Contains(got.Reason, "write-amp") {
		t.Fatalf("reason should name the write-amp tripwire, got %q", got.Reason)
	}
}

// TestFold_PinnedCatchesFleetWideSlide is the load-bearing distinction from
// cachevaluereport.foldSessionOutliers: when the WHOLE fleet slides down together, a
// self-relative session-median baseline moves with it and flags nothing, but the pinned floor
// still trips. Here every session sits at ~30% hit — below the 40% pinned floor — so all of
// them flag, which a median (=30%) check would miss entirely.
func TestFold_PinnedCatchesFleetWideSlide(t *testing.T) {
	rows := []cachevalueledger.Row{
		row("2026-07-01", 5, 1000, 300, 1),
		row("2026-07-02", 6, 1200, 360, 2),
		row("2026-07-03", 7, 900, 270, 3),
		row("2026-07-04", 8, 1100, 330, 4),
	}
	rep := Fold(rows, DefaultBaseline())
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("a uniform fleet-wide slide to 30%% must regress against the pin, got %s: %s", rep.Verdict, rep.Finding)
	}
	if len(rep.Regressions) != 4 {
		t.Fatalf("all 4 slid sessions should flag against the pin, got %d", len(rep.Regressions))
	}
}

func TestFold_SingleTurnExcluded(t *testing.T) {
	// Single-turn cold runs have no previous turn to reuse from; folding them in would
	// manufacture a false hit-rate regression. They must be skipped, not scored.
	rows := []cachevalueledger.Row{
		row("2026-07-01", 1, 10, 0, 1),
		row("2026-07-02", 1, 11, 0, 2),
		row("2026-07-03", 1, 5000, 0, 3), // large but single-turn: still excluded
	}
	rep := Fold(rows, DefaultBaseline())
	if rep.Verdict != "INSUFFICIENT" || !rep.OK {
		t.Fatalf("an all-single-turn corpus must fall open INSUFFICIENT, got %s ok=%v", rep.Verdict, rep.OK)
	}
	if rep.Scored != 0 || rep.Skipped != 3 {
		t.Fatalf("expected scored=0 skipped=3, got scored=%d skipped=%d", rep.Scored, rep.Skipped)
	}
}

func TestFold_MinPromptTokensFilters(t *testing.T) {
	// A tiny multi-turn session with terrible hit-rate is below the noise floor and must be
	// dropped, not flagged; a large one at the same hit-rate is scored and flags.
	base := DefaultBaseline()
	tiny := row("2026-07-01", 3, 100, 5, 1)   // 5% hit but only 100 prompt tokens -> dropped
	big := row("2026-07-02", 3, 2000, 100, 2) // 5% hit, 2000 prompt tokens -> scored + flagged

	repTiny := Fold([]cachevalueledger.Row{tiny}, base)
	if repTiny.Scored != 0 || repTiny.Skipped != 1 || repTiny.Verdict != "INSUFFICIENT" {
		t.Fatalf("tiny session must be dropped as noise, got %+v", repTiny)
	}
	repBig := Fold([]cachevalueledger.Row{big}, base)
	if repBig.Verdict != "REGRESSED" || repBig.Scored != 1 {
		t.Fatalf("big low-hit session must score and regress, got %+v", repBig)
	}
}

func TestFold_RegressionsSortedWorstFirst(t *testing.T) {
	rows := []cachevalueledger.Row{
		row("2026-07-01", 4, 1000, 350, 1), // 35% hit
		row("2026-07-02", 4, 1000, 100, 2), // 10% hit (worst)
		row("2026-07-03", 4, 1000, 390, 3), // 39% hit
	}
	rep := Fold(rows, DefaultBaseline())
	if len(rep.Regressions) != 3 {
		t.Fatalf("all 3 below-floor sessions should flag, got %d", len(rep.Regressions))
	}
	if rep.Regressions[0].HitRatePct >= rep.Regressions[1].HitRatePct ||
		rep.Regressions[1].HitRatePct >= rep.Regressions[2].HitRatePct {
		t.Fatalf("regressions must be worst(lowest-hit)-first, got %.1f, %.1f, %.1f",
			rep.Regressions[0].HitRatePct, rep.Regressions[1].HitRatePct, rep.Regressions[2].HitRatePct)
	}
}

func TestFold_EmptyIsInsufficientNotFailure(t *testing.T) {
	rep := Fold(nil, DefaultBaseline())
	if rep.Verdict != "INSUFFICIENT" || !rep.OK {
		t.Fatalf("empty corpus must be INSUFFICIENT and green, got %s ok=%v", rep.Verdict, rep.OK)
	}
}

func TestScoreLedgerFile_SeededLedger(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cache-value.jsonl")

	rows := append(healthyCorpus(), row("2026-07-07", 12, 2000, 300, 7)) // 15% hit — regresses
	var lines []string
	for _, r := range rows {
		line, err := cachevalueledger.AppendLedgerLine(r)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		lines = append(lines, line)
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write seeded ledger: %v", err)
	}

	rep := ScoreLedgerFile(path, DefaultBaseline())
	if rep.Verdict != "REGRESSED" || rep.OK {
		t.Fatalf("seeded below-floor ledger must regress, got %s: %s", rep.Verdict, rep.Finding)
	}
	if len(rep.Regressions) != 1 || rep.Regressions[0].Date != "2026-07-07" {
		t.Fatalf("only the seeded 2026-07-07 session should flag, got %+v", rep.Regressions)
	}

	// A missing file falls open INSUFFICIENT (never errors).
	missing := ScoreLedgerFile(filepath.Join(dir, "nope.jsonl"), DefaultBaseline())
	if missing.Verdict != "INSUFFICIENT" || !missing.OK {
		t.Fatalf("missing ledger must fall open INSUFFICIENT, got %s ok=%v", missing.Verdict, missing.OK)
	}
}

func TestDefaultBaseline_KeepsRealCorpusGreen(t *testing.T) {
	// Guard against a future tightening of DefaultBaseline silently reddening the real corpus.
	base := DefaultBaseline()
	if base.HitRatePctFloor <= 0 || base.HitRatePctFloor >= 100 {
		t.Fatalf("hit-rate floor must be a sane percentage, got %.1f", base.HitRatePctFloor)
	}
	if base.WriteAmpCeiling <= 0 {
		t.Fatalf("write-amp ceiling must be positive, got %.2f", base.WriteAmpCeiling)
	}
	if rep := Fold(healthyCorpus(), base); !rep.OK {
		t.Fatalf("DefaultBaseline must keep the real healthy corpus green, got %s: %s", rep.Verdict, rep.Finding)
	}
}
