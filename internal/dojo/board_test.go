package dojo

import "testing"

func TestBoardFromEpisodesGroupsAndGrades(t *testing.T) {
	band := DefaultCalibBand()
	// lever A: one calibrated (0.0 err) + one over-claim (big err) -> mean pulls its grade down.
	a1 := Score("s", Prediction{Lever: "alpha", Metric: "m1", Claimed: 1.0}, Outcome{Realized: 1.0, Measured: true, Sample: 10, Provenance: Observed}, band)
	a2 := Score("s", Prediction{Lever: "alpha", Metric: "m2", Claimed: 1.0}, Outcome{Realized: 0.4, Measured: true, Sample: 10, Provenance: Observed}, band)
	// lever B: one perfectly calibrated.
	b1 := Score("s", Prediction{Lever: "beta", Metric: "m1", Claimed: 0.0}, Outcome{Realized: 0.0, Measured: true, Sample: 5, Provenance: Witnessed}, band)
	// lever C: only an UNMEASURED episode -> grade n/a, sorts last.
	c1 := Score("s", Prediction{Lever: "gamma", Metric: "m1", Claimed: 1.0}, Outcome{Measured: false}, band)

	board := BoardFromEpisodes([]Episode{a1, a2, b1, c1})
	if board.Schema != BoardSchema {
		t.Fatalf("schema = %q, want %q", board.Schema, BoardSchema)
	}
	if len(board.Rows) != 3 {
		t.Fatalf("want 3 lever rows, got %d", len(board.Rows))
	}

	byLever := map[string]BoardRow{}
	for _, r := range board.Rows {
		byLever[r.Lever] = r
	}
	alpha := byLever["alpha"]
	if alpha.Episodes != 2 || alpha.Measured != 2 || alpha.Calibrated != 1 || alpha.OverClaim != 1 {
		t.Fatalf("alpha rollup wrong: %+v", alpha)
	}
	if alpha.WorstMetric != "m2" {
		t.Fatalf("alpha worst metric = %q, want m2 (the over-claim)", alpha.WorstMetric)
	}
	beta := byLever["beta"]
	if beta.Measured != 1 || beta.Calibrated != 1 || beta.Grade != "A" {
		t.Fatalf("beta rollup wrong: %+v", beta)
	}
	gamma := byLever["gamma"]
	if gamma.Measured != 0 || gamma.Unmeasured != 1 || gamma.Grade != gradeNA {
		t.Fatalf("gamma rollup wrong: %+v (an all-unmeasured lever must grade n/a)", gamma)
	}

	// Worst-first: alpha (measured, higher mean err) before beta (measured, 0 err);
	// gamma (all-unmeasured) sorts last.
	if board.Rows[0].Lever != "alpha" || board.Rows[1].Lever != "beta" || board.Rows[2].Lever != "gamma" {
		t.Fatalf("sort order wrong: %s, %s, %s", board.Rows[0].Lever, board.Rows[1].Lever, board.Rows[2].Lever)
	}
}

func TestBoardFromEpisodesEmpty(t *testing.T) {
	board := BoardFromEpisodes(nil)
	if len(board.Rows) != 0 {
		t.Fatalf("empty input should yield no rows, got %d", len(board.Rows))
	}
	if got := RenderBoard(board); got == "" {
		t.Fatal("RenderBoard should still produce a header + empty note")
	}
}
