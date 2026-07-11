package modver

import (
	"testing"
	"time"
)

// weeklyFixture is an append-only module-versions ledger spanning a baseline
// (pre-window) era and the trailing week. It is deliberately NOT in timestamp
// order (cmd/fak's window-close row is written before its baseline) to witness
// that FoldWeekly bounds the window by ts, not file order. It exercises every
// digest bucket: a rev mover with a baseline (cmd/fak), a rev+score mover
// (internal/gateway), a module born inside the window (internal/newthing), a
// module that existed before the window and never moved in it (internal/quiet),
// and a ledger-seen module dropped from the live set (internal/dead). One scar
// line must be skipped.
const weeklyFixture = `{"schema":"fak-module-versions/1","ts":"2026-07-05T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":25,"last_commit":"ccc"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":10,"last_commit":"aaa"}
{"schema":"fak-module-versions/1","ts":"2026-07-02T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g11","score":3}
{"schema":"fak-module-versions/1","ts":"2026-07-06T00:00:00Z","module":"internal/gateway","kind":"internal","rev":8,"last_commit":"g22","score":7.5}
{"schema":"fak-module-versions/1","ts":"2026-07-04T00:00:00Z","module":"internal/newthing","kind":"internal","rev":1,"last_commit":"n11"}
{"schema":"fak-module-versions/1","ts":"2026-07-07T00:00:00Z","module":"internal/newthing","kind":"internal","rev":3,"last_commit":"n22"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/quiet","kind":"internal","rev":2,"last_commit":"q00"}
{"schema":"fak-module-versions/1","ts":"2026-06-20T00:00:00Z","module":"internal/dead","kind":"internal","rev":4,"last_commit":"d00"}
not json — a fleet-written ledger has scars, must be skipped
`

func TestFoldWeeklyDigest(t *testing.T) {
	now := time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC) // window opens 2026-07-03
	live := map[string]bool{
		"cmd/fak":           true,
		"internal/gateway":  true,
		"internal/newthing": true,
		"internal/quiet":    true,
		// internal/dead deliberately absent → retired.
	}
	d := FoldWeekly([]byte(weeklyFixture), live, now)

	if d.Schema != WeeklyDigestSchema {
		t.Errorf("Schema = %q, want %q", d.Schema, WeeklyDigestSchema)
	}
	if !d.OK || d.Verdict != "OK" {
		t.Errorf("a measured week must be OK, got ok=%v verdict=%q", d.OK, d.Verdict)
	}
	if d.WindowStart != "2026-07-03T00:00:00Z" || d.WindowEnd != "2026-07-10T00:00:00Z" {
		t.Errorf("window = [%s, %s], want [2026-07-03, 2026-07-10]", d.WindowStart, d.WindowEnd)
	}
	// In-window parseable rows: cmd/fak@07-05, gateway@07-06, newthing@07-04, newthing@07-07.
	if d.LedgerRows != 4 {
		t.Errorf("LedgerRows = %d, want 4 (rows whose ts falls in the window)", d.LedgerRows)
	}

	// Top movers: cmd/fak (+15) leads internal/gateway (+3); the window bounds come
	// from ts, so cmd/fak's out-of-order 07-05 close still reads r10 -> r25.
	if len(d.TopMovers) != 2 {
		t.Fatalf("got %d top movers, want 2: %+v", len(d.TopMovers), d.TopMovers)
	}
	fak := d.TopMovers[0]
	if fak.Module != "cmd/fak" || fak.StartRev != 10 || fak.EndRev != 25 || fak.RevDelta != 15 || fak.Direction != "up" {
		t.Errorf("cmd/fak mover = %+v, want r10->r25 delta+15 up", fak)
	}
	if gw := d.TopMovers[1]; gw.Module != "internal/gateway" || gw.RevDelta != 3 {
		t.Errorf("internal/gateway mover = %+v, want delta 3", d.TopMovers[1])
	}

	// Score movers: only internal/gateway carries a score on both bounds (+4.5).
	if len(d.ScoreMovers) != 1 || d.ScoreMovers[0].Module != "internal/gateway" {
		t.Fatalf("score movers = %+v, want [internal/gateway]", d.ScoreMovers)
	}
	if sd := d.ScoreMovers[0].ScoreDelta; sd == nil || *sd != 4.5 {
		t.Errorf("internal/gateway score delta = %v, want 4.5", d.ScoreMovers[0].ScoreDelta)
	}

	// Born this week: internal/newthing (first row 07-04, inside the window). It has
	// no baseline, so it must NOT appear among the rev movers.
	if len(d.NewModules) != 1 || d.NewModules[0] != "internal/newthing" {
		t.Errorf("new modules = %v, want [internal/newthing]", d.NewModules)
	}
	for _, m := range d.TopMovers {
		if m.Module == "internal/newthing" {
			t.Errorf("a born-this-week module must not be double-counted as a rev mover: %+v", m)
		}
	}

	// Retired: internal/dead is in the ledger but absent from the live set.
	if len(d.Deaths) != 1 || d.Deaths[0] != "internal/dead" {
		t.Errorf("deaths = %v, want [internal/dead]", d.Deaths)
	}

	// internal/quiet existed before the window and never moved inside it — no bucket.
	for _, m := range d.TopMovers {
		if m.Module == "internal/quiet" {
			t.Errorf("a module that did not move in-window must not be a mover: %+v", m)
		}
	}

	if got := d.Render(); got == "" {
		t.Error("Render must produce a non-empty section for a measured week")
	}
}

// TestFoldWeeklyUnmeasured witnesses the one INCOMPLETE case: an empty/all-scar
// ledger folds to the unmeasured token and an ACTION verdict, and a nil live set
// reports no deaths rather than guessing.
func TestFoldWeeklyUnmeasured(t *testing.T) {
	now := time.Date(2026, 7, 10, 0, 0, 0, 0, time.UTC)

	empty := FoldWeekly([]byte("not json\n\n"), nil, now)
	if empty.OK || empty.Verdict != "ACTION" {
		t.Errorf("an all-scar ledger must be INCOMPLETE, got ok=%v verdict=%q", empty.OK, empty.Verdict)
	}
	if empty.Finding != weeklyUnmeasured {
		t.Errorf("Finding = %q, want the unmeasured token %q", empty.Finding, weeklyUnmeasured)
	}

	// nil live set: movers still fold, but deaths are not guessed.
	d := FoldWeekly([]byte(weeklyFixture), nil, now)
	if len(d.Deaths) != 0 {
		t.Errorf("with a nil live set deaths must be empty (undecidable), got %v", d.Deaths)
	}
	if len(d.TopMovers) != 2 {
		t.Errorf("movers must still fold with a nil live set, got %d", len(d.TopMovers))
	}
}
