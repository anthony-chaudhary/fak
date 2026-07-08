package modver

import "testing"

// ledgerFixture is three stamps of an append-only module-versions ledger,
// deliberately NOT in timestamp order (the middle stamp is written last) to
// witness that Trend orders by TS rather than trusting file order. Two modules
// move; one (internal/idle) is stamped once and never moves.
const ledgerFixture = `{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":10,"last_commit":"aaa","last_date":"2026-07-01T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":25,"last_commit":"ccc","last_date":"2026-07-03T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-02T00:00:00Z","module":"cmd/fak","kind":"cmd","rev":12,"last_commit":"bbb","last_date":"2026-07-02T00:00:00Z"}
{"schema":"fak-module-versions/1","ts":"2026-07-01T00:00:00Z","module":"internal/gateway","kind":"internal","rev":5,"last_commit":"g11","last_date":"2026-07-01T00:00:00Z","score":3}
{"schema":"fak-module-versions/1","ts":"2026-07-03T00:00:00Z","module":"internal/gateway","kind":"internal","rev":8,"last_commit":"g22","last_date":"2026-07-03T00:00:00Z","score":7.5}
{"schema":"fak-module-versions/1","ts":"2026-07-02T00:00:00Z","module":"internal/idle","kind":"internal","rev":2,"last_commit":"i00","last_date":"2026-07-02T00:00:00Z"}
not json — a fleet-written ledger has scars, must be skipped
`

func TestTrendFoldsLedger(t *testing.T) {
	rep := Trend([]byte(ledgerFixture))
	if rep.Rows != 6 {
		t.Errorf("Rows = %d, want 6 (the non-JSON scar is skipped)", rep.Rows)
	}
	if rep.Window != [2]string{"2026-07-01T00:00:00Z", "2026-07-03T00:00:00Z"} {
		t.Errorf("Window = %v, want the full 07-01..07-03 span", rep.Window)
	}
	if len(rep.Modules) != 3 {
		t.Fatalf("got %d modules, want 3: %+v", len(rep.Modules), rep.Modules)
	}
	// Default sort is rev-delta desc: cmd/fak (+15) leads internal/gateway (+3),
	// internal/idle (0) last — even though cmd/fak's biggest rev was written on
	// the middle (out-of-order) line, the window bounds are picked by timestamp.
	fak := rep.Modules[0]
	if fak.Module != "cmd/fak" || fak.FirstRev != 10 || fak.LastRev != 25 || fak.RevDelta != 15 {
		t.Errorf("cmd/fak trend = %+v, want first 10 last 25 delta 15", fak)
	}
	if fak.Stamps != 3 {
		t.Errorf("cmd/fak stamps = %d, want 3", fak.Stamps)
	}
	gw := rep.Modules[1]
	if gw.Module != "internal/gateway" || gw.RevDelta != 3 {
		t.Errorf("internal/gateway trend = %+v, want delta 3", gw)
	}
	if gw.ScoreDelta == nil || *gw.ScoreDelta != 4.5 {
		t.Errorf("internal/gateway score delta = %v, want 4.5", gw.ScoreDelta)
	}
	if idle := rep.Modules[2]; idle.Module != "internal/idle" || idle.RevDelta != 0 {
		t.Errorf("internal/idle should be last with delta 0, got %+v", idle)
	}
}

func TestTrendSelect(t *testing.T) {
	rep := Trend([]byte(ledgerFixture))

	only, err := rep.Select("internal/", "name", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(only.Modules) != 2 || only.Modules[0].Module != "internal/gateway" {
		t.Errorf("prefix+name select = %+v, want [gateway, idle]", only.Modules)
	}
	if only.Rows != 6 {
		t.Errorf("Select must preserve the full-ledger Rows=6, got %d", only.Rows)
	}

	top, err := rep.Select("", "delta", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(top.Modules) != 1 || top.Modules[0].Module != "cmd/fak" {
		t.Errorf("top-1 by delta = %+v, want [cmd/fak]", top.Modules)
	}

	if _, err := rep.Select("", "bogus", 0); err == nil {
		t.Error("Select must reject an unknown sort key")
	}
}
