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

// TestTrendVelocityAndDormantSplit is the #2469 witness over the whole ledger:
// rev velocity in revs/week, the per-module score series, and the movers /
// dormant partition. cmd/fak added 15 revs across a 2-day span (15 ÷ 2/7 weeks =
// 52.5/wk) and internal/gateway 3 (10.5/wk); internal/idle never moved, so it is
// the dormant list and nothing else is.
func TestTrendVelocityAndDormantSplit(t *testing.T) {
	rep := Trend([]byte(ledgerFixture))

	movers := rep.TopMovers(0)
	if len(movers) != 2 {
		t.Fatalf("got %d movers, want 2: %+v", len(movers), movers)
	}
	if movers[0].Module != "cmd/fak" || movers[0].RevsPerWeek != 52.5 {
		t.Errorf("fastest mover = %s at %g/wk, want cmd/fak at 52.5", movers[0].Module, movers[0].RevsPerWeek)
	}
	if movers[1].Module != "internal/gateway" || movers[1].RevsPerWeek != 10.5 {
		t.Errorf("second mover = %s at %g/wk, want internal/gateway at 10.5", movers[1].Module, movers[1].RevsPerWeek)
	}
	for _, m := range movers {
		if m.Dormant {
			t.Errorf("%s is a mover and must not be flagged dormant", m.Module)
		}
	}

	dormant := rep.DormantModules(0)
	if len(dormant) != 1 || dormant[0].Module != "internal/idle" {
		t.Fatalf("dormant = %+v, want just internal/idle", dormant)
	}
	if !dormant[0].Dormant || dormant[0].RevsPerWeek != 0 {
		t.Errorf("dormant internal/idle = %+v, want Dormant with 0 velocity", dormant[0])
	}
	if len(movers)+len(dormant) != len(rep.Modules) {
		t.Errorf("movers+dormant = %d, must partition the %d modules", len(movers)+len(dormant), len(rep.Modules))
	}

	// The series is the score curve, not just its endpoints: cmd/fak was
	// stamped three times, and the middle (out-of-order) stamp lands in the
	// middle of the series because the fold orders by timestamp.
	fak := movers[0]
	if len(fak.Series) != 3 {
		t.Fatalf("cmd/fak series = %+v, want 3 points", fak.Series)
	}
	if fak.Series[1].TS != "2026-07-02T00:00:00Z" || fak.Series[1].Rev != 12 {
		t.Errorf("cmd/fak series[1] = %+v, want the 07-02 r12 stamp", fak.Series[1])
	}
	gw := movers[1]
	if len(gw.Series) != 2 || gw.Series[0].Score == nil || *gw.Series[0].Score != 3 || *gw.Series[1].Score != 7.5 {
		t.Errorf("internal/gateway score series = %+v, want 3 then 7.5", gw.Series)
	}
}

// TestTrendSinceBaselinesAndSurfacesDormant pins the --since semantics. The
// ledger is delta-encoded, so a module with ONE row inside the window still
// grew: the fold keeps its last stamp at or before the bound as the baseline
// (cmd/fak reads r12→r25, not r25→r25), and a module with NO row inside the
// window is reported dormant at its baseline rather than dropped.
func TestTrendSinceBaselinesAndSurfacesDormant(t *testing.T) {
	rep := TrendSince([]byte(ledgerFixture), "2026-07-03T00:00:00Z")

	if rep.Rows != 6 {
		t.Errorf("Rows = %d, want all 6 parseable rows regardless of the bound", rep.Rows)
	}
	if rep.Since != "2026-07-03T00:00:00Z" {
		t.Errorf("Since = %q, want the bound echoed back", rep.Since)
	}
	if rep.Window != [2]string{"2026-07-03T00:00:00Z", "2026-07-03T00:00:00Z"} {
		t.Errorf("Window = %v, want only the in-window stamps", rep.Window)
	}

	byName := map[string]ModuleTrend{}
	for _, m := range rep.Modules {
		byName[m.Module] = m
	}
	if len(byName) != 3 {
		t.Fatalf("got %d modules, want all 3 still reported: %+v", len(byName), rep.Modules)
	}
	fak := byName["cmd/fak"]
	if fak.FirstRev != 12 || fak.LastRev != 25 || fak.RevDelta != 13 || fak.Stamps != 1 {
		t.Errorf("cmd/fak since 07-03 = %+v, want baseline r12 → r25 delta 13 over 1 stamp", fak)
	}
	if fak.RevsPerWeek != 91 { // 13 revs across the 1-day baseline→last span
		t.Errorf("cmd/fak velocity = %g/wk, want 91", fak.RevsPerWeek)
	}
	idle := byName["internal/idle"]
	if !idle.Dormant || idle.Stamps != 0 || idle.LastRev != 2 || idle.LastTS != "2026-07-02T00:00:00Z" {
		t.Errorf("internal/idle since 07-03 = %+v, want dormant at its 07-02 baseline with 0 in-window stamps", idle)
	}
}

// TestTrendSelectModuleAndCap pins the --module focus and the per-section --top
// cap: capping must keep a dormant module visible, which a single combined
// truncation under the rev-delta sort would not.
func TestTrendSelectModuleAndCap(t *testing.T) {
	rep := Trend([]byte(ledgerFixture))

	one := rep.SelectModule("internal/gateway")
	if len(one.Modules) != 1 || one.Modules[0].Module != "internal/gateway" {
		t.Errorf("SelectModule = %+v, want just internal/gateway", one.Modules)
	}
	if one.Rows != 6 {
		t.Errorf("SelectModule must preserve the full-ledger Rows=6, got %d", one.Rows)
	}
	if none := rep.SelectModule("internal/nope"); len(none.Modules) != 0 {
		t.Errorf("an unknown module is an empty answer, not %+v", none.Modules)
	}

	capped := rep.Cap(1)
	if len(capped.Modules) != 2 {
		t.Fatalf("Cap(1) = %+v, want 1 mover + 1 dormant", capped.Modules)
	}
	if capped.Modules[0].Module != "cmd/fak" || capped.Modules[1].Module != "internal/idle" {
		t.Errorf("Cap(1) = %+v, want [cmd/fak, internal/idle]", capped.Modules)
	}
	if same := rep.Cap(0); len(same.Modules) != 3 {
		t.Errorf("Cap(0) must be a no-op, got %+v", same.Modules)
	}
}

func TestNormalizeSince(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"", ""},
		{"2026-07-03", "2026-07-03T00:00:00Z"},
		{"2026-07-03T18:43:00Z", "2026-07-03T18:43:00Z"},
		// An offset is converted to UTC — the fold compares timestamps as text
		// against a Z-normalized ledger, so a raw "+02:00" would mis-bound it.
		{"2026-07-03T20:43:00+02:00", "2026-07-03T18:43:00Z"},
	} {
		got, err := NormalizeSince(tc.in)
		if err != nil {
			t.Errorf("NormalizeSince(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("NormalizeSince(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if _, err := NormalizeSince("last tuesday"); err == nil {
		t.Error("NormalizeSince must reject an unparseable bound rather than silently ignore it")
	}
}
