package supportmaturity

import "testing"

// TestNextActionGoldenPerRegime is the #1252 witness: a golden cell at EACH regime band
// produces the expected routed next-action — R0→idea-scout, R1→dispatch,
// R2→rsiloop+shipgate, R3→self-tax — with a non-empty directive bound to that loop.
func TestNextActionGoldenPerRegime(t *testing.T) {
	golden := []struct {
		cell     Rung
		regimeID string
		loop     Loop
		token    string
	}{
		{M0None, "R0", LoopIdeaScout, "idea-scout"},
		{M3Runs, "R1", LoopDispatch, "dispatch"},
		{M5Optimized, "R2", LoopRSIShipgate, "rsiloop+shipgate"},
		{M7BeyondSOTA, "R3", LoopSelfTax, "self-tax"},
	}
	for _, g := range golden {
		na := NextActionFor(g.cell)
		if na.Rung != g.cell {
			t.Fatalf("NextActionFor(%s).Rung = %s, want %s", g.cell, na.Rung, g.cell)
		}
		if na.RegimeID != g.regimeID {
			t.Fatalf("NextActionFor(%s).RegimeID = %q, want %q", g.cell, na.RegimeID, g.regimeID)
		}
		if na.Loop != g.loop {
			t.Fatalf("NextActionFor(%s).Loop = %s, want %s", g.cell, na.Loop, g.loop)
		}
		if na.Loop.String() != g.token {
			t.Fatalf("NextActionFor(%s).Loop token = %q, want %q", g.cell, na.Loop.String(), g.token)
		}
		if na.Action == "" {
			t.Fatalf("NextActionFor(%s) has empty Action directive", g.cell)
		}
		if na.Action != g.loop.Directive() {
			t.Fatalf("NextActionFor(%s).Action = %q, want the %s directive %q", g.cell, na.Action, g.loop, g.loop.Directive())
		}
	}
}

// TestEveryRungEmitsAValidRoutedAction asserts the routing is total and monotone over the
// closed ladder: every rung emits a valid loop, a non-empty regime id, an Action equal to
// its loop directive, and the regime band never decreases as the rung rises (R0→R1→R2→R3).
func TestEveryRungEmitsAValidRoutedAction(t *testing.T) {
	var prev string
	for i, r := range Rungs {
		na := NextActionFor(r)
		if !na.Loop.Valid() {
			t.Fatalf("NextActionFor(%s).Loop = %v is not a valid loop", r, na.Loop)
		}
		if na.RegimeID == "" {
			t.Fatalf("NextActionFor(%s) has empty RegimeID", r)
		}
		if na.Rung != r {
			t.Fatalf("NextActionFor(%s).Rung = %s, want %s", r, na.Rung, r)
		}
		if na.Action != na.Loop.Directive() {
			t.Fatalf("NextActionFor(%s).Action = %q, want loop directive %q", r, na.Action, na.Loop.Directive())
		}
		if i > 0 && na.RegimeID < prev {
			t.Fatalf("regime band not monotone: rung %s routes to %s after %s", r, na.RegimeID, prev)
		}
		prev = na.RegimeID
	}
}

// TestRoutingIsDeterministic asserts the routing is a pure function: NextActionFor returns
// an IDENTICAL action for the same rung every call — the issue's "the routing is
// deterministic" clause. NextAction is a comparable struct, so == is the exact check.
func TestRoutingIsDeterministic(t *testing.T) {
	for _, r := range Rungs {
		if a, b := NextActionFor(r), NextActionFor(r); a != b {
			t.Fatalf("NextActionFor(%s) not deterministic: %+v vs %+v", r, a, b)
		}
	}
}

// TestRegimeBandToLoopIsBijection asserts the four regime bands route to four DISTINCT
// loops — no two bands collapse onto the same loop, so a cell's regime names its next-loop
// unambiguously — and that exactly four bands (R0–R3) appear across the whole ladder.
func TestRegimeBandToLoopIsBijection(t *testing.T) {
	loopForBand := map[string]Loop{}
	for _, r := range Rungs {
		na := NextActionFor(r)
		if l, seen := loopForBand[na.RegimeID]; seen {
			if l != na.Loop {
				t.Fatalf("band %s routes to both %s and %s — routing must be consistent", na.RegimeID, l, na.Loop)
			}
			continue
		}
		loopForBand[na.RegimeID] = na.Loop
	}
	if len(loopForBand) != 4 {
		t.Fatalf("ladder routes to %d regime bands, want 4 (R0–R3)", len(loopForBand))
	}
	seenLoop := map[Loop]string{}
	for band, l := range loopForBand {
		if prev, dup := seenLoop[l]; dup {
			t.Fatalf("bands %s and %s both route to loop %s — band→loop must be a bijection", prev, band, l)
		}
		seenLoop[l] = band
	}
}

// TestLoopRender asserts every loop renders a distinct, non-empty wire token and a
// distinct, non-empty directive, and that an out-of-range loop is both not-Valid and
// rendered as the explicit unknown/empty forms — the closed-vocabulary guard the floor
// default (NextActionFor's out-of-range route) relies on.
func TestLoopRender(t *testing.T) {
	if len(Loops) != 4 {
		t.Fatalf("Loops roster has %d entries, want 4 (one per regime band)", len(Loops))
	}
	seenTok, seenDir := map[string]bool{}, map[string]bool{}
	for _, l := range Loops {
		if !l.Valid() {
			t.Fatalf("loop %v reports not Valid", l)
		}
		tok := l.String()
		if tok == "" || tok == "unknown" {
			t.Fatalf("loop %v has no token", l)
		}
		if seenTok[tok] {
			t.Fatalf("duplicate loop token %q", tok)
		}
		seenTok[tok] = true
		dir := l.Directive()
		if dir == "" {
			t.Fatalf("loop %q has no directive", tok)
		}
		if seenDir[dir] {
			t.Fatalf("duplicate loop directive %q", dir)
		}
		seenDir[dir] = true
	}
	bad := Loop(len(Loops))
	if bad.Valid() {
		t.Fatalf("out-of-range loop %d reports Valid", uint8(bad))
	}
	if got := bad.String(); got != "unknown" {
		t.Fatalf("out-of-range loop String() = %q, want \"unknown\"", got)
	}
	if got := bad.Directive(); got != "" {
		t.Fatalf("out-of-range loop Directive() = %q, want \"\"", got)
	}
}

// TestNextActionForOutOfRangeFloors asserts an unwitnessed (out-of-range) rung floors to
// the R0/idea-scout explore action — the honest "we can claim nothing higher, so explore"
// default, matching the From* lowerings' floor-to-M0 convention in supportmaturity.go.
func TestNextActionForOutOfRangeFloors(t *testing.T) {
	na := NextActionFor(Rung(len(Rungs)))
	if na.RegimeID != "R0" {
		t.Fatalf("NextActionFor(out-of-range).RegimeID = %q, want R0", na.RegimeID)
	}
	if na.Loop != LoopIdeaScout {
		t.Fatalf("NextActionFor(out-of-range).Loop = %s, want idea-scout", na.Loop)
	}
	if na.Action != LoopIdeaScout.Directive() {
		t.Fatalf("NextActionFor(out-of-range).Action = %q, want the idea-scout directive", na.Action)
	}
}
