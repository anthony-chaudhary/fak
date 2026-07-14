package milestonereport

import (
	"strings"
	"testing"
)

// velocityLedgerFixture is a hermetic module-versions ledger (schema
// fak-module-versions/1): two movers stamped twice each (a first + last window
// bound so RevDelta > 0) and one DORMANT lane stamped twice at the same rev, to
// prove a lane that did not move is counted in the module total but kept out of
// the rendered top-movers list.
const velocityLedgerFixture = `{"schema":"fak-module-versions/1","ts":"2026-06-20T00:00:00Z","head":"aaa","module":"internal/modver","kind":"go","rev":3,"version":"v0.0.3","last_commit":"aaa111"}
{"schema":"fak-module-versions/1","ts":"2026-06-29T00:00:00Z","head":"bbb","module":"internal/modver","kind":"go","rev":7,"version":"v0.0.7","last_commit":"bbb222"}
{"schema":"fak-module-versions/1","ts":"2026-06-20T00:00:00Z","head":"aaa","module":"internal/milestonereport","kind":"go","rev":1,"version":"v0.0.1","last_commit":"aaa111"}
{"schema":"fak-module-versions/1","ts":"2026-06-29T00:00:00Z","head":"bbb","module":"internal/milestonereport","kind":"go","rev":2,"version":"v0.0.2","last_commit":"ccc333"}
{"schema":"fak-module-versions/1","ts":"2026-06-20T00:00:00Z","head":"aaa","module":"internal/dormant","kind":"go","rev":5,"version":"v0.0.5","last_commit":"aaa111"}
{"schema":"fak-module-versions/1","ts":"2026-06-29T00:00:00Z","head":"bbb","module":"internal/dormant","kind":"go","rev":5,"version":"v0.0.5","last_commit":"aaa111"}
`

// TestInterpretVelocityFoldsLedgerMovers proves the pure fold: three modules seen,
// two movers, the summed rev delta, and the top list ordered by delta with the
// dormant lane excluded — the numbers the render then carries.
func TestInterpretVelocityFoldsLedgerMovers(t *testing.T) {
	v := InterpretVelocity([]byte(velocityLedgerFixture))
	if v.Err != "" || !v.OK {
		t.Fatalf("velocity should measure a populated ledger: err=%q ok=%v", v.Err, v.OK)
	}
	if v.Modules != 3 || v.Movers != 2 {
		t.Fatalf("velocity = %d module(s) / %d mover(s), want 3 / 2", v.Modules, v.Movers)
	}
	if v.LedgerRows != 6 {
		t.Fatalf("velocity folded %d ledger rows, want 6", v.LedgerRows)
	}
	if v.TotalRevDelta != 5 { // 4 (modver) + 1 (milestonereport) + 0 (dormant)
		t.Fatalf("total rev delta = %d, want 5", v.TotalRevDelta)
	}
	if len(v.Rows) != 2 {
		t.Fatalf("top-mover rows = %d, want 2 (dormant excluded)", len(v.Rows))
	}
	if v.Rows[0].Module != "internal/modver" || v.Rows[0].RevDelta != 4 || v.Rows[0].LastRev != 7 {
		t.Fatalf("top mover = %+v, want internal/modver +4 -> r7", v.Rows[0])
	}
	for _, row := range v.Rows {
		if row.Module == "internal/dormant" {
			t.Fatalf("a dormant lane (0 rev movement) must not appear in the top-mover list: %+v", row)
		}
	}
}

// TestRenderShowsModuleVelocityBesideEpics is the issue's named witness (#2494):
// the human render carries the module-velocity lens, and it renders BESIDE the
// epic roadmap (after the discrete-epics section), so an operator reads "issues
// closing" and "code moving" on one card.
func TestRenderShowsModuleVelocityBesideEpics(t *testing.T) {
	specs := []EpicSpec{
		{Number: 1315, Title: "native harness", Generation: "now"}, // discrete
		{Number: 1010, Title: "GLM kernel", Generation: "next"},    // ongoing
	}
	counts := []EpicCounts{
		{Number: 1315, Closed: 3, Total: 4, Source: "label"},
		{Number: 1010, Closed: 7, Total: 10, Source: "label"},
	}
	r := Fold(goodMaturity(), InterpretEpics(specs, counts, ""), FoldOpts{Date: "2026-06-29", Commit: "abc"})
	r = r.WithVelocity(InterpretVelocity([]byte(velocityLedgerFixture)))
	out := Render(r)

	for _, want := range []string{
		"module velocity:",
		"2 mover(s) of 3 module(s)",
		"internal/modver",
		"+4 rev(s) -> r7",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
	// The lens renders BESIDE the epics — after the discrete-epics section, not before it.
	if ve, ep := strings.Index(out, "module velocity:"), strings.Index(out, "discrete epics"); ve < ep {
		t.Fatalf("module velocity must render beside/after the epics (velocity@%d, epics@%d)\n%s", ve, ep, out)
	}
	// A dormant lane is counted but never rendered as a mover.
	if strings.Contains(out, "internal/dormant") {
		t.Fatalf("a dormant lane must not render in the velocity movers list\n%s", out)
	}
}

// TestRenderVelocityUnmeasuredIsHonest proves an empty ledger renders a single
// honest "unmeasured" line rather than a fabricated zero, matching the
// maturity/roadmap honesty rule.
func TestRenderVelocityUnmeasuredIsHonest(t *testing.T) {
	v := InterpretVelocity(nil)
	if v.OK || v.Err == "" {
		t.Fatalf("an empty ledger must be an errored lens, got %+v", v)
	}
	r := Fold(goodMaturity(), Epics{}, FoldOpts{Date: "2026-06-29", Commit: "abc"}).WithVelocity(v)
	out := Render(r)
	if !strings.Contains(out, "module velocity: unmeasured") {
		t.Fatalf("unmeasured velocity must render honestly\n%s", out)
	}
}
