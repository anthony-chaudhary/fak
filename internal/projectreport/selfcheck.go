package projectreport

// selfcheck.go is the deterministic, no-I/O proof behind `fak project selfcheck`: it
// exercises the fold's verdict ladder and the ledger/trend round-trip against fixed
// oracles, so a green report on the weekly cadence tick is a real signal — the fold and
// trend that produced it are themselves proven, not a tool that always passes. It is the
// project twin of milestonereport.TriageSelfcheck (which delegates to the shared
// trendreport proof); this one proves THIS package's fold + ledger invariants directly,
// since the project fold carries its own OK/ACTION/UNMEASURED vocabulary.

import "fmt"

// Selfcheck proves the fold + ledger + trend invariants deterministically. It returns
// the first violated invariant, or nil when every oracle holds.
func Selfcheck() error {
	opts := FoldOpts{Commit: "deadbeef", GeneratedAt: "2026-07-11T00:00:00Z", Date: "2026-07-11"}

	// 1. A fully-classified board is OK and carries no unclassified drift.
	healthy := Fold([]Item{
		{Issue: 1, Status: "In progress", Generation: "now", Priority: "P1"},
		{Issue: 2, Status: "Todo", Generation: "next", Priority: "P2"},
	}, opts)
	if !healthy.OK || healthy.Verdict != "OK" || len(healthy.Unclassified) != 0 {
		return fmt.Errorf("healthy board must be OK with no drift, got verdict=%q ok=%v unclassified=%v", healthy.Verdict, healthy.OK, healthy.Unclassified)
	}

	// 2. An item missing Status or Generation is ACTION drift, not OK.
	drift := Fold([]Item{
		{Issue: 1, Status: "Todo", Generation: "now"},
		{Issue: 2, Status: "Todo"}, // no generation
	}, opts)
	if drift.OK || drift.Verdict != "ACTION" || len(drift.Unclassified) != 1 || drift.Unclassified[0] != 2 {
		return fmt.Errorf("a board with an unclassified item must be ACTION, got verdict=%q unclassified=%v", drift.Verdict, drift.Unclassified)
	}

	// 3. An unreachable board is a VISIBLE UNMEASURED verdict, never a silent green.
	un := Unmeasured("", opts)
	if un.Measured || un.Verdict != "UNMEASURED" {
		return fmt.Errorf("an unreachable board must fold to a visible UNMEASURED, got measured=%v verdict=%q", un.Measured, un.Verdict)
	}

	// 4. The ledger row normalizes the Generation distribution into stable horizon columns.
	row := RowFromReport(healthy)
	if row.Now != 1 || row.Next != 1 || row.Total != 2 || row.Unclassified != 0 || row.Schema != LedgerSchema {
		return fmt.Errorf("ledger row mis-projected: %+v", row)
	}

	// 5. Trend directions: first tick is "new"; shedding an unclassified item is
	//    "improved"; gaining one is "regressed".
	first := TrendVsLast(row, nil)
	if first.Direction != "new" {
		return fmt.Errorf("first tick must be new, got %q", first.Direction)
	}
	base := LedgerRow{Date: "2026-07-10", GeneratedAt: "2026-07-10T00:00:00Z", Total: 4, Unclassified: 2}
	improved := TrendVsLast(LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 4, Unclassified: 1}, []LedgerRow{base})
	if improved.Direction != "improved" {
		return fmt.Errorf("shedding an unclassified item must be improved, got %q", improved.Direction)
	}
	regressed := TrendVsLast(LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 4, Unclassified: 3}, []LedgerRow{base})
	if regressed.Direction != "regressed" {
		return fmt.Errorf("gaining an unclassified item must be regressed, got %q", regressed.Direction)
	}

	// 6. The ledger line marshals and re-parses to the same dated row (round-trip).
	line, err := AppendLedgerLine(row)
	if err != nil {
		return fmt.Errorf("append ledger line: %w", err)
	}
	back := ParseLedger(line)
	if len(back) != 1 || back[0].Total != row.Total || back[0].Now != row.Now {
		return fmt.Errorf("ledger round-trip lost data: wrote %+v, parsed %+v", row, back)
	}
	return nil
}
