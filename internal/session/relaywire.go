// Rung H6 (issue #1899): the session-side wiring of the relay driver's Recontinue
// seam. internal/relay's driver (driver.go) deliberately never imports this package —
// its LegConfig.Recontinue hook is "wired in by a later floor", and this file is that
// floor. A relay leg IS a session generation, so a rotation reuses the table's
// existing budget-reset lineage verb (Recontinue: ContinuationID / Generation /
// ParentTrace) instead of minting a parallel lineage type. The baton stays an opaque
// type parameter so this package never imports internal/relay either — the seam is
// cycle-proof in BOTH directions, matching how armtriggers.go takes its axis numbers
// as bare scalars.
package session

import "fmt"

// RelayRecontinueHook binds tbl to relay's LegConfig.Recontinue seam: instantiated
// with B = relay.Baton it returns exactly `func(relay.Baton) (successorTrace string,
// err error)` — the driver's hook type — so it drops straight into LegConfig. When a
// rotation fires, the closure asks mint for the lineage pair — the closing leg's
// trace (the baton's parent_trace) and the fresh child trace to re-arm under
// (canonically the ContinuationID the context drain already minted) — then re-arms
// the child via tbl.Recontinue with the fresh budget: Generation = parent+1,
// ParentTrace = the closing leg, and the parent's terminal record left exactly as
// the drain wrote it (Recontinue never revives a Stopped parent). B is opaque here
// on purpose (no internal/relay import); the wiring site, which holds the concrete
// baton type, supplies the two-line mint.
func RelayRecontinueHook[B any](tbl *Table, fresh Budget, mint func(b B) (parent, child string)) func(B) (string, error) {
	return func(b B) (string, error) {
		parent, child := mint(b)
		if parent == "" {
			return "", fmt.Errorf("session: relay recontinue: mint returned no parent trace for the closing leg")
		}
		if child == "" {
			return "", fmt.Errorf("session: relay recontinue: mint returned no child trace for parent %q", parent)
		}
		return tbl.Recontinue(parent, child, fresh).TraceID, nil
	}
}
