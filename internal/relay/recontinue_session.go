// Rung H6 (issue #1899): the Recontinue wiring — the "later floor" driver.go's header
// reserves, supplied through the LegConfig.Recontinue hook. A relay leg IS a
// generation, so a rotation reuses session's existing budget-reset lineage
// (ContinuationID / Generation / ParentTrace) instead of minting a parallel one: the
// closing leg is left Stopped with its relay reason preserved for audit, and the
// successor is minted under a fresh trace with ParentTrace = the closing leg and
// Generation = parent+1. This rung adds NO new lineage type (issue: "Out of scope:
// No new lineage type; reuse ContinuationID/Generation") — it only binds the
// session-free driver seam to the live session.Table.
package relay

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// DefaultRotationReason is the reason a rotation stops the closing leg under when the baton's tombstone names none.
const DefaultRotationReason = "RELAY_ROTATED"

// RecontinueHook returns the LegConfig.Recontinue hook bound to tbl: one rotation
// drives the live session re-arm verb. fresh is the budget the successor leg is
// re-armed with. It is the H6 binding of the driver's Recontinue seam.
func RecontinueHook(tbl *session.Table, fresh session.Budget) func(Baton) (string, error) {
	return func(b Baton) (string, error) {
		parent := b.ParentTrace
		if parent == "" {
			return "", fmt.Errorf("relay: recontinue: baton carries no ParentTrace (the closing leg's trace)")
		}
		reason := b.Tombstone.Reason
		if reason == "" {
			reason = DefaultRotationReason
		}
		// Leave the closing leg Stopped with its relay reason. An already-terminal leg
		// is left exactly as it is (Transition refuses a terminal session).
		tbl.Transition(parent, session.Stopped, reason)
		// Reuse the continuation id the parent already minted; derive one only if absent.
		parentSt := tbl.Get(parent)
		child := parentSt.ContinuationID
		if child == "" {
			child = session.ContinuationID(parent, parentSt.Rev)
		}
		return tbl.Recontinue(parent, child, fresh).TraceID, nil
	}
}
