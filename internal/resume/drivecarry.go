package resume

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// DriveCarryRow is the append-only transcript-UUID-keyed projection of the
// remaining drive budget. Scalars keep this foundation leaf independent of
// session; the objective fields carry only the pin's SAFE extractive triple
// (PinID/Text/Digest, ctxplan/objective.go) — never sealed transcript bytes —
// so the leaf stays free of internal/session while still carrying WHAT the work
// is across a relaunch, not just how-much is left.
type DriveCarryRow struct {
	TS                   string `json:"ts,omitempty"`
	Session              string `json:"session"`
	TurnsLeft            int64  `json:"turns_left,omitempty"`
	TokensLeft           int64  `json:"tokens_left,omitempty"`
	ContextTokensLeft    int64  `json:"context_tokens_left,omitempty"`
	SpendMicroCentsLeft  int64  `json:"spend_micro_cents_left,omitempty"`
	TimeLeftNanos        int64  `json:"time_left_nanos,omitempty"`
	Priority             int    `json:"priority,omitempty"`
	PaceMaxTokensPerTurn int    `json:"pace_max_tokens_per_turn,omitempty"`
	PaceMinTurnGapMs     int    `json:"pace_min_turn_gap_ms,omitempty"`
	Generation           int    `json:"generation,omitempty"`
	// Objective pin (the standing goal), carried as its safe extractive triple so a
	// relaunched child can re-pin the SAME objective its first reset reconciles as
	// ObjectivePreserved instead of ObjectiveDropped (#1583). Tags mirror the sibling
	// DriveStateRow so both carry stores speak the same objective grammar.
	ObjectivePinID  string `json:"objective_pin_id,omitempty"`
	ObjectiveText   string `json:"objective_text,omitempty"`
	ObjectiveDigest string `json:"objective_digest,omitempty"`
}

// WithObjectivePin returns a copy of the row carrying the pin's safe extractive
// triple (PinID/Text/Digest). A zero pin carries nothing (the fields stay empty),
// so a session with no standing objective projects a byte-identical budget-only
// row. The Digest is carried VERBATIM — not recomputed — so a later
// ctxplan.ReconcileObjective can still catch a Text that was truncated in transit
// (its Verify would fail, yielding ObjectiveQueryUser rather than a false Preserved).
func (r DriveCarryRow) WithObjectivePin(p ctxplan.ObjectivePin) DriveCarryRow {
	if p.IsZero() {
		return r
	}
	r.ObjectivePinID = p.PinID
	r.ObjectiveText = p.Text
	r.ObjectiveDigest = p.Digest
	return r
}

// ObjectivePin reconstructs the carried objective pin from the row's extractive
// triple — the AFTER pin a relaunched child hands to its first reset so
// ctxplan.ReconcileObjective(before, after) can reconcile the standing objective.
// A row that carried no objective (blank PinID) reconstructs the zero pin, which
// reconciles as ObjectiveDropped against a real prior pin — exactly the visible
// failure #1583 forbids, made observable rather than silent.
func (r DriveCarryRow) ObjectivePin() ctxplan.ObjectivePin {
	if strings.TrimSpace(r.ObjectivePinID) == "" {
		return ctxplan.ObjectivePin{}
	}
	return ctxplan.ObjectivePin{
		PinID:  r.ObjectivePinID,
		Text:   r.ObjectiveText,
		Digest: r.ObjectiveDigest,
	}
}

// FoldDriveCarryRows returns the latest carry row per non-blank session.
func FoldDriveCarryRows(rows []DriveCarryRow) map[string]DriveCarryRow {
	out := make(map[string]DriveCarryRow)
	for _, row := range rows {
		sid := strings.TrimSpace(row.Session)
		if sid == "" {
			continue
		}
		row.Session = sid
		out[sid] = row
	}
	return out
}
