package resume

import "strings"

// DriveCarryRow is the append-only transcript-UUID-keyed projection of the
// remaining drive budget. Scalars keep this foundation leaf independent of session.
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
