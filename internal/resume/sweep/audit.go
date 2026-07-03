// audit.go — the row-assembly fold of the RELAUNCH-OUTCOME audit (the second half of
// the tools/resume_relaunch_audit.py port; relaunch.go carries the per-transcript
// verdict core). Given the ledger roster (which sids a relaunch was ATTEMPTED on) and
// the on-disk transcript copies, it emits one audited row per attempted session,
// sorted still-broken-first — the exact fold the Python's audit() performed.
//
// Pure by construction: the shell reads the ledger, globs and parses the copies, and
// censuses the live `claude --resume` processes; this leaf only folds and sorts.
package sweep

import "sort"

// RelaunchRow is one audited session — the roster facts joined with the transcript
// verdict, the same fields the Python emitted so a machine-record consumer sees an
// unchanged shape.
type RelaunchRow struct {
	SID string `json:"sid"`
	// Actions is the sorted distinct ledger actions recorded for the sid (LedgerActions).
	Actions []string `json:"actions"`
	// Account is the superset copy's account dir (empty for NO_TRANSCRIPT).
	Account string `json:"account"`
	// N is the superset copy's record count (0 for NO_TRANSCRIPT).
	N int `json:"n"`
	// Live: a `claude --resume <sid>` process is running RIGHT NOW — a session that is
	// mid-relaunch is shown as live rather than mistaken for finished.
	Live         bool   `json:"live"`
	SupersetPath string `json:"superset_path,omitempty"`
	RelaunchResult
}

// AuditRelaunches folds the ledger roster and the per-sid transcript copies into the
// audited rows. A rostered sid with no on-disk copy is NO_TRANSCRIPT (the attempt is
// still shown — a vanished transcript is a finding, not a skip). Rows sort
// still-broken-first (RelaunchOrder), then by account, then most records first, then
// sid — fully deterministic where the Python inherited dict order.
func AuditRelaunches(roster map[string][]string, copiesBySID map[string][]Copy, live map[string]bool) []RelaunchRow {
	rows := make([]RelaunchRow, 0, len(roster))
	for sid, actions := range roster {
		row := RelaunchRow{SID: sid, Actions: actions, Live: live[sid]}
		if best := SupersetIndex(copiesBySID[sid]); best < 0 {
			row.RelaunchResult = RelaunchResult{Verdict: VerdictNoTranscript}
		} else {
			c := copiesBySID[sid][best]
			row.Account = c.Account
			row.N = len(c.Records)
			row.SupersetPath = c.Path
			row.RelaunchResult = RelaunchVerdict(c.Records)
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		if a, b := RelaunchOrder(rows[i].Verdict), RelaunchOrder(rows[j].Verdict); a != b {
			return a < b
		}
		if rows[i].Account != rows[j].Account {
			return rows[i].Account < rows[j].Account
		}
		if rows[i].N != rows[j].N {
			return rows[i].N > rows[j].N
		}
		return rows[i].SID < rows[j].SID
	})
	return rows
}

// CountNotOK is how many audited sessions did NOT properly relaunch — the audit's exit
// signal (the Python exited 1 iff any verdict != RELAUNCHED_OK) and its --not-ok filter
// cardinality.
func CountNotOK(rows []RelaunchRow) int {
	n := 0
	for _, r := range rows {
		if r.Verdict != VerdictRelaunchedOK {
			n++
		}
	}
	return n
}
