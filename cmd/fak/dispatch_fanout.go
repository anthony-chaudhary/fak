package main

// THE ONE SEAM THAT MAKES THE FAN-OUT DEFAULT REAL (#2523).
//
// internal/issuefanout already held every pure piece: the taxonomy expander, the durable
// invocation ledger, the outcome fold, and the path→leaf reduction. What it never had was
// a caller in the loop. Fanning out at spine-ship time was a step written down in the
// spine-fanout skill and remembered — or not — by whichever agent had just shipped, which
// is the exact failure this seam removes: a default that lives in an agent's memory is
// not a default.
//
// It hangs off the witness sweep because that is where the loop LEARNS a spine shipped.
// The sweep already resolves each finished worker's commit and grades it; a record that
// comes back CLAIM_WITNESSED is a diff that provably resolved its issue, which is the
// spine-ship moment the verb documents as its trigger. Nothing else in the tick knows
// that fact, and no second place should re-derive it.
//
// Double-firing is structurally impossible rather than merely avoided: witnessExitedWorkers
// skips any slot that already carries a .witness sidecar, so a finished worker is graded —
// and so fanned out — exactly once no matter how many ticks run, and issuefanout.Turn folds
// several commits in one leaf into a single row.
//
// Two deliberate refusals, both pointing away from over-claiming:
//
//   - It PLANS, it does not FILE. Filing is `fak-dev issue fanout --live`: it wants the
//     tracker, a bounded marker-key dedupe scan and an operator's judgement, and a tick
//     that opened issues on its own would be a loop redesign, not a call site.
//   - The leaf comes from the commit's own changed paths, never from the issue's assigned
//     lane. The assigned lane is a guess made when the issue was filed; the diff is what
//     actually shipped.
//
// Fail-open throughout, like the sweep it hangs off: a ledger that cannot be opened
// reports the error in the payload and the tick dispatches regardless.

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// dispatchFanoutLedgerName is the append-only invocation ledger, written inside the runs
// directory so it lives and dies with the dispatch state it describes. `fak-dev issue fanout`
// folds it per week (issuefanout.FoldWeekly), which is how "was the verb used, without
// being asked" gets answered from a file instead of an anecdote.
const dispatchFanoutLedgerName = "issue-fanout.jsonl"

// dispatchFanoutShips turns this sweep's graded slots into the shipped spines the fan-out
// plans from: a witnessed claim with a resolving commit, plus that commit's changed paths.
//
// The path read goes through the same injectable seam the sweep's footprint grading uses,
// so a turn is reproducible in a test without a repo. A commit whose paths cannot be read
// contributes no ship — the count of those is surfaced by the caller rather than being
// quietly folded into "nothing shipped".
func dispatchFanoutShips(root string, records []dispatchtick.WitnessRecord) (ships []issuefanout.Ship, unread int) {
	for _, r := range records {
		if r.Claim != dispatchtick.ClaimWitnessed || strings.TrimSpace(r.SHA) == "" {
			continue
		}
		paths, ok := dispatchWitnessCommitPaths(root, r.SHA)
		if !ok {
			unread++
			continue
		}
		ships = append(ships, issuefanout.Ship{Issue: r.Issue, SpineRef: r.SHA, Paths: paths})
	}
	return ships, unread
}

// dispatchIssueFanout is the call site. It runs the planner over everything this turn
// shipped, appends one durable ledger row per invocation, and returns the captured
// loop-turn artifact for the tick payload.
//
// A turn that shipped nothing returns nil and adds no payload key, so an idle tick stays
// byte-identical to before this seam existed.
func dispatchIssueFanout(root, runsDir string, records []dispatchtick.WitnessRecord, at time.Time) map[string]any {
	ships, unread := dispatchFanoutShips(root, records)
	if len(ships) == 0 && unread == 0 {
		return nil
	}
	// Only a successfully opened file becomes the writer: assigning a nil *os.File to the
	// interface would give Turn a non-nil writer that panics on the first append.
	var ledger io.Writer
	ledgerErr := ""
	if len(ships) > 0 {
		f, err := os.OpenFile(filepath.Join(runsDir, dispatchFanoutLedgerName), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			ledgerErr = err.Error()
		} else {
			defer f.Close()
			ledger = f
		}
	}
	res := issuefanout.Turn(ships, at, ledger)
	out := map[string]any{
		"schema": res.Schema,
		"ships":  res.Ships,
		"counts": res.Counts,
		"rows":   res.Rows,
	}
	if ledger != nil && len(res.Rows) > 0 {
		// The ledger is named RELATIVE to the runs dir on purpose. The row itself is
		// privacy-clean by contract (schema, UTC stamp, lane, outcome — never a path or a
		// host), and stamping the absolute file name into a payload that gets pasted into
		// issues would put back exactly the host hint the row leaves out.
		out["ledger"] = dispatchFanoutLedgerName
	}
	if res.NoLeaf > 0 {
		out["no_leaf"] = res.NoLeaf
	}
	if len(res.Dropped) > 0 {
		out["dropped"] = res.Dropped
	}
	if unread > 0 {
		out["unread_commits"] = unread
	}
	if ledgerErr != "" {
		out["ledger_error"] = ledgerErr
	}
	return out
}
