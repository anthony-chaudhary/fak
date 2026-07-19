// Overlay MAINTENANCE loop (#5023): the tick that keeps the operator overlay
// current. `fak steer prs` is a PULL surface — it computes the fold when an
// operator asks; nothing recomputes it as commits land, records how the
// residual pile moved over a night, or gives internal/loopfleet a ledger to
// read the overlay's liveness from. This file is that something.
//
// Shape (a NORMAL loop, not a super loop): its unit of work is one tick —
// read the development-trunk commits landed since the last tick, assign each
// to its unit (or surface it as an orphan), re-fold each touched unit's band,
// append ONE ledger row — and the only thing it mutates is its own
// append-only ledger, docs/nightrun/steerpr-overlay.jsonl (schema
// fak.steerpr-overlay.v1). It never mutates the repo, the trunk, or an issue.
//
// The done-condition is EXTERNALLY witnessed: "every commit in the tick's
// range is assigned to exactly one unit or reported as an orphan, and every
// touched unit's band was re-folded" is a claim an oracle outside this
// package (dos commit-audit over the range) corroborates. The loop host binds
// that claim through loopgate.TurnForSteerprTick, and loopgate.Adjudicate
// admits it only on an external OutcomeWitnessed — a self-reported done
// re-arms with LOOP_DONE_UNWITNESSED. This package stays pure and git-free
// (pureRoot): it takes already-read git log text and caller-supplied
// verdicts, and its only I/O is the append-only ledger write.
//
// The no-silent-drop invariant is checkable FROM THE ROW ALONE:
// commits_seen == assigned + orphans (and the band tallies partition
// units_total). CheckOverlayRow enforces it on every append, so a row that
// silently dropped a commit can never reach the ledger.
package steerpr

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// OverlaySchema is the versioned identifier for one overlay-maintenance
// ledger row.
const OverlaySchema = "fak.steerpr-overlay.v1"

// OverlayLedgerRelPath is the repo-relative ledger the loop appends to. Like
// every docs/nightrun ledger it is append-only and peer-written: rows are
// only ever appended, never rewritten.
const OverlayLedgerRelPath = "docs/nightrun/steerpr-overlay.jsonl"

// OverlayRow is one appended ledger row — one tick's fold state. Every field
// is scalar, so rows compare with == and the idempotency check needs no
// deep-equal. Assigned (the units' member sum) is carried explicitly so the
// no-silent-drop invariant commits_seen == assigned + orphans is checkable
// from the row alone, the acceptance gate the issue names.
type OverlayRow struct {
	Schema       string `json:"schema"`
	TS           string `json:"ts"`
	Base         string `json:"base"`
	Head         string `json:"head"`
	CommitsSeen  int    `json:"commits_seen"`
	UnitsTotal   int    `json:"units_total"`
	Assigned     int    `json:"assigned"`
	Residual     int    `json:"residual"`
	Cleared      int    `json:"cleared"`
	Unverifiable int    `json:"unverifiable"`
	Orphans      int    `json:"orphans"`
	// Witness is the external witness binding the tick's done-claim was
	// ADMITTED on (loopgate verdict WITNESSED), rendered as the re-runnable
	// audit reference. Empty when the claim was not (or not yet) witnessed —
	// the field is set from the loopgate decision, never self-reported.
	Witness string `json:"witness,omitempty"`
}

// SameState reports whether two rows describe the identical fold state —
// every field equal except the wall-clock ts. It is the idempotency
// comparator: a re-tick over the same range with the same verdicts is the
// same state, not a new row.
func (r OverlayRow) SameState(o OverlayRow) bool {
	r.TS, o.TS = "", ""
	return r == o
}

// DoneClaim renders the tick's done-claim: the assigned-or-orphaned totality
// plus the re-folded band tallies. It is deliberately a CLAIM — text the
// external witness corroborates or refuses — never evidence by itself.
func (r OverlayRow) DoneClaim() string {
	return fmt.Sprintf(
		"steerpr overlay tick %s..%s: %d commit(s) each assigned to exactly one of %d unit(s) (%d member(s)) or reported among %d orphan(s); every touched unit's band re-folded (residual=%d cleared=%d unverifiable=%d)",
		r.Base, r.Head, r.CommitsSeen, r.UnitsTotal, r.Assigned, r.Orphans,
		r.Residual, r.Cleared, r.Unverifiable)
}

// TickResult is one tick's full output: the ledger row plus the folded view
// behind it (for a host that wants to render or cross-check), and the
// done-clean bit.
type TickResult struct {
	Row       OverlayRow
	Units     []Unit
	Unstamped []Commit
	// Clean is true when the range held zero new commits. An idle trunk is a
	// DONE-CLEAN tick, not a dark loop — do not conflate the two.
	Clean bool
}

// Tick is the loop's unit of work, pure over its inputs: parse the range's
// git log, apply the caller-supplied witness verdicts (keyed by full SHA or a
// dos-style unique short-SHA prefix), fold commits into units, and derive the
// ledger row. The partition over (units, unstamped) is TOTAL and DISJOINT
// (FoldUnits' contract), which is what makes commits_seen == assigned +
// orphans hold by construction — Tick surfaces it in the row so it stays
// checkable after the fold is gone.
func Tick(base, head, rawLog string, verdicts map[string]Verdict, now time.Time) TickResult {
	commits := ParseLog(rawLog)
	for i := range commits {
		if v, ok := verdictFor(verdicts, commits[i].SHA); ok {
			commits[i].Verdict = v
		}
	}
	units, unstamped := FoldUnits(commits)
	row := OverlayRow{
		Schema:      OverlaySchema,
		TS:          now.UTC().Format(time.RFC3339),
		Base:        strings.TrimSpace(base),
		Head:        strings.TrimSpace(head),
		CommitsSeen: len(commits),
		UnitsTotal:  len(units),
		Orphans:     len(unstamped),
	}
	for _, u := range units {
		row.Assigned += len(u.Commits)
		switch u.Band {
		case BandResidual:
			row.Residual++
		case BandCleared:
			row.Cleared++
		default:
			row.Unverifiable++
		}
	}
	return TickResult{Row: row, Units: units, Unstamped: unstamped, Clean: len(commits) == 0}
}

// verdictFor resolves one commit's verdict from the supplied map: exact
// full-SHA key first, else the LONGEST matching short-SHA prefix
// (lexicographic tiebreak), so the lookup is deterministic across map
// iteration order — the fold's determinism contract extends to the tick.
func verdictFor(verdicts map[string]Verdict, sha string) (Verdict, bool) {
	if v, ok := verdicts[sha]; ok {
		return v, true
	}
	best := ""
	var bestV Verdict
	for k, v := range verdicts {
		if k == "" || !strings.HasPrefix(sha, k) {
			continue
		}
		if len(k) > len(best) || (len(k) == len(best) && k < best) {
			best, bestV = k, v
		}
	}
	if best == "" {
		return "", false
	}
	return bestV, true
}

// CheckOverlayRow validates one ledger row's invariants from the row alone:
// the schema tag, a parseable ts, non-negative counts, the no-silent-drop
// equation commits_seen == assigned + orphans, and the band partition
// residual + cleared + unverifiable == units_total. AppendOverlayRow runs it
// before every write, so a malformed row never reaches the ledger.
func CheckOverlayRow(r OverlayRow) error {
	if r.Schema != OverlaySchema {
		return fmt.Errorf("row schema %q, want %q", r.Schema, OverlaySchema)
	}
	if strings.TrimSpace(r.Head) == "" {
		return errors.New("row head is empty")
	}
	if _, err := time.Parse(time.RFC3339, r.TS); err != nil {
		return fmt.Errorf("row ts %q is not RFC3339: %w", r.TS, err)
	}
	for name, n := range map[string]int{
		"commits_seen": r.CommitsSeen, "units_total": r.UnitsTotal, "assigned": r.Assigned,
		"residual": r.Residual, "cleared": r.Cleared, "unverifiable": r.Unverifiable, "orphans": r.Orphans,
	} {
		if n < 0 {
			return fmt.Errorf("row %s is negative (%d)", name, n)
		}
	}
	if r.CommitsSeen != r.Assigned+r.Orphans {
		return fmt.Errorf("no-silent-drop violated: commits_seen %d != assigned %d + orphans %d",
			r.CommitsSeen, r.Assigned, r.Orphans)
	}
	if r.Residual+r.Cleared+r.Unverifiable != r.UnitsTotal {
		return fmt.Errorf("band partition not total: residual %d + cleared %d + unverifiable %d != units_total %d",
			r.Residual, r.Cleared, r.Unverifiable, r.UnitsTotal)
	}
	return nil
}

// AppendOverlayRow appends one validated row to the ledger at path. It is
// IDEMPOTENT on a re-tick: when the last ledger row describes the same fold
// state (SameState — everything but ts), nothing is written and appended is
// false. The ledger is append-only: existing rows are never rewritten, the
// file is opened O_APPEND, and the first writer creates it.
func AppendOverlayRow(path string, row OverlayRow) (appended bool, err error) {
	if err := CheckOverlayRow(row); err != nil {
		return false, err
	}
	if last, ok := lastOverlayRow(path); ok && last.SameState(row) {
		return false, nil
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return false, err
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return false, err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	b, err := json.Marshal(row)
	if err != nil {
		return false, err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		return false, err
	}
	return true, nil
}

// LastOverlayState returns the ledger's last parseable row, false when the
// ledger is absent or holds none. It is how a host closes the loop "since the
// last tick": the next tick's base is the last row's head, so the ranges
// chain with no gap and no overlap.
func LastOverlayState(path string) (OverlayRow, bool) {
	return lastOverlayRow(path)
}

// lastOverlayRow returns the last parseable schema-matching row of the
// ledger, tolerating an absent file and malformed lines (a foreign or broken
// line never wedges the loop; it just does not count as prior state).
func lastOverlayRow(path string) (OverlayRow, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return OverlayRow{}, false
	}
	lines := strings.Split(string(b), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		var row OverlayRow
		if json.Unmarshal([]byte(line), &row) == nil && row.Schema == OverlaySchema {
			return row, true
		}
	}
	return OverlayRow{}, false
}
