package rsiloop

// curator.go models a skill curator's lifecycle actions (archive / consolidate)
// as RSI keep/revert decisions with a STRUCTURED, queryable reason — the fak
// answer to Hermes' coarse whole-snapshot rollback (#2841, part of #2834).
//
// Hermes' curator auto-archives stale agent-created skills and consolidates
// overlaps, taking a pre-run tar.gz snapshot so users "never lose skills". But
// the *reason* a skill was archived is not a first-class record, and restore is
// manual and whole-snapshot: reverting one wrong call rolls back every sibling
// decision taken in the same run.
//
// Here every action is one append-only journal row carrying a structured reason
// (stale N days | superseded by skill X | slop-scored), and a REVERT is one more
// row that names exactly ONE prior decision by its sequence number. Because the
// governing state is FOLDED from the journal, revert is per-decision and free:
// reverting decision N restores only N's skill and never touches a sibling. The
// read-path (Log / Why) answers "why is this skill gone?" from the journal alone.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
)

// CuratorAction is the CLOSED set of lifecycle actions a decision row can carry.
// archive and consolidate remove a skill from the live set; revert is the undo
// row that names one prior decision by sequence.
type CuratorAction string

const (
	// CuratorArchive retires a single agent-created skill (it is now "gone").
	CuratorArchive CuratorAction = "archive"
	// CuratorConsolidate folds an overlapping skill into another (also "gone").
	CuratorConsolidate CuratorAction = "consolidate"
	// CuratorRevert is the undo row: it names exactly one prior decision seq and
	// carries no reason of its own.
	CuratorRevert CuratorAction = "revert"
)

// CuratorReasonKind is the CLOSED, structured vocabulary for WHY a curator action
// fired. Keeping the tokens distinct is the point of #2841's confusion risk:
// "archived because stale" and "archived because superseded" must never collapse
// into one generic reason — an operator asking "why is this skill gone?" gets a
// token they can act on, not prose.
type CuratorReasonKind string

const (
	// ReasonStale — the skill was untouched for at least StaleDays days.
	ReasonStale CuratorReasonKind = "stale"
	// ReasonSuperseded — a newer skill (SupersededBy) now covers this one.
	ReasonSuperseded CuratorReasonKind = "superseded"
	// ReasonSlopScored — the slop scorecard flagged the skill (SlopScore).
	ReasonSlopScored CuratorReasonKind = "slop_scored"
	// ReasonSelfFulfilling — the self-fulfilling-skill detector flagged the skill
	// (#2842): its use_count looked valuable, but its value net of its own
	// invocations (CounterfactualValue) collapsed to zero or negative, so its only
	// "improvement" was being invoked. Distinct from ReasonSlopScored so an
	// operator can tell a low-quality skill from a metric-artifact skill.
	ReasonSelfFulfilling CuratorReasonKind = "self_fulfilling"
	// ReasonUnwitnessed — a PROPOSED skill was refused promotion because it did not
	// witness a strict gain over baseline on the held fixture corpus (#2872): a
	// net-negative, flat, or unbenchmarked candidate, or one whose suite/truth
	// witness was dirty. Distinct from the tokens above (which retire a skill ALREADY
	// in the active set) so an operator can tell "never beat baseline, so never
	// promoted" from "decayed / superseded / gamed after promotion." The absence of a
	// witnessed improvement is itself the complete, answerable reason — it carries no
	// numeric parameter (the metric gap travels on the SkillPromotion witness).
	ReasonUnwitnessed CuratorReasonKind = "unwitnessed"
)

// CuratorReason is the structured reason attached to an archive/consolidate
// decision. Exactly one Kind is set, and the field that Kind requires must be
// populated — Valid enforces that so a journaled reason is always answerable.
type CuratorReason struct {
	Kind CuratorReasonKind `json:"kind"`
	// StaleDays is set iff Kind == ReasonStale (days since the skill was touched).
	StaleDays int `json:"stale_days,omitempty"`
	// SupersededBy is set iff Kind == ReasonSuperseded (the skill that replaced it).
	SupersededBy string `json:"superseded_by,omitempty"`
	// SlopScore is set iff Kind == ReasonSlopScored (the flagging slop score).
	SlopScore float64 `json:"slop_score,omitempty"`
	// UseCount is set iff Kind == ReasonSelfFulfilling (the gameable use_count that
	// made the skill look valuable — the metric the loop can raise on its own).
	UseCount int `json:"use_count,omitempty"`
	// CounterfactualValue is set iff Kind == ReasonSelfFulfilling (the skill's value
	// net of its own invocations — <= 0 for a flagged self-fulfilling skill).
	CounterfactualValue float64 `json:"counterfactual_value,omitempty"`
}

// Valid reports whether the reason names a known Kind AND carries the field that
// Kind requires — the check that keeps a generic/empty reason out of the journal.
func (r CuratorReason) Valid() bool {
	switch r.Kind {
	case ReasonStale:
		return r.StaleDays > 0
	case ReasonSuperseded:
		return r.SupersededBy != ""
	case ReasonSlopScored:
		return r.SlopScore > 0
	case ReasonSelfFulfilling:
		// A self-fulfilling reason must name the positive use_count that made the
		// skill look valuable; its counterfactual value is <= 0 by construction
		// (that is the flag), so use_count is the field that must be populated.
		return r.UseCount > 0
	case ReasonUnwitnessed:
		// A skill refused promotion for showing no witnessed gain needs no numeric
		// parameter — the absence of a strict, suite-green, truth-clean gain is the
		// whole, answerable reason, so the kind alone is a valid record.
		return true
	default:
		return false
	}
}

// String renders the reason as a short, human-readable phrase for the read-path.
func (r CuratorReason) String() string {
	switch r.Kind {
	case ReasonStale:
		return "stale " + strconv.Itoa(r.StaleDays) + " days"
	case ReasonSuperseded:
		return "superseded by skill " + r.SupersededBy
	case ReasonSlopScored:
		return "slop-scored " + strconv.FormatFloat(r.SlopScore, 'g', -1, 64)
	case ReasonSelfFulfilling:
		return "self-fulfilling (use_count " + strconv.Itoa(r.UseCount) +
			", counterfactual value " + strconv.FormatFloat(r.CounterfactualValue, 'g', -1, 64) + ")"
	case ReasonUnwitnessed:
		return "unwitnessed (no measured gain over baseline on the held fixture corpus)"
	default:
		return "unknown reason"
	}
}

// CuratorRow is one append-only decision record. Seq is a monotonic per-ledger id
// so a revert can name exactly one prior decision. For an archive/consolidate row
// Reason is set and Reverts is 0; for a revert row Reason is empty and Reverts is
// the Seq it undoes.
type CuratorRow struct {
	Seq     int           `json:"seq"`
	Action  CuratorAction `json:"action"`
	Skill   string        `json:"skill"`
	Reason  CuratorReason `json:"reason,omitzero"`
	Reverts int           `json:"reverts,omitempty"`
	Note    string        `json:"note,omitempty"`
}

// CuratorEntry is the folded, current status of one skill the read-path returns:
// whether it is presently gone, and if so the governing decision and its reason.
type CuratorEntry struct {
	Skill        string        // the skill this entry describes
	Gone         bool          // true if a live (un-reverted) archive/consolidate governs it
	Action       CuratorAction // the governing action (archive|consolidate) when Gone
	Reason       CuratorReason // the governing reason when Gone
	GoverningSeq int           // the Seq of the decision that made it gone
}

// CuratorLedger is the append-only journal of curator decisions plus the folded
// read-path over it. It is the per-decision, structured-reason substitute for
// Hermes' whole-snapshot restore: every action is a row, and Log/Why answer from
// those rows alone.
type CuratorLedger struct {
	path string
	rows []CuratorRow
}

// OpenCuratorLedger opens (or creates) the JSONL ledger at path and loads any
// existing rows so Seq assignment and the fold continue across process restarts.
// A path of "" keeps the ledger purely in-memory (useful for a fast test). The
// load is CORRUPTION-TOLERANT: a torn final line (an O_APPEND process killed
// mid-write) is skipped, not fatal — the same discipline LastTrack uses.
func OpenCuratorLedger(path string) (*CuratorLedger, error) {
	l := &CuratorLedger{path: path}
	if path == "" {
		return l, nil
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return l, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r CuratorRow
		if err := json.Unmarshal(line, &r); err != nil {
			continue // skip a torn / non-JSON line rather than fail the whole read
		}
		l.rows = append(l.rows, r)
	}
	return l, nil
}

// Archive records an archive decision for skill with a structured reason, returning
// the assigned Seq. The reason must be Valid — an unstructured or empty reason is
// refused so every journaled decision stays answerable.
func (l *CuratorLedger) Archive(skill string, reason CuratorReason) (int, error) {
	return l.decide(CuratorArchive, skill, reason)
}

// Consolidate records a consolidate decision for skill (folding it into the skill
// named by a ReasonSuperseded reason, typically), returning the assigned Seq.
func (l *CuratorLedger) Consolidate(skill string, reason CuratorReason) (int, error) {
	return l.decide(CuratorConsolidate, skill, reason)
}

func (l *CuratorLedger) decide(action CuratorAction, skill string, reason CuratorReason) (int, error) {
	if skill == "" {
		return 0, fmt.Errorf("curator: %s needs a skill name", action)
	}
	if !reason.Valid() {
		return 0, fmt.Errorf("curator: %s of %q needs a valid structured reason, got %+v", action, skill, reason)
	}
	seq := len(l.rows) + 1
	return seq, l.append(CuratorRow{Seq: seq, Action: action, Skill: skill, Reason: reason})
}

// Revert undoes exactly ONE prior decision by its Seq. It is scoped to that single
// decision: the fold recomputes each skill's status from the live (un-reverted)
// rows, so reverting seq N restores only N's skill and never rolls back a sibling
// decision (the per-decision, not whole-snapshot, guarantee of #2841). Reverting an
// unknown seq, a revert row, or an already-reverted decision is refused.
func (l *CuratorLedger) Revert(seq int) error {
	target, ok := l.rowBySeq(seq)
	if !ok {
		return fmt.Errorf("curator: cannot revert unknown decision seq %d", seq)
	}
	if target.Action == CuratorRevert {
		return fmt.Errorf("curator: seq %d is itself a revert, not a decision", seq)
	}
	if l.isReverted(seq) {
		return fmt.Errorf("curator: decision seq %d is already reverted", seq)
	}
	next := len(l.rows) + 1
	return l.append(CuratorRow{Seq: next, Action: CuratorRevert, Skill: target.Skill, Reverts: seq})
}

// Log folds the journal into the current status of every skill it has touched,
// ordered by first appearance. It is the "curator log" read-path: an operator (or
// a test) can read the returned entries to see which skills are gone and why —
// answered from the journal alone, no snapshot needed.
func (l *CuratorLedger) Log() []CuratorEntry {
	reverted := l.revertedSet()
	// Governing decision per skill = the LAST live archive/consolidate row for it.
	governing := map[string]CuratorRow{}
	var order []string
	seen := map[string]bool{}
	for _, r := range l.rows {
		if r.Action == CuratorRevert {
			continue
		}
		if !seen[r.Skill] {
			seen[r.Skill] = true
			order = append(order, r.Skill)
		}
		if reverted[r.Seq] {
			delete(governing, r.Skill) // this decision no longer governs
			continue
		}
		governing[r.Skill] = r
	}
	entries := make([]CuratorEntry, 0, len(order))
	for _, skill := range order {
		if g, gone := governing[skill]; gone {
			entries = append(entries, CuratorEntry{
				Skill:        skill,
				Gone:         true,
				Action:       g.Action,
				Reason:       g.Reason,
				GoverningSeq: g.Seq,
			})
			continue
		}
		entries = append(entries, CuratorEntry{Skill: skill, Gone: false})
	}
	return entries
}

// Why answers "why is this skill gone?" from the journal alone: it returns the
// governing structured reason and true if the skill is currently gone, or a zero
// reason and false if it is live (never archived, or its archive was reverted).
func (l *CuratorLedger) Why(skill string) (CuratorReason, bool) {
	for _, e := range l.Log() {
		if e.Skill == skill {
			return e.Reason, e.Gone
		}
	}
	return CuratorReason{}, false
}

// Rows returns a copy of the append-only journal for inspection/telemetry.
func (l *CuratorLedger) Rows() []CuratorRow {
	return append([]CuratorRow(nil), l.rows...)
}

func (l *CuratorLedger) rowBySeq(seq int) (CuratorRow, bool) {
	for _, r := range l.rows {
		if r.Seq == seq {
			return r, true
		}
	}
	return CuratorRow{}, false
}

func (l *CuratorLedger) revertedSet() map[int]bool {
	reverted := map[int]bool{}
	for _, r := range l.rows {
		if r.Action == CuratorRevert && r.Reverts != 0 {
			reverted[r.Reverts] = true
		}
	}
	return reverted
}

func (l *CuratorLedger) isReverted(seq int) bool {
	return l.revertedSet()[seq]
}

// appendLedgerLine durably appends r as one JSON line to a file-backed ledger at
// path; an empty path marks an in-memory-only ledger and is a no-op. It is the
// shared write half of every rsiloop ledger's append method.
func appendLedgerLine(path string, r any) error {
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		f.Close()
		return err
	}
	b = append(b, '\n')
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// append records the row in memory and, if the ledger is file-backed, durably
// appends it as one JSON line so the journal survives a restart.
func (l *CuratorLedger) append(r CuratorRow) error {
	if err := appendLedgerLine(l.path, r); err != nil {
		return err
	}
	l.rows = append(l.rows, r)
	return nil
}
