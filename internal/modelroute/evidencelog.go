package modelroute

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"time"
)

// THE DURABLE EVIDENCE JOURNAL: the record a capability grade is allowed to be built from
// (epic #5416, tracks D and F).
//
// GradeCapability turns observed outcomes into a grade. This is where those outcomes come
// from: an append-only line-per-turn journal that survives the process, because
// outcome.go's existing journal is in-memory only and a grade rebuilt from nothing every
// restart is a grade nobody can ever earn.
//
// The fold from turns to evidence is short, and every rule in it exists because the
// alternative INFLATES a claim:
//
//  1. A repeated outcome is dropped. Retries, replayed logs and double-shipped rows are
//     normal in a fleet, and an outcome counted twice is a stronger claim than the
//     evidence supports — it is the cheapest possible way to manufacture a grade. Records
//     that carry no id cannot be protected this way, so the count of them is reported:
//     an operator who sees the whole corpus is undeduplicable knows the number they are
//     looking at can be inflated by their own producer.
//  2. Undated evidence cannot satisfy a freshness window. Capability is a property of a
//     model AS DEPLOYED — quantisation, a swapped checkpoint, a context-length change —
//     so an operator who asks for the last 30 days is asking a real question. A record
//     with no timestamp cannot be shown to be inside the window, so it is excluded when
//     one is asked for, and counted. (With no window, dates do not matter and nothing is
//     dropped for lacking one.)
//  3. Provenance is not merged here. Rows are aggregated per (model, class, VERIFY), not
//     per (model, class), so a hundred self-reported turns cannot dilute their way into a
//     block of witnessed ones. GradeCapability does the merging, and keeps the weakest
//     provenance of what it merged.
//  4. A malformed line is skipped and counted, never fatal. A journal an appending fleet
//     is writing to WILL have a torn final line at some point, and refusing the whole file
//     would throw away every good record for one bad one. Refusing to read is not a safety
//     property here; reading silently would be.
//
// Pure and stdlib-only, like the rest of this leaf: producing the rows is the caller's
// job, exactly as producing a Verification is.

// TurnOutcome is ONE observed turn: which model served it, what class of work it was,
// whether it succeeded, and how that success was established.
//
// Class comes from the WORK, never from the model. Inferring it from the model that ran
// is how a small model gets graded on the easy work it was given precisely because it was
// small, and then graded up for succeeding at it.
type TurnOutcome struct {
	// ID is the dedupe key — a turn id, a request id, anything stable per outcome.
	ID string `json:"id,omitempty"`
	// Model is the routed model id, matching the Binding.Model a placement candidate uses.
	Model string `json:"model"`
	// Class is the declared class of the work, not a guess from the model.
	Class WorkClass `json:"class"`
	// Zone records where it ran, when known. Advisory: grading is per model.
	Zone PlacementZone `json:"zone,omitempty"`
	// Success is whether the work was judged to have succeeded.
	Success bool `json:"success"`
	// Verify is HOW that judgement was reached. VerifyNone (the zero value) means the
	// model's own word, which never buys a grade.
	Verify Verification `json:"verify,omitempty"`
	// At is when the turn happened. Optional, and required only to satisfy a window.
	At time.Time `json:"at,omitempty"`
}

// AppendTurnOutcome writes one outcome as a single JSON line. Callers append; nothing in
// this file rewrites or compacts a journal, so a reader can never observe a partially
// rewritten record — only, at worst, a torn final line.
func AppendTurnOutcome(w io.Writer, o TurnOutcome) error {
	b, err := json.Marshal(o)
	if err != nil {
		return fmt.Errorf("marshal outcome: %w", err)
	}
	if _, err := w.Write(append(b, '\n')); err != nil {
		return fmt.Errorf("append outcome: %w", err)
	}
	return nil
}

// JournalStats is what a read of the journal saw, including what it could not use.
type JournalStats struct {
	// Lines is every non-blank line examined.
	Lines int `json:"lines"`
	// Malformed is the lines that would not parse. They are skipped, not fatal.
	Malformed int `json:"malformed,omitempty"`
}

// maxOutcomeLine bounds one journal line. A line longer than this is treated as malformed
// rather than grown into: an unbounded buffer on an append-only file a fleet writes to is
// a memory bug waiting for one corrupt byte.
const maxOutcomeLine = 1 << 20

// ReadTurnOutcomes parses a journal, skipping and counting what it cannot use.
//
// The returned error is reserved for a failure of the READER (an I/O error), never for bad
// content — content problems are reported in JournalStats so the caller can decide whether
// the corpus is good enough, which is a judgement this function does not get to make.
func ReadTurnOutcomes(r io.Reader) ([]TurnOutcome, JournalStats, error) {
	var out []TurnOutcome
	stats, err := scanJSONLines(r, func(line []byte) bool {
		var o TurnOutcome
		if json.Unmarshal(line, &o) != nil {
			return false
		}
		out = append(out, o)
		return true
	})
	if err != nil {
		return out, stats, fmt.Errorf("read outcome journal: %w", err)
	}
	return out, stats, nil
}

// scanJSONLines walks an append-only JSON-lines file, calling parse on each non-blank line
// and counting the ones it rejects. It is shared by this leaf's two journals ON PURPOSE:
// they draw OPPOSITE conclusions from a torn line (see escalatelog.go's header), and the
// only way that difference stays a deliberate policy rather than a drift is for the
// detection of a torn line to be one piece of code with one definition.
//
// parse returns false for a line it could not use; the count comes back as Malformed. The
// error is reserved for a genuine reader failure.
func scanJSONLines(r io.Reader, parse func(line []byte) bool) (JournalStats, error) {
	var stats JournalStats
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxOutcomeLine)
	for sc.Scan() {
		line := sc.Bytes()
		if len(trimSpaceBytes(line)) == 0 {
			continue
		}
		stats.Lines++
		if !parse(line) {
			stats.Malformed++
		}
	}
	if err := sc.Err(); err != nil {
		// A too-long line is a content problem wearing an I/O error's clothes: report it
		// as malformed and keep everything already read, rather than discarding a good
		// corpus over one bad row.
		if errors.Is(err, bufio.ErrTooLong) {
			stats.Lines++
			stats.Malformed++
			return stats, nil
		}
		return stats, err
	}
	return stats, nil
}

// trimSpaceBytes reports the line with leading/trailing ASCII whitespace removed. Kept
// local so the blank-line test does not depend on a package that this leaf's purity rules
// would otherwise have to justify.
func trimSpaceBytes(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && isSpaceByte(b[i]) {
		i++
	}
	for j > i && isSpaceByte(b[j-1]) {
		j--
	}
	return b[i:j]
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'
}

// FoldOptions narrows which outcomes count toward a grade.
type FoldOptions struct {
	// Since, when non-zero, excludes outcomes older than it — AND outcomes that carry no
	// timestamp at all, which cannot be shown to be inside the window.
	Since time.Time
}

// FoldStats is what the fold used and what it refused, so a thin grade can be explained
// without re-reading the journal.
type FoldStats struct {
	// Counted is the outcomes that reached the evidence.
	Counted int `json:"counted"`
	// Duplicates is the outcomes dropped because their id had already been seen.
	Duplicates int `json:"duplicates,omitempty"`
	// Undeduplicable is the counted outcomes that carried no id. They were kept — a
	// missing id is not proof of a duplicate — but they are the part of the corpus that
	// a broken producer could inflate.
	Undeduplicable int `json:"undeduplicable,omitempty"`
	// OutsideWindow is the outcomes excluded by FoldOptions.Since.
	OutsideWindow int `json:"outside_window,omitempty"`
	// Undated is the outcomes excluded because a window was requested and they carried no
	// timestamp. Counted apart from OutsideWindow because the fix is different: one needs
	// a wider window, the other needs a producer that stamps its rows.
	Undated int `json:"undated,omitempty"`
	// Unattributed is the outcomes dropped for naming no model.
	Unattributed int `json:"unattributed,omitempty"`
}

// EvidenceTrail is the audit path from ONE evidence row back to the turns that produced
// it: the (model, class, provenance) key the fold tallies, plus the ids it counted.
//
// It exists because a placement is only as auditable as its evidence. The fold reduces a
// journal to counts, and a count cannot be re-checked — an operator shown "24 of 25
// witnessed attempts at ultra-hard work" has no way to ask WHICH 25, so a producer that
// double-counts, mis-attributes, or grades a model on another model's turns is invisible
// at exactly the moment that number decides where traffic goes.
//
// Turns holds the ids in journal order. Anonymous counts the turns that reached the same
// row carrying no id: they are still evidence and the fold still counts them, but they
// cannot be NAMED, so they are reported apart rather than quietly missing from the
// citation. A row with a non-zero Anonymous is one an operator cannot fully re-walk —
// which is a fact about the producer, not about the model.
type EvidenceTrail struct {
	Model     string       `json:"model"`
	Class     WorkClass    `json:"class"`
	Verify    Verification `json:"verify,omitempty"`
	Turns     []string     `json:"turns,omitempty"`
	Anonymous int          `json:"anonymous,omitempty"`
}

// FoldTurnOutcomes turns a journal into the per-model evidence GradeCapability consumes.
//
// The result is deterministic: evidence rows come back sorted by class then provenance, so
// two runs over the same journal produce byte-identical grades.
func FoldTurnOutcomes(outcomes []TurnOutcome, opt FoldOptions) (map[string][]ClassEvidence, FoldStats) {
	evidence, stats, _ := FoldTurnOutcomesTrail(outcomes, opt)
	return evidence, stats
}

// FoldTurnOutcomesTrail is FoldTurnOutcomes plus the citation: which turns fed which
// evidence row. The evidence and stats are identical — the trail is recorded by the same
// pass that does the counting ON PURPOSE, because a citation assembled by a second walk
// over the journal could disagree with the counts it claims to explain, and an audit trail
// that can drift from the thing it audits is worse than none.
//
// Trails come back sorted by model, then class, then provenance, so the citation of a
// given journal is as reproducible as the grade built from it.
func FoldTurnOutcomesTrail(outcomes []TurnOutcome, opt FoldOptions) (map[string][]ClassEvidence, FoldStats, []EvidenceTrail) {
	type key struct {
		model  string
		class  WorkClass
		verify Verification
	}
	type row struct {
		evidence ClassEvidence
		trail    EvidenceTrail
	}
	var stats FoldStats
	tally := map[key]*row{}
	seen := map[string]bool{}
	window := !opt.Since.IsZero()
	for _, o := range outcomes {
		if o.Model == "" {
			stats.Unattributed++
			continue
		}
		if window {
			if o.At.IsZero() {
				stats.Undated++
				continue
			}
			if o.At.Before(opt.Since) {
				stats.OutsideWindow++
				continue
			}
		}
		if o.ID == "" {
			stats.Undeduplicable++
		} else {
			if seen[o.ID] {
				stats.Duplicates++
				continue
			}
			seen[o.ID] = true
		}
		k := key{model: o.Model, class: o.Class, verify: o.Verify}
		r := tally[k]
		if r == nil {
			r = &row{
				evidence: ClassEvidence{Class: o.Class, Verify: o.Verify},
				trail:    EvidenceTrail{Model: o.Model, Class: o.Class, Verify: o.Verify},
			}
			tally[k] = r
		}
		r.evidence.Attempts++
		if o.Success {
			r.evidence.Successes++
		}
		if o.ID == "" {
			r.trail.Anonymous++
		} else {
			r.trail.Turns = append(r.trail.Turns, o.ID)
		}
		stats.Counted++
	}
	if len(tally) == 0 {
		return nil, stats, nil
	}
	out := map[string][]ClassEvidence{}
	trails := make([]EvidenceTrail, 0, len(tally))
	for k, r := range tally {
		out[k.model] = append(out[k.model], r.evidence)
		trails = append(trails, r.trail)
	}
	for model := range out {
		rows := out[model]
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].Class != rows[j].Class {
				return rows[i].Class < rows[j].Class
			}
			return rows[i].Verify < rows[j].Verify
		})
		out[model] = rows
	}
	sort.Slice(trails, func(i, j int) bool {
		if trails[i].Model != trails[j].Model {
			return trails[i].Model < trails[j].Model
		}
		if trails[i].Class != trails[j].Class {
			return trails[i].Class < trails[j].Class
		}
		return trails[i].Verify < trails[j].Verify
	})
	return out, stats, trails
}

// TurnsBehind names the turns that earned a grade: the ids the fold counted toward the
// class the grade was built from, plus the number of contributing turns that carried no
// id and therefore cannot be named.
//
// Two refusals keep the citation from claiming more than the grade did:
//
//   - Rows the floor refused are NOT cited. GradeCapability drops evidence whose
//     provenance is forgeable or whose class the policy table does not know, and counts
//     it as Dropped instead. Citing those turns would attribute the grade to work that
//     bought none of it — an audit trail that over-claims in the exact direction the
//     grader already refused to.
//   - An UNMEASURED grade cites nothing, because it names no class: there is no set of
//     turns that earned a capability it did not award. Explaining a refusal is a
//     different question, and it is answered by reading the whole trail rather than by
//     inventing a citation for a grade that does not exist.
//
// The invariant a caller can check: len(turns)+anonymous equals Grade.Attempts. A
// citation that does not add up to the number it explains is a producer bug, and it is
// visible here rather than after the traffic has moved.
func TurnsBehind(g Grade, trails []EvidenceTrail, floor GradeFloor) (turns []string, anonymous int) {
	if !g.Measured || g.Model == "" {
		return nil, 0
	}
	for _, t := range trails {
		if t.Model != g.Model || t.Class != g.Class {
			continue
		}
		if !floor.counts(ClassEvidence{Class: t.Class, Verify: t.Verify}) {
			continue
		}
		turns = append(turns, t.Turns...)
		anonymous += t.Anonymous
	}
	return turns, anonymous
}
