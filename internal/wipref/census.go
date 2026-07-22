package wipref

import "sort"

// census.go is the PURE, read-only classifier behind `fak wip reap --census` (#5340).
// It carries zero git I/O — the cmd shell (cmd/fak/wip_census.go) gathers the git
// facts about each ref and hands this package (records + facts); this package turns
// them into an owner-state census. It NEVER decides a delete: the census is a
// reporting path that runs beside — and changes nothing about — the reap delete path.
//
// #5340 exists because refs/fak/wip/* checkpoints grow unbounded (thousands of refs
// over days) while `fak wip reap` collects ~none: the reap resolver only ever proves
// OwnerLanded (a verbatim landing) or falls back to OwnerUnknown (fail-safe keep), so
// a dead session's redundant checkpoint is kept forever. The census answers the
// question the eventual fix needs to size itself against: of every live ref, how many
// are dead-session checkpoints whose delta is already reflected in HEAD and could be
// safely collected (CensusClosedCleanEstimate — the headline number), versus how
// many carry unlanded recoverable work that MUST be kept (CensusClosedDirtyRecoverable).

// CensusClass is the closed vocabulary a single checkpoint ref is classified into.
// It deliberately mirrors the OwnerState lifecycle facts the reap fold already
// understands, with the CLOSED_* states suffixed "_ESTIMATE"/"_RECOVERABLE" to mark
// that this is a read-only estimate the census reports — not a proven-safe delete the
// reap path acts on.
type CensusClass string

const (
	// CensusLanded: the checkpoint's delta is already present in HEAD verbatim (the
	// owner committed it). Today's `fak wip reap` already collects exactly this set.
	CensusLanded CensusClass = "LANDED"
	// CensusLive: the owning session still holds a live lease. Its WIP is in active
	// use — never a collection estimate.
	CensusLive CensusClass = "LIVE"
	// CensusClosedCleanEstimate: the owning session is NOT live and the delta is
	// empty or fully subsumed by HEAD (every added line already present in HEAD's
	// copy of those files, and no removed line still lives there — reflected, even if
	// not byte-identical). This is the headline #5340 number: the refs the eventual
	// fix could safely collect that today's reap leaves behind.
	CensusClosedCleanEstimate CensusClass = "CLOSED_CLEAN_ESTIMATE"
	// CensusClosedDirtyRecoverable: the owning session is NOT live and the delta is
	// non-empty and NOT subsumed by HEAD. This is exactly the snapshot a later recover
	// exists to restore — it MUST be kept. The safety invariant of the whole census.
	CensusClosedDirtyRecoverable CensusClass = "CLOSED_DIRTY_RECOVERABLE"
	// CensusUnknown: nothing could be positively resolved (e.g. no reachable parent to
	// diff). Kept, fail-safe — the same default the reap fold collapses to.
	CensusUnknown CensusClass = "UNKNOWN"
)

// CensusFacts is the closed set of git-witnessed facts about one checkpoint ref that
// Classify folds into a CensusClass. The cmd shell fills these from git (liveness of
// the owning session, whether the delta landed verbatim in HEAD, and — for a
// dead-session unlanded ref — whether the delta is empty or line-subsumed by HEAD).
// Every field defaults false, and an all-false fact set classifies CensusUnknown, so
// a ref the shell could not resolve is kept, never collected.
type CensusFacts struct {
	// Live: the owning session still holds a live lease (refs/fak/locks/*).
	Live bool
	// Landed: the delta is present verbatim in HEAD (OwnerLanded — the reap resolver).
	Landed bool
	// Resolved: the delta was computable at all (a parent existed and diffed). When
	// false — and the ref is neither Live nor Landed — the ref is CensusUnknown.
	Resolved bool
	// DeltaEmpty: the resolved delta changes nothing (no textual/file change).
	DeltaEmpty bool
	// Subsumed: every added line of the delta already appears in HEAD's copy of its
	// file, and no removed line still appears there — the delta is reflected in HEAD.
	Subsumed bool
}

// Classify folds one ref's facts into its census class. The precedence mirrors the
// reconcile spine: liveness is decided FIRST (a live owner is never a collection
// estimate, whatever its delta), then a verbatim landing, then — for a dead session
// only — an unresolved ref is kept as UNKNOWN, an empty-or-subsumed delta is a clean
// estimate, and anything else is recoverable and kept. It is a total function: any
// fact set maps to exactly one class, and the DEFAULT of the final split is the
// safe, keep-side CensusClosedDirtyRecoverable — so a delta that cannot be proven
// subsumed is never called clean.
func Classify(f CensusFacts) CensusClass {
	switch {
	case f.Live:
		return CensusLive
	case f.Landed:
		return CensusLanded
	case !f.Resolved:
		return CensusUnknown
	case f.DeltaEmpty || f.Subsumed:
		return CensusClosedCleanEstimate
	default:
		return CensusClosedDirtyRecoverable
	}
}

// CensusReason renders the human, auditable reason a ref landed in class c, so a
// `--census --json` snapshot is self-explaining the way a reap verdict is.
func CensusReason(c CensusClass) string {
	switch c {
	case CensusLanded:
		return "delta already present in HEAD verbatim — today's reap collects this"
	case CensusLive:
		return "owning session still holds a live lease"
	case CensusClosedCleanEstimate:
		return "dead session; delta empty or fully subsumed by HEAD — safe to collect"
	case CensusClosedDirtyRecoverable:
		return "dead session; unlanded delta not subsumed by HEAD — recoverable, kept"
	default:
		return "owner state unresolved — kept (fail-safe)"
	}
}

// DeltaSubsumed is the PURE line-level subsumption oracle: it reports whether a
// checkpoint delta is already reflected in HEAD even when the file is not byte-
// identical to the checkpoint's version. added/removed map a file path to the delta's
// added / removed CONTENT lines (blank lines dropped, trimmed by the caller); head
// maps a file path to the set of lines in HEAD's current version of that path (a path
// HEAD does not have maps to an empty/absent set).
//
// The delta is subsumed iff it changed at least one line AND every added line already
// appears in HEAD's copy of its file AND no removed line still appears in HEAD's copy
// of its file. The removed-line clause is the safety teeth: a checkpoint that DELETES
// a line still present in HEAD (its deletion not yet landed) is recoverable work, and
// must NOT be called subsumed — the added-line test alone (vacuously true for a
// pure-deletion delta) would wrongly wave it through. A delta with no added or removed
// content lines (binary/mode/rename-only) is not subsumed here; emptiness is a
// separate fact carried by CensusFacts.DeltaEmpty.
func DeltaSubsumed(added, removed map[string][]string, head map[string]map[string]bool) bool {
	changed := false
	for path, lines := range added {
		h := head[path]
		for _, ln := range lines {
			changed = true
			if !h[ln] {
				return false // an added line HEAD does not carry — not reflected
			}
		}
	}
	for path, lines := range removed {
		h := head[path]
		for _, ln := range lines {
			changed = true
			if h[ln] {
				return false // a removed line still lives in HEAD — deletion unlanded
			}
		}
	}
	return changed
}

// CensusVerdict is one ref's census classification plus the reason behind it — the
// per-ref row a `--census --json` breakdown carries, mirroring ReapVerdict.
type CensusVerdict struct {
	Session string      `json:"session"`
	Ref     string      `json:"ref"`
	Object  string      `json:"object"`
	Class   CensusClass `json:"class"`
	Reason  string      `json:"reason"`
}

// CensusCounts is the aggregate headline of a census: the total ref count and the
// per-class tally. CensusCounts.ClosedCleanEstimate is the #5340 headline number.
type CensusCounts struct {
	Total                  int `json:"total"`
	Landed                 int `json:"landed"`
	Live                   int `json:"live"`
	ClosedCleanEstimate    int `json:"closed_clean_estimate"`
	ClosedDirtyRecoverable int `json:"closed_dirty_recoverable"`
	Unknown                int `json:"unknown"`
}

// CensusReport is the full read-only census: the aggregate counts plus every per-ref
// verdict, session-sorted so a before/after snapshot diffs cleanly.
type CensusReport struct {
	Counts   CensusCounts    `json:"counts"`
	Verdicts []CensusVerdict `json:"verdicts"`
}

// BuildCensus folds per-ref verdicts into a deterministic report: it session-sorts the
// verdicts (tie-broken by object id) and tallies the per-class counts. Pure — the cmd
// shell does the git I/O to produce the verdicts, this fold produces the report.
func BuildCensus(verdicts []CensusVerdict) CensusReport {
	out := make([]CensusVerdict, len(verdicts))
	copy(out, verdicts)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Session != out[j].Session {
			return out[i].Session < out[j].Session
		}
		return out[i].Object < out[j].Object
	})
	c := CensusCounts{Total: len(out)}
	for _, v := range out {
		switch v.Class {
		case CensusLanded:
			c.Landed++
		case CensusLive:
			c.Live++
		case CensusClosedCleanEstimate:
			c.ClosedCleanEstimate++
		case CensusClosedDirtyRecoverable:
			c.ClosedDirtyRecoverable++
		default:
			c.Unknown++
		}
	}
	return CensusReport{Counts: c, Verdicts: out}
}
