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
//
// #5940 SHARPENED THAT CLASSIFICATION and, in doing so, moved the headline DOWN. The
// class is now decided from a PER-PAYLOAD-FILE A/M/- reading (payload.go) before any
// line-level test, because "landed vs unlanded" is a false dichotomy: a file HEAD
// carries with DIFFERENT content is neither, and the naive remedy for it — a checkout of
// the ref's blob — silently reverts HEAD when HEAD holds the newer copy. That third
// state is CensusDiverged, and CensusClosedCleanEstimate excludes it. The other half of
// #5940 is trap A: a ref whose payload could not be READ is CensusUnknown (kept), never
// an empty — hence CensusFacts.PayloadRead.

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
	// CensusClosedDirtyRecoverable: the owning session is NOT live and the checkpoint
	// holds at least one payload file HEAD does NOT CARRY AT ALL (an `A` file — see
	// payload.go), or — for a ref with an empty payload — a delta that is non-empty and
	// not subsumed by HEAD. This is exactly the snapshot a later recover exists to
	// restore — it MUST be kept. The safety invariant of the whole census.
	CensusClosedDirtyRecoverable CensusClass = "CLOSED_DIRTY_RECOVERABLE"
	// CensusDiverged: the owning session is NOT live, HEAD carries EVERY one of the
	// checkpoint's payload paths, and at least one of them differs in content (an `M`
	// file). It is neither landed (the bytes are not HEAD's) nor at-risk salvage (HEAD
	// has no missing path), and WHICH SIDE IS NEWER IS NOT KNOWN — HEAD may hold a
	// larger, later form. The only correct action is a three-way diff, never a
	// checkout (#5940 trap B), which is why it is its own class: folding it into
	// CLOSED_CLEAN_ESTIMATE would inflate the headline collection number with refs
	// whose files still differ from HEAD, and folding it into
	// CLOSED_DIRTY_RECOVERABLE would recommend a recipe that reverts newer work.
	CensusDiverged CensusClass = "DIVERGED"
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
	// NOTE (#5940): subsumption is a LINE-SET test with no ordering. "Every added line
	// appears somewhere in HEAD's copy" does not prove HEAD is newer, only that the
	// lines are present, so it is retained as a REFINEMENT reported alongside DIVERGED
	// and is no longer sufficient on its own to promote a ref whose payload files still
	// differ from HEAD into CensusClosedCleanEstimate. Byte-identity is.
	Subsumed bool

	// ---- #5940: the per-payload-FILE reading (payload.go). ----

	// PayloadRead: the ref's payload file list AND its per-file A/M/- status against
	// HEAD were READ — every git plumbing call in that path exited 0.
	//
	// FALSE IS NOT "this ref holds no payload". It is "this ref could not be
	// measured", and it classifies CensusUnknown (kept). That distinction is the whole
	// of trap A: `git diff <obj>^ <obj>` on a ref with no reachable parent exits 128
	// with EMPTY STDOUT, and a caller that reads only stdout scores it as an empty —
	// therefore collectible — checkpoint. The field defaults false, so a shell that
	// never fills it reports every ref unmeasured rather than every ref empty.
	PayloadRead bool
	// PayloadFiles is how many files the checkpoint contributes. Meaningful only when
	// PayloadRead is true.
	PayloadFiles int
	// PayloadAbsent is how many of those files HEAD does not carry AT ALL (`A`) — the
	// only true at-risk salvage.
	PayloadAbsent int
	// PayloadDiverged is how many of those files HEAD carries with DIFFERENT content
	// (`M`) — DIVERGED, a three-way diff and never a checkout.
	PayloadDiverged int
}

// Classify folds one ref's facts into its census class. The precedence mirrors the
// reconcile spine: liveness is decided FIRST (a live owner is never a collection
// estimate, whatever its delta), then a verbatim landing. For a dead session it then
// decides on the PER-PAYLOAD-FILE reading (#5940) before it ever looks at the
// line-level delta:
//
//	!PayloadRead      -> UNKNOWN                   trap A: unmeasured, NEVER "empty"
//	PayloadAbsent > 0 -> CLOSED_DIRTY_RECOVERABLE  HEAD lacks a path outright: at risk
//	PayloadDiverged>0 -> DIVERGED                  present, differs: three-way diff only
//	PayloadFiles  > 0 -> CLOSED_CLEAN_ESTIMATE     every payload file byte-identical
//
// Only a ref whose payload was read and is EMPTY falls through to the #5340 line-level
// tail (unresolved -> UNKNOWN, empty-or-subsumed -> clean estimate, else recoverable).
//
// The absent-before-diverged order is deliberate: a ref holding BOTH names the work
// that can actually be LOST, and its per-file lists still carry the diverged paths, so
// nothing is hidden — while the remedy for each file stays per-file (PayloadRemedy),
// never per-ref. Both classes are kept, so the ordering cannot cost a ref its content.
//
// It is a total function: any fact set maps to exactly one class, and every default is
// the safe, keep side — an unmeasured ref is UNKNOWN, and a delta that cannot be proven
// subsumed is never called clean.
func Classify(f CensusFacts) CensusClass {
	switch {
	case f.Live:
		return CensusLive
	case f.Landed:
		return CensusLanded
	case !f.PayloadRead:
		// Trap A. The payload could not be READ. An empty stdout from a plumbing call
		// that also exited non-zero is not a measurement, so this is kept, not collected.
		return CensusUnknown
	case f.PayloadAbsent > 0:
		return CensusClosedDirtyRecoverable
	case f.PayloadDiverged > 0:
		return CensusDiverged
	case f.PayloadFiles > 0:
		// Read, non-empty, nothing absent and nothing diverged: every payload file is
		// byte-identical to HEAD. This is the strongest possible clean evidence.
		return CensusClosedCleanEstimate
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
		return "dead session; payload HEAD does not carry at all (or an unlanded, un-subsumed delta) — recoverable, kept"
	case CensusDiverged:
		return "dead session; every payload path is on HEAD but at least one DIFFERS — HEAD may hold the newer copy; three-way diff, never a checkout"
	default:
		return "owner state unresolved or payload unreadable — kept (fail-safe)"
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

	// ---- #5940: the per-payload-FILE breakdown behind the class. ----

	// PayloadRead is false when the ref's payload could not be measured. The three
	// counts below are then 0 because NOTHING WAS MEASURED — not because the ref is
	// empty. A consumer must read this flag before reading the counts.
	PayloadRead bool `json:"payload_read"`
	// PayloadUnreadable is the reason PayloadRead is false.
	PayloadUnreadable string `json:"payload_unreadable,omitempty"`
	// Payload is the number of files the checkpoint contributes.
	Payload int `json:"payload"`
	// Absent / Diverged / Landed are the A / M / - tallies over those files.
	Absent   int `json:"absent"`
	Diverged int `json:"diverged"`
	Landed   int `json:"landed"`
	// AbsentPaths and DivergedPaths are BOUNDED samples (CensusPathSample) of the
	// paths behind those tallies — enough to act on without letting one tree-wide
	// snapshot's file list dominate a several-thousand-ref JSON report. Truncated is
	// true when either list was cut, so a reader is never shown a short list that
	// looks complete.
	AbsentPaths   []string `json:"absent_paths,omitempty"`
	DivergedPaths []string `json:"diverged_paths,omitempty"`
	Truncated     bool     `json:"paths_truncated,omitempty"`
	// Subsumed is the #5340 line-level refinement, retained as a REPORTED fact: on a
	// DIVERGED row it marks the tail whose lines all already appear in HEAD (probably
	// already reflected — still a three-way diff, never a checkout).
	Subsumed bool `json:"subsumed,omitempty"`
}

// CensusPathSample bounds how many payload paths one verdict carries in each direction.
const CensusPathSample = 10

// SamplePaths bounds a payload path list to CensusPathSample entries, reporting whether
// it was cut. The cut flag travels WITH the list so a truncated sample can never be read
// as the whole set — the same discipline as PayloadReading.Read: a short answer must
// never be indistinguishable from a complete one.
func SamplePaths(paths []string) ([]string, bool) {
	if len(paths) <= CensusPathSample {
		return paths, false
	}
	out := make([]string, CensusPathSample)
	copy(out, paths[:CensusPathSample])
	return out, true
}

// WithPayload attaches a per-file payload census to a verdict, bounding the path
// samples. It is the only intended way to fill the payload fields, so the read/unread
// distinction and the truncation flag cannot be dropped by a caller filling them by hand.
func (v CensusVerdict) WithPayload(p PayloadCensus) CensusVerdict {
	if !p.Read {
		v.PayloadRead, v.PayloadUnreadable = false, p.Unreadable
		return v
	}
	absent, cutA := SamplePaths(p.AbsentPaths)
	diverged, cutD := SamplePaths(p.DivergedPaths)
	v.PayloadRead = true
	v.Payload, v.Absent, v.Diverged, v.Landed = p.Files, len(p.AbsentPaths), len(p.DivergedPaths), p.Landed
	v.AbsentPaths, v.DivergedPaths, v.Truncated = absent, diverged, cutA || cutD
	return v
}

// CensusCounts is the aggregate headline of a census: the total ref count and the
// per-class tally. CensusCounts.ClosedCleanEstimate is the #5340 headline number.
type CensusCounts struct {
	Total                  int `json:"total"`
	Landed                 int `json:"landed"`
	Live                   int `json:"live"`
	ClosedCleanEstimate    int `json:"closed_clean_estimate"`
	ClosedDirtyRecoverable int `json:"closed_dirty_recoverable"`
	// Diverged (#5940) is the third state: dead-session refs whose every payload path
	// is on HEAD but whose content differs. It is EXCLUDED from ClosedCleanEstimate on
	// purpose — the headline a future collector sizes itself against must not include
	// a ref whose files still differ from HEAD.
	Diverged int `json:"diverged"`
	Unknown  int `json:"unknown"`

	// ---- per-payload-FILE totals across the whole census. ----

	// FilesAbsent is how many payload files, across every ref, HEAD does not carry at
	// all — the true at-risk salvage surface.
	FilesAbsent int `json:"files_absent"`
	// FilesDiverged is how many payload files are present on HEAD with different
	// content. Every one of them needs a three-way diff, never a checkout.
	FilesDiverged int `json:"files_diverged"`
	// PayloadUnreadable is how many refs could not be measured at all. It is reported
	// separately from Unknown's other causes because a non-zero value means the census
	// has BLIND SPOTS, and a blind spot is not an empty checkpoint.
	PayloadUnreadable int `json:"payload_unreadable"`
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
		case CensusDiverged:
			c.Diverged++
		default:
			c.Unknown++
		}
		c.FilesAbsent += v.Absent
		c.FilesDiverged += v.Diverged
		// Keyed on the RECORDED REASON, not on !PayloadRead: the shell short-circuits a
		// LIVE or LANDED ref before it ever reads a payload, and counting those as
		// blind spots would bury the ones that were attempted AND FAILED — the only
		// rows that mean the census is missing something.
		if v.PayloadUnreadable != "" {
			c.PayloadUnreadable++
		}
	}
	return CensusReport{Counts: c, Verdicts: out}
}
