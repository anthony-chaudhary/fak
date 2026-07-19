package knownbad

import "strings"

// Crash-event coalescing (#3586) — the general "N crashes, 1 cause" fold.
//
// The failure it fixes: one root cause (a poisoned tree, a provider outage) kills
// N workers, and each death emits its own CHILD_CRASH event. At 100x that is
// hundreds of identical events on the operator surface and, if an auto-filer is
// wired, hundreds of duplicate issues — the observability and ticketing layer melts
// exactly when a SYSTEMATIC fault hits, which is the moment one clear signal matters
// most. Filing a ticket is itself fleet work, so the fan-out is not merely noisy, it
// is expensive.
//
// The fold: key every crash event by the SAME Signature the rest of this package
// uses (reason class + normalized tree globs + failure hash), then emit ONE
// occurrence-counted record per signature per window instead of one alert per worker
// death. Two distinct causes still produce two rows — the coalescer dedups a cause,
// never distinct causes.
//
// "Per window" is the existing TTL liveness window, not a new clock: a signature
// with a LIVE latest row is REFRESHED (its count climbs, its discovery instant and
// therefore its TTL stay pinned to the first sighting, so a sustained storm cannot
// extend its own hold forever), and a signature with no live row — never seen, or
// resolved/revoked/expired — OPENS a fresh row starting a new window. A crash
// arriving after a resolve is genuine evidence the failure came back, so reopening
// is correct rather than a duplicate.
//
// One honest caveat this package's append-to-supersede idiom imposes: a refresh
// appends a superseding ROW, so the ledger does physically grow by one line per
// coalesce pass. What is deduped — and what the acceptance criteria mean by "not 100
// rows/issues" — is the SIGNATURE: LiveRecords/Match collapse to exactly one live
// row per signature (the operator card and any auto-filer read that), and Compact
// folds the superseded lines away. It is one issue per cause, not one line per event.
//
// Pure, like the rest of this core: nowUnix is data, never a clock read.

// CrashClassUnclassified is the reason class a crash event with a blank class folds
// into, mirroring guardrsi's crash-class accounting. Keeping unclassified crashes in
// a single named bucket stops a blank class from splitting one cause across several
// signatures (which would defeat the coalescing) while still keeping it distinct from
// a genuinely classified failure.
const CrashClassUnclassified = "UNCLASSIFIED"

// CrashEvent is one supervised-child abnormal-exit observation offered to the
// coalescer — the pure-data projection of a journal CHILD_CRASH row. It carries only
// what the signature is derived from plus the observation instant, so the fold never
// needs to know the journal's row shape: the impure shell reads the journal (or the
// worker's crash report) and hands these in.
type CrashEvent struct {
	// ReasonClass is the crash class (guardrsi's closed vocabulary: SIGNAL_CRASH,
	// OOM, NONZERO_EXIT, …). Blank folds to CrashClassUnclassified.
	ReasonClass string
	// TreeGlobs is the tree the crashed worker was operating over — the containment
	// scope the resulting known-bad row holds. An event whose globs all normalize away
	// is DROPPED (counted in CoalesceStats.DroppedNoTree) rather than emitted as a row
	// that could never match anything.
	TreeGlobs []string
	// FailureHash is the optional guardrsi correlation hash. Two crashes sharing a
	// class and tree but carrying different hashes are DIFFERENT causes, so this
	// participates in the signature exactly as it does everywhere else.
	FailureHash string
	// AtUnix is when the crash was observed; the newest one in a group becomes the
	// row's LastSeenAtUnix.
	AtUnix int64
	// ObservedBy names the reporting worker. The FIRST non-empty one in a group is
	// recorded as the row's discoverer — a coalesced row names one representative
	// reporter, not all N.
	ObservedBy string
}

// Signature is the id this event coalesces under: the same fold the ledger, the
// dispatcher hold, and the same-signature resume backoff key on, so a parked class
// and a coalesced crash row name the SAME signature.
func (e CrashEvent) Signature() string {
	return Signature(crashClass(e.ReasonClass), e.TreeGlobs, e.FailureHash)
}

// crashClass normalizes a crash reason class, bucketing blank into UNCLASSIFIED.
func crashClass(raw string) string {
	c := strings.TrimSpace(raw)
	if c == "" {
		return CrashClassUnclassified
	}
	return c
}

// CoalesceStats reports what one coalesce pass did, so the shell can print an honest
// readout ("300 crashes, 1 cause") and a test can assert the reduction. The
// arithmetic balances: Events == DroppedNoTree + (the sum of the emitted rows'
// per-pass occurrence deltas), and Rows == Refreshed + Opened == Signatures.
type CoalesceStats struct {
	Events        int // crash events offered to the fold
	Rows          int // rows emitted (one per distinct signature — the coalesced output)
	Signatures    int // distinct signatures seen (equals Rows; named for legibility)
	Refreshed     int // signatures that already had a live row, whose count climbed
	Opened        int // signatures that opened a fresh window (new, or previously retracted)
	DroppedNoTree int // events dropped for having no usable tree glob (never silently)
}

// CoalesceCrashes folds crash events into ONE occurrence-counted record per
// signature, given the ledger rows already written (prior). It returns the rows to
// append — refreshed rows for signatures whose window is still live, freshly opened
// rows for the rest — plus the stats of what it folded.
//
// The three properties it guarantees, which are the point of the whole exercise:
//
//   - N events sharing a signature yield ONE row carrying occurrence_count = N,
//     not N rows.
//   - Two distinct signatures yield two rows — a cause is deduped, causes are not
//     merged.
//   - Re-emission while the window is live INCREMENTS the existing count (the
//     returned row supersedes the prior one for that signature) rather than opening
//     a second live row, so LiveRecords/Match still show exactly one.
//
// A refreshed row keeps the original DiscoveredAtUnix — the window is anchored at
// first sighting, so an ongoing storm does not indefinitely renew its own hold, and
// the row still self-heals via the existing TTL. A newly opened row is discovered at
// nowUnix (the emit instant, clock-injected by the shell) rather than at its oldest
// event, so replaying a batch of old events cannot mint a row that is born already
// expired.
//
// Emission order is first-seen-event order, so the output is deterministic.
func CoalesceCrashes(prior []Record, events []CrashEvent, nowUnix, ttlSeconds int64) ([]Record, CoalesceStats) {
	stats := CoalesceStats{Events: len(events)}

	type group struct {
		class    string
		globs    []string
		hash     string
		by       string
		count    int64
		lastSeen int64
	}
	groups := make(map[string]*group, len(events))
	order := make([]string, 0, len(events))

	for _, ev := range events {
		globs := normalizeAll(ev.TreeGlobs)
		if len(globs) == 0 {
			// A row with no tree could never intersect a query — it would be an
			// unmatchable ghost on the operator surface. Drop it, but COUNT it: a
			// silently swallowed crash is the same honesty hole this issue exists to close.
			stats.DroppedNoTree++
			continue
		}
		class := crashClass(ev.ReasonClass)
		hash := strings.TrimSpace(ev.FailureHash)
		sig := Signature(class, globs, hash)
		g, seen := groups[sig]
		if !seen {
			g = &group{class: class, globs: globs, hash: hash}
			groups[sig] = g
			order = append(order, sig)
		}
		g.count++
		if ev.AtUnix > g.lastSeen {
			g.lastSeen = ev.AtUnix
		}
		if g.by == "" {
			g.by = strings.TrimSpace(ev.ObservedBy)
		}
	}
	stats.Signatures = len(order)

	out := make([]Record, 0, len(order))
	for _, sig := range order {
		g := groups[sig]
		if live, ok := FindLatestLive(prior, sig, nowUnix); ok {
			// The window is still open: supersede the live row with a higher count.
			// Everything else (discovery instant, TTL, tree, any claim bookkeeping) is
			// carried forward untouched — this is a refresh, not a re-discovery.
			out = append(out, live.WithOccurrences(live.OccurrenceCount+g.count, g.lastSeen))
			stats.Refreshed++
			continue
		}
		rec := NewRecord(g.class, g.globs, "", g.by, g.hash, nowUnix, ttlSeconds).
			WithOccurrences(g.count, g.lastSeen)
		out = append(out, rec)
		stats.Opened++
	}
	stats.Rows = len(out)
	return out, stats
}

// WithOccurrences returns a copy of r stamped with a coalesced occurrence count and
// the newest observation instant behind it. It follows the same builder idiom as
// WithDerivedFrom: a count of 0 (and a lastSeen of 0) leaves the row byte-identical
// to a pre-#3586 row, because both fields are omitempty — so the coalescer is fully
// backward-compatible with every ledger row already written.
//
// A lastSeenAtUnix older than the one already on the row is ignored: last-seen only
// ever moves forward, so folding an out-of-order batch cannot rewind a signature's
// most-recent sighting. A negative count clamps to 0.
func (r Record) WithOccurrences(count, lastSeenAtUnix int64) Record {
	out := r
	if count < 0 {
		count = 0
	}
	out.OccurrenceCount = count
	if lastSeenAtUnix > out.LastSeenAtUnix {
		out.LastSeenAtUnix = lastSeenAtUnix
	}
	return out
}

// Coalesced reports whether a row carries a coalesced crash count (more than one
// occurrence folded into it) — the predicate an operator surface uses to render
// "300 crashes, 1 cause" instead of a bare signature line.
func (r Record) Coalesced() bool { return r.OccurrenceCount > 1 }
