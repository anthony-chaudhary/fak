package metrics

// crash_spam.go — the TICKETING half of the crash-signature coalescer (#3586):
// turn the coalesced crash ledger into ONE deduped, occurrence-counted ticket per
// signature per window, and render the "300 crashes, 1 cause" operator line.
//
// knownbad.CoalesceCrashes already folds N CHILD_CRASH events into one
// occurrence-counted ledger ROW per signature, and LiveRecords collapses the
// append-to-supersede ledger to one live row per cause. That closes the row half of
// the spam. It does NOT close the ISSUE half: nothing decides whether a coalesced
// row should mint a ticket, so an auto-filer that files per row per pass still opens
// a fresh issue every window even though the cause has not changed — and one
// poisoned tree that kills 300 workers still lands 300 filings on the first pass.
// Filing a ticket is itself fleet work, so an unbudgeted filer is not merely noisy:
// it burns the fleet exactly when a systematic fault is already burning it.
//
// This fold is that budget. Given the ledger and the filer's persisted dedup state,
// it decides per signature: OPEN one ticket (this cause has no ticket for this
// window), REFRESH it (the cause is the same, only the count climbed — edit in
// place), or SUPPRESS (nothing moved since the last filing; make no call at all).
// The cost of a crash storm therefore scales with the number of CAUSES, never with
// the number of dead workers.
//
// "Per window" is the ledger's own TTL window, not a new clock: a signature whose
// live row was DISCOVERED after the filed ticket's window anchor is a genuinely new
// window (the prior one lapsed, or was resolved and the failure came back), so it
// opens a new ticket rather than resurrecting a stale thread. Inside one window the
// count only ever climbs onto the SAME ticket, which is what makes re-emission an
// increment instead of a duplicate.
//
// It lives in the metrics layer as a PURE fold — nowUnix is data, never a clock
// read, and it makes no gh call — because the shells that would wire it
// (dogfoodissues.Sync at tier 3, guardcomplaint at tier 4) sit far above this leaf:
// importing either would red architest with ARCH_LAYER_VIOLATION. The I/O (reading
// the ledger, loading the seen-cache, the create/edit itself) belongs to whichever
// lane wires it; the decision and the readout belong here. The emitted action maps
// one-to-one onto a dogfoodissues PlanRow, and its Key is derived from
// knownbad.LeaseID so the ticket, the ledger row, and the fixer-election lease all
// name the same cause with the same sanitized id.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// CrashSpamSchema versions the readout so a consumer (an operator card, a tick-stream
// row, a JSON reader) can pin the shape.
const CrashSpamSchema = "fak-crash-spam-fold/1"

// Ticket actions — the CLOSED vocabulary a per-signature decision lands on. A closed
// set (mirroring the ledger's status vocabulary) keeps the filer's accounting stable:
// every live crash signature maps to exactly one of these on every pass.
const (
	// CrashTicketOpen — this cause has no ticket covering its current window: file
	// exactly ONE issue carrying the occurrence count. Emitted at most once per
	// signature per window, however many workers died.
	CrashTicketOpen = "open"
	// CrashTicketRefresh — a ticket already covers this window and the count climbed:
	// edit that issue in place so it reads the new total. This is the re-emission path,
	// and it is an UPDATE precisely so a sustained storm never opens a second thread.
	CrashTicketRefresh = "refresh"
	// CrashTicketSuppress — a ticket already covers this window and nothing moved since
	// it was last written: make no call. This is the action that carries the savings; a
	// steady-state pass over a known storm costs zero filings.
	CrashTicketSuppress = "suppress"
)

// crashTicketKeyPrefix tags the filer's dedup key so the marker in a filed body is
// greppable and cannot collide with another filing surface's key space.
const crashTicketKeyPrefix = "crash-"

// FiledCrashTicket is the auto-filer's persisted dedup state for ONE signature — the
// projection of whatever seen-cache the wiring lane keeps (a dogfoodissues SeenCache
// row, a gh issue carrying the marker). It records what the existing ticket already
// SAYS, which is the only thing the fold needs to tell an increment from a no-op.
type FiledCrashTicket struct {
	// Signature is the knownbad signature the ticket was filed for.
	Signature string
	// Number is the filed issue number, carried through onto a refresh/suppress so the
	// shell edits rather than searches. Zero when the state came from a cache that does
	// not track numbers.
	Number int
	// Occurrences is the count the ticket's body currently states — NOT the count of
	// crashes seen. A refresh fires only when the ledger has moved past this.
	Occurrences int64
	// WindowAtUnix is the DiscoveredAtUnix of the ledger row this ticket was filed for:
	// the anchor of the window it covers. A live row discovered after this anchor is a
	// NEW window and opens a new ticket instead of editing a lapsed one.
	WindowAtUnix int64
}

// CrashTicketAction is the fold's decision for one live crash signature: what the
// filer should do, plus everything it needs to do it without re-reading the ledger.
// It maps one-to-one onto a dogfoodissues PlanRow (Action/Key/Number/Title).
type CrashTicketAction struct {
	// Action is one of CrashTicketOpen / CrashTicketRefresh / CrashTicketSuppress.
	Action string `json:"action"`
	// Key is the stable dedup key the filed body carries in its marker.
	Key string `json:"key"`
	// Signature is the knownbad signature this ticket covers — one ticket, one cause.
	Signature string `json:"signature"`
	// ReasonClass and TreeGlobs are the human-readable halves of the signature.
	ReasonClass string   `json:"reason_class"`
	TreeGlobs   []string `json:"tree_globs,omitempty"`
	// Title is the operator-legible summary carrying the count, so "300 crashes" reads
	// at a glance from an issue list without opening the body. A refresh restates it.
	Title string `json:"title"`
	// Occurrences is the cumulative crash count this window has folded — the number the
	// ticket should state.
	Occurrences int64 `json:"occurrences"`
	// NewSince is how many crashes arrived since the ticket last stated its count (the
	// full count on an open, the delta on a refresh, 0 on a suppress) — the "what
	// changed" line an edit can lead with.
	NewSince int64 `json:"new_since"`
	// Number is the existing issue to edit on a refresh/suppress; 0 on an open.
	Number int `json:"number,omitempty"`
	// FirstSeenAtUnix / LastSeenAtUnix span the window, so the ticket reads as a
	// duration ("first seen X, still crashing at Y") rather than a bare count.
	FirstSeenAtUnix int64 `json:"first_seen_at_unix"`
	LastSeenAtUnix  int64 `json:"last_seen_at_unix,omitempty"`
}

// CrashSpamReadout is one pass of the fold: the per-signature actions plus the
// at-a-glance accounting an operator (or a cost check) reads to see that hundreds of
// deaths cost a handful of filings.
type CrashSpamReadout struct {
	Schema string `json:"schema"`
	// Crashes is the total occurrence count across every live crash signature.
	Crashes int64 `json:"crashes"`
	// Causes is the number of distinct live crash signatures — the "1 cause" half.
	Causes int `json:"causes"`
	// Tickets is how many gh calls this pass costs (opens + refreshes). This is the
	// number that must stay small while Crashes grows.
	Tickets int `json:"tickets"`
	// Opened / Refreshed split Tickets by kind.
	Opened    int `json:"opened"`
	Refreshed int `json:"refreshed"`
	// Suppressed counts signatures whose ticket was already current — filings avoided.
	Suppressed int `json:"suppressed"`
	// DroppedUnkeyed counts live rows whose signature yielded no usable dedup key. Such
	// a row could only be filed as an UNDEDUPABLE issue — the exact spam this fold
	// exists to stop — so it is dropped, but counted, never silently swallowed.
	DroppedUnkeyed int `json:"dropped_unkeyed"`
	// Actions are the per-signature decisions, loudest cause first.
	Actions []CrashTicketAction `json:"actions,omitempty"`
}

// Line renders the at-a-glance readout the issue asks for: "300 crashes, 1 cause".
// It is the one string an operator surface prints when a storm is in flight, and it
// is legible precisely because the second number stays small while the first does not.
func (r CrashSpamReadout) Line() string {
	return fmt.Sprintf("%d %s, %d %s",
		r.Crashes, plural(r.Crashes, "crash", "crashes"),
		r.Causes, plural(int64(r.Causes), "cause", "causes"))
}

// Amplification is how many crashes each filing carries — the spam-reduction factor
// (300 crashes on 1 ticket is 300). It is 0 when the pass files nothing, which is the
// best case rather than a degenerate one: a storm already ticketed costs nothing.
func (r CrashSpamReadout) Amplification() float64 {
	if r.Tickets == 0 {
		return 0
	}
	return float64(r.Crashes) / float64(r.Tickets)
}

// CrashTicketKey derives the filer's stable dedup key for a crash signature. It is
// built from knownbad.LeaseID so the ticket, the ledger row, and the fixer-election
// lease all name the SAME cause under the same sanitized id — a parked class and its
// ticket are greppably one thing. It returns "" for a signature with no usable
// content, which the fold treats as unkeyable rather than filing an undedupable issue.
func CrashTicketKey(signature string) string {
	id := knownbad.LeaseID(signature)
	if id == "" {
		return ""
	}
	return crashTicketKeyPrefix + id
}

// FoldCrashSpam decides, for every live crash signature on the ledger, whether the
// auto-filer should open, refresh, or suppress its ticket — given what it has already
// filed. It returns the actions plus the pass accounting.
//
// The three properties that are the point of the exercise, and that its tests pin:
//
//   - N crashes sharing a signature cost ONE ticket carrying N, not N tickets.
//   - Two distinct signatures cost two tickets — a cause is deduped, causes are not
//     merged.
//   - Re-emission inside the window REFRESHES the count on the existing ticket rather
//     than opening a second one, and a pass where nothing moved files nothing at all.
//
// Only rows carrying an occurrence count are considered: a count of at least 1 is the
// mark of a row the crash coalescer minted, so a hand-recorded known-bad (count 0) is
// left to whatever surface owns it rather than being auto-ticketed here. Resolved,
// revoked, and expired signatures drop out via LiveRecords, so a fixed cause stops
// costing filings the moment it is fixed.
//
// Emission order is loudest-cause-first (signature as the tiebreak), so the readout's
// head is the cause worth looking at and the output is deterministic.
func FoldCrashSpam(ledger []knownbad.Record, filed []FiledCrashTicket, nowUnix int64) CrashSpamReadout {
	index := make(map[string]FiledCrashTicket, len(filed))
	for _, f := range filed {
		sig := strings.TrimSpace(f.Signature)
		if sig == "" {
			continue
		}
		// Last write wins: a later entry is the fresher dedup state for that signature.
		index[sig] = f
	}

	rows := make([]knownbad.Record, 0, len(ledger))
	for _, rec := range knownbad.LiveRecords(ledger, nowUnix) {
		if rec.OccurrenceCount >= 1 {
			rows = append(rows, rec)
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].OccurrenceCount != rows[j].OccurrenceCount {
			return rows[i].OccurrenceCount > rows[j].OccurrenceCount
		}
		return rows[i].Signature < rows[j].Signature
	})

	out := CrashSpamReadout{Schema: CrashSpamSchema}
	for _, rec := range rows {
		key := CrashTicketKey(rec.Signature)
		if key == "" {
			out.DroppedUnkeyed++
			continue
		}
		out.Causes++
		out.Crashes += rec.OccurrenceCount

		act := CrashTicketAction{
			Key:             key,
			Signature:       rec.Signature,
			ReasonClass:     rec.ReasonClass,
			TreeGlobs:       rec.TreeGlobs,
			Occurrences:     rec.OccurrenceCount,
			FirstSeenAtUnix: rec.DiscoveredAtUnix,
			LastSeenAtUnix:  rec.LastSeenAtUnix,
		}
		prior, seen := index[strings.TrimSpace(rec.Signature)]
		switch {
		case !seen, rec.DiscoveredAtUnix > prior.WindowAtUnix:
			// No ticket, or the only ticket covers a window that has since lapsed (or was
			// resolved and the failure came back). Either way this window needs its own.
			act.Action = CrashTicketOpen
			act.NewSince = rec.OccurrenceCount
			out.Opened++
		case rec.OccurrenceCount > prior.Occurrences:
			// Same window, higher count: the storm is ongoing. Edit the count onto the
			// SAME issue — this is the branch that makes re-emission an increment.
			act.Action = CrashTicketRefresh
			act.Number = prior.Number
			act.NewSince = rec.OccurrenceCount - prior.Occurrences
			out.Refreshed++
		default:
			// The ticket already states this count. Making a call here would be pure
			// spam, so the pass costs nothing.
			act.Action = CrashTicketSuppress
			act.Number = prior.Number
			out.Suppressed++
		}
		act.Title = crashTicketTitle(act)
		out.Actions = append(out.Actions, act)
	}
	out.Tickets = out.Opened + out.Refreshed
	return out
}

// crashTicketTitle renders the issue title for one coalesced cause. The count rides
// the title (not just the body) so an operator scanning an issue list sees "300
// crashes" without opening anything; a refresh restates it, which is one edit per
// window pass rather than one per death.
func crashTicketTitle(act CrashTicketAction) string {
	return fmt.Sprintf("crash storm: %s over %s (%d %s, 1 cause)",
		act.ReasonClass, crashTreeSummary(act.TreeGlobs),
		act.Occurrences, plural(act.Occurrences, "crash", "crashes"))
}

// crashTreeSummary names the tree a cause holds, keeping a multi-glob signature to
// one legible title-length phrase.
func crashTreeSummary(globs []string) string {
	switch len(globs) {
	case 0:
		return "(no tree)"
	case 1:
		return globs[0]
	default:
		return fmt.Sprintf("%s +%d more", globs[0], len(globs)-1)
	}
}

// plural picks the singular or plural noun for n.
func plural(n int64, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
