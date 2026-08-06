package hooks

// candidates.go — the CANDIDATE DENOMINATOR (#5602): how many staged items each gate's own
// filter admitted for judgement.
//
// `fak hooks pre-commit --json` reported verdicts and nothing else, so one payload covered two
// different worlds: a staged set where every gate judged forty files and each found nothing,
// and a staged set where every gate ran but its own filter admitted ZERO items — no .go file
// for GOFMT, no doc for INDEX_SYNC, no new internal/<leaf>/ for UNTIERED_LEAF. Both render as
// {"findings": [], "count": 0}. On the human path a fully clean run printed nothing at all,
// and silence reads as "checked, nothing owed" when it equally means "nothing was checked".
//
// #5299 already counts the GATE denominator ("%d of %d gate(s) skipped"), so a gate that
// produced no verdict is visible. This is the other half — the domain size of a gate that DID
// produce one. Together they answer "what was actually checked", which is the trust floor under
// every individual gate's precision (epic #5601).
//
// WHERE THE COUNT COMES FROM, and why it is not a separate derivation. Each gate records its
// own denominator at the point its filter is ALREADY computed, inside Check. Deriving the count
// somewhere else — a second function per gate, or a re-walk of StagedDiff by the runner — would
// be a second implementation of every gate's filter, free to drift from the set the gate
// actually judged. A denominator that disagrees with the decision it describes is worse than no
// denominator: it is a wrong number wearing the authority of a measurement. Recording it in
// place also satisfies the wall-clock constraint (#5335) for free, since the count is a length
// of something already in hand and spawns no further git read.
//
// UNREPORTED IS NOT ZERO. A gate that records nothing reports UNREPORTED — Candidates returns
// ok=false, and the CLI renders JSON null. Zero cannot double as "no answer" here the way
// Finding.Severity's zero means UNGRADED, because for a denominator zero is itself a real and
// load-bearing answer: "this gate ran and judged nothing", which is precisely the state this
// issue exists to make visible. Collapsing the two would rebuild the ambiguity one level up.

import "sort"

// candidateNote is one gate's recorded denominator: how many items it admitted, and the unit
// they are counted in. The unit is per gate on purpose — a file-scoped gate counts files and a
// line-scoped gate counts added lines, and naming which one it is matters more than forcing a
// single unit across gates that do not share one.
type candidateNote struct {
	n    int
	unit string
}

// NoteCandidates records how many staged items this gate's filter admitted for judgement.
// A gate calls it from inside its own Check, at the site the filter is computed.
//
// The last call for a gate wins, so a gate that narrows its filter in stages can record the
// final domain it actually judged rather than an intermediate one.
//
// Safe to call from an ABANDONED gate. The pre-commit CLI bounds each gate with a wall clock
// and abandons one that overruns (#5335) without cancelling it — Gate.Check takes no context —
// so that Check keeps running against this same StagedDiff while the loop hands it to the next
// gate. Two gates can therefore reach this ledger at once, and an unsynchronized map write is a
// Go RUNTIME FATAL that would kill the very commit the budget exists to let through. Hence the
// same mutex discipline fileCache uses.
func (d *StagedDiff) NoteCandidates(gate string, n int, unit string) {
	if d == nil || gate == "" {
		return
	}
	d.candMu.Lock()
	defer d.candMu.Unlock()
	if d.candidates == nil {
		d.candidates = make(map[string]candidateNote)
	}
	d.candidates[gate] = candidateNote{n: n, unit: unit}
}

// Candidates returns the denominator gate recorded and the unit it counted in. ok=false means
// the gate recorded NOTHING — it is UNREPORTED, which a caller must not render as 0.
func (d *StagedDiff) Candidates(gate string) (n int, unit string, ok bool) {
	if d == nil {
		return 0, "", false
	}
	d.candMu.Lock()
	defer d.candMu.Unlock()
	note, ok := d.candidates[gate]
	return note.n, note.unit, ok
}

// ReportedGates lists, sorted, every gate that recorded a denominator this run. Used by tests
// and by the CLI to tell "no gate reports a count" from "every gate reported zero".
func (d *StagedDiff) ReportedGates() []string {
	if d == nil {
		return nil
	}
	d.candMu.Lock()
	defer d.candMu.Unlock()
	names := make([]string, 0, len(d.candidates))
	for name := range d.candidates {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
