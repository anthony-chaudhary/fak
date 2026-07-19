package superloop

// roster.go — the ONE canonical loop-fleet roster (issue #4955): the single answer
// to "which loops + super loops does the operator supervise?". Before this file the
// answer was split and hand-maintained: the super-loop registry names loops
// one-by-one as KindLoop refs, while loopfleet.Fold independently folds EVERY
// ledgered loop into a health row — and the two never reconciled, so a loop that
// was ledgered-and-folding but not hand-named in any intent was invisible to the
// walk. [BuildRoster] unions the three sources — loopfleet's folded loops, the
// loopmgr job registry, and the registered super loops — deduped on each loop's
// stable identity, so everything supervised is listed exactly once.
//
// Same shell/core split as the rest of the package: this stays PURE (no file, no
// clock — it folds inputs the impure shell read), and honest in the
// declared-vs-measured sense: a loop declared somewhere but backed by no foldable
// ledger reads UNMEASURED (never dropped, never a healthy zero), and a skipped
// ledger stays a surfaced KNOWN gap. The worst-first walk over this roster is the
// meta-walker follow-on (#4958); the SPINNING verdict is #4956.

import (
	"fmt"
	"sort"
	"strings"
)

// RosterSchema is the versioned payload tag the `fak superloop roster --json`
// surface emits.
const RosterSchema = "fak.superloop-roster.v1"

// The roster source tokens: which of the three unioned sources claims an entry.
const (
	// RosterSourceFold: the cross-ledger loop-health fold (loopfleet) folded a real
	// ledger row for this loop — the MEASURED source.
	RosterSourceFold = "fold"
	// RosterSourceLoopRegistry: the loopmgr job registry declares this loop (a
	// persisted schedule definition), whether or not its ledger has rows yet.
	RosterSourceLoopRegistry = "loop-registry"
	// RosterSourceSuperloop: the super-loop registry claims it — either the entry IS
	// a registered intent, or an intent hand-names the loop as a KindLoop member ref.
	RosterSourceSuperloop = "superloop"
)

// RosterLoop is the shell-read identity + folded state of ONE ledgered loop —
// the plain-data mirror of a loopfleet.LoopHealth row (kept as data so this
// package stays pure and import-light). Kind is the loop's stable identity
// ("cadence", "dispatch", "loopmgr:<loop_id>", ...), the dedupe key.
type RosterLoop struct {
	Kind  string `json:"kind"`
	State string `json:"state,omitempty"`
	Dark  bool   `json:"dark,omitempty"`
}

// RosterGap is a ledger that could not be folded — the plain-data mirror of a
// loopfleet.Skipped row. It is surfaced on the roster (and enumerated UNMEASURED
// by [LoopFleetStatuses]) so "absent" reads as a known gap, never a healthy zero.
type RosterGap struct {
	Ledger string `json:"ledger"`
	Path   string `json:"path,omitempty"`
	Reason string `json:"reason"`
}

// RosterEntry is one supervised thing — a ledgered loop or a registered super
// loop — counted exactly once. ID is the stable identity the union dedupes on:
// the loop kind for loops, "superloop:<name>" for intents.
type RosterEntry struct {
	ID   string     `json:"id"`
	Kind MemberKind `json:"kind"`
	// Sources lists which of the three unioned sources claim this entry (sorted).
	Sources []string `json:"sources"`
	// Named is true when some registered intent hand-names this loop as a KindLoop
	// member ref — false marks the exact loops that were invisible to the walk
	// before this roster (ledgered-and-folding but never named).
	Named bool `json:"named"`
	// Measured is true when a folded ledger row backs this entry (or, for a
	// registered super loop, always: its status is computable by walking its
	// members). A declared-but-unfoldable loop reads UNMEASURED — a known gap.
	Measured bool `json:"measured"`
	// State/Dark carry the folded health verdict when Measured (loop entries only).
	State  string `json:"state,omitempty"`
	Dark   bool   `json:"dark,omitempty"`
	Detail string `json:"detail,omitempty"`
}

// Roster is the canonical deduped union — the source of truth for what the
// operator supervises — plus the surfaced known gaps and an honesty rollup.
type Roster struct {
	Schema  string        `json:"schema"`
	Entries []RosterEntry `json:"entries"`
	// Gaps are the ledgers the fold could not read — surfaced, never silent.
	Gaps []RosterGap `json:"gaps,omitempty"`
	// Rollup: Total == len(Entries); Unnamed counts the loops no intent hand-names
	// (visible here, invisible to the hand-named walk); Unmeasured counts entries
	// with no foldable backing — the declared-vs-measured honesty line.
	Total      int `json:"total"`
	Loops      int `json:"loops"`
	Supers     int `json:"supers"`
	Measured   int `json:"measured"`
	Unmeasured int `json:"unmeasured"`
	Unnamed    int `json:"unnamed"`
}

// BuildRoster is the canonical roster function: the deduped union of the folded
// loops (loopfleet.Fold, adapted by the shell), the loopmgr job registry's loop
// ids, and the registered super loops. Dedupe key is each loop's stable identity
// — a registry job id shares the loopmgr ledger's loop-id space, so it normalizes
// to "loopmgr:<id>" and lands on the SAME entry as its folded row. Entries come
// back sorted by ID (deterministic), each tagged with every source that claims it.
func BuildRoster(folded []RosterLoop, registryIDs []string, supers []Super, gaps []RosterGap) Roster {
	entries := map[string]*RosterEntry{}
	ensure := func(id string, kind MemberKind) *RosterEntry {
		if e, ok := entries[id]; ok {
			return e
		}
		e := &RosterEntry{ID: id, Kind: kind}
		entries[id] = e
		return e
	}
	addSource := func(e *RosterEntry, src string) {
		for _, s := range e.Sources {
			if s == src {
				return
			}
		}
		e.Sources = append(e.Sources, src)
	}

	// Source 1: every folded loop — the measured spine of the roster.
	for _, l := range folded {
		id := strings.TrimSpace(l.Kind)
		if id == "" {
			continue
		}
		e := ensure(id, KindLoop)
		addSource(e, RosterSourceFold)
		e.Measured = true
		e.State = l.State
		e.Dark = l.Dark
	}

	// Source 2: the loopmgr job registry — declared schedules, folded or not. A
	// registry job id shares the ledger's loop-id space, so it dedupes onto the
	// folded "loopmgr:<id>" row when one exists.
	for _, id := range registryIDs {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		e := ensure("loopmgr:"+id, KindLoop)
		addSource(e, RosterSourceLoopRegistry)
	}

	// Source 3: the registered super loops — each intent is itself supervised, and
	// each hand-named KindLoop member ref marks its loop Named.
	for _, s := range supers {
		e := ensure("superloop:"+s.Name, KindSuperloop)
		addSource(e, RosterSourceSuperloop)
		// A registered Super's status is always computable (walk its members), so it
		// is never an unmeasured gap on the roster.
		e.Measured = true
		e.Named = true
		if e.Detail == "" {
			e.Detail = "registered intent — status via `fak superloop walk " + s.Name + "`"
		}
		for _, m := range s.Members {
			if m.Kind != KindLoop {
				continue
			}
			ref := strings.TrimSpace(m.Ref)
			if ref == "" {
				continue
			}
			le := ensure(ref, KindLoop)
			addSource(le, RosterSourceSuperloop)
			le.Named = true
		}
	}

	r := Roster{Schema: RosterSchema, Gaps: gaps}
	ids := make([]string, 0, len(entries))
	for id := range entries {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		e := entries[id]
		sort.Strings(e.Sources)
		if e.Kind == KindLoop {
			r.Loops++
			if !e.Named {
				r.Unnamed++
			}
			if e.Measured {
				if e.Detail == "" {
					e.Detail = "folded state " + e.State
				}
			} else if e.Detail == "" {
				e.Detail = "no foldable ledger — UNMEASURED (known gap, not a healthy zero)"
			}
		} else {
			r.Supers++
		}
		if e.Measured {
			r.Measured++
		} else {
			r.Unmeasured++
		}
		r.Entries = append(r.Entries, *e)
	}
	r.Total = len(r.Entries)
	return r
}

// LoopFleetStatuses expands ONE KindLoopFleet member into the per-loop statuses a
// walk weighs — the KindTrajectory/"open" enumeration precedent applied to the
// whole fleet. A Ref of "all" (or empty) enumerates every folded loop; any other
// Ref selects one loop by its stable identity. Honesty rules, in order:
//
//   - every folded loop becomes one MEASURED status carrying its identity as Ref
//     (dark carries Dark; a stale loop carries one unit of debt; live is clean);
//   - every skipped ledger becomes one UNMEASURED status — a known gap that blocks
//     Satisfied, never dropped, never counted as a healthy zero;
//   - a selected loop with no folded row, or an empty fleet, is UNMEASURED too.
//
// Output order is deterministic (loops sorted by identity, then gaps by ledger).
func LoopFleetStatuses(src Member, folded []RosterLoop, gaps []RosterGap) []MemberStatus {
	ref := strings.TrimSpace(src.Ref)
	selectOne := ref != "" && ref != "all"

	loops := append([]RosterLoop(nil), folded...)
	sort.Slice(loops, func(i, j int) bool { return loops[i].Kind < loops[j].Kind })

	out := make([]MemberStatus, 0, len(loops)+len(gaps))
	for _, l := range loops {
		if selectOne && l.Kind != ref {
			continue
		}
		// The liveness slice of the meta-walk's worst-first product (#4958): stale = 1
		// unit, live = 0, and a dark loop's urgency rides the Dark flag (tier 0, never
		// double-counted). The shell re-applies FleetDebt once the progress (#4956) and
		// follow-on (#4957) verdicts are read, so this stays the single source of truth.
		out = append(out, MemberStatus{
			Member:   Member{Kind: KindLoopFleet, Ref: l.Kind, Why: src.Why},
			Measured: true,
			Debt:     FleetDebt(l.State, l.Dark, "", ""),
			Dark:     l.Dark,
			Detail:   fmt.Sprintf("fleet loop %s — state %s", l.Kind, l.State),
		})
	}
	if !selectOne {
		sorted := append([]RosterGap(nil), gaps...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Ledger < sorted[j].Ledger })
		for _, g := range sorted {
			out = append(out, MemberStatus{
				Member:   Member{Kind: KindLoopFleet, Ref: g.Ledger, Why: src.Why},
				Measured: false,
				Detail:   fmt.Sprintf("ledger %s (%s): %s — known gap, loop health unmeasured", g.Ledger, g.Path, g.Reason),
			})
		}
	}
	if len(out) == 0 {
		detail := "no loop ledger folded on this host — fleet health unmeasured"
		if selectOne {
			detail = fmt.Sprintf("roster loop %q has no foldable ledger here — UNMEASURED, not clean", ref)
		}
		return []MemberStatus{{Member: Member{Kind: KindLoopFleet, Ref: src.Ref, Why: src.Why}, Measured: false, Detail: detail}}
	}
	return out
}
