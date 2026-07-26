package wipattr

import (
	"fmt"
	"sort"
)

// This file adds the COST half of orphan attribution (#4320 step-1 finding).
//
// Attribute/Orphans (attr.go) answer "is this dirty hunk at risk?" — a safety
// question. They deliberately say nothing about worth: an ORPHAN blocking 87
// dispatch admissions and an ORPHAN blocking none are indistinguishable in that
// output, so an operator reading it has no work queue, only a list. Rank closes
// that: it joins the dirty set against the dispatch ledger's refusal counts, so
// the orphan actually throttling the fleet sorts to the top.
//
// Measured motivation (docs/dispatch/cmd-lane-split-plan.md, "Step 1 result"):
// 146 of 172 dispatch refusals in the ledger window were DIRTY_PATH_COLLISION,
// and 125 of those named a path dirty for >=2 days — abandoned WIP, not peer
// contention. One file (cmd/fak/version_modules.go) carried 87 of them. So on
// that ledger dispatch concurrency is gated by orphan-WIP hygiene, and the
// highest-leverage sweep is exactly this ranking.
//
// The load-bearing subtlety is STALENESS IS A PROPERTY OF THE CHANGE SET, NOT
// THE FILE. A file can sit untouched for days while being one half of a change
// set whose other half was edited minutes ago — landing the stale half alone
// puts a test referencing a not-yet-committed symbol on the trunk and reds it.
// So Rank classifies on the set's freshest member (SetAgeDays), never on the
// file's own mtime, and names the sibling responsible (FreshestSibling) so the
// verdict is auditable rather than merely asserted.

// BlockState is the closed verdict vocabulary for a dirty path, ordered as a work
// queue: BlockLand is actionable now, BlockWait must not be touched, and the rest
// are hygiene with no throughput to win.
type BlockState string

const (
	// BlockLand: this path blocks >=1 dispatch admission AND its whole change set
	// is idle — the throughput lever. Landing it converts refusals into admissions.
	BlockLand BlockState = "LAND"
	// BlockWait: this path blocks admissions but its change set is live (it, or a
	// sibling, was edited recently). The refusal is a CORRECT transient one; landing
	// here would either clobber a peer mid-edit or commit half a change set.
	BlockWait BlockState = "WAIT"
	// BlockIdle: an idle change set blocking nothing. Landable hygiene, but it buys
	// no concurrency — always ranked below every BlockLand row.
	BlockIdle BlockState = "IDLE"
	// BlockActive: a live change set blocking nothing — ordinary work in flight.
	BlockActive BlockState = "ACTIVE"
)

// DefaultStaleAfterDays is the idleness threshold separating abandoned WIP from a
// live edit. Two days is deliberately conservative: it is longer than any plausible
// pause inside one working session, so a BlockLand verdict means the set was left,
// not merely set down.
const DefaultStaleAfterDays = 2.0

// Blocker is one dirty path as observed in the working tree. Set is the change-set
// key the caller groups by — the package/directory is the usual choice, since a Go
// change set almost never spans packages without also spanning their tests. An empty
// Set makes the path its own singleton set.
type Blocker struct {
	Path    string  `json:"path"`
	Set     string  `json:"set,omitempty"`
	AgeDays float64 `json:"age_days"`
}

// Blocked is the ranked verdict for one dirty path.
type Blocked struct {
	Path   string     `json:"path"`
	Set    string     `json:"set,omitempty"`
	State  BlockState `json:"state"`
	Blocks int        `json:"blocks"` // dispatch admissions this path has refused

	// AgeDays is the path's own idleness; SetAgeDays is its change set's, i.e. the
	// MINIMUM age across the set. SetAgeDays is what State is decided on — see the
	// change-set note at the top of this file. SetAgeDays <= AgeDays always.
	AgeDays    float64 `json:"age_days"`
	SetAgeDays float64 `json:"set_age_days"`

	// FreshestSibling is the set member whose recency set SetAgeDays, populated only
	// when it is a DIFFERENT path than this one — i.e. exactly when this path looks
	// idle on its own mtime but is pinned live by a peer's edit. It is the audit trail
	// for a BlockWait/BlockActive verdict that the file's own age would contradict.
	FreshestSibling string `json:"freshest_sibling,omitempty"`

	// Reason states why this row got its State, in operator-readable terms.
	Reason string `json:"reason"`
}

// Rank joins the dirty working-tree paths against per-path dispatch refusal counts
// and returns one verdict per input path (totality — nothing is dropped), ordered as
// a work queue: LAND first (highest blocks first), then WAIT, IDLE, ACTIVE.
//
// blocks maps a repo-relative path to the number of dispatch admissions it has
// refused (parsed from the ledger by the caller); a path absent from the map blocks
// nothing. staleAfterDays <= 0 falls back to DefaultStaleAfterDays.
//
// Pure: no git, no clock, no I/O. The caller supplies ages so the fold stays
// testable and a run over fixed inputs is byte-identical.
func Rank(dirty []Blocker, blocks map[string]int, staleAfterDays float64) []Blocked {
	if staleAfterDays <= 0 {
		staleAfterDays = DefaultStaleAfterDays
	}

	// Pass 1: per change set, find the freshest (minimum-age) member. Ties resolve to
	// the lexically first path so the chosen sibling — and every Reason quoting it —
	// is deterministic.
	type freshest struct {
		path string
		age  float64
	}
	sets := make(map[string]freshest, len(dirty))
	for _, b := range dirty {
		key := setKey(b)
		cur, seen := sets[key]
		if !seen || b.AgeDays < cur.age || (b.AgeDays == cur.age && b.Path < cur.path) {
			sets[key] = freshest{path: b.Path, age: b.AgeDays}
		}
	}

	out := make([]Blocked, 0, len(dirty))
	for _, b := range dirty {
		fresh := sets[setKey(b)]
		row := Blocked{
			Path:       b.Path,
			Set:        b.Set,
			Blocks:     blocks[b.Path],
			AgeDays:    b.AgeDays,
			SetAgeDays: fresh.age,
		}
		if fresh.path != b.Path {
			row.FreshestSibling = fresh.path
		}
		row.State, row.Reason = classify(row, staleAfterDays)
		out = append(out, row)
	}

	sort.SliceStable(out, func(i, j int) bool {
		if pi, pj := statePriority(out[i].State), statePriority(out[j].State); pi != pj {
			return pi < pj
		}
		if out[i].Blocks != out[j].Blocks {
			return out[i].Blocks > out[j].Blocks // most admissions recovered first
		}
		if out[i].SetAgeDays != out[j].SetAgeDays {
			return out[i].SetAgeDays > out[j].SetAgeDays // then most-abandoned first
		}
		return out[i].Path < out[j].Path
	})
	return out
}

// Landable returns just the BlockLand rows — the sweep's actual work queue, in rank
// order. Mirrors Orphans: a convenience filter that preserves Rank's ordering.
func Landable(rows []Blocked) []Blocked {
	out := make([]Blocked, 0)
	for _, r := range rows {
		if r.State == BlockLand {
			out = append(out, r)
		}
	}
	return out
}

// BlocksRecovered totals the dispatch admissions the Landable rows would unblock —
// the one number that says whether a sweep is worth running at all.
func BlocksRecovered(rows []Blocked) int {
	total := 0
	for _, r := range Landable(rows) {
		total += r.Blocks
	}
	return total
}

func classify(r Blocked, staleAfterDays float64) (BlockState, string) {
	stale := r.SetAgeDays >= staleAfterDays
	switch {
	case r.Blocks > 0 && stale:
		return BlockLand, fmt.Sprintf("blocks %s; change set idle %.1fd — land it to recover them",
			plural(r.Blocks, "dispatch admission"), r.SetAgeDays)
	case r.Blocks > 0 && r.FreshestSibling != "":
		// The change-set trap: idle on its own mtime, pinned live by a sibling. Landing
		// this path alone would commit half a change set and red the trunk.
		return BlockWait, fmt.Sprintf("blocks %s, but sibling %s was touched %.1fd ago — landing this alone commits half a change set",
			plural(r.Blocks, "dispatch admission"), r.FreshestSibling, r.SetAgeDays)
	case r.Blocks > 0:
		return BlockWait, fmt.Sprintf("blocks %s, but was touched %.1fd ago — a peer is mid-edit; the refusal is correct",
			plural(r.Blocks, "dispatch admission"), r.AgeDays)
	case stale:
		return BlockIdle, fmt.Sprintf("idle %.1fd but blocking no admission — hygiene, no concurrency to win", r.SetAgeDays)
	case r.FreshestSibling != "":
		return BlockActive, fmt.Sprintf("change set live (sibling %s touched %.1fd ago)", r.FreshestSibling, r.SetAgeDays)
	default:
		return BlockActive, fmt.Sprintf("touched %.1fd ago — work in flight", r.AgeDays)
	}
}

func statePriority(s BlockState) int {
	switch s {
	case BlockLand:
		return 0
	case BlockWait:
		return 1
	case BlockIdle:
		return 2
	default:
		return 3
	}
}

func setKey(b Blocker) string {
	if b.Set == "" {
		return "\x00path\x00" + b.Path // singleton set, unambiguous against a real key
	}
	return b.Set
}

func plural(n int, noun string) string {
	if n == 1 {
		return fmt.Sprintf("1 %s", noun)
	}
	return fmt.Sprintf("%d %ss", n, noun)
}
