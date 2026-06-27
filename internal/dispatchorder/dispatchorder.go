// Package dispatchorder is the deterministic decision the fak issue-dispatch loop is
// missing: given a set of candidate work units, which one should a worker pick FIRST, and
// which ones are stale duplicates that should not run at all? It is the computable answer to
// the operator question "25 tasks were spawned for the same thing — only the freshest should
// run, and the rest should be superseded, not re-attempted."
//
// # The gap it closes
//
// The dispatcher's current pick (tools/issue_resolve_dispatch.py: pick_target_issue) walks a
// lane's issue numbers in reverse-numeric order and returns the first that is not SKIPPED by
// the in-flight set (a live worker) or the cooldown set (attempted within the window). That is
// a SKIP policy, never a COLLAPSE policy: N units targeting the same thing all stay eligible,
// and "freshest" means the largest issue NUMBER, not the most recently updated unit. This
// package adds the collapse: among units that share a supersede KEY it keeps only the
// most-recently-updated one and marks the others SUPERSEDED, then orders the survivors by real
// recency. The existing live/cooldown skips are folded in as dispositions so one pass yields
// both the pick order and a full, auditable account of every unit.
//
// # Pure and total
//
// Plan takes a clock reading (NowUnix) as data, never reads one, and imports nothing internal
// — same inputs, same Result, no I/O. The impure half (gather the candidates from `gh` and the
// run-dir sidecars, then act on the order) lives in the cmd/fak shell, exactly the leaf/shell
// split internal/resume (the decision) and cmd/fak/resume_scan.go (the wire) use.
package dispatchorder

import "sort"

// DefaultCooldownSeconds mirrors the dispatcher's --cooldown-min default (120 minutes) so the
// leaf agrees with the live picker when the caller does not pin a window.
const DefaultCooldownSeconds = 120 * 60

// Disposition is what the planner decided to do with one candidate this tick.
type Disposition string

const (
	// DispKeep: the freshest live-eligible unit for its supersede key — a worker should pick it.
	// The Keep slice lists the DispKeep units in dispatch order.
	DispKeep Disposition = "keep"
	// DispSuperseded: an older duplicate of a kept (or running) unit sharing its key — it should
	// NOT run; a fresher unit covers the same target. SupersededBy names the unit that won.
	DispSuperseded Disposition = "superseded"
	// DispLive: a worker is already running this unit — skip it this tick (not a duplicate to run,
	// not stale to collapse; it is in flight).
	DispLive Disposition = "live"
	// DispCooling: this unit is the freshest for its key but was attempted within the cooldown
	// window — skip it THIS tick (and do not fall back to an older duplicate), retry once it cools.
	DispCooling Disposition = "cooling"
)

// The closed reason vocabulary for a Ranked.Reason, so an observability sink records WHY
// without any free text.
const (
	// ReasonFreshest: the most recently updated unit for its key (or the sole unit for it).
	ReasonFreshest = "freshest"
	// ReasonSuperseded: a fresher unit shares this unit's supersede key.
	ReasonSuperseded = "superseded_by_fresher"
	// ReasonWorkerLive: a worker is already running this unit.
	ReasonWorkerLive = "worker_live"
	// ReasonCooldown: the freshest unit for its key was attempted within the cooldown window.
	ReasonCooldown = "cooldown"
)

// Candidate is one unit of dispatchable work — all the facts the order needs, none of the
// payload. The caller supplies Key: units that share a non-empty Key are duplicates of one
// target and collapse to the freshest. A unit with an empty Key is unique by construction
// (never superseded), the opt-out for work whose target identity is unknown.
type Candidate struct {
	// ID is the unit's identity (an issue number as a string, a task id). Echoed in the result.
	ID string `json:"id"`
	// Key is the supersede/target identity. Units sharing a non-empty Key are the same target;
	// only the freshest survives. Empty Key => unique (its own group, never superseded).
	Key string `json:"key"`
	// CreatedUnix is when the unit was created (0 = unknown); the recency fallback.
	CreatedUnix int64 `json:"created_unix"`
	// UpdatedUnix is when the unit was last updated (0 = unknown); the PRIMARY recency signal.
	UpdatedUnix int64 `json:"updated_unix"`
	// LastAttemptUnix is when a worker was last spawned for this unit (0 = never); the cooldown input.
	LastAttemptUnix int64 `json:"last_attempt_unix"`
	// Live reports that a worker is currently running this unit (the in-flight skip).
	Live bool `json:"live"`
}

// recency is the unit's freshness: its last update, falling back to its creation time.
func (c Candidate) recency() int64 {
	if c.UpdatedUnix > 0 {
		return c.UpdatedUnix
	}
	return c.CreatedUnix
}

// Ranked is one candidate with the planner's verdict attached.
type Ranked struct {
	Candidate
	// Disposition is what to do with this unit this tick (keep / superseded / live / cooling).
	Disposition Disposition `json:"disposition"`
	// Reason is the closed token explaining the disposition.
	Reason string `json:"reason"`
	// SupersededBy is the winning unit's ID when Disposition is DispSuperseded; empty otherwise.
	SupersededBy string `json:"superseded_by,omitempty"`
	// Recency is the freshness value the unit was judged on (echoed for transparency).
	Recency int64 `json:"recency"`
	// Rank is the 0-based dispatch position among DispKeep units; -1 for everything else.
	Rank int `json:"rank"`
}

// Input is everything Plan needs: the candidates, the clock as data, and the cooldown window.
type Input struct {
	Candidates []Candidate `json:"candidates"`
	// NowUnix is the current time as data (the leaf never reads a clock).
	NowUnix int64 `json:"now_unix"`
	// CooldownSeconds is the attempt-cooldown window (0 => DefaultCooldownSeconds). Negative
	// disables the cooldown (no unit is ever held for it).
	CooldownSeconds int64 `json:"cooldown_seconds"`
}

// Result is the full deterministic verdict: every candidate's disposition plus the freshest-
// first pick list.
type Result struct {
	// Order is every candidate, DispKeep units first in dispatch order, then the rest by recency.
	Order []Ranked `json:"order"`
	// Keep is the IDs a worker should pick, freshest-first — Order's DispKeep units, in rank order.
	Keep []string `json:"keep"`
	// Counts of each disposition, so a one-line summary needs no fold.
	KeepCount       int `json:"keep_count"`
	SupersededCount int `json:"superseded_count"`
	LiveCount       int `json:"live_count"`
	CoolingCount    int `json:"cooling_count"`
}

// Pick is the single unit a worker should take this tick — Keep[0], or "" when nothing is
// dispatchable (every candidate is superseded, live, or cooling).
func (r Result) Pick() string {
	if len(r.Keep) == 0 {
		return ""
	}
	return r.Keep[0]
}

// Plan is THE deterministic dispatch-order decision: same Input in, same Result out — no clock,
// no I/O. It collapses same-key duplicates to the freshest unit, folds in the live/cooldown
// skips, and returns the survivors in freshest-first order. Total over any input (an empty
// candidate set yields an empty, defined Result).
//
// The policy, in order:
//  1. Group candidates by Key. A non-empty Key groups duplicates; an empty Key is its own group.
//  2. In each group the WINNER is the unit with the greatest recency (tie: greater CreatedUnix,
//     then greater ID), INCLUDING a live or cooling winner — a duplicate never out-ranks the
//     freshest just because the freshest is busy.
//  3. Disposition per unit, by precedence: a live unit is DispLive; a non-winner (with a Key)
//     is DispSuperseded by the winner; the winner is DispCooling if it was attempted within the
//     cooldown window, else DispKeep.
//  4. DispKeep units are ordered freshest-first and assigned a rank; Keep lists their IDs.
//
// A group whose winner is live or cooling yields NO keep this tick (the dispatcher waits for the
// freshest rather than running a stale duplicate) — the deliberate v1 posture; a max-backoff
// fallback to the next-freshest is a separate, later rung.
func Plan(in Input) Result {
	cooldown := in.CooldownSeconds
	if cooldown == 0 {
		cooldown = DefaultCooldownSeconds
	}

	winner := winnersByKey(in.Candidates)

	ranked := make([]Ranked, 0, len(in.Candidates))
	for _, c := range in.Candidates {
		r := Ranked{Candidate: c, Recency: c.recency(), Rank: -1}
		switch {
		case c.Live:
			r.Disposition, r.Reason = DispLive, ReasonWorkerLive
		case c.Key != "" && winner[c.Key] != c.ID:
			r.Disposition, r.Reason, r.SupersededBy = DispSuperseded, ReasonSuperseded, winner[c.Key]
		case cooldown > 0 && c.LastAttemptUnix > 0 && in.NowUnix-c.LastAttemptUnix < cooldown:
			r.Disposition, r.Reason = DispCooling, ReasonCooldown
		default:
			r.Disposition, r.Reason = DispKeep, ReasonFreshest
		}
		ranked = append(ranked, r)
	}

	// Order: DispKeep first by recency (freshest-first), then the rest by recency, stable and
	// deterministic. Ranks and the Keep list are assigned from the kept prefix.
	sort.SliceStable(ranked, func(i, j int) bool {
		ki, kj := ranked[i].Disposition == DispKeep, ranked[j].Disposition == DispKeep
		if ki != kj {
			return ki // kept units sort ahead of skipped ones
		}
		return moreRecent(ranked[i], ranked[j])
	})

	out := Result{Order: ranked}
	for i := range out.Order {
		switch out.Order[i].Disposition {
		case DispKeep:
			out.Order[i].Rank = len(out.Keep)
			out.Keep = append(out.Keep, out.Order[i].ID)
			out.KeepCount++
		case DispSuperseded:
			out.SupersededCount++
		case DispLive:
			out.LiveCount++
		case DispCooling:
			out.CoolingCount++
		}
	}
	return out
}

// winnersByKey returns, for each non-empty Key, the ID of the freshest candidate sharing it
// (the supersede winner). Units with an empty Key are not grouped (each is its own winner).
func winnersByKey(cands []Candidate) map[string]string {
	best := make(map[string]Candidate)
	for _, c := range cands {
		if c.Key == "" {
			continue
		}
		if cur, ok := best[c.Key]; !ok || beats(c, cur) {
			best[c.Key] = c
		}
	}
	winner := make(map[string]string, len(best))
	for k, c := range best {
		winner[k] = c.ID
	}
	return winner
}

// beats reports whether a is the fresher duplicate than b: greater recency, then greater
// CreatedUnix, then greater ID (a total, deterministic order with no ties).
func beats(a, b Candidate) bool {
	if a.recency() != b.recency() {
		return a.recency() > b.recency()
	}
	if a.CreatedUnix != b.CreatedUnix {
		return a.CreatedUnix > b.CreatedUnix
	}
	return a.ID > b.ID
}

// moreRecent is beats lifted to Ranked, for the final ordering of equally-disposed units.
func moreRecent(a, b Ranked) bool { return beats(a.Candidate, b.Candidate) }
