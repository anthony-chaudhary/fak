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

import (
	"sort"
	"strconv"
	"strings"
)

// DefaultCooldownSeconds mirrors the dispatcher's --cooldown-min default (120 minutes) so the
// leaf agrees with the live picker when the caller does not pin a window.
const DefaultCooldownSeconds = 120 * 60

const exactSafeSetLimit = 24

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
	// DispCollisionRisk: this unit was otherwise dispatchable, but the collision-priced fan-out
	// preflight found that launching it in this wave would overlap another kept worker's tree.
	// It is serialized before launch, so the caller can run the safe set now and retry later.
	DispCollisionRisk Disposition = "collision_risk"
	// DispGenerationHeld: this unit is otherwise present in the candidate set, but the current
	// generation window does not admit its horizon. It is held before supersede/order/collision
	// pricing so later-horizon work cannot consume the default dispatch slot.
	DispGenerationHeld Disposition = "generation_held"
	// DispBlocked: this unit names a prerequisite (BlockedBy) that is still an OPEN candidate this
	// tick — a SOFT hold, so it is excluded from Keep but kept in Order with the reason (legible,
	// not silently dropped). Fail-open: a prerequisite ABSENT from the candidate set (already
	// closed) does NOT block. Cycle-safe: a mutual A<->B block breaks toward the lowest ID — the
	// lower keeps the dispatch, the higher is held — so exploratory work never deadlocks.
	DispBlocked Disposition = "blocked"
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
	// ReasonCollisionRisk: the closed DOS refusal token for a pre-launch tree collision.
	ReasonCollisionRisk = "COLLISION_RISK"
	// ReasonGenerationHeld: the candidate's generation is outside the requested/default window.
	ReasonGenerationHeld = "GENERATION_HELD"
	// ReasonBlockedByOpenPrereq: a prerequisite named in BlockedBy is still an open candidate this
	// tick, so the unit is soft-held until the prerequisite closes (leaves the candidate set).
	ReasonBlockedByOpenPrereq = "blocked_by_open_prereq"
	// ReasonWedgedObjective: the unit's bound objective shows a sustained-STALL trajectory curve
	// (ObjectiveSignal "STALL" — flat progress across attempts), so the dispatch order deprioritized
	// it behind a fresh alternative. A DEMOTION, never a refusal: the unit stays DispKeep, stays in
	// Order and Keep, and a lone wedged candidate (no fresh alternative outranking it) is still
	// picked with ReasonFreshest.
	ReasonWedgedObjective = "wedged_objective_stall"
)

// ObjectiveSignalStall is the trajctl curve token for a wedged objective (flat progress plus
// divergence across attempts). It is mirrored here as PLAIN DATA — this pureRoot leaf imports
// nothing internal, so the caller passes the internal/trajctl Signal's string value, never the
// type. Any other token ("HEALTHY", "DRIFT", "DETOUR_OVERRUN", or empty for no objective /
// unknown) leaves the order untouched.
const ObjectiveSignalStall = "STALL"

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
	// Lane is the dos.toml lane this worker would acquire. It is optional for legacy order-only
	// callers, but when any candidate declares a Lane/Tree/Mode the input is treated as a
	// collision-priced fan-out. Same exclusive lane requests serialize before launch.
	Lane string `json:"lane,omitempty"`
	// Tree is the repo-relative file tree(s) this worker would touch, using the same prefix/glob
	// shape as dos arbitrate's lane tree. In priced mode an empty Tree is an unknown blast radius
	// and conservatively collides with every other candidate.
	Tree []string `json:"tree,omitempty"`
	// Mode is the requested lock mode ("exclusive" or "shared"). Empty means exclusive, matching
	// the dispatch lease default. shared/shared may overlap; any exclusive participant must be
	// tree-disjoint.
	Mode string `json:"mode,omitempty"`
	// Compute is the OPTIONAL compute-region claim this candidate makes alongside (or instead of)
	// its file-tree claim — the disaggregation axis of the claim-space (#3268): a device, a
	// prefill/decode phase-pool, an expert-rank interval, a KV-tier span. It is priced through the
	// SAME collision machinery as the file tree — a compute-region overlap is a COLLISION_RISK and
	// emits RepartitionAdvice exactly as a tree overlap does — but on its own resource axis, so a
	// file claim and a compute claim never contend with each other. Nil for every file-only caller,
	// which keeps their pricing byte-identical (the additive-no-regression guarantee).
	Compute *ComputeClaim `json:"compute,omitempty"`
	// CreatedUnix is when the unit was created (0 = unknown); the recency fallback.
	CreatedUnix int64 `json:"created_unix"`
	// UpdatedUnix is when the unit was last updated (0 = unknown); the PRIMARY recency signal.
	UpdatedUnix int64 `json:"updated_unix"`
	// LastAttemptUnix is when a worker was last spawned for this unit (0 = never). It holds
	// recent work for cooldown; after cooldown it marks already-started WIP, which sorts before
	// never-attempted work within the same explicit priority tier.
	LastAttemptUnix int64 `json:"last_attempt_unix"`
	// BlockedBy names the ids of prerequisites this unit depends on — a dependency edge honored as
	// a SOFT hold, never a hard ban. When a listed prerequisite is still an OPEN candidate in the
	// SAME tick's set, this unit is held (DispBlocked): excluded from Keep but kept in Order with
	// the reason, so the dependency is legible instead of silently dropped. Fail-open: a BlockedBy
	// id ABSENT from the candidate set is an already-closed prerequisite and never holds. Cycle-safe:
	// a mutual A<->B block breaks toward the lowest ID (the lower keeps the dispatch, the higher is
	// held), so exploratory work never deadlocks or freezes on a dependency cycle. Empty for every
	// legacy candidate, which keeps their disposition byte-identical (the no-regression guarantee).
	BlockedBy []string `json:"blocked_by,omitempty"`
	// Live reports that a worker is currently running this unit (the in-flight skip).
	Live bool `json:"live"`
	// Generation is the issue generation label ("gen/now", "gen/next", "gen/second-next",
	// "gen/future"). When any candidate declares a generation, the default window admits only
	// now/next work; second-next, future, and unclassified candidates are held unless Input
	// requests that horizon explicitly.
	Generation string `json:"generation,omitempty"`
	// Priority is the declared do-first weight the caller maps from the unit's priority/P*
	// label (0 = unknown/lowest, higher = do-first; the candidate builder maps
	// priority/P0>P1>P2 to descending integers). It LEADS the dispatch order: DispKeep units
	// sort by Priority descending first, then fall back to the existing recency/PreferOldest/ID
	// tiebreak. Purely additive — an all-zero-priority candidate set orders exactly as it did
	// before this field existed (the no-regression guarantee), and Priority changes only the
	// ORDER among survivors: supersede-collapse, live, cooldown, generation-hold, and collision
	// pricing are untouched.
	Priority int `json:"priority,omitempty"`
	// ReadOnly declares that this unit writes NOTHING — a provably empty WRITE footprint, which
	// is the third state the file-tree axis distinguishes from a declared tree (a known blast
	// radius) and an ABSENT tree (an UNKNOWN blast radius that collides conservatively). A
	// read-only candidate never collides on the file plane: it cannot clobber a concurrent
	// writer and cannot be clobbered by one, so it is admitted alongside anything, even an empty
	// tree or the same lane. This is the opt-out for a pure read (a `git log`, a `grep`) that the
	// bare empty-Tree rule would otherwise fold into unknown-blast-radius and serialize. Purely
	// additive — a false ReadOnly (every existing caller) prices byte-identically to before, and
	// the compute axis is untouched (a read-only file footprint asserts nothing about compute).
	ReadOnly bool `json:"read_only,omitempty"`
	// ObjectiveSignal is the OPTIONAL trajctl curve signal for this unit's bound objective,
	// carried as plain data (a string token, never the internal/trajctl type — the pureRoot
	// no-internal-imports contract): "HEALTHY", "STALL", "DRIFT", "DETOUR_OVERRUN". Only a
	// sustained-STALL objective (ObjectiveSignalStall, matched case-insensitively) changes the
	// order: within its priority tier it sorts AFTER every non-wedged candidate — deprioritized,
	// never refused — and carries ReasonWedgedObjective when a fresh alternative outranks it. It
	// stays DispKeep and Keep-eligible, so a lone wedged candidate is still picked. Empty (every
	// existing caller) orders byte-identically to before this field existed — the additive
	// no-regression guarantee.
	ObjectiveSignal string `json:"objective_signal,omitempty"`
}

// ComputeClaim is a candidate's claim over a compute region — the finer-grained twin of the
// file-tree claim (see #3268, the disaggregation axis of the claim-space). It carries exactly
// three fields, mirroring the file claim's Lane/Tree/Mode:
//
//   - Class is the resource class being claimed: "device", "phase-pool", "expert-set", "kv-tier",
//     or any caller-defined label. It is the compute-plane analogue of a dos.toml Lane — two
//     claims in DIFFERENT classes address disjoint resource spaces and NEVER contend.
//   - Range is the address range within that class, a compact integer form. Accepted shapes,
//     comma-joined for a union: a single rank/device "0"; an inclusive interval "4-7" or "4..7";
//     a start+len span "4+4"; the KV half-open "[start,len)". An empty or unparseable Range is an
//     unknown blast radius that conservatively collides with any same-class claim — the compute
//     twin of an empty file tree.
//   - Mode reuses the file claim's lock discipline verbatim: "" or "exclusive" runs alone on its
//     region; "shared" may overlap another shared claim on the same region.
type ComputeClaim struct {
	Class string `json:"class"`
	Range string `json:"range,omitempty"`
	Mode  string `json:"mode,omitempty"`
}

// recency is the unit's freshness: its last update, falling back to its creation time.
func (c Candidate) recency() int64 {
	if c.UpdatedUnix > 0 {
		return c.UpdatedUnix
	}
	return c.CreatedUnix
}

// ageStamp is the unit's age key for oldest-first ordering: its creation time, falling
// back to recency when creation is unknown (0). Smaller == older == picked first under
// Input.PreferOldest.
func (c Candidate) ageStamp() int64 {
	if c.CreatedUnix > 0 {
		return c.CreatedUnix
	}
	return c.recency()
}

// wedgedObjective reports whether the candidate's bound objective is sustained-STALL — the
// wedged case the dispatch order deprioritizes. Case-insensitive and whitespace-tolerant over
// the plain-data token; every other signal (including empty) is not wedged.
func wedgedObjective(c Candidate) bool {
	return strings.EqualFold(strings.TrimSpace(c.ObjectiveSignal), ObjectiveSignalStall)
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
	// CollidesWith names the unit(s) that caused a collision-risk serialization. Empty otherwise.
	CollidesWith []string `json:"collides_with,omitempty"`
	// BlockedByOpen names the still-OPEN prerequisite(s) that caused a DispBlocked soft hold —
	// the subset of BlockedBy present in this tick's candidate set (sorted). Empty otherwise.
	BlockedByOpen []string `json:"blocked_by_open,omitempty"`
	// CoolingSince / CoolingUntil declare the cooldown window behind a DispCooling verdict
	// (unix seconds): the span from the unit's last attempt to the moment it re-enters the
	// pool. The keys deliberately mirror internal/dispatchaging.Candidate's cooling_since /
	// cooling_until, so an aging caller building Candidates from these dispositions (or
	// re-feeding a Result's order JSON) inherits the window verbatim and PAUSES the unit's
	// starvation clock over the ineligible span instead of dropping it (#3715). Zero on every
	// other disposition — the omitempty keeps non-cooling rows byte-identical.
	CoolingSince int64 `json:"cooling_since,omitempty"`
	CoolingUntil int64 `json:"cooling_until,omitempty"`
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
	// PreferOldest orders the distinct kept units OLDEST-first (by creation, then recency,
	// then ID) instead of the default freshest-first, so a worker drains the longest-waiting
	// backlog item first. It changes only the dispatch ORDER among survivors: the same-key
	// supersede collapse still keeps the FRESHEST duplicate (the most recent update of one
	// target), and the live/cooldown/collision skips are unchanged.
	PreferOldest bool `json:"prefer_oldest,omitempty"`
	// FinishFirst orders already-attempted survivors before never-attempted survivors within
	// the same explicit priority tier. It does not bypass cooldown, dependencies, generations,
	// or collision pricing. False preserves the legacy recency order.
	FinishFirst bool `json:"finish_first,omitempty"`
	// Generation narrows the admitted horizon: "", "default", or "auto" admits gen/now and
	// gen/next; "now", "next", "second-next", and "future" admit only that horizon; "all"
	// admits every classified generation while still holding unclassified candidates. Legacy
	// callers with no candidate generation keep the pre-generation behavior.
	Generation string `json:"generation,omitempty"`
}

// Result is the full deterministic verdict: every candidate's disposition plus the freshest-
// first pick list.
type Result struct {
	// Order is every candidate, DispKeep units first in dispatch order, then the rest by recency.
	Order []Ranked `json:"order"`
	// Keep is the IDs a worker should pick, freshest-first — Order's DispKeep units, in rank order.
	Keep []string `json:"keep"`
	// Counts of each disposition, so a one-line summary needs no fold.
	KeepCount           int `json:"keep_count"`
	SupersededCount     int `json:"superseded_count"`
	LiveCount           int `json:"live_count"`
	CoolingCount        int `json:"cooling_count"`
	CollisionCount      int `json:"collision_count"`
	GenerationHeldCount int `json:"generation_held_count"`
	BlockedCount        int `json:"blocked_count"`
	// Collisions is the priced collision graph over otherwise dispatchable fan-out candidates.
	Collisions []Collision `json:"collisions,omitempty"`
	// Repartition names the colliding candidates that need narrower scope before a later wave
	// can admit them. It is geometric advice: declare/narrow the tree and then re-price.
	Repartition []RepartitionAdvice `json:"repartition,omitempty"`
	// S0-facing accounting: collisions_avoided is the number of colliding pairs serialized before
	// launch; lanes_utilized is the count of safe lanes/workers admitted this wave; and
	// serialization_wasted is the count of otherwise-dispatchable workers held for a later wave.
	CollisionsAvoided   int `json:"collisions_avoided"`
	LanesUtilized       int `json:"lanes_utilized"`
	SerializationWasted int `json:"serialization_wasted"`
	SafeConcurrency     int `json:"safe_concurrency"`
}

// Collision is one edge in the pre-launch collision graph. It is pure geometry: no worker has
// launched yet, and no live lease was acquired by this planner.
type Collision struct {
	A      string   `json:"a"`
	B      string   `json:"b"`
	Reason string   `json:"reason"`
	Lane   []string `json:"lane,omitempty"`
	Tree   []string `json:"tree,omitempty"`
	// Region is the pair of overlapping compute regions ("class:range") when the edge is a
	// compute-plane collision rather than a file-tree one. Omitted for a file-only edge, so a
	// file-only collision graph serializes byte-identically to before the compute axis existed.
	Region []string `json:"region,omitempty"`
}

// RepartitionAdvice is the price's operator-facing "how to clear this collision next" row.
// It never guesses semantic intent; it tells the caller which geometric scope is too broad
// or absent, which peers it collides with, and what evidence must be supplied before re-price.
type RepartitionAdvice struct {
	Candidate    string   `json:"candidate"`
	Lane         string   `json:"lane,omitempty"`
	CollidesWith []string `json:"collides_with,omitempty"`
	CurrentTree  []string `json:"current_tree,omitempty"`
	OverlapTree  []string `json:"overlap_tree,omitempty"`
	// CurrentRegion/OverlapRegion carry the compute-plane scope ("class:range") when the advice is
	// for a compute-region collision rather than a file-tree one. Omitted for file-only advice, so
	// a file-only repartition row is byte-identical to before the compute axis existed.
	CurrentRegion []string `json:"current_region,omitempty"`
	OverlapRegion []string `json:"overlap_region,omitempty"`
	Action        string   `json:"action"`
	Reason        string   `json:"reason"`
	Detail        string   `json:"detail"`
}

// Pick is the single unit a worker should take this tick — Keep[0], or "" when nothing is
// dispatchable (every candidate is superseded, live, or cooling).
func (r Result) Pick() string {
	if len(r.Keep) == 0 {
		return ""
	}
	return r.Keep[0]
}

// Invariant: dispatch ordering is fail-closed and deterministic across all candidate inputs.
// Given identical candidate sets and clock inputs, Plan guarantees reproducible ordering,
// deterministic tiebreaking, and fail-closed isolation of unknown blast radiuses.
//
// Guard: candidates with unresolvable tree or compute collisions are serialized before launch,
// and unknown blast radiuses fail closed by colliding conservatively against concurrent participants.
//
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
//  4. DispKeep units are ordered by declared Priority (descending) first, then non-wedged before
//     sustained-STALL (ObjectiveSignal "STALL" sorts after fresh work WITHIN its priority tier —
//     deprioritized, never refused), then freshest-first (oldest-first when Input.PreferOldest),
//     and assigned a rank; Keep lists their IDs. A wedged kept unit outranked by a fresh
//     alternative carries ReasonWedgedObjective; a lone wedged unit is still picked (ReasonFreshest).
//
// A group whose winner is live or cooling yields NO keep this tick (the dispatcher waits for the
// freshest rather than running a stale duplicate) — the deliberate v1 posture; a max-backoff
// fallback to the next-freshest is a separate, later rung.
func Plan(in Input) Result {
	cooldown := in.CooldownSeconds
	if cooldown == 0 {
		cooldown = DefaultCooldownSeconds
	}

	generationActive := generationGateActive(in)
	generationHeld := make(map[string]bool)
	eligible := make([]Candidate, 0, len(in.Candidates))
	for _, c := range in.Candidates {
		if generationActive && !c.Live && !generationAllowed(c.Generation, in.Generation) {
			generationHeld[c.ID] = true
			continue
		}
		eligible = append(eligible, c)
	}
	winner := winnersByKey(eligible)
	blocked := blockedByOpenPrereq(in.Candidates)

	ranked := make([]Ranked, 0, len(in.Candidates))
	for _, c := range in.Candidates {
		r := Ranked{Candidate: c, Recency: c.recency(), Rank: -1}
		switch {
		case c.Live:
			r.Disposition, r.Reason = DispLive, ReasonWorkerLive
		case generationHeld[c.ID]:
			r.Disposition, r.Reason = DispGenerationHeld, ReasonGenerationHeld
		case c.Key != "" && winner[c.Key] != c.ID:
			r.Disposition, r.Reason, r.SupersededBy = DispSuperseded, ReasonSuperseded, winner[c.Key]
		case len(blocked[c.ID]) > 0:
			r.Disposition, r.Reason, r.BlockedByOpen = DispBlocked, ReasonBlockedByOpenPrereq, blocked[c.ID]
		case cooldown > 0 && c.LastAttemptUnix > 0 && in.NowUnix-c.LastAttemptUnix < cooldown:
			r.Disposition, r.Reason = DispCooling, ReasonCooldown
			r.CoolingSince, r.CoolingUntil = c.LastAttemptUnix, c.LastAttemptUnix+cooldown
		default:
			r.Disposition, r.Reason = DispKeep, ReasonFreshest
		}
		ranked = append(ranked, r)
	}

	collisions := priceFanout(ranked)
	if len(collisions) > 0 {
		applyCollisionPrice(ranked, collisions, in.PreferOldest)
	}

	// Order: DispKeep first by recency (freshest-first), then the rest by recency, stable and
	// deterministic. Ranks and the Keep list are assigned from the kept prefix.
	sort.SliceStable(ranked, func(i, j int) bool {
		ki, kj := ranked[i].Disposition == DispKeep, ranked[j].Disposition == DispKeep
		if ki != kj {
			return ki // kept units sort ahead of skipped ones
		}
		if ranked[i].Priority != ranked[j].Priority {
			return ranked[i].Priority > ranked[j].Priority // declared priority leads recency; higher = do-first
		}
		// A prior attempt is the durable signal that this issue is already WIP. Once its
		// cooldown clears, finish it before opening a never-attempted issue in the same
		// priority tier. This is deliberately a sort, not a hold: collision and dependency
		// pricing can still choose the largest safe set.
		if in.FinishFirst {
			if ai, aj := ranked[i].LastAttemptUnix > 0, ranked[j].LastAttemptUnix > 0; ai != aj {
				return ai
			}
		}
		if wi, wj := wedgedObjective(ranked[i].Candidate), wedgedObjective(ranked[j].Candidate); wi != wj {
			return !wi // within a priority tier, fresh work outranks a sustained-STALL (wedged) objective
		}
		if in.PreferOldest {
			return olderFirst(ranked[i], ranked[j]) // drain the longest-waiting backlog first
		}
		return moreRecent(ranked[i], ranked[j])
	})

	out := Result{Order: ranked}
	freshKeptAhead := false // has a non-wedged kept unit already been ranked?
	for i := range out.Order {
		switch out.Order[i].Disposition {
		case DispKeep:
			if wedgedObjective(out.Order[i].Candidate) {
				if freshKeptAhead {
					// Deprioritized behind a fresh alternative: ledger WHY with the closed token.
					// Still kept, still ranked — a demotion, never a refusal.
					out.Order[i].Reason = ReasonWedgedObjective
				}
			} else {
				freshKeptAhead = true
			}
			out.Order[i].Rank = len(out.Keep)
			out.Keep = append(out.Keep, out.Order[i].ID)
			out.KeepCount++
		case DispSuperseded:
			out.SupersededCount++
		case DispLive:
			out.LiveCount++
		case DispCooling:
			out.CoolingCount++
		case DispCollisionRisk:
			out.CollisionCount++
		case DispGenerationHeld:
			out.GenerationHeldCount++
		case DispBlocked:
			out.BlockedCount++
		}
	}
	out.Collisions = collisions
	out.Repartition = repartitionAdvice(out.Order, collisions)
	out.CollisionsAvoided = len(collisions)
	out.LanesUtilized = lanesUtilized(out.Order)
	out.SerializationWasted = out.CollisionCount
	out.SafeConcurrency = out.KeepCount
	return out
}

func generationGateActive(in Input) bool {
	if strings.TrimSpace(in.Generation) != "" {
		return true
	}
	for _, c := range in.Candidates {
		if strings.TrimSpace(c.Generation) != "" {
			return true
		}
	}
	return false
}

func generationAllowed(label, window string) bool {
	label = normalizeGeneration(label)
	window = normalizeGenerationWindow(window)
	if label == "" {
		return false
	}
	switch window {
	case "", "default", "auto":
		return label == "gen/now" || label == "gen/next"
	case "all":
		return isKnownGeneration(label)
	default:
		return label == window
	}
}

func normalizeGeneration(label string) string {
	s := strings.ToLower(strings.TrimSpace(label))
	switch s {
	case "now":
		return "gen/now"
	case "next":
		return "gen/next"
	case "second-next", "second_next", "secondnext":
		return "gen/second-next"
	case "future":
		return "gen/future"
	default:
		return s
	}
}

func normalizeGenerationWindow(window string) string {
	s := strings.ToLower(strings.TrimSpace(window))
	switch s {
	case "", "default", "auto", "all":
		return s
	default:
		return normalizeGeneration(s)
	}
}

func isKnownGeneration(label string) bool {
	switch label {
	case "gen/now", "gen/next", "gen/second-next", "gen/future":
		return true
	default:
		return false
	}
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

// BlockedByOpenPrereq is the exported view of blockedByOpenPrereq: given a candidate set carrying
// BlockedBy edges, it returns per-ID the sorted set of prerequisites still open this tick (a SOFT
// hold). It exists so the live tick's post-route hold (cmd/fak/dispatch_prereq.go) reuses the same
// tested fail-open / cycle-safe engine the ordering leaf uses, rather than re-deriving it.
func BlockedByOpenPrereq(cands []Candidate) map[string][]string { return blockedByOpenPrereq(cands) }

// blockedByOpenPrereq computes, per candidate ID, the sorted set of prerequisites it names in
// BlockedBy that are still OPEN candidates this tick — a SOFT hold, never a hard ban. A candidate
// with a non-empty result is held (DispBlocked) rather than dispatched. Two invariants keep the
// hold from ever deadlocking or freezing exploratory work:
//
//   - Fail-open: a prerequisite id ABSENT from this tick's candidate set is already closed (no
//     longer an open unit) and never holds — so a dependency clears itself the moment its
//     prerequisite leaves the set.
//   - Cycle-safe: a mutual A<->B block (each names the other) breaks toward the LOWEST ID — the
//     lower keeps the right to dispatch, only the higher is held — so a dependency cycle resolves
//     deterministically to one dispatchable unit instead of hanging. A self-edge never blocks.
func blockedByOpenPrereq(cands []Candidate) map[string][]string {
	present := make(map[string]bool, len(cands))
	names := make(map[string]map[string]bool, len(cands))
	for _, c := range cands {
		present[c.ID] = true
	}
	for _, c := range cands {
		for _, p := range c.BlockedBy {
			if names[c.ID] == nil {
				names[c.ID] = make(map[string]bool, len(c.BlockedBy))
			}
			names[c.ID][p] = true
		}
	}
	blocked := make(map[string][]string)
	for _, c := range cands {
		var open []string
		for _, p := range c.BlockedBy {
			if p == c.ID || !present[p] {
				continue // a self-edge, or an absent (already-closed) prerequisite: fail-open, no hold
			}
			if names[p][c.ID] && c.ID < p {
				continue // mutual A<->B cycle: the lowest ID keeps the dispatch, only the higher is held
			}
			open = appendUniqueString(open, p)
		}
		if len(open) > 0 {
			sort.Strings(open)
			blocked[c.ID] = open
		}
	}
	return blocked
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

// olderFirst is the oldest-first total order over Ranked: smaller age (creation, then
// recency), then smaller ID — deterministic with no ties. The inverse of moreRecent, used
// for the final ordering when Input.PreferOldest drains the longest-waiting backlog first.
func olderFirst(a, b Ranked) bool {
	if as, bs := a.ageStamp(), b.ageStamp(); as != bs {
		return as < bs
	}
	if a.recency() != b.recency() {
		return a.recency() < b.recency()
	}
	return a.ID < b.ID
}

// priceFanout builds the collision graph over the otherwise-kept candidates. Legacy order-only
// callers that provide no lane/tree/mode facts keep the pre-existing behavior; as soon as any
// candidate carries those facts, the whole candidate set is the proposed multi-agent fan-out and
// unknown trees collide conservatively.
func priceFanout(ranked []Ranked) []Collision {
	priced := false
	var keep []Ranked
	for _, r := range ranked {
		if r.Disposition == DispKeep {
			keep = append(keep, r)
		}
		if candidatePriced(r.Candidate) {
			priced = true
		}
	}
	if !priced || len(keep) < 2 {
		return nil
	}

	var collisions []Collision
	for i := 0; i < len(keep); i++ {
		for j := i + 1; j < len(keep); j++ {
			if c, ok := collisionOf(keep[i].Candidate, keep[j].Candidate); ok {
				collisions = append(collisions, c)
			}
		}
	}
	return collisions
}

// applyCollisionPrice serializes the colliding losers: the kept units are sorted into the same
// fairness order the final dispatch sort uses (freshest-first, or oldest-first under
// Input.PreferOldest), and maxSafeSet admits from that order — so the safe set agrees with the
// caller's declared fairness policy instead of always favoring the fresher unit. Without this,
// PreferOldest ranks the oldest unit first only for maxSafeSet to serialize it behind a fresher
// collider, starving the backlog item the flag exists to drain.
func applyCollisionPrice(ranked []Ranked, collisions []Collision, preferOldest bool) {
	collides := make(map[string][]string)
	for _, c := range collisions {
		collides[c.A] = append(collides[c.A], c.B)
		collides[c.B] = append(collides[c.B], c.A)
	}

	var keep []Ranked
	for _, r := range ranked {
		if r.Disposition == DispKeep {
			keep = append(keep, r)
		}
	}
	admitFirst := moreRecent
	if preferOldest {
		admitFirst = olderFirst
	}
	sort.SliceStable(keep, func(i, j int) bool { return admitFirst(keep[i], keep[j]) })
	safe := maxSafeSet(keep, collisions)
	for i := range ranked {
		if ranked[i].Disposition != DispKeep {
			continue
		}
		if safe[ranked[i].ID] {
			continue
		}
		ranked[i].Disposition = DispCollisionRisk
		ranked[i].Reason = ReasonCollisionRisk
		ranked[i].CollidesWith = append([]string(nil), collides[ranked[i].ID]...)
		sort.Strings(ranked[i].CollidesWith)
	}
}

func repartitionAdvice(order []Ranked, collisions []Collision) []RepartitionAdvice {
	if len(collisions) == 0 {
		return nil
	}
	peers := map[string][]string{}
	trees := map[string][]string{}
	regions := map[string][]string{}
	candidateTrees := map[string][]string{}
	for _, r := range order {
		candidateTrees[r.ID] = cleanTree(r.Tree)
	}
	for _, c := range collisions {
		peers[c.A] = appendUniqueString(peers[c.A], c.B)
		peers[c.B] = appendUniqueString(peers[c.B], c.A)
		for _, t := range c.Tree {
			trees[c.A] = appendUniqueString(trees[c.A], t)
			trees[c.B] = appendUniqueString(trees[c.B], t)
		}
		for _, rg := range c.Region {
			regions[c.A] = appendUniqueString(regions[c.A], rg)
			regions[c.B] = appendUniqueString(regions[c.B], rg)
		}
	}
	var out []RepartitionAdvice
	for _, r := range order {
		if r.Disposition != DispCollisionRisk {
			continue
		}
		// A candidate whose collision was priced on the compute plane gets compute-region advice;
		// a file-tree collision keeps the original tree advice. A candidate that somehow collided on
		// both axes is advised on the compute axis first (its file advice re-surfaces once the
		// compute region is disjoint and the pair is re-priced).
		if len(regions[r.ID]) > 0 {
			out = append(out, computeRepartition(r, peers[r.ID], regions[r.ID]))
			continue
		}
		cur := cleanTree(r.Tree)
		action := "narrow_to_issue_paths"
		detail := "replace the broad lane tree with path-confirmed issue paths, then re-price"
		if len(cur) == 0 {
			action = "declare_tree_scope"
			detail = "declare this worker's repo-relative tree before launch; unknown scope collides conservatively"
		} else if anyPeerUnknown(peers[r.ID], candidateTrees) {
			action = "peer_declare_tree_scope"
			detail = "a colliding peer has unknown scope; declare that peer's tree and re-price before running together"
		}
		adv := RepartitionAdvice{
			Candidate:    r.ID,
			Lane:         strings.TrimSpace(r.Lane),
			CollidesWith: sortedStrings(peers[r.ID]),
			CurrentTree:  cur,
			OverlapTree:  sortedStrings(trees[r.ID]),
			Action:       action,
			Reason:       ReasonCollisionRisk,
			Detail:       detail,
		}
		out = append(out, adv)
	}
	return out
}

func anyPeerUnknown(peers []string, trees map[string][]string) bool {
	for _, peer := range peers {
		if len(trees[peer]) == 0 {
			return true
		}
	}
	return false
}

// computeRepartition builds the compute-plane repartition row: narrow the claimed rank/device
// range to a disjoint sub-region of its class, or declare a region at all when the claim left it
// unknown. It is the compute twin of the file-tree advice above.
func computeRepartition(r Ranked, peers, overlap []string) RepartitionAdvice {
	var cur []string
	action := "narrow_compute_range"
	detail := "narrow the claimed compute range to a disjoint sub-region of its class, then re-price"
	if r.Compute != nil && strings.TrimSpace(r.Compute.Range) != "" {
		cur = []string{regionLabel(r.Compute)}
	} else {
		action = "declare_compute_range"
		detail = "declare this worker's compute region (class:range) before launch; an unknown range collides conservatively"
	}
	return RepartitionAdvice{
		Candidate:     r.ID,
		Lane:          strings.TrimSpace(r.Lane),
		CollidesWith:  sortedStrings(peers),
		CurrentRegion: cur,
		OverlapRegion: sortedStrings(overlap),
		Action:        action,
		Reason:        ReasonCollisionRisk,
		Detail:        detail,
	}
}

// maxSafeSet returns the largest collision-free subset for normal fan-out widths. When several
// subsets have the same size, ties break toward candidates EARLIER in cands — the caller's
// admission order (freshest-first by default, oldest-first under PreferOldest), because the DFS
// explores include-before-exclude in position order and only a strictly larger subset replaces
// the best. Very large candidate lists fall back to the same deterministic position-order
// admission rule so a planning helper never turns one issue-lane backlog into an exponential
// search.
func maxSafeSet(cands []Ranked, collisions []Collision) map[string]bool {
	n := len(cands)
	graph := make(map[string]map[string]bool, len(cands))
	for _, c := range collisions {
		if graph[c.A] == nil {
			graph[c.A] = map[string]bool{}
		}
		if graph[c.B] == nil {
			graph[c.B] = map[string]bool{}
		}
		graph[c.A][c.B] = true
		graph[c.B][c.A] = true
	}
	if n > exactSafeSetLimit {
		return greedySafeSet(cands, graph)
	}

	var best []string
	var dfs func(pos int, chosen []string)
	dfs = func(pos int, chosen []string) {
		if len(chosen)+n-pos < len(best) {
			return
		}
		if pos == n {
			if len(chosen) > len(best) {
				best = append([]string(nil), chosen...)
			}
			return
		}
		id := cands[pos].ID
		ok := true
		for _, prev := range chosen {
			if graph[id][prev] {
				ok = false
				break
			}
		}
		if ok {
			dfs(pos+1, append(chosen, id))
		}
		dfs(pos+1, chosen)
	}
	dfs(0, nil)

	out := make(map[string]bool, len(best))
	for _, id := range best {
		out[id] = true
	}
	return out
}

func greedySafeSet(cands []Ranked, graph map[string]map[string]bool) map[string]bool {
	out := map[string]bool{}
	for _, cand := range cands {
		ok := true
		for id := range out {
			if graph[cand.ID][id] {
				ok = false
				break
			}
		}
		if ok {
			out[cand.ID] = true
		}
	}
	return out
}

func candidatePriced(c Candidate) bool {
	return strings.TrimSpace(c.Lane) != "" || len(cleanTree(c.Tree)) > 0 || strings.TrimSpace(c.Mode) != "" || c.Compute != nil
}

// hasFileClaim reports whether a candidate addresses the file plane (a lane or a non-empty tree).
// A bare Mode is a modifier, not a claim, so it does not by itself make a file claim.
func hasFileClaim(c Candidate) bool {
	return strings.TrimSpace(c.Lane) != "" || len(cleanTree(c.Tree)) > 0
}

// fileParticipant reports whether a candidate is priced on the file axis. Every legacy candidate
// (no compute claim) participates — preserving the pre-#3268 behavior where a bare exclusive
// candidate collides conservatively — while a compute-ONLY candidate stays off the file axis, so
// it never collides with a file worker via the empty-tree conservative rule.
func fileParticipant(c Candidate) bool {
	if c.Compute == nil {
		return true
	}
	return hasFileClaim(c)
}

// TreesOverlap reports whether any tree in a and b overlaps under the same prefix geometry the
// fan-out price uses. An empty side is unknown blast radius and collides conservatively.
func TreesOverlap(a, b []string) bool {
	ta, tb := cleanTree(a), cleanTree(b)
	if len(ta) == 0 || len(tb) == 0 {
		return true
	}
	for _, x := range ta {
		for _, y := range tb {
			if treeOverlap(x, y) {
				return true
			}
		}
	}
	return false
}

// collisionOf prices one pre-launch collision edge across BOTH claim axes: the file tree and the
// compute region. A pair collides if it contends on EITHER — the natural "resource request"
// semantics, where two workers serialize if they share any resource. The two axes are independent:
// a file claim and a compute claim address disjoint resource spaces and never contend with each
// other. Each axis keeps its own lock mode, so a file-shared/file-shared pair can still collide on
// an exclusive compute region and vice versa.
func collisionOf(a, b Candidate) (Collision, bool) {
	if fileParticipant(a) && fileParticipant(b) {
		if c, ok := fileCollision(a, b); ok {
			return c, true
		}
	}
	if a.Compute != nil && b.Compute != nil {
		if c, ok := computeCollision(a, b); ok {
			return c, true
		}
	}
	return Collision{}, false
}

// fileCollision is the file-tree axis: the original lane/tree/mode geometry, unchanged
// except for the read-only opt-out. A participant that declares ReadOnly writes NOTHING, so
// it cannot contend on the write plane with anyone — not even an empty-tree (unknown blast
// radius) or same-lane peer the geometry would otherwise serialize. This is the branch that
// distinguishes a provably empty WRITE footprint from an ABSENT tree: gated here, ABOVE the
// empty-tree conservative-overlap rule in TreesOverlap, so that pure geometry (TreesOverlap)
// keeps its "empty collides" contract for every unknown-footprint caller.
func fileCollision(a, b Candidate) (Collision, bool) {
	if a.ReadOnly || b.ReadOnly {
		return Collision{}, false
	}
	ma, mb := lockMode(a), lockMode(b)
	if ma == "shared" && mb == "shared" {
		return Collision{}, false
	}
	c := Collision{A: a.ID, B: b.ID, Reason: ReasonCollisionRisk}
	if laneA, laneB := strings.TrimSpace(a.Lane), strings.TrimSpace(b.Lane); laneA != "" || laneB != "" {
		c.Lane = []string{laneA, laneB}
		if laneA != "" && laneA == laneB {
			return c, true
		}
	}
	if TreesOverlap(a.Tree, b.Tree) {
		for _, x := range cleanTree(a.Tree) {
			for _, y := range cleanTree(b.Tree) {
				if treeOverlap(x, y) {
					c.Tree = []string{x, y}
					return c, true
				}
			}
		}
		return c, true
	}
	return Collision{}, false
}

// computeCollision is the compute-region axis, the exact structural twin of fileCollision: a
// shared/shared pair may overlap; different resource CLASSES never contend (the compute-plane
// analogue of "different lanes decide on tree geometry"); and within one class the integer RANGES
// decide, with an empty/unparseable range colliding conservatively as unknown blast radius.
func computeCollision(a, b Candidate) (Collision, bool) {
	ca, cb := a.Compute, b.Compute
	if ca == nil || cb == nil {
		return Collision{}, false
	}
	if !ComputeClaimsContend(*ca, *cb) {
		return Collision{}, false
	}
	return Collision{
		A:      a.ID,
		B:      b.ID,
		Reason: ReasonCollisionRisk,
		Region: []string{regionLabel(ca), regionLabel(cb)},
	}, true
}

// computeMode reuses the file lock discipline verbatim: empty defaults to exclusive.
func computeMode(cc *ComputeClaim) string {
	switch strings.ToLower(strings.TrimSpace(cc.Mode)) {
	case "shared":
		return "shared"
	default:
		return "exclusive"
	}
}

// normClass folds a resource class to its comparison key (trimmed, lowercased).
func normClass(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// regionLabel renders a compute claim as "class:range" for the collision graph and advice, with a
// "*" range standing in for an unknown (empty) range.
func regionLabel(cc *ComputeClaim) string {
	class := strings.TrimSpace(cc.Class)
	rng := strings.TrimSpace(cc.Range)
	if rng == "" {
		rng = "*"
	}
	if class == "" {
		return rng
	}
	return class + ":" + rng
}

// interval is one inclusive integer range [lo,hi] within a compute class's address space.
type interval struct{ lo, hi int64 }

// rangesOverlap reports whether two compute Range strings address any common integer. An empty or
// unparseable range on either side is unknown blast radius and collides conservatively — the exact
// discipline TreesOverlap holds for an empty file tree.
func rangesOverlap(a, b string) bool {
	ia, oka := parseRange(a)
	ib, okb := parseRange(b)
	if !oka || !okb {
		return true
	}
	for _, x := range ia {
		for _, y := range ib {
			if x.lo <= y.hi && y.lo <= x.hi {
				return true
			}
		}
	}
	return false
}

// parseRange parses a compute Range into a union of inclusive integer intervals. It accepts a
// bracketed KV half-open span "[start,len)" and a comma-joined union of tokens, each a single int
// "n", an inclusive interval "a-b"/"a..b", or a start+len span "a+len". It returns known=false for
// an empty or unparseable string so the caller can collide conservatively; it never panics.
func parseRange(s string) ([]interval, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, false
	}
	if strings.HasPrefix(s, "[") || strings.HasPrefix(s, "(") {
		inner := strings.Trim(s, "[](){} \t")
		parts := strings.Split(inner, ",")
		if len(parts) != 2 {
			return nil, false
		}
		start, ok1 := parseInt(parts[0])
		length, ok2 := parseInt(parts[1])
		if !ok1 || !ok2 {
			return nil, false
		}
		return []interval{spanInterval(start, length)}, true
	}
	var out []interval
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		iv, ok := parseRangeToken(tok)
		if !ok {
			return nil, false
		}
		out = append(out, iv)
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// parseRangeToken parses one non-empty Range token into an inclusive interval.
func parseRangeToken(tok string) (interval, bool) {
	if i := strings.Index(tok, ".."); i >= 0 {
		return parseRangePair(tok[:i], tok[i+2:], false)
	}
	if i := strings.Index(tok, "+"); i > 0 {
		return parseRangePair(tok[:i], tok[i+1:], true)
	}
	if i := strings.Index(tok, "-"); i > 0 { // i>0 so a leading '-' stays part of a negative int
		return parseRangePair(tok[:i], tok[i+1:], false)
	}
	n, ok := parseInt(tok)
	if !ok {
		return interval{}, false
	}
	return interval{n, n}, true
}

// parseRangePair builds an interval from two parsed halves: an inclusive [a,b] pair (ordered) or,
// when span is set, a start+length span.
func parseRangePair(a, b string, span bool) (interval, bool) {
	lo, ok1 := parseInt(a)
	x, ok2 := parseInt(b)
	if !ok1 || !ok2 {
		return interval{}, false
	}
	if span {
		return spanInterval(lo, x), true
	}
	if lo > x {
		lo, x = x, lo
	}
	return interval{lo, x}, true
}

// spanInterval turns a start+length span into an inclusive interval; a non-positive length claims
// just the start index.
func spanInterval(start, length int64) interval {
	if length <= 0 {
		return interval{start, start}
	}
	return interval{start, start + length - 1}
}

// parseInt parses a base-10 integer half of a Range token; whitespace-tolerant, no panic.
func parseInt(s string) (int64, bool) {
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

func lockMode(c Candidate) string {
	switch strings.ToLower(strings.TrimSpace(c.Mode)) {
	case "shared":
		return "shared"
	default:
		return "exclusive"
	}
}

func cleanTree(tree []string) []string {
	var out []string
	for _, t := range tree {
		if n := normalizeTree(t); n != "" {
			out = append(out, n)
		}
	}
	return out
}

func treeOverlap(a, b string) bool {
	a, b = normalizeTree(a), normalizeTree(b)
	if a == "" || b == "" {
		return true
	}
	if a == "**/*" || a == "**" || b == "**/*" || b == "**" {
		return true
	}
	return a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")
}

func normalizeTree(t string) string {
	t = strings.TrimSpace(strings.ReplaceAll(t, "\\", "/"))
	t = strings.TrimPrefix(t, "./")
	t = strings.TrimSuffix(t, "/")
	t = strings.TrimSuffix(t, "/**")
	t = strings.TrimSuffix(t, "/*")
	return strings.TrimSuffix(t, "/")
}

func lanesUtilized(order []Ranked) int {
	lanes := map[string]bool{}
	kept := 0
	for _, r := range order {
		if r.Disposition != DispKeep {
			continue
		}
		kept++
		if lane := strings.TrimSpace(r.Lane); lane != "" {
			lanes[lane] = true
		}
	}
	if len(lanes) > 0 {
		return len(lanes)
	}
	return kept
}

func appendUniqueString(xs []string, s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return xs
	}
	for _, x := range xs {
		if x == s {
			return xs
		}
	}
	return append(xs, s)
}

func sortedStrings(xs []string) []string {
	out := append([]string(nil), xs...)
	sort.Strings(out)
	return out
}
