package routing

import (
	"math"
	"sort"
	"strings"
)

// BalancePlanSchema identifies the balance-selection receipt shape.
const BalancePlanSchema = "routing-balance-plan/1"

// MemberScore is one member's measurement: a value carrying no clock, no handle, and
// nothing that could make a decision un-replayable. Freshness is a field: Select never
// reads a clock, so the caller decides whether the measurement is fresh.
type MemberScore struct {
	// Load is the member's saturation, in [0,1]: 0 idle, 1 fully loaded. Clamped on
	// read so an unbounded number cannot pass through the weight multiplication.
	Load float64
	// Latency is the measured latency relative to the group, in [0,1]: 0 fastest.
	// Clamped on read for the same reason.
	Latency float64
	// Measured is the freshness verdict. False means the number above is not believed:
	// least-load and least-latency explore such a member as idle, weighted-hash
	// deprioritises it.
	Measured bool
}

// clampedLoad returns Load clamped to [0,1].
func (m MemberScore) clampedLoad() float64 { return clamp01(m.Load) }

// clampedLatency returns Latency clamped to [0,1].
func (m MemberScore) clampedLatency() float64 { return clamp01(m.Latency) }

// clamp01 bounds a score to [0,1], so a caller that ships 5 or -3 gets a saturation or
// an idle reading rather than a runaway weight. Clamping each component before it is
// weighted is load-bearing.
func clamp01(v float64) float64 {
	switch {
	case math.IsNaN(v):
		return 0
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

// BalanceSnapshot is the measurement input: one score per member, tagged with the
// group it was measured for. It is data, not a live handle: Select receives it by
// value and reads no clock, which is what makes a plan replayable.
type BalanceSnapshot struct {
	// Group is the group these measurements belong to; a snapshot taken for another
	// group is not silently used for this one.
	Group string
	// Members is the per-member measurement. A member absent from the map is
	// unmeasured, which is never an error.
	Members map[string]MemberScore
	// Observed is true when a live signal owner produced this snapshot, so an
	// unmeasured member is a quiet one rather than an unknowable one. A least-*
	// strategy explores an observed group with no measurement instead of failing open.
	Observed bool
}

// scoreOf returns the member's measurement, treating an absent entry as unmeasured
// rather than zero-and-measured. That distinction is the fail-open rule: "idle" and
// "unknown" are not the same claim, so a group with no measurement keeps its declared order.
func (s BalanceSnapshot) scoreOf(member string) MemberScore {
	if s.Members == nil {
		return MemberScore{}
	}
	return s.Members[member]
}

// hasAnyMeasured reports whether at least one member carries a fresh
// measurement.
func (s BalanceSnapshot) hasAnyMeasured(members []string) bool {
	for _, m := range members {
		if s.scoreOf(m).Measured {
			return true
		}
	}
	return false
}

// BalanceExclusion is the closed vocabulary of reasons a member is not at the head of
// a balanced order. Every one is a reorder: the member moves down the chain, is never
// removed, and never becomes a refusal. The set is closed so a caller cannot invent a
// reason meaning "do not serve".
type BalanceExclusion string

const (
	// BalanceExcludeSaturated: measured and fully loaded, so its effective weight is
	// zero. It stays in the chain behind the eligible members — a saturation valve,
	// not an ejection.
	BalanceExcludeSaturated BalanceExclusion = "saturated"
	// BalanceExcludeUnmeasured: no fresh measurement and the strategy (weighted-hash)
	// does not explore. Its peers still balance among themselves.
	BalanceExcludeUnmeasured BalanceExclusion = "unmeasured"
	// BalanceExcludeNoSnapshot: NOTHING in the group has a fresh measurement, so the
	// group keeps its declared order — the fail-open verdict that makes a dead metrics
	// endpoint unable to deny service.
	BalanceExcludeNoSnapshot BalanceExclusion = "no_snapshot"
	// BalanceExcludeNoEligibleMember: the group is non-empty but no member can take
	// traffic (all drained or saturated); the declared order is still returned. A
	// merely missing metric is no_snapshot, not this.
	BalanceExcludeNoEligibleMember BalanceExclusion = "no_eligible_member"
)

// BalanceExclusionReasons returns the closed exclusion vocabulary in
// declaration order.
func BalanceExclusionReasons() []string {
	return []string{
		string(BalanceExcludeSaturated),
		string(BalanceExcludeUnmeasured),
		string(BalanceExcludeNoSnapshot),
		string(BalanceExcludeNoEligibleMember),
	}
}

// BalanceMemberPlan is one member's line in the selection receipt: what was
// declared, what was measured, what it reduced to, and why it is not first.
type BalanceMemberPlan struct {
	// Member is the member identity.
	Member string `json:"member"`
	// Weight is the declared weight.
	Weight int `json:"weight"`
	// Load is the CLAMPED measured load: the plan never carries a raw out-of-range
	// number.
	Load float64 `json:"load"`
	// Latency is the CLAMPED measured latency.
	Latency float64 `json:"latency"`
	// Measured is the freshness verdict for this member.
	Measured bool `json:"measured"`
	// EffectiveWeight is weight * (1 - clamp01(load)), or the declared weight
	// for weighted-share. Zero cannot win the rendezvous race.
	EffectiveWeight float64 `json:"effective_weight"`
	// Rank is the member's rendezvous rank (lower wins). Only meaningful for
	// the weighted-hash strategy and for tie-breaks.
	Rank float64 `json:"rank"`
	// Chosen is true for the member the order puts first.
	Chosen bool `json:"chosen"`
	// Excluded is the closed reason this member is not first, or "" when it is
	// eligible.
	Excluded BalanceExclusion `json:"excluded,omitempty"`
}

// BalancePlan is the receipt: why this member serves this request — the balancer's
// answer to the question DecisionReason answers for a fallback hop. The raw key is
// deliberately absent, since it may be a session identifier and receipts get pasted into
// issues; KeyKind records what kind of identity was hashed instead.
type BalancePlan struct {
	// Schema is the receipt shape tag.
	Schema string `json:"schema"`
	// Group is the group that was selected within.
	Group string `json:"group"`
	// Strategy is the resolved strategy.
	Strategy BalanceStrategy `json:"strategy"`
	// KeyKind names the KIND of identity hashed (e.g. "session", "request",
	// "model"). The value itself is never recorded.
	KeyKind string `json:"key_kind"`
	// Chosen is the member the order puts first.
	Chosen string `json:"chosen"`
	// Members is every declared member, in the returned order, with its line.
	Members []BalanceMemberPlan `json:"members"`
	// Reason is the closed verdict when the group failed open, or "" when the
	// group was balanced normally.
	Reason BalanceExclusion `json:"reason,omitempty"`
	// Audit is the legible trace: one line for the group, one for the verdict,
	// and one per member.
	Audit []string `json:"audit,omitempty"`
}

// BalanceOrder is the selection result: a full order plus its receipt. It is an order, not
// a winner, because failover still needs the rungs behind the head; balance never removes a hop.
type BalanceOrder struct {
	// Members is the ordered member list, winner first.
	Members []string
	// Plan is the receipt explaining Members.
	Plan BalancePlan
}

// LeastLoadTieBand is the load margin within which least-load treats members as
// equally loaded and orders them by weighted rendezvous instead: without it a
// continuous ttfb-derived load never ties and the declared weights never matter. It is
// below one in-flight step, so one extra queued request never ties an idle peer.
const LeastLoadTieBand = 0.05

// BalancePolicy selects a member order. It is a zero-size value on purpose: nothing to
// configure, nothing to remember: no counter, random source, clock, or previous order.
// Rendezvous hashing, unlike a round-robin counter, makes the same (group, snapshot, key)
// yield the same order forever while distinct keys spread across members.
type BalancePolicy struct{}

// Select orders the group's members for one request. It is pure and fails open: a
// missing, unmeasured, or saturated member is reordered, never removed, and no input
// yields an empty order or an error, since a balancer that could refuse service would be
// a quota gate. Least-* strategies explore unmeasured members as idle so a cold member
// earns the measurement it lacks.
func (BalancePolicy) Select(group BalanceGroup, snap BalanceSnapshot, key string) BalanceOrder {
	members := group.Members
	strategy := group.Strategy.orDefault()

	plan := BalancePlan{
		Schema:   BalancePlanSchema,
		Group:    group.Name,
		Strategy: strategy,
		KeyKind:  keyKindOf(key),
		Audit: []string{
			"balance: group=" + group.Name + " strategy=" + string(strategy) +
				" members=" + itoa(len(members)) + " snapshot=" + snapshotVerdict(group, snap),
		},
	}

	// An unvalidated group is a caller bug, and the safe response is declared order
	// rather than a panic: a router that panics on a bad config is a worse outage.
	if err := group.Validate(); err != nil {
		plan.Reason = BalanceExcludeNoEligibleMember
		plan.Chosen = firstOrEmpty(members)
		plan.Members = declaredMemberPlans(group, snap, members)
		plan.Audit = append(plan.Audit, "balance: group="+group.Name+" invalid_declaration=true order=declared")
		plan.Audit = append(plan.Audit, "balance: group="+group.Name+" invalid_declaration_reason="+err.Error())
		return BalanceOrder{Members: append([]string(nil), members...), Plan: plan}
	}

	// Weighted-share needs no measurements. Other strategies fail open unless
	// a live owner reports the whole group quiet under an exploring strategy.
	quiet := !snap.hasAnyMeasured(members)
	if quiet && strategy != BalanceStrategyWeightedShare && !(snap.Observed && strategy.exploresUnmeasured() && snap.Group == group.Name) {
		return failOpen(group, snap, members, strategy, keyKindOf(key), BalanceExcludeNoSnapshot,
			"no member carries a fresh measurement, so the declared order is the order")
	}

	// (2) priority: declared order, scores read and recorded.
	if strategy == BalanceStrategyPriority {
		plan.Members = declaredMemberPlans(group, snap, members)
		plan.Chosen = firstOrEmpty(members)
		for _, m := range plan.Members {
			plan.Audit = append(plan.Audit, fmtScoreLine(group.Name, m, "note=ignored_by=priority"))
		}
		plan.Audit = append(plan.Audit, "balance: group="+group.Name+" order=declared reason=priority_does_not_split")
		return BalanceOrder{Members: append([]string(nil), members...), Plan: plan}
	}

	// Build the eligible set and the per-member receipt lines.
	type scored struct {
		member string
		load   float64
		lat    float64
		rank   float64
		weight float64
		index  int
		line   int
	}
	eligible := make([]scored, 0, len(members))
	lines := make([]BalanceMemberPlan, 0, len(members))
	tail := make([]string, 0, len(members))
	excluded := make([]BalanceMemberPlan, 0, len(members))

	for i, m := range members {
		sc := snap.scoreOf(m)
		w := float64(group.WeightOf(m))
		line := BalanceMemberPlan{
			Member:   m,
			Weight:   group.WeightOf(m),
			Load:     sc.clampedLoad(),
			Latency:  sc.clampedLatency(),
			Measured: sc.Measured,
		}
		// A declared drain and a saturated measurement are the same verdict for the
		// balancer, reported differently so an operator can tell them apart.
		if w <= 0 {
			line.EffectiveWeight = 0
			line.Excluded = BalanceExcludeSaturated
			excluded = append(excluded, line)
			tail = append(tail, m)
			continue
		}
		if !sc.Measured {
			if !strategy.competesUnmeasured() {
				line.EffectiveWeight = 0
				line.Excluded = BalanceExcludeUnmeasured
				excluded = append(excluded, line)
				tail = append(tail, m)
				continue
			}
			line.Load, line.Latency = 0, 0
			line.EffectiveWeight = w
			line.Rank = rendezvousRank(key, group.Name, m, w)
			eligible = append(eligible, scored{member: m, rank: line.Rank, weight: w, index: i, line: len(lines)})
			lines = append(lines, line)
			continue
		}
		wPrime := w
		if strategy != BalanceStrategyWeightedShare {
			wPrime *= 1 - sc.clampedLoad()
		}
		line.EffectiveWeight = wPrime
		if wPrime <= 0 {
			line.Excluded = BalanceExcludeSaturated
			excluded = append(excluded, line)
			tail = append(tail, m)
			continue
		}
		line.Rank = rendezvousRank(key, group.Name, m, wPrime)
		eligible = append(eligible, scored{
			member: m, load: sc.clampedLoad(), lat: sc.clampedLatency(),
			rank: line.Rank, weight: wPrime, index: i, line: len(lines),
		})
		lines = append(lines, line)
	}

	// (5) Nothing can take traffic right now: fail open to declared order.
	if len(eligible) == 0 {
		return failOpen(group, snap, members, strategy, keyKindOf(key), BalanceExcludeNoEligibleMember,
			"every member is drained or saturated, so the declared order is the order")
	}

	minLoad := math.Inf(1)
	for _, e := range eligible {
		minLoad = math.Min(minLoad, e.load)
	}
	headTier := func(e scored) bool { return e.load <= minLoad+LeastLoadTieBand }
	sort.Slice(eligible, func(i, j int) bool {
		switch strategy {
		case BalanceStrategyLeastLoad:
			hi, hj := headTier(eligible[i]), headTier(eligible[j])
			if hi != hj {
				return hi
			}
			if !hi && eligible[i].load != eligible[j].load {
				return eligible[i].load < eligible[j].load
			}
		case BalanceStrategyLeastLatency:
			if eligible[i].lat != eligible[j].lat {
				return eligible[i].lat < eligible[j].lat
			}
		}
		if eligible[i].rank != eligible[j].rank {
			return eligible[i].rank < eligible[j].rank
		}
		return eligible[i].index < eligible[j].index
	})

	ordered := make([]string, 0, len(members))
	out := make([]BalanceMemberPlan, 0, len(members))
	for i, e := range eligible {
		ordered = append(ordered, e.member)
		line := lines[e.line]
		line.Chosen = i == 0
		out = append(out, line)
	}
	// Deprioritised members keep their declared relative order at the tail and stay
	// reachable: balance reorders, it never removes a hop.
	for _, m := range tail {
		ordered = append(ordered, m)
	}
	for _, e := range excluded {
		out = append(out, e)
	}

	plan.Members = out
	plan.Chosen = firstOrEmpty(ordered)
	if strategy == BalanceStrategyWeightedShare {
		plan.Audit = append(plan.Audit, "balance: group="+group.Name+" weights=declared load_and_latency=ignored")
	}
	if quiet {
		plan.Audit = append(plan.Audit, "balance: group="+group.Name+" explored=all_unmeasured order=rendezvous")
	}
	for _, m := range out {
		plan.Audit = append(plan.Audit, fmtScoreLine(group.Name, m, ""))
	}
	plan.Audit = append(plan.Audit, "balance: group="+group.Name+" chosen="+plan.Chosen+
		" deferred="+itoa(len(excluded))+" of "+itoa(len(members)))

	return BalanceOrder{Members: ordered, Plan: plan}
}

// failOpen returns the declared order with a closed verdict. Every fail-open path goes
// through here, so the rule is stated once: a group that cannot balance still serves, in
// declared order, with the reason on the receipt.
func failOpen(group BalanceGroup, snap BalanceSnapshot, members []string, strategy BalanceStrategy, keyKind string, reason BalanceExclusion, why string) BalanceOrder {
	plan := BalancePlan{
		Schema:   BalancePlanSchema,
		Group:    group.Name,
		Strategy: strategy,
		KeyKind:  keyKind,
		Reason:   reason,
		Members:  declaredMemberPlans(group, snap, members),
		Audit: []string{
			"balance: group=" + group.Name + " strategy=" + string(strategy) + " reason=" + string(reason),
			"balance: group=" + group.Name + " fail_open=true " + why,
		},
	}
	plan.Chosen = firstOrEmpty(members)
	return BalanceOrder{Members: append([]string(nil), members...), Plan: plan}
}

// declaredMemberPlans renders every member's receipt line without ranking.
func declaredMemberPlans(group BalanceGroup, snap BalanceSnapshot, members []string) []BalanceMemberPlan {
	out := make([]BalanceMemberPlan, 0, len(members))
	for _, m := range members {
		sc := snap.scoreOf(m)
		w := group.WeightOf(m)
		line := BalanceMemberPlan{
			Member:   m,
			Weight:   w,
			Load:     sc.clampedLoad(),
			Latency:  sc.clampedLatency(),
			Measured: sc.Measured,
		}
		switch {
		case w <= 0:
			line.Excluded = BalanceExcludeSaturated
		case !sc.Measured:
			line.Excluded = BalanceExcludeUnmeasured
		}
		out = append(out, line)
	}
	return out
}

// snapshotVerdict names the freshness of the snapshot as a whole, for the audit line.
// It never reads a clock: freshness is the caller's verdict, already in Measured.
func snapshotVerdict(group BalanceGroup, snap BalanceSnapshot) string {
	if snap.Group != "" && snap.Group != group.Name {
		return "group_mismatch(" + snap.Group + ")"
	}
	if snap.hasAnyMeasured(group.Members) {
		return "measured"
	}
	return "none"
}

// keyKindOf names the KIND of identity that was hashed, never the value: the key may be
// a session identifier, and a receipt is a thing operators paste into issues.
func keyKindOf(key string) string {
	if key == "" {
		return "absent"
	}
	if looksLikeSessionID(key) {
		return "session"
	}
	if len(key) >= 8 && strings.ContainsAny(key, "-_") {
		return "request"
	}
	return "opaque"
}

// looksLikeSessionID is a cheap, deliberately conservative shape test, not an identity
// check: it classifies a string so a receipt can say "this was a session key" without
// recording the session.
func looksLikeSessionID(key string) bool {
	if len(key) < 8 || len(key) > 128 {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		isLower := c >= 'a' && c <= 'z'
		isDigit := c >= '0' && c <= '9'
		if !isLower && !isDigit && c != '-' && c != '_' {
			return false
		}
	}
	// A bare hex run is a request id far more often than a session id, so require a
	// separator before the verdict is "session".
	return strings.ContainsAny(key, "-_")
}

// fmtScoreLine renders one member's audit line: what was declared, what was measured,
// what it reduced to, and why it is not first.
func fmtScoreLine(group string, m BalanceMemberPlan, extra string) string {
	return "balance: group=" + group +
		" member=" + m.Member +
		" weight=" + itoa(m.Weight) +
		" load=" + itoa(int(math.Round(m.Load*1e3))) +
		" latency=" + itoa(int(math.Round(m.Latency*1e3))) +
		" measured=" + boolWord(m.Measured) +
		" effective_weight=" + itoa(int(math.Round(m.EffectiveWeight*1e3))) +
		" rank=" + itoa(int(math.Round(m.Rank*1e3))) +
		" chosen=" + boolWord(m.Chosen) +
		" excluded=" + string(m.Excluded) +
		" " + extra
}

// boolWord renders a bool as a word, so an audit line never reads "measured=truefalse".
func boolWord(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// rendezvousRank is the weighted rendezvous (exponential race) rank of one member:
// rank = -ln(u) / w', with u drawn from hashing (key, group, member) and w' the effective
// weight. The lowest rank wins, giving each member probability w' / sum(w') with no
// counter or history; as w' nears zero the rank diverges upward, so a saturated member
// stops winning before it is formally excluded.
func rendezvousRank(key, group, member string, effectiveWeight float64) float64 {
	h := BalanceScore(key, group, member)
	// Map the 64-bit hash into (0,1): 53 significant bits is what a float64 carries
	// exactly, and the +0.5 keeps u strictly inside (0,1) so ln(u) is never 0 or -Inf.
	u := (float64(h>>11) + 0.5) / float64(uint64(1)<<53)
	if u <= 0 {
		u = math.SmallestNonzeroFloat64
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	if effectiveWeight <= 0 {
		return math.Inf(1)
	}
	return -math.Log(u) / effectiveWeight
}

// BalanceScore is the fabric's hash of a request identity within a group for one
// member: 64-bit FNV-1a over key || ':' || group || ':' || member, with a SplitMix64
// avalanche finalizer. It is deliberately identical to platform/gateway.ComputeHRWScore
// with node identity group + ":" + member, so both axes agree on one identity without
// this package importing the gateway.
func BalanceScore(key, group, member string) uint64 {
	return hrwScore(key, group+":"+member)
}

// hrwScore is the FNV-1a 64 + SplitMix64 construction, kept separate so
// BalanceScore reads as the policy and this reads as the primitive.
func hrwScore(key, nodeID string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	hash := uint64(offset64)
	for i := 0; i < len(key); i++ {
		hash ^= uint64(key[i])
		hash *= prime64
	}
	hash ^= uint64(':')
	hash *= prime64
	for i := 0; i < len(nodeID); i++ {
		hash ^= uint64(nodeID[i])
		hash *= prime64
	}

	// SplitMix64 avalanche: FNV-1a alone leaves visible bit bias on keys that differ
	// only in a short suffix — the exact shape of a session id — so the finalizer is
	// load-bearing.
	hash ^= hash >> 30
	hash *= 0xbf58476d1ce4e5b9
	hash ^= hash >> 27
	hash *= 0x94d049bb133111eb
	hash ^= hash >> 31
	return hash
}

// firstOrEmpty returns members[0] or "".
func firstOrEmpty(members []string) string {
	if len(members) == 0 {
		return ""
	}
	return members[0]
}

// itoa keeps the audit lines allocation-light without pulling strconv in for two call
// sites.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
