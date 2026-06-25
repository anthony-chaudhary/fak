package vcachechain

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// recall.go is the cost-gated rebuild decision — the heart of M4 (issue #719) and
// the design note's headline correction (§11.0). The goal imagines recalling a unit
// by "rebuilding and reordering a series of requests"; the economics say that for a
// single ~10-token unit pulled from a long prefix, this ALMOST ALWAYS LOSES.
//
// Replaying a warm prefix of P tokens at cached-read multiplier r costs P·r
// token-equivalents of READ to avoid U token-equivalents of fresh prefill (the unit
// being recalled). Rebuild is net-negative whenever P·r ≥ U:
//
//	§11.0 example — P=30000, r=0.1, U=10: replay costs 3000 to save 10 → a 300× LOSS.
//
// So the gate REFUSES almost every single-unit chain rebuild — and that is correct.
// Rebuild only wins in the OPPOSITE regime: a large warm prefix AMORTIZED across S
// sibling units recalled TOGETHER while the prefix is still hot, where one replay
// (cost P·r, paid once) saves S fresh prefills (S·U). Rebuild is net-positive iff
//
//	P·r < S·U            (the amortized cost gate)
//
// All functions here are pure and deterministic; the caller injects r (the M1
// calibration read multiplier) and S (the live loop's co-recalled sibling count).

// DefaultEnabled is false: M4 chains & recall is GATED OFF BY DEFAULT (issue #719
// title). With the gate off, PlanRecall returns DecisionGatedOff for every recall
// and the caller sends the unit cold. The live provider recall loop (built on the
// M1–M3 calibration and warm set) is what flips this on; until then the machinery
// is an off-path decision engine, the same posture as the M5 Governor.
const DefaultEnabled = false

// RecallDecision is the verdict PlanRecall returns for one recall.
type RecallDecision string

const (
	// DecisionGatedOff — M4 is disabled (DefaultEnabled). The caller sends the unit
	// cold via a fresh prefill; no chain is replayed. This is the default state.
	DecisionGatedOff RecallDecision = "gated_off"
	// DecisionNoCache — the chain carries secret/regulated content (Law D4): a
	// secret ancestor makes the whole chain non-warmable, so replaying it would push
	// the secret byte-for-byte through the provider prefix cache. Take the no-cache
	// path (full prefix always re-sent, no breakpoint before a secret byte).
	DecisionNoCache RecallDecision = "no_cache"
	// DecisionColdPrefill — the cost gate REFUSED the rebuild (P·r ≥ S·U): replaying
	// the chain costs at least as much as the fresh prefills it would save. Send the
	// unit cold. This is the correct decision for almost every single-unit recall.
	DecisionColdPrefill RecallDecision = "cold_prefill"
	// DecisionRebuild — the cost gate CLEARED (P·r < S·U): amortized fan-out makes
	// one chain replay cheaper than the S fresh prefills it saves. Rebuild, following
	// the TopologicalReplay schedule.
	DecisionRebuild RecallDecision = "rebuild"
)

// IsRebuild reports whether a decision replays the chain (the only verdict that
// touches the provider cache beyond a cold send).
func (d RecallDecision) IsRebuild() bool { return d == DecisionRebuild }

// RecallRequest is what the caller injects to plan a chain recall.
type RecallRequest struct {
	// TargetNodeID is the unit to recall.
	TargetNodeID string
	// SiblingsRecalled is S — how many sibling units share this warm prefix in THIS
	// recall batch (the amortization factor). 1 (or 0, treated as 1) is the
	// single-unit case the gate almost always refuses; the live loop raises S when it
	// coalesces a fan-out of co-recalled siblings onto one hot prefix.
	SiblingsRecalled int
	// ReadMult is r — the provider's cached-read token multiplier (M1 calibration;
	// ~0.1 on the major providers). The replay cost is P·r.
	ReadMult float64
	// WarmDepth is how deep the prefix is currently warm: the chain index at/above
	// which the prefix is cached. PlanRecall replays only the cold tail (nodes at
	// depth >= WarmDepth); the live loop derives this from
	// cachemeta.FirstDivergeTokenOffset. 0 means the whole chain is cold.
	WarmDepth int
}

// RecallPlan is PlanRecall's verdict plus the economics and replay schedule that
// produced it, so a caller (or a `fak vcache prove-recall` proof) can show its work.
type RecallPlan struct {
	Decision          RecallDecision
	Reason            string
	PrefixTokens      int64      // P — the replayed prefix length (ancestors of the target)
	UnitTokens        int64      // U — the target's fresh-prefill length
	ReplayCost        float64    // P·r — read token-equivalents to replay the prefix
	FreshPrefillCost  float64    // U — fresh-prefill token-equivalents a cold recall pays
	AmortizedSavings  float64    // S·U — fresh prefills saved by one replay across S siblings
	Siblings          int        // S — the amortization factor used
	BreakEvenSiblings int        // smallest S that clears the gate (math.Ceil(P·r / U), floored at 1)
	Replay            ReplayPlan // the send-one-then-fan schedule (only meaningful on DecisionRebuild)
}

// ReplayCost is P·r — the read token-equivalents to replay a warm prefix of
// prefixTokens tokens at cached-read multiplier readMult (§11.0). Exported so the
// live loop and the CLI proof can show the cost before deciding.
func ReplayCost(prefixTokens int64, readMult float64) float64 {
	if prefixTokens < 0 || readMult < 0 {
		return 0
	}
	return float64(prefixTokens) * readMult
}

// RebuildNetPositive reports whether amortized chain rebuild beats fresh prefill —
// the §11.0 amortized cost gate:
//
//	rebuild iff P·r < S·U
//
// where P is the replayed prefix length, r the read multiplier, U the unit's
// fresh-prefill length, and S the sibling count (floored at 1). This REFUSES almost
// every single-unit rebuild (S=1: P·r ≥ U for any realistic prefix) and allows
// rebuild only for amortized fan-out.
func RebuildNetPositive(prefixTokens, unitTokens int64, siblings int, readMult float64) bool {
	siblings = floorAtOne(siblings)
	if unitTokens <= 0 {
		// Nothing to save: rebuild cannot beat a zero-cost cold send.
		return false
	}
	return ReplayCost(prefixTokens, readMult) < float64(siblings)*float64(unitTokens)
}

// BreakEvenSiblings is the smallest S (≥1) that clears the amortized cost gate —
// the crossover at which one chain replay becomes cheaper than the S fresh prefills
// it saves. Because the gate is a STRICT inequality (rebuild iff P·r < S·U), the
// break-even is floor(P·r / U) + 1 (NOT ceil: at an exact-integer ratio ceil returns
// the ratio itself, which does NOT clear a strict <). It is floored at 1 and +Inf
// when U≤0 (nothing to save). The §11.0 example (P=30000, r=0.1, U=10) gives
// floor(3000/10)+1 = 301: rebuild only wins once 301 sibling units share the hot
// prefix (at S=300 the replay cost 3000 is not strictly less than the 3000 saved).
func BreakEvenSiblings(prefixTokens, unitTokens int64, readMult float64) float64 {
	if unitTokens <= 0 {
		return math.Inf(1)
	}
	ratio := ReplayCost(prefixTokens, readMult) / float64(unitTokens)
	be := math.Floor(ratio) + 1
	if be < 1 {
		be = 1
	}
	return be
}

// ChainWarmable reports whether every node on a node's recall chain is warmable
// (Law D4). A secret/regulated ancestor makes the whole chain no-cache: rebuilding
// it would replay the secret byte-for-byte through the provider prefix cache. The
// live loop's canonicalizer (§12 net-new #6) classifies each node; this checks the
// chain before any economics run, the same short-circuit the M5 Governor applies.
func ChainWarmable(dag PrefixDAG, nodeID string) (bool, error) {
	chain, err := dag.ChainTo(nodeID)
	if err != nil {
		return false, err
	}
	for _, n := range chain {
		if !vcachegov.Warmable(n.Secret) {
			return false, nil
		}
	}
	return true, nil
}

// PlanRecall is the M4 entry point: given a prefix DAG, a recall request, and the
// enable flag, decide whether to rebuild the chain or send the unit cold. The
// decision tree, top to bottom:
//
//  1. Gated off (DefaultEnabled) → DecisionGatedOff: send the unit cold. M4 is off
//     by default; the live loop is what enables it.
//  2. Law D4 — a non-warmable ancestor → DecisionNoCache: never replay a secret.
//  3. Cost gate (§11.0) — P·r ≥ S·U → DecisionColdPrefill: replay would cost at
//     least as much as the fresh prefills it saves. Send the unit cold. This is the
//     correct decision for almost every single-unit recall.
//  4. Cost gate cleared — P·r < S·U → DecisionRebuild: amortized fan-out makes one
//     replay cheaper than the S fresh prefills it saves. Rebuild on the schedule
//     TopologicalReplay produces.
//
// Correctness never depends on the outcome (Law A2): whatever the verdict, the
// caller must always be able to re-send the full prefix; a rebuild is only ever a
// cost/latency win, never a license to elide resent context.
func PlanRecall(dag PrefixDAG, req RecallRequest, enabled bool) (RecallPlan, error) {
	target, ok := dag.node(req.TargetNodeID)
	if !ok {
		return RecallPlan{}, ErrMissingNode
	}
	prefixTokens, err := dag.PrefixTokens(req.TargetNodeID)
	if err != nil {
		return RecallPlan{}, err
	}
	plan := RecallPlan{
		PrefixTokens:      prefixTokens,
		UnitTokens:        target.Tokens,
		ReplayCost:        ReplayCost(prefixTokens, req.ReadMult),
		FreshPrefillCost:  float64(target.Tokens),
		Siblings:          floorAtOne(req.SiblingsRecalled),
		BreakEvenSiblings: breakEvenSiblingCount(prefixTokens, target.Tokens, req.ReadMult),
	}
	plan.AmortizedSavings = float64(plan.Siblings) * float64(target.Tokens)

	if !enabled {
		plan.Decision = DecisionGatedOff
		plan.Reason = "M4 chains & recall is gated off by default (DefaultEnabled=false); send the unit cold"
		return plan, nil
	}
	warmable, err := ChainWarmable(dag, req.TargetNodeID)
	if err != nil {
		return plan, err
	}
	if !warmable {
		plan.Decision = DecisionNoCache
		plan.Reason = "chain carries secret/regulated content (Law D4); take the no-cache path"
		return plan, nil
	}
	if !RebuildNetPositive(prefixTokens, target.Tokens, req.SiblingsRecalled, req.ReadMult) {
		plan.Decision = DecisionColdPrefill
		plan.Reason = "cost gate: replay cost >= amortized fresh prefills (§11.0); send the unit cold"
		return plan, nil
	}
	replay, err := dag.TopologicalReplay([]string{req.TargetNodeID}, req.WarmDepth)
	if err != nil {
		return plan, err
	}
	plan.Decision = DecisionRebuild
	plan.Replay = replay
	plan.Reason = "cost gate cleared: amortized fan-out (P·r < S·U); rebuild on the send-one-then-fan schedule"
	return plan, nil
}

func floorAtOne(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// breakEvenSiblingCount is the int form of BreakEvenSiblings, safe for the
// RecallPlan struct: +Inf (U≤0, nothing to save) reports as 0 (undefined).
func breakEvenSiblingCount(prefixTokens, unitTokens int64, readMult float64) int {
	be := BreakEvenSiblings(prefixTokens, unitTokens, readMult)
	if math.IsInf(be, 1) {
		return 0
	}
	return int(be)
}
