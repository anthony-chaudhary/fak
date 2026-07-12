package model

import "math"

// selfspecgov.go — the runtime acceptance governor for MTP self-speculation
// (issue #4354; colibri `spec_decode` auto-off, `c/glm.c:1606`, verdict `inspire`,
// clean-room, no bytes vendored). It is the missing consumer half the seam at
// config.go:789 (SelfSpeculationSubstrateReady) named: the metadata halves exist
// (HasMTPHead / NumMTPLayers / RetainMTP) but nothing yet decides, at runtime,
// whether drafting is still worth its cost. This file adds that decision — and
// ONLY the decision. It is a pure tier-1 function: no draft/verify decode, no
// I/O, no package deps beyond math, so it changes no default runtime behavior
// (a zero-value governor speculates nothing) exactly like the readiness seam.
//
// It mirrors the shape of abi/speculate.go's empirical-α gate (warmupFloor +
// effectiveProb): below a warmup floor the measured rate is not yet trustworthy
// so the configured depth governs (you MUST speculate through warmup — a cold
// expert cache cannot warm if you never draft); past the floor the MEASURED
// signal governs and can drive the depth to 0 (auto-off).
//
// THE COLIBRI LESSON — charge the tax against PAGE-INS, not FLOPs. colibri's
// speculation cost is dominated by disk expert page-ins (SSD reads), not compute:
// on a cold cache each verified draft routes to extra experts (~660 -> ~1100
// expert-loads/token), so speculation is a net TIME loss until the cache warms.
// The only honest disable criterion therefore charges a rejected draft against
// the expert page-in cost it caused — the same lesson transfers to any fak
// backend where experts stream from a slow tier. So the governor's decision is
// an economic one in page-in units: keep drafting only while the page-in tax the
// drafts incur is repaid by the decode work the accepted drafts avoid.

// defaultSelfSpecWarmupDrafts is the warmup floor used when SelfSpecGovernor
// .WarmupDrafts is unset: until this many drafts have been observed the measured
// accept-rate is too small a sample to trust, so the governor keeps drafting at
// the configured depth (the "invest through warmup" prior). Mirrors
// abi.defaultMinTrials — small enough to adapt quickly, large enough that a few
// unlucky early rejections cannot auto-off a genuinely good pattern.
const defaultSelfSpecWarmupDrafts = 32

// SelfSpecGovernor is the pure tier-1 acceptance governor: given the rolling
// self-speculation accept-rate and the measured per-draft expert page-in cost, it
// decides the next draft depth — the configured MaxDraftDepth while drafting pays
// for its page-ins, or 0 (auto-off) once a rejected draft's page-in cost outweighs
// the decode time it saves. It holds only thresholds; the decision (DraftDepth) is
// a pure function of its inputs, so it is trivially unit-testable and carries no
// runtime state.
//
// DEFAULT-OFF: the zero value (MaxDraftDepth == 0) returns 0 from DraftDepth for
// every input — a kernel that never opts in speculates nothing, exactly as today.
type SelfSpecGovernor struct {
	// MaxDraftDepth is the speculation depth N to use while drafting is paying off
	// (the "on" state). 0 disables the governor entirely (the default-off posture).
	MaxDraftDepth int

	// WarmupDrafts is the minimum number of observed drafts before the measured
	// accept-rate is trusted. Below it DraftDepth returns MaxDraftDepth regardless
	// of the economics — you cannot warm a cold expert cache without first paying
	// the page-in tax, so the governor invests through warmup. 0 => the default.
	WarmupDrafts int

	// BasePageInsPerToken is the expert page-in count a plain (non-speculative)
	// decode of one token pays — the ~660 baseline in colibri's figures. It is the
	// reference the per-draft page-in tax is charged against. Must be > 0 for the
	// economic auto-off path; when unset (<= 0) the cost model is unarmed and the
	// governor never spuriously disables (it keeps MaxDraftDepth past warmup).
	BasePageInsPerToken float64
}

// warmupFloor is the effective WarmupDrafts (defaulted). Mirrors
// abi.Speculator.warmupFloor.
func (g SelfSpecGovernor) warmupFloor() int {
	if g.WarmupDrafts > 0 {
		return g.WarmupDrafts
	}
	return defaultSelfSpecWarmupDrafts
}

// DraftDepth is the governor's decision: the draft depth to use for the next
// step given the rolling acceptance rate, the measured per-draft expert page-in
// count (the MARGINAL expert page-ins — SSD reads — one drafted token adds to the
// verify pass over a plain decode; high on a cold cache, ~0 once the experts are
// resident), and how many drafts have been observed so far. It returns either the
// configured MaxDraftDepth (keep drafting) or 0 (auto-off).
//
//   - Disabled / default-off: MaxDraftDepth <= 0 => 0 (never speculate).
//   - Warmup: fewer than warmupFloor drafts observed => MaxDraftDepth. The sample
//     is too small to trust and the cache needs drafting to warm, so the governor
//     invests (mirrors abi's declared-prior-governs-until-warm rule).
//   - Post-warmup: the economic decision (speculationPaysOff). Keep MaxDraftDepth
//     while the page-in tax is repaid; return 0 the moment it is not — the
//     cold-cache -> 0 transition colibri's DRAFT=0 auto-off performs.
func (g SelfSpecGovernor) DraftDepth(acceptRate, pageInsPerDraft float64, observedDrafts int) int {
	n := g.MaxDraftDepth
	if n <= 0 {
		return 0 // disabled / default-off: never speculate
	}
	if observedDrafts < g.warmupFloor() {
		return n // warmup: invest at the configured depth until the sample is trustworthy
	}
	if g.speculationPaysOff(acceptRate, pageInsPerDraft, n) {
		return n
	}
	return 0 // auto-off: the page-in tax is no longer repaid
}

// speculationPaysOff is the honest page-in economics, evaluated post-warmup. A
// draft+verify cycle at depth n with acceptance rate a commits
//
//	yield = (1 - a^(n+1)) / (1 - a)
//
// tokens (the standard speculative-decoding expectation, Leviathan et al.
// 2211.17192): the one token a plain step already yields plus, in expectation,
// the run of accepted drafts. Speculation's EXTRA cost is the page-in tax the n
// drafted tokens add to the verify — n * pageInsPerDraft marginal expert page-ins.
// Its benefit is the decode work the BONUS tokens avoid: the (yield - 1) tokens
// beyond the guaranteed one, each of which a plain decoder would have paid
// BasePageInsPerToken to produce. So drafting pays iff
//
//	n * pageInsPerDraft  <  base * (yield - 1)
//
// i.e. the tax is less than the avoided-decode saving. On a cold cache
// pageInsPerDraft is elevated AND acceptance is low (yield -> 1, saving -> 0), so
// the tax dominates and the governor disables. As the cache warms (pageInsPerDraft
// -> 0) and acceptance rises, the saving overtakes the tax and drafting re-enables.
// The tax is charged against PAGE-INS, never FLOPs — the whole point of the axis.
//
// When BasePageInsPerToken <= 0 the cost model is unarmed (no baseline to charge
// against); the governor cannot honestly prove a loss and so keeps drafting.
func (g SelfSpecGovernor) speculationPaysOff(acceptRate, pageInsPerDraft float64, n int) bool {
	base := g.BasePageInsPerToken
	if base <= 0 {
		return true // cost model unarmed: never spuriously auto-off
	}
	if pageInsPerDraft <= 0 {
		// No extra page-ins: drafting is free on the cost side, so it pays as long
		// as it yields any bonus token at all (acceptance > 0).
		return acceptRate > 0
	}
	tax := float64(n) * pageInsPerDraft
	saving := base * (expectedTokensPerCycle(acceptRate, n) - 1)
	return tax < saving
}

// expectedTokensPerCycle is the expected number of tokens a single draft+verify
// cycle commits at depth n with acceptance rate a: (1 - a^(n+1)) / (1 - a), the
// standard speculative-decoding yield. a is clamped to [0,1]; at a==1 every draft
// is accepted so the yield is the full n+1, and at a==0 only the one guaranteed
// correction token lands (yield 1). It never returns less than 1 — a verify always
// commits at least the guaranteed token — so (yield - 1) is a well-defined
// non-negative bonus.
func expectedTokensPerCycle(acceptRate float64, n int) float64 {
	a := clamp01(acceptRate)
	if n <= 0 {
		return 1 // no drafts: only the guaranteed token
	}
	if a >= 1 {
		return float64(n + 1)
	}
	return (1 - math.Pow(a, float64(n+1))) / (1 - a)
}
