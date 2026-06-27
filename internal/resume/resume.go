// Package resume is the deterministic, observable RESUME decision: given the facts
// about a dormant agent session about to be brought back, it computes — with no
// clock, no network, and no model — exactly what happens to the provider prompt
// cache on the first turn after resume, what each re-entry STRATEGY costs, and which
// one the kernel recommends. It is the computable answer to the operator question
// "I am resuming a 250k-token session — what happens, and what does it cost?"
//
// # The gap it closes
//
// fak already has every durable resume PRIMITIVE — the drive re-attach
// (internal/session.Restore), the portable model-agnostic image
// (internal/sessionimage, KVIncluded=false: the KV cache is rebuilt on resume), the
// budget-reset carryover (internal/sessionreset), and the in-flight cut-vs-reset
// policy (internal/gateway.ResetScore). What was missing was the thing the operator
// actually asks for at the resume boundary: a FIRST-CLASS, deterministic, observable
// process that turns those primitives into a priced verdict. ResetScore decides
// cut-vs-reset for a LIVE session from rolling, observed cache counters; resume is the
// canonical COLD boundary — a session that has been dormant long enough that the
// provider's prompt-cache prefix has, by construction, aged out of its TTL — and it
// had no computed plan of its own. This package is that plan.
//
// # Why resume is the cold-cache case
//
// A provider prompt cache is ephemeral: Anthropic's default `cache_control` breakpoint
// is a 5-minute TTL (300s); the extended tier is 1 hour (3600s). A resumed session has
// almost always been idle longer than that — a crashed-and-relaunched Claude Code
// session, a session re-homed onto a fresh account, an image carried to another host.
// So the first turn after resume re-prefills the WHOLE resident transcript at the cache
// WRITE price (1.25x base input at 5m, 2.0x at 1h), with no offsetting read. For a 250k
// session that is the single most expensive turn the session will ever run. This package
// makes that cost a number you can see BEFORE you pay it, and prices the two ways to make
// it smaller — CUT (shed the middle, keep prefix + recent tail) and RESET (distill to a
// compact seed) — against keeping the whole transcript (RESUME_FULL).
//
// # Honest fences
//
//   - PROJECTION, not a claim. The cache POSTURE (cold/warm) is a deterministic
//     projection over idle-time-vs-TTL, not an observation of the provider's cache. fak
//     guarantees the prefix bytes it ships; whether the provider actually still has the
//     cache is the provider's call. Same discipline gateway/cache_pricing.go applies to
//     the raw counters: a dollar figure here is a COST PROJECTION over the resident-token
//     count, never a fak-witnessed bill.
//   - CUT-BY-DEFAULT, reset is the bigger hammer. On a cold resume the package recommends
//     CUT (it preserves the most context per dollar shed). RESET is always PRICED and
//     reported as the cheaper alternative, but never auto-recommended in v1 — the same
//     posture internal/sessionreset ships under (the auto-reset path is default-off).
//   - On ambiguity, do nothing. When idle time is unknown the package cannot tell a warm
//     prefix from a cold one, so it recommends RESUME_FULL (never shed on a guess) — the
//     same conservatism the byte-splice compactor takes when a span is ambiguous.
//
// This is a tier-1 foundation leaf: stdlib-only, imports nothing internal, registers
// nothing, off the request path. The pure decision lives here; the I/O (load a session
// image, count its resident tokens, pick model pricing) lives in the cmd/fak/resume.go
// shell, exactly as session (the decision) and cmd/fak/session_cmd.go (the wire) split.
package resume

import "math"

// CacheTTL names the provider cache_control TTL the dormant session used, mirroring the
// Anthropic grammar: a bare {"type":"ephemeral"} breakpoint is the 5-minute tier, and
// {"type":"ephemeral","ttl":"1h"} is the 1-hour tier. It drives both the cold/warm
// posture cutoff (Seconds) and the cache-write premium (WriteMultiplier).
type CacheTTL string

const (
	// TTL5m is the default ephemeral cache tier (300s TTL, 1.25x write premium).
	TTL5m CacheTTL = "5m"
	// TTL1h is the extended ephemeral cache tier (3600s TTL, 2.0x write premium).
	TTL1h CacheTTL = "1h"
)

// Seconds is the TTL in wall-clock seconds — the cutoff a session's idle time is compared
// against to decide whether the provider prefix has aged out. An unset/unknown tier falls
// back to the 5-minute default (the shorter window: a conservative cold verdict, never a
// falsely-warm one).
func (t CacheTTL) Seconds() int64 {
	if t == TTL1h {
		return 3600
	}
	return 300
}

// WriteMultiplier is the cache-WRITE price relative to base input for this TTL — the
// premium the cold first turn pays to (re)establish the warm prefix. Unset falls back to
// the 5-minute tier, so a missing TTL is priced as the cheaper write, never as a free one.
func (t CacheTTL) WriteMultiplier() float64 {
	if t == TTL1h {
		return CacheWrite1hMultiplier
	}
	return CacheWrite5mMultiplier
}

// The published Anthropic prompt-cache price multipliers, RELATIVE to the model's base
// input per-token price. These mirror gateway/cache_pricing.go's constants verbatim; they
// are redeclared here (not imported) because gateway is a tier-4 integrator and this is a
// tier-1 foundation leaf — the layered-DAG rule forbids the upward import. The values are
// stable published economics, not a fak measurement.
const (
	// CacheReadMultiplier is the price of a cached-prefix READ relative to base input.
	CacheReadMultiplier = 0.1
	// CacheWrite5mMultiplier is the price of a 5-minute-TTL cache WRITE relative to base input.
	CacheWrite5mMultiplier = 1.25
	// CacheWrite1hMultiplier is the price of a 1-hour-TTL cache WRITE relative to base input.
	CacheWrite1hMultiplier = 2.0
)

// Conservative defaults applied to a zero/unset Input axis. They match the shipped
// surfaces they mirror so the plan agrees with what the live path would actually do.
const (
	// DefaultShedBudgetTokens is the CUT target — the resident-token budget a shed keeps
	// (prefix + recent tail). It mirrors the ~48k default fak guard's byte-splice
	// compactor fires at (docs/explainers/long-sessions-keep-the-cache-hit.md).
	DefaultShedBudgetTokens = 48000
	// DefaultSeedTokens is the RESET seed size — a compact carryover (durable facts +
	// task recap + verbatim tail) the distiller produces (internal/sessionreset). A
	// deliberately small default; the real seed size is the caller's to supply.
	DefaultSeedTokens = 4000
	// DefaultHorizonTurns is the number of turns assumed to remain after resume, over
	// which the one-time cold re-prefill is amortized.
	DefaultHorizonTurns = 20
	// DefaultOutputTokensPerTurn is the per-turn completion size used to price the output
	// axis of each modeled turn.
	DefaultOutputTokensPerTurn = 512
)

// Posture is the deterministic projection of the provider cache's state at resume.
type Posture string

const (
	// PostureCold: idle time has met or exceeded the TTL, so the cached prefix has aged
	// out — the first turn re-prefills the whole resident transcript at the WRITE price.
	PostureCold Posture = "cold"
	// PostureWarm: idle time is within the TTL, so the prefix MAY still be served from the
	// provider cache (a projection, never a guarantee) — the first turn may bill as a READ.
	PostureWarm Posture = "warm"
	// PostureUnknown: idle time was not supplied, so warm-vs-cold cannot be told apart —
	// the plan recommends RESUME_FULL (do not shed on a guess) and prices the cold case.
	PostureUnknown Posture = "unknown"
)

// Strategy names a way to re-enter a dormant session.
type Strategy string

const (
	// StrategyResumeFull re-prefills the entire resident transcript — full context, the
	// most expensive cold turn.
	StrategyResumeFull Strategy = "resume_full"
	// StrategyCut sheds the middle turns and keeps a prefix + recent tail down to the shed
	// budget — most context preserved per dollar shed (the cut-by-default recommendation).
	StrategyCut Strategy = "cut"
	// StrategyReset distills the transcript to a compact carryover seed — cheapest, most
	// context dropped (priced and reported, but not auto-recommended in v1).
	StrategyReset Strategy = "reset"
)

// The closed reason vocabulary for a Report.Recommendation, so an observability sink can
// record WHY without exposing any transcript content.
const (
	// ReasonUnknownIdle: idle time unknown — recommend RESUME_FULL, never shed on a guess.
	ReasonUnknownIdle = "unknown_idle"
	// ReasonSmallSession: the resident transcript already fits the shed budget — there is
	// nothing meaningful to shed, so RESUME_FULL and CUT are the same plan.
	ReasonSmallSession = "small_session"
	// ReasonWarmPrefixIntact: the prefix is likely still cached and the horizon is short —
	// keep the whole transcript; bursting a live cache to shed bytes would cost more.
	ReasonWarmPrefixIntact = "warm_prefix_intact"
	// ReasonWarmHorizonPaysBurst: the prefix is warm, but enough turns remain that the
	// per-turn read savings from shedding repay the one-time re-prefill — CUT pays back.
	ReasonWarmHorizonPaysBurst = "warm_horizon_pays_burst"
	// ReasonColdPrefillShed: the prefix has aged out, so the cold re-prefill is unavoidable
	// — shed the middle to make that one expensive turn as small as the kept tail allows.
	ReasonColdPrefillShed = "cold_prefill_shed"
)

// Pricing is the model's BASE per-million-token price on the two billed axes. The cache
// multipliers apply on top of InputPerMTokUSD; OutputPerMTokUSD prices the completion. The
// caller supplies the numbers for the model in play (Opus 4.8 = {5, 25}, Sonnet 4.6 =
// {3, 15}, Haiku 4.5 = {1, 5}), so this leaf stays correct as prices change without edits.
type Pricing struct {
	InputPerMTokUSD  float64 `json:"input_per_mtok_usd"`
	OutputPerMTokUSD float64 `json:"output_per_mtok_usd"`
}

// Input is everything Plan needs — all the facts about the resume, none of the transcript.
type Input struct {
	// ResidentTokens is the size of the context that would be re-prefilled on RESUME_FULL
	// (the whole transcript the client would re-send). This is the 250k in the headline.
	ResidentTokens int `json:"resident_tokens"`
	// IdleSeconds is how long the session was dormant before this resume, the input that
	// decides cold-vs-warm against the TTL. A negative value means "unknown".
	IdleSeconds int64 `json:"idle_seconds"`
	// TTL is the provider cache tier the session used (5m default, 1h extended).
	TTL CacheTTL `json:"ttl"`
	// Pricing is the model's base per-MTok price (caller-supplied, provider-agnostic).
	Pricing Pricing `json:"pricing"`
	// HorizonTurns is how many turns are expected to remain after resume (0 => default).
	HorizonTurns int `json:"horizon_turns"`
	// ShedBudgetTokens is the CUT target (0 => DefaultShedBudgetTokens).
	ShedBudgetTokens int `json:"shed_budget_tokens"`
	// SeedTokens is the RESET carryover size (0 => DefaultSeedTokens).
	SeedTokens int `json:"seed_tokens"`
	// OutputTokensPerTurn prices each modeled turn's completion (0 => default).
	OutputTokensPerTurn int `json:"output_tokens_per_turn"`
}

// withDefaults returns a copy with zero/unset axes filled and out-of-range values clamped,
// so Plan is total: any Input yields a defined Report. Negative token counts clamp to 0;
// a shed budget or seed larger than the resident size clamps to it (you cannot shed to
// MORE than you have); a non-positive horizon becomes the default (at least one turn).
func (in Input) withDefaults() Input {
	out := in
	if out.ResidentTokens < 0 {
		out.ResidentTokens = 0
	}
	if out.HorizonTurns <= 0 {
		out.HorizonTurns = DefaultHorizonTurns
	}
	if out.ShedBudgetTokens <= 0 {
		out.ShedBudgetTokens = DefaultShedBudgetTokens
	}
	if out.ShedBudgetTokens > out.ResidentTokens {
		out.ShedBudgetTokens = out.ResidentTokens
	}
	if out.SeedTokens <= 0 {
		out.SeedTokens = DefaultSeedTokens
	}
	if out.SeedTokens > out.ResidentTokens {
		out.SeedTokens = out.ResidentTokens
	}
	if out.OutputTokensPerTurn <= 0 {
		out.OutputTokensPerTurn = DefaultOutputTokensPerTurn
	}
	if out.TTL != TTL1h {
		out.TTL = TTL5m
	}
	return out
}

// StrategyCost is one re-entry strategy priced end to end. Every dollar figure is a
// projection over the resident-token count at the supplied base price (see the package
// fence); none is a witnessed bill.
type StrategyCost struct {
	Strategy Strategy `json:"strategy"`
	// PrefillTokens is the context this strategy re-prefills on the first turn (N, the shed
	// budget, or the seed).
	PrefillTokens int `json:"prefill_tokens"`
	// ContextKeptFraction is PrefillTokens / ResidentTokens (1.0 for RESUME_FULL): how much
	// of the original context the strategy carries forward.
	ContextKeptFraction float64 `json:"context_kept_fraction"`
	// ColdReprefillUSD is the cache-WRITE cost of establishing this strategy's prefill from
	// cold — the price tag of the cold re-prefill itself, reported even when the posture is
	// warm so the "what if it has aged out" cost is always visible.
	ColdReprefillUSD float64 `json:"cold_reprefill_usd"`
	// FirstTurnUSD is the first turn's total cost under the projected posture: a cache WRITE
	// of the prefill (cold/unknown) or a cache READ (warm), plus the turn's output.
	FirstTurnUSD float64 `json:"first_turn_usd"`
	// HorizonUSD is the first turn plus (HorizonTurns-1) steady-state warm turns (each
	// re-reading the now-established prefix at the read multiplier, plus output).
	HorizonUSD float64 `json:"horizon_usd"`
}

// Report is the full deterministic verdict — the observable record a CLI emits as JSON or
// a table, and the same value a gateway/guard resume hook would fold into an audit row.
type Report struct {
	// Echoed, defaulted inputs so the record is self-describing.
	ResidentTokens      int      `json:"resident_tokens"`
	IdleSeconds         int64    `json:"idle_seconds"`
	TTL                 CacheTTL `json:"ttl"`
	TTLSeconds          int64    `json:"ttl_seconds"`
	HorizonTurns        int      `json:"horizon_turns"`
	OutputTokensPerTurn int      `json:"output_tokens_per_turn"`
	Pricing             Pricing  `json:"pricing"`

	// Posture is the projected cache state and PostureReason the closed token explaining it.
	Posture       Posture `json:"posture"`
	PostureReason string  `json:"posture_reason"`

	// Strategies is the priced set, always in the fixed order resume_full, cut, reset.
	Strategies []StrategyCost `json:"strategies"`

	// Recommended is the strategy the kernel picks and Reason the closed token for why.
	Recommended Strategy `json:"recommended"`
	Reason      string   `json:"reason"`
	// RecommendedSavingsUSD is RESUME_FULL.HorizonUSD - Recommended.HorizonUSD (>= 0): the
	// projected horizon savings of following the recommendation over resuming the whole
	// transcript. Zero when the recommendation IS RESUME_FULL.
	RecommendedSavingsUSD float64 `json:"recommended_savings_usd"`
	// BreakEvenTurns is how many post-resume turns a CUT takes to repay its one-time
	// re-prefill via the per-turn read savings on the shed tokens — the warm-burst gate
	// (when HorizonTurns exceeds it, a cut pays back even on a warm prefix). 0 when there
	// is nothing to shed (the resident size already fits the shed budget).
	BreakEvenTurns int `json:"break_even_turns"`
}

// Plan is THE deterministic resume decision: same Input in, same Report out — no clock, no
// I/O, no model. It prices RESUME_FULL / CUT / RESET against the projected cache posture
// and returns the recommended re-entry with a closed reason. It is total over every Input
// (see withDefaults).
func Plan(in Input) Report {
	in = in.withDefaults()
	post, postReason := projectPosture(in.IdleSeconds, in.TTL)

	full := priceStrategy(StrategyResumeFull, in.ResidentTokens, in, post)
	cut := priceStrategy(StrategyCut, in.ShedBudgetTokens, in, post)
	reset := priceStrategy(StrategyReset, in.SeedTokens, in, post)

	breakEven := breakEvenTurns(in.ResidentTokens, in.ShedBudgetTokens, in.TTL)
	rec, reason := recommend(in, post, breakEven)

	chosen := full
	switch rec {
	case StrategyCut:
		chosen = cut
	case StrategyReset:
		chosen = reset
	}
	savings := full.HorizonUSD - chosen.HorizonUSD
	if savings < 0 {
		savings = 0
	}

	return Report{
		ResidentTokens:        in.ResidentTokens,
		IdleSeconds:           in.IdleSeconds,
		TTL:                   in.TTL,
		TTLSeconds:            in.TTL.Seconds(),
		HorizonTurns:          in.HorizonTurns,
		OutputTokensPerTurn:   in.OutputTokensPerTurn,
		Pricing:               in.Pricing,
		Posture:               post,
		PostureReason:         postReason,
		Strategies:            []StrategyCost{full, cut, reset},
		Recommended:           rec,
		Reason:                reason,
		RecommendedSavingsUSD: savings,
		BreakEvenTurns:        breakEven,
	}
}

// projectPosture maps idle time to a cache posture. Negative idle is unknown; idle at or
// past the TTL is cold (the prefix has aged out); within the TTL is warm (may still be
// cached — a projection, never a guarantee).
func projectPosture(idle int64, ttl CacheTTL) (Posture, string) {
	if idle < 0 {
		return PostureUnknown, "idle_unknown"
	}
	if idle >= ttl.Seconds() {
		return PostureCold, "idle_exceeds_ttl"
	}
	return PostureWarm, "idle_within_ttl"
}

// priceStrategy prices one strategy that re-prefills `prefill` tokens. The first turn is a
// cache WRITE of the prefill when the prefix is cold or unknown (establish it from scratch)
// and a cache READ when warm (it may still be served); every later turn in the horizon is a
// steady-state warm read of the established prefix. Output is priced on every modeled turn.
func priceStrategy(s Strategy, prefill int, in Input, post Posture) StrategyCost {
	if prefill < 0 {
		prefill = 0
	}
	inPerTok := perToken(in.Pricing.InputPerMTokUSD)
	outPerTok := perToken(in.Pricing.OutputPerMTokUSD)
	output := float64(in.OutputTokensPerTurn) * outPerTok

	coldReprefill := float64(prefill) * inPerTok * in.TTL.WriteMultiplier()
	warmRead := float64(prefill) * inPerTok * CacheReadMultiplier

	firstTurn := coldReprefill + output // cold / unknown: establish the prefix
	if post == PostureWarm {
		firstTurn = warmRead + output // warm: the prefix may still be served from cache
	}
	warmTurn := warmRead + output
	horizon := firstTurn + float64(in.HorizonTurns-1)*warmTurn

	frac := 0.0
	if in.ResidentTokens > 0 {
		frac = float64(prefill) / float64(in.ResidentTokens)
	} else if prefill == 0 {
		frac = 1.0 // a zero-token session: every strategy "keeps" all (none) of it
	}

	return StrategyCost{
		Strategy:            s,
		PrefillTokens:       prefill,
		ContextKeptFraction: frac,
		ColdReprefillUSD:    coldReprefill,
		FirstTurnUSD:        firstTurn,
		HorizonUSD:          horizon,
	}
}

// recommend applies the cut-by-default rungs in order. The order IS the policy:
//  1. Unknown idle -> RESUME_FULL (never shed on a guess).
//  2. The resident transcript already fits the shed budget -> RESUME_FULL (nothing to shed).
//  3. Warm prefix: keep it (RESUME_FULL) unless the horizon is long enough that the cut
//     repays its burst (then CUT).
//  4. Cold prefix: the re-prefill is unavoidable -> CUT to shrink that one expensive turn.
//
// RESET is never returned here: it is priced and reported as the cheaper alternative, but
// auto-recommending the bigger hammer is deferred (the same default-off posture
// internal/sessionreset ships under).
func recommend(in Input, post Posture, breakEven int) (Strategy, string) {
	switch {
	case post == PostureUnknown:
		return StrategyResumeFull, ReasonUnknownIdle
	case in.ResidentTokens <= in.ShedBudgetTokens:
		return StrategyResumeFull, ReasonSmallSession
	case post == PostureWarm:
		if in.HorizonTurns > breakEven {
			return StrategyCut, ReasonWarmHorizonPaysBurst
		}
		return StrategyResumeFull, ReasonWarmPrefixIntact
	default: // PostureCold
		return StrategyCut, ReasonColdPrefillShed
	}
}

// breakEvenTurns is how many post-resume turns a CUT takes to repay its one-time re-prefill
// of the kept tail via the per-turn read savings on the shed tokens. It is the exact
// formula from docs/explainers/long-sessions-keep-the-cache-hit.md, instantiated for the
// resume cut: the kept portion (the shed budget) is the invalidated suffix that must be
// re-written; the shed portion (resident - budget) is the cached tokens whose per-turn read
// you stop paying:
//
//	break_even = ceil((write_mult - read_mult) * kept / (read_mult * shed))
//
// 0 when there is nothing to shed (resident <= budget). The TTL picks the write multiplier.
func breakEvenTurns(resident, shedBudget int, ttl CacheTTL) int {
	shed := resident - shedBudget
	if shed <= 0 || shedBudget <= 0 {
		return 0
	}
	num := (ttl.WriteMultiplier() - CacheReadMultiplier) * float64(shedBudget)
	den := CacheReadMultiplier * float64(shed)
	if den <= 0 {
		return 0
	}
	return int(math.Ceil(num / den))
}

// perToken converts a per-MTok price to a per-token price.
func perToken(perMTok float64) float64 { return perMTok / 1_000_000 }
