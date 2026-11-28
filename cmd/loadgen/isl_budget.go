// ISL (input-sequence-length) budget compensation for composed workloads.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/dataset/composer/base.py,
// disposition: the budget accounting is re-expressed in idiomatic Go; no code
// copied.
//
// Upstream keeps the synthetic wire ISL matching the user's --isl /
// --shared-system-prompt-length once cache-bust markers and chat-template
// wrapping are added at request time. Three components are subtracted from
// the target ISL budget so the composed workload respects the ISL constraint:
//
//	(a) cache-bust marker token cost (0 for NONE and warmup-isolation
//	    targets), routed to the system message for SYSTEM_* targets with a
//	    system prompt present, otherwise to the first user turn;
//	(b) chat-template wrapping: a per-request fixed cost (generation-prompt
//	    suffix) charged once to the first user turn, plus a per-message wrap
//	    (role header + end-of-turn marker) charged to every user turn;
//	(c) shared-system-prompt shrink: when the marker lands on the synthetic
//	    system prompt, the prompt is generated (marker cost) tokens shorter so
//	    its wire length still matches the configured length.
//
// Deviations forced by loadgen's pure-Go shape (no tokenizer):
//   - The chat-template overheads and the marker cost are caller-measured
//     inputs; EstimateChatTemplateOverheads ports the source's probe formula
//     from measured token lengths.
//   - A configured shared system length of 0 stands in for the source's
//     "not configured" (None): Go has no optional int, and a zero-length
//     shared prompt is degenerate (upstream's prefix_prompt_enabled requires
//     length > 0).
//   - Per-turn budgets floor at 1, mirroring the source's downstream sampler
//     (sample_positive_normal_integer floors draws at 1).
//   - Rounding uses math.Round (half away from zero) where the source uses
//     Python's round (half to even); probe means landing exactly on .5 are
//     the only divergence.
package main

import "math"

// CacheBustTarget mirrors aiperf's CacheBustTarget enum: where (and how) to
// inject a per-conversation cache-bust marker. Prefix variants diverge at
// token 0 of the prompt; suffix variants append after existing content.
type CacheBustTarget int

const (
	CacheBustNone CacheBustTarget = iota
	CacheBustSystemPrefix
	CacheBustSystemSuffix
	CacheBustFirstTurnPrefix
	CacheBustFirstTurnSuffix
	// Warmup-isolation targets carry no profiling-phase token cost: upstream
	// estimate_marker_token_cost returns 0 for them (their marker is only
	// injected during WARMUP).
	CacheBustWarmupIsolationSystem
	CacheBustWarmupIsolationFirstTurn
)

// ISLCompensation holds the three ISL-budget compensation components resolved
// at composer init, mirroring BaseDatasetComposer._init_isl_compensation.
type ISLCompensation struct {
	// PerRequestFixedTokens is the chat-template per-request fixed cost
	// (generation-prompt suffix), component (b). Charged once, to the first
	// user turn.
	PerRequestFixedTokens int
	// PerMsgWrapTokens is the chat-template per-message wrap cost (role
	// header + end-of-turn marker), component (b). Charged to every user
	// turn.
	PerMsgWrapTokens int
	// CacheBustMarkerTokens is the average token cost of the cache-bust
	// marker, component (a). 0 for NONE and warmup-isolation targets.
	CacheBustMarkerTokens int
	// MarkerOnSystemPrompt reports whether the marker lands on the system
	// message (SYSTEM_* target with a system prompt present).
	MarkerOnSystemPrompt bool
	// MarkerOnFirstUserTurn reports whether the marker lands on the first
	// user turn (any non-NONE target that did not land on the system
	// prompt).
	MarkerOnFirstUserTurn bool
}

// NewISLCompensation resolves the marker routing and the effective marker
// cost, mirroring _init_isl_compensation's component (a) routing:
// SYSTEM_* + system prompt -> marker on the system message; otherwise (incl.
// SYSTEM_* with no system message, and FIRST_TURN_*) -> first user turn.
//
// markerTokenCost is the caller-measured average marker cost
// (estimate_marker_token_cost); it is 0 for NONE and warmup-isolation
// targets, which the port enforces here so a mis-measured cost cannot leak
// into the budget. perRequestFixed/perMsgWrap are the caller-measured
// chat-template overheads; they are zeroed when applyChatTemplate is false
// (the source's --apply-chat-template opt-out).
func NewISLCompensation(target CacheBustTarget, hasSystemPrompt, applyChatTemplate bool, markerTokenCost, perRequestFixed, perMsgWrap int) ISLCompensation {
	c := ISLCompensation{}
	if applyChatTemplate {
		c.PerRequestFixedTokens = perRequestFixed
		c.PerMsgWrapTokens = perMsgWrap
	}
	if target == CacheBustNone {
		return c
	}
	if target != CacheBustWarmupIsolationSystem && target != CacheBustWarmupIsolationFirstTurn {
		c.CacheBustMarkerTokens = markerTokenCost
	}
	isSystemTarget := target == CacheBustSystemPrefix || target == CacheBustSystemSuffix
	c.MarkerOnSystemPrompt = isSystemTarget && hasSystemPrompt
	c.MarkerOnFirstUserTurn = !c.MarkerOnSystemPrompt
	return c
}

// FirstTurnAdjustment is the total tokens subtracted from the FIRST user
// turn's synthetic ISL: the per-request chat-template fixed cost, the
// per-message chat-template wrap, and the cache-bust marker when it lands on
// the first user turn. Mirrors first_turn_isl_adjustment.
//
// Prefix prompts are deliberately absent: they are additive, so the body
// keeps the full configured ISL and the prefix is prepended on top.
func (c ISLCompensation) FirstTurnAdjustment() int {
	adj := c.PerRequestFixedTokens + c.PerMsgWrapTokens
	if c.MarkerOnFirstUserTurn {
		adj += c.CacheBustMarkerTokens
	}
	return adj
}

// SubsequentTurnAdjustment is the tokens subtracted from each non-first user
// turn's synthetic ISL: just the per-message chat-template wrap. The
// per-request fixed cost and the cache-bust marker apply only to the first
// turn (later turns' raw messages are not mutated; the generation-prompt
// suffix emits once per request, not per message). Mirrors
// subsequent_turn_isl_adjustment.
func (c ISLCompensation) SubsequentTurnAdjustment() int {
	return c.PerMsgWrapTokens
}

// CompensatedSharedSystemLength shrinks the synthetic shared system prompt by
// the marker cost when the marker lands on the system prompt, so its wire
// length still matches the configured length. Mirrors component (c):
// max(1, configured - marker).
//
// ok is false when no compensation applies (marker not on the system prompt,
// or no configured shared system length); the configured value passes through
// unchanged in that case.
func (c ISLCompensation) CompensatedSharedSystemLength(configured int) (int, bool) {
	if !c.MarkerOnSystemPrompt || configured <= 0 {
		return configured, false
	}
	v := configured - c.CacheBustMarkerTokens
	if v < 1 {
		v = 1
	}
	return v, true
}

// ISLRequest describes one composed workload request: how many user turns it
// carries and, when it carries a synthetic shared system prompt, that
// prompt's configured length (0 = none).
type ISLRequest struct {
	Turns int
	// SharedSystemLength is the configured synthetic shared system prompt
	// length; 0 means the request carries none.
	SharedSystemLength int
}

// ISLCompensatedRequest is the compensated budget for one request.
type ISLCompensatedRequest struct {
	// TurnBudgets is the synthetic ISL to generate for each user turn, in
	// turn order.
	TurnBudgets []int
	// SharedSystemLength is the compensated shared system prompt length
	// (0 when the request carries none).
	SharedSystemLength int
}

// CompensateISLBudget returns the synthetic ISL to generate for each turn of
// a request so the composed wire ISL respects the target budget: the first
// user turn is charged the full first-turn adjustment, each later turn only
// the per-message wrap. Budgets floor at 1, mirroring the source's
// downstream sampler.
func CompensateISLBudget(target int, c ISLCompensation, turns int) []int {
	if turns <= 0 {
		return nil
	}
	out := make([]int, turns)
	for i := range out {
		adj := c.SubsequentTurnAdjustment()
		if i == 0 {
			adj = c.FirstTurnAdjustment()
		}
		out[i] = target - adj
		if out[i] < 1 {
			out[i] = 1
		}
	}
	return out
}

// CompensateWorkload applies the ISL budget compensation to a set of
// requests, returning per-request, per-turn synthetic ISL budgets plus the
// compensated shared system prompt length for each request that carries one.
func CompensateWorkload(target int, c ISLCompensation, requests []ISLRequest) []ISLCompensatedRequest {
	out := make([]ISLCompensatedRequest, len(requests))
	for i, r := range requests {
		out[i].TurnBudgets = CompensateISLBudget(target, c, r.Turns)
		if r.SharedSystemLength > 0 {
			if v, ok := c.CompensatedSharedSystemLength(r.SharedSystemLength); ok {
				out[i].SharedSystemLength = v
			} else {
				out[i].SharedSystemLength = r.SharedSystemLength
			}
		}
	}
	return out
}

// ChatTemplateProbe carries the measured token lengths for one probe sample:
// the chat template applied to a single user message (with generation
// prompt), to a three-message conversation (user/assistant/user, with
// generation prompt), and to the bare sample text.
type ChatTemplateProbe struct {
	SingleLen int
	TripleLen int
	BareLen   int
}

// EstimateChatTemplateOverheads decomposes chat-template overhead into
// (perRequestFixed, perMsgWrap) from measured probe lengths, mirroring
// _estimate_chat_template_overheads:
//
//	wire_tokens = per_request_fixed + Σ_msgs (per_msg_wrap + content_tokens)
//
// so per_msg_wrap = (triple - single - 2*bare) / 2 and per_request_fixed =
// single - bare - per_msg_wrap - bos. bosTokens is the tokenizer's prompt
// special-token count (BOS et al.), deliberately excluded from
// perRequestFixed by the source (every ISL path subtracts special tokens on
// its own; double-counting would shorten prompts). A non-positive wrap or
// fixed cost on any probe yields (0, 0): skipping compensation beats
// over-correcting.
func EstimateChatTemplateOverheads(probes []ChatTemplateProbe, bosTokens int) (int, int) {
	if len(probes) == 0 {
		return 0, 0
	}
	var fixedSum, wrapSum float64
	for _, p := range probes {
		avgWrap := float64(p.TripleLen-p.SingleLen-2*p.BareLen) / 2
		perRequestFixed := float64(p.SingleLen-p.BareLen) - avgWrap - float64(bosTokens)
		if avgWrap < 0 || perRequestFixed < 0 {
			return 0, 0
		}
		wrapSum += avgWrap
		fixedSum += perRequestFixed
	}
	n := float64(len(probes))
	return int(math.Round(fixedSum / n)), int(math.Round(wrapSum / n))
}
