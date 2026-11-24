package routing

// switch_gate.go — the per-session stay-or-switch gate (OR-09).
//
// Moving an agent session to another model mid-conversation throws away the
// provider's prompt cache (and the local KV cache): the next turn re-pays the
// whole warm context at the candidate's uncached input price. The gate prices
// that loss and lets a switch through only when the configured quality
// advantage outweighs it. It is pure and fail-safe: a missing price or prior,
// or a session inside a tool loop, always stays. It decides nothing on its
// own; the router journals its verdict in shadow mode (OR-09 leaf 2).

import "sort"

// SwitchDecision is the gate's closed outcome.
type SwitchDecision string

const (
	SwitchStay   SwitchDecision = "stay"
	SwitchSwitch SwitchDecision = "switch"
)

// Closed reasons a SwitchVerdict carries.
const (
	// SwitchReasonNoCandidate: no candidate model, or the candidate is the
	// current model.
	SwitchReasonNoCandidate = "no_candidate"
	// SwitchReasonActiveToolLoop: the session is mid tool loop (the last
	// message is a tool result, or a tool call is required); a switch there
	// can break tool-call ids and signed reasoning blocks.
	SwitchReasonActiveToolLoop = "active_tool_loop"
	// SwitchReasonMissingSignals: a price or the quality prior is unknown, so
	// the gate cannot price the switch and stays (MissingSignals names them).
	SwitchReasonMissingSignals = "missing_signals"
	// SwitchReasonCacheLoss: the priced cache loss (plus the minimum
	// advantage) is at least the quality advantage.
	SwitchReasonCacheLoss = "cache_loss_exceeds_advantage"
	// SwitchReasonAdvantage: the quality advantage beats the priced cache
	// loss by more than the minimum advantage.
	SwitchReasonAdvantage = "advantage_exceeds_cache_loss"
)

// Closed names of the signals a SwitchVerdict can report missing.
const (
	SwitchSignalCurrentCachedPrice     = "current_cached_input_price"
	SwitchSignalCandidateUncachedPrice = "candidate_uncached_input_price"
	SwitchSignalQualityAdvantage       = "quality_advantage"
)

// switchTokensPerMillion converts per-million-token prices.
const switchTokensPerMillion = 1_000_000.0

// SwitchGateInput is everything the gate reads. Prices are USD per million
// input tokens; a nil price or prior is a missing signal, not zero.
type SwitchGateInput struct {
	CurrentModel   string
	CandidateModel string
	// WarmContextTokens is the session's context the current model holds
	// cached; a switch re-sends it uncached. Negative counts as zero.
	WarmContextTokens int
	// CurrentCachedInputPrice is what the warm context costs per million
	// tokens if the session stays (the current model's cached-read price).
	CurrentCachedInputPrice *float64
	// CandidateUncachedInputPrice is what the warm context costs per million
	// tokens on the candidate's first turn, before its own cache is warm.
	CandidateUncachedInputPrice *float64
	// QualityAdvantage is the static per-model-pair prior: the value, in USD
	// per turn, of serving the next turn on the candidate rather than the
	// current model. Negative means the candidate is worse.
	QualityAdvantage *float64
	// MinAdvantage is the net value a switch must exceed (zero by default).
	MinAdvantage float64
	// ActiveToolLoop is true while the session is inside a tool loop.
	ActiveToolLoop bool
}

// SwitchVerdict is the gate's answer with its audit.
type SwitchVerdict struct {
	Decision SwitchDecision `json:"decision"`
	Reason   string         `json:"reason"`
	// MissingSignals names the unknown inputs, sorted; empty when none.
	MissingSignals []string `json:"missing_signals,omitempty"`
	// CacheLossUSD is the warm context's extra cost on the candidate's first
	// turn: tokens x (candidate uncached - current cached) / 1e6, floored at 0.
	CacheLossUSD float64 `json:"cache_loss_usd"`
	// AdvantageUSD is the quality prior; NetUSD is AdvantageUSD - CacheLossUSD.
	AdvantageUSD float64 `json:"advantage_usd"`
	NetUSD       float64 `json:"net_usd"`
}

// SwitchGate decides whether a session should move from its current model to
// the candidate. Order: no candidate, then an active tool loop, then missing
// signals each stay before any pricing; only a fully priced, positive net
// above MinAdvantage switches.
func SwitchGate(in SwitchGateInput) SwitchVerdict {
	if in.CandidateModel == "" || in.CandidateModel == in.CurrentModel {
		return SwitchVerdict{Decision: SwitchStay, Reason: SwitchReasonNoCandidate}
	}
	if in.ActiveToolLoop {
		return SwitchVerdict{Decision: SwitchStay, Reason: SwitchReasonActiveToolLoop}
	}
	var missing []string
	if in.CurrentCachedInputPrice == nil {
		missing = append(missing, SwitchSignalCurrentCachedPrice)
	}
	if in.CandidateUncachedInputPrice == nil {
		missing = append(missing, SwitchSignalCandidateUncachedPrice)
	}
	if in.QualityAdvantage == nil {
		missing = append(missing, SwitchSignalQualityAdvantage)
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return SwitchVerdict{Decision: SwitchStay, Reason: SwitchReasonMissingSignals, MissingSignals: missing}
	}
	warm := max(in.WarmContextTokens, 0)
	loss := float64(warm) * (*in.CandidateUncachedInputPrice - *in.CurrentCachedInputPrice) / switchTokensPerMillion
	loss = max(loss, 0)
	v := SwitchVerdict{CacheLossUSD: loss, AdvantageUSD: *in.QualityAdvantage}
	v.NetUSD = v.AdvantageUSD - v.CacheLossUSD
	if v.NetUSD > in.MinAdvantage {
		v.Decision, v.Reason = SwitchSwitch, SwitchReasonAdvantage
	} else {
		v.Decision, v.Reason = SwitchStay, SwitchReasonCacheLoss
	}
	return v
}
