package harnesskit

import (
	"errors"
	"fmt"
	"time"
)

// ContextEnvelope is the single canonical derivation of how much of a served model
// window an agent harness may use, and where its auto-compaction must fire. It exists so
// a "safe" fraction is applied exactly once, to the RAW served window, and never to a
// number that was already derived (the 256k -> 128k -> 64k stacking bug).
//
// Two different bounds shape it:
//
//   - a QUALITY cap that only bites on large windows (256k, 1M): long-context quality
//     degrades well before such a window is full, so ContextWindow = min(served, cap);
//   - COMPACTION-RISK bounds for every window: the reserve must hold the model's answer,
//     the compaction summary request must fit the hard cap, and a compaction must reclaim
//     enough room that the next one is not a turn or two away (a compaction storm).
//
// The compaction model follows Pi's semantics: compaction fires when
// contextTokens > contextWindow - reserveTokens; the summary is generated with
// maxTokens = min(0.8*reserveTokens, model.maxTokens); the kept tail is at least
// keepRecentTokens. All numbers are MODELED priors, not a same-task witness.

// WindowSource names where a window number came from. Only a served window may enter
// the derivation; a derived window (a previous envelope's ContextWindow, a halved
// value, a resident target) is refused so the derivation cannot be stacked.
type WindowSource string

const (
	// WindowServed is the raw hard cap of one request slot (input + output) as the
	// backend or router advertises it.
	WindowServed WindowSource = "served"
	// WindowDerived is any number already produced by a budget derivation.
	WindowDerived WindowSource = "derived"
)

// Closed refusal sentinels for DeriveContextEnvelope.
var (
	ErrDerivedWindow  = errors.New("context envelope input is a derived window, not a served window")
	ErrNoServedWindow = errors.New("context envelope input has no positive served window")
)

// Closed reason tokens carried by a non-viable envelope.
const (
	ReasonWindowTooSmallForHarness = "WINDOW_TOO_SMALL_FOR_HARNESS"
	ReasonSummaryExceedsWindow     = "SUMMARY_EXCEEDS_SERVED_WINDOW"
)

// ContextEnvelopeProvenance labels every envelope: a modeled prior, not a witness.
const ContextEnvelopeProvenance = "MODELED"

const (
	// QualityCapTokens is the largest context window a harness is configured to use.
	// MODELED prior: the cap only bites above it, so 256k and 1M windows get 160Ki while
	// a 128k slot is used in full.
	QualityCapTokens = 163840
	// DefaultFixedPromptTokens is the harness's fixed per-request prefix (system prompt
	// plus tool schemas). Provenance: pi first-turn prompt p50 ~14500 tokens, measured
	// 2026-10-09 on Halo Qwen sessions, rounded up to 16Ki.
	DefaultFixedPromptTokens = 16384
	// DefaultKeepRecentTokens is Pi's own keepRecentTokens default.
	DefaultKeepRecentTokens = 20000
	// SummaryPromptOverheadTokens is the summarization instruction wrapped around the
	// history when a compaction summary is requested.
	SummaryPromptOverheadTokens = 2048
	// MinKeepRecentTokens floors the kept tail so a compaction leaves usable context.
	MinKeepRecentTokens = 4096
	// MinOutputTokens and MaxOutputTokens bound the derived per-turn output budget.
	MinOutputTokens = 4096
	MaxOutputTokens = 32768
	// MinMarginTokens floors the tokenizer/estimate skew margin (llama.cpp
	// exceed_context_size errors were observed with estimate-based harnesses).
	MinMarginTokens = 1024
)

// ContextEnvelopeInput is the derivation input. ServedWindow must be the raw served
// window and Source must be WindowServed.
type ContextEnvelopeInput struct {
	ServedWindow int
	Source       WindowSource
	// FixedPromptTokens is the system prompt plus tool schemas; 0 means
	// DefaultFixedPromptTokens.
	FixedPromptTokens int
	// MaxOutputTokens overrides the derived output budget when positive.
	MaxOutputTokens int
	// MaxCompactTrigger, when positive and below the derived trigger, lowers the
	// trigger to it by enlarging the reserve (see ColdAdmission.MaxTrigger).
	MaxCompactTrigger int
}

// ColdAdmission models a deadline-admitting backend that prices every request as
// a cold prompt: a context that grows past what it admits within the client's
// deadline is refused (503 deadline_infeasible) however warm its KV cache is.
type ColdAdmission struct {
	// ClientDeadline is the time budget the harness declares per request.
	ClientDeadline time.Duration
	// Headroom is the fraction of ClientDeadline an estimate may use.
	Headroom float64
	// PrefillTokensPerSec is the backend's cold (uncached) prefill rate.
	PrefillTokensPerSec float64
	// DecodeTokens and DecodeTokensPerSec price the completion term.
	DecodeTokens       int
	DecodeTokensPerSec float64
	// TurnGrowthTokens is one turn's context growth past the trigger before the
	// harness compacts.
	TurnGrowthTokens int
}

// MaxTrigger returns the largest compaction trigger whose next request the
// backend admits cold, rounded down to 1Ki; 0 when the model is incomplete.
//
//	trigger = (Headroom*ClientDeadline - DecodeTokens/DecodeRate) * PrefillRate - TurnGrowth
func (c ColdAdmission) MaxTrigger() int {
	if c.ClientDeadline <= 0 || c.Headroom <= 0 || c.PrefillTokensPerSec <= 0 || c.DecodeTokensPerSec <= 0 {
		return 0
	}
	secs := c.Headroom*c.ClientDeadline.Seconds() - float64(c.DecodeTokens)/c.DecodeTokensPerSec
	tokens := int(secs*c.PrefillTokensPerSec) - c.TurnGrowthTokens
	tokens -= tokens % 1024
	return max(tokens, 1)
}

// ContextEnvelope is the derived harness configuration for one served window.
type ContextEnvelope struct {
	ServedWindow      int    `json:"served_window"`
	ContextWindow     int    `json:"context_window"`
	QualityCapped     bool   `json:"quality_capped"`
	OutputTokens      int    `json:"output_tokens"`
	ReserveTokens     int    `json:"reserve_tokens"`
	CompactTrigger    int    `json:"compact_trigger"`
	DeadlineCapped    bool   `json:"deadline_capped,omitempty"`
	KeepRecentTokens  int    `json:"keep_recent_tokens"`
	SummaryTokens     int    `json:"summary_tokens"`
	PostCompactTokens int    `json:"post_compact_tokens"`
	ReclaimTokens     int    `json:"reclaim_tokens"`
	Viable            bool   `json:"viable"`
	Reason            string `json:"reason,omitempty"`
	Provenance        string `json:"provenance"`
}

// DeriveContextEnvelope derives the harness context envelope from a served window.
//
//	ContextWindow = min(served, QualityCapTokens)
//	Output        = MaxOutputTokens override, else clamp(ContextWindow/8, 4096, 32768)
//	Margin        = max(1024, ContextWindow/32)
//	Reserve       = Output + SummaryPromptOverhead + Margin
//	Summary       = min(0.8*Reserve, Output)            (Pi's summary maxTokens)
//	Trigger       = ContextWindow - Reserve, lowered to MaxCompactTrigger when set
//	                (Reserve then grows to ContextWindow - Trigger)
//	Keep          = max(4096, min(20000, Trigger/2 - Fixed - Summary))
//	PostCompact   = Fixed + Summary + Keep;  Reclaim = Trigger - PostCompact
//
// The envelope is viable when the summary request fits the served window and a
// compaction reclaims at least a quarter of the trigger; otherwise Viable is false and
// Reason carries a closed token. The derivation is a fixed point: feeding ContextWindow
// back as a served window yields the same ContextWindow.
func DeriveContextEnvelope(in ContextEnvelopeInput) (ContextEnvelope, error) {
	if in.Source != WindowServed {
		return ContextEnvelope{}, fmt.Errorf("%w: source %q", ErrDerivedWindow, in.Source)
	}
	if in.ServedWindow <= 0 {
		return ContextEnvelope{}, ErrNoServedWindow
	}
	fixed := in.FixedPromptTokens
	if fixed <= 0 {
		fixed = DefaultFixedPromptTokens
	}

	window := min(in.ServedWindow, QualityCapTokens)
	output := in.MaxOutputTokens
	if output <= 0 {
		output = min(max(window/8, MinOutputTokens), MaxOutputTokens)
	}
	margin := max(MinMarginTokens, window/32)
	// Pi's summary is min(0.8*reserve, maxTokens); with maxTokens = output the summary
	// request needs output + its prompt overhead, which dominates output alone.
	reserve := output + SummaryPromptOverheadTokens + margin
	trigger := window - reserve
	capped := in.MaxCompactTrigger > 0 && in.MaxCompactTrigger < trigger
	if capped {
		trigger = in.MaxCompactTrigger
		reserve = window - trigger
	}
	summary := min(reserve*4/5, output)
	keep := max(MinKeepRecentTokens, min(DefaultKeepRecentTokens, trigger/2-fixed-summary))
	post := fixed + summary + keep
	reclaim := trigger - post

	env := ContextEnvelope{
		ServedWindow:      in.ServedWindow,
		ContextWindow:     window,
		QualityCapped:     window < in.ServedWindow,
		OutputTokens:      output,
		ReserveTokens:     reserve,
		CompactTrigger:    trigger,
		DeadlineCapped:    capped,
		KeepRecentTokens:  keep,
		SummaryTokens:     summary,
		PostCompactTokens: post,
		ReclaimTokens:     reclaim,
		Viable:            true,
		Provenance:        ContextEnvelopeProvenance,
	}
	switch {
	case trigger+SummaryPromptOverheadTokens+summary > in.ServedWindow:
		env.Viable, env.Reason = false, ReasonSummaryExceedsWindow
	case trigger <= 0 || reclaim < trigger/4:
		env.Viable, env.Reason = false, ReasonWindowTooSmallForHarness
	}
	return env, nil
}
