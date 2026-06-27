package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// resume_projection.go — the LIVE, SHADOW-mode PROJECTED-vs-OBSERVED resume-cache RESIDUAL for
// issue #941 (the live rung of epic #745). resume.Plan and `fak resume validate` are OFFLINE: a
// deterministic projection over historical facts (c449055 + 860a5e0), never a plan auto-fired on a
// live resume and never differenced against the bill the provider actually charged. This file is
// the gateway half of the live wiring: given the live facts of a resume boundary (resident tokens
// from the resumed transcript, idle from the wall clock, the model's real pricing) and the
// provider's FIRST-POST-RESUME usage, it computes resume.Plan, prices the projection and the
// observed bill on the SAME prompt-cache axis, emits BOTH side by side plus the residual, and
// exports the residual as a metric — the exact OBSERVED-vs-WITNESSED discipline
// gateway/cache_pricing.go and reset_shadow.go keep.
//
// SHADOW / observe-only. Nothing here resumes, cuts, or resets a session, and nothing alters the
// turn: observeResumeProjection only computes the residual, records the metric, and logs a
// content-free audit row. It is INERT until the host's opt-in resume hook calls it, mirroring the
// default-off posture internal/sessionreset and reset_shadow.go ship under. The host (cmd/fak) owns
// the boundary detection — the gateway is clock-free and never sees idle time (reset_shadow.go: "no
// provider exposes a TTL/idle hint today") — so it supplies the resume.Input; this leaf supplies the
// wire-neutral residual + metric the host folds into its audit, the same injected-hook split the
// gateway keeps with internal/session. It is a self-contained metric family (its accumulator lives
// on Server.resumeProj with its own mutex), so it does not touch the shared metrics struct.
//
// PROVENANCE (the issue's "witnessed bill" reconciled to the codebase's vocabulary): the PROJECTION
// (posture / recommendation / projected prompt cost) is WITNESSED in the cache_pricing.go sense —
// fak authored it from its own calibrated resume.Plan model over inputs it controls. The first-turn
// bill it is differenced against (cache_read / cache_creation) is OBSERVED — relayed verbatim from
// the upstream, never a fak claim. The residual is the live delta between the two; ~0 means the
// calibrated projection tracked the live bill.

// ResumeProjectionResidual is one resume boundary's projected-vs-observed comparison — the
// structured, content-free audit row the host folds into its resume audit (#941 ask 1+2). It
// carries fak's resume.Plan verdict (posture, recommendation) AND the side-by-side prompt-cache
// cost the projection MODELED vs the bill the provider actually CHARGED on the first post-resume
// turn, plus the residual between them. Every dollar is priced over a token count at the model's
// base input price; none is a fak-authored claim about money (the projected side is a model, the
// observed side is the provider's relayed bill).
type ResumeProjectionResidual struct {
	// Posture / Recommended / Reason / ResidentTokens are fak's resume.Plan verdict over the live
	// facts — the plan recommendation the audit row carries (ask 1).
	Posture        resume.Posture  `json:"posture"`
	Recommended    resume.Strategy `json:"recommended"`
	Reason         string          `json:"reason"`
	ResidentTokens int             `json:"resident_tokens"`

	// PROJECTED — fak's own resume.Plan over the live facts (fak-authored / WITNESSED). The cost is
	// the prompt-cache cost of the first post-resume turn the projection MODELS under the projected
	// posture (a cold re-prefill of the resident at the calibrated cold-write multiplier, or a warm
	// read), priced WITHOUT output so it differences exactly against the observed prompt bill.
	ProjectedColdWriteShare float64 `json:"projected_cold_write_share"`
	ProjectedPromptCostUSD  float64 `json:"projected_prompt_cost_usd"`

	// OBSERVED — the provider's first-post-resume bill, relayed verbatim (the issue's "witnessed
	// bill"). The cost is the same prompt-cache axis (uncached input at 1×, cache_read at 0.1×,
	// cache_creation at the write premium), output excluded.
	ObservedInputTokens         int     `json:"observed_input_tokens"`
	ObservedCacheReadTokens     int     `json:"observed_cache_read_tokens"`
	ObservedCacheCreationTokens int     `json:"observed_cache_creation_tokens"`
	ObservedColdWriteShare      float64 `json:"observed_cold_write_share"`
	ObservedPromptCostUSD       float64 `json:"observed_prompt_cost_usd"`

	// RESIDUAL = observed − projected (the live per-resume delta ask 3 exports as a metric).
	// PromptCostDeltaUSD ~0 means the projection's cost call tracked the live bill; positive means
	// the live first turn cost MORE than projected. ColdWriteShareResidual is the live cold-write
	// share minus the calibrated ColdWriteShare (#955) — the live validation of that constant.
	PromptCostDeltaUSD     float64 `json:"prompt_cost_delta_usd"`
	ColdWriteShareResidual float64 `json:"cold_write_share_residual"`
}

// computeResumeProjectionResidual is THE pure residual: same facts in, same row out — no clock, no
// I/O. It runs resume.Plan over the live `in`, prices the projected prompt-cache cost of the first
// post-resume turn under the projected posture, prices the OBSERVED first-turn bill on the same
// axis, and differences them. The projected cost is posture-consistent: a COLD/UNKNOWN posture pays
// the calibrated cold re-prefill (resume_full's ColdReprefillUSD), a WARM posture pays a cache read
// — so the residual is the gap between fak's posture-and-cost call and what the provider charged.
func computeResumeProjectionResidual(in resume.Input, observed CacheUsage) ResumeProjectionResidual {
	rep := resume.Plan(in)

	// The projected prompt-cache cost of the first post-resume turn, output excluded. resume_full is
	// always Strategies[0]; its ColdReprefillUSD is the cold re-prefill of the whole resident at the
	// calibrated cold-write multiplier. On a projected-WARM prefix the modeled first turn is a cache
	// read instead, so price that directly (the same 0.1× read multiplier cache_pricing.go uses).
	projectedPrompt := rep.Strategies[0].ColdReprefillUSD
	if rep.Posture == resume.PostureWarm {
		projectedPrompt = float64(rep.ResidentTokens) * perToken(in.Pricing.InputPerMTokUSD) * CacheReadMultiplier
	}

	// Price the OBSERVED prompt-cache bill: uncached input + cache_read + cache_creation, output
	// excluded so it differences apples-to-apples with the projection. WriteTTL defaults to 5m.
	pricing := CachePricing{InputPerMTokUSD: in.Pricing.InputPerMTokUSD, OutputPerMTokUSD: in.Pricing.OutputPerMTokUSD}
	observedPrompt := CacheUsage{
		InputTokens:         observed.InputTokens,
		CacheReadTokens:     observed.CacheReadTokens,
		CacheCreationTokens: observed.CacheCreationTokens,
		OutputTokens:        0,
		WriteTTL:            observed.WriteTTL,
	}
	observedCost := pricing.CostUSD(observedPrompt)

	// The OBSERVED cold-write share is the live counterpart of the calibrated ColdWriteShare:
	// cache_creation / (input + cache_read + cache_creation). Zero-safe.
	observedTotal := observed.InputTokens + observed.CacheReadTokens + observed.CacheCreationTokens
	observedShare := 0.0
	if observedTotal > 0 {
		observedShare = float64(observed.CacheCreationTokens) / float64(observedTotal)
	}

	return ResumeProjectionResidual{
		Posture:                     rep.Posture,
		Recommended:                 rep.Recommended,
		Reason:                      rep.Reason,
		ResidentTokens:              rep.ResidentTokens,
		ProjectedColdWriteShare:     resume.ColdWriteShare,
		ProjectedPromptCostUSD:      projectedPrompt,
		ObservedInputTokens:         observed.InputTokens,
		ObservedCacheReadTokens:     observed.CacheReadTokens,
		ObservedCacheCreationTokens: observed.CacheCreationTokens,
		ObservedColdWriteShare:      observedShare,
		ObservedPromptCostUSD:       observedCost,
		PromptCostDeltaUSD:          observedCost - projectedPrompt,
		ColdWriteShareResidual:      observedShare - resume.ColdWriteShare,
	}
}

// observeResumeProjection is the host's resume hook: at a live resume boundary, the host (cmd/fak,
// which owns the wall clock and the resumed transcript) supplies the resume.Input and the provider's
// first-post-resume usage; this computes the residual, records the SHADOW metric, logs the
// content-free audit row, and returns the residual so the host can fold it into its own audit. It
// acts on NOTHING — the live turn is byte-identical whatever the residual says. A nil server is a
// safe no-op returning the zero residual.
func (s *Server) observeResumeProjection(trace string, in resume.Input, observed CacheUsage) ResumeProjectionResidual {
	if s == nil {
		return ResumeProjectionResidual{}
	}
	r := computeResumeProjectionResidual(in, observed)
	s.resumeProj.record(r)
	s.logResumeProjection(trace, r)
	return r
}

// logResumeProjection writes the resume residual as a structured, content-free audit row (gated on
// the --log sink). It carries the plan verdict and the projected-vs-observed prompt-cache cost and
// residual — only posture/strategy/reason tokens and token/dollar counts, never a prompt byte — so
// an operator (or a dashboard) sees how the calibrated projection tracked the live bill without the
// gateway leaking any conversation content. The observed cost is labeled the provider's bill (the
// issue's "witnessed bill"); the projected cost is labeled fak's model, keeping the two provenances
// distinct on the wire.
func (s *Server) logResumeProjection(trace string, r ResumeProjectionResidual) {
	if s == nil || s.logf == nil {
		return
	}
	b, err := json.Marshal(map[string]any{
		"event":                      "gateway_resume_projection",
		"trace_id":                   trace,
		"posture":                    string(r.Posture),
		"recommended":                string(r.Recommended), // the plan recommendation (ask 1)
		"reason":                     r.Reason,
		"resident_tokens":            r.ResidentTokens,
		"projected_prompt_cost_usd":  r.ProjectedPromptCostUSD, // fak's calibrated model
		"observed_prompt_cost_usd":   r.ObservedPromptCostUSD,  // the provider's relayed bill
		"prompt_cost_delta_usd":      r.PromptCostDeltaUSD,     // observed − projected (ask 2/3)
		"projected_cold_write_share": r.ProjectedColdWriteShare,
		"observed_cold_write_share":  r.ObservedColdWriteShare,
		"cold_write_share_residual":  r.ColdWriteShareResidual,
	})
	if err != nil {
		return
	}
	s.logf("%s", b)
}

// resumeProjMetrics is the SHADOW / observe-only accumulator for the resume residual family (#941),
// a self-contained metric surface stored on Server (gateway.go). Each resume boundary is a one-shot
// event the host's opt-in hook fires; nothing acts on it. The posture/strategy/projection are
// WITNESSED (fak's own resume.Plan); the first-turn cache bill they are differenced against is
// OBSERVED (provider-relayed). The maps are minted lazily, so a zero-value Server records correctly.
type resumeProjMetrics struct {
	mu             sync.Mutex
	postures       map[string]uint64 // resume.Posture -> boundaries projected that way
	strategies     map[string]uint64 // resume.Strategy -> boundaries recommended that way
	boundaries     uint64            // total resume boundaries observed (host hook fires)
	lastCostDelta  float64           // most recent boundary's OBSERVED−PROJECTED prompt-cache cost (USD)
	lastShareResid float64           // most recent boundary's OBSERVED−calibrated cold-write-share residual
}

// record folds one resume boundary's residual into the observe-only accumulators. It only COUNTS
// what the projection said, bucketed by the closed resume.Posture and resume.Strategy, and retains
// the latest cost/share residual so a dashboard can track whether the calibrated projection stays
// accurate on live traffic. It acts on nothing (SHADOW).
func (m *resumeProjMetrics) record(r ResumeProjectionResidual) {
	if m == nil {
		return
	}
	m.mu.Lock()
	if m.postures == nil {
		m.postures = map[string]uint64{}
	}
	if m.strategies == nil {
		m.strategies = map[string]uint64{}
	}
	m.postures[string(r.Posture)]++
	m.strategies[string(r.Recommended)]++
	m.boundaries++
	m.lastCostDelta = r.PromptCostDeltaUSD
	m.lastShareResid = r.ColdWriteShareResidual
	m.mu.Unlock()
}

// resumeProjSnapshot is a lock-free copy of the resume residual accumulators for rendering.
type resumeProjSnapshot struct {
	postures       map[string]uint64
	strategies     map[string]uint64
	boundaries     uint64
	lastCostDelta  float64
	lastShareResid float64
}

func (m *resumeProjMetrics) snapshot() resumeProjSnapshot {
	out := resumeProjSnapshot{postures: map[string]uint64{}, strategies: map[string]uint64{}}
	if m == nil {
		return out
	}
	m.mu.Lock()
	for k, v := range m.postures {
		out.postures[k] = v
	}
	for k, v := range m.strategies {
		out.strategies[k] = v
	}
	out.boundaries = m.boundaries
	out.lastCostDelta = m.lastCostDelta
	out.lastShareResid = m.lastShareResid
	m.mu.Unlock()
	return out
}

// writeMetrics renders the resume PROJECTED-vs-OBSERVED RESIDUAL family (#941). It is SHADOW /
// observe-only: nothing acts on these numbers — they exist so a dashboard can track whether the
// calibrated resume.Plan stays accurate against the bill the provider actually charged. Every
// posture and strategy bucket is emitted at 0 so the panel exists before the first resume boundary.
// The projection is WITNESSED (fak's resume.Plan); the bill it is differenced against is OBSERVED
// (provider-relayed), the same split the compaction/reset-shadow families keep.
func (m *resumeProjMetrics) writeMetrics(b *strings.Builder) {
	snap := m.snapshot()

	writeCounter(b, "fak_gateway_resume_projection_boundaries_total",
		"WITNESSED (fak hook, observe-only): live resume boundaries the host fired the resume residual hook on. Default-off: 0 until the opt-in hook is enabled. Nothing acts on a resume here.", int64(snap.boundaries))

	writeHelpType(b, "fak_gateway_resume_projection_posture_total",
		"WITNESSED (fak resume.Plan, observe-only): live resume boundaries by PROJECTED cache posture (cold|warm|unknown) — the cold/warm call resume.Plan made from idle-vs-TTL. Emitted at 0 so the panel exists pre-first-boundary.", "counter")
	for _, p := range []string{string(resume.PostureCold), string(resume.PostureWarm), string(resume.PostureUnknown)} {
		fmt.Fprintf(b, "fak_gateway_resume_projection_posture_total{posture=%q} %d\n", p, snap.postures[p])
	}

	writeHelpType(b, "fak_gateway_resume_projection_recommendation_total",
		"WITNESSED (fak resume.Plan, observe-only): live resume boundaries by RECOMMENDED re-entry strategy (resume_full|cut|reset). SHADOW: the recommendation is surfaced, never auto-applied — the host owns whether to act. Emitted at 0 so the panel exists pre-first-boundary.", "counter")
	for _, st := range []string{string(resume.StrategyResumeFull), string(resume.StrategyCut), string(resume.StrategyReset)} {
		fmt.Fprintf(b, "fak_gateway_resume_projection_recommendation_total{strategy=%q} %d\n", st, snap.strategies[st])
	}

	writeHelpType(b, "fak_gateway_resume_projection_cost_delta_usd",
		"The most recent resume boundary's prompt-cache cost RESIDUAL: the provider's first-post-resume bill (OBSERVED, provider-relayed cache_read/cache_creation priced at this model's base input) MINUS fak's PROJECTED cold re-prefill cost (resume.Plan, fak-authored). ~0 means the calibrated projection tracked the live bill; positive means the live resume cost MORE than projected. Output excluded so this is purely the prompt-cache effect.", "gauge")
	fmt.Fprintf(b, "fak_gateway_resume_projection_cost_delta_usd %s\n", promFloat(snap.lastCostDelta))

	writeHelpType(b, "fak_gateway_resume_projection_cold_write_share_residual",
		"The most recent resume boundary's cold-write-share RESIDUAL: the OBSERVED live cold-write share (cache_creation / prompt, provider-relayed) MINUS fak's calibrated ColdWriteShare (#955). ~0 confirms the calibrated constant holds on live traffic; a persistent drift is the signal to refit it.", "gauge")
	fmt.Fprintf(b, "fak_gateway_resume_projection_cold_write_share_residual %s\n", promFloat(snap.lastShareResid))
}
