package ctxplan

import (
	"path/filepath"
	"strings"
)

// envelope.go — the measured (or fallback-prior) effective-context envelope that the default
// resident-budget path selects from. It is the code seam behind the long-context defaults
// doctrine (docs/long-context-defaults.md): the provider's advertised context window is a hard
// CAP, never the target resident budget. A turn's default budget is chosen from the minimum
// viable evidence floor (MVC), the measured effective ceiling (MECW), and a reserved output
// slice — not from the raw model max. Each row carries a provenance label so a fallback prior is
// never headlined as if it were witnessed.
//
// This is deliberately a small, conservative table: it encodes the two shipped default paths
// (the ctxplan planned view and history compaction) as first-class envelopes so the token-
// defaults scorecard can report their hard cap / evidence floor / target / effective ceiling /
// provenance and raise debt if any default is derived only from a raw window with no witness.

// Provenance labels the evidence class behind an envelope's budget. Ordered weakest→strongest:
// FALLBACK (a deliberately conservative prior, no evidence) < MODELED (arithmetic/inspection) <
// OBSERVED (external/telemetry, not the exact fak route) < WITNESSED (a local same-task run).
const (
	ProvenanceFallback  = "FALLBACK"
	ProvenanceModeled   = "MODELED"
	ProvenanceObserved  = "OBSERVED"
	ProvenanceWitnessed = "WITNESSED"
)

// EffectiveContextEnvelope is the default context budget for one model + task class. It
// separates the four quantities the doctrine keeps distinct so the default path never conflates
// the provider cap with the resident target:
//
//   - HardContextCap: the provider limit on request+output. A cap, never a goal.
//   - OutputReserve: tokens held back for the answer/tool args/thinking. Subtracted first.
//   - MinViableEvidenceTokens: the smallest resident evidence set that still meets the task SLO.
//   - TargetResidentTokens: the tokens fak tries to materialize this turn — the real default.
//   - MaxEffectiveTokens: the measured (or prior) effective ceiling; min(cap-reserve, this).
//
// Provenance/Witness record how strong the evidence for the row is. A row whose target is the
// raw hard cap without a WITNESSED provenance is a raw-window-target debt (see RawWindowTarget).
type EffectiveContextEnvelope struct {
	ModelPattern            string
	TaskClass               string
	HardContextCap          int
	OutputReserve           int
	MinViableEvidenceTokens int
	TargetResidentTokens    int
	MaxEffectiveTokens      int
	Provenance              string
	Witness                 string
}

// SafeCap is the largest resident view the envelope permits: min(HardContextCap-OutputReserve,
// MaxEffectiveTokens). Output reserve is subtracted before the effective ceiling is applied, so a
// request that would consume the whole window as input can never be a default.
func (e EffectiveContextEnvelope) SafeCap() int {
	capMinusReserve := e.HardContextCap - e.OutputReserve
	if capMinusReserve < 0 {
		capMinusReserve = 0
	}
	if e.MaxEffectiveTokens > 0 && e.MaxEffectiveTokens < capMinusReserve {
		return e.MaxEffectiveTokens
	}
	return capMinusReserve
}

// Target is the resident budget the default path should use: the task's target clamped into
// [MinViableEvidenceTokens, SafeCap]. It never returns the raw hard cap; the target is bounded by
// the effective ceiling and floored at the minimum viable evidence set.
func (e EffectiveContextEnvelope) Target() int {
	lo, hi := e.MinViableEvidenceTokens, e.SafeCap()
	if hi < lo {
		hi = lo
	}
	t := e.TargetResidentTokens
	if t < lo {
		t = lo
	}
	if t > hi {
		t = hi
	}
	return t
}

// RawWindowTarget is the doctrine violation the scorecard raises as debt: a default that treats a
// provider's advertised hard cap as the target resident budget without a same-task WITNESS. It is
// true when the target reaches (or exceeds) the effective ceiling — i.e. the row is not being held
// below the raw window by a measured envelope — and the provenance is weaker than WITNESSED, or the
// row carries no provenance label at all.
func (e EffectiveContextEnvelope) RawWindowTarget() bool {
	if e.Provenance == ProvenanceWitnessed {
		return false
	}
	if e.Provenance == "" {
		return true
	}
	ceiling := e.MaxEffectiveTokens
	if ceiling <= 0 || e.HardContextCap-e.OutputReserve < ceiling {
		ceiling = e.HardContextCap - e.OutputReserve
	}
	return e.TargetResidentTokens >= ceiling
}

// EnvelopeForModel returns the routine-agent-turn envelope for model. Specific
// ModelPattern rows are checked before the wildcard fallback.
func EnvelopeForModel(model string) EffectiveContextEnvelope {
	model = strings.ToLower(strings.TrimSpace(model))
	var fallback EffectiveContextEnvelope
	for _, e := range DefaultEnvelopes() {
		if e.TaskClass != "routine-agent-turn" {
			continue
		}
		if e.ModelPattern == "*" {
			fallback = e
			continue
		}
		if ok, _ := filepath.Match(strings.ToLower(e.ModelPattern), model); ok {
			return e
		}
	}
	return fallback
}

// DefaultEnvelopes is the conservative fallback-prior table the no-flag default path selects
// from. Both rows are labeled MODELED (a code/doctrine-derived prior on the shipped default), not
// WITNESSED: the targets are the currently shipped constants (the ctxplan planned view's 8K
// resident budget and history compaction's 48K budget), held far below each provider's advertised
// window and below the modeled effective ceiling. They upgrade to WITNESSED only when a same-task
// fak bench measures the exact route (docs/long-context-defaults.md).
func DefaultEnvelopes() []EffectiveContextEnvelope {
	return []EffectiveContextEnvelope{
		{ModelPattern: "*fable*", TaskClass: "routine-agent-turn", HardContextCap: 64000, OutputReserve: 32000, MinViableEvidenceTokens: 2000, TargetResidentTokens: 8000, MaxEffectiveTokens: 32000, Provenance: ProvenanceModeled, Witness: "#3611 worker-model envelope"},
		{ModelPattern: "*haiku*", TaskClass: "routine-agent-turn", HardContextCap: 96000, OutputReserve: 32000, MinViableEvidenceTokens: 2000, TargetResidentTokens: 8000, MaxEffectiveTokens: 48000, Provenance: ProvenanceModeled, Witness: "#3611 worker-model envelope"},
		{ModelPattern: "*codex*", TaskClass: "routine-agent-turn", HardContextCap: 272000, OutputReserve: 32000, MinViableEvidenceTokens: 2000, TargetResidentTokens: 8000, MaxEffectiveTokens: 96000, Provenance: ProvenanceModeled, Witness: "#3611 worker-model envelope"},
		{
			ModelPattern:            "*",
			TaskClass:               "routine-agent-turn",
			HardContextCap:          200000,
			OutputReserve:           32000,
			MinViableEvidenceTokens: 2000,
			TargetResidentTokens:    8000,
			MaxEffectiveTokens:      32000,
			Provenance:              ProvenanceModeled,
			Witness:                 "docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md; docs/long-context-defaults.md fallback prior (8K-32K routine)",
		},
		{
			ModelPattern:            "*",
			TaskClass:               "long-doc-history",
			HardContextCap:          200000,
			OutputReserve:           32000,
			MinViableEvidenceTokens: 8000,
			TargetResidentTokens:    48000,
			MaxEffectiveTokens:      128000,
			Provenance:              ProvenanceModeled,
			Witness:                 "docs/long-context-defaults.md fallback prior (32K-128K cross-file/long-doc)",
		},
	}
}

// GenericTurnEnvelope is the envelope the unconstrained default budget path seeds from — the
// routine-agent-turn row. DefaultBudgetBounds derives its resident ceiling from this envelope's
// Target so the seed spectrum tracks the doctrine's effective envelope rather than a bare literal.
func GenericTurnEnvelope() EffectiveContextEnvelope {
	env := EnvelopeForModel("")
	// The seed ceiling matches the historical 8192-token generous resident view; the envelope's
	// 8000 target is the shipped ctxview budget. Keep the seed at the doctrine-consistent 8192 so
	// an unconstrained caller's spectrum is unchanged while it is now sourced from the envelope.
	env.TargetResidentTokens = 8192
	return env
}
