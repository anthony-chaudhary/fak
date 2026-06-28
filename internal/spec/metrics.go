package spec

// metrics.go — acceptance-rate metrics for speculative decode (issue #284, the
// acceptance criterion "Metrics for acceptance rate" / scope "Acceptance rate
// tracking"). The decode round-driver SpeculativeGreedy already returns the per-run
// TOTALS (drafted, accepted, rolled back), but the acceptance criterion asks for the
// two DERIVED quantities a serving lane reports and an operator watches:
//
//   - the per-token ACCEPTANCE RATE (accepted/drafted) — the headline spec-decode
//     metric and the same `a` that feeds the modeled polymodel.EffectiveTokensPerVerify
//     speedup curve; and
//   - the EFFECTIVE real tokens advanced per target verify pass (advanced/rounds) —
//     the realized, measured-on-THIS-run analogue of that modeled quantity, i.e. the
//     decode-speedup proxy (a non-speculative decode advances exactly 1 real token per
//     bandwidth-bound pass, so an effective E>1 is the speculative win).
//
// AcceptanceMeter accumulates the per-round counts into both. It is a pure accumulator
// — it owns no model, holds no KV, and does no I/O — so it is cheap to embed on a decode
// lane and to Snapshot for a metrics export or a JSON artifact (the shape
// experiments/spec-decode/*.json already reports per run). It is fed from ONE serial
// decode lane (polymodel.Schedule keeps decode strictly single-lane), so it needs no
// locking; do not share one meter across concurrent decode lanes.
//
// Honest boundary: the meter reports the acceptance rate and the effective-tokens-per-
// verify speedup PROXY on whatever model actually ran (a CPU synthetic in the witness,
// a real draft/target pair on a backend). The absolute wall-clock "2×+ decode speedup"
// acceptance item is a measured tokens/sec number that still needs the GPU bench harness
// — this meter is the metrics substrate that number is read off, never a substitute for it.

// AcceptanceMeter tracks speculative-decode acceptance across rounds. Feed it one Observe
// per speculative round (the unit SpeculativeGreedy iterates); read the derived metrics
// with AcceptanceRate / EffectiveTokensPerVerify, or take a whole AcceptanceStats with
// Snapshot. The zero value is an empty, ready-to-use meter.
type AcceptanceMeter struct {
	rounds   int
	drafted  int
	accepted int
	advanced int
}

// Observe records one speculative round: drafted = tokens the drafter proposed (and the
// target verified) this round, accepted = the leading drafts the target confirmed
// (0..drafted), advanced = REAL tokens committed this round (accepted drafts + any
// correction/bonus, >= 1 on a progressing lane). Negative inputs clamp to 0 and accepted
// is capped at drafted, so a miscounting caller cannot drive the rate above 1.
func (m *AcceptanceMeter) Observe(drafted, accepted, advanced int) {
	if drafted < 0 {
		drafted = 0
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > drafted {
		accepted = drafted
	}
	if advanced < 0 {
		advanced = 0
	}
	m.rounds++
	m.drafted += drafted
	m.accepted += accepted
	m.advanced += advanced
}

// Rounds reports the number of observed speculative rounds (== target verify passes).
func (m *AcceptanceMeter) Rounds() int { return m.rounds }

// Drafted reports the total tokens proposed (and verified) across all rounds.
func (m *AcceptanceMeter) Drafted() int { return m.drafted }

// Accepted reports the total accepted draft tokens across all rounds.
func (m *AcceptanceMeter) Accepted() int { return m.accepted }

// Advanced reports the total REAL tokens committed across all rounds.
func (m *AcceptanceMeter) Advanced() int { return m.advanced }

// AcceptanceRate is the per-token acceptance rate accepted/drafted in [0,1] — the
// headline spec-decode metric. It is 0 when nothing was drafted (no division by zero).
func (m *AcceptanceMeter) AcceptanceRate() float64 {
	if m.drafted == 0 {
		return 0
	}
	return float64(m.accepted) / float64(m.drafted)
}

// EffectiveTokensPerVerify is the REAL tokens advanced per speculative round
// (advanced/rounds) — the realized, measured-on-this-run analogue of the modeled
// polymodel.EffectiveTokensPerVerify, and the decode-speedup proxy. It is 0 when no
// round was observed.
func (m *AcceptanceMeter) EffectiveTokensPerVerify() float64 {
	if m.rounds == 0 {
		return 0
	}
	return float64(m.advanced) / float64(m.rounds)
}

// AcceptanceStats is a flat snapshot of an AcceptanceMeter for a metrics export or a
// JSON artifact. The two derived fields are computed from the counts (never stored), so
// a Snapshot can never disagree with the raw totals it carries.
type AcceptanceStats struct {
	Rounds                   int     `json:"rounds"`
	Drafted                  int     `json:"drafted_tokens"`
	Accepted                 int     `json:"accepted_tokens"`
	Advanced                 int     `json:"advanced_tokens"`
	AcceptanceRate           float64 `json:"acceptance_rate"`
	EffectiveTokensPerVerify float64 `json:"effective_tokens_per_verify"`
}

// Snapshot returns the current metrics as a flat AcceptanceStats.
func (m *AcceptanceMeter) Snapshot() AcceptanceStats {
	return AcceptanceStats{
		Rounds:                   m.rounds,
		Drafted:                  m.drafted,
		Accepted:                 m.accepted,
		Advanced:                 m.advanced,
		AcceptanceRate:           m.AcceptanceRate(),
		EffectiveTokensPerVerify: m.EffectiveTokensPerVerify(),
	}
}
