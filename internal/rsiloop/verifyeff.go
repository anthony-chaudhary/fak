package rsiloop

// verifyeff.go wires a generic, deterministic, wall-clock-free verification
// effectiveness harness into the RSI engine. It scores a VerifyEffSnapshot
// across seven orthogonal axes (attribution, introducer linkage, gate
// precision, catcher retention, queue classification, judge separation, and
// verifier exposure penalty) into a 0..100 Verification Effectiveness Index
// (VEI) and structured Scorecard, then drives candidate proposals through
// rsiloop.Run and shipgate.Evaluate.

import (
	"fmt"
	"math"
)

// VerifyEffMetricName labels the verification effectiveness index in the RSI journal.
const VerifyEffMetricName = "verify_effectiveness_index"

// VerifyEffSnapshot captures normalized [0,1] verification effectiveness signals.
// All fields are deterministic and wall-clock-free.
type VerifyEffSnapshot struct {
	EscapeAttributionRate   float64 `json:"escape_attribution_rate"`
	IntroducerLinkRate      float64 `json:"introducer_link_rate"`
	GatePrecision           float64 `json:"gate_precision"`
	CatcherRetentionRate    float64 `json:"catcher_retention_rate"`
	QueueClassifiedRate     float64 `json:"queue_classified_rate"`
	JudgeSeparationRate     float64 `json:"judge_separation_rate"`
	VerifierExposurePenalty float64 `json:"verifier_exposure_penalty"`
}

// VerifyEffProposal is one proposed verification effectiveness state to score as
// an RSI candidate.
type VerifyEffProposal struct {
	Label    string
	Snapshot VerifyEffSnapshot
}

// ScoreVerifyEff computes a deterministic 0..100 Verification Effectiveness Index
// (VEI) and a structured multi-axis Scorecard (Name="verify_effectiveness") from
// a VerifyEffSnapshot. Inputs are clamped to [0,1] (NaN/Inf treated as 0).
func ScoreVerifyEff(s VerifyEffSnapshot) (vei float64, score Scorecard) {
	attr := clampVerifyEffUnit(s.EscapeAttributionRate)
	link := clampVerifyEffUnit(s.IntroducerLinkRate)
	prec := clampVerifyEffUnit(s.GatePrecision)
	ret := clampVerifyEffUnit(s.CatcherRetentionRate)
	qcls := clampVerifyEffUnit(s.QueueClassifiedRate)
	sep := clampVerifyEffUnit(s.JudgeSeparationRate)
	pen := clampVerifyEffUnit(s.VerifierExposurePenalty)

	weighted := 0.16*attr +
		0.14*link +
		0.16*prec +
		0.14*ret +
		0.15*qcls +
		0.15*sep +
		0.10*(1.0-pen)

	vei = math.Round(clampVerifyEffUnit(weighted)*10000.0) / 100.0

	score = Scorecard{
		Name:  "verify_effectiveness",
		Value: vei,
		Grade: verifyEffGrade(vei),
		Components: []ScoreComponent{
			{Name: "escape_attribution_rate", Value: attr, Unit: "ratio"},
			{Name: "introducer_link_rate", Value: link, Unit: "ratio"},
			{Name: "gate_precision", Value: prec, Unit: "ratio"},
			{Name: "catcher_retention_rate", Value: ret, Unit: "ratio"},
			{Name: "queue_classified_rate", Value: qcls, Unit: "ratio"},
			{Name: "judge_separation_rate", Value: sep, Unit: "ratio"},
			{Name: "verifier_exposure_penalty", Value: pen, Unit: "ratio"},
		},
	}
	return vei, score
}

// NewVerifyEffHarness returns a deterministic RSI Harness that evaluates a
// representative low-visibility baseline and three verification proposals
// through rsiloop.Run / shipgate.Evaluate:
//   - Candidate 1: no-op proposal -> REVERT
//   - Candidate 2: attribution + queue-classification uplift -> KEEP
//   - Candidate 3: gate signal-to-friction + separation uplift -> KEEP
func NewVerifyEffHarness() Harness {
	baseline := VerifyEffSnapshot{
		EscapeAttributionRate:   0.25,
		IntroducerLinkRate:      0.20,
		GatePrecision:           0.55,
		CatcherRetentionRate:    0.50,
		QueueClassifiedRate:     0.30,
		JudgeSeparationRate:     0.50,
		VerifierExposurePenalty: 0.35,
	}

	attributionAndQueue := baseline
	attributionAndQueue.EscapeAttributionRate = 0.80
	attributionAndQueue.IntroducerLinkRate = 0.75
	attributionAndQueue.QueueClassifiedRate = 0.85

	gateAndSeparation := attributionAndQueue
	gateAndSeparation.GatePrecision = 0.88
	gateAndSeparation.CatcherRetentionRate = 0.85
	gateAndSeparation.JudgeSeparationRate = 0.90
	gateAndSeparation.VerifierExposurePenalty = 0.10

	proposals := []VerifyEffProposal{
		{Label: "verifyeff:no-op", Snapshot: baseline},
		{Label: "verifyeff:attribution-and-queue-uplift", Snapshot: attributionAndQueue},
		{Label: "verifyeff:gate-signal-and-separation-uplift", Snapshot: gateAndSeparation},
	}

	return NewVerifyEffHarnessWith(baseline, proposals)
}

// NewVerifyEffHarnessWith builds a verification effectiveness Harness from an
// explicit baseline snapshot and proposal sequence.
func NewVerifyEffHarnessWith(baseline VerifyEffSnapshot, proposals []VerifyEffProposal) Harness {
	props := append([]VerifyEffProposal(nil), proposals...)
	return Harness{
		MetricName:      VerifyEffMetricName,
		LowerBetter:     false,
		BaselineRefName: "verifyeff-baseline",
		BaselineMetric: func() (float64, string, error) {
			vei, _ := ScoreVerifyEff(baseline)
			return vei, fmt.Sprintf("verifyeff@vei=%.2f", vei), nil
		},
		Candidates: func() []Candidate {
			out := make([]Candidate, len(props))
			for i, p := range props {
				out[i] = Candidate{Label: p.Label, Payload: i}
			}
			return out
		},
		Measure: func(c Candidate) (Measurement, error) {
			i, ok := c.Payload.(int)
			if !ok || i < 0 || i >= len(props) {
				return Measurement{}, fmt.Errorf("verifyeff proposal index %v out of range", c.Payload)
			}
			snap := props[i].Snapshot
			valid := verifyEffSnapshotValid(snap)
			vei, sc := ScoreVerifyEff(snap)
			return Measurement{
				Metric:     vei,
				SuiteGreen: valid,
				TruthClean: valid && snap.VerifierExposurePenalty < 1.0,
				Score:      &sc,
				Note: fmt.Sprintf("vei=%.2f grade=%s attr=%.2f qcls=%.2f prec=%.2f sep=%.2f pen=%.2f",
					vei, sc.Grade, snap.EscapeAttributionRate, snap.QueueClassifiedRate,
					snap.GatePrecision, snap.JudgeSeparationRate, snap.VerifierExposurePenalty),
			}, nil
		},
	}
}

func verifyEffSnapshotValid(s VerifyEffSnapshot) bool {
	vals := []float64{
		s.EscapeAttributionRate,
		s.IntroducerLinkRate,
		s.GatePrecision,
		s.CatcherRetentionRate,
		s.QueueClassifiedRate,
		s.JudgeSeparationRate,
		s.VerifierExposurePenalty,
	}
	for _, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			return false
		}
	}
	return true
}

func clampVerifyEffUnit(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return v
}

func verifyEffGrade(vei float64) string {
	switch {
	case vei >= 90:
		return "A"
	case vei >= 75:
		return "B"
	case vei >= 60:
		return "C"
	case vei >= 40:
		return "D"
	default:
		return "F"
	}
}
