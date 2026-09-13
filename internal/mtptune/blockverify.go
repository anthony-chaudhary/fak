package mtptune

import (
	"fmt"
	"math"
	"strings"
)

// BlockVerifyConfig configures a deterministic block-verification sweep over draft
// lengths K=KMin..KMax for speculative decoding.
//
// PerfectDraftAccept is the per-position acceptance probability p of the modeled
// draft profile. A value of 1.0 is a true "perfect draft": the draft matches the
// reference on every position, so acceptance is deterministic (all K proposed
// tokens are always accepted). A value below 1.0 models a realistic profile where
// acceptance decays geometrically with position: expected accepted drafts at depth
// K is sum_{i=1..K} p^i, which stops growing as K increases and produces a plateau.
//
// BonusEnabled selects the 1+N bonus layout: the target model contributes one real
// token per verify step regardless of how many drafts are accepted, so committed
// tokens per step is 1 + (accepted drafts). With BonusEnabled=false the step's real
// token output is exactly the accepted drafts (no bonus token).
type BlockVerifyConfig struct {
	KMin               int     `json:"k_min"`
	KMax               int     `json:"k_max"`
	PerfectDraftAccept float64 `json:"perfect_draft_accept"`
	BonusEnabled       bool    `json:"bonus_enabled"`
}

// DefaultBlockVerifyAccept is the documented default per-position acceptance
// probability for the modeled realistic (decayed) draft profile. Accepting every
// position is unrealistic, so the default uses p=0.9: the expected accepted
// drafts at depth K is sum_{i=1..K} 0.9^i, which saturates near 9 as K grows and
// yields an observable plateau well inside K<=32.
//
// The TRUE perfect draft is p=1.0, exercised by TestBlockVerificationBonusLayout:
// accepted/step is exactly K+1 under the 1+N bonus layout (K without it), it
// grows without bound, and it has no plateau in range. The default p=0.9 profile
// is the decayed case that plateaus; it is not a perfect draft.
const DefaultBlockVerifyAccept = 0.9

// DefaultBlockVerifyConfig returns the standard sweep configuration: K=1..32, the
// realistic decayed draft profile at p=0.9, and the 1+N bonus layout enabled.
func DefaultBlockVerifyConfig() BlockVerifyConfig {
	return BlockVerifyConfig{
		KMin:               1,
		KMax:               32,
		PerfectDraftAccept: DefaultBlockVerifyAccept,
		BonusEnabled:       true,
	}
}

// BlockVerifyPoint is one measured point of the sweep at draft length K.
type BlockVerifyPoint struct {
	K                     int     `json:"k"`
	AcceptedTokensPerStep float64 `json:"accepted_tokens_per_step"`
	TotalAccepted         int     `json:"total_accepted"`
	Steps                 int     `json:"steps"`
	StepLatencyMs         float64 `json:"step_latency_ms"`
	ThroughputTPS         float64 `json:"throughput_tps"`
}

// BlockVerifyReport is the deterministic result of a block-verification sweep.
//
// AcceptedTokensPerStep is the measured (modeled) series the plateau detector runs
// on. PlateauK is the smallest K>=2 where the relative marginal gain over the
// previous K falls below plateauEpsilon and stays below for all larger K.
// PeakK/PeakAcceptedPerStep is the K maximizing AcceptedTokensPerStep — the honest
// "pick K from measured data" answer.
type BlockVerifyReport struct {
	Config                   BlockVerifyConfig  `json:"config"`
	Points                   []BlockVerifyPoint `json:"points"`
	PlateauK                 int                `json:"plateau_k"`
	PlateauAcceptedPerStep   float64            `json:"plateau_accepted_per_step"`
	PeakK                    int                `json:"peak_k"`
	PeakAcceptedPerStep      float64            `json:"peak_accepted_per_step"`
	ImprovementPctVsBaseline float64            `json:"improvement_pct_vs_baseline"`
}

// plateauEpsilon is the relative marginal-gain threshold below which a step's
// contribution is considered flat. A 5% relative gain is the documented cutoff.
const plateauEpsilon = 0.05

// Validate checks the block-verification configuration.
func (cfg BlockVerifyConfig) Validate() error {
	if cfg.KMin < 1 {
		return fmt.Errorf("invalid KMin: %d (must be >= 1)", cfg.KMin)
	}
	if cfg.KMax < cfg.KMin {
		return fmt.Errorf("invalid K range: [%d, %d] (KMax must be >= KMin)", cfg.KMin, cfg.KMax)
	}
	if cfg.KMax > 64 {
		return fmt.Errorf("invalid KMax: %d (must be <= 64)", cfg.KMax)
	}
	if cfg.PerfectDraftAccept < 0 || cfg.PerfectDraftAccept > 1 {
		return fmt.Errorf("invalid PerfectDraftAccept: %f (must be in [0, 1])", cfg.PerfectDraftAccept)
	}
	return nil
}

// acceptedDraftsAtDepth returns the modeled expected accepted draft tokens for a
// draft of length K under per-position acceptance probability p, using geometric
// decay: sum_{i=1..K} p^i. At p=1 this reduces exactly to K (a perfect draft).
func acceptedDraftsAtDepth(k int, p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return float64(k)
	}
	// Closed form of the geometric series sum_{i=1..K} p^i avoids per-step drift.
	return p * (1 - math.Pow(p, float64(k))) / (1 - p)
}

// RunBlockVerifySweep sweeps K=KMin..KMax and computes accepted-tokens/step
// for the configured draft profile. It validates cfg, is deterministic (no
// wall-clock or float-from-time inputs), and never claims upstream's +17-20%.
//
// Committed tokens per step = accepted drafts, plus one bonus target token when
// BonusEnabled. Step latency/throughput are modeled with the existing
// EstimateThroughput helper (Apple Silicon Metal) when available; the same helper
// already incorporates the K+1 traffic and rollback penalty, so no new hardware
// numbers are invented here.
func RunBlockVerifySweep(cfg BlockVerifyConfig) (*BlockVerifyReport, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	hw := AppleSiliconMetalSweepConfig()
	points := make([]BlockVerifyPoint, 0, cfg.KMax-cfg.KMin+1)

	for k := cfg.KMin; k <= cfg.KMax; k++ {
		acceptedDrafts := acceptedDraftsAtDepth(k, cfg.PerfectDraftAccept)
		committedPerStep := acceptedDrafts
		if cfg.BonusEnabled {
			committedPerStep = acceptedDrafts + 1.0
		}

		// Deterministic measurement accounting: one verify step per point,
		// TotalAccepted is the integer-truncated committed tokens for that step.
		steps := 1
		totalAccepted := int(math.Floor(committedPerStep + 1e-9))

		// Per-step acceptance rate seen by the throughput model is the fraction of
		// proposed drafts that survive; at p=1 this is 1.0 (perfect draft).
		acceptanceRate := 0.0
		if k > 0 {
			acceptanceRate = acceptedDrafts / float64(k)
		}

		stepLatencyMs := 0.0
		throughput := 0.0
		if hw.ModelWeightGB > 0 {
			// EstimateThroughput returns committed tokens/sec for the modeled step.
			throughput = EstimateThroughput(k, acceptanceRate, hw)
			if throughput > 0 {
				stepLatencyMs = (committedPerStep / throughput) * 1000.0
			}
		}

		points = append(points, BlockVerifyPoint{
			K:                     k,
			AcceptedTokensPerStep: committedPerStep,
			TotalAccepted:         totalAccepted,
			Steps:                 steps,
			StepLatencyMs:         stepLatencyMs,
			ThroughputTPS:         throughput,
		})
	}

	report := &BlockVerifyReport{
		Config: cfg,
		Points: points,
	}

	// Plateau: smallest K>=2 whose relative marginal gain over K-1 is below
	// plateauEpsilon and remains below for every larger K.
	report.PlateauK, report.PlateauAcceptedPerStep = detectPlateau(points)

	// Peak: K maximizing accepted-tokens/step; ties resolve to the smallest K.
	report.PeakK = points[0].K
	report.PeakAcceptedPerStep = points[0].AcceptedTokensPerStep
	for _, pt := range points {
		if pt.AcceptedTokensPerStep > report.PeakAcceptedPerStep+1e-12 {
			report.PeakK = pt.K
			report.PeakAcceptedPerStep = pt.AcceptedTokensPerStep
		}
	}

	// Improvement vs the baseline K=KMin point.
	baseline := points[0].AcceptedTokensPerStep
	if baseline > 0 {
		report.ImprovementPctVsBaseline = (report.PeakAcceptedPerStep - baseline) / baseline * 100.0
	}

	return report, nil
}

// detectPlateau returns the smallest K>=2 where the relative marginal gain from
// K-1 to K is below plateauEpsilon and stays below for all larger K, along with
// that point's accepted-tokens/step. If no such K exists, it returns the last
// point's K and value (no plateau detected within range).
func detectPlateau(points []BlockVerifyPoint) (int, float64) {
	for i := 1; i < len(points); i++ {
		prev := points[i-1].AcceptedTokensPerStep
		gain := 0.0
		if prev > 0 {
			gain = (points[i].AcceptedTokensPerStep - prev) / prev
		}
		if gain >= plateauEpsilon {
			continue
		}
		// Candidate: verify the gain stays below epsilon for every later step.
		flat := true
		for j := i + 1; j < len(points); j++ {
			p := points[j-1].AcceptedTokensPerStep
			g := 0.0
			if p > 0 {
				g = (points[j].AcceptedTokensPerStep - p) / p
			}
			if g >= plateauEpsilon {
				flat = false
				break
			}
		}
		if flat {
			return points[i].K, points[i].AcceptedTokensPerStep
		}
	}
	last := points[len(points)-1]
	return last.K, last.AcceptedTokensPerStep
}

// FormatBlockVerifyReport renders the sweep as an aligned, deterministic text
// table. The header names the profile honestly: the default p=0.9 is a decayed
// draft profile (it plateaus), whereas p=1.0 is the true perfect draft used by
// the bonus-layout check (accepted/step K+1, no plateau). The plateau point is
// marked "PLATEAU" and the peak point "PEAK".
func FormatBlockVerifyReport(r *BlockVerifyReport) string {
	if r == nil {
		return "block-verify: <nil report>\n"
	}

	var sb strings.Builder
	sb.WriteString("===================================================================\n")
	sb.WriteString("         BLOCK-VERIFICATION SWEEP (decayed-draft profile)\n")
	sb.WriteString("===================================================================\n")
	sb.WriteString(fmt.Sprintf("K=%d..%d | p=%.3f | bonus-1+N=%v | plateau eps=%.0f%%\n",
		r.Config.KMin, r.Config.KMax, r.Config.PerfectDraftAccept, r.Config.BonusEnabled, plateauEpsilon*100))
	sb.WriteString("-------------------------------------------------------------------\n")
	sb.WriteString(fmt.Sprintf("%-4s  %-12s  %-14s  %-12s  %s\n",
		"K", "acc/step", "throughput_tps", "latency_ms", "mark"))
	sb.WriteString("-------------------------------------------------------------------\n")

	for _, pt := range r.Points {
		mark := ""
		switch {
		case pt.K == r.PlateauK && pt.K == r.PeakK:
			mark = "PLATEAU+PEAK"
		case pt.K == r.PlateauK:
			mark = "PLATEAU"
		case pt.K == r.PeakK:
			mark = "PEAK"
		}
		sb.WriteString(fmt.Sprintf("%-4d  %-12.4f  %-14.2f  %-12.3f  %s\n",
			pt.K, pt.AcceptedTokensPerStep, pt.ThroughputTPS, pt.StepLatencyMs, mark))
	}

	sb.WriteString("-------------------------------------------------------------------\n")
	sb.WriteString(fmt.Sprintf("plateau: K=%d (%.4f acc/step) | peak: K=%d (%.4f acc/step)\n",
		r.PlateauK, r.PlateauAcceptedPerStep, r.PeakK, r.PeakAcceptedPerStep))
	sb.WriteString(fmt.Sprintf("improvement vs K=%d baseline: %.1f%%\n",
		r.Config.KMin, r.ImprovementPctVsBaseline))
	sb.WriteString("===================================================================\n")
	return sb.String()
}
