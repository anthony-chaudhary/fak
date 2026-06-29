// changepoint.go — CHANGE-POINT DETECTION on the overhead/benchmark series (issue #1163, T7 of
// the self-tax assurance epic #1147). The L3 regression gate (T6, #1162) watches a stream of
// per-run overhead measurements — the fak-ON vs fak-OFF delta on the frozen canonical workload —
// and must decide: did the kernel's overhead REGRESS, persistently, or is this just benchmark
// noise? A BRITTLE FIXED THRESHOLD ("red if p50 > 600ns") fails both ways: it false-reds on a
// single noisy spike, and it misses a small-but-real drift that stays under the line. This file
// replaces that threshold with CHANGE-POINT DETECTION over the stored series — the Hunter /
// USENIX SREcon approach (§4 of the epic design note) — so the gate fires on a DISTRIBUTION SHIFT
// THAT PERSISTS, not on one sample.
//
// THE METHOD (stated honestly). This is BINARY SEGMENTATION with a PERMUTATION significance test:
//
//  1. SPLIT STATISTIC. Over a segment x[lo:hi], scan every candidate split k and compute the
//     two-sample t-statistic of the mean shift across it:
//
//         t(k) = |mean(x[k:hi]) − mean(x[lo:k])| / sqrt( sp² · (1/nL + 1/nR) )
//
//     with sp² the pooled within-segment variance. The split k* with the largest |t| is the most
//     change-point-like position in the segment. (This is the standardized CUSUM / maximum-
//     likelihood change-in-mean estimator for a Gaussian — the same family Hunter's t-test uses.)
//
//  2. PERMUTATION SIGNIFICANCE (the no-false-red guard). The max over k of n−1 correlated
//     statistics has a far heavier null than a single t — so a single-test critical value would
//     inflate false reds. Under the null "no change point" the segment is EXCHANGEABLE, so we
//     shuffle it `Permutations` times, recompute the SAME max statistic each time, and read the
//     p-value off that null: p = (1 + #{perm stat ≥ observed}) / (1 + Permutations). This is
//     multiple-comparison-correct BY CONSTRUCTION — the observed max is compared to the null
//     distribution OF THE MAX — which is exactly why stationary noise does not red (its observed
//     max is a typical draw from its own permutation null, so p ≈ uniform, mostly > Alpha).
//
//  3. PERSISTENCE (§4 "exceeds AND persists"). A change is accepted only if each side has at least
//     `MinSegment` runs — a one-sample spike at the edge cannot be a change point. This is the
//     design note's persistence rule (≥N runs), enforced as the minimum segment width.
//
//  4. RECURSE. On an accepted split, recurse into both halves to find further change points; a
//     half that no longer carries a significant split terminates. The result is the ordered set
//     of persistent distribution shifts in the series.
//
// THE ACCEPTANCE (this ticket's witness, changepoint_test.go). An injected step-change in the
// series is flagged (a change point at the step, the shift sign correct); a stationary-noise
// series is NOT flagged (no false red) — asserted both on a single deterministic fixture and as a
// bounded false-positive RATE over many seeded noise series (the principled "no false red", the
// way sprt_test.go witnesses the ~half-samples claim over 400 fixtures rather than one lucky seed).
//
// HONESTY FENCE (the lane boundary, mirroring sprt.go / the §12 contract). This file ships the
// DETECTOR — the self-contained statistical unit the gate's persistence rule consumes. WIRING it
// into a live `make ci` RED is T6 (#1162), which is blocked on the T2 budget (#1150) that defines
// "over-budget": no budget ⇒ no calibrated breach ⇒ no gate (the epic's §8 sequencing law). The
// detector does not need the budget — it asks "did the SERIES distribution-shift?", not "is it
// over budget?" — so it is buildable and witnessable today, exactly as its sibling T8 SPRT
// (sprt.go) shipped its sequential decision without the budget-gated red.
package turnbench

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// defaultChangePointSeed seeds the permutation null when a caller leaves ChangePointConfig.Seed
// unset, so DetectChangePoints is deterministic out of the box (the permutation p-value, and thus
// every verdict, is a pure function of the series + config). A caller varies it only to draw an
// independent null.
const defaultChangePointSeed int64 = 0x1163C9A11

// ChangePointConfig parameterizes the detector. The zero value is valid: withDefaults fills every
// field, so DetectChangePoints(label, series, ChangePointConfig{}) is the canonical call.
type ChangePointConfig struct {
	// Alpha is the permutation-test significance level — a segment splits only when its
	// multiple-comparison-corrected p-value ≤ Alpha. Lower Alpha ⇒ fewer false reds, less power.
	// Default 0.01.
	Alpha float64 `json:"alpha"`

	// MinSegment is the persistence rule (§4): the minimum number of runs on EACH side of a change
	// point, so a one-sample spike cannot flag. Must be ≥ 2 (a t-statistic needs df > 0). Default 5.
	MinSegment int `json:"min_segment"`

	// Permutations is the number of shuffles used to estimate the null distribution of the max
	// split statistic. The smallest achievable p-value is 1/(Permutations+1), so Permutations must
	// exceed 1/Alpha for a change to be able to reach significance. Default 500.
	Permutations int `json:"permutations"`

	// Seed makes the permutation null deterministic. Default defaultChangePointSeed.
	Seed int64 `json:"seed"`
}

// withDefaults returns a copy with every unset/degenerate field filled, so the detector has a
// well-formed config regardless of what the caller passed.
func (c ChangePointConfig) withDefaults() ChangePointConfig {
	if c.Alpha <= 0 || c.Alpha >= 1 {
		c.Alpha = 0.01
	}
	if c.MinSegment < 2 {
		c.MinSegment = 5
	}
	if c.Permutations <= 0 {
		c.Permutations = 500
	}
	if c.Seed == 0 {
		c.Seed = defaultChangePointSeed
	}
	return c
}

// ChangePoint is one detected persistent distribution shift in the series.
type ChangePoint struct {
	Index      int     `json:"index"`       // series index where the NEW segment begins (the split position)
	Statistic  float64 `json:"statistic"`   // the max two-sample t at the split (the change-point evidence)
	PValue     float64 `json:"p_value"`     // permutation p-value (multiple-comparison corrected)
	Confidence float64 `json:"confidence"`  // 1 − p_value
	MeanBefore float64 `json:"mean_before"` // mean of the segment left of the split
	MeanAfter  float64 `json:"mean_after"`  // mean of the segment right of the split
	Shift      float64 `json:"shift"`       // mean_after − mean_before (signed: + is a regression on an overhead series)
}

// ChangePointReport is the detector's artifact: the ordered change points and the headline
// Shifted boolean the gate's persistence rule reads (a persistent distribution shift vs a single
// spike). It mirrors the other turnbench reports (Provenance + JSON()).
type ChangePointReport struct {
	Provenance Provenance `json:"provenance"`

	N            int     `json:"n"`            // series length
	Alpha        float64 `json:"alpha"`        // significance level used
	MinSegment   int     `json:"min_segment"`  // persistence width used
	Permutations int     `json:"permutations"` // permutation count used

	// Shifted is true iff at least one persistent change point was found — the signal the L3 gate
	// consumes: "the series distribution-shifted" (red-worthy), distinct from a single noisy spike.
	Shifted      bool          `json:"shifted"`
	ChangePoints []ChangePoint `json:"change_points"`

	Note string `json:"note"`
}

// JSON renders the report in the canonical turnbench artifact encoding.
func (r *ChangePointReport) JSON() []byte { return marshalArtifact(r) }

// DetectChangePoints runs binary-segmentation change-point detection over series and returns the
// ordered persistent distribution shifts (each significant at cfg.Alpha under the permutation
// null, each with ≥ cfg.MinSegment runs per side). label is stamped as the artifact's SliceID. A
// series too short to carry a persistence-respecting split returns a valid "stationary" report
// (not enough runs yet is a legitimate answer for a gate called as the series grows); only an
// empty series is an error.
func DetectChangePoints(label string, series []float64, cfg ChangePointConfig) (*ChangePointReport, error) {
	if len(series) == 0 {
		return nil, fmt.Errorf("turnbench: DetectChangePoints needs a non-empty series")
	}
	cfg = cfg.withDefaults()
	rng := rand.New(rand.NewSource(cfg.Seed))

	var cps []ChangePoint
	// Binary segmentation: find the most-significant split in a segment, accept it iff it clears
	// the permutation gate, then recurse into both halves. The rng is consumed in a deterministic
	// (left-before-right) order, so the whole detection is reproducible.
	var search func(lo, hi int)
	search = func(lo, hi int) {
		seg := series[lo:hi]
		if len(seg) < 2*cfg.MinSegment {
			return // too short to split with a MinSegment-wide side on each end
		}
		stat, kLocal := maxSplitStat(seg, cfg.MinSegment)
		if kLocal < 0 {
			return
		}
		p := permutationPValue(seg, stat, cfg.MinSegment, cfg.Permutations, rng)
		if p > cfg.Alpha {
			return // not a distribution shift — just noise; do NOT red
		}
		k := lo + kLocal
		before, after := meanOf(series[lo:k]), meanOf(series[k:hi])
		cps = append(cps, ChangePoint{
			Index:      k,
			Statistic:  stat,
			PValue:     p,
			Confidence: 1 - p,
			MeanBefore: before,
			MeanAfter:  after,
			Shift:      after - before,
		})
		search(lo, k)
		search(k, hi)
	}
	search(0, len(series))
	sort.Slice(cps, func(i, j int) bool { return cps[i].Index < cps[j].Index })

	shifted := len(cps) > 0
	note := fmt.Sprintf("stationary — no persistent distribution shift over %d runs (no false red); "+
		"alpha=%g min_segment=%d permutations=%d.", len(series), cfg.Alpha, cfg.MinSegment, cfg.Permutations)
	if shifted {
		note = fmt.Sprintf("%d persistent change-point(s) over %d runs; first at index %d "+
			"(shift %+.4g, p=%.4g); alpha=%g min_segment=%d permutations=%d.",
			len(cps), len(series), cps[0].Index, cps[0].Shift, cps[0].PValue,
			cfg.Alpha, cfg.MinSegment, cfg.Permutations)
	}

	return &ChangePointReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.DetectChangePoints",
			SliceID:     label,
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (change-point detection — binary segmentation + permutation test)",
		},
		N:            len(series),
		Alpha:        cfg.Alpha,
		MinSegment:   cfg.MinSegment,
		Permutations: cfg.Permutations,
		Shifted:      shifted,
		ChangePoints: cps,
		Note:         note,
	}, nil
}

// maxSplitStat scans every split of seg that leaves ≥ minSeg points on each side and returns the
// largest two-sample t-statistic of the mean shift and the local split index achieving it. It uses
// prefix sums so the whole scan is O(len(seg)). Returns (best, -1) when no split respects minSeg.
func maxSplitStat(seg []float64, minSeg int) (best float64, splitLocal int) {
	m := len(seg)
	if m < 2*minSeg {
		return -1, -1
	}
	// Prefix sums of x and x² for O(1) per-split mean and sum-of-squares.
	pre := make([]float64, m+1)
	pre2 := make([]float64, m+1)
	for i, v := range seg {
		pre[i+1] = pre[i] + v
		pre2[i+1] = pre2[i] + v*v
	}
	best, splitLocal = -1, -1
	for k := minSeg; k <= m-minSeg; k++ {
		nL, nR := float64(k), float64(m-k)
		sumL, sumR := pre[k], pre[m]-pre[k]
		meanL, meanR := sumL/nL, sumR/nR
		// Within-side sum of squared deviations: Σx² − (Σx)²/n.
		ssL := pre2[k] - sumL*sumL/nL
		ssR := (pre2[m] - pre2[k]) - sumR*sumR/nR
		df := nL + nR - 2
		if df <= 0 {
			continue
		}
		sp2 := (ssL + ssR) / df       // pooled variance
		scale := sp2 * (1/nL + 1/nR)  // variance of the mean difference
		diff := meanR - meanL
		var t float64
		switch {
		case scale > 0:
			t = math.Abs(diff) / math.Sqrt(scale)
		case diff == 0:
			t = 0 // no within-variance and no shift ⇒ no evidence
		default:
			t = math.Inf(1) // a shift with zero within-variance ⇒ maximal evidence
		}
		if t > best {
			best, splitLocal = t, k
		}
	}
	return best, splitLocal
}

// permutationPValue estimates the p-value of `observed` under the null "seg is exchangeable" (no
// change point) by reshuffling seg `permutations` times and recomputing the max split statistic.
// p = (1 + #{perm ≥ observed}) / (1 + permutations) — the standard permutation p-value, which can
// never be 0 (the observed arrangement is itself one valid permutation). Because the null is the
// distribution OF THE MAX, this is multiple-comparison-correct, which is what keeps stationary
// noise from reding. An infinite observed statistic (zero within-variance, nonzero shift) is
// maximally significant and short-circuits to the smallest achievable p.
func permutationPValue(seg []float64, observed float64, minSeg, permutations int, rng *rand.Rand) float64 {
	if math.IsInf(observed, 1) {
		return 1.0 / float64(permutations+1)
	}
	perm := make([]float64, len(seg))
	copy(perm, seg)
	ge := 0
	for b := 0; b < permutations; b++ {
		rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		stat, _ := maxSplitStat(perm, minSeg)
		if stat >= observed {
			ge++
		}
	}
	return float64(1+ge) / float64(1+permutations)
}

// meanOf is the arithmetic mean of xs (0 for an empty slice).
func meanOf(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
