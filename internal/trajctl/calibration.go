package trajctl

// calibration.go — the SCORER CALIBRATION meter (issue #2566, epic #2533): the
// improve-the-scoring-method half of the doctrine. The curve fold (curve.go) reads a
// scorer's numbers at face value; nothing yet asks whether those numbers TRACK REALITY.
// A judge scorer that reports steady progress while the witnessed outcome goes nowhere is
// worse than no scorer — it steers sessions wrong, silently. This fold measures, per
// scorer method+version, how well its scores CORRELATE with the eventual W3 witnessed
// outcome, and ranks the scorers so a mis-calibrated one is visible (annotated, never
// deleted — a low-calibration scorer is a repair target, not garbage).
//
// The ground truth is the W3 witnessed-commit-progress method (the deterministic,
// zero-model-call outcome scorer): for each objective its FINAL W3 value is where the
// objective actually ended up. A scorer is calibrated to the extent its readings across
// objectives move WITH those outcomes — high where the objective was met, low where it
// was abandoned. The measure is the Pearson correlation of (scorer value, objective
// outcome) pairs pooled across objectives; it needs at least two objectives with DIFFERING
// outcomes, so a scorer that has only ever seen one outcome reads INSUFFICIENT (surfaced,
// never scored as if calibrated). The fold is pure and tier-1: it reads the already-folded
// State and no clock/disk.

import (
	"fmt"
	"math"
	"sort"
)

// CalibrationSchema is the pinned schema id for a calibration report.
const CalibrationSchema = "fak-trajctl-calibration/1"

// GroundTruthMethod is the W3 outcome method every other scorer is calibrated against —
// the witnessed-commit-progress scorer, the deterministic outcome signal.
const GroundTruthMethod = CommitScorerMethod

// CalibrationVerdict is the closed calibration vocabulary. WELL/WEAK/MISCALIBRATED are
// measured bands; INSUFFICIENT means there was not enough varying outcome to correlate.
type CalibrationVerdict string

const (
	// CalibrationWell: the scorer's numbers track the witnessed outcome strongly
	// (r >= calibrationWellThreshold) — trustworthy enough to steer on.
	CalibrationWell CalibrationVerdict = "WELL_CALIBRATED"
	// CalibrationWeak: a positive but weak correlation (0 <= r < well) — better than
	// noise, not trustworthy alone.
	CalibrationWeak CalibrationVerdict = "WEAKLY_CALIBRATED"
	// CalibrationMiscalibrated: the scorer ANTI-correlates with the outcome (r < 0) — its
	// high scores precede low outcomes. The named enemy: it steers the wrong way.
	CalibrationMiscalibrated CalibrationVerdict = "MISCALIBRATED"
	// CalibrationInsufficient: fewer than two paired samples, or no variance to correlate
	// (every paired outcome identical) — unknown, not bad.
	CalibrationInsufficient CalibrationVerdict = "INSUFFICIENT"
)

// calibrationWellThreshold is the pinned correlation at or above which a scorer reads
// WELL_CALIBRATED. Below it (but non-negative) is WEAK; negative is MISCALIBRATED.
const calibrationWellThreshold = 0.5

// ScorerCalibration is one scorer method+version's calibration verdict: how many paired
// samples fed it, its Pearson coefficient against the W3 outcome (0 when unmeasured), and
// the closed-vocabulary verdict. Measured is false exactly when the verdict is INSUFFICIENT.
type ScorerCalibration struct {
	Method      string             `json:"method"`
	Version     string             `json:"version"`
	Samples     int                `json:"samples"`
	Coefficient float64            `json:"coefficient"`
	Measured    bool               `json:"measured"`
	Verdict     CalibrationVerdict `json:"verdict"`
	Detail      string             `json:"detail"`
}

// CalibrationReport is the schema-pinned leaderboard. Scorers are ranked best-first
// (highest correlation at rank 1), so the WORST-calibrated measured scorer — the one most
// in need of repair — sits at the bottom; INSUFFICIENT scorers (nothing to rank yet) trail
// the measured rows. GroundTruth names the outcome method the correlation is against.
type CalibrationReport struct {
	Schema      string              `json:"schema"`
	GroundTruth string              `json:"ground_truth_method"`
	Scorers     []ScorerCalibration `json:"scorers"`
}

// Calibrate folds every non-ground-truth scorer's numbers against the W3 outcome and
// returns the ranked leaderboard. The outcome for an objective is its FINAL W3
// witnessed-commit-progress value (latest by UnixMillis, ties by append order); a scorer's
// value rows are each paired with their objective's outcome and pooled per method+version.
func (s State) Calibrate() CalibrationReport {
	outcome, haveOutcome := s.finalGroundTruth()

	type key struct{ method, version string }
	pairsX := map[key][]float64{}
	pairsY := map[key][]float64{}
	order := make([]key, 0)
	seen := map[key]bool{}
	for _, row := range s.Scores {
		if row.Method == GroundTruthMethod {
			continue // the ground truth cannot calibrate against itself
		}
		if row.Method == MetaScorerMethod {
			continue // the meta scorer is never its own repair target — the one-level fence (#2567)
		}
		if !haveOutcome[row.ObjectiveID] {
			continue // no witnessed outcome to correlate this objective against
		}
		k := key{row.Method, row.Version}
		if !seen[k] {
			seen[k] = true
			order = append(order, k)
		}
		pairsX[k] = append(pairsX[k], row.Value)
		pairsY[k] = append(pairsY[k], outcome[row.ObjectiveID])
	}

	scorers := make([]ScorerCalibration, 0, len(order))
	for _, k := range order {
		sc := ScorerCalibration{Method: k.method, Version: k.version, Samples: len(pairsX[k])}
		if r, ok := pearson(pairsX[k], pairsY[k]); ok {
			sc.Measured = true
			sc.Coefficient = r
			sc.Verdict = calibrationVerdict(r)
			sc.Detail = fmt.Sprintf("r=%+.2f over %d sample(s) vs the %s outcome", r, sc.Samples, GroundTruthMethod)
		} else {
			sc.Verdict = CalibrationInsufficient
			sc.Detail = fmt.Sprintf("insufficient signal: %d paired sample(s) with no outcome variance to correlate — needs >=2 objectives with differing outcomes", sc.Samples)
		}
		scorers = append(scorers, sc)
	}

	sort.SliceStable(scorers, func(i, j int) bool {
		a, b := scorers[i], scorers[j]
		if a.Measured != b.Measured {
			return a.Measured // measured rows rank above unmeasured (INSUFFICIENT) ones
		}
		if a.Measured && a.Coefficient != b.Coefficient {
			return a.Coefficient > b.Coefficient // best-first: the worst measured scorer sinks to the bottom
		}
		if a.Method != b.Method {
			return a.Method < b.Method
		}
		return a.Version < b.Version
	})
	return CalibrationReport{Schema: CalibrationSchema, GroundTruth: GroundTruthMethod, Scorers: scorers}
}

// WorstCalibrated returns the lowest-ranked MEASURED scorer — the worst-first repair
// target the meta loop (#2567) enters — and true, or a zero value and false when no scorer
// has enough signal to rank. Reads the best-first report's measured tail.
func (r CalibrationReport) WorstCalibrated() (ScorerCalibration, bool) {
	for i := len(r.Scorers) - 1; i >= 0; i-- {
		if r.Scorers[i].Measured {
			return r.Scorers[i], true
		}
	}
	return ScorerCalibration{}, false
}

// finalGroundTruth folds the latest W3 witnessed-commit-progress value per objective — the
// outcome each scorer is calibrated against. Latest wins by UnixMillis; an unstamped or
// tied row keeps append order (the last seen).
func (s State) finalGroundTruth() (outcome map[string]float64, have map[string]bool) {
	outcome = map[string]float64{}
	have = map[string]bool{}
	at := map[string]int64{}
	for _, row := range s.Scores {
		if row.Method != GroundTruthMethod || row.Witness != W3 {
			continue
		}
		if !have[row.ObjectiveID] || row.UnixMillis >= at[row.ObjectiveID] {
			outcome[row.ObjectiveID] = row.Value
			at[row.ObjectiveID] = row.UnixMillis
			have[row.ObjectiveID] = true
		}
	}
	return outcome, have
}

// calibrationVerdict bands a Pearson coefficient into the measured vocabulary.
func calibrationVerdict(r float64) CalibrationVerdict {
	switch {
	case r >= calibrationWellThreshold:
		return CalibrationWell
	case r >= 0:
		return CalibrationWeak
	default:
		return CalibrationMiscalibrated
	}
}

// pearson returns the Pearson correlation of xs and ys and true, or 0 and false when the
// sample is too small (<2) or either series has zero variance (a constant series has no
// correlation to measure). Total and pure.
func pearson(xs, ys []float64) (float64, bool) {
	n := len(xs)
	if n < 2 || n != len(ys) {
		return 0, false
	}
	var sx, sy float64
	for i := 0; i < n; i++ {
		sx += xs[i]
		sy += ys[i]
	}
	mx, my := sx/float64(n), sy/float64(n)
	var cov, vx, vy float64
	for i := 0; i < n; i++ {
		dx, dy := xs[i]-mx, ys[i]-my
		cov += dx * dy
		vx += dx * dx
		vy += dy * dy
	}
	if vx == 0 || vy == 0 {
		return 0, false
	}
	return cov / math.Sqrt(vx*vy), true
}
