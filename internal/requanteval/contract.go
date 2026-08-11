// Package requanteval evaluates fixed-grid discrete refinement without selecting a
// universal quantization method.
package requanteval

import (
	"errors"
	"fmt"
	"math"
)

const (
	ContractVersion = "requanteval/v1"
	RecipeVersion   = "requant-arxiv-2608.07019v1-fixed-grid-v1"
	RuntimeVersion  = "fak-go-reference-cpu/v1"
	PaperID         = "arXiv:2608.07019v1"
	PaperSHA256     = "5505cab5060f170e5e7a03b07fce83682c7af3bb0c3cfbb9cdb5920c220d4beb"
)

type Outcome string

const (
	OutcomeSupported   Outcome = "supported"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeDelegate    Outcome = "delegate"
)

type ReasonCode string

const (
	ReasonEvaluated         ReasonCode = "REQUANT_EVALUATED"
	ReasonUnknownContract   ReasonCode = "REQUANT_UNKNOWN_CONTRACT"
	ReasonUnknownRecipe     ReasonCode = "REQUANT_UNKNOWN_RECIPE"
	ReasonInvalidGrid       ReasonCode = "REQUANT_INVALID_GRID"
	ReasonInvalidShape      ReasonCode = "REQUANT_INVALID_SHAPE"
	ReasonInvalidQuality    ReasonCode = "REQUANT_INVALID_QUALITY"
	ReasonRuntimeDelegation ReasonCode = "REQUANT_RUNTIME_DELEGATION"
	ReasonInvalidProvenance ReasonCode = "REQUANT_INVALID_PROVENANCE"
)

type EvidenceKind string

const (
	EvidenceModeled  EvidenceKind = "modeled"
	EvidenceObserved EvidenceKind = "observed"
)

type Provenance struct {
	ArtifactID         string `json:"artifact_id"`
	ArtifactVersion    string `json:"artifact_version"`
	ArtifactSHA256     string `json:"artifact_sha256"`
	Initializer        string `json:"initializer"`
	InitializerVersion string `json:"initializer_version"`
	Source             string `json:"source"`
}

type Request struct {
	ContractVersion string       `json:"contract_version"`
	RecipeVersion   string       `json:"recipe_version"`
	RuntimeVersion  string       `json:"runtime_version"`
	Seed            uint64       `json:"seed"`
	Grid            []float64    `json:"grid"`
	InitialCodes    []int        `json:"initial_codes"`
	Target          []float64    `json:"target"`
	Hessian         [][]float64  `json:"hessian"`
	MaxSweeps       int          `json:"max_sweeps"`
	Provenance      Provenance   `json:"provenance"`
	Quality         QualityProbe `json:"quality"`
}

// QualityProbe evaluates the initial and refined weight vectors on the same
// pinned synthetic linear-model examples. Expected contains one scalar output
// per input row.
type QualityProbe struct {
	ID       string      `json:"id"`
	Version  string      `json:"version"`
	Inputs   [][]float64 `json:"inputs"`
	Expected []float64   `json:"expected"`
}

type Metrics struct {
	InitialMSE      float64 `json:"initial_mse"`
	FinalMSE        float64 `json:"final_mse"`
	AbsoluteMSEGain float64 `json:"absolute_mse_gain"`
	RelativeMSEGain float64 `json:"relative_mse_gain"`
}

type QualityMetrics struct {
	Metric               string  `json:"metric"`
	Examples             int     `json:"examples"`
	InitialPredictionMSE float64 `json:"initial_prediction_mse"`
	FinalPredictionMSE   float64 `json:"final_prediction_mse"`
	AbsoluteGain         float64 `json:"absolute_gain"`
}

type ConversionCost struct {
	Sweeps             int `json:"sweeps"`
	CoordinatesVisited int `json:"coordinates_visited"`
	Evaluations        int `json:"candidate_evaluations"`
	AcceptedUpdates    int `json:"accepted_updates"`
}

type ClaimCheck struct {
	Verdict string `json:"verdict"`
	Basis   string `json:"basis"`
}

type Result struct {
	Outcome          Outcome         `json:"outcome"`
	Reason           ReasonCode      `json:"reason"`
	Detail           string          `json:"detail,omitempty"`
	ContractVersion  string          `json:"contract_version"`
	RecipeVersion    string          `json:"recipe_version"`
	RuntimeVersion   string          `json:"runtime_version"`
	PaperID          string          `json:"paper_id"`
	PaperSHA256      string          `json:"paper_sha256"`
	Seed             uint64          `json:"seed"`
	Grid             []float64       `json:"grid,omitempty"`
	InitialCodes     []int           `json:"initial_codes,omitempty"`
	RefinedCodes     []int           `json:"refined_codes,omitempty"`
	Metrics          *Metrics        `json:"metrics,omitempty"`
	Quality          *QualityMetrics `json:"quality,omitempty"`
	ConversionCost   *ConversionCost `json:"conversion_cost,omitempty"`
	Evidence         EvidenceKind    `json:"evidence"`
	HardwareEnvelope string          `json:"hardware_envelope,omitempty"`
	Provenance       Provenance      `json:"provenance"`
	ClaimCheck       ClaimCheck      `json:"claim_check"`
}

// Evaluate runs the paper's fixed-grid, one-coordinate-at-a-time acceptance
// rule on a bounded neutral fixture. It is an evaluator, not a production
// model converter. The seed is recorded but no randomness is used by v1.
func Evaluate(req Request) Result {
	out := baseResult(req)
	refuse := func(reason ReasonCode, detail string) Result {
		out.Outcome, out.Reason, out.Detail = OutcomeUnsupported, reason, detail
		return out
	}
	if req.ContractVersion != ContractVersion {
		return refuse(ReasonUnknownContract, "contract version is not recognized")
	}
	if req.RecipeVersion != RecipeVersion {
		return refuse(ReasonUnknownRecipe, "recipe version is not recognized")
	}
	if req.RuntimeVersion != RuntimeVersion {
		out.Outcome, out.Reason = OutcomeDelegate, ReasonRuntimeDelegation
		out.Detail = "requested runtime is not the pinned reference CPU evaluator"
		return out
	}
	if req.Provenance.ArtifactID == "" || req.Provenance.ArtifactVersion == "" || req.Provenance.ArtifactSHA256 == "" || req.Provenance.Initializer == "" || req.Provenance.InitializerVersion == "" || req.Provenance.Source == "" {
		return refuse(ReasonInvalidProvenance, "artifact and initializer provenance must be complete")
	}
	if err := validateGrid(req.Grid, req.InitialCodes); err != nil {
		return refuse(ReasonInvalidGrid, err.Error())
	}
	if err := validateShape(req); err != nil {
		return refuse(ReasonInvalidShape, err.Error())
	}
	if err := validateQuality(req.Quality, len(req.InitialCodes)); err != nil {
		return refuse(ReasonInvalidQuality, err.Error())
	}

	codes := append([]int(nil), req.InitialCodes...)
	initial := objective(req, codes)
	current := initial
	cost := ConversionCost{}
	for sweep := 0; sweep < req.MaxSweeps; sweep++ {
		changed := false
		cost.Sweeps++
		for i := range codes {
			cost.CoordinatesVisited++
			bestCode, best := codes[i], current
			for candidate := range req.Grid {
				if candidate == codes[i] {
					continue
				}
				trial := append([]int(nil), codes...)
				trial[i] = candidate
				score := objective(req, trial)
				cost.Evaluations++
				if score < best && !almostEqual(score, best) {
					bestCode, best = candidate, score
				}
			}
			if bestCode != codes[i] {
				codes[i], current = bestCode, best
				cost.AcceptedUpdates++
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	m := Metrics{InitialMSE: initial, FinalMSE: current, AbsoluteMSEGain: initial - current}
	if initial > 0 {
		m.RelativeMSEGain = (initial - current) / initial
	}
	out.Outcome, out.Reason = OutcomeSupported, ReasonEvaluated
	out.Grid = append([]float64(nil), req.Grid...)
	out.InitialCodes = append([]int(nil), req.InitialCodes...)
	out.RefinedCodes = codes
	q := qualityMetrics(req, req.InitialCodes, codes)
	out.Metrics, out.Quality, out.ConversionCost = &m, &q, &cost
	out.Evidence = EvidenceModeled
	out.HardwareEnvelope = "none: deterministic objective evaluation; no wall-time, throughput, GPU, or model-quality measurement"
	out.ClaimCheck = ClaimCheck{Verdict: "not-yet", Basis: "fixture MSE is measured by the reference evaluator, but downstream model quality and hardware performance are not observed"}
	return out
}

func baseResult(req Request) Result {
	return Result{ContractVersion: req.ContractVersion, RecipeVersion: req.RecipeVersion, RuntimeVersion: req.RuntimeVersion, PaperID: PaperID, PaperSHA256: PaperSHA256, Seed: req.Seed, Evidence: EvidenceModeled, Provenance: req.Provenance, ClaimCheck: ClaimCheck{Verdict: "not-yet", Basis: "no supported evaluation completed"}}
}

func validateGrid(grid []float64, codes []int) error {
	if len(grid) < 2 || len(grid) > 256 {
		return errors.New("grid must contain 2..256 levels")
	}
	for i, v := range grid {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("grid levels must be finite")
		}
		if i > 0 && !(v > grid[i-1]) {
			return errors.New("grid levels must be strictly increasing")
		}
	}
	for _, c := range codes {
		if c < 0 || c >= len(grid) {
			return errors.New("initial code is outside grid")
		}
	}
	return nil
}

func validateShape(req Request) error {
	n := len(req.InitialCodes)
	if n == 0 || n > 4096 || len(req.Target) != n || len(req.Hessian) != n {
		return fmt.Errorf("codes, target and square Hessian must have equal non-zero size up to 4096")
	}
	if req.MaxSweeps < 1 || req.MaxSweeps > 1000 {
		return errors.New("max_sweeps must be 1..1000")
	}
	for _, v := range req.Target {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("target must be finite")
		}
	}
	for i, row := range req.Hessian {
		if len(row) != n {
			return errors.New("Hessian must be square")
		}
		for j, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return errors.New("Hessian must be finite")
			}
			if math.Abs(v-req.Hessian[j][i]) > 1e-12 {
				return errors.New("Hessian must be symmetric")
			}
		}
		if row[i] < 0 {
			return errors.New("Hessian diagonal must be non-negative")
		}
	}
	return nil
}

func validateQuality(q QualityProbe, width int) error {
	if q.ID == "" || q.Version == "" || len(q.Inputs) == 0 || len(q.Inputs) != len(q.Expected) {
		return errors.New("quality probe must pin an ID/version and equal non-zero inputs/expected")
	}
	for _, row := range q.Inputs {
		if len(row) != width {
			return errors.New("quality input width must equal weight count")
		}
		for _, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return errors.New("quality inputs must be finite")
			}
		}
	}
	for _, v := range q.Expected {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("quality expected outputs must be finite")
		}
	}
	return nil
}

func qualityMetrics(req Request, initial, refined []int) QualityMetrics {
	score := func(codes []int) float64 {
		var sum float64
		for i, row := range req.Quality.Inputs {
			var prediction float64
			for j, x := range row {
				prediction += x * req.Grid[codes[j]]
			}
			d := prediction - req.Quality.Expected[i]
			sum += d * d
		}
		return sum / float64(len(req.Quality.Inputs))
	}
	a, b := score(initial), score(refined)
	return QualityMetrics{Metric: "synthetic-linear-prediction-mse", Examples: len(req.Quality.Inputs), InitialPredictionMSE: a, FinalPredictionMSE: b, AbsoluteGain: a - b}
}

func objective(req Request, codes []int) float64 {
	e := make([]float64, len(codes))
	for i, c := range codes {
		e[i] = req.Grid[c] - req.Target[i]
	}
	var sum float64
	for i := range e {
		for j := range e {
			sum += e[i] * req.Hessian[i][j] * e[j]
		}
	}
	return sum / float64(len(e))
}
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 1e-15*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
