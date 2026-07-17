package model

import (
	"fmt"
	"math"
	"sort"
)

// NegationPair is one affirmed/negated residual-activation contrast.
type NegationPair struct {
	Concept  string    `json:"concept"`
	Affirmed []float64 `json:"affirmed"`
	Negated  []float64 `json:"negated"`
	Split    string    `json:"split"`
}

// NegationGeometryReport records cross-concept direction consistency and held-out
// reconstruction. SharedMap is the affine map h -> h + MeanDisplacement learned on train.
type NegationGeometryReport struct {
	MeanCosine                float64   `json:"mean_cosine"`
	TopSingularValueFraction  float64   `json:"top_singular_value_fraction"`
	SharedMapHeldOutRMSE      float64   `json:"shared_map_held_out_rmse"`
	PerConceptBaselineRMSE    float64   `json:"per_concept_baseline_rmse"`
	MeanDisplacement          []float64 `json:"mean_displacement"`
	Verdict                   string    `json:"verdict"`
	CosineThreshold           float64   `json:"cosine_threshold"`
	SingularFractionThreshold float64   `json:"singular_fraction_threshold"`
}

// MeasureNegationGeometry fits a shared affine negation map on train concepts and evaluates
// unseen concepts. The per-concept baseline cannot transfer a memorized displacement to an
// unseen concept, so its held-out prediction is the affirmed activation (identity).
func MeasureNegationGeometry(pairs []NegationPair, cosineThreshold, singularThreshold float64) (NegationGeometryReport, error) {
	if cosineThreshold <= -1 || cosineThreshold > 1 || singularThreshold <= 0 || singularThreshold > 1 {
		return NegationGeometryReport{}, fmt.Errorf("invalid negation geometry thresholds")
	}
	ordered := append([]NegationPair(nil), pairs...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Split == ordered[j].Split {
			return ordered[i].Concept < ordered[j].Concept
		}
		return ordered[i].Split < ordered[j].Split
	})
	var train [][]float64
	dim := 0
	for _, p := range ordered {
		if p.Concept == "" || len(p.Affirmed) == 0 || len(p.Affirmed) != len(p.Negated) {
			return NegationGeometryReport{}, fmt.Errorf("invalid pair %q", p.Concept)
		}
		if dim == 0 {
			dim = len(p.Affirmed)
		}
		if len(p.Affirmed) != dim {
			return NegationGeometryReport{}, fmt.Errorf("pair %s dimension drift", p.Concept)
		}
		if p.Split == "train" {
			train = append(train, displacement(p))
		}
	}
	if len(train) < 2 {
		return NegationGeometryReport{}, fmt.Errorf("need at least two train concepts")
	}
	mean := make([]float64, dim)
	for _, d := range train {
		for i, v := range d {
			mean[i] += v
		}
	}
	for i := range mean {
		mean[i] /= float64(len(train))
	}
	meanCos := meanPairwiseCosine(train)
	topFraction := topEnergyFraction(train)
	var sharedSSE, baselineSSE float64
	var heldValues int
	for _, p := range ordered {
		if p.Split != "test" {
			continue
		}
		for i := range p.Affirmed {
			sharedErr := p.Affirmed[i] + mean[i] - p.Negated[i]
			baseErr := p.Affirmed[i] - p.Negated[i]
			sharedSSE += sharedErr * sharedErr
			baselineSSE += baseErr * baseErr
			heldValues++
		}
	}
	if heldValues == 0 {
		return NegationGeometryReport{}, fmt.Errorf("missing held-out concepts")
	}
	verdict := "entangled"
	if meanCos >= cosineThreshold && topFraction >= singularThreshold {
		verdict = "shared"
	}
	return NegationGeometryReport{MeanCosine: meanCos, TopSingularValueFraction: topFraction, SharedMapHeldOutRMSE: math.Sqrt(sharedSSE / float64(heldValues)), PerConceptBaselineRMSE: math.Sqrt(baselineSSE / float64(heldValues)), MeanDisplacement: mean, Verdict: verdict, CosineThreshold: cosineThreshold, SingularFractionThreshold: singularThreshold}, nil
}

func displacement(p NegationPair) []float64 {
	d := make([]float64, len(p.Affirmed))
	for i := range d {
		d[i] = p.Negated[i] - p.Affirmed[i]
	}
	return d
}
func negGeomDot(a, b []float64) float64 {
	var v float64
	for i := range a {
		v += a[i] * b[i]
	}
	return v
}
func negGeomNorm(a []float64) float64 { return math.Sqrt(negGeomDot(a, a)) }
func meanPairwiseCosine(ds [][]float64) float64 {
	var sum float64
	var n int
	for i := 0; i < len(ds); i++ {
		for j := i + 1; j < len(ds); j++ {
			den := negGeomNorm(ds[i]) * negGeomNorm(ds[j])
			if den != 0 {
				sum += negGeomDot(ds[i], ds[j]) / den
				n++
			}
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// topEnergyFraction computes sigma_1^2 / sum sigma_i^2 via power iteration on D^T D.
func topEnergyFraction(ds [][]float64) float64 {
	dim := len(ds[0])
	cov := make([][]float64, dim)
	var trace float64
	for i := range cov {
		cov[i] = make([]float64, dim)
	}
	for _, d := range ds {
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				cov[i][j] += d[i] * d[j]
			}
		}
	}
	for i := 0; i < dim; i++ {
		trace += cov[i][i]
	}
	if trace == 0 {
		return 0
	}
	v := make([]float64, dim)
	for i := range v {
		v[i] = 1 / math.Sqrt(float64(dim))
	}
	for iter := 0; iter < 64; iter++ {
		next := make([]float64, dim)
		for i := range next {
			next[i] = negGeomDot(cov[i], v)
		}
		n := negGeomNorm(next)
		if n == 0 {
			return 0
		}
		for i := range v {
			v[i] = next[i] / n
		}
	}
	lambda := negGeomDot(v, mulMatrixVector(cov, v))
	return lambda / trace
}
func mulMatrixVector(m [][]float64, v []float64) []float64 {
	out := make([]float64, len(m))
	for i := range m {
		out[i] = negGeomDot(m[i], v)
	}
	return out
}
