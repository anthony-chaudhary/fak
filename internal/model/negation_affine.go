package model

import (
	"fmt"
	"math"
	"sort"
)

// NegationAffinePair is one residual-space affirmative/negated pair at a candidate layer.
type NegationAffinePair struct {
	Concept  string    `json:"concept"`
	Split    string    `json:"split"`
	Affirmed []float64 `json:"affirmed"`
	Negated  []float64 `json:"negated"`
}

// NegationAffineLayer groups paired residuals captured at one transformer layer.
type NegationAffineLayer struct {
	Layer int                  `json:"layer"`
	Pairs []NegationAffinePair `json:"pairs"`
}

// NegationAffineOperator contains both fitted candidates. ApplyAffine is the general
// transform; ApplySteering is the single-vector baseline.
type NegationAffineOperator struct {
	Layer    int         `json:"layer"`
	Steering []float64   `json:"steering"`
	Matrix   [][]float64 `json:"matrix"`
	Bias     []float64   `json:"bias"`
}

// NegationAffineLayerResult records held-out reconstruction and behavioral effects.
type NegationAffineLayerResult struct {
	Layer                int     `json:"layer"`
	SteeringRMSE         float64 `json:"steering_rmse"`
	AffineRMSE           float64 `json:"affine_rmse"`
	UnpatchedTargetScore float64 `json:"unpatched_target_score"`
	PatchedTargetScore   float64 `json:"patched_target_score"`
	RandomTargetScore    float64 `json:"random_target_score"`
	PatchEffect          float64 `json:"patch_effect"`
	RandomEffect         float64 `json:"random_effect"`
}

// NegationAffineSweepReport names the layer selected strictly on held-out affine RMSE.
type NegationAffineSweepReport struct {
	BestLayer  int                         `json:"best_layer"`
	BestEffect float64                     `json:"best_effect"`
	Layers     []NegationAffineLayerResult `json:"layers"`
}

// FitNegationAffine fits a mean steering displacement and an affine least-squares map.
func FitNegationAffine(layer int, pairs []NegationAffinePair) (NegationAffineOperator, error) {
	if layer < 0 {
		return NegationAffineOperator{}, fmt.Errorf("negation affine layer must be non-negative")
	}
	train, dim, err := validateNegationAffinePairs(pairs, "train")
	if err != nil {
		return NegationAffineOperator{}, err
	}
	if len(train) < dim+1 {
		return NegationAffineOperator{}, fmt.Errorf("affine fit needs at least %d training pairs, got %d", dim+1, len(train))
	}
	op := NegationAffineOperator{Layer: layer, Steering: make([]float64, dim), Matrix: make([][]float64, dim), Bias: make([]float64, dim)}
	for _, p := range train {
		for j := range op.Steering {
			op.Steering[j] += (p.Negated[j] - p.Affirmed[j]) / float64(len(train))
		}
	}
	// Solve each output coordinate against [h,1] using normal equations. The tiny
	// diagonal term only stabilizes degenerate fixtures; it does not encode a direction.
	width := dim + 1
	gram := make([][]float64, width)
	for i := range gram {
		gram[i] = make([]float64, width)
	}
	rhs := make([][]float64, dim)
	for j := range rhs {
		rhs[j] = make([]float64, width)
	}
	for _, p := range train {
		x := append(append([]float64(nil), p.Affirmed...), 1)
		for i := range x {
			for k := range x {
				gram[i][k] += x[i] * x[k]
			}
			for out := 0; out < dim; out++ {
				rhs[out][i] += x[i] * p.Negated[out]
			}
		}
	}
	for i := range gram {
		gram[i][i] += 1e-10
	}
	for out := 0; out < dim; out++ {
		coef, err := solveLinear(gram, rhs[out])
		if err != nil {
			return NegationAffineOperator{}, err
		}
		op.Matrix[out] = append([]float64(nil), coef[:dim]...)
		op.Bias[out] = coef[dim]
	}
	return op, nil
}

func (o NegationAffineOperator) ApplySteering(hidden []float64) ([]float64, error) {
	if len(hidden) != len(o.Steering) {
		return nil, fmt.Errorf("steering shape mismatch: hidden=%d operator=%d", len(hidden), len(o.Steering))
	}
	out := append([]float64(nil), hidden...)
	for i := range out {
		out[i] += o.Steering[i]
	}
	return out, nil
}
func (o NegationAffineOperator) ApplyAffine(hidden []float64) ([]float64, error) {
	if len(hidden) != len(o.Bias) {
		return nil, fmt.Errorf("affine shape mismatch: hidden=%d operator=%d", len(hidden), len(o.Bias))
	}
	out := append([]float64(nil), o.Bias...)
	for i := range out {
		if len(o.Matrix[i]) != len(hidden) {
			return nil, fmt.Errorf("affine matrix row %d shape mismatch", i)
		}
		for j, v := range hidden {
			out[i] += o.Matrix[i][j] * v
		}
	}
	return out, nil
}

// Hook returns a residual hook that applies this operator only at its fitted layer.
// Shape or operator errors panic because they indicate invalid forward-pass setup.
func (o NegationAffineOperator) Hook() ResidualHook {
	return func(layer int, hidden []float32) {
		if layer != o.Layer {
			return
		}
		x := make([]float64, len(hidden))
		for i := range hidden {
			x[i] = float64(hidden[i])
		}
		y, err := o.ApplyAffine(x)
		if err != nil {
			panic(err)
		}
		for i := range y {
			hidden[i] = float32(y[i])
		}
	}
}

// PatchActivation applies the affine map through ActivationPatch's owned capture/inject seam.
func (o NegationAffineOperator) PatchActivation(p *ActivationPatch, hidden []float32) error {
	if p == nil || p.Layer() != o.Layer {
		return fmt.Errorf("activation patch layer does not match operator layer %d", o.Layer)
	}
	x := make([]float64, len(hidden))
	for i := range hidden {
		x[i] = float64(hidden[i])
	}
	y, err := o.ApplyAffine(x)
	if err != nil {
		return err
	}
	patched := make([]float32, len(y))
	for i := range y {
		patched[i] = float32(y[i])
	}
	p.Capture(o.Layer, patched)
	_, err = p.Inject(o.Layer, hidden)
	return err
}

// SweepNegationAffine fits every candidate and chooses the best held-out layer.
func SweepNegationAffine(layers []NegationAffineLayer) (NegationAffineSweepReport, error) {
	if len(layers) == 0 {
		return NegationAffineSweepReport{}, fmt.Errorf("negation affine sweep is empty")
	}
	r := NegationAffineSweepReport{BestLayer: -1}
	best := math.Inf(1)
	for _, layer := range layers {
		op, err := FitNegationAffine(layer.Layer, layer.Pairs)
		if err != nil {
			return r, fmt.Errorf("layer %d: %w", layer.Layer, err)
		}
		test, _, err := validateNegationAffinePairs(layer.Pairs, "test")
		if err != nil {
			return r, fmt.Errorf("layer %d: %w", layer.Layer, err)
		}
		var seS, seA, seBase, seRandom float64
		var n int
		control := orthogonalControl(op.Steering)
		for _, p := range test {
			s, _ := op.ApplySteering(p.Affirmed)
			a, _ := op.ApplyAffine(p.Affirmed)
			rand := append([]float64(nil), p.Affirmed...)
			for i := range rand {
				rand[i] += control[i]
			}
			for j := range p.Negated {
				seS += sq(s[j] - p.Negated[j])
				seA += sq(a[j] - p.Negated[j])
				seBase += sq(p.Affirmed[j] - p.Negated[j])
				seRandom += sq(rand[j] - p.Negated[j])
				n++
			}
		}
		rmS, rmA, rmB, rmR := math.Sqrt(seS/float64(n)), math.Sqrt(seA/float64(n)), math.Sqrt(seBase/float64(n)), math.Sqrt(seRandom/float64(n))
		row := NegationAffineLayerResult{Layer: layer.Layer, SteeringRMSE: rmS, AffineRMSE: rmA, UnpatchedTargetScore: scoreRMSE(rmB), PatchedTargetScore: scoreRMSE(rmA), RandomTargetScore: scoreRMSE(rmR)}
		row.PatchEffect = row.PatchedTargetScore - row.UnpatchedTargetScore
		row.RandomEffect = row.RandomTargetScore - row.UnpatchedTargetScore
		r.Layers = append(r.Layers, row)
		if rmA < best {
			best = rmA
			r.BestLayer = layer.Layer
			r.BestEffect = row.PatchEffect
		}
	}
	sort.Slice(r.Layers, func(i, j int) bool { return r.Layers[i].Layer < r.Layers[j].Layer })
	return r, nil
}

func validateNegationAffinePairs(pairs []NegationAffinePair, split string) ([]NegationAffinePair, int, error) {
	var selected []NegationAffinePair
	dim := 0
	for _, p := range pairs {
		if p.Split != split {
			continue
		}
		if dim == 0 {
			dim = len(p.Affirmed)
		}
		if dim == 0 || len(p.Affirmed) != dim || len(p.Negated) != dim {
			return nil, 0, fmt.Errorf("%s pair %q has inconsistent shape", split, p.Concept)
		}
		selected = append(selected, p)
	}
	if len(selected) == 0 {
		return nil, 0, fmt.Errorf("no %s pairs", split)
	}
	return selected, dim, nil
}
func solveLinear(a [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	m := make([][]float64, n)
	for i := range m {
		m[i] = append(append([]float64(nil), a[i]...), b[i])
	}
	for col := 0; col < n; col++ {
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(m[row][col]) > math.Abs(m[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(m[pivot][col]) < 1e-12 {
			return nil, fmt.Errorf("singular affine fit")
		}
		m[col], m[pivot] = m[pivot], m[col]
		d := m[col][col]
		for j := col; j <= n; j++ {
			m[col][j] /= d
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			f := m[row][col]
			for j := col; j <= n; j++ {
				m[row][j] -= f * m[col][j]
			}
		}
	}
	x := make([]float64, n)
	for i := range x {
		x[i] = m[i][n]
	}
	return x, nil
}
func orthogonalControl(v []float64) []float64 {
	out := make([]float64, len(v))
	if len(v) == 1 {
		out[0] = -v[0]
		return out
	}
	out[0] = -v[1]
	out[1] = v[0]
	for i := 2; i < len(v); i++ {
		out[i] = -v[i]
	}
	return out
}
func scoreRMSE(x float64) float64 { return 1 / (1 + x) }
func sq(x float64) float64        { return x * x }
