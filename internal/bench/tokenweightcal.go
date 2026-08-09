package bench

// Token-weight calibration (#5778). The scheduler weights shipped in
// tokenprofile.DefaultWeights are explicit illustrative policy inputs (#5771);
// they are not hardware truth. This fold turns a measured service-time ledger
// from one sanctioned compute node into a versioned calibration artifact:
// per-class service rates with confidence provenance, a held-out error bound,
// and a comparison against the *tuned* scalar-total baseline.
//
// The comparison is deliberately not a strawman. The scalar baseline is refit
// by least squares on the same fit split, so it gets its own best scalar before
// it is beaten. A net-true gain claim is refused unless that measured
// alternative and the measured per-decision scheduling overhead are both
// supplied, and unless the ledger declares measured (not synthetic) rows.

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/tokenprofile"
)

const (
	// TokenServiceRowSchema is the one-row-per-observation ledger contract.
	TokenServiceRowSchema = "fak-token-service-sample/1"
	// TokenWeightCalibrationSchema versions the fitted calibration artifact.
	TokenWeightCalibrationSchema = "fak-token-weight-calibration/1"

	// TokenSplitFit rows fit the weights; TokenSplitHoldout rows are never
	// seen by either fit and score the declared error bound.
	TokenSplitFit     = "fit"
	TokenSplitHoldout = "holdout"

	// MeasuredProvenancePrefix marks a row taken from real hardware on a
	// sanctioned compute node. SyntheticProvenancePrefix marks a fixture.
	// A row must declare one or the other: an undeclared provenance is how
	// illustrative inputs quietly become "hardware truth", which is the exact
	// defect #5778 exists to close.
	MeasuredProvenancePrefix  = "measured:"
	SyntheticProvenancePrefix = "synthetic:"

	// minTokenFitRows keeps two residual degrees of freedom over the four
	// fitted parameters, so the reported confidence interval means something.
	minTokenFitRows     = 6
	minTokenHoldoutRows = 3
)

// TokenServiceRow is one measured service-time observation for one controlled
// token shape. The three token counts are the classes #5771 separates: prefill
// (uncached input), cache read/transfer (cached input), and decode.
type TokenServiceRow struct {
	Schema              string  `json:"schema"`
	ShapeID             string  `json:"shape_id"`
	Split               string  `json:"split"`
	UncachedInputTokens int64   `json:"uncached_input_tokens"`
	CachedInputTokens   int64   `json:"cached_input_tokens"`
	DecodeTokens        int64   `json:"decode_tokens"`
	ServiceMS           float64 `json:"service_ms"`
	Model               string  `json:"model"`
	Engine              string  `json:"engine"`
	Hardware            string  `json:"hardware"`
	BatchSize           int     `json:"batch_size"`
	Repetitions         int     `json:"repetitions"`
	Provenance          string  `json:"provenance"`
}

// TokenWeightProvenance binds a calibration to the one configuration it was
// measured on. Weights do not pool across model, engine, hardware, or batch.
type TokenWeightProvenance struct {
	Model          string `json:"model"`
	Engine         string `json:"engine"`
	Hardware       string `json:"hardware"`
	BatchSize      int    `json:"batch_size"`
	MinRepetitions int    `json:"min_repetitions"`
	FitSamples     int    `json:"fit_samples"`
	HoldoutSamples int    `json:"holdout_samples"`
	HardwareTruth  bool   `json:"hardware_truth"`
	Sources        string `json:"sources"`
}

// TokenWeightRate is one fitted per-token service rate. Identified is false
// when the 95% interval does not exclude zero — that is a design-matrix
// failure (the token shapes did not vary this class enough), not a weight.
type TokenWeightRate struct {
	MSPerToken    float64 `json:"ms_per_token"`
	CI95HalfWidth float64 `json:"ci95_half_width_ms_per_token"`
	Identified    bool    `json:"identified"`
}

// TokenServiceWeights is the fitted service-time model:
// service_ms ≈ fixed + Σ class_tokens × class_rate.
type TokenServiceWeights struct {
	UncachedInput        TokenWeightRate `json:"uncached_input"`
	CachedInput          TokenWeightRate `json:"cached_input"`
	Decode               TokenWeightRate `json:"decode"`
	FixedMS              float64         `json:"fixed_ms"`
	FixedCI95HalfWidthMS float64         `json:"fixed_ci95_half_width_ms"`
}

// TokenWeightConfidence reports how well the fit split is explained.
type TokenWeightConfidence struct {
	FitSamples       int     `json:"fit_samples"`
	DegreesOfFreedom int     `json:"degrees_of_freedom"`
	ResidualStdMS    float64 `json:"residual_std_ms"`
	RSquared         float64 `json:"r_squared"`
	Note             string  `json:"note"`
}

// TokenPredictionError scores one predictor on the held-out split.
type TokenPredictionError struct {
	Samples        int     `json:"samples"`
	MAPEPct        float64 `json:"mape_pct"`
	RMSEMS         float64 `json:"rmse_ms"`
	MaxAbsErrorPct float64 `json:"max_abs_error_pct"`
}

// TokenScalarBaseline is the tuned scalar-total alternative: one rate over
// total tokens plus a fixed term, refit by least squares on the same fit split.
type TokenScalarBaseline struct {
	MSPerTotalToken float64              `json:"ms_per_total_token"`
	FixedMS         float64              `json:"fixed_ms"`
	Holdout         TokenPredictionError `json:"holdout"`
}

// TokenWeightCalibration is the versioned calibration artifact.
type TokenWeightCalibration struct {
	Schema                string                `json:"schema"`
	Provenance            TokenWeightProvenance `json:"provenance"`
	Weights               TokenServiceWeights   `json:"weights"`
	SchedulerWeights      tokenprofile.Weights  `json:"scheduler_weights"`
	Confidence            TokenWeightConfidence `json:"confidence"`
	Holdout               TokenPredictionError  `json:"holdout"`
	Baseline              TokenScalarBaseline   `json:"scalar_total_baseline"`
	DeclaredErrorBoundPct float64               `json:"declared_error_bound_pct"`
	WithinDeclaredBound   bool                  `json:"within_declared_bound"`
	Digest                string                `json:"digest"`
}

// ReadTokenServiceJSONL decodes the auditable one-row-per-observation ledger.
func ReadTokenServiceJSONL(r io.Reader) ([]TokenServiceRow, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []TokenServiceRow
	for line := 1; scanner.Scan(); line++ {
		if len(strings.TrimSpace(scanner.Text())) == 0 {
			continue
		}
		var row TokenServiceRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("token service row %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// CalibrateTokenWeights fits per-class service rates on the fit split and
// scores both the fitted model and the tuned scalar-total baseline on the
// held-out split. boundPct is the caller's declared held-out error bound
// (MAPE, percent); the artifact records whether the fit stayed inside it
// rather than adjusting the bound to whatever was achieved.
func CalibrateTokenWeights(rows []TokenServiceRow, boundPct float64) (TokenWeightCalibration, error) {
	if boundPct <= 0 || math.IsNaN(boundPct) || math.IsInf(boundPct, 0) {
		return TokenWeightCalibration{}, errors.New("declared error bound must be a positive percentage")
	}
	prov, fit, holdout, err := validateTokenServiceRows(rows)
	if err != nil {
		return TokenWeightCalibration{}, err
	}
	weights, conf, err := fitTokenServiceWeights(fit)
	if err != nil {
		return TokenWeightCalibration{}, err
	}
	baseRate, baseFixed, err := fitScalarTotal(fit)
	if err != nil {
		return TokenWeightCalibration{}, err
	}

	cal := TokenWeightCalibration{
		Schema:     TokenWeightCalibrationSchema,
		Provenance: prov,
		Weights:    weights,
		Confidence: conf,
		Holdout: scoreTokenPredictor(holdout, func(r TokenServiceRow) float64 {
			return weights.FixedMS +
				float64(r.UncachedInputTokens)*weights.UncachedInput.MSPerToken +
				float64(r.CachedInputTokens)*weights.CachedInput.MSPerToken +
				float64(r.DecodeTokens)*weights.Decode.MSPerToken
		}),
		Baseline: TokenScalarBaseline{
			MSPerTotalToken: baseRate,
			FixedMS:         baseFixed,
			Holdout: scoreTokenPredictor(holdout, func(r TokenServiceRow) float64 {
				return baseFixed + float64(totalTokens(r))*baseRate
			}),
		},
		DeclaredErrorBoundPct: boundPct,
	}
	cal.SchedulerWeights = relativeSchedulerWeights(weights)
	cal.WithinDeclaredBound = cal.Holdout.MAPEPct <= boundPct
	digest, err := tokenCalibrationDigest(cal)
	if err != nil {
		return TokenWeightCalibration{}, err
	}
	cal.Digest = digest
	return cal, nil
}

func validateTokenServiceRows(rows []TokenServiceRow) (TokenWeightProvenance, []TokenServiceRow, []TokenServiceRow, error) {
	var out TokenWeightProvenance
	if len(rows) == 0 {
		return out, nil, nil, errors.New("empty token service ledger")
	}
	var fit, holdout []TokenServiceRow
	measured, synthetic := 0, 0
	minReps := 0
	shapes := make(map[string]string, len(rows))
	for i, row := range rows {
		if row.Schema != TokenServiceRowSchema {
			return out, nil, nil, fmt.Errorf("row %d: want schema %q, got %q", i, TokenServiceRowSchema, row.Schema)
		}
		if row.ShapeID == "" || row.Model == "" || row.Engine == "" || row.Hardware == "" {
			return out, nil, nil, fmt.Errorf("row %d has incomplete calibration identity (shape/model/engine/hardware)", i)
		}
		if row.BatchSize <= 0 || row.Repetitions <= 0 {
			return out, nil, nil, fmt.Errorf("row %d has invalid batch size or repetitions", i)
		}
		if row.UncachedInputTokens < 0 || row.CachedInputTokens < 0 || row.DecodeTokens < 0 {
			return out, nil, nil, fmt.Errorf("row %d has negative token counts", i)
		}
		if totalTokens(row) == 0 {
			return out, nil, nil, fmt.Errorf("row %d carries no tokens in any class", i)
		}
		if row.ServiceMS <= 0 || math.IsNaN(row.ServiceMS) || math.IsInf(row.ServiceMS, 0) {
			return out, nil, nil, fmt.Errorf("row %d has non-positive or non-finite service time", i)
		}
		switch {
		case strings.HasPrefix(row.Provenance, MeasuredProvenancePrefix):
			measured++
		case strings.HasPrefix(row.Provenance, SyntheticProvenancePrefix):
			synthetic++
		default:
			return out, nil, nil, fmt.Errorf("row %d provenance %q must declare %q or %q", i, row.Provenance, MeasuredProvenancePrefix, SyntheticProvenancePrefix)
		}
		if out.Model == "" {
			out.Model, out.Engine, out.Hardware, out.BatchSize = row.Model, row.Engine, row.Hardware, row.BatchSize
			minReps = row.Repetitions
		}
		if row.Model != out.Model || row.Engine != out.Engine || row.Hardware != out.Hardware || row.BatchSize != out.BatchSize {
			return out, nil, nil, fmt.Errorf("row %d mixes configurations: weights do not pool across model/engine/hardware/batch", i)
		}
		if row.Repetitions < minReps {
			minReps = row.Repetitions
		}
		if prior, ok := shapes[row.ShapeID]; ok && prior != row.Split {
			return out, nil, nil, fmt.Errorf("shape %q appears in both splits: held-out rows must not leak into the fit", row.ShapeID)
		}
		shapes[row.ShapeID] = row.Split
		switch row.Split {
		case TokenSplitFit:
			fit = append(fit, row)
		case TokenSplitHoldout:
			holdout = append(holdout, row)
		default:
			return out, nil, nil, fmt.Errorf("row %d has unknown split %q", i, row.Split)
		}
	}
	if len(fit) < minTokenFitRows {
		return out, nil, nil, fmt.Errorf("need at least %d fit rows for a confidence interval, got %d", minTokenFitRows, len(fit))
	}
	if len(holdout) < minTokenHoldoutRows {
		return out, nil, nil, fmt.Errorf("need at least %d held-out rows to score an error bound, got %d", minTokenHoldoutRows, len(holdout))
	}
	if measured > 0 && synthetic > 0 {
		return out, nil, nil, errors.New("ledger mixes measured and synthetic rows: a calibration is one or the other")
	}
	out.MinRepetitions = minReps
	out.FitSamples = len(fit)
	out.HoldoutSamples = len(holdout)
	out.HardwareTruth = measured > 0
	out.Sources = fmt.Sprintf("%d fit + %d holdout rows, %s", len(fit), len(holdout), map[bool]string{true: "measured hardware", false: "synthetic fixture"}[measured > 0])
	return out, fit, holdout, nil
}

func totalTokens(r TokenServiceRow) int64 {
	return r.UncachedInputTokens + r.CachedInputTokens + r.DecodeTokens
}

// fitTokenServiceWeights solves the ordinary-least-squares normal equations for
// service_ms ≈ fixed + u·wu + c·wc + d·wd. A singular normal matrix means the
// token shapes were not controlled — they did not vary the three classes
// independently — and is refused rather than pseudo-inverted into a fake fit.
func fitTokenServiceWeights(fit []TokenServiceRow) (TokenServiceWeights, TokenWeightConfidence, error) {
	var w TokenServiceWeights
	var conf TokenWeightConfidence
	n := len(fit)
	design := make([][4]float64, n)
	y := make([]float64, n)
	for i, r := range fit {
		design[i] = [4]float64{float64(r.UncachedInputTokens), float64(r.CachedInputTokens), float64(r.DecodeTokens), 1}
		y[i] = r.ServiceMS
	}
	var normal [4][4]float64
	var rhs [4]float64
	for i := 0; i < n; i++ {
		for a := 0; a < 4; a++ {
			rhs[a] += design[i][a] * y[i]
			for b := 0; b < 4; b++ {
				normal[a][b] += design[i][a] * design[i][b]
			}
		}
	}
	inv, ok := invert4(normal)
	if !ok {
		return w, conf, errors.New("token shapes do not independently vary prefill, cache-read, and decode: the design is singular and no per-class weight is identifiable")
	}
	var beta [4]float64
	for a := 0; a < 4; a++ {
		for b := 0; b < 4; b++ {
			beta[a] += inv[a][b] * rhs[b]
		}
	}
	dof := n - 4
	rss, tss, meanY := 0.0, 0.0, 0.0
	for _, v := range y {
		meanY += v
	}
	meanY /= float64(n)
	for i := 0; i < n; i++ {
		pred := beta[3]
		for a := 0; a < 3; a++ {
			pred += beta[a] * design[i][a]
		}
		rss += (y[i] - pred) * (y[i] - pred)
		tss += (y[i] - meanY) * (y[i] - meanY)
	}
	sigma2 := rss / float64(dof)
	half := func(idx int) float64 {
		v := sigma2 * inv[idx][idx]
		if v <= 0 {
			return 0
		}
		return 1.96 * math.Sqrt(v)
	}
	rate := func(idx int) TokenWeightRate {
		h := half(idx)
		return TokenWeightRate{MSPerToken: beta[idx], CI95HalfWidth: h, Identified: beta[idx] > 0 && h < beta[idx]}
	}
	w = TokenServiceWeights{
		UncachedInput:        rate(0),
		CachedInput:          rate(1),
		Decode:               rate(2),
		FixedMS:              beta[3],
		FixedCI95HalfWidthMS: half(3),
	}
	r2 := 0.0
	if tss > 0 {
		r2 = 1 - rss/tss
	}
	conf = TokenWeightConfidence{
		FitSamples:       n,
		DegreesOfFreedom: dof,
		ResidualStdMS:    math.Sqrt(sigma2),
		RSquared:         r2,
		Note:             "95% intervals use a normal approximation; Identified=false means the shapes did not vary that class enough to sign its weight",
	}
	return w, conf, nil
}

// fitScalarTotal tunes the alternative before comparing against it: the scalar
// gets its own least-squares rate over total tokens on the same fit split.
func fitScalarTotal(fit []TokenServiceRow) (rate, fixed float64, err error) {
	n := float64(len(fit))
	var sx, sy, sxx, sxy float64
	for _, r := range fit {
		x := float64(totalTokens(r))
		sx += x
		sy += r.ServiceMS
		sxx += x * x
		sxy += x * r.ServiceMS
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0, 0, errors.New("every fit row carries the same total token count: the scalar-total baseline is not tunable")
	}
	rate = (n*sxy - sx*sy) / den
	fixed = (sy - rate*sx) / n
	return rate, fixed, nil
}

func scoreTokenPredictor(holdout []TokenServiceRow, predict func(TokenServiceRow) float64) TokenPredictionError {
	out := TokenPredictionError{Samples: len(holdout)}
	if len(holdout) == 0 {
		return out
	}
	var sumAbsPct, sumSq float64
	for _, r := range holdout {
		diff := predict(r) - r.ServiceMS
		pct := math.Abs(diff) / r.ServiceMS * 100
		sumAbsPct += pct
		sumSq += diff * diff
		if pct > out.MaxAbsErrorPct {
			out.MaxAbsErrorPct = pct
		}
	}
	out.MAPEPct = sumAbsPct / float64(len(holdout))
	out.RMSEMS = math.Sqrt(sumSq / float64(len(holdout)))
	return out
}

// relativeSchedulerWeights renormalizes the measured rates onto the
// tokenprofile.Weights axis, where uncached input is 1 by construction. This is
// the drop-in replacement for the illustrative tokenprofile.DefaultWeights.
func relativeSchedulerWeights(w TokenServiceWeights) tokenprofile.Weights {
	base := w.UncachedInput.MSPerToken
	if base <= 0 {
		return tokenprofile.Weights{}
	}
	nonNeg := func(v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	}
	return tokenprofile.Weights{
		InputUncached: 1,
		InputCached:   nonNeg(w.CachedInput.MSPerToken) / base,
		Output:        nonNeg(w.Decode.MSPerToken) / base,
	}
}

// SchedulingOverhead is the measured per-admission cost of *running* each
// scheduler arm. A calibrated three-class weighting is not free relative to a
// single scalar multiply, and #5778 refuses to call the accuracy win a gain
// until that cost is on the books.
type SchedulingOverhead struct {
	CalibratedDecisionMS float64 `json:"calibrated_decision_ms"`
	BaselineDecisionMS   float64 `json:"baseline_decision_ms"`
	Measured             bool    `json:"measured"`
	Provenance           string  `json:"provenance"`
}

// NetSchedulingValue is the net-true accounting: what better service-time
// prediction buys, minus what computing it costs.
type NetSchedulingValue struct {
	Schema                  string  `json:"schema"`
	BaselineHoldoutRMSEMS   float64 `json:"baseline_holdout_rmse_ms"`
	CalibratedHoldoutRMSEMS float64 `json:"calibrated_holdout_rmse_ms"`
	MispredictionReducedMS  float64 `json:"misprediction_reduced_ms_per_request"`
	AddedOverheadMS         float64 `json:"added_overhead_ms_per_request"`
	NetValueMSPerRequest    float64 `json:"net_value_ms_per_request"`
	NetTrueGain             bool    `json:"net_true_gain"`
	WhyNot                  string  `json:"why_not,omitempty"`
	OverheadProvenance      string  `json:"overhead_provenance"`
	CalibrationDigest       string  `json:"calibration_digest"`
}

// EvaluateNetSchedulingValue refuses a net-true gain claim unless the measured
// alternative and the measured overhead are both included. Refusals are hard
// errors, not a false result: a caller must not be able to read "no gain" where
// the honest answer is "you did not measure enough to say".
func EvaluateNetSchedulingValue(cal TokenWeightCalibration, oh SchedulingOverhead) (NetSchedulingValue, error) {
	if cal.Schema != TokenWeightCalibrationSchema {
		return NetSchedulingValue{}, fmt.Errorf("want calibration schema %q, got %q", TokenWeightCalibrationSchema, cal.Schema)
	}
	if cal.Holdout.Samples == 0 || cal.Baseline.Holdout.Samples == 0 {
		return NetSchedulingValue{}, errors.New("net-true gain refused: the measured alternative was not scored on the held-out split")
	}
	if cal.Baseline.MSPerTotalToken == 0 && cal.Baseline.FixedMS == 0 {
		return NetSchedulingValue{}, errors.New("net-true gain refused: the tuned scalar-total alternative is missing")
	}
	if !oh.Measured || oh.Provenance == "" {
		return NetSchedulingValue{}, errors.New("net-true gain refused: per-decision scheduling overhead is not measured")
	}
	for _, v := range []float64{oh.CalibratedDecisionMS, oh.BaselineDecisionMS} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return NetSchedulingValue{}, errors.New("net-true gain refused: overhead must be finite and non-negative")
		}
	}
	out := NetSchedulingValue{
		Schema:                  "fak-net-scheduling-value/1",
		BaselineHoldoutRMSEMS:   cal.Baseline.Holdout.RMSEMS,
		CalibratedHoldoutRMSEMS: cal.Holdout.RMSEMS,
		MispredictionReducedMS:  cal.Baseline.Holdout.RMSEMS - cal.Holdout.RMSEMS,
		AddedOverheadMS:         oh.CalibratedDecisionMS - oh.BaselineDecisionMS,
		OverheadProvenance:      oh.Provenance,
		CalibrationDigest:       cal.Digest,
	}
	out.NetValueMSPerRequest = out.MispredictionReducedMS - out.AddedOverheadMS
	switch {
	case !cal.Provenance.HardwareTruth:
		out.WhyNot = "calibration is a synthetic fixture, not measured hardware truth"
	case !cal.WithinDeclaredBound:
		out.WhyNot = fmt.Sprintf("held-out MAPE %.2f%% exceeds the declared bound %.2f%%", cal.Holdout.MAPEPct, cal.DeclaredErrorBoundPct)
	case !cal.Weights.UncachedInput.Identified || !cal.Weights.CachedInput.Identified || !cal.Weights.Decode.Identified:
		out.WhyNot = "at least one per-class weight is unidentified by the fit design"
	case out.NetValueMSPerRequest <= 0:
		out.WhyNot = "added scheduling overhead cancels the misprediction reduction"
	default:
		out.NetTrueGain = true
	}
	return out, nil
}

func tokenCalibrationDigest(cal TokenWeightCalibration) (string, error) {
	cal.Digest = ""
	buf, err := json.Marshal(cal)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:]), nil
}

// invert4 inverts a 4x4 matrix by Gauss-Jordan elimination with partial
// pivoting, reporting false when the matrix is singular at a scale-relative
// tolerance.
func invert4(a [4][4]float64) ([4][4]float64, bool) {
	var m [4][8]float64
	scale := 0.0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			m[i][j] = a[i][j]
			if v := math.Abs(a[i][j]); v > scale {
				scale = v
			}
		}
		m[i][4+i] = 1
	}
	if scale == 0 {
		return [4][4]float64{}, false
	}
	tol := 1e-12 * scale
	for col := 0; col < 4; col++ {
		piv := col
		for r := col + 1; r < 4; r++ {
			if math.Abs(m[r][col]) > math.Abs(m[piv][col]) {
				piv = r
			}
		}
		if math.Abs(m[piv][col]) <= tol {
			return [4][4]float64{}, false
		}
		m[col], m[piv] = m[piv], m[col]
		p := m[col][col]
		for j := 0; j < 8; j++ {
			m[col][j] /= p
		}
		for r := 0; r < 4; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			if f == 0 {
				continue
			}
			for j := 0; j < 8; j++ {
				m[r][j] -= f * m[col][j]
			}
		}
	}
	var inv [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[i][j] = m[i][4+j]
		}
	}
	return inv, true
}
