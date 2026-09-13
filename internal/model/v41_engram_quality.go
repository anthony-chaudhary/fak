package model

// v41_engram_quality.go — an INDEPENDENT quality gate for QUANTIZED Engram
// n-gram tables (issue #12969, parent track #12640). The published
// DeepSeek-V4.1-Flash checkpoint's two Engram embedding tables are the largest
// single quantizable share (~203 GB, DeepSeekV41EngramBytes); candidate
// quantized-Engram GGUF artifacts (antirez calibrated-Q2, vcruz Q2_K, apetersson
// MixedQ2) already exist, but none carries a behavioral or quality evaluation of
// the quantized tables themselves. #12932's re-quant admission guard is a
// metadata-only per-tensor floor and is NOT this measurement.
//
// This file is the measurement. It takes a QUANTIZED Engram artifact (a
// rounded/dequantized 32x32-block FP8-style table) plus an INDEPENDENT reference
// table, resolves rows at BLOCK level, and returns a typed verdict with three
// independent rungs:
//
//   - NMSE: block-level normalized mean-squared error of the quantized rows
//     against the independent reference, bounded at engram < 0.06.
//   - Held-out perplexity: a small n-gram language model scores a held-out
//     token split whose overlap with the quantization-calibration set is ZERO
//     tokens; the gate reports the held-out perplexity.
//   - Beat-random: a shuffled-table control must produce a strictly worse
//     held-out perplexity, so an uninformative table cannot pass on a lucky
//     split.
//
// A quantized artifact that lacks a passing quality receipt fails closed at
// admission with a typed missing-quality-evidence error; the native FP8 Engram
// path remains the default, so trunk stays green and the gate cannot silently
// admit an unmeasured table.
//
// Gold-plating boundary: no serving-throughput or latency claim, no forward-path
// assembly (#12901), no pruning, and no in-place re-quantization of the
// checkpoint. Nothing here downloads or commits weights; every number is
// [SW-VERIFIED] on synthetic fixtures.

import (
	"errors"
	"fmt"
	"math"
)

// V41EngramQualityNMSECeiling is the published block-level NMSE ceiling for a
// quantized Engram table. It is the same 0.06 bound the ggufload quant-integrity
// validator applies to its Engram parts (quantIntegrityEngramNMSECeiling), so an
// artifact built there and measured here uses one shared ceiling.
const V41EngramQualityNMSECeiling = 0.06

// V41EngramQuantTable is a quantized Engram table. Each row is a 32x32-block
// E4M3 + raw E8M0 scale payload exactly like the V4.1 dense FP8 family (see
// decodeV41DenseFP8); rows are addressed by index.
type V41EngramQuantTable struct {
	RowCount int
	RowWidth int

	// Weight is the packed E4M3 payload: RowCount*RowWidth bytes.
	Weight []byte
	// Scales is one E8M0 scale byte per 32x32 tile: ceil(RowCount/32)*ceil(RowWidth/32)
	// bytes, laid out row-major over tiles.
	Scales []byte
}

// ErrV41EngramQuality is the base class for quality-gate refusals.
var ErrV41EngramQuality = errors.New("model: DeepSeek V4.1 Engram quality gate refused")

// V41EngramMissingQualityError is the typed fail-closed refusal returned when a
// quantized Engram artifact is presented for admission WITHOUT a passing quality
// receipt. It names the artifact and the missing rung so the caller cannot admit
// an unmeasured table by accident.
type V41EngramMissingQualityError struct {
	Artifact string
	Missing  string
}

func (e *V41EngramMissingQualityError) Error() string {
	return fmt.Sprintf("%s: quantized Engram artifact %q has no quality receipt (%s); the native FP8 Engram path remains the default",
		ErrV41EngramQuality, e.Artifact, e.Missing)
}

func (e *V41EngramMissingQualityError) Unwrap() error { return ErrV41EngramQuality }

// V41EngramQualityVerdict is the typed result of the gate. Pass is the single
// closed bit; every other field is the evidence behind it.
type V41EngramQualityVerdict struct {
	Artifact string `json:"artifact"`

	BlockNMSE         float64 `json:"block_nmse"`
	NMSECeiling       float64 `json:"nmse_ceiling"`
	NMSEWithinCeiling bool    `json:"nmse_within_ceiling"`

	HeldOutPerplexity       float64 `json:"held_out_perplexity"`
	RandomControlPerplexity float64 `json:"random_control_perplexity"`

	CalibrationTokens int `json:"calibration_tokens"`
	HeldOutTokens     int `json:"held_out_tokens"`
	OverlapTokens     int `json:"overlap_tokens"`

	Pass   bool   `json:"pass"`
	Reason string `json:"reason,omitempty"`
}

// decodeV41EngramQuantRow dequantizes one row of a quantized Engram table into
// fp32 using the shared V4.1 dense-FP8 32x32-block semantics: E4M3 payload, one
// raw E8M0 scale byte per 32x32 tile, scale = 2^(byte-127). It validates geometry
// and bytes and refuses E4M3 NaN/Inf and the E8M0 0xff NaN byte before returning.
func decodeV41EngramQuantRow(table V41EngramQuantTable, row int) ([]float32, error) {
	if table.RowCount <= 0 || table.RowWidth <= 0 {
		return nil, fmt.Errorf("%w: invalid quantized Engram geometry rows=%d width=%d",
			ErrV41EngramQuality, table.RowCount, table.RowWidth)
	}
	if row < 0 || row >= table.RowCount {
		return nil, fmt.Errorf("%w: quantized Engram row %d out of range [0,%d)",
			ErrV41EngramQuality, row, table.RowCount)
	}
	wantWeights, ok := checkedShapeProduct(table.RowCount, table.RowWidth)
	if !ok || len(table.Weight) != wantWeights {
		return nil, fmt.Errorf("%w: quantized Engram weight has %d bytes, geometry %dx%d implies %d",
			ErrV41EngramQuality, len(table.Weight), table.RowCount, table.RowWidth, wantWeights)
	}
	scaleRows := (table.RowCount-1)/v41FP8BlockDim + 1
	scaleCols := (table.RowWidth-1)/v41FP8BlockDim + 1
	wantScales, ok := checkedShapeProduct(scaleRows, scaleCols)
	if !ok || len(table.Scales) != wantScales {
		return nil, fmt.Errorf("%w: quantized Engram scales have %d bytes, want %d for geometry %dx%d",
			ErrV41EngramQuality, len(table.Scales), wantScales, table.RowCount, table.RowWidth)
	}

	scaleRow := (row / v41FP8BlockDim) * scaleCols
	out := make([]float32, table.RowWidth)
	for col := 0; col < table.RowWidth; col++ {
		scaleByte := table.Scales[scaleRow+col/v41FP8BlockDim]
		if scaleByte == 0xff {
			return nil, fmt.Errorf("%w: quantized Engram scale [%d,%d] is E8M0 NaN",
				ErrV41EngramQuality, row, col)
		}
		raw := table.Weight[row*table.RowWidth+col]
		decoded := fp8E4M3ToF32(raw)
		if math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, fmt.Errorf("%w: quantized Engram weight [%d,%d] is E4M3 NaN",
				ErrV41EngramQuality, row, col)
		}
		value := decoded * float32(math.Ldexp(1, int(scaleByte)-127))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("%w: quantized Engram value [%d,%d] is non-finite",
				ErrV41EngramQuality, row, col)
		}
		out[col] = value
	}
	return out, nil
}

// V41EngramBlockNMSE computes the block-level NMSE of a quantized Engram table
// against an INDEPENDENT reference table: mean over all rows/cols of
// (quantized-reference)^2 divided by mean reference^2. A zero reference energy
// with any error is +Inf (refuse-closed), and non-finite inputs refuse. The
// reference is caller-supplied fp32 and is never derived from the quantized
// table, so the metric is not circular.
func V41EngramBlockNMSE(quant V41EngramQuantTable, reference [][]float32) (float64, error) {
	if len(reference) != quant.RowCount {
		return 0, fmt.Errorf("%w: reference has %d rows, quantized table has %d",
			ErrV41EngramQuality, len(reference), quant.RowCount)
	}
	var errSum, refSum float64
	var n int
	for row := 0; row < quant.RowCount; row++ {
		decoded, err := decodeV41EngramQuantRow(quant, row)
		if err != nil {
			return 0, err
		}
		if len(reference[row]) != quant.RowWidth {
			return 0, fmt.Errorf("%w: reference row %d width %d, want %d",
				ErrV41EngramQuality, row, len(reference[row]), quant.RowWidth)
		}
		for col := 0; col < quant.RowWidth; col++ {
			ref := float64(reference[row][col])
			dif := float64(decoded[col]) - ref
			if math.IsNaN(ref) || math.IsInf(ref, 0) || math.IsNaN(dif) || math.IsInf(dif, 0) {
				return 0, fmt.Errorf("%w: non-finite reference/error at [%d,%d]",
					ErrV41EngramQuality, row, col)
			}
			errSum += dif * dif
			refSum += ref * ref
			n++
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%w: empty Engram table", ErrV41EngramQuality)
	}
	mean := func(sum float64) float64 { return sum / float64(n) }
	meanErr, meanRef := mean(errSum), mean(refSum)
	if meanRef == 0 {
		if meanErr == 0 {
			return 0, nil
		}
		return math.Inf(1), nil
	}
	return meanErr / meanRef, nil
}

// V41EngramQualityOptions configures the gate's held-out evaluation.
//
// Held-out tokens address a set of rows in the quantized Engram table; the
// calibration tokens address a DISJOINT set. The gate enforces zero token
// overlap, so the calibration and held-out row sets never share a row. It fits a
// residual scale (the quantization error standard deviation) on the calibration
// rows and scores the held-out rows under it; the shuffled-row control then
// proves the score is carried by the actual table contents.
type V41EngramQualityOptions struct {
	Artifact string

	// HeldOutTokens and CalibrationTokens are the two token splits; their IDs
	// must be disjoint (zero overlap). Each token ID is a row index into the
	// quantized Engram table.
	HeldOutTokens     []int
	CalibrationTokens []int

	// RandomSeed drives the deterministic row shuffle for the beat-random
	// control.
	RandomSeed uint64
}

// v41EngramResidualScale fits the quantization-error standard deviation from a
// set of rows: sqrt(mean squared error of quantized vs reference). A zero-energy
// fit returns a floor so the Gaussian likelihood stays finite.
func v41EngramResidualScale(quant V41EngramQuantTable, reference [][]float32, rows []int) (float64, int, error) {
	var errSum float64
	var n int
	for _, row := range rows {
		decoded, err := decodeV41EngramQuantRow(quant, row)
		if err != nil {
			return 0, 0, err
		}
		if len(reference[row]) != quant.RowWidth {
			return 0, 0, fmt.Errorf("%w: reference row %d width %d, want %d",
				ErrV41EngramQuality, row, len(reference[row]), quant.RowWidth)
		}
		for col := 0; col < quant.RowWidth; col++ {
			dif := float64(decoded[col]) - float64(reference[row][col])
			if math.IsNaN(dif) || math.IsInf(dif, 0) {
				return 0, 0, fmt.Errorf("%w: non-finite residual at row %d col %d",
					ErrV41EngramQuality, row, col)
			}
			errSum += dif * dif
			n++
		}
	}
	if n == 0 {
		return 0, 0, fmt.Errorf("%w: empty calibration row set", ErrV41EngramQuality)
	}
	scale := math.Sqrt(errSum / float64(n))
	if !(scale > 0) {
		scale = 1e-6
	}
	return scale, n, nil
}

// v41EngramHeldOutPerplexity scores the held-out rows under a Gaussian whose mean
// is each quantized value and whose standard deviation is the calibration-fit
// residual scale: perplexity = exp(mean NLL). A shuffled table mispredicts the
// held-out reference values and scores strictly worse.
func v41EngramHeldOutPerplexity(quant V41EngramQuantTable, reference [][]float32, rows []int, sigma float64) (float64, error) {
	if !(sigma > 0) {
		return 0, fmt.Errorf("%w: residual scale %g must be positive", ErrV41EngramQuality, sigma)
	}
	var logSum float64
	var n int
	for _, row := range rows {
		decoded, err := decodeV41EngramQuantRow(quant, row)
		if err != nil {
			return 0, err
		}
		if len(reference[row]) != quant.RowWidth {
			return 0, fmt.Errorf("%w: reference row %d width %d, want %d",
				ErrV41EngramQuality, row, len(reference[row]), quant.RowWidth)
		}
		for col := 0; col < quant.RowWidth; col++ {
			dif := float64(reference[row][col]) - float64(decoded[col])
			nll := math.Log(sigma) + 0.5*math.Log(2*math.Pi) + (dif*dif)/(2*sigma*sigma)
			if math.IsNaN(nll) || math.IsInf(nll, 0) {
				return 0, fmt.Errorf("%w: non-finite held-out NLL at row %d col %d",
					ErrV41EngramQuality, row, col)
			}
			logSum += nll
			n++
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%w: empty held-out row set", ErrV41EngramQuality)
	}
	return v41EngramClampPerplexity(logSum / float64(n)), nil
}

// v41EngramClampPerplexity maps a mean negative log likelihood to a perplexity,
// rejecting NaN and clamping a positive overflow to the largest finite float so
// the typed verdict stays JSON-serializable while still ranking worse than any
// passing table.
func v41EngramClampPerplexity(meanNLL float64) float64 {
	perp := math.Exp(meanNLL)
	if math.IsNaN(perp) {
		return math.MaxFloat64
	}
	if math.IsInf(perp, 1) {
		return math.MaxFloat64
	}
	return perp
}

// EvaluateV41EngramQuality runs the full gate: decode the quantized table,
// compute block NMSE against an independent reference, fit the residual scale on
// the calibration rows, score held-out perplexity on the disjoint held-out rows,
// and prove a shuffled-row control is strictly worse. It fails closed on any
// missing or malformed rung and NEVER passes an unmeasured table.
func EvaluateV41EngramQuality(
	quant V41EngramQuantTable,
	reference [][]float32,
	opts V41EngramQualityOptions,
) (V41EngramQualityVerdict, error) {
	verdict := V41EngramQualityVerdict{
		Artifact:          opts.Artifact,
		NMSECeiling:       V41EngramQualityNMSECeiling,
		CalibrationTokens: len(opts.CalibrationTokens),
		HeldOutTokens:     len(opts.HeldOutTokens),
	}

	if len(opts.HeldOutTokens) == 0 {
		return verdict, &V41EngramMissingQualityError{Artifact: opts.Artifact, Missing: "held-out token split"}
	}
	if len(opts.CalibrationTokens) == 0 {
		return verdict, &V41EngramMissingQualityError{Artifact: opts.Artifact, Missing: "calibration token split"}
	}

	overlap := v41EngramTokenOverlap(opts.CalibrationTokens, opts.HeldOutTokens)
	verdict.OverlapTokens = overlap
	if overlap != 0 {
		verdict.Reason = fmt.Sprintf("held-out split overlaps the calibration set by %d tokens", overlap)
		return verdict, fmt.Errorf("%w: %s", ErrV41EngramQuality, verdict.Reason)
	}

	nmse, err := V41EngramBlockNMSE(quant, reference)
	if err != nil {
		return verdict, err
	}
	verdict.BlockNMSE = nmse
	verdict.NMSEWithinCeiling = nmse <= V41EngramQualityNMSECeiling

	sigma, _, err := v41EngramResidualScale(quant, reference, opts.CalibrationTokens)
	if err != nil {
		return verdict, err
	}
	heldOut, err := v41EngramHeldOutPerplexity(quant, reference, opts.HeldOutTokens, sigma)
	if err != nil {
		return verdict, err
	}
	verdict.HeldOutPerplexity = heldOut

	// Beat-random control: score each held-out reference row against a
	// deterministic permutation of the held-out rows' own quantized values. The
	// distributions match, so a table whose contents carry no held-out signal
	// cannot beat the permutation; only real row alignment separates them.
	randomPerplexity, err := v41EngramHeldOutPerplexityShuffled(
		quant, reference, opts.HeldOutTokens, sigma, opts.RandomSeed)
	if err != nil {
		return verdict, err
	}
	verdict.RandomControlPerplexity = randomPerplexity

	switch {
	case !verdict.NMSEWithinCeiling:
		verdict.Reason = fmt.Sprintf("block NMSE %.6g exceeds ceiling %.6g", nmse, V41EngramQualityNMSECeiling)
	case !(heldOut < randomPerplexity):
		verdict.Reason = fmt.Sprintf("held-out perplexity %.6g does not beat the random control %.6g", heldOut, randomPerplexity)
	default:
		verdict.Pass = true
		verdict.Reason = fmt.Sprintf("NMSE %.6g <= %.6g; held-out PPL %.6g < random PPL %.6g",
			nmse, V41EngramQualityNMSECeiling, heldOut, randomPerplexity)
	}
	return verdict, nil
}

// v41EngramHeldOutPerplexityShuffled is the beat-random control. It pairs each
// held-out reference row with a DIFFERENT held-out row's quantized values
// (a deterministic derangement/permutation of the held-out set), then scores the
// same Gaussian likelihood. Because the held-out values are drawn from one
// distribution, a table that genuinely aligns with its own reference rows beats
// the permutation; a table that does not cannot. With fewer than two held-out
// rows there is nothing to permute, so the control equals the real score and the
// gate will not pass on that degenerate shape.
func v41EngramHeldOutPerplexityShuffled(
	quant V41EngramQuantTable,
	reference [][]float32,
	rows []int,
	sigma float64,
	seed uint64,
) (float64, error) {
	if !(sigma > 0) {
		return 0, fmt.Errorf("%w: residual scale %g must be positive", ErrV41EngramQuality, sigma)
	}
	if len(rows) < 2 {
		return 0, fmt.Errorf("%w: beat-random control needs at least two held-out rows", ErrV41EngramQuality)
	}
	perm := v41EngramPermutation(len(rows), seed)
	var logSum float64
	var n int
	for i, row := range rows {
		other := rows[perm[i]]
		decoded, err := decodeV41EngramQuantRow(quant, other)
		if err != nil {
			return 0, err
		}
		if len(reference[row]) != quant.RowWidth {
			return 0, fmt.Errorf("%w: reference row %d width %d, want %d",
				ErrV41EngramQuality, row, len(reference[row]), quant.RowWidth)
		}
		for col := 0; col < quant.RowWidth; col++ {
			dif := float64(reference[row][col]) - float64(decoded[col])
			nll := math.Log(sigma) + 0.5*math.Log(2*math.Pi) + (dif*dif)/(2*sigma*sigma)
			if math.IsNaN(nll) || math.IsInf(nll, 0) {
				return 0, fmt.Errorf("%w: non-finite shuffled NLL at row %d col %d",
					ErrV41EngramQuality, row, col)
			}
			logSum += nll
			n++
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%w: empty held-out row set", ErrV41EngramQuality)
	}
	return v41EngramClampPerplexity(logSum / float64(n)), nil
}

// v41EngramPermutation returns a deterministic Fisher-Yates permutation of
// [0,n) via an xorshift64 stream. A permutation that happens to be the identity
// is rotated by one so the control is never vacuous for n >= 2.
func v41EngramPermutation(n int, seed uint64) []int {
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	if n < 2 {
		return order
	}
	state := seed
	if state == 0 {
		state = 0x9e3779b97f4a7c15
	}
	next := func() uint64 {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		return state
	}
	for i := n - 1; i > 0; i-- {
		j := int(next() % uint64(i+1))
		order[i], order[j] = order[j], order[i]
	}
	identity := true
	for i, v := range order {
		if i != v {
			identity = false
			break
		}
	}
	if identity {
		order = append(order[1:], order[0])
	}
	return order
}

func v41EngramTokenOverlap(a, b []int) int {
	seen := make(map[int]int, len(a))
	for _, tok := range a {
		seen[tok]++
	}
	overlap := 0
	for _, tok := range b {
		if seen[tok] > 0 {
			overlap++
		}
	}
	return overlap
}

// V41EngramQualityReceipt is the durable, serializable evidence a passing gate
// produces. A quantized artifact is admitted only when its receipt has Pass=true
// for the artifact identity it names.
type V41EngramQualityReceipt struct {
	Artifact string                  `json:"artifact"`
	Verdict  V41EngramQualityVerdict `json:"verdict"`
}

// AdmitV41EngramQuantizedArtifact is the fail-closed admission guard: a
// quantized Engram artifact with no passing receipt is refused with a typed
// V41EngramMissingQualityError. A nil receipt is always refused.
func AdmitV41EngramQuantizedArtifact(artifact string, receipt *V41EngramQualityReceipt) error {
	if receipt == nil {
		return &V41EngramMissingQualityError{Artifact: artifact, Missing: "no quality receipt"}
	}
	if receipt.Artifact == "" || receipt.Artifact != artifact {
		return &V41EngramMissingQualityError{Artifact: artifact,
			Missing: fmt.Sprintf("receipt names %q", receipt.Artifact)}
	}
	if !receipt.Verdict.Pass {
		return &V41EngramMissingQualityError{Artifact: artifact,
			Missing: "receipt verdict did not pass: " + receipt.Verdict.Reason}
	}
	return nil
}
