// bitplane_head.go — Bit-plane certified LM-head argmax host oracle.
//
// ADAPTED from magnitudedev/magnitude (Apache-2.0):
//   inference/engine/kernels/kernels/readout.seismic:278@8d7dbcc1ba
//   inference/engine/kernels/kernels/cuda/lib/projection/planes.cuh:64@814f72a
//
// No upstream source bytes are copied; this file provides a clean-room Go
// host oracle and bit-plane encoder modeling the exact certification contract
// paired with the unwired CUDA kernels in sampling_topk.cu:
//   1. Decompose each row of weights into MSB-first bit-planes with affine
//      per-row parameters (min, scale) and exact elementwise residual bound.
//   2. Given query vector x, read ONLY the top `topBits` bit-planes and derive
//      a sound interval [low_r, high_r] guaranteed to contain the exact F32
//      dot-product logit mathx.FDot(W_r, x).
//   3. Certify the argmax margin: let bestLowIdx be the lowest index achieving
//      bestLow = max_r low_r. Any row r != bestLowIdx with high_r < bestLow (or
//      high_r == bestLow && r > bestLowIdx under lowest-index tie-breaking) is
//      provably pruned.
//   4. If zero other rows remain in contention, exit early with bestLowIdx and
//      RefinedRows == 0. Otherwise run exact F32 dot-product refinement ONLY on
//      the uncertified contenders and return bit-exact argmax parity.
//
// Status: [SW-VERIFIED] host oracle only; CUDA kernels in sampling_topk.cu remain
// unwired (token_time: unmeasured / HW-qualification-pending).

package compute

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

var (
	// ErrBitplaneInvalidShape refuses malformed matrix, bit-plane, or query dimensions.
	ErrBitplaneInvalidShape = errors.New("compute: bitplane head: invalid shape or bit configuration")
	// ErrBitplaneNonFinite refuses weights, activations, or scores containing NaN or Inf.
	ErrBitplaneNonFinite = errors.New("compute: bitplane head: non-finite weights, activation, or score")
)

const (
	bitplaneMaxTotalBits = 16
	bitplaneWordBits     = 32
)

// BitplaneHead stores an LM-head projection matrix decomposed into MSB-first
// bit-planes alongside the original F32 rows for selective exact refinement.
type BitplaneHead struct {
	Vocab       int
	HiddenDim   int
	TotalBits   int
	TopBits     int
	WordsPerRow int

	// Planes stores MSB-first bit-planes: plane 0 is bit (TotalBits-1), plane
	// TotalBits-1 is bit 0. Layout: [(row*TotalBits + plane)*WordsPerRow + word].
	Planes []uint32

	RowMin    []float32
	RowScale  []float32
	RowEps    []float32
	RowMaxAbs []float32

	// RawWeights holds the exact [Vocab * HiddenDim] F32 rows used exclusively
	// during fallback refinement of uncertified contenders.
	RawWeights []float32
}

// BitplaneArgmaxResult records the certified or selectively refined argmax
// outcome together with per-row error-bound intervals and pruning counts.
type BitplaneArgmaxResult struct {
	Token       int
	Certified   bool
	PrunedRows  int
	RefinedRows int
	Contenders  []int
	Bounds      [][2]float64
}

func isFiniteF32(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

func float32Up(v float64) float32 {
	if v <= 0 {
		return 0
	}
	f := float32(v)
	if float64(f) < v {
		f = math.Nextafter32(f, float32(math.Inf(1)))
	}
	return f
}

// NewBitplaneHead packs row-major F32 weights [vocab, hiddenDim] into TotalBits
// MSB-first bit-planes with default read depth TopBits, recording exact
// per-row residual error bounds so certification is mathematically sound.
func NewBitplaneHead(weights []float32, vocab, hiddenDim, totalBits, topBits int) (*BitplaneHead, error) {
	if vocab <= 0 || hiddenDim <= 0 || totalBits <= 0 || totalBits > bitplaneMaxTotalBits ||
		topBits <= 0 || topBits > totalBits {
		return nil, fmt.Errorf("%w: vocab=%d hidden=%d totalBits=%d topBits=%d",
			ErrBitplaneInvalidShape, vocab, hiddenDim, totalBits, topBits)
	}
	maxInt := int(^uint(0) >> 1)
	if vocab > maxInt/hiddenDim || len(weights) != vocab*hiddenDim {
		return nil, fmt.Errorf("%w: weights len=%d want %d*%d",
			ErrBitplaneInvalidShape, len(weights), vocab, hiddenDim)
	}
	for i, w := range weights {
		if !isFiniteF32(w) {
			return nil, fmt.Errorf("%w: weight[%d]=%v", ErrBitplaneNonFinite, i, w)
		}
	}

	wordsPerRow := (hiddenDim + bitplaneWordBits - 1) / bitplaneWordBits
	if vocab > maxInt/totalBits || vocab*totalBits > maxInt/wordsPerRow {
		return nil, fmt.Errorf("%w: bitplane buffer overflow", ErrBitplaneInvalidShape)
	}

	h := &BitplaneHead{
		Vocab:       vocab,
		HiddenDim:   hiddenDim,
		TotalBits:   totalBits,
		TopBits:     topBits,
		WordsPerRow: wordsPerRow,
		Planes:      make([]uint32, vocab*totalBits*wordsPerRow),
		RowMin:      make([]float32, vocab),
		RowScale:    make([]float32, vocab),
		RowEps:      make([]float32, vocab),
		RowMaxAbs:   make([]float32, vocab),
		RawWeights:  append([]float32(nil), weights...),
	}

	maxCode := (1 << uint(totalBits)) - 1
	codes := make([]uint32, hiddenDim)

	for r := 0; r < vocab; r++ {
		row := weights[r*hiddenDim : (r+1)*hiddenDim]
		mn, mx := row[0], row[0]
		maxAbs := math.Abs(float64(row[0]))
		for _, w := range row[1:] {
			if w < mn {
				mn = w
			}
			if w > mx {
				mx = w
			}
			if aw := math.Abs(float64(w)); aw > maxAbs {
				maxAbs = aw
			}
		}

		var scale32 float32
		if mx > mn && maxCode > 0 {
			scale32 = float32((float64(mx) - float64(mn)) / float64(maxCode))
		}
		h.RowMin[r] = mn
		h.RowScale[r] = scale32
		h.RowMaxAbs[r] = float32Up(maxAbs)

		var maxErr float64
		for k, w := range row {
			var code int
			if scale32 > 0 {
				norm := (float64(w) - float64(mn)) / float64(scale32)
				code = int(math.Round(norm))
				if code < 0 {
					code = 0
				} else if code > maxCode {
					code = maxCode
				}
			}
			codes[k] = uint32(code)
			recon := float64(mn) + float64(scale32)*float64(code)
			if errAbs := math.Abs(float64(w) - recon); errAbs > maxErr {
				maxErr = errAbs
			}
		}
		h.RowEps[r] = float32Up(maxErr)

		// Scatter codes into MSB-first bit-planes: plane 0 holds bit (totalBits-1).
		for p := 0; p < totalBits; p++ {
			bitShift := uint(totalBits - 1 - p)
			planeBase := (r*totalBits + p) * wordsPerRow
			for k := 0; k < hiddenDim; k++ {
				if (codes[k]>>bitShift)&1 != 0 {
					h.Planes[planeBase+(k>>5)] |= uint32(1) << uint(k&31)
				}
			}
		}
	}

	return h, nil
}

// ReconstructCode reconstructs the full TotalBits integer code for (row, col)
// from the packed bit-planes.
func (h *BitplaneHead) ReconstructCode(row, col int) (uint32, error) {
	return h.ReconstructTopCode(row, col, h.TotalBits)
}

// ReconstructTopCode reads only the top `topBits` MSB planes for (row, col)
// and returns the code aligned to the full TotalBits range (low bits zeroed).
func (h *BitplaneHead) ReconstructTopCode(row, col, topBits int) (uint32, error) {
	if h == nil || row < 0 || row >= h.Vocab || col < 0 || col >= h.HiddenDim ||
		topBits <= 0 || topBits > h.TotalBits {
		return 0, ErrBitplaneInvalidShape
	}
	wordIdx := col >> 5
	bitIdx := uint(col & 31)
	var topCode uint32
	for p := 0; p < topBits; p++ {
		planeBase := (row*h.TotalBits + p) * h.WordsPerRow
		bit := (h.Planes[planeBase+wordIdx] >> bitIdx) & 1
		topCode = (topCode << 1) | bit
	}
	return topCode << uint(h.TotalBits-topBits), nil
}

// DequantWeight reconstructs the full-plane quantized weight value for (row, col).
func (h *BitplaneHead) DequantWeight(row, col int) (float32, error) {
	code, err := h.ReconstructCode(row, col)
	if err != nil {
		return 0, err
	}
	return float32(float64(h.RowMin[row]) + float64(h.RowScale[row])*float64(code)), nil
}

// CertifyAndRefineArgmax runs bit-plane top-bit certification with h.TopBits
// and exact F32 fallback refinement exclusively for contending rows.
func (h *BitplaneHead) CertifyAndRefineArgmax(x []float32) (BitplaneArgmaxResult, error) {
	if h == nil {
		return BitplaneArgmaxResult{}, ErrBitplaneInvalidShape
	}
	return h.CertifyAndRefineArgmaxWithTopBits(x, h.TopBits)
}

// CertifyAndRefineArgmaxWithTopBits runs bit-plane certification using an
// explicit top-plane read depth `topBits` in [1, h.TotalBits].
func (h *BitplaneHead) CertifyAndRefineArgmaxWithTopBits(x []float32, topBits int) (BitplaneArgmaxResult, error) {
	if h == nil || h.Vocab <= 0 || h.HiddenDim <= 0 || len(x) != h.HiddenDim ||
		topBits <= 0 || topBits > h.TotalBits {
		return BitplaneArgmaxResult{}, ErrBitplaneInvalidShape
	}

	var sumX, posSumX, negSumX, absSumX, maxAbsX float64
	for i, xk := range x {
		if !isFiniteF32(xk) {
			return BitplaneArgmaxResult{}, fmt.Errorf("%w: x[%d]=%v", ErrBitplaneNonFinite, i, xk)
		}
		dx := float64(xk)
		sumX += dx
		if dx > 0 {
			posSumX += dx
		} else if dx < 0 {
			negSumX += dx
		}
		ax := math.Abs(dx)
		absSumX += ax
		if ax > maxAbsX {
			maxAbsX = ax
		}
	}

	remBits := uint(h.TotalBits - topBits)
	tailMask := uint32(0)
	if remBits > 0 {
		tailMask = (uint32(1) << remBits) - 1
	}
	maxCode := uint32((1 << uint(h.TotalBits)) - 1)

	nF32 := float64(h.HiddenDim + 2)
	const u32 = 0x1p-24
	gamma32 := 1.0
	if nF32*u32 < 0.5 {
		gamma32 = (nF32 * u32) / (1.0 - nF32*u32)
	}
	const u64 = 0x1p-52

	bounds := make([][2]float64, h.Vocab)
	bestLow := math.Inf(-1)
	bestLowIdx := -1

	// Stage 1: top-bit-plane scan only (topBits of TotalBits planes).
	for r := 0; r < h.Vocab; r++ {
		minVal := float64(h.RowMin[r])
		scale := float64(h.RowScale[r])
		eps := float64(h.RowEps[r])
		maxAbsW := float64(h.RowMaxAbs[r])

		if maxAbsW*maxAbsX*float64(h.HiddenDim) >= math.MaxFloat32*0.25 {
			bounds[r] = [2]float64{math.Inf(-1), math.Inf(1)}
			if bestLowIdx < 0 {
				bestLow = math.Inf(-1)
				bestLowIdx = r
			}
			continue
		}

		var topCodeDot float64
		rowPlaneBase := r * h.TotalBits * h.WordsPerRow
		for wordIdx := 0; wordIdx < h.WordsPerRow; wordIdx++ {
			kStart := wordIdx << 5
			kEnd := kStart + bitplaneWordBits
			if kEnd > h.HiddenDim {
				kEnd = h.HiddenDim
			}
			// Read only the top `topBits` plane words for this 32-weight block.
			var words [bitplaneMaxTotalBits]uint32
			for p := 0; p < topBits; p++ {
				words[p] = h.Planes[rowPlaneBase+p*h.WordsPerRow+wordIdx]
			}
			for k := kStart; k < kEnd; k++ {
				bit := uint(k & 31)
				var topCode uint32
				for p := 0; p < topBits; p++ {
					topCode = (topCode << 1) | ((words[p] >> bit) & 1)
				}
				shiftedCode := topCode << remBits
				topCodeDot += float64(shiftedCode) * float64(x[k])
			}
		}

		baseDot := minVal*sumX + scale*topCodeDot
		tailMin := scale * float64(tailMask) * negSumX
		tailMax := scale * float64(tailMask) * posSumX
		f32Slack := gamma32*maxAbsW*absSumX + nF32*float64(math.SmallestNonzeroFloat32)
		f64Slack := float64(h.HiddenDim+8) * u64 * ((math.Abs(minVal)+scale*float64(maxCode))*absSumX + 1.0)
		pad := eps*absSumX + f32Slack + f64Slack

		lo := baseDot + tailMin - pad
		hi := baseDot + tailMax + pad
		if math.IsNaN(lo) || math.IsNaN(hi) {
			return BitplaneArgmaxResult{}, ErrBitplaneNonFinite
		}
		bounds[r] = [2]float64{lo, hi}
		if bestLowIdx < 0 || lo > bestLow {
			bestLow = lo
			bestLowIdx = r
		}
	}

	// Stage 2: margin certification — prune any row r != bestLowIdx whose upper
	// bound cannot beat bestLowIdx (respecting lowest-index tie-breaking).
	var contenders []int
	for r := 0; r < h.Vocab; r++ {
		if r == bestLowIdx {
			continue
		}
		hi := bounds[r][1]
		if hi > bestLow || (hi == bestLow && r < bestLowIdx) {
			contenders = append(contenders, r)
		}
	}

	// Zero-contender early exit: bestLowIdx is certified without refining any row.
	if len(contenders) == 0 {
		return BitplaneArgmaxResult{
			Token:       bestLowIdx,
			Certified:   true,
			PrunedRows:  h.Vocab - 1,
			RefinedRows: 0,
			Contenders:  nil,
			Bounds:      bounds,
		}, nil
	}

	// Stage 3: exact F32 dot-product refinement ONLY on contending rows (plus bestLowIdx).
	contenders = append(contenders, bestLowIdx)
	sort.Ints(contenders)

	bestToken := -1
	bestScore := float32(math.Inf(-1))
	for _, r := range contenders {
		wRow := h.RawWeights[r*h.HiddenDim : (r+1)*h.HiddenDim]
		score := mathx.FDot(wRow, x)
		if !isFiniteF32(score) {
			return BitplaneArgmaxResult{}, fmt.Errorf("%w: refined row %d score=%v", ErrBitplaneNonFinite, r, score)
		}
		if bestToken < 0 || score > bestScore || (score == bestScore && r < bestToken) {
			bestScore = score
			bestToken = r
		}
	}

	return BitplaneArgmaxResult{
		Token:       bestToken,
		Certified:   false,
		PrunedRows:  h.Vocab - len(contenders),
		RefinedRows: len(contenders),
		Contenders:  contenders,
		Bounds:      bounds,
	}, nil
}

// ExactArgmaxF32 computes the full F32 dot-product logits and argmax using
// mathx.FDot and mathx.ArgmaxF32, serving as the reference oracle.
func ExactArgmaxF32(weights, x []float32, vocab, hiddenDim int) (int, []float32, error) {
	if vocab <= 0 || hiddenDim <= 0 {
		return 0, nil, ErrBitplaneInvalidShape
	}
	maxInt := int(^uint(0) >> 1)
	if vocab > maxInt/hiddenDim || len(weights) != vocab*hiddenDim || len(x) != hiddenDim {
		return 0, nil, ErrBitplaneInvalidShape
	}
	for i, w := range weights {
		if !isFiniteF32(w) {
			return 0, nil, fmt.Errorf("%w: weight[%d]=%v", ErrBitplaneNonFinite, i, w)
		}
	}
	for i, xk := range x {
		if !isFiniteF32(xk) {
			return 0, nil, fmt.Errorf("%w: x[%d]=%v", ErrBitplaneNonFinite, i, xk)
		}
	}
	logits := make([]float32, vocab)
	for r := 0; r < vocab; r++ {
		s := mathx.FDot(weights[r*hiddenDim:(r+1)*hiddenDim], x)
		if !isFiniteF32(s) {
			return 0, nil, fmt.Errorf("%w: logits[%d]=%v", ErrBitplaneNonFinite, r, s)
		}
		logits[r] = s
	}
	return mathx.ArgmaxF32(logits), logits, nil
}
