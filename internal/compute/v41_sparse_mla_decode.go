package compute

import (
	"errors"
	"fmt"
	"math"
)

// errV41SparseMLADecodeGeometry identifies a malformed request to
// v41SparseMLADecodeReference. It is a caller contract failure, never an
// arithmetic one, so no partial output exists when it is returned.
var errV41SparseMLADecodeGeometry = errors.New("compute: invalid V4.1 sparse MLA decode geometry")

// v41SparseMLADecodeOptions selects the FlashMLA-class decode partition the
// reference reproduces. TileRows is the number of selected rows one online-
// softmax step consumes (FlashMLA's sparse decode uses 64-row top-k blocks);
// Splits is the number of independent split-KV partials the combine step merges
// (FlashMLA's scheduler metadata chooses it per batch). Both must be positive;
// Splits larger than the tile count is clamped to the tile count, exactly as a
// split that owns no tile produces no partial.
type v41SparseMLADecodeOptions struct {
	TileRows int
	Splits   int
}

// v41SparseMLADecodeResult is a fresh output [heads,headDim] plus one natural-
// log LSE per head over the selected scores only. The sink never enters LSE: it
// only rescales the output, so a split-KV caller can merge LSEs before folding
// it. A head with no selected rows has a zero output and LSE -Inf.
type v41SparseMLADecodeResult struct {
	Out []float32
	LSE []float32
}

// v41SparseMLADecodeReference is the host reference for the FlashMLA numerical
// class of the V4.1 shared-latent sink attention (fak#13993). It accepts the
// same already-gathered input as the shared-attention backend - q[heads,headDim],
// sharedKV[selectedRows,headDim] supplying both K and V to every head, and an
// optional per-head sink - and differs from that contract only in arithmetic
// order: scores are taken to base 2 (scale*log2(e)), each TileRows tile updates
// a running maximum, rescaled sum and rescaled value accumulator, each split
// emits a normalized partial and its base-2 LSE, the combine step weights the
// partials by exp2(lse_s - lse), and the sink is folded last as
// out *= 1/(1+exp(sink-lse)).
//
// That is the online-softmax/split-combine substitution the serial three-pass
// contract forbids, so a CUDA FlashMLA adapter can never be bit-compared with
// the serial oracle. This reference is what such an adapter is compared with;
// its distance from the serial oracle is pinned by tests, not assumed. It is a
// pure function: no backend, device, model weight, or fallback selection.
//
// Products and sums round to binary32 explicitly so no architecture fuses them
// into FMA. Inputs must be finite; a nonfinite intermediate returns a
// *v41SparseMLADecodeError and no output.
func v41SparseMLADecodeReference(q, sharedKV, sink []float32, selectedRows, heads, headDim int, scale float32, hasSink bool, opt v41SparseMLADecodeOptions) (v41SparseMLADecodeResult, error) {
	if err := validateV41SparseMLADecode(q, sharedKV, sink, selectedRows, heads, headDim, scale, hasSink, opt); err != nil {
		return v41SparseMLADecodeResult{}, err
	}
	res := v41SparseMLADecodeResult{Out: make([]float32, heads*headDim), LSE: make([]float32, heads)}
	if selectedRows == 0 {
		for h := range res.LSE {
			res.LSE[h] = float32(math.Inf(-1))
		}
		return res, nil
	}
	tiles := (selectedRows + opt.TileRows - 1) / opt.TileRows
	splits := min(opt.Splits, tiles)
	log2e := float32(math.Log2E)
	scale2 := float32(scale * log2e)
	partOut := make([]float32, splits*headDim)
	partLSE := make([]float32, splits)
	acc := make([]float32, headDim)
	scores := make([]float32, opt.TileRows)
	for h := range heads {
		qh := q[h*headDim : (h+1)*headDim]
		for s := range splits {
			lo := s * tiles / splits * opt.TileRows
			hi := min((s+1)*tiles/splits*opt.TileRows, selectedRows)
			lse, err := v41SparseMLADecodeSplit(qh, sharedKV, lo, hi, headDim, scale2, opt.TileRows, h, acc, scores)
			if err != nil {
				return v41SparseMLADecodeResult{}, err
			}
			copy(partOut[s*headDim:(s+1)*headDim], acc)
			partLSE[s] = lse
		}
		out := res.Out[h*headDim : (h+1)*headDim]
		lse2, err := v41SparseMLADecodeCombine(partOut, partLSE, splits, headDim, h, out)
		if err != nil {
			return v41SparseMLADecodeResult{}, err
		}
		if hasSink {
			// 1/(1+exp2(sink2-lse2)) with sink2 = sink*log2(e).
			gap := float32(float32(sink[h]*log2e) - lse2)
			factor := float32(1 / float32(1+v41SparseMLAExp2(gap)))
			if !finite32(factor) {
				return v41SparseMLADecodeResult{}, v41SparseMLADecodeFault("denominator", h, -1, -1, factor)
			}
			for d := range out {
				out[d] = float32(out[d] * factor)
			}
		}
		res.LSE[h] = float32(lse2 * float32(math.Ln2))
	}
	return res, nil
}

// v41SparseMLADecodeSplit runs the tiled online softmax over rows [lo,hi) and
// leaves the normalized partial in acc. It returns the split's base-2 LSE.
func v41SparseMLADecodeSplit(qh, kv []float32, lo, hi, dim int, scale2 float32, tileRows, h int, acc, scores []float32) (float32, error) {
	negInf := float32(math.Inf(-1))
	running, sum := negInf, float32(0)
	clear(acc)
	for t := lo; t < hi; t += tileRows {
		end := min(t+tileRows, hi)
		tileMax := negInf
		for r := t; r < end; r++ {
			row := kv[r*dim : (r+1)*dim]
			var dot float32
			for d := range dim {
				dot = float32(dot + float32(qh[d]*row[d]))
			}
			v := float32(dot * scale2)
			if !finite32(v) {
				return 0, v41SparseMLADecodeFault("scaled score", h, r, -1, v)
			}
			scores[r-t] = v
			tileMax = max(tileMax, v)
		}
		next := max(running, tileMax)
		alpha := float32(0)
		if running != negInf {
			alpha = v41SparseMLAExp2(float32(running - next))
		}
		sum = float32(sum * alpha)
		for d := range acc {
			acc[d] = float32(acc[d] * alpha)
		}
		for r := t; r < end; r++ {
			p := v41SparseMLAExp2(float32(scores[r-t] - next))
			sum = float32(sum + p)
			row := kv[r*dim : (r+1)*dim]
			for d := range acc {
				acc[d] = float32(acc[d] + float32(p*row[d]))
			}
		}
		running = next
		if !finite32(sum) {
			return 0, v41SparseMLADecodeFault("denominator", h, end-1, -1, sum)
		}
	}
	// sum >= 1 because the maximum score contributes exp2(0); no zero divide.
	for d := range acc {
		acc[d] = float32(acc[d] / sum)
		if !finite32(acc[d]) {
			return 0, v41SparseMLADecodeFault("normalized value", h, -1, d, acc[d])
		}
	}
	return float32(running + float32(math.Log2(float64(sum)))), nil
}

// v41SparseMLADecodeCombine merges normalized split partials by their base-2
// LSEs into out and returns the merged base-2 LSE.
func v41SparseMLADecodeCombine(partOut, partLSE []float32, splits, dim, h int, out []float32) (float32, error) {
	top := partLSE[0]
	for _, v := range partLSE[1:splits] {
		top = max(top, v)
	}
	var sum float32
	for _, v := range partLSE[:splits] {
		sum = float32(sum + v41SparseMLAExp2(float32(v-top)))
	}
	lse := float32(top + float32(math.Log2(float64(sum))))
	if !finite32(lse) {
		return 0, v41SparseMLADecodeFault("denominator", h, -1, -1, lse)
	}
	clear(out)
	for s := range splits {
		w := v41SparseMLAExp2(float32(partLSE[s] - lse))
		part := partOut[s*dim : (s+1)*dim]
		for d := range out {
			out[d] = float32(out[d] + float32(w*part[d]))
		}
	}
	for d, v := range out {
		if !finite32(v) {
			return 0, v41SparseMLADecodeFault("combined output", h, -1, d, v)
		}
	}
	return lse, nil
}

func v41SparseMLAExp2(v float32) float32 { return float32(math.Exp2(float64(v))) }

// v41SparseMLADecodeError names the first nonfinite intermediate. Head, Row and
// Element are zero-based gathered positions; -1 means not applicable. ValueBits
// keeps the offending binary32 payload, including a NaN payload.
type v41SparseMLADecodeError struct {
	Stage     string
	Head      int
	Row       int
	Element   int
	ValueBits uint32
}

func (e *v41SparseMLADecodeError) Error() string {
	return fmt.Sprintf("compute: V4.1 sparse MLA decode nonfinite %s at head=%d row=%d element=%d bits=0x%08x",
		e.Stage, e.Head, e.Row, e.Element, e.ValueBits)
}

func v41SparseMLADecodeFault(stage string, head, row, element int, value float32) error {
	return &v41SparseMLADecodeError{Stage: stage, Head: head, Row: row, Element: element, ValueBits: math.Float32bits(value)}
}

func validateV41SparseMLADecode(q, kv, sink []float32, rows, heads, dim int, scale float32, hasSink bool, opt v41SparseMLADecodeOptions) error {
	bad := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", errV41SparseMLADecodeGeometry, fmt.Sprintf(format, args...))
	}
	switch {
	case rows < 0 || heads <= 0 || dim <= 0:
		return bad("rows=%d heads=%d headDim=%d", rows, heads, dim)
	case opt.TileRows <= 0 || opt.Splits <= 0:
		return bad("tileRows=%d splits=%d", opt.TileRows, opt.Splits)
	case !finite32(scale) || scale == 0:
		return bad("scale %g must be finite and nonzero", scale)
	case heads > math.MaxInt/dim || rows > math.MaxInt/dim:
		return bad("shape overflows int")
	case len(q) != heads*dim:
		return bad("q length %d, want %d", len(q), heads*dim)
	case len(kv) != rows*dim:
		return bad("sharedKV length %d, want %d", len(kv), rows*dim)
	case hasSink && len(sink) != heads:
		return bad("sink length %d, want %d", len(sink), heads)
	}
	for _, values := range [][]float32{q, kv} {
		for i, v := range values {
			if !finite32(v) {
				return bad("nonfinite input element %d", i)
			}
		}
	}
	if hasSink {
		for i, v := range sink {
			if !finite32(v) {
				return bad("nonfinite sink %d", i)
			}
		}
	}
	return nil
}
