package model

import (
	"errors"
	"fmt"
	"math"
)

// v4_flash_attn_ratio0.go - the native ratio-0 (window-only) attention
// reference for DeepSeek-V4-Flash-0731 (parent #12636, epic #12635).
//
// The published Attention.forward at revision
// 7872f01b1d1fe23eabc4c98b48bffcef5a386062 for a layer whose compress_ratio is
// 0 executes, in order:
//
//   q  = wq_b(wq_norm(wq_a(x)))            // low-rank Q, reshaped [seq,n_heads,head_dim]
//   q *= rsqrt(mean(q^2, -1) + eps)        // head RMS scale, NO learnable gain
//   rope(q[..., -rd:])                     // RoPE on the FINAL rope_head_dim dims only
//   kv = wkv(x); kv = kv_norm(kv)          // single latent K/V row per position
//   rope(kv[..., -rd:])
//   o  = sparse_attn(q, window, attn_sink, softmax_scale)
//   rope(o[..., -rd:], inverse=True)       // INVERSE rope on the final rd dims
//   o  = wo_b(wo_a(o.view(seq, o_groups, -1)))
//
// The 128-slot window is causal by construction: each row's kv is appended
// BEFORE that row's attention is computed, so a query always sees itself and
// the prior rows still resident in the window, oldest-first.
//
// sparse_attn keeps the LEARNABLE SINK in the softmax DENOMINATOR ONLY:
//
//   Z = exp(sink - m) + sum_t exp(score_t - m),  m = max(sink, max_t score_t)
//   o = sum_t (exp(score_t - m)/Z) * k_t
//
// so the sink can suppress attention but never contributes a value vector.

var (
	// ErrV4FlashRatio0Geometry reports a DeepSeek-V4-Flash config whose
	// dimensions cannot describe the published ratio-0 attention: a non-positive
	// or inconsistent head/lora/group geometry, or a window that is not the
	// published 128.
	ErrV4FlashRatio0Geometry = errors.New("model: DeepSeek V4 Flash ratio-0 geometry is invalid")

	// ErrV4FlashRatio0Shape reports a runtime shape violation in the ratio-0
	// reference: an input row or weight matrix that does not match the geometry,
	// a negative start position, a nil window, or a mis-sized inverse-frequency
	// vector. The reference fails closed rather than panic.
	ErrV4FlashRatio0Shape = errors.New("model: DeepSeek V4 Flash ratio-0 tensor shape is invalid")
)

// V4FlashRatio0Geometry is the fixed head/lora/group geometry of a ratio-0
// attention layer. NormEps is the head-RMS epsilon (Config.RMSNormEps).
type V4FlashRatio0Geometry struct {
	Dim         int
	NumHeads    int
	HeadDim     int
	RopeHeadDim int
	QLoraRank   int
	OGroups     int
	OLoraRank   int
	WindowSize  int
	NormEps     float64
}

// V4FlashRatio0Weights is the row-major weight set consumed by the ratio-0
// reference. Every inner slice is copied out of the checkpoint row-major.
//
//	WqA      [QLoraRank][Dim]
//	WqB      [NumHeads*HeadDim][QLoraRank]
//	Wkv      [HeadDim][Dim]
//	Woa      [OGroups*OLoraRank][NumHeads*HeadDim/OGroups]
//	Wob      [Dim][OGroups*OLoraRank]
//	AttnSink [NumHeads]
type V4FlashRatio0Weights struct {
	WqA      [][]float32
	WqB      [][]float32
	Wkv      [][]float32
	Woa      [][]float32
	Wob      [][]float32
	AttnSink []float32
}

// V4FlashRatio0GeometryFromConfig derives and validates the ratio-0 geometry
// from a V4-Flash config without mutating it. Any violation returns
// ErrV4FlashRatio0Geometry.
func V4FlashRatio0GeometryFromConfig(cfg Config) (V4FlashRatio0Geometry, error) {
	g := V4FlashRatio0Geometry{
		Dim:         cfg.HiddenSize,
		NumHeads:    cfg.NumHeads,
		HeadDim:     cfg.HeadDim,
		RopeHeadDim: cfg.QKRopeHeadDim,
		QLoraRank:   cfg.QLoraRank,
		OGroups:     cfg.OGroups,
		OLoraRank:   cfg.OLoraRank,
		WindowSize:  V4FlashWindowSize,
		NormEps:     cfg.RMSNormEps,
	}
	switch {
	case g.HeadDim <= 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: head_dim=%d", ErrV4FlashRatio0Geometry, g.HeadDim)
	case g.NumHeads <= 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: num_heads=%d", ErrV4FlashRatio0Geometry, g.NumHeads)
	case g.RopeHeadDim <= 0 || g.RopeHeadDim >= g.HeadDim:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: qk_rope_head_dim=%d head_dim=%d", ErrV4FlashRatio0Geometry, g.RopeHeadDim, g.HeadDim)
	case g.QLoraRank <= 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: q_lora_rank=%d", ErrV4FlashRatio0Geometry, g.QLoraRank)
	case g.OGroups <= 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: o_groups=%d", ErrV4FlashRatio0Geometry, g.OGroups)
	case g.HeadDim%g.OGroups != 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: head_dim=%d not divisible by o_groups=%d", ErrV4FlashRatio0Geometry, g.HeadDim, g.OGroups)
	case g.OLoraRank <= 0:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: o_lora_rank=%d", ErrV4FlashRatio0Geometry, g.OLoraRank)
	case g.WindowSize != V4FlashWindowSize:
		return V4FlashRatio0Geometry{}, fmt.Errorf("%w: window_size=%d want %d", ErrV4FlashRatio0Geometry, g.WindowSize, V4FlashWindowSize)
	}
	return g, nil
}

// v4FlashMatVecRows applies one row-major matrix-vector product per matrix row.
// Every row shares the input x and lands in the corresponding dst slot.
func v4FlashMatVecRows(dst []float32, rows [][]float32, x []float32) {
	for r, row := range rows {
		var acc float32
		for c := range x {
			acc += float32(row[c] * x[c])
		}
		dst[r] = acc
	}
}

// v4FlashApplyRoPE rotates the FINAL ropeDim dims of row in place as adjacent
// complex pairs (0,1),(2,3),... over that suffix. out[2j] = a*cos - b*sin and
// out[2j+1] = b*cos + a*sin with a=row[2j], b=row[2j+1]. inverse conjugates the
// rotation by negating sin. cos/sin come from inv and the absolute position p.
func v4FlashApplyRoPE(row []float32, ropeDim int, inv []float64, p int, inverse bool) {
	half := ropeDim / 2
	off := len(row) - ropeDim
	for j := 0; j < half; j++ {
		ang := float64(p) * inv[j]
		c := float32(math.Cos(ang))
		s := float32(math.Sin(ang))
		if inverse {
			s = -s
		}
		a := row[off+2*j]
		b := row[off+2*j+1]
		row[off+2*j] = float32(a*c) - float32(b*s)
		row[off+2*j+1] = float32(b*c) + float32(a*s)
	}
}

// v4FlashRMSHeadScale scales the head row in place by
// 1/sqrt(mean(row^2)+eps). The published ratio-0 Q head scale has NO learnable
// gain, so only the epsilon is applied.
func v4FlashRMSHeadScale(row []float32, eps float32) {
	var ss float32
	for _, v := range row {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(row))+eps)))
	for i := range row {
		row[i] = float32(row[i] * inv)
	}
}

// v4FlashSparseAttnHead is the published sparse-attention reduction for one
// head: q and each key are one HeadDim row. The learnable sink contributes to
// the softmax DENOMINATOR ONLY, so it can suppress attention without supplying
// a value.
func v4FlashSparseAttnHead(q []float32, keys [][]float32, sink float32, scale float32) []float32 {
	out := make([]float32, len(q))
	if len(keys) == 0 {
		return out
	}
	scores := make([]float32, len(keys))
	m := sink
	for t, k := range keys {
		var dot float32
		for d := range q {
			dot += float32(q[d] * k[d])
		}
		s := float32(dot * scale)
		scores[t] = s
		if s > m {
			m = s
		}
	}
	z := float32(math.Exp(float64(sink - m)))
	for _, s := range scores {
		z += float32(math.Exp(float64(s - m)))
	}
	for t := range scores {
		p := float32(math.Exp(float64(scores[t]-m))) / z
		for d := range out {
			out[d] += float32(p * keys[t][d])
		}
	}
	return out
}

// v4FlashRatio0Attention runs the published ratio-0 attention forward for one
// layer over x ([seq][Dim] hidden rows) and returns [seq][Dim] output rows.
//
// inv is the RoPE inverse-frequency vector (length RopeHeadDim/2), injected so
// the caller controls position math. window is the shared
// *V4FlashCircularWindow; each row's kv is appended BEFORE that row's attention
// (the causal ordering), then all retained rows oldest-first are attended with
// the per-head learnable sink in the softmax denominator. startPos is the first
// row's absolute position (0 for prefill, >0 for decode).
func v4FlashRatio0Attention(g V4FlashRatio0Geometry, w V4FlashRatio0Weights, x [][]float32, startPos int, inv []float64, window *V4FlashCircularWindow) ([][]float32, error) {
	if startPos < 0 {
		return nil, fmt.Errorf("%w: startPos=%d", ErrV4FlashRatio0Shape, startPos)
	}
	if window == nil {
		return nil, fmt.Errorf("%w: nil window", ErrV4FlashRatio0Shape)
	}
	if len(inv) != g.RopeHeadDim/2 {
		return nil, fmt.Errorf("%w: len(inv)=%d want %d", ErrV4FlashRatio0Shape, len(inv), g.RopeHeadDim/2)
	}
	if err := v4FlashRatio0ValidateWeights(g, w); err != nil {
		return nil, err
	}

	groupDim := g.HeadDim / g.OGroups
	packedDim := g.NumHeads * g.HeadDim
	midDim := g.OGroups * g.OLoraRank

	out := make([][]float32, len(x))
	for i, row := range x {
		if len(row) != g.Dim {
			return nil, fmt.Errorf("%w: x row %d len=%d want %d", ErrV4FlashRatio0Shape, i, len(row), g.Dim)
		}

		qLo := make([]float32, g.QLoraRank)
		v4FlashMatVecRows(qLo, w.WqA, row)
		v4FlashRMSHeadScale(qLo, float32(g.NormEps))
		qPacked := make([]float32, packedDim)
		v4FlashMatVecRows(qPacked, w.WqB, qLo)
		for h := 0; h < g.NumHeads; h++ {
			head := qPacked[h*g.HeadDim : (h+1)*g.HeadDim]
			v4FlashRMSHeadScale(head, float32(g.NormEps))
			v4FlashApplyRoPE(head, g.RopeHeadDim, inv, startPos+i, false)
		}

		kv := make([]float32, g.HeadDim)
		v4FlashMatVecRows(kv, w.Wkv, row)
		v4FlashRMSHeadScale(kv, float32(g.NormEps))
		v4FlashApplyRoPE(kv, g.RopeHeadDim, inv, startPos+i, false)

		window.Append(kv)
		keys := window.Rows()

		o := make([]float32, packedDim)
		for h := 0; h < g.NumHeads; h++ {
			qh := qPacked[h*g.HeadDim : (h+1)*g.HeadDim]
			oh := o[h*g.HeadDim : (h+1)*g.HeadDim]
			copy(oh, v4FlashSparseAttnHead(qh, keys, w.AttnSink[h], float32(1.0/math.Sqrt(float64(g.HeadDim)))))
			v4FlashApplyRoPE(oh, g.RopeHeadDim, inv, startPos+i, true)
		}

		var mid []float32
		for grp := 0; grp < g.OGroups; grp++ {
			oGrp := make([]float32, 0, g.NumHeads*groupDim)
			for h := 0; h < g.NumHeads; h++ {
				head := o[h*g.HeadDim : (h+1)*g.HeadDim]
				oGrp = append(oGrp, head[grp*groupDim:(grp+1)*groupDim]...)
			}
			midGrp := make([]float32, g.OLoraRank)
			v4FlashMatVecRows(midGrp, w.Woa[grp*g.OLoraRank:(grp+1)*g.OLoraRank], oGrp)
			mid = append(mid, midGrp...)
		}
		y := make([]float32, g.Dim)
		v4FlashMatVecRows(y, w.Wob, mid[:midDim])
		out[i] = y
	}
	return out, nil
}

// v4FlashRatio0ValidateWeights fails closed on any weight matrix whose row or
// column count does not match the geometry.
func v4FlashRatio0ValidateWeights(g V4FlashRatio0Geometry, w V4FlashRatio0Weights) error {
	packedDim := g.NumHeads * g.HeadDim
	groupDim := g.HeadDim / g.OGroups
	woaIn := g.NumHeads * groupDim
	if err := v4FlashCheckMatrix("WqA", w.WqA, g.QLoraRank, g.Dim); err != nil {
		return err
	}
	if err := v4FlashCheckMatrix("WqB", w.WqB, packedDim, g.QLoraRank); err != nil {
		return err
	}
	if err := v4FlashCheckMatrix("Wkv", w.Wkv, g.HeadDim, g.Dim); err != nil {
		return err
	}
	if err := v4FlashCheckMatrix("Woa", w.Woa, g.OGroups*g.OLoraRank, woaIn); err != nil {
		return err
	}
	if err := v4FlashCheckMatrix("Wob", w.Wob, g.Dim, g.OGroups*g.OLoraRank); err != nil {
		return err
	}
	if len(w.AttnSink) != g.NumHeads {
		return fmt.Errorf("%w: AttnSink len=%d want %d", ErrV4FlashRatio0Shape, len(w.AttnSink), g.NumHeads)
	}
	return nil
}

// v4FlashCheckMatrix enforces a rows x cols shape on one row-major matrix.
func v4FlashCheckMatrix(name string, m [][]float32, rows, cols int) error {
	if len(m) != rows {
		return fmt.Errorf("%w: %s rows=%d want %d", ErrV4FlashRatio0Shape, name, len(m), rows)
	}
	for r, row := range m {
		if len(row) != cols {
			return fmt.Errorf("%w: %s row %d len=%d want %d", ErrV4FlashRatio0Shape, name, r, len(row), cols)
		}
	}
	return nil
}
