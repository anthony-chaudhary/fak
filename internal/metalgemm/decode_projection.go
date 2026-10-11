//go:build darwin && arm64 && cgo

package metalgemm

import "errors"

// KQuantProjection is a resident native k-quant decode projection with exactly one handle set.
// A decode graph encodes it from the f32 activation where a Q8Weight takes the Q8-quantized one,
// so a projection the checkpoint ships as Q4_K/Q5_K/Q6_K streams its own bytes instead of a
// requantized Q8_0 copy.
type KQuantProjection struct {
	Q4K *Q4KWeight
	Q5K *Q5KWeight
	Q6K *Q6KWeight
}

// IsSet reports whether any native handle is present.
func (p KQuantProjection) IsSet() bool { return p.Q4K != nil || p.Q5K != nil || p.Q6K != nil }

func (p KQuantProjection) shape() (out, in int, ok bool) {
	n := 0
	if p.Q4K != nil {
		n++
		out, in, ok = p.Q4K.Out, p.Q4K.In, p.Q4K.id >= 0
	}
	if p.Q5K != nil {
		n++
		out, in, ok = p.Q5K.Out, p.Q5K.In, p.Q5K.ID() >= 0
	}
	if p.Q6K != nil {
		n++
		out, in, ok = p.Q6K.Out, p.Q6K.In, p.Q6K.ID() >= 0
	}
	return out, in, ok && n == 1
}

func (p KQuantProjection) encode(g *ProjectionGraph, x *GraphResult) (*GraphResult, error) {
	switch {
	case p.Q4K != nil:
		return g.EncodeQ4KFrom(p.Q4K, x)
	case p.Q5K != nil:
		return g.EncodeQ5KFrom(p.Q5K, x)
	case p.Q6K != nil:
		return g.EncodeQ6KFrom(p.Q6K, x)
	}
	return nil, errors.New("metalgemm: empty k-quant projection")
}

// decodeProjection is one resolved decode projection: a Q8 handle or a native k-quant handle.
type decodeProjection struct {
	name string
	q8   *Q8Weight
	kq   KQuantProjection
}

// shape is valid only while the caller holds the Q8 handle's read lock (lockDecodeQ8).
func (p decodeProjection) shape() (out, in int, ok bool) {
	if p.q8 != nil {
		return p.q8.Out, p.q8.In, p.q8.id >= 0 && !p.kq.IsSet()
	}
	return p.kq.shape()
}

// liveShape is shape for a caller that does not hold the Q8 read lock; it takes it itself.
func (p decodeProjection) liveShape() (out, in int, ok bool) {
	if p.q8 != nil {
		if p.kq.IsSet() {
			return 0, 0, false
		}
		return p.q8.Out, p.q8.In, p.q8.ID() >= 0
	}
	return p.kq.shape()
}

// encode projects x. A Q8 projection quantizes x at most once per source, caching the codes in
// *quantized and counting the quantizer encoder in *quantizers.
func (p decodeProjection) encode(g *ProjectionGraph, x *GraphResult, quantized **QuantizedGraphResult, quantizers *int) (*GraphResult, error) {
	if p.q8 == nil {
		return p.kq.encode(g, x)
	}
	if *quantized == nil {
		q, err := g.QuantizeQ8(x)
		if err != nil {
			return nil, err
		}
		*quantized = q
		*quantizers++
	}
	return g.EncodeQ8From(p.q8, *quantized)
}

// lockDecodeQ8 read-locks every Q8 handle (in order) and returns the matching unlock.
func lockDecodeQ8(ps []decodeProjection) func() {
	for _, p := range ps {
		if p.q8 != nil {
			p.q8.mu.RLock()
		}
	}
	return func() {
		for i := len(ps) - 1; i >= 0; i-- {
			if ps[i].q8 != nil {
				ps[i].q8.mu.RUnlock()
			}
		}
	}
}

func (w Qwen35DecodeWeights) projections() []decodeProjection {
	q8 := [5]*Q8Weight{w.InQKV, w.InZ, w.InB, w.InA, w.Out}
	names := [5]string{"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
	out := make([]decodeProjection, 5)
	for i := range out {
		out[i] = decodeProjection{name: names[i], q8: q8[i], kq: w.Native[i]}
	}
	return out
}
