package compute

import (
	"fmt"
	"math"
)

// SpecVerifyBackend is the optional capability interface a backend implements
// to run split-KV multi-query speculative verify attention (#11100).
type SpecVerifyBackend interface {
	Backend
	SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error
}

// SpecVerifyAttentionBackend exposes the speculative verify attention kernel method.
type SpecVerifyAttentionBackend interface {
	SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error
}

// specVerifyAttentionHost is the byte-for-byte CPU reference implementation for
// split-KV multi-query speculative verify attention (#11100).
// q is [qLen, nH, d], k is [kvLen, nHkv, d], v is [kvLen, nHkv, d], out is [qLen, nH, d].
// Causal masking is applied: each draft query token qi attends to all prefix keys
// plus preceding draft tokens up to qi (prefix = kvLen - qLen).
func specVerifyAttentionHost(q, k, v, out []float32, qLen, kvLen, nH, nHkv, d int, scale float32) error {
	if qLen <= 0 || kvLen <= 0 || kvLen < qLen {
		return fmt.Errorf("compute: specVerifyAttentionHost invalid lengths qLen=%d kvLen=%d", qLen, kvLen)
	}
	if nH <= 0 || nHkv <= 0 || (nH%nHkv) != 0 {
		return fmt.Errorf("compute: specVerifyAttentionHost invalid heads nH=%d nHkv=%d", nH, nHkv)
	}
	if d <= 0 {
		return fmt.Errorf("compute: specVerifyAttentionHost invalid head dim d=%d", d)
	}
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(d)))
	}
	grp := nH / nHkv
	w := nHkv * d
	prefix := kvLen - qLen

	// Clear out
	for i := range out {
		out[i] = 0
	}

	scores := make([]float32, kvLen)

	for qi := 0; qi < qLen; qi++ {
		attendLen := prefix + qi + 1
		if attendLen > kvLen {
			attendLen = kvLen
		}

		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[(qi*nH+h)*d : (qi*nH+h+1)*d]

			for j := 0; j < attendLen; j++ {
				kj := k[j*w+kvh*d : j*w+(kvh+1)*d]
				scores[j] = dot(qh, kj) * scale
			}

			softmaxInPlace(scores[:attendLen])

			oh := out[(qi*nH+h)*d : (qi*nH+h+1)*d]
			for j := 0; j < attendLen; j++ {
				vj := v[j*w+kvh*d : j*w+(kvh+1)*d]
				weight := scores[j]
				for dim := 0; dim < d; dim++ {
					oh[dim] += weight * vj[dim]
				}
			}
		}
	}
	return nil
}

// SpecVerifyAttention on cpuBackend is the CPU Reference implementation (#11100).
func (c *cpuBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil || k == nil || v == nil || out == nil {
		return fmt.Errorf("compute: SpecVerifyAttention nil tensor argument")
	}
	if qLen <= 0 || kvLen <= 0 || kvLen < qLen {
		return fmt.Errorf("compute: SpecVerifyAttention invalid lengths qLen=%d kvLen=%d", qLen, kvLen)
	}
	if nH <= 0 || nHkv <= 0 || (nH%nHkv) != 0 {
		return fmt.Errorf("compute: SpecVerifyAttention invalid heads nH=%d nHkv=%d", nH, nHkv)
	}
	if d <= 0 {
		return fmt.Errorf("compute: SpecVerifyAttention invalid head dim d=%d", d)
	}
	expectedQ := qLen * nH * d
	expectedKV := kvLen * nHkv * d
	if q.Numel() != expectedQ {
		return fmt.Errorf("compute: SpecVerifyAttention q numel %d != expected %d", q.Numel(), expectedQ)
	}
	if k.Numel() != expectedKV {
		return fmt.Errorf("compute: SpecVerifyAttention k numel %d != expected %d", k.Numel(), expectedKV)
	}
	if v.Numel() != expectedKV {
		return fmt.Errorf("compute: SpecVerifyAttention v numel %d != expected %d", v.Numel(), expectedKV)
	}

	qf := c.f32(*q)
	kf := c.f32(*k)
	vf := c.f32(*v)

	outSlice := make([]float32, expectedQ)
	scale := float32(1.0 / math.Sqrt(float64(d)))
	if err := specVerifyAttentionHost(qf, kf, vf, outSlice, qLen, kvLen, nH, nHkv, d, scale); err != nil {
		return err
	}
	*out = c.result([]int{qLen, nH, d}, outSlice)
	return nil
}

// SpecVerifyAttention executes split-KV multi-query speculative verify attention (#11100).
// If the backend owning q implements SpecVerifyBackend, it dispatches to that backend;
// otherwise it falls back to the CPU reference.
func SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil {
		return fmt.Errorf("compute: SpecVerifyAttention nil q tensor")
	}
	be := q.Backend()
	if be == nil {
		be = Default()
	}
	if svb, ok := be.(SpecVerifyBackend); ok {
		return svb.SpecVerifyAttention(q, k, v, out, qLen, kvLen, nH, nHkv, d)
	}
	ref, ok := Default().(SpecVerifyBackend)
	if !ok {
		return fmt.Errorf("compute: Default backend does not implement SpecVerifyBackend")
	}
	return ref.SpecVerifyAttention(q, k, v, out, qLen, kvLen, nH, nHkv, d)
}
