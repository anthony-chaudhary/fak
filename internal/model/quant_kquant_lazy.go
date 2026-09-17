package model

import "fmt"

// quant_kquant_lazy.go — the lazy/on-demand materialization seam for the NON-Q4_K dense
// k-quants, symmetric with the Q4_K path in quant_q4k.go (#13202). The Q4_K store holds a
// checkpoint range and reads it only when a consumer actually needs the bytes; this file
// gives the dense k-quant stores (Q2_K/Q3_K/Q5_K/Q6_K and the IQ family) the same primitive,
// so the bounded-dense MoE route (#13201) can retain a non-Q4_K dense k-quant by descriptor
// instead of paying its full payload in host RAM up front.
//
// The range descriptor is the SAME format-agnostic LazyQ4KRange (bytes + reader + optional
// mapped span), and the mapped-span validation is the SHARED mappedLazyRaw, so the two
// stores cannot drift. The only k-quant-specific fact is the block geometry that turns a
// [out,in] shape into an exact byte count.

// materializeRaw returns the resident raw payload of a lazy k-quant tensor, mirroring
// q4kTensor.materializeRaw exactly: a memoized raw copy wins, then a validated zero-copy
// mapped span, else a bounded-window page-aligned chunked ReadAt that fails closed on a
// missing reader, a reader error, or a short read.
func (qt *kQuantTensor) materializeRaw() ([]byte, error) {
	if len(qt.raw) > 0 {
		return qt.raw, nil
	}
	if span, offset, ok := mappedLazyRaw(qt.lazy); ok {
		return span[offset : offset+qt.lazy.Bytes], nil
	}
	if qt.lazy == nil || qt.lazy.Reader == nil || qt.lazy.Bytes <= 0 {
		return nil, fmt.Errorf("model: k-quant tensor has no resident or lazy payload")
	}
	raw := makePageAlignedResidentBytes(qt.lazy.Bytes)
	window := q4kMaterializeWindowBytes
	if window <= 0 || window > qt.lazy.Bytes {
		window = qt.lazy.Bytes
	}
	buf := make([]byte, window)
	for done := 0; done < qt.lazy.Bytes; {
		n := qt.lazy.Bytes - done
		if n > len(buf) {
			n = len(buf)
		}
		if _, err := qt.lazy.Reader.ReadAt(buf[:n], qt.lazy.Offset+int64(done)); err != nil {
			return nil, err
		}
		copy(raw[done:], buf[:n])
		done += n
	}
	return raw, nil
}

// requireRawCPU is the #13202 legibility guardrail for the CPU k-quant matmul entry points
// (kQuantMatRowsRangeRaw and its siblings). A lazy k-quant tensor holds no resident raw
// bytes until materializeRaw runs, and the CPU fallback does `if len(raw)==0 { raw = qt.raw }`
// — over a lazy tensor both are empty, so the row loop would silently read a nil slice and
// emit ZEROS instead of failing. That is a silent-wrong-answer bug, so this turns it into a
// named panic: the tensor must be materialized (or read through a device path) first.
func (qt *kQuantTensor) requireRawCPU(op string) {
	if qt != nil && len(qt.raw) == 0 && qt.lazy != nil && qt.out > 0 {
		panic(fmt.Sprintf("model: k-quant %s on a lazy tensor (kind=%s out=%d in=%d): the resident "+
			"raw bytes are a checkpoint range that has not been materialized, and a CPU k-quant "+
			"matmul ran — reading nil raw would silently produce zeros. Call materializeRaw first, "+
			"or route the matmul through a device path that reads the lazy range (#13202).",
			op, qt.kind, qt.out, qt.in))
	}
}

// AddLazyKQuant stores a checkpoint-backed k-quant descriptor without reading its payload.
// It runs the SAME eligibility gate as the resident k-quant entries (residentQuantTarget),
// validates the byte count against the kind's block geometry, and stores a kQuantTensor
// that holds only the range — no payload read. Symmetric with AddLazyQ4K.
func (b *QuantBuilder) AddLazyKQuant(canon string, shape []int, kind kQuantKind, src LazyQ4KRange) error {
	return b.addResidentQuant(canon, shape, func(name string) {
		if b.m.kqw == nil {
			b.m.kqw = map[string]*kQuantTensor{}
		}
		blockWeights := kind.blockWeights()
		if shape[1]%blockWeights != 0 {
			panic("model: lazy k-quant reduction dim not a multiple of block size")
		}
		want := shape[0] * (shape[1] / blockWeights) * kind.blockBytes()
		if src.Bytes != want {
			panic("model: lazy k-quant payload size mismatch")
		}
		b.m.kqw[name] = &kQuantTensor{
			out:  shape[0],
			in:   shape[1],
			nblk: shape[1] / blockWeights,
			kind: kind,
			lazy: &src,
		}
	})
}

// KQuantLazy reports whether a k-quant tensor is backed by a checkpoint range rather than
// resident bytes. Exported diagnostic twin of Q4KLazy.
func (m *Model) KQuantLazy(name string) bool { return m.kqw[name] != nil && m.kqw[name].lazy != nil }
