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

// ensureRawCPU materializes the bounded lazy range of a checkpoint-backed k-quant tensor on
// demand and memoizes it into qt.raw, so a CPU k-quant matmul over a lazy bounded-dense tensor
// (#13201/#13202) executes instead of hitting the requireRawCPU panic (#13216). It is the
// k-quant twin of the `qt.materializeRaw()` call the device staging path already makes in
// weightHALQ4K (hal.go): the retention primitive has always been able to fault the range, and
// the CPU forward was simply never asking it to. The read is bounded by the fak#13199 window
// (q4kMaterializeWindowBytes) into page-aligned resident bytes, so a dense side larger than host
// RAM is faulted through a small window rather than retained whole.
//
// Fail-closed is preserved: a tensor with no resident and no lazy payload is left untouched for
// requireRawCPU to name, and a read error panics legibly here rather than silently reading nil
// (which would emit zeros — the exact #13202 wrong answer). No silent zeros in any branch.
func (qt *kQuantTensor) ensureRawCPU(op string) {
	if qt == nil || len(qt.raw) > 0 || qt.lazy == nil {
		return
	}
	raw, err := qt.materializeRaw()
	if err != nil {
		panic(fmt.Sprintf("model: lazy k-quant %s materialization failed (kind=%s out=%d in=%d): %v "+
			"(#13216; the lazy range could not be faulted and a CPU k-quant matmul ran — refusing "+
			"to read nil raw and silently produce zeros).", op, qt.kind, qt.out, qt.in, err))
	}
	// #13253: memoization RETENTION is bounded by the model's declared streamed-dense working
	// set. A nil ledger (no bound declared) skips this entirely, so the default path is
	// byte-for-byte unchanged. When a bound is declared, refuse to grow retained dense bytes
	// past it by name rather than silently exceeding the declared budget and inviting the
	// kernel OOM this bound exists to prevent. A refused charge must not retain: panic before
	// the memoizing assignment.
	if qt.denseBound != nil {
		incoming := int64(len(raw))
		if !qt.denseBound.chargeRetained(incoming) {
			panic(denseResidentBoundPanic(op, qt.denseBound.bound, incoming, qt.denseBound.retained))
		}
	}
	qt.raw = raw
}

// requireRawCPU is the #13202 legibility guardrail for the CPU k-quant matmul entry points
// (kQuantMatRowsRangeRaw and its siblings). A lazy k-quant tensor holds no resident raw
// bytes until materializeRaw runs, and the CPU fallback does `if len(raw)==0 { raw = qt.raw }`
// — over a lazy tensor both are empty, so the row loop would silently read a nil slice and
// emit ZEROS instead of failing. That is a silent-wrong-answer bug, so this turns it into a
// named panic: the tensor must be materialized (or read through a device path) first.
//
// The CPU entry points call ensureRawCPU first, so this is now the BACKSTOP for a tensor that
// is genuinely unmaterializable (no reader / no lazy range / a failed read), not the ordinary
// path for a bounded lazy range.
func (qt *kQuantTensor) requireRawCPU(op string) {
	if qt != nil && len(qt.raw) == 0 && qt.lazy != nil && qt.out > 0 {
		panic(fmt.Sprintf("model: k-quant %s on a lazy tensor (kind=%s out=%d in=%d): the resident "+
			"raw bytes are a checkpoint range that has not been materialized, and a CPU k-quant "+
			"matmul ran — reading nil raw would silently produce zeros. Call materializeRaw first, "+
			"or route the matmul through a device path that reads the lazy range (#13202).",
			op, qt.kind, qt.out, qt.in))
	}
}

// The exported per-type lazy k-quant entries below mirror the AddResidentQ2K/AddResidentQ3K/
// AddResidentQ5K/AddResidentQ6K family in quant_kquant.go: kQuantKind is unexported, so
// AddLazyKQuant's signature is unreachable for a caller outside package model. These wrappers
// let such a caller key the primitive on a CONCRETE format instead of the private kind, while
// the block geometry is still selected inside package model — so a caller cannot supply a
// mismatched kind/geometry pair. Each is a one-line delegation to AddLazyKQuant.

// AddLazyKQuantQ2K stores a checkpoint-backed Q2_K descriptor without reading its payload.
func (b *QuantBuilder) AddLazyKQuantQ2K(canon string, shape []int, src LazyQ4KRange) error {
	return b.AddLazyKQuant(canon, shape, kindQ2K, src)
}

// AddLazyKQuantQ3K stores a checkpoint-backed Q3_K descriptor without reading its payload.
func (b *QuantBuilder) AddLazyKQuantQ3K(canon string, shape []int, src LazyQ4KRange) error {
	return b.AddLazyKQuant(canon, shape, kindQ3K, src)
}

// AddLazyKQuantQ5K stores a checkpoint-backed Q5_K descriptor without reading its payload.
func (b *QuantBuilder) AddLazyKQuantQ5K(canon string, shape []int, src LazyQ4KRange) error {
	return b.AddLazyKQuant(canon, shape, kindQ5K, src)
}

// AddLazyKQuantQ6K stores a checkpoint-backed Q6_K descriptor without reading its payload.
func (b *QuantBuilder) AddLazyKQuantQ6K(canon string, shape []int, src LazyQ4KRange) error {
	return b.AddLazyKQuant(canon, shape, kindQ6K, src)
}

// AddLazyKQuant stores a checkpoint-backed k-quant descriptor without reading its payload.
// It runs the SAME eligibility gate as the resident k-quant entries (residentQuantTarget),
// validates the byte count against the kind's block geometry, and stores a kQuantTensor
// that holds only the range — no payload read. Symmetric with AddLazyQ4K.
//
// Reachability constraint: kind is the unexported kQuantKind, so this entry point is callable
// only from within package model. Out-of-package callers must use the exported per-type
// wrappers above (AddLazyKQuantQ2K/Q3K/Q5K/Q6K), which pin the kind for them.
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
			// Shared model-level ledger; nil unless a dense bound was declared.
			denseBound: b.m.denseLedger(),
		}
	})
}

// KQuantLazy reports whether a k-quant tensor is backed by a checkpoint range rather than
// resident bytes. Exported diagnostic twin of Q4KLazy.
func (m *Model) KQuantLazy(name string) bool { return m.kqw[name] != nil && m.kqw[name].lazy != nil }
