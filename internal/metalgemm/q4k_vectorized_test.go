//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"math"
	"slices"
	"sort"
	"testing"
	"time"
)

// q4kVectorizedReference is an independent CPU transcription of the 144-byte Q4_K layout.
// It proves both Metal arms against the format rather than treating the existing scalar kernel
// as its own oracle.
func q4kVectorizedReference(raw []byte, out, in int, x []float32) []float32 {
	nblk := in / q4kTestBlockWeights
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q4kTestBlockBytes
			blk := raw[base : base+q4kTestBlockBytes]
			d := math.Float32frombits(q4kTestF16Bits(binary.LittleEndian.Uint16(blk[0:2])))
			dmin := math.Float32frombits(q4kTestF16Bits(binary.LittleEndian.Uint16(blk[2:4])))
			scales := blk[4:16]
			q := blk[16:]
			xb := x[b*q4kTestBlockWeights:]
			qi, is := 0, 0
			for j := 0; j < q4kTestBlockWeights; j += 64 {
				s0, m0 := q4kVectorizedScaleMin(is, scales)
				s1, m1 := q4kVectorizedScaleMin(is+1, scales)
				d0, dm0 := d*float32(s0), dmin*float32(m0)
				d1, dm1 := d*float32(s1), dmin*float32(m1)
				for l := 0; l < 32; l++ {
					acc += (d0*float32(q[qi+l]&0x0f) - dm0) * xb[j+l]
				}
				for l := 0; l < 32; l++ {
					acc += (d1*float32(q[qi+l]>>4) - dm1) * xb[j+32+l]
				}
				qi += 32
				is += 2
			}
		}
		y[o] = acc
	}
	return y
}

func q4kVectorizedScaleMin(j int, scales []byte) (byte, byte) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	return (scales[j+4] & 0x0f) | ((scales[j-4] >> 6) << 4),
		(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
}

func q4kTestF16Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return sign | uint32(e+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(int32(exp)-15+127)<<23 | frac<<13
	}
}

func q4kVectorizedMedian(ds []time.Duration) time.Duration {
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	return ds[len(ds)/2]
}

func TestQ4KVectorizedP1Candidate(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetGEMVUseVectorized(false)

	const out, in = 5120, 5120 // Qwen3.8-27B hidden projection shape.
	raw := q4kTestRaw(out, in, 0x8965)
	x := q4kTestVector(in, 8965)
	w := UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	control := make([]float32, out)
	candidate := make([]float32, out)
	SetGEMVUseVectorized(false)
	controlObservation := NewExecutionObservation(ExecutionQ4KGEMV)
	if executed := w.gemvWithEvents(x, control, controlObservation); executed != q4kGEMVExecutedScalar {
		t.Fatalf("scalar selector executed status = %d, want %d", executed, q4kGEMVExecutedScalar)
	}
	requireCompletedExecution(t, controlObservation, ExecutionQ4KGEMV)
	SetGEMVUseVectorized(true)
	candidateObservation := NewExecutionObservation(ExecutionQ4KGEMV)
	if executed := w.gemvWithEvents(x, candidate, candidateObservation); executed != q4kGEMVExecutedVectorized {
		t.Fatalf("vector selector executed status = %d, want %d", executed, q4kGEMVExecutedVectorized)
	}
	requireCompletedExecution(t, candidateObservation, ExecutionQ4KGEMV)
	t.Logf("executed pipeline identity: control=scalar candidate=vectorized")

	// A requested vector pipeline that is unavailable must not substitute the scalar PSO. The
	// native result status, empty lifecycle observation, and untouched output jointly prove no
	// command buffer was dispatched and no scalar result was smuggled through.
	unavailable := make([]float32, out)
	for i := range unavailable {
		unavailable[i] = 8965
	}
	unavailableBefore := slices.Clone(unavailable)
	unavailableObservation := NewExecutionObservation(ExecutionQ4KGEMV)
	if executed := w.gemvWithEventsMode(x, unavailable, unavailableObservation, q4kGEMVModeVectorizedUnavailable); executed != q4kGEMVNotExecuted {
		t.Fatalf("unavailable vector pipeline executed status = %d, want no dispatch", executed)
	}
	unavailableSnapshot, err := unavailableObservation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(unavailableSnapshot.Events) != 0 {
		t.Fatalf("unavailable vector pipeline events = %+v, want none", unavailableSnapshot.Events)
	}
	if !slices.Equal(unavailable, unavailableBefore) {
		t.Fatal("unavailable vector pipeline changed output; scalar fallback or partial dispatch occurred")
	}
	t.Log("unavailable vector pipeline: status=not-executed events=0 output=untouched")
	reference := q4kVectorizedReference(raw, out, in, x)

	for name, got := range map[string][]float32{"control": control, "candidate": candidate} {
		cosine, maxRel := q4kTestCosineMaxRel(reference, got)
		if cosine < 0.999999 || maxRel > 5e-3 {
			t.Fatalf("%s vs CPU reference: cosine=%g maxRel=%g, want cosine >= 0.999999 and maxRel <= 5e-3", name, cosine, maxRel)
		}
		t.Logf("%s vs CPU reference: cosine=%.9f maxRel=%g", name, cosine, maxRel)
	}
	cosine, maxRel := q4kTestCosineMaxRel(control, candidate)
	if cosine < 0.999999 || maxRel > 5e-3 {
		t.Fatalf("candidate vs scalar control: cosine=%g maxRel=%g, want cosine >= 0.999999 and maxRel <= 5e-3", cosine, maxRel)
	}
	t.Logf("candidate vs scalar control: cosine=%.9f maxRel=%g", cosine, maxRel)

	// Time both pipelines in this process, against the same resident weight, input, buffers, and
	// compiled library. Alternating first-arm order keeps thermal/order bias out of the median.
	const rounds, callsPerRound = 11, 32
	controlTimes := make([]time.Duration, 0, rounds)
	candidateTimes := make([]time.Duration, 0, rounds)
	measure := func(vectorized bool) time.Duration {
		SetGEMVUseVectorized(vectorized)
		start := time.Now()
		for i := 0; i < callsPerRound; i++ {
			w.GEMV(x, candidate)
		}
		return time.Since(start) / callsPerRound
	}
	measure(false)
	measure(true)
	for round := 0; round < rounds; round++ {
		if round%2 == 0 {
			controlTimes = append(controlTimes, measure(false))
			candidateTimes = append(candidateTimes, measure(true))
		} else {
			candidateTimes = append(candidateTimes, measure(true))
			controlTimes = append(controlTimes, measure(false))
		}
	}
	controlMedian := q4kVectorizedMedian(controlTimes)
	candidateMedian := q4kVectorizedMedian(candidateTimes)
	speedup := float64(controlMedian) / float64(candidateMedian)
	t.Logf("same-binary P=1 [out=%d,in=%d]: scalar=%s candidate=%s speedup=%.3fx", out, in, controlMedian, candidateMedian, speedup)
	if candidateMedian >= controlMedian {
		t.Fatalf("vectorized candidate did not beat scalar control: speedup=%.3fx", speedup)
	}
}
