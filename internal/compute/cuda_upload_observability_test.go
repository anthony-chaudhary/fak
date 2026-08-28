//go:build cuda

package compute

import "testing"

type cudaImmutableUploadCounters struct {
	calls         uint64
	transferBytes uint64
	residentBytes uint64
}

func cudaImmutableUploadCountersNow(cb *cudaBackend) cudaImmutableUploadCounters {
	calls, transferBytes, residentBytes := cb.CUDAImmutableWeightUploadSnapshot()
	return cudaImmutableUploadCounters{calls: calls, transferBytes: transferBytes, residentBytes: residentBytes}
}

func cudaImmutableUploadDelta(before, after cudaImmutableUploadCounters) cudaImmutableUploadCounters {
	return cudaImmutableUploadCounters{
		calls:         after.calls - before.calls,
		transferBytes: after.transferBytes - before.transferBytes,
		residentBytes: after.residentBytes - before.residentBytes,
	}
}

func TestCUDAImmutableWeightUploadSnapshotCountsOnlySuccessfulCacheMisses(t *testing.T) {
	cb := cudaOrSkip(t)
	data := make([]float32, 2*32)
	for i := range data {
		data[i] = float32(i + 1)
	}
	host := NewF32(Default(), []int{2, 32}, data)
	before := cudaImmutableUploadCountersNow(cb)
	first := cb.Upload(host, F16)
	afterFirst := cudaImmutableUploadCountersNow(cb)
	if got, want := cudaImmutableUploadDelta(before, afterFirst), (cudaImmutableUploadCounters{calls: 1, transferBytes: 2 * 32 * 4, residentBytes: 2 * 32 * 2}); got != want {
		t.Fatalf("first F16 immutable upload delta = %+v, want %+v", got, want)
	}

	cached := cb.Upload(host, F16)
	if cached.buf != first.buf {
		t.Fatal("same immutable host pointer did not hit the CUDA upload cache")
	}
	if got := cudaImmutableUploadDelta(afterFirst, cudaImmutableUploadCountersNow(cb)); got != (cudaImmutableUploadCounters{}) {
		t.Fatalf("cached immutable upload delta = %+v, want zero", got)
	}

	freshData := append([]float32(nil), data...)
	fresh := cb.Upload(NewF32(Default(), []int{2, 32}, freshData), F16)
	afterFresh := cudaImmutableUploadCountersNow(cb)
	if got, want := cudaImmutableUploadDelta(afterFirst, afterFresh), (cudaImmutableUploadCounters{calls: 1, transferBytes: 2 * 32 * 4, residentBytes: 2 * 32 * 2}); got != want {
		t.Fatalf("fresh immutable upload delta = %+v, want %+v", got, want)
	}

	activation := cb.UploadClass(NewF32(Default(), []int{2, 2}, []float32{1, 2, 3, 4}), F32, MemoryActivation, "cuda-upload-observability-activation")
	if got := cudaImmutableUploadDelta(afterFresh, cudaImmutableUploadCountersNow(cb)); got != (cudaImmutableUploadCounters{}) {
		t.Fatalf("activation upload delta = %+v, want zero", got)
	}

	afterActivation := cudaImmutableUploadCountersNow(cb)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("classed activation narrowing did not refuse")
			}
		}()
		cb.UploadClass(NewF32(Default(), []int{2, 2}, []float32{1, 2, 3, 4}), F16, MemoryActivation, "cuda-upload-observability-refused")
	}()
	if got := cudaImmutableUploadDelta(afterActivation, cudaImmutableUploadCountersNow(cb)); got != (cudaImmutableUploadCounters{}) {
		t.Fatalf("refused upload delta = %+v, want zero", got)
	}

	cb.Free(first)
	cb.Free(fresh)
	cb.Free(activation)
}

func TestCUDAImmutableWeightUploadSnapshotUsesActualTransferAndResidentLayouts(t *testing.T) {
	cb := cudaOrSkip(t)
	const out, in = 2, 256
	cases := []struct {
		name          string
		host          func() Tensor
		as            Dtype
		transferBytes uint64
		residentBytes uint64
	}{
		{name: "f32", host: func() Tensor { return NewF32(Default(), []int{out, in}, make([]float32, out*in)) }, as: F32, transferBytes: out * in * 4, residentBytes: out * in * 4},
		{name: "f16", host: func() Tensor { return NewF32(Default(), []int{out, in}, make([]float32, out*in)) }, as: F16, transferBytes: out * in * 4, residentBytes: out * in * 2},
		{name: "q8-from-f32", host: func() Tensor { return NewF32(Default(), []int{out, in}, make([]float32, out*in)) }, as: Q8_0, transferBytes: out*in + out*(in/q8DeviceBlock)*4, residentBytes: out*in + out*(in/q8DeviceBlock)*4},
		{name: "q8-resident", host: func() Tensor {
			return NewQ8(Default(), []int{out, in}, make([]int8, out*in), make([]float32, out*(in/q8DeviceBlock)), q8DeviceBlock)
		}, as: Q8_0, transferBytes: out*in + out*(in/q8DeviceBlock)*4, residentBytes: out*in + out*(in/q8DeviceBlock)*4},
		{name: "q2-resident", host: func() Tensor {
			return NewQ2(Default(), []int{out, in}, make([]byte, out*in/4), make([]float32, out*(in/q8DeviceBlock)), q8DeviceBlock)
		}, as: Q2_0, transferBytes: out*in/4 + out*(in/q8DeviceBlock)*4, residentBytes: out*in/4 + out*(in/q8DeviceBlock)*4},
		{name: "q4-raw", host: func() Tensor { return NewQ4K(Default(), []int{out, in}, make([]byte, out*144)) }, as: Q4_K, transferBytes: out * 144, residentBytes: out * 144},
		{name: "q5-raw", host: func() Tensor { return NewQ5K(Default(), []int{out, in}, make([]byte, out*176)) }, as: Q5_K, transferBytes: out * 176, residentBytes: out * 176},
		{name: "q6-raw", host: func() Tensor { return NewQ6K(Default(), []int{out, in}, make([]byte, out*210)) }, as: Q6_K, transferBytes: out * 210, residentBytes: out * 210},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := cudaImmutableUploadCountersNow(cb)
			resident := cb.Upload(tc.host(), tc.as)
			after := cudaImmutableUploadCountersNow(cb)
			want := cudaImmutableUploadCounters{calls: 1, transferBytes: tc.transferBytes, residentBytes: tc.residentBytes}
			if got := cudaImmutableUploadDelta(before, after); got != want {
				t.Fatalf("immutable upload delta = %+v, want %+v", got, want)
			}
			cb.Free(resident)
		})
	}
}
