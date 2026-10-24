//go:build cuda && cgo

package compute

import (
	"math"
	"os"
	"testing"
)

const cudaNVFP4RequireEnv = "FAK_REQUIRE_CUDA_NVFP4"

func cudaNVFP4OrSkip(t *testing.T) *cudaBackend {
	t.Helper()
	be := Pick("cuda")
	cb, ok := be.(*cudaBackend)
	if !ok {
		message := "cuda NVFP4 backend not registered (no reachable CUDA device); set " + cudaNVFP4RequireEnv + "=1 on HIL to fail instead of skip"
		if os.Getenv(cudaNVFP4RequireEnv) == "1" {
			t.Fatal(message)
		}
		t.Skip(message)
	}
	return cb
}

func cudaNVFP4Resident(t *testing.T, cb *cudaBackend, shape []int, packed, scales []byte, global float32) Tensor {
	t.Helper()
	host, err := NewNVFP4(cb, shape, packed, scales, global)
	if err != nil {
		t.Fatal(err)
	}
	return cb.Upload(host, NVFP4)
}

func cudaNVFP4CPUWeight(t *testing.T, ref *cpuBackend, shape []int, packed, scales []byte, global float32) Tensor {
	t.Helper()
	w, err := NewNVFP4(ref, shape, packed, scales, global)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func requireNVFP4Exact(t *testing.T, tag string, want, got []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", tag, len(got), len(want))
	}
	for i := range want {
		if !nvfp4Finite(got[i]) {
			t.Fatalf("%s[%d] is non-finite: %v", tag, i, got[i])
		}
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %v (%08x), want CPU W4A16 %v (%08x)", tag, i,
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// fak-test:runtime integration est=1s lane=optin
func TestCUDANVFP4W4A16Parity(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	ref := &cpuBackend{}

	t.Run("finite-product-domain", func(t *testing.T) {
		// Every E2M1 code against every finite non-negative E4M3FN scale byte.
		// One active K element per row makes each pair independently observable.
		const out, in = 16 * 127, 16
		packed := make([]byte, out*in/2)
		scales := make([]byte, out)
		for row := 0; row < out; row++ {
			packed[row*in/2] = byte(row % 16)
			scales[row] = byte(row / 16)
		}
		x := make([]float32, in)
		x[0] = 1

		cpuW := cudaNVFP4CPUWeight(t, ref, []int{out, in}, packed, scales, 1)
		gpuW := cudaNVFP4Resident(t, cb, []int{out, in}, packed, scales, 1)
		want := ref.Read(ref.MatMul(cpuW, NewF32(ref, []int{in}, x)))
		got := cb.Read(cb.MatMul(gpuW, cb.Upload(NewF32(cb, []int{in}, x), F32)))
		requireNVFP4Exact(t, "product-domain MatMul", want, got)
	})

	t.Run("bf16-rne-batch-and-late-global", func(t *testing.T) {
		const out, in, batch = 3, 32, 3
		packed := make([]byte, out*in/2)
		for i := range packed {
			packed[i] = byte(i*17 + 0x32)
		}
		scales := []byte{0x01, 0x39, 0x7e, 0x28, 0, 0x40}
		x := make([]float32, batch*in)
		for i := range x {
			x[i] = float32(i%13-6)*0.137 + 0.003
		}
		// Exact BF16 RNE ties: even lower BF16 stays at 1; odd rounds upward.
		x[0] = math.Float32frombits(0x3f808000)
		x[1] = math.Float32frombits(0x3f818000)
		x[2] = math.Float32frombits(0xbf818000)

		cpuW := cudaNVFP4CPUWeight(t, ref, []int{out, in}, packed, scales, 0.1)
		gpuW := cudaNVFP4Resident(t, cb, []int{out, in}, packed, scales, 0.1)
		want := ref.Read(ref.BatchedMatMul(cpuW, NewF32(ref, []int{batch, in}, x), batch))
		got := cb.Read(cb.BatchedMatMul(gpuW, cb.Upload(NewF32(cb, []int{batch, in}, x), F32), batch))
		requireNVFP4Exact(t, "BatchedMatMul", want, got)
		for token := 0; token < batch; token++ {
			row := x[token*in : (token+1)*in]
			wantRow := ref.Read(ref.MatMul(cpuW, NewF32(ref, []int{in}, row)))
			gotRow := cb.Read(cb.MatMul(gpuW, cb.Upload(NewF32(cb, []int{in}, row), F32)))
			requireNVFP4Exact(t, "MatMul", wantRow, gotRow)
		}

		// This cancellation fixture has a stable bit pattern only when the global
		// scale is applied once after the serial FP32 reduction, never per weight.
		codes := []byte{10, 7, 14, 5, 6, 5, 3, 14, 13, 13, 10, 12, 5, 2, 1, 4}
		latePacked := make([]byte, 8)
		for i, code := range codes {
			latePacked[i/2] |= code << uint(4*(i%2))
		}
		lateX := []float32{0.125, 0.125, -1, -2, 0.125, 1.375, 1, 1.375, 123, 1.375, 123, -2, 0.75, 0.75, -1, 0.125}
		lateCPU := cudaNVFP4CPUWeight(t, ref, []int{1, 16}, latePacked, []byte{0x38}, 0.1)
		lateGPU := cudaNVFP4Resident(t, cb, []int{1, 16}, latePacked, []byte{0x38}, 0.1)
		lateWant := ref.Read(ref.MatMul(lateCPU, NewF32(ref, []int{16}, lateX)))
		if bits := math.Float32bits(lateWant[0]); bits != 0xc2440ccd {
			t.Fatalf("late-global CPU fixture changed: got %08x, want c2440ccd", bits)
		}
		lateGot := cb.Read(cb.MatMul(lateGPU, cb.Upload(NewF32(cb, []int{16}, lateX), F32)))
		requireNVFP4Exact(t, "late-global MatMul", lateWant, lateGot)
	})
}

// fak-test:runtime integration est=100ms lane=optin
func TestCUDANVFP4ResidentContract(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	if !cb.SupportsDeviceWeightDtype(NVFP4) {
		t.Fatal("cuda backend does not advertise resident NVFP4 MatMul capability")
	}

	const out, in = 7, 32
	packed := make([]byte, out*in/2)
	scales := make([]byte, out*(in/NVFP4BlockSize))
	for i := range packed {
		packed[i] = byte(i*37 + 11)
	}
	for i := range scales {
		scales[i] = byte(i % 0x7f)
	}
	w := cudaNVFP4Resident(t, cb, []int{out, in}, packed, scales, 0.25)
	if w.Dtype != NVFP4 || w.Layout != RowMajor {
		t.Fatalf("resident tensor = dtype %s layout %v, want row-major NVFP4", w.Dtype, w.Layout)
	}
	b, ok := w.buf.(*cudaBuf)
	if !ok || b == nil {
		t.Fatalf("resident NVFP4 buffer = %T, want *cudaBuf", w.buf)
	}
	wantBytes := len(packed) + len(scales)
	if b.n != wantBytes || b.scalesN != 0 || b.residentBytes() != wantBytes {
		t.Fatalf("resident NVFP4 bytes = primary %d side-scale %d total %d, want packed payload %d",
			b.n, b.scalesN, b.residentBytes(), wantBytes)
	}
	cb.Free(w)

	// Uploading one host owner twice must produce independent device allocations.
	// Otherwise freeing either tensor leaves the other pointing at released VRAM.
	host, err := NewNVFP4(cb, []int{out, in}, packed, scales, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	first := cb.Upload(host, NVFP4)
	second := cb.Upload(host, NVFP4)
	firstBuf, firstOK := first.buf.(*cudaBuf)
	secondBuf, secondOK := second.buf.(*cudaBuf)
	if !firstOK || !secondOK || firstBuf == nil || secondBuf == nil {
		t.Fatalf("repeat upload buffers = (%T, %T), want two *cudaBuf values", first.buf, second.buf)
	}
	if firstBuf.ptr == secondBuf.ptr {
		t.Fatal("repeat NVFP4 uploads alias one device allocation")
	}
	cb.Free(first)

	ref := &cpuBackend{}
	x := make([]float32, in)
	x[0], x[16] = math.Float32frombits(0x3f818000), -0.25
	cpuW := cudaNVFP4CPUWeight(t, ref, []int{out, in}, packed, scales, 0.25)
	want := ref.Read(ref.MatMul(cpuW, NewF32(ref, []int{in}, x)))
	got := cb.Read(cb.MatMul(second, cb.Upload(NewF32(cb, []int{in}, x), F32)))
	requireNVFP4Exact(t, "second upload after first free", want, got)
	cb.Free(second)
}
