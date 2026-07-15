//go:build cuda

package compute

import (
	"errors"
	"testing"
	"unsafe"
)

func cudaInvalidStateFixture(be *cudaBackend, dtype Dtype, shape []int, invalid bool) Tensor {
	storage := new(byte)
	buffer := &cudaBuf{
		ptr:   unsafe.Pointer(storage),
		n:     1,
		class: MemoryKVCache,
	}
	if invalid {
		buffer.invalid = 1
	}
	var quant *QuantSpec
	if dtype == Q8_0 {
		quant = &QuantSpec{Block: q8DeviceBlock, Axis: 2, Bits: 8, Symmetric: true}
	}
	return makeTensor(be, dtype, RowMajor, append([]int(nil), shape...), quant, buffer)
}

func requireCUDAInvalidStatePanic(t *testing.T, invoke func()) {
	t.Helper()
	defer func() {
		recovered := recover()
		var invalid *Qwen35GDNInvalidStateError
		if !errors.As(asErrorCompute(recovered), &invalid) {
			t.Fatalf("consumer panic = %T %v, want *Qwen35GDNInvalidStateError", recovered, recovered)
		}
		if invalid.Operand != "buffer" {
			t.Fatalf("invalid-state operand = %q, want buffer", invalid.Operand)
		}
	}()
	invoke()
	t.Fatal("invalid CUDA buffer reached consumer without a panic")
}

// TestCUDAInvalidStateDirectConsumersRefuseBeforeLaunch exercises the direct
// pointer paths found by the #4743 audit. The fixture uses non-device sentinel
// pointers deliberately: every subtest must reject the poison bit before any C
// allocator, copy, or kernel can observe those pointers, so it is deterministic
// even on a tagged build host with no registered CUDA device.
func TestCUDAInvalidStateDirectConsumersRefuseBeforeLaunch(t *testing.T) {
	be := &cudaBackend{name: "cuda"}
	assertNoTransient := func(t *testing.T) {
		t.Helper()
		if got := len(be.transient); got != 0 {
			t.Fatalf("invalid consumer retained %d transient allocation(s), want 0", got)
		}
	}

	t.Run("rope", func(t *testing.T) {
		x := cudaInvalidStateFixture(be, F32, []int{2}, true)
		requireCUDAInvalidStatePanic(t, func() { be.RoPE(x, 0, 1, 2, 10000) })
		assertNoTransient(t)
	})

	t.Run("q8-matmul", func(t *testing.T) {
		w := cudaInvalidStateFixture(be, Q8_0, []int{2, q8DeviceBlock}, true)
		x := cudaInvalidStateFixture(be, F32, []int{q8DeviceBlock}, false)
		requireCUDAInvalidStatePanic(t, func() { be.MatMul(w, x) })
		assertNoTransient(t)
	})

	t.Run("q4k-batched-matmul", func(t *testing.T) {
		w := cudaInvalidStateFixture(be, Q4_K, []int{2, 256}, true)
		x := cudaInvalidStateFixture(be, F32, []int{1, 256}, false)
		requireCUDAInvalidStatePanic(t, func() { be.BatchedMatMul(w, x, 1) })
		assertNoTransient(t)
	})

	t.Run("awq", func(t *testing.T) {
		w := cudaInvalidStateFixture(be, Q4_K, []int{2, 4}, false)
		scales := cudaInvalidStateFixture(be, F32, []int{2}, false)
		x := cudaInvalidStateFixture(be, F32, []int{4}, true)
		requireCUDAInvalidStatePanic(t, func() { be.AWQMatMul(w, scales, x) })
		assertNoTransient(t)
	})

	t.Run("batched-awq", func(t *testing.T) {
		w := cudaInvalidStateFixture(be, Q4_K, []int{2, 4}, false)
		scales := cudaInvalidStateFixture(be, F32, []int{2}, false)
		x := cudaInvalidStateFixture(be, F32, []int{1, 4}, true)
		requireCUDAInvalidStatePanic(t, func() { be.AWQBatchedMatMul(w, scales, x, 1) })
		assertNoTransient(t)
	})

	t.Run("gptq", func(t *testing.T) {
		qweight := cudaInvalidStateFixture(be, F32, []int{1}, false)
		qzeros := cudaInvalidStateFixture(be, F32, []int{1}, false)
		scales := cudaInvalidStateFixture(be, F32, []int{1}, false)
		gidx := cudaInvalidStateFixture(be, F32, []int{1}, true)
		x := cudaInvalidStateFixture(be, F32, []int{1}, false)
		requireCUDAInvalidStatePanic(t, func() {
			be.GPTQMatMul(qweight, qzeros, scales, gidx, x, 1, 1, 4, 1, 1)
		})
		assertNoTransient(t)
	})

	t.Run("kv-append-preflights-all-sources", func(t *testing.T) {
		kv := &cudaKV{
			be:   be,
			cfg:  KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 1},
			K:    make([]dslice, 1),
			Kraw: make([]dslice, 1),
			V:    make([]dslice, 1),
		}
		kRaw := cudaInvalidStateFixture(be, F32, []int{1}, false)
		kRoPE := cudaInvalidStateFixture(be, F32, []int{1}, true)
		value := cudaInvalidStateFixture(be, F32, []int{1}, false)
		requireCUDAInvalidStatePanic(t, func() { kv.AppendKV(0, kRaw, kRoPE, value, 0) })
		if kv.K[0].len != 0 || kv.Kraw[0].len != 0 || kv.V[0].len != 0 || len(kv.pos) != 0 {
			t.Fatalf("invalid append mutated KV lengths: K=%d Kraw=%d V=%d pos=%d", kv.K[0].len, kv.Kraw[0].len, kv.V[0].len, len(kv.pos))
		}
		assertNoTransient(t)
	})
}
