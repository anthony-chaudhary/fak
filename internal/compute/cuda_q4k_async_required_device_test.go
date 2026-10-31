//go:build fak_cuda_q4k_async_witness && cuda && cgo && fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select && fak_cuda_q4k_qkv_async

package compute

import (
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
)

// Required-device, isolated-process qualification only. This proves the explicit
// async owner boundary, not ordinary serving admission or model throughput.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAQ4KAsyncOwnedPhysical(t *testing.T) {
	if os.Getenv("FAK_CUDA_Q4K_ASYNC_ACCEPT") != "1" || os.Getenv("FAK_CUDA_GRAPH") != "0" {
		t.Fatal("required Q4 async witness needs FAK_CUDA_Q4K_ASYNC_ACCEPT=1 and FAK_CUDA_GRAPH=0")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	be, registered := Lookup("cuda")
	c, ok := be.(*cudaBackend)
	if !registered || !ok || c == nil || c != cudaDev || RequireReference(c) {
		t.Fatal("required registered actual CUDA backend unavailable; no fallback")
	}
	if CUDAQ4KDecodeArmInUse() != CUDAQ4KDecodeMMVQ {
		t.Fatal("required Q4 MMVQ arm not selected; set FAK_CUDA_Q4K_DECODE=mmvq before process startup")
	}
	cudaMu.Lock()
	clean := !c.capturing && len(c.transient) == 0 && cudaQ4KAsyncOwner.count == 0 && cudaQ4KAsyncOwner.failure == 0 && cudaQ4KAsyncOwner.owner == nil
	cudaMu.Unlock()
	if !clean {
		t.Fatal("required isolated fresh owner/transient state unavailable")
	}
	// Cleanup uses the existing checked Free/Recycle paths. A sticky failed drain
	// retains quarantine; it must not be bypassed to satisfy an allocator check.
	var owned []Tensor
	defer func() {
		defer func() {
			if p := recover(); p != nil {
				t.Errorf("checked cleanup failed (primary failure retained): %v", p)
			}
		}()
		for _, tensor := range owned {
			c.Free(tensor)
		}
		c.Recycle()
	}()
	deviceBuffer := func(tensor Tensor) *cudaBuf {
		t.Helper()
		b, ok := tensor.buf.(*cudaBuf)
		if !ok || b == nil || b.ptr == nil {
			t.Fatal("required live CUDA tensor owner missing")
		}
		return b
	}
	const in = 256
	input := make([]float32, in)
	for i := range input {
		input[i] = 1
		if i%4 == 0 {
			input[i] = -1
		}
	}
	x := c.Upload(NewF32(c, []int{in}, input), F32)
	owned = append(owned, x)
	inputBuf := deviceBuffer(x)
	ref, foundReference := Lookup("cpu-ref")
	if !foundReference || !RequireReference(ref) {
		t.Fatal("required exact CPU reference unavailable")
	}
	var weights [3]Tensor
	var wants [3][]float32
	var weightBufs [3]*cudaBuf
	for matrix, rows := range []int{1, 2, 3} {
		raw := make([]byte, rows*144)
		for row := 0; row < rows; row++ {
			block := raw[row*144 : (row+1)*144]
			binary.LittleEndian.PutUint16(block, 0x3c00) // d=1, dmin=0.
			for i := 4; i < 16; i++ {
				block[i] = 1 // Existing Q4_K scale packing; every group scale=1.
			}
			code := byte(matrix + row + 1)
			for i := 16; i < len(block); i++ {
				block[i] = code | code<<4
			}
		}
		wants[matrix] = ref.Read(ref.MatMul(NewQ4K(ref, []int{rows, in}, raw), NewF32(ref, []int{in}, input)))
		for row, value := range wants[matrix] {
			if value != float32(128*(matrix+row+1)) {
				t.Fatal("deterministic packed Q4 fixture disagrees with existing CPU reference")
			}
		}
		weights[matrix] = c.Upload(NewQ4K(c, []int{rows, in}, raw), Q4_K)
		owned = append(owned, weights[matrix])
		weightBufs[matrix] = deviceBuffer(weights[matrix])
	}
	before := c.Q4KQKVSharedInputQualificationOperations()
	var outputs [2][3]Tensor
	var outputBufs [2][3]*cudaBuf
	for call := range outputs {
		q, k, v, used, err := c.Q4KQKVSharedInputQualification(weights[0], weights[1], weights[2], x)
		if err != nil || !used {
			t.Fatalf("required selected async call %d used=%t error=%v; no retry", call, used, err)
		}
		outputs[call] = [3]Tensor{q, k, v}
		owned = append(owned, q, k, v)
		for i, output := range outputs[call] {
			outputBufs[call][i] = deviceBuffer(output)
		}
	}
	// The counter advances only inside async_finish after a successful checked
	// drain. Hardware may already have finished; timing is NOT the assertion.
	if got := c.Q4KQKVSharedInputQualificationOperations(); got != before {
		t.Fatalf("enqueue mislabeled as checked-owner completion: %d -> %d", before, got)
	}
	var scratch [2]*cudaBuf
	cudaMu.Lock()
	p := &cudaQ4KAsyncOwner
	valid := p.owner == c && p.count == 2 && p.bytes == 2*(in/32)*36 && p.failure == 0
	if valid {
		for call := range outputs {
			scratch[call] = p.scratch[call]
			valid = valid && scratch[call] != nil && scratch[call].ptr != nil && scratch[call].n == (in/32)*36
			want := [7]*cudaBuf{weightBufs[0], weightBufs[1], weightBufs[2], inputBuf, outputBufs[call][0], outputBufs[call][1], outputBufs[call][2]}
			valid = valid && p.dependencies[call] == want
			for _, output := range outputs[call] {
				b, ok := output.buf.(*cudaBuf)
				valid = valid && ok && b != nil && b.ptr != nil && !b.Ready()
			}
		}
		valid = valid && scratch[0] != scratch[1]
	}
	cudaMu.Unlock()
	if !valid {
		t.Fatal("pending scratch/dependency ownership or unpublished readiness changed before checked drain")
	}
	// Public Free must checked-drain both queued calls BEFORE recycling input.
	c.Free(x)
	if got := c.Q4KQKVSharedInputQualificationOperations(); got != before+2 {
		t.Fatalf("checked Free completion=%d, want %d", got, before+2)
	}
	cudaMu.Lock()
	valid = inputBuf.ptr == nil && p.owner == nil && p.count == 0 && p.bytes == 0 && p.failure == 0
	for i := range p.scratch {
		valid = valid && p.scratch[i] == nil && p.dependencies[i] == [7]*cudaBuf{}
	}
	cudaMu.Unlock()
	if !valid {
		t.Fatal("checked Free did not clear owner inventory before releasing input")
	}
	for _, call := range outputs {
		for matrix, output := range call {
			if !output.Ready() {
				t.Fatal("checked completion did not publish output readiness")
			}
			got := c.Read(output)
			if len(got) != len(wants[matrix]) {
				t.Fatal("Q4 result extent changed")
			}
			for row, value := range got {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-wants[matrix][row])) > 0.02 {
					t.Fatalf("Q4 output[%d,%d]=%g reference=%g", matrix, row, value, wants[matrix][row])
				}
			}
		}
	}
	c.Recycle()
	cudaMu.Lock()
	valid = len(c.transient) == 0
	for _, b := range scratch {
		valid = valid && b.ptr == nil
	}
	for _, call := range outputBufs {
		for _, output := range call {
			valid = valid && output.ptr == nil
		}
	}
	cudaMu.Unlock()
	if !valid {
		t.Fatal("checked owner retirement left test transients allocated")
	}
}
