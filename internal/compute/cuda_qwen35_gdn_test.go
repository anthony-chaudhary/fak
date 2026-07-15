//go:build cuda

package compute

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"unsafe"
)

// Set FAK_CUDA_GDN_REQUIRED=1 on a hardware acceptance node. In required mode
// absence of the exact CUDA backend is a hard failure, never a skip or cpu-ref
// substitution.
const cudaGDNRequiredEnv = "FAK_CUDA_GDN_REQUIRED"

func cudaGDNBackend(t *testing.T) *cudaBackend {
	t.Helper()
	be, ok := Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatalf("%s=1: real CUDA GDN fixture required, but exact backend cuda is not registered", cudaGDNRequiredEnv)
		}
		t.Skip("exact cuda backend not registered (set FAK_CUDA_GDN_REQUIRED=1 on an acceptance node to fail rather than skip)")
	}
	cuda, ok := be.(*cudaBackend)
	if !ok {
		t.Fatalf("registered cuda backend has unexpected concrete type %T", be)
	}
	return cuda
}

type cudaGDNGeometry struct {
	hidden, nK, nV, kHd, vHd, kernel int
	eps                              float32
}

func (g cudaGDNGeometry) keyDim() int   { return g.nK * g.kHd }
func (g cudaGDNGeometry) valueDim() int { return g.nV * g.vHd }
func (g cudaGDNGeometry) convDim() int  { return 2*g.keyDim() + g.valueDim() }

type cudaGDNOperands struct {
	x, inQKV, inZ, inB, inA         Tensor
	convW, aLog, dtBias, norm       Tensor
	outW, convState, recurrentState Tensor
}

type cudaGDNLCG uint64

func (r *cudaGDNLCG) vector(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		*r = *r*6364136223846793005 + 1442695040888963407
		out[i] = (float32(uint32(*r>>32))/float32(uint64(1)<<32) - 0.5) * scale
	}
	return out
}

func uploadCUDAGDN(t *testing.T, be *cudaBackend, shape []int, data []float32, class MemoryClass, site string) Tensor {
	t.Helper()
	host := NewF32(Default(), shape, data)
	var resident Tensor
	if class == MemoryWeights {
		resident = be.Upload(host, F32)
	} else {
		resident = be.UploadClass(host, F32, class, site)
	}
	t.Cleanup(func() { be.Free(resident) })
	return resident
}

func newCUDAGDNOperands(t *testing.T, be *cudaBackend, g cudaGDNGeometry) cudaGDNOperands {
	t.Helper()
	rng := cudaGDNLCG(0x4738c0da)
	weight := func(shape []int, scale float32, site string) Tensor {
		n := 1
		for _, dimension := range shape {
			n *= dimension
		}
		return uploadCUDAGDN(t, be, shape, rng.vector(n, scale), MemoryWeights, site)
	}
	state := func(shape []int, site string) Tensor {
		n := 1
		for _, dimension := range shape {
			n *= dimension
		}
		return uploadCUDAGDN(t, be, shape, make([]float32, n), MemoryKVCache, site)
	}
	x := uploadCUDAGDN(t, be, []int{g.hidden}, rng.vector(g.hidden, 0.5), MemoryActivation, "qwen35-gdn-input")
	normData := rng.vector(g.vHd, 0.1)
	for i := range normData {
		normData[i] += 1
	}
	aLogData := rng.vector(g.nV, 0.2)
	for i := range aLogData {
		aLogData[i] -= 0.6
	}
	return cudaGDNOperands{
		x:              x,
		inQKV:          weight([]int{g.convDim(), g.hidden}, 0.2, "in-qkv"),
		inZ:            weight([]int{g.valueDim(), g.hidden}, 0.2, "in-z"),
		inB:            weight([]int{g.nV, g.hidden}, 0.2, "in-b"),
		inA:            weight([]int{g.nV, g.hidden}, 0.2, "in-a"),
		convW:          weight([]int{g.convDim(), 1, g.kernel}, 0.3, "conv"),
		aLog:           uploadCUDAGDN(t, be, []int{g.nV}, aLogData, MemoryWeights, "a-log"),
		dtBias:         weight([]int{g.nV}, 0.2, "dt-bias"),
		norm:           uploadCUDAGDN(t, be, []int{g.vHd}, normData, MemoryWeights, "norm"),
		outW:           weight([]int{g.hidden, g.valueDim()}, 0.2, "out"),
		convState:      state([]int{g.kernel - 1, g.convDim()}, "qwen35-gdn-conv-state"),
		recurrentState: state([]int{g.nV, g.kHd, g.vHd}, "qwen35-gdn-recurrent-state"),
	}
}

func (o cudaGDNOperands) decode(be *cudaBackend, g cudaGDNGeometry) (Tensor, Tensor, Tensor, error) {
	return be.Qwen35GDNDecode(
		o.x, o.inQKV, o.inZ, o.inB, o.inA, o.convW, o.aLog, o.dtBias, o.norm, o.outW,
		o.convState, o.recurrentState,
		g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps,
	)
}

func requireFiniteNonzeroCUDA(t *testing.T, name string, values []float32) {
	t.Helper()
	var norm float64
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s[%d] is non-finite: %v", name, i, value)
		}
		norm += float64(value) * float64(value)
	}
	if norm == 0 {
		t.Fatalf("%s has degenerate zero norm", name)
	}
}

func TestCUDAQwen35GDNKernelOneCompletesInPlaceDeviceOnly(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	g := cudaGDNGeometry{hidden: 8, nK: 1, nV: 2, kHd: 2, vHd: 2, kernel: 1, eps: 1e-5}
	o := newCUDAGDNOperands(t, be, g)
	convIdentity, recurrentIdentity := o.convState.Buf(), o.recurrentState.Buf()

	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.ResetQwen35GDNOperationCount()
	out, nextConv, nextRecurrent, err := o.decode(be, g)
	if err != nil {
		t.Fatalf("K=1 CUDA GDN decode: %v", err)
	}
	if got := be.Qwen35GDNOperationCount(); got != 1 {
		t.Fatalf("completed operation count = %d, want 1", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("H2D bytes inside K=1 operation = %d, want 0", got)
	}
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("D2H bytes inside K=1 operation = %d, want 0", got)
	}
	if nextConv.Buf() != convIdentity || nextRecurrent.Buf() != recurrentIdentity {
		t.Fatal("K=1 mutable state did not preserve in-place buffer identity")
	}
	if !out.Ready() {
		t.Fatal("synchronized whole-operation output is not Ready")
	}
	if outputBuf := out.buf.(*cudaBuf); outputBuf.managed || outputBuf.class != MemoryScratchpad {
		t.Fatalf("strict output allocation managed=%v class=%q, want device-only scratchpad", outputBuf.managed, outputBuf.class)
	}
	gotOut := be.Read(out)
	recurrent := be.Read(nextRecurrent)
	recycledConv, recycledRecurrent := nextConv.Buf(), nextRecurrent.Buf()
	be.Recycle()
	if nextConv.Buf() != recycledConv || nextRecurrent.Buf() != recycledRecurrent || !nextConv.Ready() || !nextRecurrent.Ready() {
		t.Fatal("MemoryKVCache state identity/readiness did not survive Recycle")
	}
	if got := be.Read(nextConv); len(got) != 0 {
		t.Fatalf("K=1 conv state length = %d, want 0", len(got))
	}
	requireFiniteNonzeroCUDA(t, "K=1 output", gotOut)
	requireFiniteNonzeroCUDA(t, "K=1 recurrent state", recurrent)
}

func TestCUDAQwen35GDNLaunchAndAsyncFailuresInvalidateState(t *testing.T) {
	for _, tc := range []struct {
		name, stageName string
		stage           int
	}{
		{name: "launch", stage: 3, stageName: "causal-conv-state"},
		{name: "async", stage: 7, stageName: "stream-synchronize"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := cudaGDNBackend(t)
			t.Cleanup(be.Recycle)
			t.Cleanup(func() { qwen35GDNInjectFaultForTest(0) })
			g := cudaGDNGeometry{hidden: 8, nK: 1, nV: 2, kHd: 2, vHd: 2, kernel: 3, eps: 1e-5}
			o := newCUDAGDNOperands(t, be, g)
			be.ResetQwen35GDNOperationCount()
			qwen35GDNInjectFaultForTest(tc.stage)
			out, nextConv, nextRecurrent, err := o.decode(be, g)
			var kernel *Qwen35GDNKernelError
			if !errors.As(err, &kernel) || kernel.Stage != tc.stageName {
				t.Fatalf("fault stage %d error = %T %v, want stage %q", tc.stage, err, err, tc.stageName)
			}
			if out.Backend() != nil || nextConv.Backend() != nil || nextRecurrent.Backend() != nil {
				t.Fatal("failed operation returned usable output/state tensors")
			}
			if got := be.Qwen35GDNOperationCount(); got != 0 {
				t.Fatalf("failed operation count = %d, want 0", got)
			}
			if o.convState.Ready() || o.recurrentState.Ready() {
				t.Fatal("failed in-place state still reports Ready")
			}
			for name, state := range map[string]Tensor{"conv_state": o.convState, "recurrent_state": o.recurrentState} {
				func() {
					defer func() {
						recovered := recover()
						var invalid *Qwen35GDNInvalidStateError
						if !errors.As(asErrorCompute(recovered), &invalid) {
							t.Fatalf("Read(%s) panic = %T %v, want *Qwen35GDNInvalidStateError", name, recovered, recovered)
						}
					}()
					_ = be.Read(state)
				}()
			}
		})
	}
}

func TestCUDAQwen35GDNManagedOperandRefusesBeforeMutation(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	g := cudaGDNGeometry{hidden: 8, nK: 1, nV: 2, kHd: 2, vHd: 2, kernel: 3, eps: 1e-5}
	o := newCUDAGDNOperands(t, be, g)
	beforeConv := append([]float32(nil), be.Read(o.convState)...)
	beforeRecurrent := append([]float32(nil), be.Read(o.recurrentState)...)

	cudaMu.Lock()
	managedBuf := be.dallocManagedClass(g.hidden*F32.Bytes(), MemoryActivation, "qwen35-gdn-forced-managed-test")
	cudaMu.Unlock()
	managedInput := makeTensor(be, F32, RowMajor, []int{g.hidden}, nil, managedBuf)
	t.Cleanup(func() { be.Free(managedInput) })
	o.x = managedInput

	be.ResetHostXfer()
	be.ResetH2DXfer()
	be.ResetQwen35GDNOperationCount()
	_, _, _, err := o.decode(be, g)
	var residency *Qwen35GDNResidencyError
	if !errors.As(err, &residency) || residency.Operand != "normalized_input" || !strings.Contains(err.Error(), "managed memory") {
		t.Fatalf("managed refusal = %T %v", err, err)
	}
	if got := be.Qwen35GDNOperationCount(); got != 0 {
		t.Fatalf("managed refusal operation count = %d, want 0", got)
	}
	if got := be.H2DXferBytes(); got != 0 {
		t.Fatalf("managed refusal H2D delta = %d, want 0", got)
	}
	if got := be.HostXferBytes(); got != 0 {
		t.Fatalf("managed refusal D2H delta = %d, want 0", got)
	}
	afterConv := be.Read(o.convState)
	afterRecurrent := be.Read(o.recurrentState)
	if !equalF32Bits(beforeConv, afterConv) || !equalF32Bits(beforeRecurrent, afterRecurrent) {
		t.Fatal("managed refusal mutated durable GDN state")
	}
}

func TestCUDAQwen35GDNWrongStateClassRefusesWithoutOperation(t *testing.T) {
	for _, class := range []MemoryClass{MemoryWeights, MemoryScratchpad} {
		t.Run(string(class), func(t *testing.T) {
			be := cudaGDNBackend(t)
			t.Cleanup(be.Recycle)
			g := cudaGDNGeometry{hidden: 8, nK: 1, nV: 2, kHd: 2, vHd: 2, kernel: 3, eps: 1e-5}
			o := newCUDAGDNOperands(t, be, g)
			beforeConv := append([]float32(nil), be.Read(o.convState)...)
			stateBuf := o.convState.buf.(*cudaBuf)
			stateBuf.class = class

			be.ResetHostXfer()
			be.ResetH2DXfer()
			be.ResetQwen35GDNOperationCount()
			_, _, _, err := o.decode(be, g)
			var residency *Qwen35GDNResidencyError
			if !errors.As(err, &residency) || residency.Operand != "conv_state" || !strings.Contains(err.Error(), "kv_cache") {
				t.Fatalf("state class %q refusal = %T %v", class, err, err)
			}
			if got := be.Qwen35GDNOperationCount(); got != 0 {
				t.Fatalf("state class %q operation count = %d, want 0", class, got)
			}
			if got := be.H2DXferBytes(); got != 0 {
				t.Fatalf("state class %q H2D delta = %d, want 0", class, got)
			}
			if got := be.HostXferBytes(); got != 0 {
				t.Fatalf("state class %q D2H delta = %d, want 0", class, got)
			}
			stateBuf.class = MemoryKVCache
			if afterConv := be.Read(o.convState); !equalF32Bits(beforeConv, afterConv) {
				t.Fatalf("state class %q refusal mutated convolution state", class)
			}
		})
	}
}

func TestCUDAQwen35GDNTensorMetadataRefusals(t *testing.T) {
	be := &cudaBackend{name: "cuda"}
	storage := make([]byte, 12)
	shapes := qwen35GDNGeometryTensors(8, 1, 2, 2, 2, 3)
	tensors := make([]Tensor, len(shapes))
	for i := range shapes {
		nbytes, ok := qwen35GDNShapeBytes(shapes[i].Shape, F32.Bytes())
		if !ok {
			t.Fatalf("fixture shape %v overflowed", shapes[i].Shape)
		}
		class := MemoryWeights
		if i == 0 {
			class = MemoryActivation
		}
		if i >= 10 {
			class = MemoryKVCache
		}
		tensors[i] = makeTensor(be, F32, RowMajor, shapes[i].Shape, nil, &cudaBuf{
			ptr: unsafe.Pointer(&storage[i]), n: nbytes, class: class,
		})
	}

	t.Run("capacity", func(t *testing.T) {
		buf := tensors[0].buf.(*cudaBuf)
		buf.n--
		err := be.validateQwen35GDNTensor("normalized_input", tensors[0])
		var residency *Qwen35GDNResidencyError
		if !errors.As(err, &residency) || !strings.Contains(err.Error(), "shape requires") {
			t.Fatalf("undersized allocation error = %T %v", err, err)
		}
		buf.n++
	})

	operands := []struct {
		name string
		t    Tensor
	}{
		{"normalized_input", tensors[0]}, {"in_proj_qkv", tensors[1]},
		{"in_proj_z", tensors[2]}, {"in_proj_b", tensors[3]}, {"in_proj_a", tensors[4]},
		{"conv1d", tensors[5]}, {"A_log", tensors[6]}, {"dt_bias", tensors[7]},
		{"norm", tensors[8]}, {"out_proj", tensors[9]},
		{"conv_state", tensors[10]}, {"recurrent_state", tensors[11]},
	}
	t.Run("state-class", func(t *testing.T) {
		buf := tensors[10].buf.(*cudaBuf)
		for _, class := range []MemoryClass{MemoryWeights, MemoryScratchpad} {
			buf.class = class
			err := be.validateQwen35GDNStateOperands(operands, tensors[10], tensors[11])
			var residency *Qwen35GDNResidencyError
			if !errors.As(err, &residency) || !strings.Contains(err.Error(), "kv_cache") {
				t.Fatalf("state class %q error = %T %v", class, err, err)
			}
		}
		buf.class = MemoryKVCache
	})

	t.Run("state-readonly-alias", func(t *testing.T) {
		state := tensors[10].buf.(*cudaBuf)
		old := state.ptr
		state.ptr = tensors[0].buf.(*cudaBuf).ptr
		err := be.validateQwen35GDNStateOperands(operands, tensors[10], tensors[11])
		var residency *Qwen35GDNResidencyError
		if !errors.As(err, &residency) || !strings.Contains(err.Error(), "aliases mutable GDN state") {
			t.Fatalf("state/read-only alias error = %T %v", err, err)
		}
		state.ptr = old
	})
}

func equalF32Bits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func asErrorCompute(value any) error {
	err, _ := value.(error)
	return err
}
