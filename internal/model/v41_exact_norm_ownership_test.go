package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ExactNormOwnershipBackend struct {
	*v41ExactNormTestBackend
	mode             string
	activation, gain compute.Tensor
	output           compute.Tensor
	foreign          compute.Tensor
	frees            map[compute.Buffer]int
	reads            int
	primary, cleanup error
}

func (b *v41ExactNormOwnershipBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 {
		panic("unexpected ownership fixture dtype")
	}
	out := compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
	switch class {
	case compute.MemoryWeights:
		b.gain = out
	case compute.MemoryActivation:
		b.activation = out
	default:
		panic("unexpected ownership fixture memory class")
	}
	return out
}

func (b *v41ExactNormOwnershipBackend) V41ExactRMSNorm(x, weight compute.Tensor, eps float32) (compute.Tensor, error) {
	switch b.mode {
	case "activation_alias":
		b.calls++
		return x, nil
	case "cached_gain_alias":
		b.calls++
		return weight, nil
	case "foreign_owner":
		b.calls++
		b.foreign = compute.NewF32(compute.Default(), append([]int(nil), x.Shape...), make([]float32, x.Shape[0]))
		return b.foreign, nil
	}
	out, err := b.v41ExactNormTestBackend.V41ExactRMSNorm(x, weight, eps)
	if err != nil {
		return out, err
	}
	b.output = out
	switch b.mode {
	case "wrong_shape", "wrong_shape_cleanup_fault":
		out.Shape = []int{x.Shape[0] / 2}
	case "wrong_layout":
		out.Layout = compute.ColMajor
	case "owned_output_and_error":
		return out, b.primary
	}
	return out, nil
}

func (b *v41ExactNormOwnershipBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(x)
}

func (b *v41ExactNormOwnershipBackend) Free(x compute.Tensor) {
	b.frees[x.Buf()]++
	b.Backend.Free(x)
	if b.cleanup != nil && x.Buf() == b.output.Buf() {
		panic(b.cleanup)
	}
}

// CPU storage witnesses model ownership and failure handling, not native frees.
// The real model-owned gain pool must survive a session's rejected output.
// fak-test:runtime fast est=50ms lane=default
func TestV41ExactNormOutputOwnership(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("exact norm ownership witness requires cpu-ref")
	}
	for _, mode := range []string{"activation_alias", "cached_gain_alias", "foreign_owner", "wrong_shape", "wrong_layout", "wrong_shape_cleanup_fault", "owned_output_and_error", "healthy"} {
		t.Run(mode, func(t *testing.T) {
			const eps = float32(1e-6)
			gain, input := v41ExactNormTestRow(128, 0.01), v41ExactNormTestRow(128, 2)
			m := v41IndexKeyNormTinyModel(t, gain, eps)
			b := &v41ExactNormOwnershipBackend{v41ExactNormTestBackend: newV41ExactNormTestBackend(), mode: mode,
				frees: map[compute.Buffer]int{}, primary: errors.New("exact norm primary failure")}
			b.outputOwner = b
			if mode == "wrong_shape_cleanup_fault" || mode == "owned_output_and_error" {
				b.cleanup = &compute.BackendError{Backend: "exact-norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New("output cleanup failure")}
			}
			t.Cleanup(func() {
				// End injected transient faults before actual model teardown.
				b.cleanup = nil
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
			s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
			t.Cleanup(s.Close)
			normalize := s.v41IndexKeyNormFunc()
			if normalize == nil {
				t.Fatal("exact operation was not selected")
			}
			values, err := normalize(0, input, gain, eps)
			if mode == "healthy" {
				if err != nil || s.BackendSessionClosed() || !v41CompressorNormEqualBits(values, v41IndexKeyNormReference(input, gain, eps)) || b.reads != 1 {
					t.Fatalf("healthy fresh output changed: values=%v error=%v", values, err)
				}
			} else {
				want := error(errV41ProjectionResult)
				if mode == "owned_output_and_error" {
					want = b.primary
				}
				var closed *BackendForwardOperationError
				if values != nil || !errors.As(err, &closed) || closed.Cause != want || closed.Path != "v41-index-key-norm" || !s.BackendSessionClosed() || b.reads != 0 {
					t.Fatalf("malformed output did not fail closed with primary cause: values=%v error=%v reads=%d", values, err, b.reads)
				}
			}
			if b.calls != 1 || b.frees[b.activation.Buf()] != 1 || b.frees[b.gain.Buf()] != 0 {
				t.Fatalf("borrowed/activation ownership changed: calls=%d activation releases=%d gain releases=%d", b.calls, b.frees[b.activation.Buf()], b.frees[b.gain.Buf()])
			}
			if b.output.Buf() != nil && b.frees[b.output.Buf()] != 1 {
				t.Fatalf("owned output released %d times", b.frees[b.output.Buf()])
			}
			if b.foreign.Buf() != nil && b.frees[b.foreign.Buf()] != 0 {
				t.Fatal("foreign output was released through the wrong backend")
			}
			residents := 0
			if !m.VisitImmutableDeviceWeights(b, func(w compute.Tensor) bool { residents++; return w.Buf() == b.gain.Buf() }) || residents != 1 {
				t.Fatal("model-owned cached gain did not survive session cleanup")
			}
			if mode != "healthy" {
				retry := v41IndexerTestRecover(func() { _, _ = normalize(0, input, gain, eps) })
				if retry != s.halFailure || b.calls != 1 || b.frees[b.activation.Buf()] != 1 || b.frees[b.gain.Buf()] != 0 {
					t.Fatal("closed session repeated the operation or release")
				}
			}
		})
	}
}
