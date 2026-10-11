package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This CPU recorder exercises the selected exact operation, not physical Vulkan.
type v41ExactNormCleanupBackend struct {
	*v41ExactNormTestBackend
	fault          string
	primary        any
	cleanup        [2]any
	owned          []compute.Buffer
	freeAttempts   map[compute.Buffer]int
	uploads, reads int
}

func (b *v41ExactNormCleanupBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt == compute.F32 && class == compute.MemoryWeights && site == "hal-weight "+layerName(0, "indexer.k_norm.weight") {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	if dt != compute.F32 || class != compute.MemoryActivation || site != "V4.1 RMSNorm activation "+layerName(0, "indexer.k_norm.weight") {
		panic("unexpected exact RMSNorm fixture upload")
	}
	b.uploads++
	out := b.v41DenseTestBackend.Upload(x, dt)
	b.owned = append(b.owned, out.Buf())
	return out
}

func (b *v41ExactNormCleanupBackend) V41ExactRMSNorm(x, weight compute.Tensor, eps float32) (compute.Tensor, error) {
	if b.fault == "dispatch_panic" {
		b.calls++
		panic(b.primary)
	}
	out, err := b.v41ExactNormTestBackend.V41ExactRMSNorm(x, weight, eps)
	if err == nil {
		b.owned = append(b.owned, out.Buf())
	}
	return out, err
}

func (b *v41ExactNormCleanupBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	if b.fault == "read_panic" {
		panic(b.primary)
	}
	if b.fault == "short_read" {
		return nil
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41ExactNormCleanupBackend) Free(x compute.Tensor) {
	for i, buf := range b.owned {
		if x.Buf() == buf {
			b.freeAttempts[buf]++
			if b.freeAttempts[buf] == 1 {
				b.v41DenseTestBackend.Free(x)
			}
			if b.cleanup[i] != nil {
				panic(b.cleanup[i])
			}
			return
		}
	}
	// Cached immutable gain is outside the transient cleanup fault injection.
	b.v41DenseTestBackend.Free(x)
}

// Ref #13668. Existing defers already attempt both transient releases. The
// regression is cause precedence through the actual index-key norm consumer,
// including its typed closed-session latch and prohibition on replay.
// fak-test:runtime fast est=50ms lane=default
func TestV41ExactNormPrimaryFailureSurvivesCleanup(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("exact norm fault witness requires cpu-ref")
	}
	backendError := func(message string) error {
		return &compute.BackendError{Backend: "exact-norm-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New(message)}
	}
	for _, cleanup := range []struct {
		name   string
		values [2]any
	}{
		{"typed_cleanup", [2]any{backendError("activation release"), backendError("output release")}},
		{"unknown_cleanup", [2]any{"activation release panic", "output release panic"}},
	} {
		for _, tc := range []struct {
			name, fault  string
			panicValue   any
			owned, reads int
		}{
			{name: "returned_error", fault: "dispatch_error", owned: 1},
			{name: "typed_dispatch_panic", fault: "dispatch_panic", owned: 1},
			{name: "typed_read_panic", fault: "read_panic", owned: 2, reads: 1},
			{name: "short_read", fault: "short_read", owned: 2, reads: 1},
			{name: "unknown_panic", fault: "read_panic", panicValue: &struct{ marker int }{13668}, owned: 2, reads: 1},
			{name: "successful_operation", owned: 2, reads: 1},
		} {
			t.Run(cleanup.name+"/"+tc.name, func(t *testing.T) {
				const eps = float32(1e-6)
				gain := v41ExactNormTestRow(128, 0.01)
				primary := backendError("primary operation failure")
				b := &v41ExactNormCleanupBackend{v41ExactNormTestBackend: newV41ExactNormTestBackend(),
					fault: tc.fault, primary: primary, cleanup: cleanup.values, freeAttempts: map[compute.Buffer]int{}}
				b.outputOwner = b
				if tc.panicValue != nil {
					b.primary = tc.panicValue
				}
				if tc.fault == "dispatch_error" {
					b.fail = primary
				}
				s := &Session{M: v41IndexKeyNormTinyModel(t, gain, eps), Backend: b, halW: map[string]compute.Tensor{}}
				t.Cleanup(s.Close)
				normalize := s.v41IndexKeyNormFunc()
				if normalize == nil {
					t.Fatal("actual exact norm callback did not bind")
				}
				var values []float32
				var err error
				input := v41ExactNormTestRow(128, 2)
				recovered := v41IndexerTestRecover(func() { values, err = normalize(0, input, gain, eps) })
				wantCause, wantPanic := any(primary), tc.panicValue
				switch tc.fault {
				case "short_read":
					wantCause = errV41ProjectionResult
				case "":
					wantCause = cleanup.values[1] // Output releases before activation.
					if _, ok := wantCause.(error); !ok {
						wantPanic = wantCause
					}
				default:
					if tc.panicValue != nil {
						wantCause = tc.panicValue
					}
				}
				if recovered != wantPanic {
					t.Fatalf("panic identity changed: got %v want %v", recovered, wantPanic)
				}
				var closed *BackendForwardOperationError
				if values != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Path != "v41-index-key-norm" || closed.Layer != 0 {
					t.Fatalf("selected failure lost closed identity: values=%v error=%v", values, s.halFailure)
				}
				if expected, ok := wantCause.(error); ok && closed.Cause != expected {
					t.Fatalf("primary cause changed: got %v want %v", closed.Cause, expected)
				}
				if wantPanic == nil && !errors.Is(err, s.halFailure) {
					t.Fatalf("returned error lost closed identity: %v", err)
				}
				if len(b.owned) != tc.owned || len(b.freeAttempts) != tc.owned || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatalf("unexpected attempts: owned=%d frees=%d uploads=%d calls=%d reads=%d", len(b.owned), len(b.freeAttempts), b.uploads, b.calls, b.reads)
				}
				for _, attempts := range b.freeAttempts {
					if attempts != 1 {
						t.Fatal("transient release was not attempted exactly once")
					}
				}
				retry := v41IndexerTestRecover(func() { _, _ = normalize(0, input, gain, eps) })
				if retry != s.halFailure || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatal("closed callback retried work or changed failure identity")
				}
			})
		}
	}
}
