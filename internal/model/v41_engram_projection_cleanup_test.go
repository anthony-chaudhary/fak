package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The recorder frees CPU storage before raising distinct cleanup faults. It
// proves release attempts and cause precedence, not physical deallocation.
type v41EngramCleanupBackend struct {
	*v41EngProjBackend
	fault                 string
	primary               any
	cleanup               [2]any
	owned                 []compute.Buffer
	freeAttempts          map[compute.Buffer]int
	uploads, calls, reads int
}

func (b *v41EngramCleanupBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 || class != compute.MemoryActivation || site != "V4.1 Engram projection activation" {
		panic("unexpected Engram cleanup fixture upload")
	}
	b.uploads++
	out := b.v41EngProjBackend.Upload(x, dt)
	b.owned = append(b.owned, out.Buf())
	return out
}

func (b *v41EngramCleanupBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.calls++
	if b.fault == "matmul" {
		panic(b.primary)
	}
	out := b.v41EngProjBackend.MatMul(w, x)
	b.owned = append(b.owned, out.Buf())
	return out
}

func (b *v41EngramCleanupBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	if b.fault == "read" {
		panic(b.primary)
	}
	values := b.v41EngProjBackend.Read(x)
	if b.fault == "length" {
		return nil
	}
	if b.fault == "finite" {
		values[len(values)-1] = math.Float32frombits(0x7f7f8000)
	}
	return values
}

func (b *v41EngramCleanupBackend) Free(x compute.Tensor) {
	for i, buf := range b.owned {
		if x.Buf() == buf {
			b.freeAttempts[buf]++
			if b.freeAttempts[buf] == 1 {
				b.v41EngProjBackend.Free(x)
			}
			if b.cleanup[i] != nil {
				panic(b.cleanup[i])
			}
			return
		}
	}
	// Immutable cached weights are outside the two transient release faults.
	b.v41EngProjBackend.Free(x)
}

// Existing defers already attempt both releases. This regression demonstrates
// only preservation of the primary cause and first-cleanup-failure ordering.
// It drives the actual session callback with the published BF16 validation.
// fak-test:runtime medium est=1s lane=default
func TestV41EngramProjectionPrimaryFailureSurvivesCleanup(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("Engram cleanup software witness requires cpu-ref")
	}
	m, _ := v41FullEngramModel(t)
	stage := m.v41EngramStageFor()
	in, out := stage.columns*stage.headDim, (stage.hc+1)*m.Cfg.HiddenSize
	backendError := func(message string) error {
		return &compute.BackendError{Backend: "engram-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New(message)}
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
			{name: "typed_matmul_panic", fault: "matmul", owned: 1},
			{name: "typed_read_panic", fault: "read", owned: 2, reads: 1},
			{name: "short_read_error", fault: "length", owned: 2, reads: 1},
			{name: "bf16_overflow_error", fault: "finite", owned: 2, reads: 1},
			{name: "pointer_panic", fault: "read", panicValue: &struct{ marker int }{13668}, owned: 2, reads: 1},
			{name: "plain_error_panic", fault: "read", panicValue: errors.New("original read panic"), owned: 2, reads: 1},
			{name: "string_panic", fault: "read", panicValue: "original read panic", owned: 2, reads: 1},
			{name: "successful_operation", owned: 2, reads: 1},
		} {
			t.Run(cleanup.name+"/"+tc.name, func(t *testing.T) {
				primary := backendError("primary failure")
				b := &v41EngramCleanupBackend{v41EngProjBackend: newV41EngProjBackend(m), fault: tc.fault, primary: primary,
					cleanup: cleanup.values, freeAttempts: map[compute.Buffer]int{}}
				if tc.panicValue != nil {
					b.primary = tc.panicValue
				}
				s := v41EngProjSession(t, m, b)
				project := s.v41EngramProjectionFunc()
				if project == nil {
					t.Fatal("actual Engram callback was not selected")
				}
				var values []float32
				var outcome v41DenseProjectionOutcome
				var err error
				recovered := v41IndexerTestRecover(func() { values, outcome, err = project(0, make([]float32, in), out, in) })
				wantCause, wantPanic := any(primary), tc.panicValue
				switch tc.fault {
				case "length", "finite":
					wantCause = errV41ProjectionResult
				case "":
					wantCause = cleanup.values[1] // output release precedes activation
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
				if values != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) || closed.Layer != 0 || closed.Path != "v41-engram-projection" {
					t.Fatalf("selected failure did not retire session: values=%v error=%v", values, s.halFailure)
				}
				if expected, ok := wantCause.(error); ok && closed.Cause != expected {
					t.Fatalf("primary cause changed: got %v want %v", closed.Cause, expected)
				}
				if wantPanic == nil && (err != s.halFailure || outcome != v41ProjectionError) {
					t.Fatalf("returned outcome/error lost closed selected identity: outcome=%v error=%v", outcome, err)
				}
				if len(b.owned) != tc.owned || len(b.freeAttempts) != tc.owned || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatalf("unexpected attempt counts: owned=%d frees=%d uploads=%d calls=%d reads=%d", len(b.owned), len(b.freeAttempts), b.uploads, b.calls, b.reads)
				}
				for _, buf := range b.owned {
					if b.freeAttempts[buf] != 1 {
						t.Fatalf("transient release attempted %d times", b.freeAttempts[buf])
					}
				}
				retry := v41IndexerTestRecover(func() { _, _, _ = project(0, make([]float32, in), out, in) })
				if retry != s.halFailure || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatal("closed callback retried work or changed failure identity")
				}
				for _, attempts := range b.freeAttempts {
					if attempts != 1 {
						t.Fatal("closed callback retried release")
					}
				}
			})
		}
	}
}
