package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CPU storage exercises the selected callback's failure contract, not hardware.
type v41IndexSelectCleanupBackend struct {
	*v41DenseTestBackend
	fault                 string
	primary               error
	primaryPanic          any
	cleanup               [2]any
	allocations           []compute.Buffer
	freeAttempts          map[compute.Buffer]int
	uploads, calls, reads int
}

func (b *v41IndexSelectCleanupBackend) V41IndexSelectAvailable() bool { return true }

func (b *v41IndexSelectCleanupBackend) allocate(shape []int, values []float32) compute.Tensor {
	x := compute.NewF32(b, append([]int(nil), shape...), append([]float32(nil), values...))
	b.allocations = append(b.allocations, x.Buf())
	return x
}

func (b *v41IndexSelectCleanupBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if dt != compute.F32 || class != compute.MemoryActivation || site != "V4.1 index-select scores" {
		panic("unexpected index-select upload")
	}
	b.uploads++
	return b.allocate(x.Shape, b.Backend.Read(x))
}

func (b *v41IndexSelectCleanupBackend) V41IndexSelect(scores compute.Tensor, rows, topK, topKBlocks, blockSize int) (compute.Tensor, error) {
	if scores.Backend() != b || !reflect.DeepEqual(scores.Shape, []int{rows}) || rows != 2 || topK != 1 || topKBlocks != 0 {
		panic("invalid index-select cleanup fixture")
	}
	b.calls++
	if b.fault == "dispatch" {
		return compute.Tensor{}, b.primary
	}
	if b.fault == "panic" {
		panic(b.primaryPanic)
	}
	out := b.allocate([]int{1}, []float32{0})
	if b.fault == "partial" {
		return out, b.primary
	}
	return out, nil
}

func (b *v41IndexSelectCleanupBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	if b.fault == "read" {
		panic(b.primaryPanic)
	}
	return b.Backend.Read(x)
}

func (b *v41IndexSelectCleanupBackend) Free(x compute.Tensor) {
	for i, buf := range b.allocations {
		if x.Buf() == buf {
			b.freeAttempts[buf]++
			if b.freeAttempts[buf] == 1 {
				b.Backend.Free(x)
			}
			if b.cleanup[i] != nil {
				panic(b.cleanup[i])
			}
			return
		}
	}
	panic("unexpected index-select release")
}

// Refs #13668. This selected-path software regression does not qualify the
// default-off indexer-score shader tracked separately in #13806.
// fak-test:runtime fast est=5ms lane=default
func TestV41IndexSelectPrimaryFailureSurvivesCleanup(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("index-select cleanup witness requires cpu-ref")
	}
	backendError := func(message string) error {
		return &compute.BackendError{Backend: "index-select-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New(message)}
	}
	for _, cleanup := range []struct {
		name   string
		values [2]any
	}{
		{"typed cleanup", [2]any{backendError("score release"), backendError("output release")}},
		{"unknown cleanup", [2]any{"score release panic", "output release panic"}},
	} {
		for _, tc := range []struct {
			name, fault  string
			panicValue   any
			owned, reads int
		}{
			{name: "returned error", fault: "dispatch", owned: 1},
			{name: "partial output error", fault: "partial", owned: 2},
			{name: "typed dispatch panic", fault: "panic", owned: 1},
			{name: "typed read panic", fault: "read", owned: 2, reads: 1},
			{name: "pointer panic", fault: "panic", panicValue: &struct{ marker int }{17}, owned: 1},
			{name: "error panic", fault: "panic", panicValue: errors.New("original unknown panic"), owned: 1},
			{name: "string panic", fault: "panic", panicValue: "original panic", owned: 1},
			{name: "successful operation", owned: 2, reads: 1},
		} {
			t.Run(cleanup.name+"/"+tc.name, func(t *testing.T) {
				primary := backendError("primary failure")
				b := &v41IndexSelectCleanupBackend{v41DenseTestBackend: newV41DenseTestBackend(), fault: tc.fault,
					primary: primary, primaryPanic: primary, cleanup: cleanup.values, freeAttempts: map[compute.Buffer]int{}}
				if tc.panicValue != nil {
					b.primaryPanic = tc.panicValue
				}
				s := &Session{M: &Model{}, Backend: b, halW: map[string]compute.Tensor{}}
				t.Cleanup(s.Close)
				selectRows := s.v41IndexSelectFunc()
				if selectRows == nil {
					t.Fatal("software callback was not selected")
				}
				var out []int32
				var err error
				recovered := v41IndexerTestRecover(func() { out, err = selectRows(4, []float32{2, 1}, 1, 0, 1) })
				wantCause, wantPanic := any(primary), tc.panicValue
				if tc.fault == "" {
					// The output is released first; a later score-release panic
					// must not replace that first cleanup-only failure.
					wantCause = cleanup.values[1]
					if _, ok := wantCause.(error); !ok {
						wantPanic = wantCause
					}
				} else if tc.panicValue != nil {
					wantCause = tc.panicValue
				}
				if recovered != wantPanic {
					t.Fatalf("panic identity changed: got %v want %v", recovered, wantPanic)
				}
				var closed *BackendForwardOperationError
				if out != nil || !s.BackendSessionClosed() || !errors.As(s.halFailure, &closed) ||
					closed.Layer != 4 || closed.Path != "v41-index-select" || closed.Forward != ForwardPathKind("deepseek41") {
					t.Fatalf("failure did not retire the selected session: out=%v failure=%v", out, s.halFailure)
				}
				if expected, ok := wantCause.(error); ok && closed.Cause != expected {
					t.Fatalf("primary cause identity changed: got %v want %v", closed.Cause, expected)
				}
				if wantPanic == nil && err != s.halFailure {
					t.Fatalf("returned failure differs from latched failure: %v / %v", err, s.halFailure)
				}
				if len(b.allocations) != tc.owned || len(b.freeAttempts) != tc.owned || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatalf("unexpected work or release count: allocated=%d freed=%d uploads=%d calls=%d reads=%d", len(b.allocations), len(b.freeAttempts), b.uploads, b.calls, b.reads)
				}
				for _, buf := range b.allocations {
					if b.freeAttempts[buf] != 1 {
						t.Fatalf("owned buffer released %d times", b.freeAttempts[buf])
					}
				}
				retry := v41IndexerTestRecover(func() { _, _ = selectRows(4, []float32{2, 1}, 1, 0, 1) })
				if retry != s.halFailure || b.uploads != 1 || b.calls != 1 || b.reads != tc.reads {
					t.Fatal("closed callback did not preserve failure or retried work")
				}
				for _, attempts := range b.freeAttempts {
					if attempts != 1 {
						t.Fatal("closed callback retried cleanup")
					}
				}
			})
		}
	}
}
