package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Actual CPU kernels exercise the live expertSwiGLU fallback. Only this call's
// transient handles are poisoned/faulted; cached weight ownership stays separate.
type expertHALCleanupRecorder struct {
	compute.Backend
	owned                            []compute.Buffer
	labels                           map[compute.Buffer]string
	frees                            map[compute.Buffer]int
	weights                          map[compute.Buffer]int
	order                            []string
	matmuls, swiglu                  int
	plainActivation, classActivation int
	fault                            string
	primary                          any
	cleanup, poison                  bool
	cleanupFirst                     error
}

func (b *expertHALCleanupRecorder) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}
func (b *expertHALCleanupRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}
func (b *expertHALCleanupRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.F32 {
		b.plainActivation++
		if b.fault == "upload" {
			panic(b.primary)
		}
		return b.track(b.Backend.Upload(x, dt), "input")
	}
	y := b.Backend.Upload(x, dt)
	b.weights[y.Buf()] = 0
	return y
}
func (b *expertHALCleanupRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if class == compute.MemoryActivation {
		b.classActivation++
		panic("unexpected narrow expert activation route: " + site)
	}
	return b.Upload(x, dt)
}
func (b *expertHALCleanupRecorder) track(x compute.Tensor, label string) compute.Tensor {
	b.owned = append(b.owned, x.Buf())
	b.labels[x.Buf()] = label
	return x
}
func (b *expertHALCleanupRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	label := []string{"gate", "up", "down"}[b.matmuls-1]
	if b.fault == label {
		panic(b.primary)
	}
	return b.track(b.Backend.MatMul(w, x), label)
}
func (b *expertHALCleanupRecorder) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	if b.fault == "swiglu" {
		panic(b.primary)
	}
	return b.track(b.Backend.SwiGLU(g, u), "fused")
}
func (b *expertHALCleanupRecorder) Read(x compute.Tensor) []float32 {
	if b.fault == "read" {
		panic(b.primary)
	}
	return b.Backend.Read(x)
}
func (b *expertHALCleanupRecorder) Free(x compute.Tensor) {
	label, owned := b.labels[x.Buf()]
	if !owned {
		if n, weight := b.weights[x.Buf()]; weight {
			b.weights[x.Buf()] = n + 1
		}
		b.Backend.Free(x)
		return
	}
	b.frees[x.Buf()]++
	b.order = append(b.order, label)
	if b.poison {
		for i := range b.Backend.Read(x) {
			b.Backend.Read(x)[i] = float32(math.NaN())
		}
	}
	b.Backend.Free(x)
	if b.cleanup {
		if len(b.order) == 1 {
			panic(b.cleanupFirst)
		}
		panic("later transient Free")
	}
}

// Existing release attempts retain the original selected panic and successful
// borrowed readback is copied before cleanup. The active expertSwiGLUHAL path
// uses plain activation Upload; the later narrow seam uses UploadClass instead.
// Ref #13815. This is software lifetime evidence, not device qualification.
// fak-test:runtime medium est=2s lane=default
func TestExpertSwiGLUHALPrimaryCleanup(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("requires the actual cpu-ref backend")
	}
	const H = 256
	m := expertRingTestModel(t, H, 1)
	x := expertRingTestInput(H)
	run := func(b *expertHALCleanupRecorder) ([]float32, *Session) {
		s := &Session{M: m, Backend: b, Q4K: true, halW: map[string]compute.Tensor{}}
		return expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s}), s
	}
	newRecorder := func() *expertHALCleanupRecorder {
		return &expertHALCleanupRecorder{Backend: compute.Default(), labels: map[compute.Buffer]string{}, frees: map[compute.Buffer]int{}, weights: map[compute.Buffer]int{}, cleanupFirst: errors.New("first transient cleanup")}
	}
	baseline := newRecorder()
	want, control := run(baseline)
	want = append([]float32(nil), want...)
	control.Close()
	if len(want) != H || baseline.matmuls != 3 || baseline.swiglu != 1 || baseline.plainActivation != 1 || baseline.classActivation != 0 {
		t.Fatal("fixture did not select the full device expert seam")
	}
	for _, value := range want {
		if !finite32(value) {
			t.Fatal("nonfinite arithmetic control")
		}
	}
	primary := &compute.BackendError{Backend: "expert-cleanup", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected primary")}
	for _, tc := range []struct {
		name, fault     string
		primary         any
		cleanup, poison bool
	}{
		{name: "success-owned-readback", poison: true},
		{name: "upload", fault: "upload", primary: primary, cleanup: true},
		{name: "gate", fault: "gate", primary: primary, cleanup: true},
		{name: "up", fault: "up", primary: primary, cleanup: true},
		{name: "swiglu", fault: "swiglu", primary: primary, cleanup: true},
		{name: "down", fault: "down", primary: primary, cleanup: true},
		{name: "read", fault: "read", primary: primary, cleanup: true},
		{name: "opaque-primary", fault: "read", primary: &struct{ id int }{13815}, cleanup: true},
		{name: "string-primary", fault: "read", primary: "selected read panic", cleanup: true},
		{name: "cleanup-only", cleanup: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newRecorder()
			b.fault, b.primary, b.cleanup, b.poison = tc.fault, tc.primary, tc.cleanup, tc.poison
			s := &Session{M: m, Backend: b, Q4K: true, halW: map[string]compute.Tensor{}}
			t.Cleanup(func() { b.cleanup, b.poison = false, false; s.Close() })
			var got []float32
			var caught any
			func() {
				defer func() { caught = recover() }()
				got = expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			}()
			wantPanic := tc.primary
			if wantPanic == nil && tc.cleanup {
				wantPanic = b.cleanupFirst
			}
			if caught != wantPanic {
				t.Errorf("primary changed: got=%v want=%v", caught, wantPanic)
			}
			if b.plainActivation != 1 || b.classActivation != 0 {
				t.Fatal("did not select the active expertSwiGLUHAL activation path")
			}
			var order []string
			for i := len(b.owned) - 1; i >= 0; i-- {
				buf := b.owned[i]
				order = append(order, b.labels[buf])
				if b.frees[buf] != 1 {
					t.Errorf("%s release attempts=%d, want 1", b.labels[buf], b.frees[buf])
				}
			}
			if !reflect.DeepEqual(b.order, order) {
				t.Errorf("cleanup order=%v want=%v", b.order, order)
			}
			for _, n := range b.weights {
				if n != 0 {
					t.Error("callback freed a borrowed weight")
				}
			}
			if tc.name == "success-owned-readback" {
				if len(got) != len(want) {
					t.Fatal("successful result has wrong width")
				}
				for i, v := range got {
					if !finite32(v) || math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Fatalf("owned arithmetic result changed at %d: %v want %v", i, v, want[i])
					}
				}
			}
		})
	}
}
