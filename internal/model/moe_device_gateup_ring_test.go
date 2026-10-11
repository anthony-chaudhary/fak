package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// This recorder executes the real CPU kernels. Freed-weight rejection makes the
// lifetime witness independent of whether CPU Free happens to poison its buffer.
type gateUpRingRecorder struct {
	compute.Backend
	weights       map[compute.Buffer]int
	weightUploads int
	matmuls       int
	freedDispatch int
	failUpload    int
	failMatMul    int
	fault         error
	onUpload      func()
	onMatMul      func()
}

func (b *gateUpRingRecorder) Caps() compute.Caps {
	return compute.Caps{DeviceMemory: true, UploadDtype: true}
}

func (b *gateUpRingRecorder) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}

func (b *gateUpRingRecorder) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt != compute.F32 {
		b.weightUploads++
		if b.onUpload != nil {
			b.onUpload()
		}
		if b.weightUploads == b.failUpload {
			panic(b.fault)
		}
	}
	y := b.Backend.Upload(x, dt)
	if dt != compute.F32 {
		b.weights[y.Buf()] = 0
	}
	return y
}

func (b *gateUpRingRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, _ string) compute.Tensor {
	return b.Upload(x, dt)
}

func (b *gateUpRingRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	if b.weights[w.Buf()] != 0 {
		b.freedDispatch++
		panic("gate/up dispatched a freed weight")
	}
	if b.onMatMul != nil {
		b.onMatMul()
	}
	if b.matmuls == b.failMatMul {
		panic(b.fault)
	}
	return b.Backend.MatMul(w, x)
}

func (b *gateUpRingRecorder) Free(x compute.Tensor) {
	if n, weight := b.weights[x.Buf()]; weight {
		b.weights[x.Buf()] = n + 1
	}
	b.Backend.Free(x)
}

func gateUpRingSession(m *Model, budget int64) (*Session, *gateUpRingRecorder) {
	b := &gateUpRingRecorder{Backend: compute.Default(), weights: map[compute.Buffer]int{}, fault: &compute.BackendError{Backend: "gate-up-ring-recorder", Class: compute.VulkanClassExecutionFailed, Err: errors.New("injected projection fault")}}
	return &Session{M: m, Backend: b, Q4K: true, halW: map[string]compute.Tensor{}, ExpertRingBytes: budget}, b
}

func gateUpRingFreedOnce(t *testing.T, b *gateUpRingRecorder) {
	t.Helper()
	for _, n := range b.weights {
		if n != 1 {
			t.Errorf("weight Free attempts=%d, want exactly one after owner teardown", n)
		}
	}
}

// fak-test:runtime medium est=7s lane=default
// fak-test:justify Pins the live gate/up staging lifetime with one-weight rings and shared holds.
func TestExpertGateUpRingLifetime(t *testing.T) {
	if compute.Default().Name() != "cpu-ref" || compute.Default().Caps().DeviceMemory {
		t.Fatal("requires the actual cpu-ref backend; recorder is software evidence only")
	}
	const H = 256
	m := expertRingTestModel(t, H, 1)
	gate, up, down := expertName(0, 0, "gate_proj.weight"), expertName(0, 0, "up_proj.weight"), expertName(0, 0, "down_proj.weight")
	bytes, x := expertRingWeightBytes(t, m), expertRingTestInput(H)
	for _, route := range []string{"generic", "full-bf16", "gate-up-down"} {
		run := func(s *Session) ([]float32, bool) {
			if route == "gate-up-down" {
				return q4kExpertGateUpDownHAL(s, gate, up, down, x, H, H)
			}
			return q4kExpertInputHALWithLimitAndBF16(s, gate, up, x, H, H, 0, route == "full-bf16", 0)
		}
		baseline, baselineBackend := gateUpRingSession(m, 0)
		want, ok := run(baseline)
		if !ok {
			t.Fatal("unbounded control declined")
		}
		want = append([]float32(nil), want...)
		baseline.Close()
		gateUpRingFreedOnce(t, baselineBackend)
		for i, v := range want {
			if !finite32(v) {
				t.Fatalf("nonfinite unbounded control at %d: %v", i, v)
			}
		}
		for _, tc := range []struct {
			name   string
			budget int64
			strict bool
		}{
			{"disabled", 0, false},
			{"one-weight-portable", bytes, false},
			{"one-weight-strict", bytes, true},
			{"sufficient-strict", 3 * bytes, true},
		} {
			t.Run(route+"/"+tc.name, func(t *testing.T) {
				s, b := gateUpRingSession(m, tc.budget)
				t.Cleanup(s.Close)
				if tc.strict {
					s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
				}
				var got []float32
				var handled bool
				caught := deviceOnlyExpertRingPanic(func() { got, handled = run(s) })
				if tc.strict && tc.budget == bytes {
					err, _ := caught.(error)
					var operation *BackendForwardOperationError
					if !errors.As(err, &operation) || !errors.Is(err, polymodel.ErrPinnedNoRoom) || !s.BackendSessionClosed() || b.matmuls != 0 {
						t.Errorf("one-weight refusal=%v closed=%v matmuls=%d; want typed capacity refusal before dispatch", caught, s.BackendSessionClosed(), b.matmuls)
					}
				} else {
					if caught != nil || !handled || len(got) != len(want) {
						t.Fatalf("valid route failed: panic=%v handled=%v length=%d", caught, handled, len(got))
					}
					for i := range want {
						if !finite32(got[i]) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("residency changed arithmetic at %d", i)
						}
					}
					if s.expertRing != nil && len(s.expertRing.holds) != 0 {
						t.Error("successful operation retained a projection hold")
					}
				}
				if b.freedDispatch != 0 || (tc.budget == 0 && s.expertRing != nil) {
					t.Error("freed-weight dispatch or disabled-ring allocation")
				}
				s.Close()
				gateUpRingFreedOnce(t, b)
			})
		}
	}

	for _, fault := range []string{"up-stage", "up-gemm", "up-decline", "down-decline", "strict-refusal", "success"} {
		t.Run("shared/"+fault, func(t *testing.T) {
			model := expertRingTestModel(t, H, 1)
			s, b := gateUpRingSession(model, 0)
			peer := &Session{M: model, Backend: b, Q4K: true, halW: map[string]compute.Tensor{}}
			budget := 3 * bytes
			if fault == "strict-refusal" {
				budget = bytes
				s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			}
			sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: model, Backend: b, BudgetBytes: budget})
			if err != nil {
				t.Fatal(err)
			}
			if err := sh.Attach(peer, "peer"); err != nil {
				t.Fatal(err)
			}
			if err := sh.Attach(s, "subject"); err != nil {
				t.Fatal(err)
			}
			r := peer.expertRing
			key := "q4k:" + gate
			peer.weightHALQ4K(gate, model.q4kw[gate])
			done := peer.ringEnter(r)
			r.hold(key)
			done()
			t.Cleanup(func() {
				b.onUpload, b.onMatMul = nil, nil
				done := peer.ringEnter(r)
				r.release(key)
				done()
				s.Close()
				peer.Close()
				if err := sh.Close(); err != nil {
					t.Error(err)
				}
				gateUpRingFreedOnce(t, b)
			})
			observations := 0
			checkHeld := func() {
				observations++
				if r.holds[polymodel.ModelID(key)] != 2 {
					t.Error("subject did not own exactly one gate hold before later staging/dispatch")
				}
			}
			b.onUpload = func() {
				checkHeld()
				if s.ringDepth < 2 {
					t.Error("up staging lacks an outer shared stage-and-hold span")
				}
			}
			b.onMatMul = checkHeld
			switch fault {
			case "up-stage":
				b.failUpload = b.weightUploads + 1
			case "up-gemm":
				b.failMatMul = 2
			case "up-decline":
				delete(model.q4kw, up)
				delete(model.manifest, up)
			case "down-decline":
				delete(model.q4kw, down)
				delete(model.manifest, down)
			}
			var handled bool
			caught := deviceOnlyExpertRingPanic(func() {
				if fault == "down-decline" {
					_, handled = q4kExpertGateUpDownHAL(s, gate, up, down, x, H, H)
				} else {
					_, handled = q4kExpertInputHAL(s, gate, up, x, H, H)
				}
			})
			if fault == "strict-refusal" {
				err, _ := caught.(error)
				var operation *BackendForwardOperationError
				if !errors.As(err, &operation) || !errors.Is(err, polymodel.ErrPinnedNoRoom) || !s.BackendSessionClosed() || b.matmuls != 0 {
					t.Errorf("shared strict refusal failed to retire before dispatch: %v", caught)
				}
				if s.sharedRing != nil || s.expertRing != nil {
					t.Error("failed shared member was not detached")
				}
				beforeUploads, beforeMatMuls := b.weightUploads, b.matmuls
				retry := deviceOnlyExpertRingPanic(func() { _, _ = q4kExpertInputHAL(s, gate, up, x, H, H) })
				if retry != s.halFailure || b.weightUploads != beforeUploads || b.matmuls != beforeMatMuls {
					t.Error("closed shared member replayed work or lost failure identity")
				}
			} else if fault == "up-stage" || fault == "up-gemm" {
				if caught != b.fault {
					t.Errorf("projection panic identity changed: %v", caught)
				}
			} else if caught != nil || handled != (fault == "success") {
				t.Errorf("unexpected result: panic=%v handled=%v", caught, handled)
			}
			if fault != "up-decline" && fault != "strict-refusal" && observations == 0 {
				t.Error("no held-lifetime observation")
			}
			if r.holds[polymodel.ModelID(key)] != 1 || len(r.holds) != 1 || s.ringDepth != 0 {
				t.Error("subject release lost the peer hold, retained its own hold, or stranded the shared span")
			}
			if b.freedDispatch != 0 {
				t.Error("freed shared weight dispatched")
			}
		})
	}

	for _, full := range []bool{false, true} {
		name := "reduced"
		if full {
			name = "full"
		}
		t.Run("session-callback/"+name, func(t *testing.T) {
			model := v41IncrementalExpertFixture(t, full, false)
			gate := layerName(0, "ffn.experts.0.w1.weight")
			s, b := gateUpRingSession(model, kQuantResidentBytes(model.kqw[gate]))
			t.Cleanup(s.Close)
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			callback := s.v41ExpertGateUpFunc()
			caught := deviceOnlyExpertRingPanic(func() { _, _, _ = callback(0, "ffn.experts.0", x) })
			err, _ := caught.(error)
			var operation *BackendForwardOperationError
			if !errors.As(err, &operation) || !errors.Is(err, polymodel.ErrPinnedNoRoom) || !s.BackendSessionClosed() || b.matmuls != 0 || b.freedDispatch != 0 {
				t.Errorf("selected callback lost pre-dispatch capacity refusal: panic=%v closed=%v dispatches=%d", caught, s.BackendSessionClosed(), b.matmuls)
			}
			beforeUploads, beforeMatMuls := b.weightUploads, b.matmuls
			retry := deviceOnlyExpertRingPanic(func() { _, _, _ = callback(0, "ffn.experts.0", x) })
			if retry != s.halFailure || b.weightUploads != beforeUploads || b.matmuls != beforeMatMuls {
				t.Error("closed selected callback replayed work or lost failure identity")
			}
			s.Close()
			gateUpRingFreedOnce(t, b)
		})
	}

	t.Run("gate-stage-failure", func(t *testing.T) {
		s, b := gateUpRingSession(m, 3*bytes)
		t.Cleanup(s.Close)
		b.failUpload = 1
		caught := deviceOnlyExpertRingPanic(func() { _, _ = q4kExpertInputHAL(s, gate, up, x, H, H) })
		if caught != b.fault || b.matmuls != 0 || s.ringDepth != 0 || len(s.expertRing.holds) != 0 {
			t.Fatal("gate staging failure changed identity, dispatched, or retained a hold")
		}
	})

	t.Run("disabled-tier", func(t *testing.T) {
		s, b := gateUpRingSession(m, bytes)
		t.Cleanup(s.Close)
		s.Q4K = false
		if got, ok := q4kExpertInputHAL(s, gate, up, x, H, H); got != nil || ok || b.weightUploads != 0 || s.expertRing != nil {
			t.Fatal("disabled resident tier no longer declines before staging")
		}
	})
}
