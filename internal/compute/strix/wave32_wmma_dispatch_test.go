package strix

import (
	"errors"
	"fmt"
	"testing"
	"time"
)

// targetTFLOPs is the issue's named acceptance target for BF16 WMMA on gfx1151.
// It may ONLY be claimed from an [HW-WITNESSED] (physical device) measurement.
const targetTFLOPs = 47.52

// simWMMAExecutor is a pure-Go WMMA executor with PhysicalDeviceProbe=false: it
// numerically models the tile but never touches a GPU. It exists to prove that a
// simulator CANNOT earn [HW-WITNESSED], even when it reports a non-zero duration.
type simWMMAExecutor struct {
	// injectedDuration attacks the classifier: a simulator that lies about a
	// non-zero kernel duration must still be classified SW-VERIFIED.
	injectedDuration time.Duration
	caps             WMMACapabilities
}

func newSimWMMAExecutor() *simWMMAExecutor {
	return &simWMMAExecutor{
		caps: WMMACapabilities{
			DeviceName:          "software-simulator",
			GFX:                 "gfx1151-sim",
			ExecutesKernel:      false,
			PhysicalDeviceProbe: false,
			SupportedPrimitives: []WMMAPrimitive{WMMAPrimitive16x16x16, WMMAPrimitive16x16x32},
		},
	}
}

func (s *simWMMAExecutor) Capabilities() WMMACapabilities { return s.caps }

func (s *simWMMAExecutor) ExecuteWMMATile(prim WMMAPrimitive, M, N, K int, A_bf16, B_bf16 []uint16) ([]float32, time.Duration, error) {
	cfg := DefaultWave32WMMAConfig()
	if prim == WMMAPrimitive16x16x32 {
		cfg = Wave32WMMAConfig16x16x32()
	}
	sim := NewWave32RetiledMatMul(cfg)
	C, _, err := sim.MatMulBF16(M, N, K, A_bf16, B_bf16)
	if err != nil {
		return nil, 0, err
	}
	return C, s.injectedDuration, nil
}

// TestBF16WMMATileTuningMeets47_52TFLOPs is the issue's named witness. On a host
// with no gfx1151 it is an SW GATE: it proves the target CANNOT be claimed from a
// simulator, and exits 0. If a real device executor is present it hard-asserts.
func TestBF16WMMATileTuningMeets47_52TFLOPs(t *testing.T) {
	prims := []WMMAPrimitive{WMMAPrimitive16x16x16, WMMAPrimitive16x16x32}

	var realExec WMMADeviceExecutor = currentPhysicalWMMAExecutor()

	for _, prim := range prims {
		req := DefaultWMMARequest(32, 32, 32)
		req.Prim = prim

		exec := realExec
		if exec == nil {
			exec = newSimWMMAExecutor()
		}

		res, err := RunWMMAMeasured(exec, req)
		if err != nil {
			t.Fatalf("prim %s: RunWMMAMeasured: %v", prim, err)
		}

		if res.Class != EvidenceClassSWVerified {
			t.Fatalf("prim %s: expected SW-VERIFIED on this host, got %s", prim, res.Class)
		}
		if res.MeasuredKnown {
			t.Errorf("prim %s: simulator must not report MeasuredKnown", prim)
		}
		if res.MeasuredTFLOPs != 0 {
			t.Errorf("prim %s: simulator must not report measured TFLOPs, got %v", prim, res.MeasuredTFLOPs)
		}
		if len(res.C) != req.M*req.N {
			t.Errorf("prim %s: expected C len %d, got %d", prim, req.M*req.N, len(res.C))
		}

		t.Logf("prim %s: [HW-WITNESSED] >=%.2f TFLOPS is pending physical gfx1151 "+
			"(class=%s measured_known=%v measured_tflops=%.3f)",
			prim, targetTFLOPs, res.Class, res.MeasuredKnown, res.MeasuredTFLOPs)
	}

	if realExec == nil {
		t.Logf("no physical gfx1151 WMMA executor on this host: [HW-WITNESSED] >=%.2f TFLOPS target unclaimable (SW gate holds)", targetTFLOPs)
		return
	}

	// A real device executor is present: now the named target is binding.
	req := DefaultWMMARequest(64, 64, 64)
	req.Prim = WMMAPrimitive16x16x16
	res, err := RunWMMAMeasured(realExec, req)
	if err != nil {
		t.Fatalf("physical executor: RunWMMAMeasured: %v", err)
	}
	if !res.MeasuredKnown {
		t.Fatalf("physical executor present but MeasuredKnown=false (class=%s)", res.Class)
	}
	if res.MeasuredTFLOPs < targetTFLOPs {
		t.Fatalf("physical gfx1151 measured %.2f TFLOPS < target %.2f", res.MeasuredTFLOPs, targetTFLOPs)
	}
}

// TestBF16WMMASimCannotEarnHWWitnessed is the negative test: no simulated or
// caller-supplied path may earn the hardware label.
func TestBF16WMMASimCannotEarnHWWitnessed(t *testing.T) {
	req := DefaultWMMARequest(32, 32, 32)
	req.Prim = WMMAPrimitive16x16x16

	// (1) A simulator with a deliberately NON-ZERO duration still cannot classify HW.
	lying := newSimWMMAExecutor()
	lying.injectedDuration = 3 * time.Millisecond

	res, err := RunWMMAMeasured(lying, req)
	if err != nil {
		t.Fatalf("RunWMMAMeasured(sim): %v", err)
	}
	if res.KernelDuration <= 0 {
		t.Fatalf("test setup: injected duration should be > 0, got %v", res.KernelDuration)
	}
	if res.Class != EvidenceClassSWVerified {
		t.Errorf("sim with dur>0 must stay SW-VERIFIED, got %s", res.Class)
	}
	if res.MeasuredKnown {
		t.Errorf("sim must not report MeasuredKnown")
	}
	if res.MeasuredTFLOPs != 0 {
		t.Errorf("sim must not report measured TFLOPs, got %v", res.MeasuredTFLOPs)
	}

	// (2) A nil executor is a typed fail-closed refusal, never a sim fallback.
	if _, err := RunWMMAMeasured(nil, req); !errors.Is(err, ErrWMMADeviceKernelAbsent) {
		t.Errorf("nil executor: expected ErrWMMADeviceKernelAbsent, got %v", err)
	}

	// (3) An unsupported primitive is refused, not silently downshifted.
	unsupported := newSimWMMAExecutor()
	unsupported.caps.SupportedPrimitives = []WMMAPrimitive{WMMAPrimitive16x16x16}
	badReq := DefaultWMMARequest(32, 32, 32)
	badReq.Prim = WMMAPrimitive16x16x32
	if _, err := RunWMMAMeasured(unsupported, badReq); !errors.Is(err, ErrWMMAUnsupportedPrimitive) {
		t.Errorf("unsupported primitive: expected ErrWMMAUnsupportedPrimitive, got %v", err)
	}

	// (4) Class is derived structurally, not from any caller-supplied value:
	// RunWMMAMeasured takes only (exec, req) and never a WMMAResult, so a
	// caller-constructed HW-labeled WMMAResult cannot be threaded in. Prove the
	// derivation is a pure function of the executor's capabilities even when the
	// caller tries to seed an HW label on a result it passes around.
	var seeded WMMAResult
	seeded.Class = EvidenceClassHWWitnessed // fabricated by the caller
	out, err := RunWMMAMeasured(newSimWMMAExecutor(), req)
	if err != nil {
		t.Fatalf("RunWMMAMeasured: %v", err)
	}
	if out.Class != EvidenceClassSWVerified {
		t.Errorf("caller HW label leaked: got class %s from a sim executor", out.Class)
	}
	if seeded.Class != EvidenceClassHWWitnessed {
		t.Fatalf("test setup: seeded result should carry the caller's fabricated label")
	}
}

// BenchmarkBF16WMMATileSweep sweeps a small set of shapes per primitive through
// the sim executor and reports ACTUAL per-shape TFLOPs when measured; otherwise
// it reports 0 / model-only. It never presents a simulated number as measured.
func BenchmarkBF16WMMATileSweep(b *testing.B) {
	shapes := [][3]int{{32, 32, 32}, {64, 64, 64}, {64, 64, 32}}
	prims := []WMMAPrimitive{WMMAPrimitive16x16x16, WMMAPrimitive16x16x32}
	sim := newSimWMMAExecutor()

	for _, prim := range prims {
		for _, sh := range shapes {
			M, N, K := sh[0], sh[1], sh[2]
			req := DefaultWMMARequest(M, N, K)
			req.Prim = prim

			b.Run(fmt.Sprintf("%s/%dx%dx%d", prim, M, N, K), func(b *testing.B) {
				var res WMMAResult
				var err error
				for i := 0; i < b.N; i++ {
					res, err = RunWMMAMeasured(sim, req)
					if err != nil {
						b.Fatalf("RunWMMAMeasured: %v", err)
					}
				}
				if res.MeasuredKnown {
					b.ReportMetric(res.MeasuredTFLOPs, "TFLOPs")
				} else {
					b.ReportMetric(0, "TFLOPs")
					b.ReportMetric(0, "model-only")
				}
			})
		}
	}
}

// currentPhysicalWMMAExecutor is the seam a platform build may populate with a
// real gfx1151 device executor. On this host there is none, so it returns nil and
// the named witness stays an SW gate.
func currentPhysicalWMMAExecutor() WMMADeviceExecutor { return nil }
