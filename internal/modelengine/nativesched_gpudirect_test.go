package modelengine

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func newTestGPUDirectSwapper(t *testing.T, cfg model.Config) *model.Qwen38GPUDirectSwapper {
	t.Helper()
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 64, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}
	engine, err := model.NewQwen38GPUDirectSwapper(slab, cfg, 4)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}
	return engine
}

func testNativeSchedulerGPUDirect(t *testing.T, mode NativePreemptionMode) {
	t.Helper()
	m := nativeSchedulerQwenSwapModel()
	engine := newTestGPUDirectSwapper(t, m.Cfg)
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:             mode,
		MaxBlocks:        8,
		BlockTokens:      4,
		GPUDirectSwapper: engine,
	})

	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	history := append(append([]int(nil), ln.prompt...), ln.gen...)
	wantLogits := copyF32(ln.logits)

	control := m.NewSession()
	defer control.Close()
	controlLogits := control.Prefill(history)
	if !reflect.DeepEqual(wantLogits, controlLogits) {
		t.Fatal("fixture logits differ before preemption")
	}

	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("gpudirect preempt failed: %v", err)
	}

	if ln.sess != nil || len(s.preempted) != 1 {
		t.Fatalf("preempted lane session=%p victims=%d, want nil/1", ln.sess, len(s.preempted))
	}
	if ln.gpuDirectKV == nil {
		t.Fatal("preempted lane expected non-nil gpuDirectKV")
	}
	if ln.hostKV != nil {
		t.Fatalf("expected nil hostKV under GPU Direct, got %d bytes", len(ln.hostKV))
	}
	if stagingCopies := ln.gpuDirectKV.StagingCopyCount(); stagingCopies != 0 {
		t.Fatalf("expected 0 staging copies, got %d", stagingCopies)
	}

	descBytes := int64(ln.gpuDirectKV.TotalBytes())
	if descBytes <= 0 {
		t.Fatalf("expected positive descriptor total bytes, got %d", descBytes)
	}

	stPreempt := s.KVPreemptionStats()
	if stPreempt.GPUDirectSwaps != 1 {
		t.Fatalf("expected 1 GPUDirectSwaps, got %d", stPreempt.GPUDirectSwaps)
	}
	if stPreempt.GPUDirectBytes != descBytes {
		t.Fatalf("expected GPUDirectBytes=%d, got %d", descBytes, stPreempt.GPUDirectBytes)
	}
	if stPreempt.GPUDirectStagingCopies != 0 {
		t.Fatalf("expected GPUDirectStagingCopies=0, got %d", stPreempt.GPUDirectStagingCopies)
	}
	if stPreempt.SwapBytes != 0 {
		t.Fatalf("expected 0 SwapBytes under GPU Direct, got %d", stPreempt.SwapBytes)
	}

	s.readmitPreemptedLocked()
	if ln.sess == nil || len(s.lanes) != 1 || len(s.preempted) != 0 {
		t.Fatalf("readmit session=%p running=%d victims=%d, want live/1/0", ln.sess, len(s.lanes), len(s.preempted))
	}
	defer ln.sess.Close()

	if ln.gpuDirectKV != nil {
		t.Fatalf("readmitted lane retained gpuDirectKV: %v", ln.gpuDirectKV)
	}
	if !reflect.DeepEqual(ln.logits, wantLogits) {
		t.Fatal("readmit did not preserve saved logits exactly")
	}
	if _, err := ln.sess.VerifyTokenLineage(history); err != nil {
		t.Fatalf("readmit lineage: %v", err)
	}

	for step := 0; step < 3; step++ {
		gotToken, wantToken := argmax(ln.logits), argmax(controlLogits)
		if gotToken != wantToken {
			t.Fatalf("continuation step %d token=%d, want %d", step, gotToken, wantToken)
		}
		history = append(history, gotToken)
		ln.logits = ln.sess.Step(gotToken)
		controlLogits = control.Step(wantToken)
		if !reflect.DeepEqual(ln.logits, controlLogits) {
			t.Fatalf("continuation step %d logits differ", step)
		}
		if _, err := ln.sess.VerifyTokenLineage(history); err != nil {
			t.Fatalf("continuation step %d lineage: %v", step, err)
		}
	}

	stReadmit := s.KVPreemptionStats()
	if stReadmit.Readmitted != 1 {
		t.Fatalf("expected 1 Readmitted, got %d", stReadmit.Readmitted)
	}
	if stReadmit.GPUDirectRestored != descBytes {
		t.Fatalf("expected GPUDirectRestored=%d, got %d", descBytes, stReadmit.GPUDirectRestored)
	}
}

func TestNativeScheduler_GPUDirect(t *testing.T) {
	t.Run("NativePreemptGPUDirectSwap", func(t *testing.T) {
		testNativeSchedulerGPUDirect(t, NativePreemptGPUDirectSwap)
	})
	t.Run("NativePreemptSwap_WithEngine", func(t *testing.T) {
		testNativeSchedulerGPUDirect(t, NativePreemptSwap)
	})
	t.Run("NilEngineFailsCleanly", func(t *testing.T) {
		m := nativeSchedulerQwenSwapModel()
		s := NewNativeScheduler(m)
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptGPUDirectSwap,
			MaxBlocks:   8,
			BlockTokens: 4,
		})
		ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
		defer ln.cancel()
		if err := s.preemptLaneLocked(ln); err == nil {
			t.Fatal("expected error preempting with NativePreemptGPUDirectSwap and nil GPUDirectSwapper, got nil")
		}
	})
	t.Run("LineageMismatchRefusesReadmit", func(t *testing.T) {
		m := nativeSchedulerQwenSwapModel()
		engine := newTestGPUDirectSwapper(t, m.Cfg)
		s := NewNativeScheduler(m)
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:             NativePreemptGPUDirectSwap,
			MaxBlocks:        8,
			BlockTokens:      4,
			GPUDirectSwapper: engine,
		})
		ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
		defer ln.cancel()
		if err := s.preemptLaneLocked(ln); err != nil {
			t.Fatalf("preempt failed: %v", err)
		}
		ln.gen = ln.gen[:len(ln.gen)-1]
		s.readmitPreemptedLocked()
		if !ln.terminal || !ln.reclaimed || ln.sess != nil || len(s.lanes) != 0 || len(s.preempted) != 0 {
			t.Fatalf("refused readmit terminal=%t reclaimed=%t session=%p running=%d victims=%d", ln.terminal, ln.reclaimed, ln.sess, len(s.lanes), len(s.preempted))
		}
		if !errors.Is(ln.err, model.ErrTokenLineageMismatch) {
			t.Fatalf("refused readmit error=%v, want ErrTokenLineageMismatch", ln.err)
		}
		if ln.gpuDirectKV != nil {
			t.Fatal("expected gpuDirectKV to be freed on refused readmit")
		}
	})
}
