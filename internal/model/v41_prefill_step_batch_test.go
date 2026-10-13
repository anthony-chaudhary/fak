package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41PrefillStepBatchBackend struct {
	*v41DenseTestBackend
	beginOK          bool
	beginCalls, ends int
	endErr           error
	events           []string
	active           bool
	requireActive    bool
	opOutside        bool
}

func newV41PrefillStepBatchBackend() *v41PrefillStepBatchBackend {
	return &v41PrefillStepBatchBackend{v41DenseTestBackend: newV41DenseTestBackend(), beginOK: true}
}

func (b *v41PrefillStepBatchBackend) BeginV41StepBatch() bool {
	b.beginCalls++
	b.events = append(b.events, "begin")
	if b.beginOK {
		b.active = true
	}
	return b.beginOK
}

func (b *v41PrefillStepBatchBackend) EndV41StepBatch() error {
	b.ends++
	b.events = append(b.events, "end")
	b.active = false
	return b.endErr
}

func (b *v41PrefillStepBatchBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.requireActive && !b.active {
		b.opOutside = true
	}
	return b.v41DenseTestBackend.MatMul(w, x)
}

func (b *v41PrefillStepBatchBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.requireActive && !b.active {
		b.opOutside = true
	}
	return b.v41DenseTestBackend.BatchedMatMul(w, x, rows)
}

func (b *v41PrefillStepBatchBackend) reset() {
	b.beginCalls, b.ends = 0, 0
	b.events = nil
	b.active, b.opOutside = false, false
}

// fak-test:runtime fast est=1s lane=default
func TestV41PrefillStepBatchWindows(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Run("cold full forward", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		b.requireActive = true
		s.Prefill([]int{1, 2, 3})
		if b.beginCalls != 1 || b.ends != 1 || b.active || b.opOutside {
			t.Fatalf("begin=%d end=%d", b.beginCalls, b.ends)
		}
	})

	t.Run("declined begin has no end", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		b.beginOK = false
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		if b.beginCalls != 1 || b.ends != 0 {
			t.Fatalf("begin=%d end=%d", b.beginCalls, b.ends)
		}
	})

	t.Run("one window per suffix step", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		b.reset()
		s.Prefill([]int{4, 5, 6})
		if b.beginCalls != 3 || b.ends != 3 {
			t.Fatalf("begin=%d end=%d", b.beginCalls, b.ends)
		}
	})

	t.Run("fallback starts after failed step closes", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		s.v41Forward.layerState(m.Cfg.NumLayers - 1).nextWindowPos += 5
		b.reset()
		s.Prefill([]int{4, 5})
		want := []string{"begin", "end", "begin", "end"}
		if !reflect.DeepEqual(b.events, want) {
			t.Fatalf("events=%v want=%v", b.events, want)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestV41PrefillStepBatchFailureLifecycle(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Run("in-forward panic retains primary failure", func(t *testing.T) {
		m := v41DenseTestFixture(t, false, false)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		b.reset()
		fault := &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: ErrV41ForwardStage}
		endErr := errors.New("secondary end sentinel")
		b.endErr = endErr
		b.fail, b.failSite = fault, "matmul"
		err := recoverError(func() { s.Prefill([]int{4}) })
		if !errors.Is(err, fault) || errors.Is(err, endErr) || b.beginCalls != 1 || b.ends != 1 || b.active {
			t.Fatalf("err=%T begin=%d end=%d", err, b.beginCalls, b.ends)
		}
	})

	t.Run("decode end stage error does not trigger fallback", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		b.reset()
		endErr := errors.New("decode end sentinel")
		b.endErr = errors.Join(endErr, ErrV41ForwardStage)
		err := recoverError(func() { s.Step(4) })
		if !errors.Is(err, endErr) || !errors.Is(err, ErrV41ForwardStage) || b.beginCalls != 1 || b.ends != 1 || b.active {
			t.Fatalf("err=%T begin=%d end=%d", err, b.beginCalls, b.ends)
		}
	})

	t.Run("structural and end failure do not trigger fallback", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		s.v41Forward.layerState(m.Cfg.NumLayers - 1).nextWindowPos += 5
		b.reset()
		endErr := errors.New("dual end sentinel")
		b.endErr = endErr
		err := recoverError(func() { s.Prefill([]int{4}) })
		if !errors.Is(err, ErrV41ForwardStage) || !errors.Is(err, endErr) || b.beginCalls != 1 || b.ends != 1 || b.active {
			t.Fatalf("err=%T begin=%d end=%d", err, b.beginCalls, b.ends)
		}
	})

	t.Run("end error is fatal", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		endErr := errors.New("end sentinel")
		b.endErr = endErr
		s := v41DenseTestSession(t, m, b)
		err := recoverError(func() { s.Prefill([]int{1, 2}) })
		if !errors.Is(err, endErr) || b.beginCalls != 1 || b.ends != 1 {
			t.Fatalf("err=%T begin=%d end=%d", err, b.beginCalls, b.ends)
		}
	})

	t.Run("end stage error does not trigger fallback", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2, 3})
		b.reset()
		endErr := errors.New("end stage sentinel")
		b.endErr = errors.Join(endErr, ErrV41ForwardStage)
		err := recoverError(func() { s.Prefill([]int{4}) })
		if !errors.Is(err, endErr) || !errors.Is(err, ErrV41ForwardStage) || b.beginCalls != 1 || b.ends != 1 {
			t.Fatalf("err=%T begin=%d end=%d", err, b.beginCalls, b.ends)
		}
	})

	t.Run("disabled", func(t *testing.T) {
		t.Setenv(v41StepBatchEnv, "0")
		m := v41IncrementalPlainModel(t, 1)
		b := newV41PrefillStepBatchBackend()
		s := v41DenseTestSession(t, m, b)
		s.Prefill([]int{1, 2})
		if b.beginCalls != 0 || b.ends != 0 {
			t.Fatalf("begin=%d end=%d", b.beginCalls, b.ends)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestV41PrefillStepBatchParity(t *testing.T) {
	m := v41IncrementalPlainModel(t, 1)
	prompt, suffix := []int{1, 2, 3}, []int{4, 5}
	run := func(backend compute.Backend, enabled bool) (cold, tail []float32, history []int) {
		t.Helper()
		if enabled {
			t.Setenv(v41StepBatchEnv, "1")
		} else {
			t.Setenv(v41StepBatchEnv, "0")
		}
		s := v41DenseTestSession(t, m, backend)
		cold = append([]float32(nil), s.Prefill(prompt)...)
		tail = append([]float32(nil), s.Prefill(suffix)...)
		history = append([]int(nil), s.v41Forward.history...)
		return cold, tail, history
	}

	enabledBackend := newV41PrefillStepBatchBackend()
	enabledBackend.requireActive = true
	enabledCold, enabledTail, enabledHistory := run(enabledBackend, true)
	disabledCold, disabledTail, disabledHistory := run(newV41PrefillStepBatchBackend(), false)
	unsupportedCold, unsupportedTail, unsupportedHistory := run(newV41DenseTestBackend(), true)
	if !reflect.DeepEqual(enabledCold, disabledCold) || !reflect.DeepEqual(enabledTail, disabledTail) || !reflect.DeepEqual(enabledHistory, disabledHistory) {
		t.Fatal("enabled and disabled results differ")
	}
	if !reflect.DeepEqual(enabledCold, unsupportedCold) || !reflect.DeepEqual(enabledTail, unsupportedTail) || !reflect.DeepEqual(enabledHistory, unsupportedHistory) {
		t.Fatal("batch-capable and unsupported backend results differ")
	}
	if enabledBackend.active || enabledBackend.opOutside {
		t.Fatalf("active=%t opOutside=%t", enabledBackend.active, enabledBackend.opOutside)
	}
}
