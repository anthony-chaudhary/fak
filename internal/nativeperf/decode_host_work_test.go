package nativeperf

import (
	"context"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func fakeProcessCPU(t *testing.T, cpu *atomic.Int64, ok bool) {
	t.Helper()
	saved := processCPUTime
	processCPUTime = func() (time.Duration, bool) { return time.Duration(cpu.Load()), ok }
	t.Cleanup(func() { processCPUTime = saved })
}

func TestCPUPerTokenFoldsDecodeWindowsIntoRequest(t *testing.T) {
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, true)
	ctx, acc := WithDecodeHostWork(context.Background())

	for range 2 {
		span := StartDecodeSpan()
		time.Sleep(2 * time.Millisecond)
		cpu.Add(int64(30 * time.Millisecond))
		w := span.Finish(ctx, 10)
		if w.CPU == nil || *w.CPU != 30*time.Millisecond {
			t.Fatalf("window cpu = %v, want 30ms", w.CPU)
		}
	}
	totals, ok := acc.Totals()
	if !ok || totals.Windows != 2 || totals.Steps != 20 {
		t.Fatalf("totals = %+v ok=%v, want 2 windows / 20 steps", totals, ok)
	}
	perToken, ok := totals.CPUMSPerToken()
	if !ok || math.Abs(perToken-3) > 1e-9 {
		t.Fatalf("cpu ms/token = %v ok=%v, want 3", perToken, ok)
	}
	share, ok := totals.HostShare()
	if !ok || share <= 0 {
		t.Fatalf("host share = %v ok=%v, want positive", share, ok)
	}
}

func TestCPUPerTokenAbsentWhenUnreadableOrOverlapped(t *testing.T) {
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, false)
	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	span.Finish(ctx, 4)
	totals, _ := acc.Totals()
	if _, ok := totals.CPUMSPerToken(); ok {
		t.Fatal("unreadable process CPU must leave cpu_ms_per_token absent, not 0")
	}

	fakeProcessCPU(t, &cpu, true)
	ctx, acc = WithDecodeHostWork(context.Background())
	outer := StartDecodeSpan()
	inner := StartDecodeSpan()
	if w := inner.Finish(context.Background(), 1); !w.Overlapped || w.CPU != nil {
		t.Fatalf("inner overlapped window = %+v, want overlapped with absent CPU", w)
	}
	outer.Finish(ctx, 4)
	totals, _ = acc.Totals()
	if _, ok := totals.HostShare(); ok {
		t.Fatal("an overlapped decode window cannot be attributed to one request")
	}
}

func TestDeviceDispatchKernelPhaseReachesReceiptMetrics(t *testing.T) {
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, true)
	var dispatches, submitWait atomic.Uint64
	t.Cleanup(compute.RegisterDeviceWorkReader("vulkan", func() compute.DeviceWork {
		return compute.DeviceWork{Dispatches: dispatches.Load(), SubmitWaitNS: submitWait.Load(), SubmitWaitObserved: true}
	}))
	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	time.Sleep(2 * time.Millisecond)
	dispatches.Add(754)
	submitWait.Add(uint64(time.Millisecond))
	w := span.Finish(ctx, 1)
	if w.Backend != "vulkan" || w.Dispatches != 754 || w.Kernel == nil || *w.Kernel != time.Millisecond {
		t.Fatalf("window = %+v, want vulkan 754 dispatches and 1ms kernel", w)
	}
	totals, _ := acc.Totals()
	receipt, ok := totals.DecodeWindowReceipt()
	if !ok {
		t.Fatal("measured kernel time must yield a phase receipt")
	}
	m := NewReceiptMetrics(0)
	if !m.ObservePhases(receipt, time.Now()) {
		t.Fatalf("phase receipt rejected: %v", receipt.Validate(time.Nanosecond))
	}
	if out := m.Prometheus(time.Now()); !strings.Contains(out, `phase="kernel",kind="wall"} 0.001`+"\n") {
		t.Fatalf("kernel phase missing from receipt metrics:\n%s", out)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDecodeSubmitWaitCoverageNeverFabricatesZero(t *testing.T) {
	for _, tc := range []struct {
		name          string
		before, after compute.DeviceWork
		observed      bool
	}{
		{"unobserved_zero", compute.DeviceWork{}, compute.DeviceWork{Dispatches: 1}, false},
		{"unobserved_nonzero", compute.DeviceWork{}, compute.DeviceWork{Dispatches: 1, SubmitWaitNS: 1}, false},
		{"observed_zero", compute.DeviceWork{SubmitWaitObserved: true}, compute.DeviceWork{Dispatches: 1, SubmitWaitObserved: true}, true},
		{"coverage_disappears", compute.DeviceWork{SubmitWaitObserved: true}, compute.DeviceWork{Dispatches: 1}, false},
		{"coverage_appears", compute.DeviceWork{}, compute.DeviceWork{Dispatches: 1, SubmitWaitObserved: true}, false},
		{"wait_counter_reset", compute.DeviceWork{SubmitWaitNS: 2, SubmitWaitObserved: true}, compute.DeviceWork{Dispatches: 1, SubmitWaitNS: 1, SubmitWaitObserved: true}, false},
		{"dispatch_counter_reset", compute.DeviceWork{Dispatches: 2, SubmitWaitObserved: true}, compute.DeviceWork{Dispatches: 1, SubmitWaitObserved: true}, false},
		{"wait_duration_overflow", compute.DeviceWork{SubmitWaitObserved: true}, compute.DeviceWork{Dispatches: 1, SubmitWaitNS: ^uint64(0), SubmitWaitObserved: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current := tc.before
			unregister := compute.RegisterDeviceWorkReader("coverage-fixture", func() compute.DeviceWork { return current })
			defer unregister()
			ctx, acc := WithDecodeHostWork(context.Background())
			span := StartDecodeSpan()
			defer span.Abandon()
			// A receipt needs positive wall time even within one host clock tick.
			span.started = span.started.Add(-time.Nanosecond)
			current = tc.after
			w := span.Finish(ctx, 1)
			if (w.Kernel != nil) != tc.observed || (w.Kernel != nil && *w.Kernel != 0) {
				t.Fatalf("submit wait=%v, observed=%v", w.Kernel, tc.observed)
			}
			totals, _ := acc.Totals()
			if _, ok := totals.DecodeWindowReceipt(); ok != tc.observed {
				t.Fatalf("phase receipt available=%v, want %v", ok, tc.observed)
			}
		})
	}
	t.Run("backend_switch", func(t *testing.T) {
		unregister := compute.RegisterDeviceWorkReader("coverage-before", func() compute.DeviceWork { return compute.DeviceWork{SubmitWaitObserved: true} })
		defer unregister()
		span := StartDecodeSpan()
		defer span.Abandon()
		unregister()
		unregisterAfter := compute.RegisterDeviceWorkReader("coverage-after", func() compute.DeviceWork { return compute.DeviceWork{Dispatches: 1, SubmitWaitObserved: true} })
		defer unregisterAfter()
		if w := span.Finish(context.Background(), 1); w.Kernel != nil {
			t.Fatal("different backend identities produced a submit-wait delta")
		}
	})
	t.Run("selected_backend_does_not_inherit_previous_values", func(t *testing.T) {
		a := compute.DeviceWork{SubmitWaitObserved: true}
		z := compute.DeviceWork{ReadbackBytes: 2}
		unregisterA := compute.RegisterDeviceWorkReader("coverage-a", func() compute.DeviceWork { return a })
		defer unregisterA()
		unregisterZ := compute.RegisterDeviceWorkReader("coverage-z", func() compute.DeviceWork { return z })
		defer unregisterZ()
		span := StartDecodeSpan()
		defer span.Abandon()
		a.Dispatches, a.ReadbackBytes = 1, 4
		z.Dispatches, z.ReadbackBytes = 1, 1
		if w := span.Finish(context.Background(), 1); w.Backend != "coverage-z" || w.Kernel != nil || w.Readback != 0 {
			t.Fatalf("selected backend inherited foreign values: %+v", w)
		}
	})
}
