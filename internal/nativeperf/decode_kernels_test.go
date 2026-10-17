package nativeperf

import (
	"context"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fakeKernelProfile is a cumulative profile a test advances between span
// start and finish.
type fakeKernelProfile struct {
	mu sync.Mutex
	p  compute.KernelProfile
}

func (f *fakeKernelProfile) read() (compute.KernelProfile, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := f.p
	p.Rows = append([]compute.KernelRow(nil), f.p.Rows...)
	return p, true
}

// step advances the profile by n decode steps of two kernels.
func (f *fakeKernelProfile) step(n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.p.Rows == nil {
		f.p.Rows = []compute.KernelRow{{Kernel: "q2k_matvec", GX: 5120, GY: 1}, {Kernel: "rmsnorm", GX: 1, GY: 1}}
	}
	f.p.Batches += uint64(n)
	f.p.Ops += uint64(3 * n)
	f.p.UntimedTransfer += uint64(n)
	f.p.GPUNS += float64(n) * 7000
	f.p.WallNS += float64(n) * 9000
	f.p.Rows[0].Calls += uint64(2 * n)
	f.p.Rows[0].NS += float64(n) * 6000
	f.p.Rows[1].Calls += uint64(n)
	f.p.Rows[1].NS += float64(n) * 1000
}

func registerFakeKernelProfile(t *testing.T) *fakeKernelProfile {
	t.Helper()
	if compute.KernelProfileReaderRegistered() {
		t.Skip("a backend kernel profile reader is registered in this process")
	}
	f := &fakeKernelProfile{}
	f.step(5)
	t.Cleanup(compute.RegisterKernelProfileReader("vulkan", f.read))
	return f
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeSpanKernelsNilWhenOff(t *testing.T) {
	if compute.KernelProfileReaderRegistered() {
		t.Skip("a backend kernel profile reader is registered in this process")
	}
	ctx, acc := WithDecodeHostWork(context.Background())
	starts := nonDecodeWorkStarts.Load()
	nd := BeginNonDecodeWork()
	if nonDecodeWorkStarts.Load() != starts || nonDecodeWorkInFlight.Load() != 0 {
		t.Fatal("non-decode marks must stay inert while no kernel profile is kept")
	}
	nd.End()
	span := StartDecodeSpan()
	if w := span.Finish(ctx, 4); w.Kernels != nil || w.NonDecodeOverlap {
		t.Fatalf("window = %+v, want no kernels with the profile off", w)
	}
	totals, ok := acc.Totals()
	if !ok || totals.Kernels != nil {
		t.Fatalf("totals = %+v ok=%v, want absent kernels", totals, ok)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeHostWorkKernelsPerStep(t *testing.T) {
	f := registerFakeKernelProfile(t)
	ctx, acc := WithDecodeHostWork(context.Background())
	for _, steps := range []int{127, 63} {
		span := StartDecodeSpan()
		f.step(steps)
		w := span.Finish(ctx, steps)
		if w.Kernels == nil || w.Kernels.Ops != uint64(3*steps) {
			t.Fatalf("window kernels = %+v, want %d ops", w.Kernels, 3*steps)
		}
	}
	totals, ok := acc.Totals()
	if !ok || totals.Steps != 190 {
		t.Fatalf("totals = %+v ok=%v, want 190 steps", totals, ok)
	}
	want := &compute.KernelProfile{
		Backend: "vulkan", Batches: 190, Ops: 570, UntimedTransfer: 190, GPUNS: 190 * 7000, WallNS: 190 * 9000,
		Rows: []compute.KernelRow{
			{Kernel: "q2k_matvec", GX: 5120, GY: 1, Calls: 380, NS: 190 * 6000},
			{Kernel: "rmsnorm", GX: 1, GY: 1, Calls: 190, NS: 190 * 1000},
		},
	}
	if !reflect.DeepEqual(totals.Kernels, want) {
		t.Fatalf("request kernels = %#v\nwant %#v", totals.Kernels, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeSpanKernelsDroppedOnOverlap(t *testing.T) {
	f := registerFakeKernelProfile(t)
	ctx, acc := WithDecodeHostWork(context.Background())
	outer := StartDecodeSpan()
	inner := StartDecodeSpan()
	f.step(8)
	if w := inner.Finish(context.Background(), 8); !w.Overlapped || w.Kernels != nil {
		t.Fatalf("inner window = %+v, want overlapped without kernels", w)
	}
	if w := outer.Finish(ctx, 8); w.Kernels != nil {
		t.Fatalf("outer window kernels = %+v, want nil", w.Kernels)
	}
	span := StartDecodeSpan()
	f.step(8)
	if w := span.Finish(ctx, 8); w.Kernels == nil {
		t.Fatal("a clean window after the overlap must measure kernels")
	}
	totals, ok := acc.Totals()
	if !ok || totals.Windows != 2 || totals.Kernels != nil {
		t.Fatalf("totals = %+v ok=%v, want a kernel gap from the overlapped window", totals, ok)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeSpanKernelsDroppedOnNonDecodeWork(t *testing.T) {
	f := registerFakeKernelProfile(t)
	t.Run("starts_inside", func(t *testing.T) {
		span := StartDecodeSpan()
		nd := BeginNonDecodeWork()
		f.step(4)
		nd.End()
		if w := span.Finish(context.Background(), 4); !w.NonDecodeOverlap || w.Kernels != nil {
			t.Fatalf("window = %+v, want non-decode overlap without kernels", w)
		}
	})
	t.Run("active_at_start", func(t *testing.T) {
		nd := BeginNonDecodeWork()
		span := StartDecodeSpan()
		nd.End()
		f.step(4)
		if w := span.Finish(context.Background(), 4); !w.NonDecodeOverlap || w.Kernels != nil {
			t.Fatalf("window = %+v, want non-decode overlap without kernels", w)
		}
	})
	nd := BeginNonDecodeWork()
	nd.End()
	nd.End()
	span := StartDecodeSpan()
	f.step(4)
	if w := span.Finish(context.Background(), 4); w.NonDecodeOverlap || w.Kernels == nil {
		t.Fatalf("window after a closed mark = %+v, want kernels", w)
	}
	if nonDecodeWorkInFlight.Load() != 0 {
		t.Fatalf("non-decode in flight = %d after every End, want 0", nonDecodeWorkInFlight.Load())
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeSpanKernelsDroppedWhenDecodeStartsDuringClosingSnapshot(t *testing.T) {
	f := registerFakeKernelProfile(t)
	span := StartDecodeSpan()
	f.step(6)
	var late DecodeSpan
	reads := 0
	unregister := compute.RegisterKernelProfileReader("vulkan", func() (compute.KernelProfile, bool) {
		reads++
		if reads == 1 {
			late = StartDecodeSpan()
			f.step(3)
		}
		return f.read()
	})
	defer unregister()
	w := span.Finish(context.Background(), 6)
	late.Abandon()
	if w.Overlapped || w.NonDecodeOverlap || w.Kernels != nil {
		t.Fatalf("window = %+v, want a kernel gap with no host overlap verdict", w)
	}
	if reads != 2 {
		t.Fatalf("kernel reads = %d, want the closing read plus the late span's start", reads)
	}
}
