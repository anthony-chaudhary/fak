package gateway

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

type gatewayFakeKernelProfile struct {
	mu sync.Mutex
	p  compute.KernelProfile
}

func (f *gatewayFakeKernelProfile) read() (compute.KernelProfile, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := f.p
	p.Rows = append([]compute.KernelRow(nil), f.p.Rows...)
	return p, true
}

func (f *gatewayFakeKernelProfile) decode(steps int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.p.Rows == nil {
		f.p.Rows = []compute.KernelRow{{Kernel: "q2k_matvec", GX: 5120, GY: 1}, {Kernel: "rmsnorm", GX: 1, GY: 1}}
	}
	f.p.Batches += uint64(steps)
	f.p.Ops += uint64(3 * steps)
	f.p.GPUNS += float64(steps) * 2e5
	f.p.WallNS += float64(steps) * 3e5
	f.p.Rows[0].Calls += uint64(2 * steps)
	f.p.Rows[0].NS += float64(steps) * 1.8e5
	f.p.Rows[1].Calls += uint64(steps)
	f.p.Rows[1].NS += float64(steps) * 2e4
}

func decodeWindow(ctx context.Context, f *gatewayFakeKernelProfile, steps int) {
	span := nativeperf.StartDecodeSpan()
	time.Sleep(5 * time.Millisecond)
	if f != nil {
		f.decode(steps)
	}
	span.Finish(ctx, steps)
}

// fak-test:runtime fast est=50ms lane=default
func TestApplyRequestObservationFoldsKernels(t *testing.T) {
	if compute.KernelProfileReaderRegistered() {
		t.Skip("a backend kernel profile reader is registered in this process")
	}
	f := &gatewayFakeKernelProfile{}
	unregister := compute.RegisterKernelProfileReader("vulkan", f.read)
	defer unregister()

	ctx, _ := nativeperf.WithDecodeHostWork(context.Background())
	decodeWindow(ctx, f, 127)
	rec := &perfledger.Record{NativeTiming: perfledger.NewNativeTiming(64, 128, 10, 1000)}
	applyRequestObservation(&requestObservation{phaseContext: ctx}, rec)
	k := rec.NativeTiming.Kernels
	if k == nil || k.Backend != "vulkan" || k.DecodeSteps != 127 || !k.Complete || k.GPUMSPerToken != 0.2 {
		t.Fatalf("kernels = %+v, want a complete 127-step vulkan table at 0.2 ms/token", k)
	}
	if len(k.SelectedKernels) != 2 || k.SelectedKernels[0].Kernel != "q2k_matvec" || k.SelectedKernels[0].Op != compute.KernelOpMatvec ||
		k.SelectedKernels[0].Share != 0.9 || k.SelectedKernels[1].Op != compute.KernelOpNorm {
		t.Fatalf("selected kernels = %+v", k.SelectedKernels)
	}

	gapCtx, _ := nativeperf.WithDecodeHostWork(context.Background())
	decodeWindow(gapCtx, f, 64)
	unregister()
	decodeWindow(gapCtx, nil, 64)
	gap := &perfledger.Record{NativeTiming: perfledger.NewNativeTiming(64, 129, 10, 1000)}
	applyRequestObservation(&requestObservation{phaseContext: gapCtx}, gap)
	if gap.NativeTiming == nil || gap.NativeTiming.Kernels != nil {
		t.Fatalf("a request with an unprofiled window must carry no kernels: %+v", gap.NativeTiming)
	}
}
