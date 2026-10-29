package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// retentionOOMBackend injects device allocation failures into a synthetic forward and records
// the pool trims and context probes the failed-request release performs. Its latch stands in
// for the CUDA session latch: an injected failure observes it the way allocateQwen35GDN does.
type retentionOOMBackend struct {
	compute.Backend
	latch     *compute.DeviceFaultLatch
	failures  int
	failWith  func() any
	observe   compute.DeviceFaultClass
	injected  int
	recycled  int
	trimLarge []int
	probes    int
	probeErr  error
}

func (b *retentionOOMBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.injected < b.failures {
		b.injected++
		if b.observe != "" {
			b.latch.Observe(b.observe, "retention-test", 0)
		}
		panic(b.failWith())
	}
	return b.Backend.MatMul(w, x)
}

func (b *retentionOOMBackend) Recycle()                                    { b.recycled++ }
func (b *retentionOOMBackend) TrimLarge(keep int)                          { b.trimLarge = append(b.trimLarge, keep) }
func (b *retentionOOMBackend) DeviceFaultLatch() *compute.DeviceFaultLatch { return b.latch }
func (b *retentionOOMBackend) DeviceContextProbe() error {
	b.probes++
	return b.probeErr
}

// forwardClosedSequenceOOM is the Raider shape: the layer-3 attention-qg scratch allocation
// misses inside the resident sequence prefill, and the model closes the session around it.
func forwardClosedSequenceOOM() any {
	return &model.BackendForwardOperationError{
		Backend: "cuda", Forward: model.ForwardQwen35GDN, Path: model.Qwen35GDNCUDAPath,
		Layer: -1, Stage: "sequence prefill",
		Cause: &compute.Qwen35SequenceError{Stage: "attention-qg", Layer: 3, Cause: &compute.DeviceAllocError{
			Bytes: 251658240, Site: "qwen35-sequence-attention-qg", Class: compute.MemoryScratchpad,
		}},
	}
}

func retentionPlanner(be compute.Backend) (*InKernelPlanner, []int) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	return &InKernelPlanner{m: m, modelID: "synthetic-oom-retention", backend: be}, synthIDs(cfg.VocabSize, 8, 9073)
}

func retentionGenerate(p *InKernelPlanner, ids []int) error {
	_, err := p.generateReusedWithOOMRetry(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { return false }, nil)
	return err
}

// fak-test:runtime fast est=1s
func TestFailedRequestOOMRetention_ForwardClosedOOMReleasesAndNextRequestSucceeds(t *testing.T) {
	be := &retentionOOMBackend{
		Backend:  compute.Default(),
		latch:    compute.NewDeviceFaultLatch("cuda", 3),
		failures: 1,
		failWith: forwardClosedSequenceOOM,
		observe:  compute.DeviceFaultAllocation,
	}
	p, ids := retentionPlanner(be)

	err := retentionGenerate(p, ids)
	var alloc *compute.DeviceAllocError
	if !errors.As(err, &alloc) || alloc.Bytes != 251658240 {
		t.Fatalf("failed request err = %v, want the typed attention-qg allocation miss", err)
	}
	// Session close also recycles; TrimLarge(0) is the release's own witness.
	if be.recycled < 1 || len(be.trimLarge) != 1 || be.trimLarge[0] != 0 {
		t.Fatalf("post-failure release = recycle %d trimLarge %v, want >=1/[0]", be.recycled, be.trimLarge)
	}
	if be.probes != 1 || be.latch.Health() != compute.DeviceHealthy {
		t.Fatalf("allocation-only suspect = probes %d health %s, want 1 probe and healthy", be.probes, be.latch.Health())
	}
	if err := compute.AdmitDevice(be, "next-request"); err != nil {
		t.Fatalf("next request refused after a classified OOM: %v", err)
	}
	if err := retentionGenerate(p, ids); err != nil {
		t.Fatalf("next request failed after release: %v", err)
	}
	if st := be.latch.Snapshot(); st.Attempts != 0 || st.Recoveries != 1 {
		t.Fatalf("latch attempts/recoveries = %d/%d, want 0/1 (no reconstruction budget spent)", st.Attempts, st.Recoveries)
	}
}

// fak-test:runtime fast est=1s
func TestFailedRequestOOMRetention_FailedRetryTrimsAfterFinalAttempt(t *testing.T) {
	be := &retentionOOMBackend{
		Backend:  compute.Default(),
		failures: 2,
		failWith: func() any {
			return &compute.DeviceAllocError{Bytes: 4096, Site: "retention-scratch", Class: compute.MemoryScratchpad}
		},
	}
	p, ids := retentionPlanner(be)

	err := retentionGenerate(p, ids)
	var oom *InKernelOOMError
	if !errors.As(err, &oom) {
		t.Fatalf("failed retry err = %v, want InKernelOOMError", err)
	}
	// One trim prepares the retry; the second releases what the failed retry left pooled.
	if len(be.trimLarge) != 2 {
		t.Fatalf("trimLarge calls = %v, want retry-prepare plus post-failure release", be.trimLarge)
	}
	if err := retentionGenerate(p, ids); err != nil {
		t.Fatalf("next request failed after release: %v", err)
	}
}

// fak-test:runtime fast est=10ms
func TestFailedRequestOOMRetention_ExecutionFaultStaysLatched(t *testing.T) {
	be := &retentionOOMBackend{Backend: compute.Default(), latch: compute.NewDeviceFaultLatch("cuda", 3)}
	be.latch.Observe(compute.DeviceFaultAllocation, "scratch", 0)
	be.latch.Observe(compute.DeviceFaultExecution, "qwen35-gdn-decode", 10002)
	p := &InKernelPlanner{modelID: "retention", backend: be}

	p.releaseFailedOOMRequest(context.Background(), &InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryScratchpad})
	if len(be.trimLarge) != 1 {
		t.Fatalf("classified OOM must still trim idle pools, got %v", be.trimLarge)
	}
	if be.probes != 0 || be.latch.Health() != compute.DevicePoisoned {
		t.Fatalf("execution fault reopened: probes %d health %s", be.probes, be.latch.Health())
	}
	var fault *compute.DeviceFaultError
	if err := compute.AdmitDevice(be, "next-request"); !errors.As(err, &fault) {
		t.Fatalf("execution-faulted session admitted next request: %v", err)
	}
}

// fak-test:runtime fast est=10ms
func TestFailedRequestOOMRetention_FailedProbeAndNonOOMLeaveStateAlone(t *testing.T) {
	be := &retentionOOMBackend{Backend: compute.Default(), latch: compute.NewDeviceFaultLatch("cuda", 3), probeErr: errors.New("fence failed")}
	be.latch.Observe(compute.DeviceFaultAllocation, "scratch", 0)
	p := &InKernelPlanner{modelID: "retention", backend: be}

	p.releaseFailedOOMRequest(context.Background(), errors.New("ordinary upstream error"))
	if len(be.trimLarge) != 0 || be.probes != 0 {
		t.Fatalf("non-OOM failure touched device state: trimLarge %v probes %d", be.trimLarge, be.probes)
	}
	p.releaseFailedOOMRequest(context.Background(), &InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryScratchpad})
	if be.probes != 1 || be.latch.Health() != compute.DevicePoisoned {
		t.Fatalf("failed probe reopened the session: probes %d health %s", be.probes, be.latch.Health())
	}
	ctx := context.WithValue(context.Background(), inKernelEngramAttemptContextKey{}, &inKernelEngramAttempt{selected: true})
	p.releaseFailedOOMRequest(ctx, &InKernelOOMError{Bytes: 1 << 20, Class: compute.MemoryScratchpad})
	if len(be.trimLarge) != 1 {
		t.Fatalf("selected engram attempt must not trim shared state: %v", be.trimLarge)
	}
}
