package agent

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type deviceAdmissionCompleteResult struct {
	comp *Completion
	err  error
}

func deviceAdmissionCompleteFixture(t *testing.T) (*InKernelPlanner, []Message, int64) {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.LayerTypes = []string{"linear_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-device-admission", true, nil, false)
	p.maxNew = 4
	p.batchDecode = true
	p.metal = true
	// Price the request under the same backend target that Complete will admit.
	// A nil backend selects the host-Metal panel path and therefore a different plan.
	p.backend = capacityProbeBackend{Backend: compute.Default(), total: 1 << 30, free: 1 << 30, known: true}
	messages := []Message{{Role: RoleUser, Content: "bounded reservation"}}
	prepared, err := p.preparePrompt(context.Background(), messages, nil, applySampleOpts())
	if err != nil {
		t.Fatal(err)
	}
	want := p.requestMemoryPlan(len(prepared.ids), prepared.maxNew).DeviceTotal()
	if want <= 0 {
		t.Fatalf("request device demand = %d, want positive", want)
	}
	free := want + want/2
	p.backend = &cohortCapabilityBackend{
		capacityProbeBackend: capacityProbeBackend{Backend: compute.Default(), total: 2 * free, free: free, known: true},
		path:                 compute.Qwen35IndependentDecodeBatchPath,
	}
	if want > ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom) || 2*want <= ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom) {
		t.Fatalf("fixture want=%d free=%d does not admit one and refuse two", want, free)
	}
	return p, messages, want
}

func awaitDeviceReservation(t *testing.T, p *InKernelPlanner, want int64) {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		p.deviceBudgetMu.Lock()
		got := p.deviceReservedBytes
		p.deviceBudgetMu.Unlock()
		if got == want {
			return
		}
		select {
		case <-timer.C:
			t.Fatalf("reserved bytes = %d, want %d", got, want)
		case <-tick.C:
		}
	}
}

func deviceReservedBytes(p *InKernelPlanner) int64 {
	p.deviceBudgetMu.Lock()
	defer p.deviceBudgetMu.Unlock()
	return p.deviceReservedBytes
}

func awaitDeviceAdmissionComplete(t *testing.T, done <-chan deviceAdmissionCompleteResult) deviceAdmissionCompleteResult {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(time.Second):
		t.Fatal("Complete did not return")
		return deviceAdmissionCompleteResult{}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInKernelCompleteCoalescedDeviceReservationLifecycle(t *testing.T) {
	p, messages, want := deviceAdmissionCompleteFixture(t)
	hookEntered := make(chan struct{})
	releaseHook := make(chan struct{})
	var enteredOnce, releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseHook) }) }
	defer release()
	p.coalesceReadyHook = func() {
		enteredOnce.Do(func() { close(hookEntered) })
		<-releaseHook
	}

	firstCtx, cancelFirst := context.WithCancel(context.Background())
	defer cancelFirst()
	firstDone := make(chan deviceAdmissionCompleteResult, 1)
	go func() {
		comp, err := p.Complete(firstCtx, messages, nil)
		firstDone <- deviceAdmissionCompleteResult{comp: comp, err: err}
	}()
	select {
	case <-hookEntered:
	case <-time.After(time.Second):
		t.Fatal("first Complete did not reach the coalescer hook")
	}
	awaitDeviceReservation(t, p, want)

	secondDone := make(chan deviceAdmissionCompleteResult, 1)
	go func() {
		comp, err := p.Complete(context.Background(), messages, nil)
		secondDone <- deviceAdmissionCompleteResult{comp: comp, err: err}
	}()
	second := awaitDeviceAdmissionComplete(t, secondDone)
	var capacityErr *InKernelCapacityError
	if second.comp != nil || !errors.As(second.err, &capacityErr) ||
		capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
		t.Fatalf("second Complete = (%#v, %T %v), want device reservation refusal", second.comp, second.err, second.err)
	}
	p.coalesceMu.Lock()
	queued := len(p.coalesceReady)
	p.coalesceMu.Unlock()
	if queued != 1 {
		t.Fatalf("coalescer queue depth = %d, want only the admitted request", queued)
	}

	cancelFirst()
	release()
	first := awaitDeviceAdmissionComplete(t, firstDone)
	if first.comp != nil || !errors.Is(first.err, context.Canceled) {
		t.Fatalf("canceled first Complete = (%#v, %v), want context.Canceled", first.comp, first.err)
	}
	if got := deviceReservedBytes(p); got != 0 {
		t.Fatalf("reservation after cancellation = %d, want 0", got)
	}

	retryEntered := make(chan struct{})
	releaseRetry := make(chan struct{})
	var retryEnteredOnce, retryReleaseOnce sync.Once
	releaseRetryHook := func() { retryReleaseOnce.Do(func() { close(releaseRetry) }) }
	defer releaseRetryHook()
	p.coalesceReadyHook = func() {
		retryEnteredOnce.Do(func() { close(retryEntered) })
		<-releaseRetry
	}
	retryCtx, cancelRetry := context.WithCancel(context.Background())
	defer cancelRetry()
	retryDone := make(chan deviceAdmissionCompleteResult, 1)
	go func() {
		comp, err := p.Complete(retryCtx, messages, nil)
		retryDone <- deviceAdmissionCompleteResult{comp: comp, err: err}
	}()
	select {
	case <-retryEntered:
	case <-time.After(time.Second):
		t.Fatal("Complete after release was not admitted")
	}
	awaitDeviceReservation(t, p, want)
	cancelRetry()
	releaseRetryHook()
	retry := awaitDeviceAdmissionComplete(t, retryDone)
	if retry.comp != nil || !errors.Is(retry.err, context.Canceled) {
		t.Fatalf("canceled retry Complete = (%#v, %v), want context.Canceled", retry.comp, retry.err)
	}
	if got := deviceReservedBytes(p); got != 0 {
		t.Fatalf("retry cancellation leaked %d reserved bytes", got)
	}
}
