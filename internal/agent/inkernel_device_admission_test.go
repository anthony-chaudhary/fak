package agent

import (
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func deviceAdmissionFixture(t *testing.T) (*InKernelPlanner, int64) {
	t.Helper()
	p := &InKernelPlanner{m: model.NewSynthetic(tinyConcurrencyConfig())}
	want := p.requestMemoryPlan(10, 5).DeviceTotal()
	if want <= 0 || want > math.MaxInt64/3 {
		t.Fatalf("device request demand = %d, want bounded positive bytes", want)
	}
	free := 2*want + want/5
	p.backend = capacityProbeBackend{total: free, free: free, known: true}
	return p, want
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionReservesSnapshot(t *testing.T) {
	p, want := deviceAdmissionFixture(t)
	free := p.backend.(capacityProbeBackend).free
	aggregate := 2 * want
	if aggregate <= ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom) {
		t.Fatalf("fixture aggregate %d fits original headroom budget for free %d", aggregate, free)
	}
	weakenedBudget := want + ApplyByteHeadroom(free-want, inKernelRequestDeviceHeadroom)
	if aggregate > weakenedBudget {
		t.Fatalf("fixture aggregate %d also exceeds weakened post-subtraction budget %d", aggregate, weakenedBudget)
	}

	if err := p.refuseOversizeRequest(10, 5); err != nil {
		t.Fatalf("first unreserved precheck = %v, want admitted", err)
	}
	if err := p.refuseOversizeRequest(10, 5); err != nil {
		t.Fatalf("second unreserved precheck over the same snapshot = %v, want admitted baseline", err)
	}

	release, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("first reserved admission = %v, want admitted", err)
	}
	if p.deviceReservedBytes != want {
		t.Fatalf("reserved bytes = %d, want %d", p.deviceReservedBytes, want)
	}
	if _, err = p.admitCoalescedDeviceMemory(10, 5); err == nil {
		t.Fatal("second admission over the held snapshot reservation succeeded")
	} else {
		var capacityErr *InKernelCapacityError
		if !errors.As(err, &capacityErr) || capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
			t.Fatalf("second admission error = %T (%v), want closed device reservation refusal", err, err)
		}
	}
	if p.deviceReservedBytes != want {
		t.Fatalf("refused admission mutated reservation to %d, want %d", p.deviceReservedBytes, want)
	}

	release()
	release()
	if p.deviceReservedBytes != 0 {
		t.Fatalf("idempotent release left %d reserved bytes", p.deviceReservedBytes)
	}
	release2, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("admission after release = %v, want admitted", err)
	}
	release2()
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionConcurrentOneWinner(t *testing.T) {
	p, want := deviceAdmissionFixture(t)
	const callers = 8
	start := make(chan struct{})
	results := make(chan struct {
		release func()
		err     error
	}, callers)
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			release, err := p.admitCoalescedDeviceMemory(10, 5)
			results <- struct {
				release func()
				err     error
			}{release: release, err: err}
		}()
	}
	close(start)
	wg.Wait()
	close(results)

	winners := 0
	for result := range results {
		if result.err == nil {
			winners++
			defer result.release()
			continue
		}
		var capacityErr *InKernelCapacityError
		if !errors.As(result.err, &capacityErr) {
			t.Fatalf("loser error = %T (%v), want *InKernelCapacityError", result.err, result.err)
		}
	}
	if winners != 1 || p.deviceReservedBytes != want {
		t.Fatalf("concurrent admissions winners=%d reserved=%d, want 1/%d", winners, p.deviceReservedBytes, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionUnknownCapacityFailsOpen(t *testing.T) {
	for _, tc := range []struct {
		name    string
		backend capacityProbeBackend
	}{
		{name: "unreported", backend: capacityProbeBackend{known: false}},
		{name: "zero-total", backend: capacityProbeBackend{total: 0, free: 1 << 30, known: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &InKernelPlanner{
				m:       model.NewSynthetic(tinyConcurrencyConfig()),
				backend: tc.backend,
			}
			for i := 0; i < 2; i++ {
				release, err := p.admitCoalescedDeviceMemory(100_000, 256)
				if err != nil {
					t.Fatalf("unknown-capacity admission %d = %v, want fail-open", i, err)
				}
				release()
			}
			if p.deviceReservedBytes != 0 {
				t.Fatalf("unknown capacity reserved %d bytes, want zero", p.deviceReservedBytes)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionOverflowDoesNotMutateReservation(t *testing.T) {
	p, _ := deviceAdmissionFixture(t)
	p.deviceReservedBytes = math.MaxInt64

	_, err := p.admitCoalescedDeviceMemory(10, 5)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
		t.Fatalf("overflow admission error = %T (%v), want closed reservation refusal", err, err)
	}
	if p.deviceReservedBytes != math.MaxInt64 {
		t.Fatalf("overflow admission mutated reservation to %d, want MaxInt64", p.deviceReservedBytes)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionPreservesPressureTrim(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	p := &InKernelPlanner{m: m, modelID: "coalesced-pressure"}
	want := p.requestMemoryPlan(10, 5).DeviceTotal()
	be := &pressureTrimBackend{Backend: compute.Default(), total: 4 * want, free: 1, trimFree: 2 * want}
	p.backend = be

	release, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("trimmed admission = %v, want admitted", err)
	}
	defer release()
	if be.recycle != 1 || be.trim != 1 || len(be.trimLarge) != 1 {
		t.Fatalf("pressure trim calls = recycle %d trim %d large %v, want one each", be.recycle, be.trim, be.trimLarge)
	}
	if p.deviceReservedBytes != want {
		t.Fatalf("trimmed admission reserved %d bytes, want %d", p.deviceReservedBytes, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionTrimsCombinedReservation(t *testing.T) {
	m := model.NewSynthetic(tinyConcurrencyConfig())
	p := &InKernelPlanner{m: m, modelID: "coalesced-combined-pressure"}
	want := p.requestMemoryPlan(10, 5).DeviceTotal()
	be := &pressureTrimBackend{Backend: compute.Default(), total: 2 << 30, free: 1 << 30}
	p.backend = be

	releaseFirst, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("first admission = %v, want admitted", err)
	}
	defer releaseFirst()
	be.free = want + want/5
	be.trimFree = 1 << 30

	releaseSecond, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("combined reservation admission after trim = %v, want admitted", err)
	}
	defer releaseSecond()
	if be.recycle != 1 || be.trim != 1 || len(be.trimLarge) != 1 {
		t.Fatalf("combined pressure trim calls = recycle %d trim %d large %v, want one each", be.recycle, be.trim, be.trimLarge)
	}
	if p.deviceReservedBytes != 2*want {
		t.Fatalf("combined admission reserved %d bytes, want %d", p.deviceReservedBytes, 2*want)
	}
}
