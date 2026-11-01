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

func knownTotalDeviceAdmissionFixture(t *testing.T, requests int) (*InKernelPlanner, int64) {
	t.Helper()
	p := &InKernelPlanner{
		m:       model.NewSynthetic(tinyConcurrencyConfig()),
		backend: capacityProbeBackend{total: 1 << 30, free: compute.FreeUnknown, known: true},
	}
	plan := p.requestMemoryPlan(10, 5)
	weights := plan.ByClass()[compute.MemoryWeights]
	want := plan.DeviceTotal() - weights
	if weights <= 0 || want <= 0 {
		t.Fatalf("fixture weights=%d request=%d, want positive demands", weights, want)
	}
	budget := weights + int64(requests)*want + want/2
	total := int64(math.Ceil(float64(budget) / (1 - inKernelRequestDeviceHeadroom)))
	p.backend = capacityProbeBackend{total: total, free: compute.FreeUnknown, known: true}
	if available := ApplyByteHeadroom(total, inKernelRequestDeviceHeadroom); available < weights+int64(requests)*want || available >= weights+int64(requests+1)*want {
		t.Fatalf("fixture budget=%d does not admit exactly %d requests", available, requests)
	}
	return p, want
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionReservesKnownTotal(t *testing.T) {
	p, want := knownTotalDeviceAdmissionFixture(t, 1)
	release, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("first known-total admission = %v", err)
	}
	defer release()
	if _, err := p.admitCoalescedDeviceMemory(10, 5); err == nil {
		t.Fatal("second admission exceeded the known total without a reservation refusal")
	} else {
		var capacityErr *InKernelCapacityError
		if !errors.As(err, &capacityErr) || errors.Is(err, ErrInKernelRuntimeExtrasUnknown) || capacityErr.Scope != compute.MemoryScopeDevice || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
			t.Fatalf("second admission = %T (%v), want typed device reservation refusal", err, err)
		}
	}
	if p.deviceReservedBytes != want {
		t.Fatalf("reserved=%d, want incremental demand %d with shared weights charged once", p.deviceReservedBytes, want)
	}
	release()
	release()
	if p.deviceReservedBytes != 0 {
		t.Fatalf("idempotent release left %d bytes reserved", p.deviceReservedBytes)
	}
	releaseAgain, err := p.admitCoalescedDeviceMemory(10, 5)
	if err != nil {
		t.Fatalf("readmission after release = %v", err)
	}
	releaseAgain()
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionSharesResidentWeights(t *testing.T) {
	p, want := knownTotalDeviceAdmissionFixture(t, 2)
	for i := 0; i < 2; i++ {
		release, err := p.admitCoalescedDeviceMemory(10, 5)
		if err != nil {
			t.Fatalf("admission %d double-counted shared weights: %v", i, err)
		}
		defer release()
	}
	if p.deviceReservedBytes != 2*want {
		t.Fatalf("reserved=%d, want %d request-local bytes", p.deviceReservedBytes, 2*want)
	}
	_, err := p.admitCoalescedDeviceMemory(10, 5)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Site != inKernelCoalescedDeviceReservationSite || p.deviceReservedBytes != 2*want {
		t.Fatalf("third admission=%v reserved=%d, want refusal preserving two reservations", err, p.deviceReservedBytes)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionCapacityTransitions(t *testing.T) {
	for _, firstFree := range []int64{compute.FreeUnknown, 1 << 30} {
		p := &InKernelPlanner{m: model.NewSynthetic(tinyConcurrencyConfig())}
		be := &capacityProbeBackend{total: 1 << 30, free: firstFree, known: true}
		p.backend = be
		be.free = 1 << 30
		want := p.requestMemoryPlan(10, 5).DeviceTotal()
		for i, free := range []int64{firstFree, compute.FreeUnknown, 1 << 30} {
			be.free = free
			release, err := p.admitCoalescedDeviceMemory(10, 5)
			if err != nil {
				t.Fatalf("transition %d free=%d: %v", i, free, err)
			}
			defer release()
			if p.deviceReservedBytes != int64(i+1)*want {
				t.Fatalf("transition %d reserved=%d, want %d", i, p.deviceReservedBytes, int64(i+1)*want)
			}
		}
	}
}

type deviceAdmissionTransitionBackend struct {
	capacityProbeBackend
	initialFree int64
	probes      int
}

func (b *deviceAdmissionTransitionBackend) DeviceMemory() (int64, int64, bool) {
	b.probes++
	if b.probes == 1 {
		return b.total, b.initialFree, true
	}
	return b.total, b.free, true
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionSnapshotTransitionChargesWeights(t *testing.T) {
	for _, freeBecomesUnknown := range []bool{true, false} {
		p, want := knownTotalDeviceAdmissionFixture(t, 1)
		weights := p.requestMemoryPlan(10, 5).ByClass()[compute.MemoryWeights]
		margin := min(weights/2, want/2)
		total := int64(math.Ceil(float64(want+margin) / (1 - inKernelRequestDeviceHeadroom)))
		if budget := ApplyByteHeadroom(total, inKernelRequestDeviceHeadroom); budget < want || budget >= want+weights {
			t.Fatalf("fixture budget=%d must fit runtime %d but not shared weights %d", budget, want, weights)
		}
		be := &deviceAdmissionTransitionBackend{capacityProbeBackend: capacityProbeBackend{total: total, free: total, known: true}, initialFree: compute.FreeUnknown}
		if freeBecomesUnknown {
			be.initialFree, be.free = total, compute.FreeUnknown
		}
		p.backend = be
		release, err := p.admitCoalescedDeviceMemory(10, 5)
		defer release()
		if freeBecomesUnknown {
			var capacityErr *InKernelCapacityError
			if !errors.As(err, &capacityErr) || capacityErr.Scope != compute.MemoryScopeDevice || p.deviceReservedBytes != 0 {
				t.Fatalf("unknown-free snapshot failed to charge shared weights: err=%v reserved=%d", err, p.deviceReservedBytes)
			}
		} else if err != nil || p.deviceReservedBytes != want {
			t.Fatalf("known-free snapshot retained shared weight charge: err=%v reserved=%d", err, p.deviceReservedBytes)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionKnownTotalConcurrentOneWinner(t *testing.T) {
	p, want := knownTotalDeviceAdmissionFixture(t, 1)
	start := make(chan struct{})
	results := make(chan func(), 8)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			release, err := p.admitCoalescedDeviceMemory(10, 5)
			if err == nil {
				results <- release
				return
			}
			var capacityErr *InKernelCapacityError
			if !errors.As(err, &capacityErr) || capacityErr.Site != inKernelCoalescedDeviceReservationSite {
				t.Errorf("concurrent refusal = %T (%v)", err, err)
			}
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	winners := 0
	for release := range results {
		winners++
		defer release()
	}
	if winners != 1 || p.deviceReservedBytes != want {
		t.Fatalf("winners=%d reserved=%d, want 1/%d", winners, p.deviceReservedBytes, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInKernelCoalescedDeviceAdmissionKnownTotalOverflow(t *testing.T) {
	p, _ := knownTotalDeviceAdmissionFixture(t, 1)
	p.deviceReservedBytes = math.MaxInt64
	_, err := p.admitCoalescedDeviceMemory(10, 5)
	var capacityErr *InKernelCapacityError
	if !errors.As(err, &capacityErr) || capacityErr.Site != inKernelCoalescedDeviceReservationSite || p.deviceReservedBytes != math.MaxInt64 {
		t.Fatalf("overflow refusal=%v reserved=%d, want unchanged reservation", err, p.deviceReservedBytes)
	}
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
