package compute

import (
	"errors"
	"testing"
)

// TestHostBudgetForPagedWeights pins the two-term budget arithmetic for demand-paged weight
// serving (#4362): the fractional headroom term and the ABSOLUTE page-cache floor term, whichever
// binds tighter. The floor exists because a squeezed page cache turns every re-touch of a cold
// mapped span back into a buffered pread, so the last GB of app residency costs more bandwidth
// than it returns — and because that cliff is a property of the backing device, a fraction
// under-reserves on a small host and over-reserves on a large one.
func TestHostBudgetForPagedWeights(t *testing.T) {
	const gib = int64(1) << 30
	const headroom = 0.15

	// Floor OFF (<= 0) is the drop-in contract: byte-identical to the fraction-only budget, so
	// every path that exists today is unchanged until a caller opts in.
	for _, floor := range []int64{0, -1, -64 * gib} {
		if got, want := HostBudgetForPagedWeights(64*gib, headroom, floor), BudgetAfterHeadroom(64*gib, headroom); got != want {
			t.Errorf("floor=%d: budget = %d, want the fraction-only budget %d (floor <= 0 must be a no-op)", floor, got, want)
		}
	}

	// Small host — the floor BINDS. 64 GiB usable at 15% reserves only ~9.6 GiB, less than the
	// 16 GiB the page cache needs to stay off the pread cliff, so the absolute term wins.
	if got, want := HostBudgetForPagedWeights(64*gib, headroom, 16*gib), 48*gib; got != want {
		t.Errorf("small host: budget = %d, want %d (MemAvailable-floor; the fraction under-reserves here)", got, want)
	}

	// Large host — the FRACTION binds. 1 TiB usable at 15% already holds back ~153 GiB, far more
	// than the 16 GiB floor, so carving the floor out again would be double-reserving.
	if got, want := HostBudgetForPagedWeights(1024*gib, headroom, 16*gib), BudgetAfterHeadroom(1024*gib, headroom); got != want {
		t.Errorf("large host: budget = %d, want the (tighter) fractional budget %d", got, want)
	}

	// A floor at or above MemAvailable clamps to 0, never negative: on such a host the reserve
	// consumes everything and 0 is the honest budget — any positive demand then refuses, whereas a
	// negative budget would read downstream as "unknown".
	for _, floor := range []int64{8 * gib, 64 * gib} {
		if got := HostBudgetForPagedWeights(8*gib, headroom, floor); got != 0 {
			t.Errorf("floor %d >= MemAvailable 8 GiB: budget = %d, want 0 (clamped, not negative)", floor, got)
		}
	}

	// A non-positive MemAvailable is passed through exactly as BudgetAfterHeadroom does, so an
	// unknown/absent host probe cannot be turned into a bogus budget by the floor term.
	for _, mem := range []int64{0, -5} {
		if got := HostBudgetForPagedWeights(mem, headroom, 16*gib); got != mem {
			t.Errorf("MemAvailable=%d: budget = %d, want it returned unchanged", mem, got)
		}
	}
}

// TestRefusePagedHostPlanForHostMem covers the demand-paged host fit refusal
// (refusePagedHostPlanForHostMem, the injectable core of RefusePagedHostPlanIfTooBig, #4362).
// The load-bearing case is the middle one: a plan that PASSES the fraction-only guard but would
// eat into the page-cache floor must be refused, so the mapped weights keep their read-through
// tier instead of trading LRU hits for disk reads.
func TestRefusePagedHostPlanForHostMem(t *testing.T) {
	const gib = int64(1) << 30
	const headroom = 0.15
	const floor = 12 * gib
	// A demand-paged serve: dense weights live in VRAM (device-scoped), the mapped/pread expert
	// weights are the host-scoped pool that faults through the OS page cache.
	plan := MemoryPlan{
		{Class: MemoryWeights, Bytes: 8 * gib, Detail: "gguf-device-dense-load", Scope: MemoryScopeDevice},
		{Class: MemoryOffload, Bytes: 50 * gib, Detail: "gguf-host-paged-weights", Scope: MemoryScopeHost},
	}

	// 60 GiB MemAvailable, 15% headroom -> 51 GiB fraction-only budget, so the 50 GiB paged pool
	// FITS the existing guard. Pin that, so the refusal below is provably the floor term's doing
	// and not just a tighter fraction.
	if err := refuseHostScopedPlanForHostMem(plan, 64*gib, 60*gib, true, headroom); err != nil {
		t.Fatalf("precondition: the fraction-only guard must ACCEPT this plan, got %v", err)
	}
	if err := refusePagedHostPlanForHostMem(plan, 64*gib, 60*gib, true, headroom, 0); err != nil {
		t.Fatalf("floor off: want the byte-identical fraction-only verdict (nil), got %v", err)
	}

	// Same host, floor ON: 60 - 12 = 48 GiB < 50 GiB paged pool -> refuse BEFORE the mapping is
	// made resident and the page cache is squeezed onto the buffered-pread cliff.
	err := refusePagedHostPlanForHostMem(plan, 64*gib, 60*gib, true, headroom, floor)
	if err == nil {
		t.Fatal("50 GiB paged pool into 60 GiB MemAvailable with a 12 GiB page-cache floor: want FitTooBig, got nil")
	}
	var fe *FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *FitError, got %T: %v", err, err)
	}
	if fe.Verdict != FitTooBig {
		t.Errorf("verdict = %s, want too_big", fe.Verdict)
	}
	if fe.Scope != MemoryScopeHost {
		t.Errorf("scope = %q, want host (the paged pool is host memory)", fe.Scope)
	}
	// Want is the HOST-scoped total only — the VRAM-resident dense weights must not count.
	if fe.Want != 50*gib {
		t.Errorf("want bytes = %d, want host-scoped-only %d (device weights must not count)", fe.Want, 50*gib)
	}
	// Avail is the floor-carved budget, so the operator message names the number that actually bound.
	if fe.Avail != 60*gib-floor {
		t.Errorf("avail = %d, want the floor-carved budget %d", fe.Avail, 60*gib-floor)
	}
	for _, d := range fe.Demands {
		if d.ScopeOrDefault() != MemoryScopeHost {
			t.Errorf("FitError carried a non-host demand %+v; the refusal must name only the paged pool", d)
		}
	}

	// A plan that KEEPS THE FLOOR INTACT loads: 40 GiB paged pool leaves 20 GiB >= the 12 GiB
	// reserve, and it is under the 51 GiB fraction term too -> nil.
	intact := MemoryPlan{
		{Class: MemoryWeights, Bytes: 8 * gib, Detail: "gguf-device-dense-load", Scope: MemoryScopeDevice},
		{Class: MemoryOffload, Bytes: 40 * gib, Detail: "gguf-host-paged-weights", Scope: MemoryScopeHost},
	}
	if err := refusePagedHostPlanForHostMem(intact, 64*gib, 60*gib, true, headroom, floor); err != nil {
		t.Errorf("40 GiB paged pool (floor intact): want nil, got %v", err)
	}

	// The fraction still binds where it is tighter: on a big box the headroom term, not the floor,
	// is what refuses an oversized pool.
	huge := MemoryPlan{{Class: MemoryOffload, Bytes: 950 * gib, Scope: MemoryScopeHost}}
	if err := refusePagedHostPlanForHostMem(huge, 1024*gib, 1000*gib, true, headroom, floor); err == nil {
		t.Error("950 GiB pool into a 850 GiB fractional budget: want FitTooBig from the headroom term, got nil")
	}

	// FreeUnknown falls back to the total ceiling, conservatively — the same rule the fraction-only
	// guard uses — and the floor is carved out of that ceiling.
	if err := refusePagedHostPlanForHostMem(plan, 60*gib, FreeUnknown, true, headroom, floor); err == nil {
		t.Error("FreeUnknown on a 60 GiB box: want the total-ceiling budget to refuse the 50 GiB pool, got nil")
	}

	// A device-only plan asks nothing of host RAM -> never refuse, even on a tiny box.
	deviceOnly := MemoryPlan{{Class: MemoryWeights, Bytes: 400 * gib, Scope: MemoryScopeDevice}}
	if err := refusePagedHostPlanForHostMem(deviceOnly, 8*gib, 1*gib, true, headroom, floor); err != nil {
		t.Errorf("device-only plan: want nil (the paged host guard ignores VRAM weights), got %v", err)
	}

	// Fail-open, unchanged from capacity.go: a host that cannot report its memory never refuses.
	if err := refusePagedHostPlanForHostMem(plan, 0, FreeUnknown, false, headroom, floor); err != nil {
		t.Errorf("unreported host memory: want fail-open nil, got %v", err)
	}
	if err := refusePagedHostPlanForHostMem(nil, 64*gib, 60*gib, true, headroom, floor); err != nil {
		t.Errorf("empty plan: want nil, got %v", err)
	}
}
