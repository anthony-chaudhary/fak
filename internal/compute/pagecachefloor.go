package compute

// pagecachefloor.go — the OS page cache as a first-class host-memory budget term (#4362).
//
// BudgetAfterHeadroom (capacity.go) reserves a FRACTION of the host budget for the bytes that
// are not in the checked plan. That is the right shape for ANONYMOUS host RAM: on the pure-CPU
// serve and the --cpu-offload-experts path the weights are copied into the process, every byte
// the app declines to take is a byte it could have taken, and the reserve should scale with the
// box.
//
// It is the wrong shape for DEMAND-PAGED weight serving. When weights are mapped and faulted in
// on reference, the OS page cache is not slack the app failed to claim — it is the serve's own
// read-through tier, and bytes handed to the app's resident pool are bytes taken away from it.
// Past the point where the resident working set stops missing, the last few GB of app residency
// return LESS than the buffered-pread bandwidth they cost, because a squeezed page cache turns
// every re-touch of a cold span back into a disk read. So the honest budget carves that reserve
// out FIRST, and carves it out as an ABSOLUTE floor rather than a fraction: the throughput cliff
// is a property of the backing device's buffered-read behaviour, not of how much RAM the box
// happens to have, so a fraction under-reserves on a small host and over-reserves on a large one.
//
// The host budget for a demand-paged plan is therefore the TIGHTER of the two terms:
//
//	min(BudgetAfterHeadroom(memAvailable, headroom), memAvailable-floorBytes)
//
// Scope, honestly stated. This file owns the budget ARITHMETIC and the fit refusal built on it;
// it does not own the NUMBER. floorBytes is caller-supplied and OFF by default (<= 0), so every
// path that exists today computes a byte-identical budget and nothing changes until a caller
// opts in. Sizing the floor to a MEASURED buffered-pread throughput cliff, and calling this from
// a mapped/pread weight loader, land with the demand-paged serve itself (#974 / #974-B) — a
// measured constant is deliberately not invented here.

// HostBudgetForPagedWeights is the host-memory budget for a DEMAND-PAGED plan: the fractional
// headroom term and the absolute page-cache floor term, whichever binds tighter.
//
// memAvailable is the host's allocatable bytes (on Linux, MemAvailable — see hostmem_linux.go),
// headroom is the same [0,1) fraction BudgetAfterHeadroom applies for the bytes outside the
// checked plan, and floorBytes is the page-cache reserve to hold back so the mapped weights keep
// a read-through tier. A floorBytes <= 0 disables the floor term and the result is exactly
// BudgetAfterHeadroom, so this is a drop-in for the fraction-only budget.
//
// A floor larger than memAvailable clamps the budget to 0 rather than going negative: on such a
// host the reserve consumes everything, and 0 is the honest budget — any positive demand then
// refuses, instead of a negative budget silently reading as "unknown".
func HostBudgetForPagedWeights(memAvailable int64, headroom float64, floorBytes int64) int64 {
	budget := BudgetAfterHeadroom(memAvailable, headroom)
	if floorBytes <= 0 || memAvailable <= 0 {
		return budget
	}
	afterFloor := memAvailable - floorBytes
	if afterFloor < 0 {
		afterFloor = 0
	}
	if afterFloor < budget {
		return afterFloor
	}
	return budget
}

// RefusePagedHostPlanIfTooBig is the demand-paged counterpart of
// RefuseHostScopedPlanIfTooBigForHost (capacity.go): it checks a plan's HOST-scoped demands
// against HostBudgetForPagedWeights instead of the fraction-only budget, so a plan that would fit
// the headroom term but eat into the page-cache floor is refused BEFORE the mapping is made
// resident and the serve starts trading LRU hits for disk reads.
//
// It keeps the capacity.go contract: fail-OPEN (a host that cannot report its memory yields nil
// and loads exactly as before), host-scoped FitError carrying just the host demands so the
// refusal names the paged pool rather than any VRAM-resident dense weights, and floorBytes <= 0
// reproducing the existing fraction-only verdict byte-for-byte.
func RefusePagedHostPlanIfTooBig(plan MemoryPlan, headroom float64, floorBytes int64) error {
	total, free, known := HostSystemMemoryInfo()
	return refusePagedHostPlanForHostMem(plan, total, free, known, headroom, floorBytes)
}

// refusePagedHostPlanForHostMem is the injectable core of RefusePagedHostPlanIfTooBig: it takes
// the host (total, free, known) explicitly so the refusal is testable without a live
// /proc/meminfo, exactly as refuseHostScopedPlanForHostMem does for the fraction-only guard.
func refusePagedHostPlanForHostMem(plan MemoryPlan, total, free int64, known bool, headroom float64, floorBytes int64) error {
	want := plan.HostTotal()
	if want <= 0 {
		return nil
	}
	if !known || total <= 0 {
		return nil
	}
	base := free
	if base < 0 { // FreeUnknown -> fall back to the total ceiling, conservatively
		base = total
	}
	avail := HostBudgetForPagedWeights(base, headroom, floorBytes)
	if want <= avail {
		return nil
	}
	hostPlan := make(MemoryPlan, 0, len(plan))
	for _, d := range plan {
		if d.Bytes > 0 && d.ScopeOrDefault() == MemoryScopeHost {
			hostPlan = append(hostPlan, d)
		}
	}
	return &FitError{Verdict: FitTooBig, Want: want, Avail: avail, Demands: hostPlan, Scope: MemoryScopeHost}
}
