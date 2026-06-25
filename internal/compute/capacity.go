package compute

// capacity.go — the eighth assumption the HAL lifts, at the type level.
//
// internal/compute/compute.go neutralizes SEVEN hardware-SHAPE assumptions (dtype
// monoculture, host-pointer aliasing, x86 build-tag dispatch, synchronous return,
// goroutine parallelism, row-major layout, eager full-RAM residency). All seven describe
// what a device LOOKS like. None describes the one thing every accelerator actually runs
// out of: finite, exhaustible memory. The forward loop, and the layers above it, have no
// way to ASK a backend "how much memory do you have, and will this fit?" — so today a
// device that is too small does not refuse, it PANICS mid-allocation (cuda/metal dalloc),
// and the one capacity number a backend already holds (the CUDA device's totalGlobalMem,
// read in fcuda_init) is discarded. Caps.DeviceMemory says only that resident tensors are
// NOT host-addressable (a SHAPE fact); it does not say how MUCH device memory exists.
//
// DeviceCapacity is the eighth lift: the REPORT half of the hardware-capacity bridge. A
// backend that can probe its own ceiling implements it; the helpers below let any caller
// ask the question without knowing the concrete backend, and — critically — they FAIL
// OPEN. A backend that cannot answer (the pure-Go cpu-ref floor, a wasm target, a device
// whose driver does not expose a memory query) reports known=false, and the fit check
// returns FitUnknown ("proceed"), never FitTooBig. So adding the capability never blocks
// a path that worked before; it only lets a backend that KNOWS its limit turn a downstream
// OOM panic into an answerable, typed pre-check.
//
// Honest scope, in the grain of the HAL doc: this lifts the capability into the type
// system, the same way the other seven are lifted "even though only the CPU reference is
// implemented today." It does NOT itself enforce a fit before load, nor spill a span when
// a tier fills — that is the rest of the bridge: a backend reports pressure HERE, the
// cachemeta placement plane (PlanPlacement / TierPressure, already built and tested) plans
// the demote-not-evict move, and an engine adapter performs it. Those planks are tracked
// in docs/explainers/hardware-limits-and-capacity.md; this file ships the first one, the
// one the cuda backend can satisfy from a number it already reads.

// FreeUnknown is the sentinel a DeviceCapacity returns for `free` when the total ceiling is
// known but the currently-allocatable bytes are not yet probeable (e.g. the cuda backend
// before a cudaMemGetInfo free-memory query is wired). It is negative so it can never be
// mistaken for a real free-byte count; FitsOnDevice falls back to the total ceiling when it
// sees it (conservative: it treats the whole device as the budget rather than guessing).
const FreeUnknown int64 = -1

// DeviceCapacity is the OPTIONAL capability a Backend implements to report the finite size
// of the memory it owns. It is discovered the same way every other optional HAL capability
// is (CollectiveBackend, DSASparseBackend): a caller type-asserts the Backend for it and
// cross-checks Caps().CapacityProbe, falling back to "unknown" when it is absent. The CPU
// reference does NOT implement it — host RAM is implicit, and probing it would need the
// syscalls the reference deliberately avoids to stay stdlib-only and wasm-clean — so the
// portable floor correctly reports its capacity as unknown.
type DeviceCapacity interface {
	Backend
	// DeviceMemory reports the device's memory ceiling and current headroom.
	//
	//   known == false : the backend cannot probe its capacity. total/free are
	//                    meaningless; callers MUST treat capacity as unknown and proceed
	//                    (the fail-open contract). This is the portable-floor answer.
	//   known == true  : total > 0 is the hard ceiling in bytes. free is the currently
	//                    allocatable bytes, or FreeUnknown when total is known but free is
	//                    not yet probeable.
	DeviceMemory() (total, free int64, known bool)
}

// FitVerdict is the typed outcome of a capacity pre-check — the answerable form of the
// question a downstream cuda/metal dalloc answers today only by panicking.
type FitVerdict uint8

const (
	// FitUnknown: capacity is not probeable on this backend. Proceed — this is the
	// fail-open verdict that keeps the portable floor and any non-reporting backend
	// working exactly as before the capability existed.
	FitUnknown FitVerdict = iota
	// FitOK: the request fits within the known (headroom-adjusted) budget.
	FitOK
	// FitTooBig: the backend KNOWS the request exceeds its ceiling. A caller can refuse
	// with a sizing message BEFORE the allocation that would otherwise panic.
	FitTooBig
)

// String renders the verdict for logs and refusal messages.
func (f FitVerdict) String() string {
	switch f {
	case FitOK:
		return "ok"
	case FitTooBig:
		return "too_big"
	default:
		return "unknown"
	}
}

// DeviceMemoryInfo reports (total, free, known) for ANY backend, hiding the type-assert.
// It returns known=false unless b both implements DeviceCapacity AND advertises
// Caps().CapacityProbe — the two-part discovery idiom the rest of the HAL uses (the cap
// is the cheap pre-check; the interface is the source of truth, and requiring both means a
// backend cannot half-advertise). A non-reporting backend yields (0, FreeUnknown, false).
func DeviceMemoryInfo(b Backend) (total, free int64, known bool) {
	if b == nil {
		return 0, FreeUnknown, false
	}
	dc, ok := b.(DeviceCapacity)
	if !ok || !b.Caps().CapacityProbe {
		return 0, FreeUnknown, false
	}
	return dc.DeviceMemory()
}

// FitsOnDevice answers whether wantBytes can be placed on b's device without exhausting it
// — turning a would-be OOM panic into a typed pre-check. It FAILS OPEN: a backend whose
// capacity is unknown yields FitUnknown (proceed), never FitTooBig, so this never blocks
// the portable floor or a device that cannot report. It fails CLOSED (FitTooBig) only when
// the backend KNOWS the request exceeds its budget.
//
// headroom in [0,1) reserves that fraction of the budget for the bytes that do NOT pass
// through this single check — the KV cache, activations, and per-op scratch — so a model
// that exactly fills weights does not leave zero room to run. A headroom outside [0,1) is
// ignored (treated as 0). The returned avail is the budget the verdict was computed
// against (headroom-applied), so a caller can build a "needs W, have A" message.
//
// When free is FreeUnknown but total is known, the budget is the total ceiling: the check
// can still catch a model that does not fit the WHOLE device, just not one that fits the
// device but not the current free headroom (that needs a live free-memory probe — the
// tracked follow-up for the cuda backend).
func FitsOnDevice(b Backend, wantBytes int64, headroom float64) (verdict FitVerdict, avail int64) {
	total, free, known := DeviceMemoryInfo(b)
	if !known || total <= 0 {
		return FitUnknown, 0
	}
	budget := free
	if budget < 0 { // FreeUnknown -> fall back to the total ceiling, conservatively
		budget = total
	}
	if headroom > 0 && headroom < 1 {
		budget = int64(float64(budget) * (1 - headroom))
	}
	if wantBytes <= budget {
		return FitOK, budget
	}
	return FitTooBig, budget
}
