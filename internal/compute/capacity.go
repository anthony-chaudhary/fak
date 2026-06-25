package compute

import "strconv"

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

// FitError is the typed refusal RefuseIfTooBig returns when a capacity-reporting backend
// KNOWS the requested bytes exceed its device — the answerable form of the allocation that
// would otherwise panic (cuda/metal dalloc) mid-load. It carries the sizing so a caller can
// surface "needs ~W, device has ~A" before a byte is allocated.
//
// It is returned ONLY for FitTooBig. A backend whose capacity is unknown (the pure-Go cpu-ref
// floor, a device that cannot probe) never produces one — RefuseIfTooBig returns nil there, so
// the portable floor and any non-reporting backend load exactly as before the capability added
// this refusal (the fail-open contract from DeviceCapacity, docs/explainers/hardware-limits-
// and-capacity.md Plank 5).
type FitError struct {
	Verdict FitVerdict // always FitTooBig
	Want    int64      // bytes the caller asked to place
	Avail   int64      // headroom-adjusted budget the verdict was computed against
}

func (e *FitError) Error() string {
	return "compute: model needs " + memSize(e.Want) + ", device has " + memSize(e.Avail) +
		" (FitTooBig: model exceeds the reported device capacity; RefuseIfTooBig turned a would-be OOM into a typed refusal)"
}

// memSize renders a byte count in the largest binary unit that keeps at least one of it, so a
// refusal message reads "needs ~4.13 GiB" / "device has ~1.00 MiB" rather than a bare integer.
// It is for human-facing refusal strings only; the exact bytes live on the FitError fields.
func memSize(b int64) string {
	switch {
	case b >= 1<<30:
		return strconv.FormatFloat(float64(b)/float64(1<<30), 'f', 2, 64) + " GiB"
	case b >= 1<<20:
		return strconv.FormatFloat(float64(b)/float64(1<<20), 'f', 2, 64) + " MiB"
	case b >= 1<<10:
		return strconv.FormatFloat(float64(b)/float64(1<<10), 'f', 2, 64) + " KiB"
	default:
		return strconv.FormatInt(b, 10) + " B"
	}
}

// RefuseIfTooBig turns FitsOnDevice's verdict into a typed, fail-open admission decision: it
// returns a *FitError (with the "needs ~W, device has ~A" sizing) ONLY when b knows the
// request exceeds its ceiling (FitTooBig), and nil otherwise — FitOK obviously, and CRUCIALLY
// FitUnknown (a backend that cannot probe, i.e. the cpu-ref floor), so wiring this into a load
// path never blocks a path that worked before. It is the load-time half of the capacity bridge
// (docs/explainers/hardware-limits-and-capacity.md Plank 5): call it with a pre-load
// weight-footprint estimate before the make/append that would otherwise OOM-panic, and a
// capacity-reporting backend turns "too big" into an answerable refusal instead of a panic.
func RefuseIfTooBig(b Backend, wantBytes int64, headroom float64) error {
	verdict, avail := FitsOnDevice(b, wantBytes, headroom)
	if verdict != FitTooBig {
		return nil
	}
	return &FitError{Verdict: verdict, Want: wantBytes, Avail: avail}
}

// DeviceAllocError is the typed form of a RUNTIME in-kernel device allocation failure — the
// allocation that actually returned nil, as opposed to FitError's pre-flight refusal. It is
// raised (as a panic, since the failing alloc sits deep below a CGO boundary with no error
// return) by the cuda backend's allocation choke points and recovered at the in-kernel decode
// boundary, where it becomes an actionable error instead of crashing the serving goroutine.
//
// It carries the requested size so a caller can render a fak-owned, leak-free message (Bytes is
// fak's own number, never upstream content). The real CUDA cause — a genuine OOM vs a context
// poisoned by a prior async kernel fault — is printed to fak-cuda stderr by fcuda_malloc /
// fcuda_malloc_managed; Site names the allocation choke point that raised it, for the operator
// log. This lives in capacity.go (plain Go, no `cuda` build tag) so packages that only import
// compute for its types — e.g. internal/agent — can errors.As it without the cuda build.
type DeviceAllocError struct {
	Bytes int    // the device allocation that failed
	Site  string // "dalloc" | "dallocManaged" | "evict-scratch" — the choke point that raised it
}

func (e *DeviceAllocError) Error() string {
	return "compute: cuda device allocation of " + memSize(int64(e.Bytes)) + " failed (" + e.Site +
		"; cudaMalloc and managed fallback both returned nil; see fak-cuda stderr for the real CUDA error)"
}
