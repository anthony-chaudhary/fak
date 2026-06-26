package compute

import (
	"strconv"
	"strings"
)

// capacity.go — the eighth assumption the HAL lifts, at the type level.
//
// internal/compute/compute.go neutralizes SEVEN hardware-SHAPE assumptions (dtype
// monoculture, host-pointer aliasing, x86 build-tag dispatch, synchronous return,
// goroutine parallelism, row-major layout, eager full-RAM residency). All seven describe
// what a device LOOKS like. None describes the one thing every accelerator actually runs
// out of: finite, exhaustible memory. The forward loop, and the layers above it, have no
// way to ASK a backend "how much memory do you have, and will this fit?" — so today a
// device that is too small does not refuse, it PANICS mid-allocation (cuda/metal dalloc),
// and capacity numbers a backend already holds (CUDA totalGlobalMem, Vulkan device-local
// heap size) used to stay outside the HAL contract. Caps.DeviceMemory says only that resident tensors are
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
// one a backend can satisfy from a number it already reads.

// FreeUnknown is the sentinel a DeviceCapacity returns for `free` when the total ceiling is
// known but the currently-allocatable bytes are not yet probeable (e.g. CUDA before
// cudaMemGetInfo, or Vulkan before a budget/free-memory extension is wired). It is negative so it can never be
// mistaken for a real free-byte count; FitsOnDevice falls back to the total ceiling when it
// sees it (conservative: it treats the whole device as the budget rather than guessing).
const FreeUnknown int64 = -1

const maxInt64Uint64 = uint64(1<<63 - 1)

func uint64ToCapInt64(v uint64) int64 {
	if v > maxInt64Uint64 {
		return int64(maxInt64Uint64)
	}
	return int64(v)
}

// MemoryClass names the pool or purpose behind a capacity request. The byte-only
// FitsOnDevice helper remains for legacy callers, but a MemoryPlan preserves the
// distinction operators actually need when a node refuses a request: weights,
// KV-cache residency, DDR/DRAM cache, offload staging, and per-op scratchpad are
// different remedies even when they all consume bytes from the same finite device.
type MemoryClass string

const (
	MemoryUnknown    MemoryClass = "unknown"
	MemoryWeights    MemoryClass = "weights"
	MemoryKVCache    MemoryClass = "kv_cache"
	MemoryDDRCache   MemoryClass = "ddr_cache"
	MemoryOffload    MemoryClass = "offload"
	MemoryScratchpad MemoryClass = "scratchpad"
	MemoryActivation MemoryClass = "activation"
)

// MemoryScope says which finite pool a demand consumes. Empty scope means device for backward
// compatibility with older MemoryDemand literals and with the HAL fit checks: a demand must opt
// into host scope before RefuseMemoryPlanIfTooBig stops counting it against device capacity.
type MemoryScope string

const (
	MemoryScopeDevice MemoryScope = "device"
	MemoryScopeHost   MemoryScope = "host"
)

// MemoryDemand is one classed demand in a fit check. Detail is optional human
// context ("gguf-q4k-load", "decode logits", "kv prefill window"); it is never
// parsed by policy code.
type MemoryDemand struct {
	Class  MemoryClass
	Bytes  int64
	Detail string
	Scope  MemoryScope
}

// MemoryPlan is the classed form of "will this fit?" A plan can hold several
// simultaneous demands against one backend capacity: e.g. weights + expected KV +
// scratch. It is intentionally pure data so loaders can build it from headers
// before allocating.
type MemoryPlan []MemoryDemand

// EstimateKVStoreBytes reports the resident bytes required by the compute.KVStore
// layout for tokens cached at once. The HAL KV contract stores three f32 rows per
// cached position per layer: pre-RoPE K (for exact re-positioning on evict),
// post-RoPE K, and V. Invalid or incomplete geometry returns 0 so callers can
// fail open when a config does not carry enough evidence for a cache plan.
func EstimateKVStoreBytes(cfg KVConfig, tokens int) int64 {
	return saturatingMulInt64(
		int64(cfg.NumLayers),
		int64(tokens),
		int64(cfg.NumKVHeads),
		int64(cfg.HeadDim),
		3, // Kraw + K + V
		4, // f32 bytes
	)
}

// EstimateKVStoreMemoryPlan is the classed form of EstimateKVStoreBytes.
func EstimateKVStoreMemoryPlan(cfg KVConfig, tokens int) MemoryPlan {
	bytes := EstimateKVStoreBytes(cfg, tokens)
	if bytes <= 0 {
		return nil
	}
	return MemoryPlan{{Class: MemoryKVCache, Bytes: bytes, Detail: "hal-kv-store"}}
}

// TransformerScratchConfig is the geometry needed to conservatively plan the f32 transient
// buffers used by the model HAL token path. It is intentionally architecture-small: it names
// only the dimensions that shape the common decoder block buffers.
type TransformerScratchConfig struct {
	HiddenSize       int
	IntermediateSize int
	VocabSize        int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	IncludeLogits    bool
}

// EstimateHALTransientMemoryPlan returns classed f32 activation/scratchpad demands for one
// token through the generic HAL path. It is a conservative token-boundary estimate: device
// backends recycle transient op outputs after the token, so this counts the unfused outputs that
// may coexist during a decode step. Incomplete geometry omits the unsupported demand rather than
// inventing a number, preserving the fail-open contract used by the other fit helpers.
func EstimateHALTransientMemoryPlan(cfg TransformerScratchConfig) MemoryPlan {
	var plan MemoryPlan
	if activation := EstimateHALActivationBytes(cfg); activation > 0 {
		plan = append(plan, MemoryDemand{Class: MemoryActivation, Bytes: activation, Detail: "hal-token-activation"})
	}
	if scratch := EstimateHALScratchpadBytes(cfg); scratch > 0 {
		plan = append(plan, MemoryDemand{Class: MemoryScratchpad, Bytes: scratch, Detail: "hal-token-scratch"})
	}
	return plan
}

// EstimateHALActivationBytes reports the per-token residual plus optional full-logits f32
// output. The served public Step/Prefill contract still materializes full logits for callers
// that request them, even when weights are Q8-resident.
func EstimateHALActivationBytes(cfg TransformerScratchConfig) int64 {
	if cfg.HiddenSize <= 0 {
		return 0
	}
	elems := int64(cfg.HiddenSize)
	if cfg.IncludeLogits && cfg.VocabSize > 0 {
		elems = saturatingAddInt64(elems, int64(cfg.VocabSize))
	}
	return saturatingMulInt64(elems, 4) // f32 bytes
}

// EstimateHALScratchpadBytes reports a conservative per-token sum of f32 transient op outputs
// for the unfused HAL chain: norms, Q/K/V projections, RoPE K copy, attention output, residual
// projection temps, FFN gate/up/SwiGLU, FFN down temp, and final norm. Fused backends may allocate
// less; this number is for safe admission, not a backend-specific profiler.
func EstimateHALScratchpadBytes(cfg TransformerScratchConfig) int64 {
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumLayers <= 0 ||
		cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return 0
	}
	h := int64(cfg.HiddenSize)
	i := int64(cfg.IntermediateSize)
	q := saturatingMulInt64(int64(cfg.NumHeads), int64(cfg.HeadDim))
	kv := saturatingMulInt64(int64(cfg.NumKVHeads), int64(cfg.HeadDim))
	if q <= 0 || kv <= 0 {
		return 0
	}
	perLayerElems := saturatingSumInt64(
		h,  // input RMSNorm
		q,  // q projection
		kv, // raw k projection
		kv, // v projection
		kv, // RoPE k copy for non-AppendKVRoPE backends
		q,  // attention output
		h,  // attention o_proj temp
		h,  // post-attention RMSNorm
		i,  // FFN gate projection
		i,  // FFN up projection
		i,  // SwiGLU output
		h,  // FFN down projection temp
	)
	elems := saturatingAddInt64(saturatingMulInt64(int64(cfg.NumLayers), perLayerElems), h) // final norm
	return saturatingMulInt64(elems, 4)                                                     // f32 bytes
}

func saturatingAddInt64(vals ...int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	var acc int64
	for _, v := range vals {
		if v <= 0 {
			continue
		}
		if acc > maxInt64-v {
			return maxInt64
		}
		acc += v
	}
	return acc
}

func saturatingSumInt64(vals ...int64) int64 {
	return saturatingAddInt64(vals...)
}

func saturatingMulInt64(vals ...int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	acc := int64(1)
	for _, v := range vals {
		if v <= 0 {
			return 0
		}
		if acc > maxInt64/v {
			return maxInt64
		}
		acc *= v
	}
	return acc
}

// Total returns the non-negative bytes requested by the plan, saturating at int64
// max on overflow so an impossible plan can only become more conservative.
func (p MemoryPlan) Total() int64 {
	return p.totalWhere(func(MemoryDemand) bool { return true })
}

// DeviceTotal returns the bytes in the plan that consume device capacity. Host-scoped
// demands remain in the plan for visibility but do not make a device fit check reject a
// valid offload placement.
func (p MemoryPlan) DeviceTotal() int64 {
	return p.totalWhere(func(d MemoryDemand) bool { return d.DeviceScoped() })
}

// HostTotal returns the bytes in the plan that consume host-side capacity. These demands
// are visible even when the backend cannot report host capacity; in that case the fit check
// fails open rather than pretending host RAM is bounded by the device ceiling.
func (p MemoryPlan) HostTotal() int64 {
	return p.totalWhere(func(d MemoryDemand) bool { return d.ScopeOrDefault() == MemoryScopeHost })
}

func (p MemoryPlan) totalWhere(include func(MemoryDemand) bool) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	var total int64
	for _, d := range p {
		if d.Bytes <= 0 || !include(d) {
			continue
		}
		if total > maxInt64-d.Bytes {
			return maxInt64
		}
		total += d.Bytes
	}
	return total
}

func (d MemoryDemand) DeviceScoped() bool {
	return d.Scope == "" || d.Scope == MemoryScopeDevice
}

func (d MemoryDemand) ScopeOrDefault() MemoryScope {
	if d.Scope == "" {
		return MemoryScopeDevice
	}
	return d.Scope
}

// ByClass folds the plan into byte totals per MemoryClass. Negative and zero
// demands are ignored, matching Total.
func (p MemoryPlan) ByClass() map[MemoryClass]int64 {
	out := map[MemoryClass]int64{}
	for _, d := range p {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = MemoryUnknown
		}
		out[class] += d.Bytes
	}
	return out
}

func cloneMemoryPlan(p MemoryPlan) MemoryPlan {
	if len(p) == 0 {
		return nil
	}
	out := make(MemoryPlan, len(p))
	copy(out, p)
	return out
}

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

// HostCapacity is the OPTIONAL companion to DeviceCapacity for host-scoped memory
// demands: CPU-offloaded expert weights, DDR/DRAM cache tiers, and host staging pools.
// Like DeviceCapacity it is trusted only when Caps().HostCapacityProbe is also set, so a
// backend cannot half-advertise a probe. Unknown host capacity fails open.
type HostCapacity interface {
	Backend
	// HostMemory reports the host-side memory ceiling and current headroom available to
	// this backend. FreeUnknown has the same meaning as DeviceMemory.
	HostMemory() (total, free int64, known bool)
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

// HostMemoryInfo reports host-side capacity for host-scoped memory demands. It follows
// the same two-part discovery contract as DeviceMemoryInfo: interface plus explicit cap
// flag, otherwise unknown/fail-open.
func HostMemoryInfo(b Backend) (total, free int64, known bool) {
	if b == nil {
		return 0, FreeUnknown, false
	}
	hc, ok := b.(HostCapacity)
	if !ok || !b.Caps().HostCapacityProbe {
		return 0, FreeUnknown, false
	}
	return hc.HostMemory()
}

func fitsWithinReportedMemory(total, free int64, known bool, wantBytes int64, headroom float64) (verdict FitVerdict, avail int64) {
	if wantBytes <= 0 {
		return FitOK, 0
	}
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
// device but not the current free headroom (that needs a live backend-specific
// free-memory probe).
func FitsOnDevice(b Backend, wantBytes int64, headroom float64) (verdict FitVerdict, avail int64) {
	total, free, known := DeviceMemoryInfo(b)
	return fitsWithinReportedMemory(total, free, known, wantBytes, headroom)
}

// FitsOnHost answers the same fit question for host-scoped demands. It is intentionally
// opt-in and fail-open: a backend must implement HostCapacity and advertise
// Caps().HostCapacityProbe before host-scoped offload/DDR bytes can refuse a plan.
func FitsOnHost(b Backend, wantBytes int64, headroom float64) (verdict FitVerdict, avail int64) {
	total, free, known := HostMemoryInfo(b)
	return fitsWithinReportedMemory(total, free, known, wantBytes, headroom)
}

// FitsMemoryPlan is FitsOnDevice for a classed memory plan. The verdict is still
// fail-open, but it now checks the demand's scope: device-scoped bytes against the device
// probe, and host-scoped bytes against the host probe only when the backend advertises one.
// Demand classes survive into the FitError and operator surfaces instead of being lost.
func FitsMemoryPlan(b Backend, plan MemoryPlan, headroom float64) (verdict FitVerdict, avail int64) {
	_, verdict, avail, _ = fitsMemoryPlanByScope(b, plan, headroom)
	return verdict, avail
}

func fitsMemoryPlanByScope(b Backend, plan MemoryPlan, headroom float64) (scope MemoryScope, verdict FitVerdict, avail int64, want int64) {
	deviceWant := plan.DeviceTotal()
	deviceVerdict, deviceAvail := FitsOnDevice(b, deviceWant, headroom)
	if deviceVerdict == FitTooBig {
		return MemoryScopeDevice, deviceVerdict, deviceAvail, deviceWant
	}
	hostWant := plan.HostTotal()
	hostVerdict, hostAvail := FitsOnHost(b, hostWant, headroom)
	if hostVerdict == FitTooBig {
		return MemoryScopeHost, hostVerdict, hostAvail, hostWant
	}
	if deviceVerdict == FitUnknown {
		return MemoryScopeDevice, deviceVerdict, deviceAvail, deviceWant
	}
	if hostVerdict == FitUnknown {
		return MemoryScopeHost, hostVerdict, hostAvail, hostWant
	}
	return MemoryScopeDevice, FitOK, deviceAvail, deviceWant
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
	Want    int64      // bytes the caller asked to place in Scope
	Avail   int64      // headroom-adjusted budget the verdict was computed against
	Demands MemoryPlan // optional classed breakdown of Want
	Scope   MemoryScope
}

func (e *FitError) Error() string {
	subject := "model"
	if len(e.Demands) > 0 {
		subject = "memory plan " + memoryPlanSummary(e.Demands)
	}
	scope := e.Scope
	if scope == "" {
		scope = MemoryScopeDevice
	}
	return "compute: " + subject + " needs " + memSize(e.Want) + ", " + string(scope) + " has " + memSize(e.Avail) +
		" (FitTooBig: request exceeds the reported " + string(scope) + " capacity; capacity pre-check turned a would-be OOM into a typed refusal)"
}

func memoryPlanSummary(plan MemoryPlan) string {
	type key struct {
		class MemoryClass
		scope MemoryScope
	}
	byClassScope := map[key]int64{}
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = MemoryUnknown
		}
		byClassScope[key{class: class, scope: d.ScopeOrDefault()}] += d.Bytes
	}
	if len(byClassScope) == 0 {
		return "{}"
	}
	order := []MemoryClass{
		MemoryWeights,
		MemoryKVCache,
		MemoryDDRCache,
		MemoryOffload,
		MemoryScratchpad,
		MemoryActivation,
		MemoryUnknown,
	}
	seen := map[key]bool{}
	parts := make([]string, 0, len(byClassScope))
	for _, class := range order {
		for _, scope := range []MemoryScope{MemoryScopeDevice, MemoryScopeHost} {
			k := key{class: class, scope: scope}
			if b, ok := byClassScope[k]; ok {
				parts = append(parts, memoryPlanSummaryLabel(class, scope)+"="+memSize(b))
				seen[k] = true
			}
		}
	}
	for k, b := range byClassScope {
		if !seen[k] {
			parts = append(parts, memoryPlanSummaryLabel(k.class, k.scope)+"="+memSize(b))
		}
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

func memoryPlanSummaryLabel(class MemoryClass, scope MemoryScope) string {
	if scope == MemoryScopeHost {
		return string(class) + "(host)"
	}
	return string(class)
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
	return &FitError{Verdict: verdict, Want: wantBytes, Avail: avail, Scope: MemoryScopeDevice}
}

// RefuseMemoryPlanIfTooBig is the class-preserving form of RefuseIfTooBig. It
// returns a FitError only for a known-too-small backend; unknown-capacity backends
// still fail open. The error carries both the total bytes and the original classed
// plan so a caller can render targeted remedies instead of a generic OOM.
func RefuseMemoryPlanIfTooBig(b Backend, plan MemoryPlan, headroom float64) error {
	scope, verdict, avail, want := fitsMemoryPlanByScope(b, plan, headroom)
	if verdict != FitTooBig {
		return nil
	}
	return &FitError{Verdict: verdict, Want: want, Avail: avail, Demands: cloneMemoryPlan(plan), Scope: scope}
}

// DeviceAllocError is the typed form of a RUNTIME in-kernel device allocation failure — the
// allocation that actually returned nil, as opposed to FitError's pre-flight refusal. It is
// raised (as a panic, since the failing alloc sits deep below a CGO boundary with no error
// return) by device backend allocation choke points and recovered at the in-kernel decode
// boundary, where it becomes an actionable error instead of crashing the serving goroutine.
//
// It carries the requested size so a caller can render a fak-owned, leak-free message (Bytes is
// fak's own number, never upstream content). The real CUDA cause — a genuine OOM vs a context
// poisoned by a prior async kernel fault — may be printed by the backend shim before nil is
// returned; Site names the allocation choke point that raised it, for the operator log. This
// lives in capacity.go (plain Go, no device build tag) so packages that only import compute for
// its types — e.g. internal/agent — can errors.As it without a GPU build.
type DeviceAllocError struct {
	Bytes int         // the device allocation that failed
	Site  string      // "dalloc" | "dallocManaged" | "evict-scratch" — the choke point that raised it
	Class MemoryClass // weights | kv_cache | offload | scratchpad | activation | unknown
}

func (e *DeviceAllocError) Error() string {
	class := e.DemandClass()
	subject := "device"
	if class != MemoryUnknown {
		subject = string(class)
	}
	site := e.Site
	if site == "" {
		site = "device allocator"
	}
	return "compute: " + subject + " allocation of " + memSize(int64(e.Bytes)) + " failed (" + site +
		"; device allocator returned nil)"
}

func (e *DeviceAllocError) DemandClass() MemoryClass {
	if e == nil || e.Class == "" {
		return MemoryUnknown
	}
	return e.Class
}
