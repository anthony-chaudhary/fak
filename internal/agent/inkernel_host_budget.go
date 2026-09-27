package agent

import (
	"context"
	"log"
	"runtime"
	"runtime/debug"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Host-memory admission for the host-session (Metal) seam (#13267).
//
// On darwin the resident `fak up` planner runs with backend == nil and metal == true:
// the KV cache, every prefix-cache clone, and the prefill panels are ordinary host (Go
// heap / unified) memory. refuseOversizeRequest asks a compute.Backend for its capacity,
// so on that seam it never ran, and nothing bounded the process before a divergent-suffix
// prefill: repeated divergent prompts grew the session KV plus up to three retained
// prefix-cache copies per request until the kernel jetsam-killed the server. This arm
// gives that seam the same typed, pre-allocation decline the device path has, priced
// against the operator's process ceiling (`fak up --max-rss`), and it only engages once a
// caller arms it — an unarmed planner is byte-identical to the historical path.

// inKernelHostAdmissionCopies is how many retained prefix-cache copies of the request's
// state one host-session turn can admit: the adaptive checkpoint snapshot, the full-prompt
// clone, and the generated-continuation clone (inkernel_decode.go steps 2-4).
const inKernelHostAdmissionCopies = 3

// inKernelHostReliefHalvings bounds the pressure-relief loop: the prefix cache is halved
// this many times before it is emptied, so relief costs at most a handful of GC passes.
const inKernelHostReliefHalvings = 3

// hostMemoryBudget is the armed per-process ceiling and its live usage probe.
type hostMemoryBudget struct {
	ceiling  int64
	used     func() (int64, bool)
	reserved int64 // bytes admitted to in-flight requests and not yet released
}

// inKernelSkipPrefixAdmissionKey marks a request whose working set fits the host ceiling
// but whose retained prefix-cache copies would not: it is served, but not cached.
type inKernelSkipPrefixAdmissionKey struct{}

func inKernelSkipPrefixAdmission(ctx context.Context) bool {
	skip, _ := ctx.Value(inKernelSkipPrefixAdmissionKey{}).(bool)
	return skip
}

// SetHostMemoryBudget arms the host-memory admission check for the host-session seam.
// ceiling is the process byte ceiling (the `fak up --max-rss` contract) and used reports
// the process's current resident bytes. A non-positive ceiling or a nil probe disarms it.
func (p *InKernelPlanner) SetHostMemoryBudget(ceiling int64, used func() (int64, bool)) {
	if p == nil {
		return
	}
	p.hostBudgetMu.Lock()
	defer p.hostBudgetMu.Unlock()
	if ceiling <= 0 || used == nil {
		p.hostBudget = nil
		return
	}
	p.hostBudget = &hostMemoryBudget{ceiling: ceiling, used: used}
}

// hostMemoryAdmissionArmed reports whether this request must pass the host arm: the
// planner is on the host-session seam (no compute.Backend to ask) and a budget is armed.
func (p *InKernelPlanner) hostMemoryAdmissionArmed() bool {
	if p == nil || p.backend != nil || p.m == nil {
		return false
	}
	p.hostBudgetMu.Lock()
	defer p.hostBudgetMu.Unlock()
	return p.hostBudget != nil
}

// hostRequestDemand prices one host-session turn. session is the request's own working
// set: the KV plus fixed recurrent state for prompt+maxNew positions (a prefix hit clones
// the whole cached prefix into the session, so the prompt is new memory either way), the
// transient scratch, and the host prefill-panel live peak. retained is the prefix-cache
// copies the turn will admit when reuse is on. It deliberately prices the base context
// plan, not requestMemoryPlanWithExtras: those extras are the Vulkan prefill-panel and
// retained-MTP terms, whose estimator refuses a panel wider than the planned tokens and
// would turn every short host turn into a capacity-unknown refusal.
func (p *InKernelPlanner) hostRequestDemand(promptTokens, maxNew int) (session, retained int64, plan compute.MemoryPlan) {
	planned := promptTokens + maxNew
	if planned < promptTokens {
		planned = promptTokens
	}
	cs := p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision))
	_, plan = compute.AutoSizeContextPlan(cs, nil, compute.FreeUnknown, planned)
	session = plan.Total() + p.hostPrefillPanelBytes(promptTokens)
	if p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		if perCopy := compute.EstimateKVStoreBytes(cs.KV, planned) + cs.SessionState.Total(); perCopy > 0 {
			retained = perCopy * inKernelHostAdmissionCopies
		}
	}
	return session, retained, plan
}

// hostPrefillPanelBytes is the live f32 peak of one host prefill panel at the MLP: the
// residual, normed and down-projected rows (3 x hidden) plus the gate and up rows
// (2 x intermediate) for each panel token. The generic runtime-extras term prices only a
// hidden-width row, which undercounts the host panel by roughly an order of magnitude.
func (p *InKernelPlanner) hostPrefillPanelBytes(promptTokens int) int64 {
	if p == nil || p.m == nil || promptTokens <= 0 {
		return 0
	}
	// Only the chunked Qwen Q4_K route walks the prompt in bounded panels; every other
	// host route prefills the whole divergent suffix in one pass (prefillDivergentSuffix).
	panel := promptTokens
	if p.qwenQ4KPrefillChunkTarget() {
		if chunk := p.effectiveQwenQ4KPrefillChunkTokens(); chunk > 0 && chunk < panel {
			panel = chunk
		}
	}
	width := int64(3*p.m.Cfg.HiddenSize + 2*p.m.Cfg.IntermediateSize)
	if width <= 0 {
		return 0
	}
	return int64(panel) * width * 4
}

// admitHostMemory is the host-session capacity admission. It runs before any session,
// clone, or prefill allocation and returns the (possibly re-annotated) request context
// plus a release func the caller must run when the turn finishes. It is fail-open: an
// unarmed planner, a device-backed planner, or an unknown usage probe admits unchanged.
//
// Decision ladder under the armed ceiling (the reservation of every in-flight turn is
// subtracted, so concurrent coalesced turns cannot all admit against the same headroom):
//  1. session + retained copies fit: admit with prefix caching.
//  2. otherwise collect garbage (a divergent prefill churns gigabytes of short-lived panel
//     buffers) and re-probe; admit with caching if both now fit.
//  3. the session alone fits: admit, but mark the turn to skip prefix-cache admission.
//     The cache is NOT trimmed for optional copies, so the prefix this turn reuses stays.
//  4. the session itself does not fit: trim the prefix cache (halve, then empty) toward
//     the session, and admit without caching if it now fits.
//  5. nothing fits: refuse with a typed host-scope InKernelCapacityError, before any
//     allocation, instead of growing until the kernel kills the process.
func (p *InKernelPlanner) admitHostMemory(ctx context.Context, promptTokens, maxNew int) (context.Context, func(), error) {
	noop := func() {}
	if !p.hostMemoryAdmissionArmed() {
		return ctx, noop, nil
	}
	session, retained, plan := p.hostRequestDemand(promptTokens, maxNew)
	if session <= 0 {
		return ctx, noop, nil
	}
	p.hostBudgetMu.Lock()
	defer p.hostBudgetMu.Unlock()
	b := p.hostBudget
	if b == nil {
		return ctx, noop, nil
	}
	avail, known := b.available()
	if !known {
		return ctx, noop, nil
	}
	if session+retained > avail {
		runtime.GC()
		debug.FreeOSMemory()
		if avail, known = b.available(); !known {
			return ctx, noop, nil
		}
	}
	// A turn that only fits after evicting cached prefixes is served without caching:
	// admitting fresh copies right after the trim would re-grow the cache it just shed.
	trimmed := false
	if session > avail {
		trimmed = true
		p.trimPrefixCacheForHostPressure(func() bool {
			avail, known = b.available()
			return !known || session <= avail
		})
		if !known {
			return ctx, noop, nil
		}
	}
	want := session + retained
	switch {
	case want <= avail && !trimmed:
	case session <= avail:
		want = session
		ctx = context.WithValue(ctx, inKernelSkipPrefixAdmissionKey{}, true)
		log.Printf("inkernel_chat host-memory model=%s action=skip-prefix-admission session_bytes=%d retained_bytes=%d avail_bytes=%d ceiling_bytes=%d",
			p.modelID, session, retained, avail, b.ceiling)
	default:
		log.Printf("inkernel_chat host-memory model=%s action=decline session_bytes=%d avail_bytes=%d ceiling_bytes=%d reserved_bytes=%d",
			p.modelID, session, avail, b.ceiling, b.reserved)
		return ctx, noop, &InKernelCapacityError{
			Want:  session,
			Avail: max(avail, 0),
			Class: primaryDemandClass(plan, compute.MemoryScopeDevice),
			Scope: compute.MemoryScopeHost,
			Site:  "host-memory-precheck",
		}
	}
	b.reserved += want
	released := false
	return ctx, func() {
		p.hostBudgetMu.Lock()
		defer p.hostBudgetMu.Unlock()
		if released {
			return
		}
		released = true
		b.reserved -= want
		if b.reserved < 0 {
			b.reserved = 0
		}
	}, nil
}

// available is the ceiling minus live usage minus in-flight reservations. The caller
// holds hostBudgetMu. A probe that cannot report usage yields known=false (fail open).
func (b *hostMemoryBudget) available() (int64, bool) {
	used, ok := b.used()
	if !ok || used < 0 {
		return 0, false
	}
	return b.ceiling - used - b.reserved, true
}

// trimPrefixCacheForHostPressure evicts prefix-cache entries until fits reports true:
// the cache is halved up to inKernelHostReliefHalvings times and then emptied, returning
// freed pages to the OS after each step. Only unleased entries are evicted, so a prefix
// being served survives. The cache's configured token budget is restored afterwards, so
// relief never changes its steady-state bound. The caller holds hostBudgetMu (never p.mu).
func (p *InKernelPlanner) trimPrefixCacheForHostPressure(fits func() bool) {
	if p.tree == nil {
		return
	}
	p.mu.Lock()
	st := p.tree.Stats()
	p.mu.Unlock()
	target := st.Tokens
	for i := 0; i <= inKernelHostReliefHalvings && target > 0; i++ {
		target /= 2
		if i == inKernelHostReliefHalvings {
			target = 0
		}
		p.mu.Lock()
		p.tree.SetRetention(target)
		p.mu.Unlock()
		debug.FreeOSMemory()
		log.Printf("inkernel_chat host-memory model=%s action=trim-prefix-cache retain_tokens=%d", p.modelID, target)
		if fits() {
			break
		}
	}
	p.mu.Lock()
	if st.MaxTokens > 0 {
		p.tree.SetRetention(st.MaxTokens)
	} else {
		p.tree.SetRetention(-1)
	}
	p.mu.Unlock()
}
