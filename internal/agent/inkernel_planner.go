package agent

// inkernel_planner.go — the in-kernel chat Planner. When fak serve boots with a
// preloaded GGUF model (modelengine.Preload / PreloadQ4K) and a tokenizer, and no
// upstream --base-url, this planner drives BOTH /v1/chat/completions and
// /v1/messages (they share s.planner.Complete) from the model fused into the
// kernel — real ChatML chat through internal/tokenizer, not the byte-tokenized
// dispatch demo in modelengine.Complete.
//
// The decode recipe is the proven cmd/fakchat hybrid path: render ChatML → Encode
// → Session.Prefill → argmax/temperature sample → Session.Step → Decode, stopping
// on <|im_end|>/<|endoftext|>. fakchat's end-to-end coherent chat (Qwen2.5-1.5B/7B,
// FAK-NATIVE-CHAT-RESULTS.md) is the witness that this recipe produces real text;
// this file factors it into a Planner so the gateway can serve it on both wires.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// InKernelPlanner is an agent.Planner backed by the in-kernel model. One Complete
// call renders the transcript as ChatML, runs a real Prefill + decode over the
// kernel-owned session cache, and returns the assistant's text. It does not itself
// emit structured tool calls — the gateway's adjudication layer still runs on
// whatever the caller proposed.
type InKernelPlanner struct {
	m                 *model.Model
	tok               *tokenizer.Tokenizer
	modelID           string
	q4k               bool            // resident-Q4_K load: decode runs Session.Q4K (SDOT int8 GEMV)
	quant             bool            // Q8_0 decode/prefill path (the served default); tests flip it to exercise the proven f32 reuse path
	backend           compute.Backend // non-nil → decode runs through the device HAL (e.g. CUDA) instead of the CPU session
	metal             bool            // Apple-Silicon metalgemm GPU forward on the CPU session (s.Metal); engaged ONLY when backend==nil (the CPU-session seam). No-op on non-Metal builds.
	cpuOffloadExperts bool            // with a backend, keep MoE experts host-resident while dense/attention use the device
	// expertSpill is the resolved graded expert placement (`--n-cpu-moe`, #5612) this planner
	// installs on every session it builds: how many MoE layers spill to host and how many device
	// bytes the routed-expert ring may hold. nil — the default and every planner that was never
	// given a grade — leaves placement exactly as cpuOffloadExperts alone decided it. Resolved once
	// by SetExpertSpill (inkernel_expert_spill.go), never per request.
	expertSpill *model.ExpertSpillPlacement
	maxNew      int
	temp        float64
	seed        int64
	// decodeTraceNow is an injectable monotonic clock used only by explicitly
	// traced requests. nil selects time.Now; the default path never reads it.
	decodeTraceNow func() time.Time

	// qwenQ4KPrefillChunkTokens and its typed parse error are resolved once at
	// construction. The error is request-gated to the exact resident hybrid path,
	// so an unrelated model remains byte-for-byte on its historical forward.
	qwenQ4KPrefillChunkTokens    int
	qwenQ4KPrefillChunkConfigErr *model.InKernelQwenQ4KPrefillChunkConfigError
	qwen35MetalGDNSequence       bool
	q4kGateUpOutputSlab          bool
	qwen35MetalGDNExecuted       atomic.Bool

	// tree is the process-scoped RadixAttention prefix cache (internal/radixkv): the
	// multi-thousand-token static system+tool-schema prefix is prefilled once and the
	// next turn REUSES its KV, prefilling only the divergent suffix — the candidate-#13
	// win, bit-identical to a full recompute (proven in internal/model's KV-prefix-reuse
	// rung). nil disables reuse (every turn full-prefills, the pre-#13 behavior); the
	// device-HAL path (backend != nil) never reuses (the reuse clone is a CPU session).
	// mu guards every tree access — the gateway can drive Complete concurrently, and the
	// tree is shared mutable state (radixkv itself is deliberately lock-free).
	//
	// MEMORY NOTE: radixkv stores the FULL-prefix KV per node, so a long single growing
	// conversation accumulates nested KV clones (see radixkv's Tokens-vs-PrefixTokens
	// note). FAK_INKERNEL_RADIX_BUDGET sets the edge-token budget (0 = unbounded, the
	// default — the maximal-reuse regime the witnesses measure), and
	// FAK_NATIVE_KV_VICTIM_RULE=cost-aware selects the KVBM victim rule for budget pressure.
	// Operators serving long sessions should set a budget; bounding the deep-chain
	// footprint is tracked.
	mu         sync.Mutex
	tree       *radixkv.Tree
	scopedTree *radixkv.ScopedTree

	// devMu serializes the WHOLE device forward pass (Prefill + the decode loop) when a
	// backend or native Metal is wired. These accelerators have one shared command stream and
	// model-level buffers. The CUDA backend's Go-side cudaMu makes
	// each INDIVIDUAL op atomic — but NOT a whole multi-op forward. Two Complete calls driven
	// concurrently by the gateway would interleave their per-token op sequences on that shared
	// device state and stomp each other's activation/KV buffers, faulting the kernel with an
	// illegal memory access that then poisons the CUDA context for every later request until a
	// process restart (observed live on an L4: a 2-way concurrent burst took the GPU serve down
	// with thousands of sticky cuda_kernels.cu illegal-access errors). The plain CPU path is
	// already session-local per turn and guards only its shared tree with p.mu, so devMu leaves
	// it untouched.
	// This serializes concurrent device requests into safe queuing — correct for a single-stream
	// device — instead of crashing; batched multi-user device decode is the separate throughput
	// follow-up (internal/model/batch.go), not a correctness fix.
	devMu sync.Mutex

	coalesceMu         sync.Mutex
	coalesceReady      []*inKernelCoalesceRequest
	coalesceRunning    bool
	coalesceReadyHook  func()
	coalesceBatchHook  func(int)
	coalesceSharedHook func(panels int, macs int64)
	coalesceCohortID   atomic.Uint64

	reqMemMu      sync.Mutex
	lastReqMemory RequestMemoryStats

	// moeResidencyState is the serve-scoped fold of every request's activated-expert residency
	// (R6/#5617, inkernel_moe_residency.go). It is embedded because the ring lives on a session
	// this planner builds and closes PER REQUEST, so without a planner-scoped ledger the whole
	// ladder's accounting is destroyed at each teardown and no serve surface can see it.
	moeResidencyState

	// inKernelTurnTaxState is the per-turn cache-decision ledger (#1538, inkernel_turntax.go),
	// embedded for the same reason as moeResidencyState directly above: the decision is taken
	// inside a session this planner builds and closes PER REQUEST, so planner-scoped storage is
	// the only place it survives that teardown. Its zero value is a usable empty ledger, so
	// every constructor — including a bare &InKernelPlanner{…} — records from its first turn.
	inKernelTurnTaxState

	oomRetryMu sync.Mutex
	oomRetry   map[string]*inKernelOOMRetryClassStats

	pressureTrimMu sync.Mutex
	pressureTrim   map[requestPressureTrimKey]*requestPressureTrimStats

	// kvSpanEvict gates the model-side KV-quarantine eviction BRIDGE (internal/kvmmu)
	// on the live serve path (issue #579). When on, a tool-result QUARANTINE drives a
	// real model.KVCache.Evict of the result's K/V span over a fresh model.Session built
	// from the loaded model — the bit-exact re-RoPE + renumber the kvmmu witnesses prove,
	// now fired by a live request instead of only a synthetic-model unit test. DEFAULT OFF
	// (FAK_INKERNEL_KVMMU=on opts in); off it is an inert no-op, so the served path is
	// byte-for-byte the pre-bridge behavior. It is independent of and additive to the
	// radixkv prefix-cache eviction above — that drops a reusable PREFIX node; this evicts
	// the per-session SPAN and is the model-independent KV-MMU floor.
	kvSpanEvict bool

	// batchDecode routes a request's decode through the continuous-batch
	// BatchSession.StepBatchActive machinery (as a batch of one on the resident chat serve)
	// instead of the serial Session.Step loop. It is the OPT-IN wiring seam
	// (FAK_INKERNEL_BATCH=on) that lets a future cross-request coalescer co-batch concurrent
	// chat turns onto the shared weight stream. DEFAULT OFF: unset, the decode path is the
	// byte-identical pre-seam serial loop, and even ON at B=1 StepBatchActive is exactly
	// Seqs[0].Step, so the served tokens are unchanged — the batched glm_moe_dsa GEMM that
	// yields throughput is the separate, box-gated lever, not this flag.
	batchDecode bool

	// kvPrefixEverAdmitted is the #3391 first-prefill latch: false until a reuse-enabled
	// turn has admitted its prompt into the radix tree (generateReused step 3). While
	// false, a turn's prompt tokens are booked as INELIGIBLE in the cacheobs hit-rate
	// denominator — nothing has been admitted, so nothing could match: the always-cold
	// first prefill the raw ratio unfairly counts against the cache. Deliberately one-way
	// and planner-scoped (matching the tree's scope): a tree later evicted back to empty
	// keeps counting prompts as eligible, and on the multi-tenant scoped tree a tenant's
	// own cold first prefill after another tenant warmed the latch counts as eligible too
	// — both over-count the denominator, which can only UNDER-state the filtered ratio,
	// never inflate it (the same honest-conservative direction as cacheobs's clamps).
	kvPrefixEverAdmitted atomic.Bool
}

type inKernelOOMRetryClassStats struct {
	attempts        uint64
	successes       uint64
	failures        uint64
	lastFailedBytes uint64
	lastSite        string
}

type requestPressureTrimKey struct {
	scope  string
	class  string
	reason string
}

type requestPressureTrimStats struct {
	attempts        uint64
	trimmed         uint64
	noHooks         uint64
	resolved        uint64
	lastWantBytes   uint64
	lastBudgetBytes uint64
	lastMarginBytes int64
}

// Model reports the model id (for /v1/models provenance + the planner seam).
func (p *InKernelPlanner) Model() string { return p.modelID }

// NativeDecodeTraceSupported declares that this planner owns the token-commit
// seam used by NativeDecodeTrace. It performs no model work.
func (p *InKernelPlanner) NativeDecodeTraceSupported() bool { return true }

// StreamingSupported enables the gateway's semantic SSE path for in-kernel runs.
// The backend projects each completed turn as one content delta; tool lifecycle
// progress still arrives independently from the owned loop.
func (p *InKernelPlanner) StreamingSupported() bool { return true }

// CompleteStream emits assistant prose while leaving tool calls buffered for adjudication.
func (p *InKernelPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	completion, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	if sink != nil && completion.Message.Content != "" {
		if err := sink(completion.Message.Content); err != nil {
			return nil, err
		}
	}
	return completion, nil
}

// KVMemoryStats reports the in-process KV prefix cache's physical resident shape.
// Native backend snapshots are split into hot device bytes, hot host metadata, and
// the independently owned host-DRAM L2. Proxy/provider counters never enter here.
func (p *InKernelPlanner) KVMemoryStats() KVMemoryStats {
	if p == nil || p.m == nil {
		return KVMemoryStats{
			MemoryClass: string(compute.MemoryKVCache),
			Scope:       string(compute.MemoryScopeHost),
			DType:       compute.F32.String(),
		}
	}
	kvCfg := compute.KVConfig{
		NumLayers:  p.m.Cfg.NumLayers,
		NumKVHeads: p.m.Cfg.NumKVHeads,
		HeadDim:    p.m.Cfg.HeadDim,
		RopeTheta:  p.m.Cfg.RopeTheta,
	}
	bytesPerToken := compute.EstimateKVStoreBytes(kvCfg, 1)
	stats := KVMemoryStats{
		Enabled:       p.tree != nil,
		Backend:       "radixkv",
		MemoryClass:   string(compute.MemoryKVCache),
		Scope:         string(compute.MemoryScopeHost),
		DType:         compute.F32.String(),
		BytesPerToken: bytesPerToken,
		HeadroomRatio: inKernelKVMemoryHeadroom,
	}
	if p.backend != nil && p.tree == nil {
		stats.Enabled = false
		stats.Backend = p.backend.Name()
		stats.Scope = string(compute.MemoryScopeDevice)
		total, free, known := compute.DeviceMemoryInfo(p.backend)
		applyKVMemoryCapacity(&stats, total, free, known)
		return stats
	}
	hostTotal, hostFree, hostKnown := compute.HostSystemMemoryInfo()
	if p.tree == nil {
		applyKVMemoryCapacity(&stats, hostTotal, hostFree, hostKnown)
		return stats
	}
	p.mu.Lock()
	st := p.tree.Stats()
	p.mu.Unlock()
	if p.backend != nil && p.backend.Caps().DeviceMemory {
		stats.Backend = p.backend.Name()
		stats.Scope = string(compute.MemoryScopeDevice)
		stats.ResidentTokens = st.DeviceSnapshotTokens
		stats.ResidentBytes = st.DeviceSnapshotBytes
		total, free, known := compute.DeviceMemoryInfo(p.backend)
		applyKVMemoryCapacity(&stats, total, free, known)
	} else {
		stats.ResidentTokens = st.PrefixTokens
		stats.ResidentBytes = compute.EstimateKVStoreBytes(kvCfg, st.PrefixTokens)
		applyKVMemoryCapacity(&stats, hostTotal, hostFree, hostKnown)
	}
	stats.BudgetTokens = st.MaxTokens
	stats.LRUTokens = st.Tokens
	stats.MaxDepthTokens = st.MaxDepthTokens
	stats.Nodes = st.Nodes
	stats.Leaves = st.Leaves
	stats.Evictions = st.Evictions
	stats.PolicyEvictions = st.PolicyEvictions
	stats.Splits = st.Splits
	stats.L1DeviceResidentBytes = st.DeviceSnapshotBytes
	stats.L1HostResidentBytes = st.DeviceSnapshotHostBytes
	stats.L2HostResidentBytes = st.HostSnapshotBytes
	stats.L2HostCapacityBytes = st.MaxHostSnapshotBytes
	stats.L1Hits = st.L1Hits
	stats.L1Misses = st.L1Misses
	stats.L1Faults = st.L1Faults
	stats.L1HitTokens = st.L1HitTokens
	stats.L2Hits = st.L2Hits
	stats.L2Misses = st.L2Misses
	stats.L2Faults = st.L2Faults
	stats.L2HitTokens = st.L2HitTokens
	stats.L2StageBytes = st.L2StageBytes
	stats.L2RestoreBytes = st.L2RestoreBytes
	stats.L2Evictions = st.L2Evictions
	stats.L3Enabled = st.L3Enabled
	stats.L3ReferencedBytes = st.L3ReferencedBytes
	stats.L3Hits = st.L3Hits
	stats.L3Misses = st.L3Misses
	stats.L3Faults = st.L3Faults
	stats.L3HitTokens = st.L3HitTokens
	stats.L3StageBytes = st.L3StageBytes
	stats.L3RestoreBytes = st.L3RestoreBytes
	stats.L3StageNanos = st.L3StageNanos
	stats.L3RestoreNanos = st.L3RestoreNanos
	stats.L3StageFaults = st.L3StageFaults
	stats.L3RestoreFaults = st.L3RestoreFaults
	return stats
}

const inKernelKVMemoryHeadroom = 0.15

func applyKVMemoryCapacity(stats *KVMemoryStats, total, free int64, known bool) {
	if stats == nil || !known || total <= 0 {
		return
	}
	stats.CapacityKnown = true
	stats.CapacityTotalBytes = total
	if free != compute.FreeUnknown && free >= 0 {
		stats.CapacityFreeKnown = true
		stats.CapacityFreeBytes = free
	}
	budgetBase := total
	if stats.CapacityFreeKnown {
		budgetBase = SaturatingAddBytes(free, stats.ResidentBytes)
		if budgetBase > total {
			budgetBase = total
		}
	}
	stats.FitBudgetBytes = ApplyByteHeadroom(budgetBase, stats.HeadroomRatio)
	stats.FitMarginBytes = stats.FitBudgetBytes - stats.ResidentBytes
}

// SaturatingAddBytes adds two byte counts without ever wrapping: a non-positive b is a
// no-op, and a sum that would overflow int64 pins at maxInt64 instead. Device capacity
// figures arrive from several backends and a wrapped negative total would read as "no
// memory" and refuse a request that fits. Exported because the gateway's request-memory
// fit view (internal/gateway/memory_fit.go) totals the SAME quantities this planner
// reports and must saturate them identically, or the two views disagree at the ceiling.
func SaturatingAddBytes(a, b int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	if b <= 0 {
		return a
	}
	if a > maxInt64-b {
		return maxInt64
	}
	return a + b
}

// ApplyByteHeadroom reserves a fraction of a byte budget: it returns bytes scaled down by
// headroom, treating a non-positive budget as zero and an out-of-range headroom (<=0 or
// >=1) as "reserve nothing" rather than as a clamp — a 0 budget must stay 0, and a bogus
// ratio must never silently zero a real budget. Exported for the same reason as
// SaturatingAddBytes: the gateway renders the headroom-adjusted view of these budgets.
func ApplyByteHeadroom(bytes int64, headroom float64) int64 {
	if bytes <= 0 {
		return 0
	}
	if headroom <= 0 || headroom >= 1 {
		return bytes
	}
	return int64(float64(bytes) * (1 - headroom))
}

func (p *InKernelPlanner) RequestMemoryStats() RequestMemoryStats {
	if p == nil {
		return RequestMemoryStats{}
	}
	p.reqMemMu.Lock()
	defer p.reqMemMu.Unlock()
	out := p.lastReqMemory
	out.MemoryPlan = append([]RequestMemoryDemand(nil), p.lastReqMemory.MemoryPlan...)
	out.Capacities = append([]RequestMemoryCapacity(nil), p.lastReqMemory.Capacities...)
	return out
}

func (p *InKernelPlanner) InKernelOOMRetryStats() InKernelOOMRetryStats {
	if p == nil {
		return InKernelOOMRetryStats{}
	}
	backend := "unknown"
	if p.backend != nil {
		backend = p.backend.Name()
	}
	p.oomRetryMu.Lock()
	defer p.oomRetryMu.Unlock()
	out := InKernelOOMRetryStats{Backend: backend, Rows: make([]InKernelOOMRetryClassStats, 0, len(p.oomRetry))}
	for class, st := range p.oomRetry {
		if st == nil {
			continue
		}
		out.Rows = append(out.Rows, InKernelOOMRetryClassStats{
			Class:           class,
			Attempts:        st.attempts,
			Successes:       st.successes,
			Failures:        st.failures,
			LastFailedBytes: st.lastFailedBytes,
			LastSite:        st.lastSite,
		})
	}
	sort.SliceStable(out.Rows, func(i, j int) bool { return out.Rows[i].Class < out.Rows[j].Class })
	return out
}

func (p *InKernelPlanner) InKernelMemoryPressureTrimStats() InKernelMemoryPressureTrimStats {
	if p == nil {
		return InKernelMemoryPressureTrimStats{}
	}
	backend := "unknown"
	if p.backend != nil {
		backend = p.backend.Name()
	}
	p.pressureTrimMu.Lock()
	defer p.pressureTrimMu.Unlock()
	out := InKernelMemoryPressureTrimStats{Backend: backend, Rows: make([]InKernelMemoryPressureTrimClassStats, 0, len(p.pressureTrim))}
	for key, st := range p.pressureTrim {
		if st == nil {
			continue
		}
		out.Rows = append(out.Rows, InKernelMemoryPressureTrimClassStats{
			Scope:           key.scope,
			Class:           key.class,
			Reason:          key.reason,
			Attempts:        st.attempts,
			Trimmed:         st.trimmed,
			NoHooks:         st.noHooks,
			Resolved:        st.resolved,
			LastWantBytes:   st.lastWantBytes,
			LastBudgetBytes: st.lastBudgetBytes,
			LastMarginBytes: st.lastMarginBytes,
		})
	}
	sort.SliceStable(out.Rows, func(i, j int) bool {
		a, b := out.Rows[i], out.Rows[j]
		if a.Scope != b.Scope {
			return a.Scope < b.Scope
		}
		if a.Class != b.Class {
			return a.Class < b.Class
		}
		return a.Reason < b.Reason
	})
	return out
}

// Complete renders the transcript as ChatML and runs one in-kernel decode turn,
// returning the generated assistant text. Mirrors cmd/fakchat's hybrid path. The
// per-request SampleOpts override this planner's configured decode length,
// temperature, TopP (nucleus cutoff), and TopK (top-k cutoff) for THIS turn, and a
// per-request Stop sequence ends the turn early (string-suffix stop, orthogonal to
// the token-ID <|im_end|>/EOS stops). All five per-request sampling controls the
// HTTP wires forward are now honored on the in-kernel path too.
// InKernelOOMError is the agent-level, recovered form of an in-kernel device allocation
// failure (a *compute.DeviceAllocError that unwound out of a device decode path). It is
// in-kernel BY CONSTRUCTION — only the in-kernel planner / compute backend can produce it,
// never a real upstream — so the gateway can safely render a specific, actionable client
// message for it (an over-large prompt on a small GPU) without any risk of leaking upstream
// content. Bytes is the device allocation that failed; Class and Site preserve the allocator
// category for operator visibility without exposing model/provider content.
type InKernelOOMError struct {
	Bytes int
	Class compute.MemoryClass
	Site  string
}

func (e *InKernelOOMError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	if class == compute.MemoryUnknown {
		return fmt.Sprintf("in-kernel GPU out of memory (device allocation of %d bytes failed)", e.Bytes)
	}
	return fmt.Sprintf("in-kernel GPU out of memory (%s allocation of %d bytes failed)", class, e.Bytes)
}

// InKernelCapacityError is the request-time companion to InKernelOOMError: a backend
// with known capacity can refuse the planned in-kernel request memory before the device
// allocator is touched. It is still a local OOM-class resource exhaustion, but it is
// earlier and more actionable than a recovered DeviceAllocError.
type InKernelCapacityError struct {
	Want  int64
	Avail int64
	Class compute.MemoryClass
	Scope compute.MemoryScope
	Site  string
}

func (e *InKernelCapacityError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	scope := e.Scope
	if scope == "" {
		scope = compute.MemoryScopeDevice
	}
	return fmt.Sprintf("in-kernel GPU capacity precheck refused request (%s %s plan needs %d bytes, available budget is %d bytes)", scope, class, e.Want, e.Avail)
}

// recoverDevicePanic is the body of Complete's deferred recover, factored out so it is
// unit-testable without a GPU (the panic payload is an ordinary Go value). It converts a
// recovered in-kernel device-allocation panic into a typed, actionable error and reports
// handled=true; for ANY other recovered value it reports handled=false so the caller
// re-panics — the recover stays surgical and never swallows a genuine bug (a nil deref, a
// validation panic, a poisoned-context launch failure).
func recoverDevicePanic(r any) (err error, handled bool) {
	var dae *compute.DeviceAllocError
	if e, ok := r.(error); ok && errors.As(e, &dae) {
		return &InKernelOOMError{Bytes: dae.Bytes, Class: dae.DemandClass(), Site: dae.Site}, true
	}
	return nil, false
}

type inKernelGenerateResult struct {
	gen, promptTok, matched int
	// cacheable is the lookup-side prefix match (#3390): tokens the radix index matched
	// BEFORE servability (nil KV, exact-hit refeed, unsupported truncate) could trim the
	// realized `matched` below it. cacheable >= matched always; 0 when reuse is off.
	cacheable         int
	sourceTier        radixkv.SnapshotTier
	prefillS, decodeS float64
	stopped           bool
	batchReceipt      InKernelBatchReceipt
}

func (p *InKernelPlanner) generateReusedRecovering(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, measurementOpt ...*nativeInferenceMeasurement) (res inKernelGenerateResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	gen, promptTok, cacheable, matched, sourceTier, prefillS, decodeS, stopped, err := p.generateReusedContextWithBias(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	if err != nil {
		return inKernelGenerateResult{}, err
	}
	return inKernelGenerateResult{
		gen:        gen,
		promptTok:  promptTok,
		cacheable:  cacheable,
		matched:    matched,
		sourceTier: sourceTier,
		prefillS:   prefillS,
		decodeS:    decodeS,
		stopped:    stopped,
	}, nil
}

func (p *InKernelPlanner) generateReusedWithOOMRetry(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, onRetry func(), measurementOpt ...*nativeInferenceMeasurement) (inKernelGenerateResult, error) {
	res, err := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	if err == nil {
		return res, nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return inKernelGenerateResult{}, ctxErr
	}
	if !p.prepareDeviceOOMRetry(err) {
		return inKernelGenerateResult{}, err
	}
	if onRetry != nil {
		onRetry()
	}
	retryRes, retryErr := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	p.recordInKernelOOMRetry(err, retryErr == nil)
	return retryRes, retryErr
}

func (p *InKernelPlanner) prepareDeviceOOMRetry(err error) bool {
	if p == nil || p.backend == nil {
		return false
	}
	var oom *InKernelOOMError
	if !errors.As(err, &oom) {
		return false
	}
	released := p.trimBackendIdlePools()
	if released {
		log.Printf("inkernel_chat oom-retry model=%s backend=%s class=%s site=%s bytes=%d action=trim-idle-pools",
			p.modelID, p.backend.Name(), oom.Class, oom.Site, oom.Bytes)
	}
	return released
}

func (p *InKernelPlanner) trimBackendIdlePools() bool {
	if p == nil || p.backend == nil {
		return false
	}
	released := false
	if r, ok := p.backend.(interface{ Recycle() }); ok {
		r.Recycle()
		released = true
	}
	if t, ok := p.backend.(interface{ Trim() }); ok {
		t.Trim()
		released = true
	}
	if t, ok := p.backend.(interface{ TrimLarge(int) }); ok {
		t.TrimLarge(0)
		released = true
	}
	return released
}

func (p *InKernelPlanner) recordInKernelOOMRetry(trigger error, success bool) {
	if p == nil {
		return
	}
	class, bytes, site := inKernelOOMRetryTrigger(trigger)
	p.oomRetryMu.Lock()
	if p.oomRetry == nil {
		p.oomRetry = map[string]*inKernelOOMRetryClassStats{}
	}
	st := p.oomRetry[class]
	if st == nil {
		st = &inKernelOOMRetryClassStats{}
		p.oomRetry[class] = st
	}
	st.attempts++
	if success {
		st.successes++
	} else {
		st.failures++
	}
	st.lastFailedBytes = bytes
	st.lastSite = site
	p.oomRetryMu.Unlock()
}

func inKernelOOMRetryTrigger(err error) (class string, bytes uint64, site string) {
	var oom *InKernelOOMError
	if errors.As(err, &oom) {
		if oom.Bytes > 0 {
			bytes = uint64(oom.Bytes)
		}
		class = strings.TrimSpace(string(oom.Class))
		site = strings.TrimSpace(oom.Site)
	}
	if class == "" {
		class = string(compute.MemoryUnknown)
	}
	return class, bytes, site
}

func (p *InKernelPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (comp *Completion, err error) {
	// An in-kernel device-allocation failure (e.g. OOM on a small GPU under a large Claude
	// Code system prompt) panics deep below a CGO boundary with no error channel. Recover it
	// HERE — the narrowest Go frame that wraps the whole device decode (generateReused's
	// Prefill/Step + NewBackendSession's NewKV) AND returns the error the gateway already maps
	// to a client response — converting it into a typed error instead of crashing the serving
	// goroutine. Everything else re-panics, preserving today's crash/stack behavior for bugs.
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				comp, err = nil, e
				return
			}
			panic(r)
		}
	}()
	if p.qwenQ4KPrefillChunkTarget() && p.qwenQ4KPrefillChunkConfigErr != nil {
		return nil, p.qwenQ4KPrefillChunkConfigErr
	}
	sp := applySampleOpts(opts...)
	if sp.NativeDecodeTokenIDs && !sp.DecodeTrace {
		return nil, fmt.Errorf("native decode token IDs require a decode trace")
	}
	var requestStarted time.Time
	if sp.NativeInferenceReceipt {
		if _, _, err := p.nativeSelectionIdentity(); err != nil {
			return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: err.Error()}
		}
		requestStarted = time.Now()
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	temp := p.temp
	if sp.Temperature != nil {
		temp = *sp.Temperature
	}
	// Per-request nucleus cutoff; 0 (the zero value) disables truncation so an omitted
	// top_p keeps the full softmax draw, identical to the pre-seam path.
	topP := 0.0
	if sp.TopP != nil {
		topP = *sp.TopP
	}
	// Per-request top-k; 0 (the zero value, and any value <=0) disables truncation so
	// an omitted top_k keeps the full distribution, identical to the pre-seam path.
	topK := 0
	if sp.TopK != nil {
		topK = *sp.TopK
	}
	var logitBias model.LogitBias
	if len(sp.LogitBias) > 0 {
		logitBias = model.LogitBias(sp.LogitBias)
	}
	// Per-request OpenAI repetition penalties; nil/omitted stays 0, which
	// sampleLogitsWithPenalty treats as a byte-for-byte no-op (the pre-#1705 path).
	var freqPenalty, presPenalty float64
	if sp.FrequencyPenalty != nil {
		freqPenalty = *sp.FrequencyPenalty
	}
	if sp.PresencePenalty != nil {
		presPenalty = *sp.PresencePenalty
	}
	if sp.NativeInferenceReceipt && (temp != 0 || topP != 0 || topK > 0 || len(logitBias) > 0 || freqPenalty != 0 || presPenalty != 0) {
		return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: "requires greedy sampling over unmodified logits"}
	}

	chat := renderInKernelChatMLRequest(messages, tools, p.m.Cfg, sp.ResponseFormat, sp.ToolChoice)
	ids, err := p.tok.Encode(chat)
	if err != nil {
		return nil, err
	}
	stops := StopIDs(p.tok, p.m.Cfg)

	// emit runs per generated token: decode the piece, accumulate the text, and apply the
	// per-request string-suffix Stop (orthogonal to the token-ID stops). Returning true
	// ends the turn with the token counted and its text trimmed (the stop string is not
	// echoed back, matching the HTTP wires). Factoring decode into this closure keeps the
	// token-level reuse/decode core (generateReused) tokenizer-free, so the candidate-#13
	// reuse and #14 eviction are witnessable on a synthetic model with no tokenizer fixture.
	var sb strings.Builder
	emit := func(next int) bool {
		if piece, derr := p.tok.Decode([]int{next}); derr == nil {
			sb.WriteString(piece)
		}
		if trimmed, hit := checkStop(sb.String(), sp.Stop); hit {
			sb.Reset()
			sb.WriteString(trimmed)
			return true
		}
		return false
	}

	// Serialize the entire device forward pass: a single-stream accelerator cannot run two
	// forwards at once without the concurrent op-streams corrupting shared device buffers
	// (see devMu). The plain CPU path owns a per-turn session and guards the shared radix tree
	// with p.mu itself, so it remains concurrent. Held across Prefill + decode.
	if p.requiresDeviceSerialization() && !p.coalescesQwenDecode() {
		p.devMu.Lock()
		defer p.devMu.Unlock()
		if err := p.refuseOversizeRequest(len(ids), maxNew); err != nil {
			return nil, err
		}
	}
	if p.coalescesQwenDecode() {
		if err := p.refuseOversizeRequest(len(ids), maxNew); err != nil {
			return nil, err
		}
	}
	var measurement *nativeInferenceMeasurement
	if sp.NativeInferenceReceipt || sp.DecodeTrace || sp.NativeDecodeTokenIDs {
		measurement = &nativeInferenceMeasurement{
			startedAt:             requestStarted,
			inferenceDisabled:     !sp.NativeInferenceReceipt,
			decodeTokenIDsEnabled: sp.NativeDecodeTokenIDs,
		}
		if sp.NativeDecodeTokenIDs {
			measurement.decodeTokenIDs = make([]int, 0, maxNew)
		}
		if sp.NativeInferenceReceipt {
			measurement.cudaImmutableWeightUploadsBefore, measurement.cudaImmutableWeightUploadsAvailable = cudaImmutableWeightUploadSnapshot(p.backend)
		}
		if sp.DecodeTrace {
			measurement.traceNow = p.decodeTraceNow
			if measurement.traceNow == nil {
				measurement.traceNow = time.Now
			}
		}
	}
	generate := func(runCtx context.Context) (inKernelGenerateResult, error) {
		return p.generateReusedWithOOMRetry(runCtx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, func() {
			sb.Reset()
			measurement.reset()
		}, measurement)
	}
	var genRes inKernelGenerateResult
	if p.coalescesQwenDecode() {
		genRes, err = p.runCoalescedGenerate(ctx, generate)
	} else {
		genRes, err = generate(ctx)
	}
	if err != nil {
		return nil, err
	}
	gen, promptTok, matched, prefillS, decodeS, stopped := genRes.gen, genRes.promptTok, genRes.matched, genRes.prefillS, genRes.decodeS, genRes.stopped
	// finishReason is honest about WHY decode ended: "stop" when a token-ID stop or a
	// per-request Stop sequence fired, "length" when maxNew was the only limit hit.
	finishReason := "length"
	if stopped {
		finishReason = "stop"
	}

	// Witness line (mirrors cmd/fakchat): real per-turn prefill/decode tok/s through the
	// in-kernel model, now also reporting the RadixAttention prefix reuse (reused vs
	// prompt) so a served chat turn self-reports the candidate-#13 win. prefill tok/s is
	// over the COMPUTED suffix (prompt minus the reused prefix) — the work actually done.
	computed := promptTok - matched
	prefTPS, decTPS := 0.0, 0.0
	if prefillS > 0 {
		prefTPS = float64(computed) / prefillS
	}
	if decodeS > 0 {
		decTPS = float64(gen) / decodeS
	}
	// #3176 Q1/Q2 decode witness: report the resolved Q8 SIMD kernel tier (+ whether the fused
	// fast decode GEMV engaged) and the effective decode-worker count, so an operator can SEE —
	// without wall-clock guessing — that the AVX2/AVX-512 lane fired (not the reference path) and
	// that decode parallelizes across a modest, capped stream count rather than either 1 core or
	// an oversubscribed all-core dispatch (the pathology 40e0afd fixed on many-core amd64).
	q8kern, q8fused := model.Q8DecodeKernel()
	q8fusedMark := ""
	if q8fused {
		q8fusedMark = "+fused"
	}
	p.logExecutionSummary(q8kern, q8fusedMark, promptTok, genRes.cacheable, matched, computed, prefillS, prefTPS, gen, decodeS, decTPS)
	// Feed the process-global KV-prefix reuse tap so this turn's split hit-rate reaches
	// /metrics, not just this log line — the live measurement of the frozen-trajectory
	// cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md). #3390: BOTH halves —
	// the lookup-side index match (cacheability) and the realized serve (matched/prompt) —
	// so the gap lost to eviction/admission is observable, not folded into "miss".
	// #3391 adds (a) the eligibility-filtered denominator — sampled BEFORE this turn is
	// latched as admitted, so the always-cold first prefill books zero eligible tokens
	// instead of depressing the fair hit-rate — and (b) the (model, tenant) attribution,
	// with the tenant drawn from the SAME authenticated prefix-cache identity the scoped
	// tree isolates on (unscoped traffic normalizes to "unknown" inside the tap).
	eligibleTok := p.kvPrefixEligiblePromptTokens(promptTok)
	p.noteKVPrefixAdmitted()
	tenant := ""
	if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped {
		tenant = owner.Tenant
	}
	cacheobs.Default.ObserveLabeled(cacheobs.Labels{Model: p.modelID, Tenant: tenant},
		promptTok, genRes.cacheable, matched, eligibleTok)
	// #3896 provenance axis: remote L3 matches are external transfers; L1/L2
	// matches stay local, and the unmatched suffix is local compute.
	localHit, externalHit := matched, 0
	if genRes.sourceTier == radixkv.SnapshotTierRemoteL3 {
		localHit, externalHit = 0, matched
	}
	cacheobs.Default.ObserveBySource(cacheobs.SourceLocalHit, localHit)
	cacheobs.Default.ObserveBySource(cacheobs.SourceExternalTransfer, externalHit)
	cacheobs.Default.ObserveBySource(cacheobs.SourceLocalCompute, promptTok-matched)
	compReuseEntry := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "fak-inkernel", ModelID: p.modelID, PromptTokens: int64(promptTok), CachedTokens: int64(matched)})

	// Split a Qwen3.5 reasoning block off the decoded text BEFORE it becomes Content
	// (and before the tool-call lift below reads it). A reasoning model (Ornith) opens
	// the turn with <think>…</think> then the final answer; renderChatMLTools does NOT
	// pre-seed the open tag, so the model emits both. splitReasoning is the in-kernel
	// equivalent of vLLM's --reasoning-parser qwen3: the reasoning lands in
	// ReasoningContent and only the post-</think> answer flows into Content (and thus
	// into Claude Code's context). It is gated — a non-reasoning turn (no think tags)
	// returns the decoded text untouched, so this is byte-identical to today for any
	// model that does not emit <think>.
	reasoning, content := splitReasoning(sb.String())
	comp = &Completion{
		Message:       Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
		FinishReason:  finishReason,
		ProviderCache: &compReuseEntry,
		Usage:         Usage{PromptTokens: promptTok, CompletionTokens: gen, TotalTokens: promptTok + gen, PromptTokensDetails: &UsageTokenDetails{CachedTokens: matched}},
	}
	if sp.NativeInferenceReceipt {
		comp.NativeInference = p.buildNativeInferenceReceipt(measurement, prefillS, decodeS)
	}
	if genRes.batchReceipt.CohortID != 0 {
		receipt := genRes.batchReceipt
		comp.InKernelBatch = &receipt
	}
	if sp.DecodeTrace {
		events := make([]NativeDecodeTraceEvent, len(measurement.traceEvents))
		copy(events, measurement.traceEvents)
		comp.DecodeTrace = &NativeDecodeTrace{
			Schema: NativeDecodeTraceSchema,
			Engine: NativeDecodeTraceEngine,
			Events: events,
		}
	}
	if sp.NativeDecodeTokenIDs {
		comp.NativeDecodeTokenIDs = &NativeDecodeTokenIDs{
			Schema:   NativeDecodeTokenIDsSchema,
			Engine:   NativeDecodeTokenIDsEngine,
			TokenIDs: append([]int(nil), measurement.decodeTokenIDs...),
		}
	}
	// Lift the model's text-form <tool_call> emissions into structured Message.ToolCalls
	// (Hermes dialect == Qwen2.5 native), set FinishReason="tool_calls", and flag a
	// claimed-but-unparseable call — the SAME normalization every proxy adapter runs, so
	// the in-kernel forward becomes a first-class tool-calling planner. Without this the
	// gateway adjudicates nothing (it reads Message.ToolCalls) and the Anthropic wire never
	// emits a tool_use block, so Claude Code's agent loop has nothing to execute.
	comp = normalizeCompletionToolCalls(comp)
	// A length finish is a conformance failure only when the caller actually
	// forced a named tool. Calling enforceForcedToolChoice for an omitted/auto
	// choice marks every max_tokens completion as dropped before it even resolves
	// the effective tool name, which would make an exact T64 receipt unreachable.
	if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
		comp = enforceForcedToolChoice(comp, sp.ToolChoice, tools, messages)
	}
	// Fail closed on a TRUNCATED tool call: the in-kernel finishReason is "stop"/"length"
	// (never "tool_calls"), so normalizeCompletionToolCalls cannot infer a drop from the
	// finish reason. If decode emitted an unclosed <tool_call> opener that the lift could
	// not recover, mark ToolCallsDropped so the conformance gate refuses the turn rather
	// than silently leaking a half-formed call into Claude Code's context.
	if len(comp.Message.ToolCalls) == 0 && strings.Contains(comp.Message.Content, "<tool_call>") {
		comp.ToolCallsDropped = true
	}
	return comp, nil
}

func (p *InKernelPlanner) buildNativeInferenceReceipt(measurement *nativeInferenceMeasurement, prefillS, decodeS float64) *model.NativeInferenceReceipt {
	backend, forwardPath := p.executionIdentity()
	nativeSelection, nativeSelectionDigest, _ := p.nativeSelectionIdentity()
	var qwen35MetalForwardSequence *model.Qwen35MetalForwardSequenceReceipt
	if measurement.qwen35MetalForwardSequence.EvidenceState != "" || measurement.qwen35MetalForwardSequence.Available {
		snapshot := measurement.qwen35MetalForwardSequence
		qwen35MetalForwardSequence = &snapshot
	}
	var qwen35MetalStateIdentity *model.Qwen35MetalStateIdentityReceipt
	if measurement.qwen35MetalStateIdentity != nil && measurement.qwen35MetalStateIdentity.Available {
		qwen35MetalStateIdentity = cloneQwen35MetalStateIdentityReceipt(*measurement.qwen35MetalStateIdentity)
	}
	var cudaImmutableWeightUploads *model.NativeCUDAImmutableWeightUploadDelta
	if after, ok := cudaImmutableWeightUploadSnapshot(p.backend); ok && measurement.cudaImmutableWeightUploadsAvailable {
		before := measurement.cudaImmutableWeightUploadsBefore
		if after.Calls >= before.Calls && after.TransferBytes >= before.TransferBytes && after.ResidentBytes >= before.ResidentBytes {
			cudaImmutableWeightUploads = &model.NativeCUDAImmutableWeightUploadDelta{
				Before: before,
				After:  after,
				Delta: model.NativeCUDAImmutableWeightUploadCounters{
					Calls:         after.Calls - before.Calls,
					TransferBytes: after.TransferBytes - before.TransferBytes,
					ResidentBytes: after.ResidentBytes - before.ResidentBytes,
				},
			}
		}
	}
	return &model.NativeInferenceReceipt{
		TokenIDs:                   append([]int(nil), measurement.tokenIDs...),
		TokenLogprobs:              append([]float64(nil), measurement.logprobs...),
		PrefillSeconds:             prefillS,
		TTFTSeconds:                measurement.ttftS,
		DecodeSeconds:              decodeS,
		Model:                      p.modelID,
		Engine:                     "inkernel",
		Planner:                    "inkernel",
		Owner:                      "fak",
		Backend:                    backend,
		ForwardPath:                forwardPath,
		Q4K:                        p.q4k,
		FallbackActive:             false,
		PrefillChunkTokens:         p.nativeInferencePrefillChunkTokens(),
		NativeSelection:            nativeSelection,
		NativeSelectionDigest:      nativeSelectionDigest,
		Qwen35MetalForwardSequence: qwen35MetalForwardSequence,
		Qwen35MetalStateIdentity:   qwen35MetalStateIdentity,
		CUDAImmutableWeightUploads: cudaImmutableWeightUploads,
	}
}

func (p *InKernelPlanner) nativeSelectionIdentity() (model.NativeSelectionIdentity, string, error) {
	backend, forwardPath := p.executionIdentity()
	identity := model.NativeSelectionIdentity{
		Schema:              model.NativeSelectionIdentitySchemaV1,
		ModelRef:            p.modelID,
		Backend:             backend,
		ForwardPath:         forwardPath,
		Quantization:        p.nativeSelectionQuantization(),
		PrefillChunkTokens:  p.nativeInferencePrefillChunkTokens(),
		CPUOffloadExperts:   p.nativeSelectionCPUOffloadExperts(),
		Q4KGateUpOutputSlab: p.q4kGateUpOutputSlab,
	}
	digest, err := identity.Digest()
	if err != nil {
		return model.NativeSelectionIdentity{}, "", err
	}
	return identity, digest, nil
}

func (p *InKernelPlanner) nativeSelectionQuantization() string {
	if p != nil && p.q4k {
		return model.NativeSelectionQuantizationQ4K
	}
	if p != nil && p.quant {
		return model.NativeSelectionQuantizationQ8_0
	}
	return model.NativeSelectionQuantizationF32
}

func (p *InKernelPlanner) nativeSelectionCPUOffloadExperts() int {
	if p == nil {
		return 0
	}
	if p.expertSpill != nil {
		return p.expertSpill.Fit.SpillLayers
	}
	if p.cpuOffloadExperts && p.m != nil {
		return len(p.m.MoEExpertLayers())
	}
	return 0
}

func cloneQwen35MetalStateIdentityReceipt(src model.Qwen35MetalStateIdentityReceipt) *model.Qwen35MetalStateIdentityReceipt {
	src.States = append([]model.Qwen35MetalStateDigest(nil), src.States...)
	return &src
}

type cudaImmutableWeightUploadSnapshotter interface {
	CUDAImmutableWeightUploadSnapshot() (calls, transferBytes, residentBytes uint64)
}

func cudaImmutableWeightUploadSnapshot(be compute.Backend) (model.NativeCUDAImmutableWeightUploadCounters, bool) {
	provider, ok := be.(cudaImmutableWeightUploadSnapshotter)
	if !ok {
		return model.NativeCUDAImmutableWeightUploadCounters{}, false
	}
	calls, transferBytes, residentBytes := provider.CUDAImmutableWeightUploadSnapshot()
	return model.NativeCUDAImmutableWeightUploadCounters{Calls: calls, TransferBytes: transferBytes, ResidentBytes: residentBytes}, true
}

func (p *InKernelPlanner) requiresDeviceSerialization() bool {
	return p != nil && (p.backend != nil || p.metal)
}

const inKernelRequestDeviceHeadroom = 0.15
const inKernelRequestPressureTrimMarginRatio = 0.10
const inKernelRequestPressureTrimMinMarginBytes = 64 << 20

func (p *InKernelPlanner) logExecutionSummary(q8kern, q8fusedMark string, promptTok, cacheable, matched, computed int, prefillS, prefTPS float64, generated int, decodeS, decTPS float64) {
	backend, forwardPath := p.executionIdentity()
	log.Printf("inkernel_chat model=%s backend=%s forward_path=%s q4k=%v q8dec=%s%s/%dw prompt=%dtok cacheable=%dtok reused=%dtok prefill=%dtok/%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s",
		p.modelID, backend, forwardPath, p.q4k, q8kern, q8fusedMark, model.Q8DecodeWorkers(), promptTok, cacheable, matched, computed, prefillS, prefTPS, generated, decodeS, decTPS)
}

// executionIdentity makes the request log say which compute path actually produced
// the token. q8dec remains useful CPU implementation metadata, but it must not be
// mistaken for a fallback when a device-backed Qwen3.6 session is selected.
func (p *InKernelPlanner) executionIdentity() (backend, forwardPath string) {
	backend, forwardPath = "cpu-ref", "cpu/reference"
	if p == nil {
		return backend, forwardPath
	}
	if p.backend != nil {
		backend, forwardPath = p.backend.Name(), "device/generic"
	} else if p.metal {
		// Metal is deliberately a CPU-session seam rather than a compute.Backend, but it still
		// owns the projection/MLP dispatch selected for this request. Report that resolved
		// accelerator instead of making a real Metal turn look like a CPU fallback (#8295).
		backend, forwardPath = "metal", "metal/session-forward"
	}
	if p.m != nil && p.m.Cfg.IsQwen35Hybrid() {
		if p.backend != nil {
			// Model.NewBackendSession has already validated the structural GDN
			// contract before this request can complete. Name its stable path here.
			forwardPath = model.Qwen35GDNCUDAPath
		} else if p.metal && p.qwen35MetalGDNExecuted.Load() {
			forwardPath = model.Qwen35MetalGDNSequenceForwardPath
		} else if p.metal {
			// Qwen3.5-family Metal uses the native Session forward with Metal projection/MLP
			// dispatch; q4k= in the same summary records the selected weight format.
			forwardPath = "metal/qwen35-hybrid-session-v1"
		} else {
			forwardPath = "cpu/qwen35-gdn-reference"
		}
	}
	return backend, forwardPath
}
func (p *InKernelPlanner) refuseOversizeRequest(promptTokens, maxNew int) error {
	if p == nil || p.backend == nil || p.m == nil {
		return nil
	}
	plan := p.requestMemoryPlan(promptTokens, maxNew)
	if len(plan) == 0 {
		return nil
	}
	p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
	if err := compute.RefuseMemoryPlanIfTooBig(p.backend, plan, inKernelRequestDeviceHeadroom); err != nil {
		var fe *compute.FitError
		if errors.As(err, &fe) {
			if p.maybeTrimRequestPressure(plan, "capacity_precheck") {
				p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
				if retryErr := compute.RefuseMemoryPlanIfTooBig(p.backend, plan, inKernelRequestDeviceHeadroom); retryErr == nil {
					p.recordRequestPressureTrimResolved(plan, "capacity_precheck")
					return nil
				} else if errors.As(retryErr, &fe) {
					err = retryErr
				} else {
					return retryErr
				}
			}
			return p.capacityErrorFromFit(fe)
		}
		return err
	}
	if p.maybeTrimRequestPressure(plan, "low_margin") {
		p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
	}
	return nil
}

type requestPressureFit struct {
	scope     compute.MemoryScope
	class     compute.MemoryClass
	want      int64
	budget    int64
	margin    int64
	freeKnown bool
}

func (p *InKernelPlanner) maybeTrimRequestPressure(plan compute.MemoryPlan, reason string) bool {
	fit, ok := p.requestDevicePressureFit(plan)
	if !ok || !shouldTrimRequestPressure(fit) {
		return false
	}
	trimmed := p.trimBackendIdlePools()
	p.recordRequestPressureTrim(fit, reason, trimmed, false)
	if trimmed {
		log.Printf("inkernel_chat pressure-trim model=%s backend=%s scope=%s class=%s reason=%s want=%d budget=%d margin=%d action=trim-idle-pools",
			p.modelID, p.backend.Name(), fit.scope, fit.class, reason, fit.want, fit.budget, fit.margin)
	}
	return trimmed
}

func (p *InKernelPlanner) recordRequestPressureTrimResolved(plan compute.MemoryPlan, reason string) {
	fit, ok := p.requestDevicePressureFit(plan)
	if !ok {
		return
	}
	p.recordRequestPressureTrim(fit, reason, false, true)
}

func (p *InKernelPlanner) requestDevicePressureFit(plan compute.MemoryPlan) (requestPressureFit, bool) {
	if p == nil || p.backend == nil {
		return requestPressureFit{}, false
	}
	total, free, known := compute.DeviceMemoryInfo(p.backend)
	if !known || total <= 0 || free < 0 {
		return requestPressureFit{}, false
	}
	want := plan.DeviceTotal()
	if want <= 0 {
		return requestPressureFit{}, false
	}
	budget := ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom)
	return requestPressureFit{
		scope:     compute.MemoryScopeDevice,
		class:     primaryDemandClass(plan, compute.MemoryScopeDevice),
		want:      want,
		budget:    budget,
		margin:    budget - want,
		freeKnown: true,
	}, true
}

func shouldTrimRequestPressure(fit requestPressureFit) bool {
	if !fit.freeKnown || fit.want <= 0 {
		return false
	}
	if fit.margin < 0 {
		return true
	}
	return fit.margin <= requestPressureTrimMarginThreshold(fit.budget)
}

func requestPressureTrimMarginThreshold(budget int64) int64 {
	if budget <= 0 {
		return 0
	}
	threshold := int64(float64(budget) * inKernelRequestPressureTrimMarginRatio)
	if threshold < inKernelRequestPressureTrimMinMarginBytes {
		threshold = inKernelRequestPressureTrimMinMarginBytes
	}
	return threshold
}

func (p *InKernelPlanner) recordRequestPressureTrim(fit requestPressureFit, reason string, trimmed, resolved bool) {
	if p == nil {
		return
	}
	scope := strings.TrimSpace(string(fit.scope))
	if scope == "" {
		scope = string(compute.MemoryScopeDevice)
	}
	class := strings.TrimSpace(string(fit.class))
	if class == "" {
		class = string(compute.MemoryUnknown)
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "unknown"
	}
	p.pressureTrimMu.Lock()
	if p.pressureTrim == nil {
		p.pressureTrim = map[requestPressureTrimKey]*requestPressureTrimStats{}
	}
	key := requestPressureTrimKey{scope: scope, class: class, reason: reason}
	st := p.pressureTrim[key]
	if st == nil {
		st = &requestPressureTrimStats{}
		p.pressureTrim[key] = st
	}
	if resolved {
		st.resolved++
	} else {
		st.attempts++
		if trimmed {
			st.trimmed++
		} else {
			st.noHooks++
		}
	}
	st.lastWantBytes = positiveInt64ToUint64(fit.want)
	st.lastBudgetBytes = positiveInt64ToUint64(fit.budget)
	st.lastMarginBytes = fit.margin
	p.pressureTrimMu.Unlock()
}

func positiveInt64ToUint64(v int64) uint64 {
	if v <= 0 {
		return 0
	}
	return uint64(v)
}

func (p *InKernelPlanner) recordRequestMemoryPlan(promptTokens, maxNew int, plan compute.MemoryPlan) {
	if p == nil || p.backend == nil {
		return
	}
	plannedTokens := promptTokens + maxNew
	if plannedTokens < promptTokens {
		plannedTokens = promptTokens
	}
	deviceTotal, deviceFree, deviceKnown := compute.DeviceMemoryInfo(p.backend)
	hostTotal, hostFree, hostKnown := compute.HostMemoryInfo(p.backend)
	stats := RequestMemoryStats{
		Observed:      len(plan) > 0,
		Backend:       p.backend.Name(),
		PromptTokens:  promptTokens,
		MaxNewTokens:  maxNew,
		PlannedTokens: plannedTokens,
		HeadroomRatio: inKernelRequestDeviceHeadroom,
		MemoryPlan:    requestMemoryDemands(plan),
		Capacities: []RequestMemoryCapacity{
			requestMemoryCapacity(string(compute.MemoryScopeDevice), deviceTotal, deviceFree, deviceKnown),
			requestMemoryCapacity(string(compute.MemoryScopeHost), hostTotal, hostFree, hostKnown),
		},
	}
	p.reqMemMu.Lock()
	p.lastReqMemory = stats
	p.reqMemMu.Unlock()
}

func requestMemoryDemands(plan compute.MemoryPlan) []RequestMemoryDemand {
	if len(plan) == 0 {
		return nil
	}
	out := make([]RequestMemoryDemand, 0, len(plan))
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		out = append(out, RequestMemoryDemand{
			Class:  string(class),
			Scope:  string(d.ScopeOrDefault()),
			DType:  d.DType,
			Bytes:  d.Bytes,
			Detail: d.Detail,
		})
	}
	return out
}

func requestMemoryCapacity(scope string, total, free int64, known bool) RequestMemoryCapacity {
	cap := RequestMemoryCapacity{
		Scope:      scope,
		TotalBytes: total,
		Known:      known,
		FreeKnown:  known && free >= 0,
	}
	if !known {
		cap.TotalBytes = 0
		return cap
	}
	if cap.FreeKnown {
		cap.FreeBytes = free
	}
	return cap
}

func (p *InKernelPlanner) requestMemoryPlan(promptTokens, maxNew int) compute.MemoryPlan {
	if p == nil || p.m == nil {
		return nil
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	if maxNew < 0 {
		maxNew = 0
	}
	plannedTokens := promptTokens + maxNew
	if plannedTokens < promptTokens {
		plannedTokens = promptTokens
	}
	// Delegate to the single context auto-sizer (#1049) — the same function the serve boot
	// path uses — so boot and per-request build a byte-identical KV+scratch plan for the
	// same (model, tokens). The per-request count is exact, so it is the explicit override
	// (>=0); resident weights (below) stay this path's own demand.
	_, plan := compute.AutoSizeContextPlan(p.m.Cfg.ContextSizeConfig(), nil, compute.FreeUnknown, plannedTokens)
	if p.backend != nil && p.includeResidentWeightsInRequestFit() {
		if r := p.m.ResidentReport(); r != nil && r.TotalResidentBytes > 0 {
			plan = append(compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: r.TotalResidentBytes, Detail: "resident-weights", DType: "mixed"}}, plan...)
		}
	}
	return plan
}

func (p *InKernelPlanner) includeResidentWeightsInRequestFit() bool {
	if p == nil || p.backend == nil {
		return false
	}
	_, free, known := compute.DeviceMemoryInfo(p.backend)
	return !known || free < 0
}

func (p *InKernelPlanner) capacityErrorFromFit(fe *compute.FitError) error {
	if fe == nil {
		return nil
	}
	scope := fe.Scope
	if scope == "" {
		scope = compute.MemoryScopeDevice
	}
	return &InKernelCapacityError{
		Want:  fe.Want,
		Avail: fe.Avail,
		Class: primaryDemandClass(fe.Demands, scope),
		Scope: scope,
		Site:  "capacity-precheck",
	}
}

func primaryDemandClass(plan compute.MemoryPlan, scope compute.MemoryScope) compute.MemoryClass {
	var bestClass compute.MemoryClass
	var bestBytes int64
	for _, d := range plan {
		if d.Bytes <= 0 || d.ScopeOrDefault() != scope {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		if d.Bytes > bestBytes {
			bestBytes = d.Bytes
			bestClass = class
		}
	}
	if bestClass == "" {
		return compute.MemoryUnknown
	}
	return bestClass
}

// generateReused runs prefill + decode for an already-encoded prompt, REUSING the longest
// cached KV prefix (the radix tree) when enabled and FAILING OPEN to a full prefill on a
// miss — the candidate-#13 core, factored out of Complete so the reuse/decode path is
// exercisable on a synthetic model with no tokenizer.
//
// emit is invoked with each generated token id AFTER sampling and BEFORE the next Step;
// returning true stops decode with that token counted (Complete's string-suffix stop
// closes over the tokenizer there). A token-id stop (stops[next]) or next<0 ends decode
// WITHOUT emitting — the served contract that a stop token is not echoed.
//
// SNAPSHOT/LEASE discipline: the full-prompt KV is snapshotted (Cloned) right after
// Prefill — BEFORE the decode loop mutates s.Cache by appending generated positions — and
// inserted under a FRESH Lookup so radixkv's lease handoff (Lookup→Insert→Done) is honored
// entirely inside the lock, with no unexported *node escaping this scope. The reuse clone
// (SessionFromPrefix) is also taken under the lock, so a concurrent eviction of the tree
// node can never race our read of its KV. Returns the generated-token count, the prompt
// length, the reused-prefix length, prefill/decode seconds, and whether a stop (not maxNew)
// ended the turn.
