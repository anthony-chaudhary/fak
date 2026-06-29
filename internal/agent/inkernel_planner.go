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
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
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
	maxNew            int
	temp              float64
	seed              int64

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
	// note). FAK_INKERNEL_RADIX_BUDGET sets the LRU edge-token budget (0 = unbounded, the
	// default — the maximal-reuse regime the witnesses measure). Operators serving long
	// sessions should set a budget; bounding the deep-chain footprint is tracked.
	mu   sync.Mutex
	tree *radixkv.Tree

	// devMu serializes the WHOLE device forward pass (Prefill + the decode loop) when a
	// backend is wired. The CUDA backend is single-stream by construction (one g_stream, one
	// cuBLAS handle, a shared size-bucketed device free-list) and its Go-side cudaMu makes
	// each INDIVIDUAL op atomic — but NOT a whole multi-op forward. Two Complete calls driven
	// concurrently by the gateway would interleave their per-token op sequences on that shared
	// device state and stomp each other's activation/KV buffers, faulting the kernel with an
	// illegal memory access that then poisons the CUDA context for every later request until a
	// process restart (observed live on an L4: a 2-way concurrent burst took the GPU serve down
	// with thousands of sticky cuda_kernels.cu illegal-access errors). The radix-reuse path
	// (backend == nil) is already CPU-session-local per turn and guards only its shared tree
	// with p.mu, so devMu engages ONLY on the backend path and leaves the CPU path untouched.
	// This serializes concurrent device requests into safe queuing — correct for a single-stream
	// device — instead of crashing; batched multi-user device decode is the separate throughput
	// follow-up (internal/model/batch.go), not a correctness fix.
	devMu sync.Mutex

	reqMemMu      sync.Mutex
	lastReqMemory RequestMemoryStats

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

// NewInKernelPlanner builds a planner over an already-loaded model + tokenizer.
// q4k flags a resident-Q4_K load so the decode engages Session.Q4K. Generation
// depth/sampling default to a greedy 256-token turn but are overridable via
// FAK_INKERNEL_MAX_TOKENS / FAK_INKERNEL_TEMP / FAK_INKERNEL_SEED.
func NewInKernelPlanner(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, cpuOffloadExpertsOpt ...bool) *InKernelPlanner {
	cpuOffloadExperts := false
	if len(cpuOffloadExpertsOpt) > 0 {
		cpuOffloadExperts = cpuOffloadExpertsOpt[0]
	}
	p := &InKernelPlanner{
		m:                 m,
		tok:               tok,
		modelID:           modelID,
		q4k:               q4k,
		quant:             true, // the served in-kernel path runs the Q8_0 forward (a quantized model)
		backend:           backend,
		metal:             metal,
		cpuOffloadExperts: cpuOffloadExperts,
		maxNew:            envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:              envFloat("FAK_INKERNEL_TEMP", 0),
		seed:              int64(envInt("FAK_INKERNEL_SEED", 0)),
	}
	if backend == nil && metal {
		m.PrepareMetalResidency(q4k)
	}
	// RadixAttention KV-prefix reuse is ON by default; FAK_INKERNEL_RADIX=off disables it
	// (the A/B "tree OFF" arm). The reuse clone is a CPU session, so it only engages when
	// no device backend is wired — the device path keeps its current full-prefill behavior.
	if os.Getenv("FAK_INKERNEL_RADIX") != "off" && backend == nil {
		p.tree = radixkv.New(envInt("FAK_INKERNEL_RADIX_BUDGET", 0))
	}
	// The model-side KV-quarantine eviction bridge (#579) is OFF unless opted in, the same
	// default-off / fail-open posture as the ctxplan seam (FAK_CTXPLAN_SEAM). It runs over a
	// CPU model.Session, so like the radix tree it does not engage a device backend.
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_KVMMU"))) {
	case "on", "1", "true", "yes":
		p.kvSpanEvict = backend == nil
	}
	return p
}

// Model reports the model id (for /v1/models provenance + the planner seam).
func (p *InKernelPlanner) Model() string { return p.modelID }

// KVMemoryStats reports the in-process KV prefix cache's resident memory shape.
// RadixAttention reuse is currently CPU-session backed, so enabled=true reports host
// scoped kv_cache bytes. A device backend uses per-request backend sessions today
// (no persistent radix tree), so it reports enabled=false with the model's per-token
// KV byte geometry for planning visibility.
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
	if p.backend != nil {
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
	stats.ResidentTokens = st.PrefixTokens
	stats.ResidentBytes = compute.EstimateKVStoreBytes(kvCfg, st.PrefixTokens)
	stats.BudgetTokens = st.MaxTokens
	stats.LRUTokens = st.Tokens
	stats.MaxDepthTokens = st.MaxDepthTokens
	stats.Nodes = st.Nodes
	stats.Leaves = st.Leaves
	stats.Evictions = st.Evictions
	stats.PolicyEvictions = st.PolicyEvictions
	stats.Splits = st.Splits
	applyKVMemoryCapacity(&stats, hostTotal, hostFree, hostKnown)
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
		budgetBase = addKVCapacityBytes(free, stats.ResidentBytes)
		if budgetBase > total {
			budgetBase = total
		}
	}
	stats.FitBudgetBytes = applyKVCapacityHeadroom(budgetBase, stats.HeadroomRatio)
	stats.FitMarginBytes = stats.FitBudgetBytes - stats.ResidentBytes
}

func addKVCapacityBytes(a, b int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	if b <= 0 {
		return a
	}
	if a > maxInt64-b {
		return maxInt64
	}
	return a + b
}

func applyKVCapacityHeadroom(bytes int64, headroom float64) int64 {
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
	prefillS, decodeS       float64
	stopped                 bool
}

func (p *InKernelPlanner) generateReusedRecovering(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (res inKernelGenerateResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	gen, promptTok, matched, prefillS, decodeS, stopped, err := p.generateReusedContext(ctx, ids, maxNew, temp, topP, topK, stops, emit)
	if err != nil {
		return inKernelGenerateResult{}, err
	}
	return inKernelGenerateResult{
		gen:       gen,
		promptTok: promptTok,
		matched:   matched,
		prefillS:  prefillS,
		decodeS:   decodeS,
		stopped:   stopped,
	}, nil
}

func (p *InKernelPlanner) generateReusedWithOOMRetry(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool, onRetry func()) (inKernelGenerateResult, error) {
	res, err := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, stops, emit)
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
	retryRes, retryErr := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, stops, emit)
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
	sp := applySampleOpts(opts...)
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

	chat := renderChatMLTools(messages, tools)
	ids, err := p.tok.Encode(chat)
	if err != nil {
		return nil, err
	}
	stops := inKernelStopIDs(p.tok, p.m.Cfg)

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

	// Serialize the entire device forward pass: the single-stream backend cannot run two
	// forwards at once without the concurrent op-streams corrupting shared device buffers
	// (see devMu). On the CPU path (backend == nil) this is a no-op hold — generateReused
	// owns a per-turn session there and guards the shared radix tree with p.mu itself — so
	// the lock costs nothing and the reuse path is unchanged. Held across Prefill + decode.
	if p.backend != nil {
		p.devMu.Lock()
		defer p.devMu.Unlock()
		if err := p.refuseOversizeRequest(len(ids), maxNew); err != nil {
			return nil, err
		}
	}
	genRes, err := p.generateReusedWithOOMRetry(ctx, ids, maxNew, temp, topP, topK, stops, emit, func() {
		sb.Reset()
	})
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
	log.Printf("inkernel_chat model=%s q4k=%v prompt=%dtok reused=%dtok prefill=%dtok/%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s",
		p.modelID, p.q4k, promptTok, matched, computed, prefillS, prefTPS, gen, decodeS, decTPS)
	// Feed the process-global KV-prefix reuse tap so this turn's realized cache-hit
	// (matched/prompt) reaches /metrics, not just this log line — the live measurement of
	// the frozen-trajectory cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md).
	cacheobs.Default.Observe(promptTok, matched)

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
		Message:      Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
		FinishReason: finishReason,
		Usage:        Usage{PromptTokens: promptTok, CompletionTokens: gen, TotalTokens: promptTok + gen},
	}
	// Lift the model's text-form <tool_call> emissions into structured Message.ToolCalls
	// (Hermes dialect == Qwen2.5 native), set FinishReason="tool_calls", and flag a
	// claimed-but-unparseable call — the SAME normalization every proxy adapter runs, so
	// the in-kernel forward becomes a first-class tool-calling planner. Without this the
	// gateway adjudicates nothing (it reads Message.ToolCalls) and the Anthropic wire never
	// emits a tool_use block, so Claude Code's agent loop has nothing to execute.
	comp = normalizeCompletionToolCalls(comp)
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

const inKernelRequestDeviceHeadroom = 0.15
const inKernelRequestPressureTrimMarginRatio = 0.10
const inKernelRequestPressureTrimMinMarginBytes = 64 << 20

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
	budget := applyKVCapacityHeadroom(free, inKernelRequestDeviceHeadroom)
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
func (p *InKernelPlanner) generateReused(ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool) {
	gen, promptTok, matched, prefillS, decodeS, stopped, _ = p.generateReusedContext(context.Background(), ids, maxNew, temp, topP, topK, stops, emit)
	return
}

func (p *InKernelPlanner) generateReusedContext(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool, err error) {
	promptTok = len(ids)
	if len(ids) == 0 {
		return
	}
	if err = ctx.Err(); err != nil {
		return
	}
	reuse := p.tree != nil && p.backend == nil

	// 1) Acquire a session, reusing the longest cached KV prefix when enabled. The clone
	// (SessionFromPrefix) happens under the lock, so once we unlock our session owns an
	// independent copy and a concurrent tree eviction cannot affect this turn's decode.
	var s *model.Session
	if reuse {
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		if k := b.KV(); k != nil {
			s = p.m.SessionFromPrefix(k) // an independent clone; cache.Len() == m
			matched = m
		}
		p.tree.Done(b) // release the lease — we have our clone (or matched nothing)
		p.mu.Unlock()
		// Fully cached (an exact-duplicate transcript): the cached KV has the prefix but
		// not the last-token logits decode must start from. Fail OPEN to a fresh full
		// prefill rather than truncate (some recurrent architectures refuse Evict); the
		// exact-replay case is not the reuse hot path.
		if s != nil && matched >= len(ids) {
			s, matched = nil, 0
		}
	}
	if s == nil {
		matched = 0
		if p.backend != nil {
			s = p.m.NewBackendSession(p.backend)
			defer s.Close()
		} else {
			s = p.m.NewSession()
		}
	}
	s.Quant = p.quant
	// resident-Q4_K decode runs on BOTH the host (cpu-ref) AND the cuda backend: the device HAL
	// copies the raw Q4_K super-blocks resident and serves them with the dequant-fused k_q4k_gemm
	// tile (internal/compute/cuda.go MatMul/BatchedMatMul case Q4_K, #485), so a device session can
	// decode Q4_K directly — no f32/Q8 round-trip. (The old gate forced Q8/F32 on any backend.)
	s.Q4K = p.q4k
	s.CPUOffloadExperts = p.cpuOffloadExperts
	// Apple-Silicon Metal GPU forward (`fak serve --metal`): engage the metalgemm GPU
	// prefill + GPU-resident Q8 decode on the CPU session. Guarded to backend==nil — Metal
	// is the CPU-session seam (s.Backend stays nil), and setting s.Metal on a device session
	// is incoherent; serve also rejects --metal with --backend up front. s.MetalQ4K mirrors
	// cmd/fakchat (s.MetalQ4K = q4k && metal). Inert on non-Metal builds (the model
	// package's metal dispatch falls back to CPU) and the resident decode self-declines any
	// non-dense-Qwen-Q8 model, so this never forces an unsupported GPU path.
	if p.backend == nil && p.metal {
		s.Metal = true
		s.MetalQ4K = p.q4k
	}

	// 2) Prefill ONLY the divergent suffix (the whole prompt on a miss).
	tp := time.Now()
	logits := s.Prefill(ids[matched:])
	prefillS = time.Since(tp).Seconds()
	if err = ctx.Err(); err != nil {
		return
	}

	// 3) Snapshot the full-prompt KV (before decode mutates s.Cache) and cache it under a
	// fresh Lookup→Insert→Done. The snapshot covers the FULL ids prefix, so it is a valid
	// leaf kv no matter how much a concurrent turn may have inserted since step 1.
	if reuse {
		snap := s.Cache.Clone()
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		leaf := p.tree.Insert(b, ids[m:], snap)
		p.tree.Done(leaf)
		p.mu.Unlock()
	}

	// 4) Decode.
	rng := rand.New(rand.NewSource(p.seed))
	td := time.Now()
	for gen = 0; gen < maxNew; gen++ {
		if err = ctx.Err(); err != nil {
			break
		}
		next := sampleLogits(logits, temp, topP, topK, rng)
		if next < 0 || stops[next] {
			stopped = true
			break
		}
		if emit != nil && emit(next) {
			gen++ // this token WAS generated; count it before exiting the loop
			stopped = true
			break
		}
		if emit != nil {
			if err = ctx.Err(); err != nil {
				gen++ // this token was emitted before cancellation became visible
				break
			}
		}
		logits = s.Step(next)
	}
	decodeS = time.Since(td).Seconds()
	return
}

// PoisonEvictor is the narrow seam the gateway drives on a tool-result QUARANTINE: the
// in-kernel KV cache must drop any cached prefix that attended to the now-poisoned result
// (candidate #14), so a later turn re-prefills instead of replaying the poisoned KV. It is
// implemented by InKernelPlanner; the gateway type-asserts its planner to it, so a proxy/
// mock planner — or an in-kernel planner with reuse disabled — simply does not engage it.
type PoisonEvictor interface {
	// EvictPoisoned drops the cached KV prefix along the transcript THROUGH and including
	// messages[throughIdx] (the quarantined tool result, rendered with its ORIGINAL content
	// AND the request's tool schemas). Returns the freed token count (0 if nothing was cached
	// / reuse is off). tools MUST be the SAME tool set the generation turn was rendered with
	// (renderChatMLTools): the tool-spec block folds into the leading system block, so a
	// tools-less eviction render is NOT a token-prefix of a tools-bearing cached turn and
	// fails open (reclaims nothing) — the reuse gap on tool-using turns that #612 closes.
	EvictPoisoned(messages []Message, throughIdx int, tools []ToolDef) int
}

// EvictPoisoned renders the transcript up to and including the poisoned message — WITH the
// request's tool schemas (renderTranscriptTools) but WITHOUT the trailing assistant-open
// marker, so the token path ends exactly on the poison's <|im_end|> turn boundary — encodes
// it, and evicts the cached branch along that path. Rendering WITH tools is load-bearing: the
// generation turn was cached as renderChatMLTools(messages, tools) with the tool-spec folded
// into the leading system block, so the eviction render must fold the SAME spec in or it is
// not a token-prefix of the cached turn and the walk reclaims nothing (the #612 fail-open on
// tool-using turns). TestPrefixInvariantWithTools proves renderTranscriptTools(prefix, tools)
// IS a string-prefix of renderChatMLTools(full, tools); because each turn ends on the atomic
// <|im_end|> special token, the encoded partial transcript is a genuine token-prefix of the
// cached turn, so the walk lands on (and EvictNode drops) the node whose KV saw the poison
// while sparing benign siblings. nil tools renders byte-identically to the historical
// renderTranscript, so a non-tool turn is unchanged. It wraps evictPoisonedIDs.
func (p *InKernelPlanner) EvictPoisoned(messages []Message, throughIdx int, tools []ToolDef) int {
	if p.tree == nil || throughIdx < 0 || throughIdx >= len(messages) {
		return 0
	}
	ids, err := p.tok.Encode(renderTranscriptTools(messages[:throughIdx+1], tools))
	if err != nil || len(ids) == 0 {
		return 0
	}
	freed := p.evictPoisonedIDs(ids)
	if freed > 0 {
		log.Printf("inkernel_chat poison-evict model=%s through_msg=%d freed=%dtok", p.modelID, throughIdx, freed)
	}
	return freed
}

// KVSpanEvictor is the model-side KV-quarantine eviction BRIDGE seam the gateway drives on
// a tool-result QUARANTINE (issue #579). Where PoisonEvictor drops a reusable radixkv PREFIX
// node, this enforces the kvmmu bridge: it rebuilds the transcript's per-message K/V spans on
// a fresh model.Session over the LOADED model and EVICTS the quarantined result's span via the
// proven model.KVCache.Evict (re-RoPE + renumber), so the session's attention state is
// bit-identical to a run that never saw the poison. It is implemented by InKernelPlanner and
// engaged ONLY when FAK_INKERNEL_KVMMU opts in; a proxy/mock planner — or the bridge left off
// — does not implement it, so the gateway's type-assert simply skips it (fail-open default).
type KVSpanEvictor interface {
	// EvictKVSpan rebuilds messages[:throughIdx+1] as labeled per-message K/V segments on a
	// fresh session over the loaded model, then quarantines (evicts) the segment for
	// messages[throughIdx] — the quarantined tool result, rendered with its ORIGINAL content
	// AND the request's tool schemas (so the per-segment spans concatenate to EXACTLY the
	// tools-bearing generation token path, not a tools-less one — #612). It returns the number
	// of K/V positions evicted (0 when the bridge is off or nothing matched) and whether the
	// post-eviction cache is bit-exact to a session that only ever prefilled the survivor spans
	// (the never-saw invariant the kvmmu witnesses certify).
	EvictKVSpan(messages []Message, throughIdx int, tools []ToolDef) (freed int, repositionExact bool)
}

// EvictKVSpan is the live-path KV-MMU bridge (#579): it lowers the transcript through the
// poisoned message into per-message token spans, prefills them as labeled kvmmu segments over
// a FRESH model.Session built from the loaded model, and quarantines the poison segment by id —
// which drives the proven model.KVCache.Evict (re-RoPE + renumber). It then proves the
// reposition was bit-exact by comparing the post-evict next-token logits against a reference
// session that only ever prefilled the survivor spans: equal logits == "the cache is identical
// to never having seen the poison" (the structural, model-independent guarantee — true for any
// weights, which is why a synthetic checkpoint is a faithful witness of the wiring). It is
// inert (returns 0,false) unless FAK_INKERNEL_KVMMU opted the bridge in, so the served path is
// unchanged by default and FAILS OPEN on any encode/cache anomaly.
func (p *InKernelPlanner) EvictKVSpan(messages []Message, throughIdx int, tools []ToolDef) (freed int, repositionExact bool) {
	if !p.kvSpanEvict || p.m == nil || p.tok == nil || throughIdx < 0 || throughIdx >= len(messages) {
		return 0, false
	}
	// Lower each message into the incremental token span it adds to the cumulative transcript.
	// Rendering renderTranscriptTools(messages[:i+1], tools) and slicing past the previous
	// cumulative length makes the per-segment spans concatenate to EXACTLY the full transcript
	// token path the generation turn cached (tool-spec folded into the leading system block, #612)
	// — so the poison segment evicts precisely its own span and the survivors renumber correctly.
	segIDs, poisonSeg, ok := p.lowerSegments(messages, throughIdx, tools)
	if !ok {
		return 0, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	// Build a fresh-session kvmmu bridge over the lowered segments. Fail OPEN on a cache whose
	// eviction is formally unsupported (a hybrid Gated-DeltaNet recurrence: kvmmu's evict would
	// panic KVCache.Evict). The byte-gate quarantine already paged the result out, so the KV-MMU
	// span eviction simply does not engage on such a model rather than crash the served turn.
	sess, bridge, ok := p.newSegmentBridge(segIDs)
	if !ok {
		return 0, false
	}
	freed, found := bridge.Quarantine(poisonSeg)
	if !found || freed == 0 {
		return 0, false
	}
	// Reference: a session that ONLY prefilled the survivor spans. Equal next-token logits
	// (within the cross-path FMA tolerance, 0 on amd64) prove the evicted cache is the never-saw
	// cache. This is the bit-exact reposition invariant, witnessed end-to-end on the live path.
	repositionExact = p.repositionIsExact(sess, segIDs, poisonSeg)
	log.Printf("inkernel_chat kvmmu-evict model=%s through_msg=%d freed=%dpos reposition_exact=%v",
		p.modelID, throughIdx, freed, repositionExact)
	return freed, repositionExact
}

// KVSpanElider is the model-side PLANNED-ELISION residency BRIDGE seam the gateway drives on
// a context-planner elision (issue #579, the kvmmu-planned-eviction half). Where KVSpanEvictor
// enforces a trust QUARANTINE (a poisoned span), this enforces a CAPACITY plan: when the live
// ctxplan view-planner decides the resident view is the last residentTail messages, this evicts
// every OLDER message's K/V span via kvmmu.ApplyPlan (the proven model.KVCache.Evict re-RoPE +
// renumber), so the kernel-owned KV residency SHRINKS to the planner's O(1) resident view
// byte-for-byte — the model's attention state stops physically holding the elided history. The
// elided spans keep a content-address page-back-in handle, so the demand-fault path is intact:
// an elision is a page fault, not a lost fact. Implemented by InKernelPlanner and engaged ONLY
// when FAK_INKERNEL_KVMMU opts in; a proxy/mock planner — or the bridge left off — does not
// implement it, so the gateway's type-assert simply skips it (fail-open default).
type KVSpanElider interface {
	// ElideKVSpans rebuilds messages as labeled per-message K/V segments on a fresh session over
	// the loaded model, then applies the context PLANNER's own ctxplan.Plan — evicting every span
	// the plan Elided via the proven model.KVCache.Evict — so the kernel-owned KV residency shrinks
	// to the plan's O(1) resident view. The plan's span ids MUST be the per-message ids segIDFor
	// mints (the adapter contract kvmmu.ApplyPlan keys on); a plan keyed on foreign ids elides
	// nothing.
	//
	// It returns the number of K/V positions freed (0 when the bridge is off, the plan elided
	// nothing, or the model cannot evict) and whether the post-elision cache is bit-exact to a
	// session that only ever prefilled the resident spans (the O(1)-residency invariant). The
	// bit-exact guarantee holds ONLY in the provable direction — every elided span positionally
	// AFTER every resident span (the over-budget-tail case the kvmmu witness proves), because a
	// re-RoPE cannot un-see attention a surviving earlier token already absorbed from a later one.
	// In any other direction the residency still shrinks and stays recoverable, but repositionExact
	// is reported false rather than asserting an invariant that does not hold.
	ElideKVSpans(messages []Message, plan ctxplan.Plan) (freed int, repositionExact bool)
}

// ElideKVSpans is the live-path planned-elision residency bridge (#579, the kvmmu-planned-eviction
// half): it lowers the full transcript into per-message token spans, prefills them as labeled kvmmu
// segments over a FRESH model.Session built from the loaded model, then applies the context
// planner's own plan via kvmmu.ApplyPlan — which drives the proven model.KVCache.Evict over the
// elided spans (re-RoPE + renumber). The planner's plan already guarantees every elided span
// carries a page-back-in handle (ctxplan.Audit faithfulness), so the demand-fault path stays
// intact — an elision is a page fault, not a lost fact.
//
// When the elided spans are all positionally AFTER the resident spans (the over-budget-tail plan
// the optimizer produces, keeping the early pins and shedding later low-density candidates), the
// post-elision cache is BIT-EXACT to a reference session that only ever prefilled the resident
// spans — proven here by comparing next-token logits, the same structural, model-independent
// guarantee EvictKVSpan asserts for a quarantine. In the other direction (eliding an old prefix a
// resident later span already attended to) a re-RoPE cannot reproduce never-having-seen it, so the
// residency still shrinks but repositionExact is reported false rather than overclaimed. It is
// inert (returns 0,false) unless FAK_INKERNEL_KVMMU opted the bridge in, so the served path is
// unchanged by default and FAILS OPEN on any encode/cache anomaly.
func (p *InKernelPlanner) ElideKVSpans(messages []Message, plan ctxplan.Plan) (freed int, repositionExact bool) {
	if !p.kvSpanEvict || p.m == nil || p.tok == nil || len(messages) == 0 {
		return 0, false
	}
	if len(plan.Elided) == 0 {
		return 0, false // the plan elided nothing — residency already matches the view
	}
	// Lower every message into its incremental token span (the same lowering EvictKVSpan uses,
	// through the LAST message so the spans concatenate to exactly the full transcript path). The
	// segment ids are segIDFor(message, i) — the same ids the plan must carry. #612 threads tools
	// into the poison-eviction render; this planned-elision residency bridge is a SEPARATE seam
	// whose driver does not yet carry the request tools, so it keeps its historical tools-less
	// lowering (nil) — byte-identical to before, a tracked follow-on, not a regression.
	segIDs, _, ok := p.lowerSegments(messages, len(messages)-1, nil)
	if !ok {
		return 0, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	// Fail OPEN on a cache whose eviction is formally unsupported (a recurrent / GDN cache):
	// the residency plan simply does not engage rather than panic the served turn.
	sess, bridge, ok := p.newSegmentBridge(segIDs)
	if !ok {
		return 0, false
	}
	freed = bridge.ApplyPlan(plan)
	if freed == 0 {
		return 0, false // no segment id matched the plan's elided set (or all were held/selected)
	}
	// Bit-exact ONLY in the provable direction: every elided span positionally after every
	// resident one. There, the reference is a prefill of just the resident-prefix spans, and equal
	// next-token logits prove the elided cache is the only-ever-saw-the-view cache (the
	// O(1)-residency invariant). Otherwise residency shrank but the never-saw invariant does not
	// hold, so report false instead of asserting it.
	repositionExact = p.residencyIsExact(sess, segIDs, plan)
	log.Printf("inkernel_chat kvmmu-elide model=%s elided=%d freed=%dpos reposition_exact=%v",
		p.modelID, len(plan.Elided), freed, repositionExact)
	return freed, repositionExact
}

// SegElisionPlan builds the ctxplan.Plan ElideKVSpans consumes from a positional resident/elided
// split of a transcript: message i is Elided iff elided[i], else Selected. Every span id is the
// SAME id lowerSegments mints (segIDFor), so kvmmu.ApplyPlan's id-correspondence contract holds —
// a segment is evicted iff its id is elided and not selected. Every elision carries a sha256
// content-address (ctxplan.Digest) of its rendered message as the page-back-in handle, so the
// plan is ctxplan.Audit-Faithful (every elided span recoverable) and the demand-fault path stays
// intact. It is the adapter the gateway uses to turn the context planner's positional view into a
// segIDFor-keyed plan the residency bridge can apply.
func SegElisionPlan(messages []Message, elided []bool) ctxplan.Plan {
	plan := ctxplan.Plan{Objective: ctxplan.ObjGreedy, Candidates: len(messages)}
	for i := range messages {
		id := segIDFor(messages[i], i)
		if i < len(elided) && elided[i] {
			plan.Elided = append(plan.Elided, ctxplan.Elision{
				ID:     id,
				Step:   i,
				Role:   segTool(messages[i]),
				Digest: ctxplan.Digest([]byte(renderTranscript(messages[i : i+1]))),
				Reason: ctxplan.ElideOverBudget,
			})
			continue
		}
		plan.Selected = append(plan.Selected, ctxplan.Selection{
			ID:   id,
			Step: i,
			Role: segTool(messages[i]),
		})
	}
	return plan
}

// residencyIsExact proves the post-elision cache is bit-identical to a run that only ever saw the
// resident spans — but ONLY in the direction where that is true: every elided span positionally
// AFTER every resident span (so no surviving resident token ever attended to an evicted one). It
// partitions segs by the plan's elided/selected sets, returns false unless the resident set is a
// contiguous positional PREFIX (elided set the suffix), then prefills just the resident-prefix ids
// on a reference session and compares the post-elision next-token distribution to it. Equal
// (within the cross-path FMA tolerance) iff the re-RoPE + renumber left the cache identical to
// never having prefilled the elided suffix. It is the planned-elision twin of repositionIsExact.
func (p *InKernelPlanner) residencyIsExact(elided *model.Session, segs []kvSegment, plan ctxplan.Plan) bool {
	elide := make(map[string]bool, len(plan.Elided))
	for _, e := range plan.Elided {
		elide[e.ID] = true
	}
	for _, s := range plan.Selected {
		delete(elide, s.ID)
	}
	// Walk the segments in cache (positional) order. The resident set is bit-exact-reconstructible
	// only if it is a contiguous PREFIX: once we have seen an elided span, every later span must
	// also be elided (the elided set is the positional suffix). Collect the resident-prefix ids.
	var refIDs []int
	seenElided := false
	for _, sg := range segs {
		if elide[sg.id] {
			seenElided = true
			continue
		}
		if seenElided {
			return false // a resident span sits AFTER an elided one — not the provable direction
		}
		refIDs = append(refIDs, sg.ids...)
	}
	return p.refLogitsExact(elided, refIDs)
}

// kvSegment is one lowered per-message K/V span: its kvmmu segment id (the message index +
// tool-call id), the tool that produced it, and the incremental token ids it occupies.
type kvSegment struct {
	id   string
	tool string
	ids  []int
}

// lowerSegments renders messages[:throughIdx+1] into per-message incremental token spans and
// returns the ordered segments plus the segment id of the poisoned message (messages[throughIdx]).
// It renders WITH the request's tool schemas (renderTranscriptTools) so the lowered spans
// concatenate to exactly the tools-bearing generation token path; nil tools is byte-identical
// to the historical renderTranscript lowering. It fails (ok=false) if any encode errors or any
// incremental span is empty, so a degenerate tokenization fails OPEN to no eviction rather than
// evicting the wrong span.
func (p *InKernelPlanner) lowerSegments(messages []Message, throughIdx int, tools []ToolDef) (segs []kvSegment, poisonID string, ok bool) {
	prev := 0
	for i := 0; i <= throughIdx; i++ {
		cum, err := p.tok.Encode(renderTranscriptTools(messages[:i+1], tools))
		if err != nil || len(cum) <= prev {
			return nil, "", false
		}
		span := append([]int(nil), cum[prev:]...)
		prev = len(cum)
		id := segIDFor(messages[i], i)
		segs = append(segs, kvSegment{id: id, tool: segTool(messages[i]), ids: span})
		if i == throughIdx {
			poisonID = id
		}
	}
	return segs, poisonID, len(segs) > 0
}

// newSegmentBridge builds a FRESH model.Session over the loaded model (carrying the planner's
// quant config) and a kvmmu bridge with every lowered segment appended — the shared session +
// bridge construction EvictKVSpan and ElideKVSpans both run under p.mu before quarantining a
// span or applying a residency plan. It returns ok=false (with a nil session and bridge) on a
// cache whose eviction is formally unsupported (a recurrent / GDN cache, whose CanEvict reads
// non-nil on the empty fresh cache), so the caller fails OPEN rather than panicking the turn.
func (p *InKernelPlanner) newSegmentBridge(segs []kvSegment) (*model.Session, *kvmmu.Context, bool) {
	sess := p.m.NewSession()
	sess.Quant, sess.Q4K = p.quant, p.q4k
	if sess.Cache.CanEvict() != nil {
		return nil, nil, false
	}
	bridge := kvmmu.NewWithGate(sess, kvmmu.FoldedGate{})
	for _, sg := range segs {
		bridge.Append(sg.id, sg.tool, sg.ids)
	}
	return sess, bridge, true
}

// repositionIsExact rebuilds a reference session that prefills ONLY the survivor spans (every
// segment except the poison) and compares the bridge session's post-eviction next-token
// distribution to the reference's. The evicted cache holds the survivor spans at compacted
// positions; decoding one step from the same final survivor token on BOTH reads the
// distribution each would continue from — equal (within the cross-path FMA tolerance) iff the
// eviction's re-RoPE + renumber left the cache bit-identical to never having seen the poison.
func (p *InKernelPlanner) repositionIsExact(evicted *model.Session, segs []kvSegment, poisonID string) bool {
	var refIDs []int
	for _, sg := range segs {
		if sg.id == poisonID {
			continue
		}
		refIDs = append(refIDs, sg.ids...)
	}
	return p.refLogitsExact(evicted, refIDs)
}

// refLogitsExact builds a reference session that prefills ONLY refIDs (the resident survivor
// token path, carrying the planner's quant config) and reports whether `cache`'s post-eviction
// next-token distribution is bit-identical to that reference within the cross-path FMA tolerance.
// It is the shared bit-exact reposition check repositionIsExact and residencyIsExact both close
// on, and returns false when the resident path is empty or the cache length already diverges
// from it.
func (p *InKernelPlanner) refLogitsExact(cache *model.Session, refIDs []int) bool {
	if len(refIDs) == 0 || cache.Cache.Len() != len(refIDs) {
		return false
	}
	ref := p.m.NewSession()
	ref.Quant, ref.Q4K = p.quant, p.q4k
	ref.Prefill(refIDs)
	last := refIDs[len(refIDs)-1]
	return logitsClose(cache.Step(last), ref.Step(last))
}

// logitsClose reports whether two next-token logit vectors are equal within the cross-path FMA
// tolerance (0 on amd64; sub-1e-4 on arches where the gc compiler auto-fuses FMA). It is the
// same max|Δ| reposition measure internal/model's rung-3 oracle and the kvmmu witnesses use.
func logitsClose(a, b []float32) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}
	const tol = 1e-4
	for i := range a {
		d := float64(a[i] - b[i])
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// segIDFor mints the stable kvmmu segment id for a message at index i: the message index keeps
// distinct messages distinct, and the tool-call id (when present) ties the span to the result
// the gateway admitted, so the poison segment is addressable by the same identity the admission
// ledger carries.
func segIDFor(m Message, i int) string {
	if m.ToolCallID != "" {
		return "m" + strconv.Itoa(i) + ":" + m.ToolCallID
	}
	return "m" + strconv.Itoa(i)
}

// segTool reports the producing tool name for the ledger/reporting (the tool result's Name, or
// the role for a non-tool message).
func segTool(m Message) string {
	if m.Name != "" {
		return m.Name
	}
	return m.Role
}

// evictPoisonedIDs drops the cached prefix lying along `ids` (a poisoned transcript token
// path) — the token-level #14 seam EvictPoisoned wraps. Guarded by mu; no-op when reuse
// is disabled.
// underTreeLock runs fn while holding mu, returning 0 (no-op) when the prefix tree
// is absent (reuse disabled). Centralizes the nil-check + lock the prefix-tree
// accessors share so a copy can't drop the guard.
func (p *InKernelPlanner) underTreeLock(fn func() int) int {
	if p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return fn()
}

func (p *InKernelPlanner) evictPoisonedIDs(ids []int) int {
	return p.underTreeLock(func() int { return p.tree.EvictPrefix(ids) })
}

// cachedPrefixLen reports how many leading tokens of `ids` are already resident in the
// prefix cache (read-only). It is the reuse-state probe the witnesses assert on; 0 when
// reuse is disabled.
func (p *InKernelPlanner) cachedPrefixLen(ids []int) int {
	return p.underTreeLock(func() int { return p.tree.MatchLen(ids) })
}

// checkStop reports whether the accumulated decode text ends with any of the
// per-request stop sequences, returning the text with the matched stop suffix
// trimmed. It mirrors the HTTP wires' contract: the stop string ends generation and
// is NOT echoed in the returned content. The LONGEST matching stop wins so the trim
// is maximal, and an empty stop string is ignored (it would otherwise match every
// text and truncate every turn to nothing). An empty stop set never fires, so the
// default in-kernel path is byte-for-byte the pre-seam behavior.
func checkStop(text string, stop []string) (string, bool) {
	best := ""
	for _, s := range stop {
		if s == "" {
			continue
		}
		if strings.HasSuffix(text, s) && len(s) > len(best) {
			best = s
		}
	}
	if best == "" {
		return text, false
	}
	return text[:len(text)-len(best)], true
}

// renderChatML renders the transcript as Qwen/SmolLM2 ChatML, terminating with an
// open assistant turn for generation. System messages fold into one leading system
// block. It is the zero-tools form: renderChatMLTools(messages, nil). The eviction /
// reuse paths use this (and renderTranscript) so their token path is byte-identical to
// the pre-tool-calling behavior — protecting the radix prefix invariant.
func renderChatML(messages []Message) string {
	return renderChatMLTools(messages, nil)
}

// renderChatMLTools is renderChatML with tool support: it advertises the tool JSON
// schemas to the model and renders prior tool-call / tool-result history in Qwen2.5's
// canonical <tool_call>/<tool_response> ChatML. It terminates with an open assistant
// turn for generation. When tools is empty AND no message carries a structured tool
// call or tool result, its output is byte-for-byte identical to the old renderChatML.
func renderChatMLTools(messages []Message, tools []ToolDef) string {
	return renderTranscriptTools(messages, tools) + "<|im_start|>assistant\n"
}

// renderTranscript renders the messages as complete ChatML turns WITHOUT the trailing
// open assistant turn. The zero-tools form: renderTranscriptTools(messages, nil). The
// poison-eviction path uses this so its token path ends exactly on a turn boundary (the
// atomic <|im_end|> special token), keeping it a clean token-prefix of any cached turn
// that began with these messages.
func renderTranscript(messages []Message) string {
	return renderTranscriptTools(messages, nil)
}

// toolSpecBlock renders the canonical Qwen2.5 tool-spec preamble for the folded system
// block: the <tools>…</tools> signatures plus the "emit a <tool_call> json object"
// instruction. It is deterministic (schemas in declaration order) so it is a stable part
// of every token-prefix when folded into the single leading system block — the constraint
// that keeps radix KV reuse valid across a tool-using session.
func toolSpecBlock(tools []ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range tools {
		fn := t.Function
		params := fn.Parameters
		if len(params) == 0 {
			params = json.RawMessage("{}")
		}
		// Marshal one OpenAI-style {"type":"function","function":{…}} signature per tool.
		// Build it from a stable field order via json.Marshal of a map alternative would
		// re-sort keys; use an explicit struct so the rendering is deterministic.
		sig := struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		}{Type: "function"}
		sig.Function.Name = fn.Name
		sig.Function.Description = fn.Description
		sig.Function.Parameters = params
		enc, err := json.Marshal(sig)
		if err != nil {
			// A malformed tool schema must not corrupt the prompt; skip it (the gateway
			// validates schemas upstream, so this is belt-and-suspenders).
			continue
		}
		b.WriteString("\n")
		b.Write(enc)
	}
	b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	return b.String()
}

// renderTranscriptTools is the single rendering core. When tools is non-empty it folds
// the tool-spec block into the leading system block; assistant tool calls render as
// canonical <tool_call> blocks and role=tool messages render as <tool_response> user
// turns. With nil tools and no structured tool call/result on any message, the output is
// byte-for-byte identical to the historical renderTranscript.
func renderTranscriptTools(messages []Message, tools []ToolDef) string {
	var b strings.Builder
	var sys []string
	for _, m := range messages {
		if m.Role == "system" && strings.TrimSpace(m.Content) != "" {
			sys = append(sys, m.Content)
		}
	}
	spec := toolSpecBlock(tools)
	// Emit a leading system block when there is any system text OR a tool spec to
	// advertise. The spec folds into the SAME block (after the system text) so it is part
	// of every token-prefix.
	if len(sys) > 0 || spec != "" {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(strings.Join(sys, "\n"))
		b.WriteString(spec)
		b.WriteString("<|im_end|>\n")
	}
	for _, m := range messages {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			continue
		case "tool":
			// A tool result reads as user-supplied context to the model. When the result
			// carries a tool name, wrap it in Qwen's canonical <tool_response> grammar so a
			// tool-trained model recognizes the multi-turn tool flow; otherwise keep the
			// historical bare "name: content" form (byte-identical to the pre-tool path).
			role = "user"
			if m.Name != "" {
				content = "<tool_response>\n" + m.Name + ": " + content + "\n</tool_response>"
			}
		case "assistant":
			for _, tc := range m.ToolCalls {
				// Canonical Qwen2.5 <tool_call> block: arguments as a JSON VALUE, not a
				// quoted string, so it round-trips cleanly through LiftTextToolCalls.
				args := strings.TrimSpace(tc.Function.Arguments)
				if args == "" || !json.Valid([]byte(args)) {
					args = "{}"
				}
				content += "\n<tool_call>\n{\"name\": " + strconv.Quote(tc.Function.Name) + ", \"arguments\": " + args + "}\n</tool_call>"
			}
		}
		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	return b.String()
}

// inKernelStopIDs mirrors cmd/fakchat.stopIDs: <|im_end|>, <|endoftext|>, and any
// EOS ids the model config declares.
func inKernelStopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
	stops := map[int]bool{}
	for id, content := range tok.SpecialTokens() {
		if content == "<|im_end|>" || content == "<|endoftext|>" {
			stops[id] = true
		}
	}
	if cfg.EOSTokenID > 0 {
		stops[cfg.EOSTokenID] = true
	}
	for _, e := range cfg.EOSTokenIDs {
		if e > 0 {
			stops[e] = true
		}
	}
	return stops
}

// sampleLogits mirrors cmd/fakchat.sample: argmax when temp<=0, else a
// temperature-scaled softmax draw. topK then topP truncate the stochastic path, in
// that order (the standard top-k → top-p pipeline): top-k keeps only the k
// highest-probability tokens, then nucleus (top-p) keeps the smallest set whose
// cumulative mass reaches topP. The tail each step excludes is zeroed before the
// draw. A topK<=0 or topK>=len(logits) disables top-k; a topP<=0 or topP>=1 disables
// nucleus — with both off the draw is the full softmax, byte-for-byte the pre-seam
// behavior. The single most-probable token is always kept so neither cutoff can
// empty the candidate set. Both shape only the stochastic path: temp<=0 stays pure
// argmax (top-k/top-p never change the argmax winner).
func sampleLogits(logits []float32, temp, topP float64, topK int, rng *rand.Rand) int {
	if temp <= 0 {
		best, bi := float32(-math.MaxFloat32), 0
		for i, x := range logits {
			if x > best {
				best, bi = x, i
			}
		}
		return bi
	}
	maxL := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	var sum float64
	probs := make([]float64, len(logits))
	for i, x := range logits {
		p := math.Exp(float64(x-maxL) / temp)
		probs[i] = p
		sum += p
	}
	if topK > 0 && topK < len(probs) {
		sum = topKTruncate(probs, sum, topK)
	}
	if topP > 0 && topP < 1 {
		sum = nucleusTruncate(probs, sum, topP)
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return i
		}
	}
	// Fall back to the last token with nonzero mass (nucleus zeroed the tail).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return len(logits) - 1
}

// nucleusTruncate zeroes every probability outside the top-p nucleus in place and
// returns the surviving mass (the new normalization sum). The nucleus is the
// smallest set of highest-probability tokens whose cumulative mass reaches topP;
// the single most-probable token is always kept so the nucleus is never empty.
// probs is unsorted on entry and stays index-aligned to the caller's logits.
func nucleusTruncate(probs []float64, sum, topP float64) float64 {
	order := descProbOrder(probs, func(i, j int) bool { return probs[i] > probs[j] })
	target := topP * sum
	var cum float64
	kept := make(map[int]bool, len(order))
	for rank, idx := range order {
		// Stop BEFORE adding this token once the nucleus already reached the target —
		// the kept set is the minimal prefix whose mass >= target. Rank 0 is always
		// kept (the head token) so the nucleus is never empty.
		if rank > 0 && cum >= target {
			break
		}
		kept[idx] = true
		cum += probs[idx]
	}
	return maskKept(probs, kept)
}

// descProbOrder returns the indices of probs ordered by the caller's less comparator, which
// ranks two ELEMENT indices (not positions in the returned slice). It is the shared
// highest-probability-first index permutation nucleusTruncate and topKTruncate sort on; each
// passes its own tie-break (nucleus leaves equal masses in arbitrary order; topK breaks ties by
// the lower index for a stable kept set).
func descProbOrder(probs []float64, less func(i, j int) bool) []int {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return less(order[a], order[b]) })
	return order
}

// maskKept zeroes every probability whose index is not in kept (in place) and returns the
// surviving mass (the new normalization sum) — the shared renormalization tail of
// nucleusTruncate and topKTruncate.
func maskKept(probs []float64, kept map[int]bool) float64 {
	var newSum float64
	for i := range probs {
		if kept[i] {
			newSum += probs[i]
		} else {
			probs[i] = 0
		}
	}
	return newSum
}

// topKTruncate zeroes every probability outside the top-k highest-probability
// tokens in place and returns the surviving mass (the new normalization sum). Ties
// at the k-th rank are broken by index order (the sort is stable on equal probs via
// the index comparator), so the kept set is deterministic. probs is unsorted on
// entry and stays index-aligned to the caller's logits. The caller guarantees
// 0 < k < len(probs); k>=len(probs) is a no-op handled before the call so the full
// distribution stays byte-for-byte the pre-seam draw.
func topKTruncate(probs []float64, sum float64, k int) float64 {
	// Highest probability first; ties resolve to the lower index so the kept set is
	// stable and reproducible across runs.
	order := descProbOrder(probs, func(i, j int) bool {
		if probs[i] != probs[j] {
			return probs[i] > probs[j]
		}
		return i < j
	})
	kept := make(map[int]bool, k)
	for rank := 0; rank < k; rank++ {
		kept[order[rank]] = true
	}
	return maskKept(probs, kept)
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
