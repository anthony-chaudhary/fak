package agent

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"log"
)

// SetPromptShrinkLevers configures the prompt-shrink levers for this planner.
func (p *InKernelPlanner) SetPromptShrinkLevers(compactBudget int, elideStaleReads, deferColdTools bool) {
	if p == nil {
		return
	}
	p.compactHistoryBudget = compactBudget
	p.elideStaleReads = elideStaleReads
	p.deferColdTools = deferColdTools
}

// SetRestoreStash attaches a content-addressed storage callback for elided / compacted turns.
func (p *InKernelPlanner) SetRestoreStash(stash func(trace, id, excerpt string, body []byte)) {
	if p == nil {
		return
	}
	p.restoreStash = stash
}

// SetIncrementalContext installs the composed system-context registry for this planner
// (inkernel_incremental.go). The durable comparison snapshot is deliberately left in
// place: it is what lets a reconcile detect that a source has LEFT the registry (and
// require a replacement) rather than treating the swap as a fresh, empty comparison. A
// nil ctx is the inert identity, preserving the historical planner byte-for-byte.
func (p *InKernelPlanner) SetIncrementalContext(ctx *IncrementalContext) {
	if p == nil {
		return
	}
	p.ctxMu.Lock()
	defer p.ctxMu.Unlock()
	p.incrementalContext = ctx
}

// IncrementalContextGeneration reports a fresh baseline generation for the installed
// registry. The bool is false (and the generation zero) when no registry is set; the
// error is *InitializationBlockedError when any source is temporarily unavailable.
func (p *InKernelPlanner) IncrementalContextGeneration() (ContextGeneration, bool, error) {
	if p == nil {
		return ContextGeneration{}, false, nil
	}
	p.ctxMu.Lock()
	ctx := p.incrementalContext
	p.ctxMu.Unlock()
	if ctx == nil {
		return ContextGeneration{}, false, nil
	}
	generation, err := InitializeIncrementalContext(ctx)
	return generation, true, err
}

// ReconcileIncrementalContextNow reconciles the installed registry against the
// planner's stored snapshot. With no registry the result is Unchanged. On Updated or
// ReplacementReady the stored snapshot ADVANCES (correct context invalidation); on
// ReplacementBlocked it is left untouched so a later retry still sees the admitted
// state.
//
// The caller must NOT mutate the returned ReconcileResult.Snapshot: on Unchanged it
// aliases the planner's stored snapshot, and on Updated/ReplacementReady the planner
// retains the same map it returns. Treat it as read-only comparison state.
//
// ctxMu is held across the source load/baseline/update/removed callbacks, so those
// callbacks must NOT re-enter the planner (SetIncrementalContext,
// IncrementalContextGeneration, or this method) on pain of deadlock.
func (p *InKernelPlanner) ReconcileIncrementalContextNow() (ReconcileResult, error) {
	if p == nil {
		return ReconcileResult{Kind: ReconcileUnchanged}, nil
	}
	p.ctxMu.Lock()
	defer p.ctxMu.Unlock()
	if p.incrementalContext == nil {
		return ReconcileResult{Kind: ReconcileUnchanged}, nil
	}
	res, err := ReconcileIncrementalContext(p.incrementalContext, p.ctxSnapshot)
	if err != nil {
		return ReconcileResult{}, err
	}
	switch res.Kind {
	case ReconcileUpdated, ReconcileReplacementReady:
		p.ctxSnapshot = res.Snapshot
	}
	return res, nil
}

// ApplyIncrementalContextUpdate reconciles the installed registry and returns the
// model-visible incremental delta. invalidated is true exactly when a
// replacement_ready occurred, signalling the caller must re-baseline resident context.
func (p *InKernelPlanner) ApplyIncrementalContextUpdate() (string, bool, error) {
	res, err := p.ReconcileIncrementalContextNow()
	if err != nil {
		return "", false, err
	}
	return res.Text, res.Kind == ReconcileReplacementReady, nil
}

// SetContextLayerMemo installs the per-session Source/Registry/Snapshot layer memo
// (inkernel_incremental.go). nil leaves the planner on the historical full-render path,
// so a planner that never calls this is byte-for-byte unchanged. It is nil-receiver safe.
func (p *InKernelPlanner) SetContextLayerMemo(m *ContextLayerMemo) {
	if p == nil {
		return
	}
	p.ctxMu.Lock()
	defer p.ctxMu.Unlock()
	p.ctxLayerMemo = m
}

// RenderIncrementalContext renders messages/tools through the installed layer memo and
// returns the prompt plus its token ids. With no memo installed (or a nil planner, model,
// or tokenizer) it returns the full rebuild and an all-rendered stats record.
//
// The memo reuses the Source and Snapshot layers and re-renders only the Registry layer
// when the request is a pure append of the previous one; its own mandatory guard falls
// back to the full rebuild whenever the incremental assembly cannot be proven
// byte-identical to it. The DeepSeek V4.1 raw-completion BOS rule (preparePrompt) is
// applied here to the returned ids exactly once Ã¢â‚¬â€ BOS is an id-only prefix, never text,
// so it is deliberately absent from the memo's fingerprint.
func (p *InKernelPlanner) RenderIncrementalContext(messages []Message, tools []ToolDef, sp SampleParams) (string, []int, ContextLayerStats, error) {
	if p == nil || p.m == nil || p.tok == nil {
		return "", nil, ContextLayerStats{}, errors.New("native incremental context requires a model and tokenizer")
	}
	if p.nativeChatTemplate != nil {
		if err := p.admitDeepSeekV41ChatProfile(messages, sp); err != nil {
			return "", nil, ContextLayerStats{}, err
		}
		segments, err := p.renderNativeChatTemplate(context.Background(), messages, tools, sp)
		if err != nil {
			return "", nil, ContextLayerStats{}, err
		}
		ids, err := p.tok.EncodeSegments(segments)
		if err != nil {
			return "", nil, ContextLayerStats{}, fmt.Errorf("%w: selected tokenizer refused", ErrNativeChatTemplate)
		}
		if len(ids) == 0 {
			return "", nil, ContextLayerStats{}, fmt.Errorf("%w: empty encoded prompt", ErrNativeChatTemplate)
		}
		// No legacy layer memo may turn marked request strings into trusted text.
		return joinPromptSegments(segments), ids, ContextLayerStats{Rendered: contextLayerAll()}, nil
	}
	full := func(ms []Message, ts []ToolDef) string { return inKernelRenderPrompt(ms, ts, p.m.Cfg, sp) }
	encode := p.tok.Encode

	p.ctxMu.Lock()
	memo := p.ctxLayerMemo
	p.ctxMu.Unlock()

	if memo == nil {
		fullText := full(messages, tools)
		ids, err := encode(fullText)
		if err != nil {
			return "", nil, ContextLayerStats{}, err
		}
		return fullText, p.applyInKernelBOS(ids, sp), ContextLayerStats{Rendered: contextLayerAll()}, nil
	}

	plan, err := memo.PlanContextLayers(messages, tools, full, encode)
	if err != nil {
		return "", nil, ContextLayerStats{}, err
	}
	return plan.Prompt, p.applyInKernelBOS(plan.IDs, sp), plan.Stats, nil
}

// applyInKernelBOS prepends the GGUF-declared BOS id for a raw DeepSeek V4.1 completion
// request, mirroring preparePrompt. It is a no-op for every other renderer.
func (p *InKernelPlanner) applyInKernelBOS(ids []int, sp SampleParams) []int {
	if p == nil || p.m == nil {
		return ids
	}
	if inKernelRawV41Prompt(p.m.Cfg, sp) && p.m.Cfg.BOSTokenID != nil {
		return append([]int{*p.m.Cfg.BOSTokenID}, ids...)
	}
	return ids
}

// ApplyPromptShrink evaluates the configured or request-level prompt-shrink levers
// over typed messages and tools before tokenization.
func (p *InKernelPlanner) ApplyPromptShrink(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) ([]Message, []ToolDef, TypedPromptShrinkOutcome) {
	return p.applyPromptShrinkResolved(ctx, messages, tools, applySampleOpts(opts...))
}

// applyPromptShrinkResolved shares the already-resolved controls used to render
// this prompt. It does not evaluate option closures or freeze caller-owned data.
func (p *InKernelPlanner) applyPromptShrinkResolved(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams) ([]Message, []ToolDef, TypedPromptShrinkOutcome) {
	compactBudget := p.compactHistoryBudget
	if sp.CompactHistoryBudget != nil {
		compactBudget = *sp.CompactHistoryBudget
	}
	elideStale := p.elideStaleReads
	if sp.ElideStaleReads != nil {
		elideStale = *sp.ElideStaleReads
	}
	deferTools := p.deferColdTools
	if sp.DeferColdTools != nil {
		deferTools = *sp.DeferColdTools
	}

	if compactBudget <= 0 && !elideStale && !deferTools {
		return messages, tools, TypedPromptShrinkOutcome{}
	}

	var stash func(id, excerpt string, body []byte)
	if p.restoreStash != nil {
		traceID := nativePhaseTraceID(ctx)
		stash = func(id, excerpt string, body []byte) {
			p.restoreStash(traceID, id, excerpt, body)
		}
	}

	return ApplyTypedPromptShrinkLevers(messages, tools, TypedPromptShrinkConfig{
		CompactHistoryBudget:    compactBudget,
		ElideStaleReads:         elideStale,
		DeferColdTools:          deferTools,
		RestoreStash:            stash,
		observeNativeCompaction: nativeCompactionObserver(ctx) != nil,
	})
}

// KVMemoryStats reports the in-process KV prefix cache's physical resident shape.
// Native backend snapshots are split into hot device bytes, hot host metadata, and
// the independently owned host-DRAM L2. Proxy/provider counters never enter here.
// effectiveKVConfig is the full context charge this planner's admission math must use:
// the model's geometry plus the realized KV storage tier (p.kvPrecision, mapped to the
// compute tier). The F32 default maps to compute.KVPrecisionF32, so a planner that
// never set a tier estimates byte-identically to before.
func (p *InKernelPlanner) effectiveKVConfig() compute.ContextSizeConfig {
	return p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision))
}

// computeKVPrecisionFor maps a model.KVPrecision tier to the compute.KVPrecision the
// byte estimate understands. Only f32 and q8_0 are realized; any other declared tier
// falls back to f32 so the estimate never claims a density the engine does not have.
func computeKVPrecisionFor(prec model.KVPrecision) compute.KVPrecision {
	switch prec {
	case model.KVPrecisionQ8_0:
		return compute.KVPrecisionQ8
	case model.KVPrecisionFP16:
		return compute.KVPrecisionF16
	case model.KVPrecisionBF16:
		return compute.KVPrecisionBF16
	default:
		return compute.KVPrecisionF32
	}
}

func deviceKV16(prec model.KVPrecision) bool {
	return prec == model.KVPrecisionFP16 || prec == model.KVPrecisionBF16
}

// pickDeviceKVTier admits a 16-bit device KV tier only where the backend's HAL
// store and the model's forward path realize it; otherwise it falls back to f32 and names
// why. Every other tier keeps the per-session host-cache checks.
func pickDeviceKVTier(cfg model.Config, backend compute.Backend, want model.KVPrecision) (model.KVPrecision, string) {
	if backend == nil || !deviceKV16(want) {
		return want, ""
	}
	if reason := cfg.DeviceKVRefusal(backend.Name(), want); reason != "" {
		return model.KVPrecisionFP32, fmt.Sprintf("kv precision %s -> fp32 (source=device-kv-fallback backend=%s): %s", want, backend.Name(), reason)
	}
	return want, ""
}

// KVPrecisionFallback reports why the requested device KV tier fell back to f32, or "".
func (p *InKernelPlanner) KVPrecisionFallback() string {
	if p == nil {
		return ""
	}
	return p.kvPrecisionFallback
}

func (p *InKernelPlanner) KVMemoryStats() KVMemoryStats {
	if p == nil || p.m == nil {
		return KVMemoryStats{
			MemoryClass: string(compute.MemoryKVCache),
			Scope:       string(compute.MemoryScopeHost),
			DType:       compute.F32.String(),
		}
	}
	kvCfg := p.effectiveKVConfig()
	bytesPerToken := kvCfg.KVBytesPerToken()
	stats := KVMemoryStats{
		Enabled:             p.tree != nil,
		Backend:             "radixkv",
		MemoryClass:         string(compute.MemoryKVCache),
		Scope:               string(compute.MemoryScopeHost),
		DType:               kvCfg.KV.Precision.StorageLabel(),
		BytesPerToken:       bytesPerToken,
		FixedBytesPerStream: kvCfg.FixedKVBytesPerStream(),
		HeadroomRatio:       inKernelKVMemoryHeadroom,
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
		stats.ResidentBytes = kvCfg.KVBytes(st.PrefixTokens)
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
	stats.LocalL3Enabled = st.LocalL3Enabled
	stats.LocalL3BudgetBytes = st.LocalL3BudgetBytes
	stats.LocalL3ReferencedBytes = st.LocalL3ReferencedBytes
	stats.LocalL3Hits = st.LocalL3Hits
	stats.LocalL3Misses = st.LocalL3Misses
	stats.LocalL3Faults = st.LocalL3Faults
	stats.LocalL3HitTokens = st.LocalL3HitTokens
	stats.LocalL3StageBytes = st.LocalL3StageBytes
	stats.LocalL3RestoreBytes = st.LocalL3RestoreBytes
	stats.LocalL3Evictions = st.LocalL3Evictions
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
// >=1) as "reserve nothing" rather than as a clamp Ã¢â‚¬â€ a 0 budget must stay 0, and a bogus
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

// EnableConcurrencyProfile turns on the opt-in device-fan-out phase-timeline
// recorder (issue #1589). It must be called before concurrent Complete calls.
// Enabling it changes no served tokens and no batching: it only records the
// admit / forward-enter / forward-exit timestamps at the existing devMu seam.
func (p *InKernelPlanner) EnableConcurrencyProfile() {
	if p == nil {
		return
	}
	if p.concurrencyProfile == nil {
		p.concurrencyProfile = newConcurrencyProfiler()
	}
}

// ConcurrencyProfile folds the recorded phase timeline into a named-mechanism
// receipt. It returns nil when no profile was enabled. Callers must invoke it
// after all profiled requests have returned.
func (p *InKernelPlanner) ConcurrencyProfile() *ConcurrencyProfile {
	if p == nil {
		return nil
	}
	_, forwardPath := p.executionIdentity()
	return p.concurrencyProfile.build(forwardPath)
}

const inKernelRequestDeviceHeadroom = 0.15
const inKernelRequestPressureTrimMarginRatio = 0.10
const inKernelRequestPressureTrimMinMarginBytes = 64 << 20

// nativeDecodeRate derives a sustained decode throughput from a generate call's
// raw counters, or reports it unavailable. The first generated token is emitted
// by the prefill block, so only the gen-1 completed post-prefill intervals are
// decode work; a rate needs at least one such interval AND positive elapsed
// decode time. When either is absent the rate is unavailable (ok=false) and the
// caller must not serialize a finite success value ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the defect #13298 where a
// single warmup token with a near-zero window reported ~6957 tok/s. Raw counters
// (gen, decodeS) are preserved untouched; only the derived rate is gated.
func nativeDecodeRate(gen int, decodeS float64) (float64, bool) {
	if gen < 2 || decodeS <= 0 {
		return 0, false
	}
	return float64(gen-1) / decodeS, true
}

func (p *InKernelPlanner) logExecutionSummary(q8kern, q8fusedMark string, promptTok, cacheable, matched, computed int, prefillS, prefTPS float64, generated int, decodeS, decTPS float64, cohortClause string) {
	backend, forwardPath := p.executionIdentity()
	hostClause, q8Clause := p.executionPlacementClauses(q8kern, q8fusedMark)
	log.Printf("inkernel_chat model=%s backend=%s forward_path=%s%s q4k=%v%s prompt=%dtok cacheable=%dtok reused=%dtok prefill=%dtok/%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s%s%s gemm=%s%s%s kv=%s",
		p.modelID, backend, forwardPath, hostClause, p.q4k, q8Clause, promptTok, cacheable, matched, computed, prefillS, prefTPS, generated, decodeS, decTPS, p.v41FaultAttributionClause(), cohortClause, p.prefillQ4KGEMMLabel(), p.hostPathDegradedClause(), p.requantizedWeightsClause(), p.kvPrecisionLabel())
}

// inKernelCohortReceiptClause renders the decode coalescer's batch receipt as
// the execution summary's cohort= clause: the cohort id, the peak lane count
// the request ran beside, and the shared projection panels the batched
// forwards attributed to this request. Empty for a non-coalesced turn
// (CohortID 0), so every pre-existing summary line and its parsers stay
// byte-identical. The clause is what lets a served device run PROVE the
// coalescer engaged from its own serve log: cohort=<id>/lanes=1/panels=0 is
// the honest solo-cohort shape (a B==1 coalesced turn is bit-identical to
// serial), lanes>1 with panels>0 is the shared-weight batched forward ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
// without it a concurrency receipt can only infer engagement from aggregate
// tok/s, which cannot distinguish never-coalesced from coalesced-then-
// governed-to-serial (fak#8395 device witness).
func inKernelCohortReceiptClause(r InKernelBatchReceipt) string {
	if r.CohortID == 0 {
		return ""
	}
	return fmt.Sprintf(" cohort=%d/lanes=%d/panels=%d", r.CohortID, r.CohortSize, r.SharedPanels)
}

// kvPrecisionLabel is the kv= value of the execution summary: the realized KV tier,
// with the unset zero value spelled as the f32 it stores.
func (p *InKernelPlanner) kvPrecisionLabel() string {
	if p == nil || p.kvPrecision == "" {
		return string(model.KVPrecisionFP32)
	}
	return string(p.kvPrecision)
}

// prefillQ4KGEMMLabel is the gemm= value of the execution summary: the Q4_K prefill GEMM
// route(s) the request's prefill observed (scalar, a newer small-P/MM32/M5 identity, cpu,
// gemv, token_loop, or "+"-joined when mixed), or model.Q4KPrefillGEMMNone when no prefill
// ran or nothing was observed. Appended last so every existing key keeps its position.
func (p *InKernelPlanner) prefillQ4KGEMMLabel() string {
	if p != nil {
		if label, _ := p.prefillQ4KGEMM.Load().(string); label != "" {
			return label
		}
	}
	return model.Q4KPrefillGEMMNone
}

// v41ExpertFaultAttributioner is the optional read seam a V4.1 model exposes
// so the execution summary can attribute the routed-expert fault cost per
// forward phase (#13294 DoD item 1). Every model WITHOUT the method ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the
// whole non-V4.1 fleet ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â fails this assertion and skips the clause, so their
// summary lines stay byte-for-byte unchanged.
type v41ExpertFaultAttributioner interface {
	V41ExpertFaultAttribution() model.V41ExpertFaultAttribution
}

// v41FaultAttributionClause renders the appended clause of the execution
// summary: the phase-split routed-expert fault attribution for the model the
// turn ran on. Empty when the model does not expose it OR when both phase
// ledgers are untouched ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a non-V4.1 turn reports nothing rather than a zero
// clause, keeping the existing line the physical receipts parse. The two
// phase ledgers are read through their exported FIELDS (field access through
// selectors never names the unexported ledger type), so the formatting stays
// entirely in the agent layer.
func (p *InKernelPlanner) v41FaultAttributionClause() string {
	if p == nil || p.m == nil {
		return ""
	}
	at := v41ExpertFaultAttributioner(p.m)
	if at == nil {
		return ""
	}
	return formatV41FaultClause(at.V41ExpertFaultAttribution())
}

// formatV41FaultClause is the pure renderer behind v41FaultAttributionClause,
// split out so a unit test can assert the exact clause without a live model.
// The byte clause is unchanged from the pre-elapsed form (every physical
// receipt already parses it); the #13299 elapsed split is APPENDED so an
// operator reading one physical serve log can rank disk wait against f32
// dequantization against the routed contraction ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the choice PLAN
// halo-ds41-100-30 Ãƒâ€šÃ‚Â§101 makes between the residency (#12952), device-contraction
// (#13128) and dequant levers. Seconds and per-token milliseconds come straight
// from the ledger's nanosecond accumulators; the contraction backend is the
// observed engine identity, never a device receipt.
func formatV41FaultClause(fa model.V41ExpertFaultAttribution) string {
	if fa == (model.V41ExpertFaultAttribution{}) {
		return ""
	}
	const mib = 1.0 / (1 << 20)
	pre, dec := fa.Prefill, fa.Decode
	hits, reads := pre.ResidentHits+dec.ResidentHits, pre.ResidentHits+dec.ResidentHits+pre.Faults+dec.Faults
	hitFraction := 0.0
	if reads > 0 {
		hitFraction = float64(hits) / float64(reads)
	}
	// The backend identity is only stamped once a contraction ran; when neither
	// phase contracted, say so with a dash rather than an ambiguous empty value.
	backend := pre.ContractionBackend
	if backend == "" {
		backend = "-"
	}
	compressorClause := ""
	if v41CompressorPhaseSnapshot(pre) != (V41CompressorPhaseSnapshot{}) || v41CompressorPhaseSnapshot(dec) != (V41CompressorPhaseSnapshot{}) {
		compressorClause = fmt.Sprintf(
			" compressor_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs]",
			pre.CompressorProjectionDeviceCalls, pre.CompressorProjectionDeviceRows, pre.CompressorProjectionHostCalls, pre.CompressorProjectionHostRows, pre.CompressorProjectionActivationUploadBytes, pre.CompressorProjectionReadbackBytes, float64(pre.CompressorProjectionNanos)/1e9,
			dec.CompressorProjectionDeviceCalls, dec.CompressorProjectionDeviceRows, dec.CompressorProjectionHostCalls, dec.CompressorProjectionHostRows, dec.CompressorProjectionActivationUploadBytes, dec.CompressorProjectionReadbackBytes, float64(dec.CompressorProjectionNanos)/1e9)
	}
	indexerClause := ""
	if v41IndexerPhaseSnapshot(pre) != (V41IndexerPhaseSnapshot{}) || v41IndexerPhaseSnapshot(dec) != (V41IndexerPhaseSnapshot{}) {
		indexerClause = fmt.Sprintf(" indexer prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB projection=%.3fs scoring=%dcalls/%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB projection=%.3fs scoring=%dcalls/%.3fs]",
			pre.IndexerProjectionDeviceCalls, pre.IndexerProjectionDeviceRows, pre.IndexerProjectionHostCalls, pre.IndexerProjectionHostRows, pre.IndexerProjectionActivationUploadBytes, pre.IndexerProjectionReadbackBytes, float64(pre.IndexerProjectionNanos)/1e9, pre.IndexerScoringCalls, float64(pre.IndexerScoringNanos)/1e9,
			dec.IndexerProjectionDeviceCalls, dec.IndexerProjectionDeviceRows, dec.IndexerProjectionHostCalls, dec.IndexerProjectionHostRows, dec.IndexerProjectionActivationUploadBytes, dec.IndexerProjectionReadbackBytes, float64(dec.IndexerProjectionNanos)/1e9, dec.IndexerScoringCalls, float64(dec.IndexerScoringNanos)/1e9)
	}
	combinedClause := ""
	if pre.HostExpertCombinedCalls > 0 || dec.HostExpertCombinedCalls > 0 {
		status := func(c model.V41PhaseFieldCoverage) string {
			if c.Status == "" {
				return "unavailable"
			}
			return c.Status
		}
		combinedClause = fmt.Sprintf(" host_expert_combined prefill=[attempts=%d elapsed=%.3fs coverage=%s] decode=[attempts=%d elapsed=%.3fs coverage=%s] split_coverage prefill=[dequant=%s contraction=%s] decode=[dequant=%s contraction=%s]",
			pre.HostExpertCombinedCalls, float64(pre.HostExpertCombinedNanos)/1e9, status(pre.Coverage.HostExpertCombinedTime),
			dec.HostExpertCombinedCalls, float64(dec.HostExpertCombinedNanos)/1e9, status(dec.Coverage.HostExpertCombinedTime),
			status(pre.Coverage.DequantTime), status(pre.Coverage.ContractionTime), status(dec.Coverage.DequantTime), status(dec.Coverage.ContractionTime))
	}
	return fmt.Sprintf(
		" v41_faults prefill=[faults=%df/%.1ffpt faulted_bytes=%.1fMiB dequant=%.2fGiB] decode=[faults=%df/%.1ffpt faulted_bytes=%.1fMiB dequant=%.2fGiB] hit_fraction=%.2f elapsed prefill=[fault=%.3fs/%.3fmspt dequant=%.3fs/%.3fmspt contraction=%.3fs/%.3fmspt] decode=[fault=%.3fs/%.3fmspt dequant=%.3fs/%.3fmspt contraction=%.3fs/%.3fmspt] contraction_backend=%s incremental_device prefill=[gate_up=%d down=%d dispatch=%.3fs] decode=[gate_up=%d down=%d dispatch=%.3fs] incremental_engram prefill=[injections=%d rows=%d hash_tokens=%d elapsed=%.3fs] decode=[injections=%d rows=%d hash_tokens=%d elapsed=%.3fs] expert_activation prefill=[device=%d host=%d readback=%dB elapsed=%.3fs] decode=[device=%d host=%d readback=%dB elapsed=%.3fs] dense_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs]%s head_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] grouped_output prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] engram_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] mhc_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs]",
		pre.Faults, pre.FaultsPerToken, mib*float64(pre.FaultedBytes), float64(pre.DequantBytes)/(1<<30),
		dec.Faults, dec.FaultsPerToken, mib*float64(dec.FaultedBytes), float64(dec.DequantBytes)/(1<<30),
		hitFraction,
		float64(pre.FaultDoorNanos)/1e9, pre.FaultNanosPerToken/1e6,
		float64(pre.DequantNanos)/1e9, pre.DequantNanosPerToken/1e6,
		float64(pre.ContractionNanos)/1e9, pre.ContractionNanosPerToken/1e6,
		float64(dec.FaultDoorNanos)/1e9, dec.FaultNanosPerToken/1e6,
		float64(dec.DequantNanos)/1e9, dec.DequantNanosPerToken/1e6,
		float64(dec.ContractionNanos)/1e9, dec.ContractionNanosPerToken/1e6,
		backend,
		pre.IncrementalDeviceGateUpCalls, pre.IncrementalDeviceDownCalls, float64(pre.IncrementalDeviceDispatchNanos)/1e9,
		dec.IncrementalDeviceGateUpCalls, dec.IncrementalDeviceDownCalls, float64(dec.IncrementalDeviceDispatchNanos)/1e9,
		pre.IncrementalEngramInjections, pre.IncrementalEngramRows, pre.IncrementalEngramHashTokens, float64(pre.IncrementalEngramNanos)/1e9,
		dec.IncrementalEngramInjections, dec.IncrementalEngramRows, dec.IncrementalEngramHashTokens, float64(dec.IncrementalEngramNanos)/1e9,
		pre.ExpertActivationDeviceCalls, pre.ExpertActivationHostCalls, pre.ExpertActivationReadbackBytes, float64(pre.ExpertActivationNanos)/1e9,
		dec.ExpertActivationDeviceCalls, dec.ExpertActivationHostCalls, dec.ExpertActivationReadbackBytes, float64(dec.ExpertActivationNanos)/1e9,
		pre.DenseProjectionDeviceCalls, pre.DenseProjectionDeviceRows, pre.DenseProjectionHostCalls, pre.DenseProjectionHostRows, pre.DenseProjectionActivationUploadBytes, pre.DenseProjectionReadbackBytes, float64(pre.DenseProjectionNanos)/1e9,
		dec.DenseProjectionDeviceCalls, dec.DenseProjectionDeviceRows, dec.DenseProjectionHostCalls, dec.DenseProjectionHostRows, dec.DenseProjectionActivationUploadBytes, dec.DenseProjectionReadbackBytes, float64(dec.DenseProjectionNanos)/1e9,
		compressorClause+indexerClause+combinedClause,
		pre.HeadProjectionDeviceCalls, pre.HeadProjectionDeviceRows, pre.HeadProjectionHostCalls, pre.HeadProjectionHostRows, pre.HeadProjectionActivationUploadBytes, pre.HeadProjectionReadbackBytes, float64(pre.HeadProjectionNanos)/1e9,
		dec.HeadProjectionDeviceCalls, dec.HeadProjectionDeviceRows, dec.HeadProjectionHostCalls, dec.HeadProjectionHostRows, dec.HeadProjectionActivationUploadBytes, dec.HeadProjectionReadbackBytes, float64(dec.HeadProjectionNanos)/1e9,
		pre.GroupedOutputDeviceCalls, pre.GroupedOutputDeviceRows, pre.GroupedOutputHostCalls, pre.GroupedOutputHostRows, pre.GroupedOutputMatMulCalls, pre.GroupedOutputActivationUploadBytes, pre.GroupedOutputReadbackBytes, pre.GroupedOutputHostWeightF32Bytes, float64(pre.GroupedOutputNanos)/1e9,
		dec.GroupedOutputDeviceCalls, dec.GroupedOutputDeviceRows, dec.GroupedOutputHostCalls, dec.GroupedOutputHostRows, dec.GroupedOutputMatMulCalls, dec.GroupedOutputActivationUploadBytes, dec.GroupedOutputReadbackBytes, dec.GroupedOutputHostWeightF32Bytes, float64(dec.GroupedOutputNanos)/1e9,
		pre.EngramProjectionDeviceCalls, pre.EngramProjectionDeviceRows, pre.EngramProjectionHostCalls, pre.EngramProjectionHostRows, pre.EngramProjectionMatMulCalls, pre.EngramProjectionActivationUploadBytes, pre.EngramProjectionReadbackBytes, pre.EngramProjectionHostWeightF32Bytes, float64(pre.EngramProjectionNanos)/1e9,
		dec.EngramProjectionDeviceCalls, dec.EngramProjectionDeviceRows, dec.EngramProjectionHostCalls, dec.EngramProjectionHostRows, dec.EngramProjectionMatMulCalls, dec.EngramProjectionActivationUploadBytes, dec.EngramProjectionReadbackBytes, dec.EngramProjectionHostWeightF32Bytes, float64(dec.EngramProjectionNanos)/1e9,
		pre.MHCProjectionDeviceCalls, pre.MHCProjectionDeviceRows, pre.MHCProjectionHostCalls, pre.MHCProjectionHostRows, pre.MHCProjectionMatMulCalls, pre.MHCProjectionActivationUploadBytes, pre.MHCProjectionReadbackBytes, pre.MHCProjectionHostWeightF32Bytes, float64(pre.MHCProjectionNanos)/1e9,
		dec.MHCProjectionDeviceCalls, dec.MHCProjectionDeviceRows, dec.MHCProjectionHostCalls, dec.MHCProjectionHostRows, dec.MHCProjectionMatMulCalls, dec.MHCProjectionActivationUploadBytes, dec.MHCProjectionReadbackBytes, dec.MHCProjectionHostWeightF32Bytes, float64(dec.MHCProjectionNanos)/1e9)
}

// executionIdentity makes the request log say which compute path actually produced
// the token. The summary prints q8dec only for a CPU session; a device turn names its
// executed host stages in host= instead (executionPlacementClauses).
// ExecutionIdentity reports the compute backend and forward path this planner
// serves on, the same pair the inkernel_chat summary line logs.
func (p *InKernelPlanner) ExecutionIdentity() (backend, forwardPath string) {
	return p.executionIdentity()
}

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
			// contract before this request can complete. Preserve the selected
			// backend's path instead of labeling every device as CUDA.
			if gdn, ok := p.backend.(interface{ Qwen35GDNPath() string }); ok {
				if path := gdn.Qwen35GDNPath(); model.IsSupportedQwen35GDNPath(path) {
					forwardPath = path
				}
			}
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

// generateReused runs prefill + decode for an already-encoded prompt, REUSING the longest
// cached KV prefix (the radix tree) when enabled and FAILING OPEN to a full prefill on a
// miss ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the candidate-#13 core, factored out of Complete so the reuse/decode path is
// exercisable on a synthetic model with no tokenizer.
//
// emit is invoked with each generated token id AFTER sampling and BEFORE the next Step;
// returning true stops decode with that token counted (Complete's string-suffix stop
// closes over the tokenizer there). A token-id stop (stops[next]) or next<0 ends decode
// WITHOUT emitting ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the served contract that a stop token is not echoed.
//
// SNAPSHOT/LEASE discipline: the full-prompt KV is snapshotted (Cloned) right after
// Prefill ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â BEFORE the decode loop mutates s.Cache by appending generated positions ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â and
// inserted under a FRESH Lookup so radixkv's lease handoff (LookupÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢InsertÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢Done) is honored
// entirely inside the lock, with no unexported *node escaping this scope. The reuse clone
// (SessionFromPrefix) is also taken under the lock, so a concurrent eviction of the tree
// node can never race our read of its KV. Returns the generated-token count, the prompt
// length, the reused-prefix length, prefill/decode seconds, and whether a stop (not maxNew)
// ended the turn.
