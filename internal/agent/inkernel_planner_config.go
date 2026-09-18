package agent

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// NewInKernelPlanner builds a planner over an already-loaded model + tokenizer.
// q4k flags a resident-Q4_K load so the decode engages Session.Q4K. Generation
// depth/sampling default to a greedy 256-token turn but are overridable via
// FAK_INKERNEL_MAX_TOKENS / FAK_INKERNEL_TEMP / FAK_INKERNEL_SEED.
func NewInKernelPlanner(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, cpuOffloadExpertsOpt ...bool) *InKernelPlanner {
	cpuOffloadExperts := false
	if len(cpuOffloadExpertsOpt) > 0 {
		cpuOffloadExperts = cpuOffloadExpertsOpt[0]
	}
	return NewInKernelPlannerWithConfig(m, tok, modelID, q4k, backend, metal, InKernelPlannerConfig{CPUOffloadExperts: cpuOffloadExperts})
}

// InKernelPlannerConfig carries settings that must be fixed at planner construction.
// Empty/zero fields preserve NewInKernelPlanner's historical defaults.
type InKernelPlannerConfig struct {
	// ContextTokens caps the total prompt plus planned decode positions accepted by
	// this planner. Zero uses the model's declared context window; when both are
	// known, the smaller bound wins.
	ContextTokens int
	// CPUCacheBytes caps retained native CPU KV payload; zero uses the environment.
	CPUCacheBytes int64
	// KVPrecision selects the realized storage tier of the per-request native KV
	// cache. Its zero value (and "") is model.KVPrecisionFP32: byte-for-byte the
	// historical f32 cache. model.KVPrecisionQ8_0 realizes the dense mixed layout
	// (Kraw f32 + q8_0 K/V) the planner's compute.KVPrecisionQ8 tier charges.
	KVPrecision               model.KVPrecision
	CPUOffloadExperts         bool
	QwenQ4KPrefillChunkTokens int
	Qwen35MetalGDNSequence    bool
	Q4KGateUpOutputSlab       bool
	DenseGPULayers            int
	CompactHistoryBudget      int
	ElideStaleReads           bool
	DeferColdTools            bool
	// RadixBudgetTokens bounds the process-scoped RadixAttention prefix cache to
	// this many cached tokens. A negative value keeps the historical unbounded
	// default (0 == unbounded); zero derives the bound from ContextTokens (see
	// resolveInKernelRadixBudgetTokens). It exists so a long-running resident
	// server (turnkey `fak up`) bounds the deep-chain footprint without an
	// operator env: radixkv stores the FULL-prefix KV per node, so an unbounded
	// tree accumulates nested KV clones on every turn of a growing conversation
	// (see the MEMORY NOTE on InKernelPlanner.tree). FAK_INKERNEL_RADIX_BUDGET
	// still wins whenever it is set, for callers that never set this field.
	RadixBudgetTokens int
	// BatchDecode opts this planner into the continuous-batch decode wiring
	// (#401/#1590) at construction time instead of relying on the process-wide
	// FAK_INKERNEL_BATCH env. It is the programmatic seam the turnkey `fak up`
	// path uses so N concurrent same-prefix requests are coalesced onto one
	// batched forward (coalescesQwenDecode) rather than serialized one-at-a-time
	// on devMu. Left false, decode is the historical serial Session.Step loop;
	// with it true, a B==1 fan-out is still bit-identical to serial (see the
	// batchDecode field note in inkernel_planner.go).
	BatchDecode bool
}

// NewInKernelPlannerWithConfig is the explicit configuration constructor for native planning.
func NewInKernelPlannerWithConfig(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, cfg InKernelPlannerConfig) *InKernelPlanner {
	prefillChunkTokens, prefillChunkErr := resolveInKernelQwenQ4KPrefillChunkTokens(cfg.QwenQ4KPrefillChunkTokens)
	prefillChunkExplicit := cfg.QwenQ4KPrefillChunkTokens != 0
	p := &InKernelPlanner{
		m:                            m,
		tok:                          tok,
		modelID:                      modelID,
		q4k:                          q4k,
		quant:                        true, // the served in-kernel path runs the Q8_0 forward (a quantized model)
		backend:                      backend,
		metal:                        metal,
		cpuOffloadExperts:            cfg.CPUOffloadExperts,
		kvPrecision:                  cfg.KVPrecision,
		contextTokens:                cfg.ContextTokens,
		denseGPULayers:               cfg.DenseGPULayers,
		maxNew:                       envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:                         envFloat("FAK_INKERNEL_TEMP", 0),
		seed:                         int64(envInt("FAK_INKERNEL_SEED", 0)),
		qwenQ4KPrefillChunkTokens:    prefillChunkTokens,
		qwenQ4KPrefillChunkExplicit:  prefillChunkExplicit,
		qwenQ4KPrefillChunkConfigErr: prefillChunkErr,
		qwen35MetalGDNSequence:       cfg.Qwen35MetalGDNSequence,
		q4kGateUpOutputSlab:          cfg.Q4KGateUpOutputSlab,
		compactHistoryBudget:         cfg.CompactHistoryBudget,
		elideStaleReads:              cfg.ElideStaleReads,
		deferColdTools:               cfg.DeferColdTools,
		batchDecode:                  cfg.BatchDecode,
	}
	if backend == nil && metal {
		m.PrepareMetalResidency(q4k)
	}
	// The GRADED expert spill (#5612, inkernel_expert_spill.go) is OFF unless the operator asks:
	// FAK_N_CPU_MOE=auto sizes it against the measured device budget, FAK_N_CPU_MOE=<N> states it.
	// Unset — every serve today — nothing is resolved and the placement stays exactly what
	// cpuOffloadExperts alone made it. Resolved HERE, once, because sizing walks every resident
	// tensor name and the device path builds a session per request.
	p.setExpertSpillFromEnv()
	// RadixAttention KV-prefix reuse is ON by default; FAK_INKERNEL_RADIX=off
	// disables it (the A/B "tree OFF" arm). Device reuse is admitted only for
	// architectures whose PrefixSnapshot owns every continuation byte: GLM's
	// host DSA state and Qwen3.5/3.6's attention plus recurrent backend state.
	if os.Getenv("FAK_INKERNEL_RADIX") != "off" && inKernelPlannerPrefixReuseSupported(m, backend) {
		p.tree = radixkv.NewWithTierBudgetsAndEvictionPolicy(
			resolveInKernelRadixBudgetTokens(cfg, contextTokensForRadixBudget(cfg, m)),
			envInt64("FAK_INKERNEL_RADIX_SNAPSHOT_BYTES", 0),
			envInt64("FAK_INKERNEL_RADIX_HOST_L2_BYTES", 0),
			inKernelRadixEvictionPolicyFromEnv(),
		)
		cpuBytes := cfg.CPUCacheBytes
		if cpuBytes == 0 {
			if value := os.Getenv("FAK_INKERNEL_RADIX_CPU_BYTES"); value != "" {
				var err error
				cpuBytes, err = strconv.ParseInt(value, 10, 64)
				if err != nil {
					panic("FAK_INKERNEL_RADIX_CPU_BYTES must be a non-negative byte count")
				}
			}
		}
		p.tree.SetCPUCacheByteBudget(cpuBytes)
		p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	}
	// The model-side KV-quarantine eviction bridge (#579) is OFF unless opted in, the same
	// default-off / fail-open posture as the ctxplan seam (FAK_CTXPLAN_SEAM). It runs over a
	// CPU model.Session, so like the radix tree it does not engage a device backend.
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_KVMMU"))) {
	case "on", "1", "true", "yes":
		p.kvSpanEvict = backend == nil
	}
	// Continuous-batch decode wiring (#401, L2; turnkey fan-out #1590). The
	// constructor may opt in structurally via cfg.BatchDecode (the turnkey
	// `fak up` path does), and the process-wide FAK_INKERNEL_BATCH env may opt in
	// or explicitly opt OUT (`off`) so an operator override always wins. Either
	// arm routes decode through BatchSession.StepBatchActive; at B==1 it is
	// bit-identical to serial — see the batchDecode field note.
	if strings.EqualFold(strings.TrimSpace(os.Getenv("FAK_INKERNEL_BATCH")), "off") {
		p.batchDecode = false
	} else {
		switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_BATCH"))) {
		case "on", "1", "true", "yes":
			p.batchDecode = true
		}
	}
	return p
}

// RuntimeConfig returns the explicit native settings fixed at planner construction.
// It is a readback seam for gateway/operator reachability receipts; changing the returned
// value cannot mutate a running planner.
func (p *InKernelPlanner) RuntimeConfig() InKernelPlannerConfig {
	var cpuBytes int64
	if p.tree != nil {
		cpuBytes = p.tree.CPUCacheByteBudget()
	}
	return InKernelPlannerConfig{
		ContextTokens:             p.contextTokens,
		CPUCacheBytes:             cpuBytes,
		KVPrecision:               p.kvPrecision,
		CPUOffloadExperts:         p.cpuOffloadExperts,
		QwenQ4KPrefillChunkTokens: p.qwenQ4KPrefillChunkTokens,
		Qwen35MetalGDNSequence:    p.qwen35MetalGDNSequence,
		Q4KGateUpOutputSlab:       p.q4kGateUpOutputSlab,
		DenseGPULayers:            p.denseGPULayers,
		CompactHistoryBudget:      p.compactHistoryBudget,
		ElideStaleReads:           p.elideStaleReads,
		DeferColdTools:            p.deferColdTools,
	}
}

func resolveInKernelQwenQ4KPrefillChunkTokens(tokens int) (int, *model.InKernelQwenQ4KPrefillChunkConfigError) {
	switch {
	case tokens == 0:
		return inKernelQwenQ4KPrefillChunkTokens, nil
	case tokens >= 128 && tokens <= 8192:
		return tokens, nil
	default:
		return 0, &model.InKernelQwenQ4KPrefillChunkConfigError{Value: fmt.Sprint(tokens)}
	}
}

func inKernelPlannerPrefixReuseSupported(m *model.Model, backend compute.Backend) bool {
	// Fail closed for a RECOMPUTE session before the host/device split (#5548). The gemma4
	// bridge keeps its prefix in the session's token history and leaves s.Cache empty, so the
	// snapshot this planner admits (step 3 of generateReusedContextWithBias) carries none of
	// it. Nothing downstream can notice: truncatePrefix returns a non-nil zero-length clone,
	// the nil-guard passes, and the tree matches on token ids — so a partial hit would prefill
	// only the divergent suffix against a session that never saw the prefix. Refusing here is
	// also honest about the saving: `matched` feeds the reused-vs-prompt witness line and the
	// KV-prefix KPI, and a recompute forward re-runs the whole prefix on every ingest, so any
	// non-zero reuse it reported would be a saving it never realized. Reuse returns when #5496
	// lands the cached path.
	if m != nil && !m.Cfg.KVPrefixReuseSupported() {
		return false
	}
	if backend == nil {
		return true
	}
	return m != nil && m.Cfg.InKernelBackendPrefixReuseSupported()
}

// kvPrefixEligiblePromptTokens is the eligibility witness for one served turn (#3391):
// how many of the turn's promptTok prefill tokens COULD have been served from the cached
// KV prefix. Two always-uncacheable cases book zero — a planner running without prefix
// reuse (no tree, or a backend that never reuses: no token can ever hit) and the
// always-cold first prefill (the kvPrefixEverAdmitted latch is still cold: nothing has
// been admitted, so nothing could match). Every other turn counts its whole prompt.
// Deliberately conservative: over-counting eligible can only UNDER-state the filtered
// ratio reused/eligible, and cacheobs clamps eligible up to the observed lookup match,
// so even a stale zero (a prewarmed tree serving the "first" turn) cannot push the ratio
// above 1. Must be sampled BEFORE noteKVPrefixAdmitted flips the latch for the turn.
func (p *InKernelPlanner) kvPrefixEligiblePromptTokens(promptTok int) int {
	if p.tree == nil || !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		return 0
	}
	if !p.kvPrefixEverAdmitted.Load() {
		return 0
	}
	return promptTok
}

// noteKVPrefixAdmitted flips the first-prefill latch after a successful reuse-enabled
// turn: generateReused's admission step has stored this turn's full prompt KV, so the
// NEXT turn's prompt is genuinely eligible to hit. A failed turn admits nothing and the
// latch stays cold; a reuse-disabled planner never flips it (its prompts stay ineligible
// forever, which is exactly true).
func (p *InKernelPlanner) noteKVPrefixAdmitted() {
	if p.tree == nil || !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		return
	}
	p.kvPrefixEverAdmitted.Store(true)
}

// contextTokensForRadixBudget picks the finite context ceiling the radix budget
// default is derived from: the planner's explicit ContextTokens when set,
// otherwise the model's declared context window. Both unknown (zero) means there
// is no honest finite ceiling to derive, so the caller falls back to the
// historical unbounded default via resolveInKernelRadixBudgetTokens.
func contextTokensForRadixBudget(cfg InKernelPlannerConfig, m *model.Model) int {
	configured := cfg.ContextTokens
	declared := 0
	if m != nil {
		declared = m.Cfg.MaxPositionEmbeddings
	}
	switch {
	case configured > 0 && declared > 0 && configured < declared:
		return configured
	case declared > 0:
		return declared
	case configured > 0:
		return configured
	default:
		return 0
	}
}

// resolveInKernelRadixBudgetTokens resolves the RadixAttention prefix-cache token
// budget with precedence:
//
//  1. FAK_INKERNEL_RADIX_BUDGET when set (>= 0) — the historical operator knob
//     still wins for every caller, so an explicit env cannot be silently
//     overridden by this new default.
//  2. cfg.RadixBudgetTokens when negative (keep-unbounded) or positive (explicit).
//  3. Otherwise, when a finite context ceiling is known, bound the tree to ONE
//     full context of cached prefix tokens. This is the leak fix: an unbounded
//     tree stores the FULL-prefix KV per node, so a resident server that serves a
//     long, growing conversation accumulates nested KV clones without limit. One
//     context's worth of cached prefix preserves the maximal-reuse win for the
//     common case (a static system+tool prefix reused turn after turn) while
//     making the deep-chain footprint bounded and predictable.
//  4. No finite ceiling known (contextTokens == 0): 0 == unbounded, preserving
//     the historical default for a planner that cannot state a bound.
//
// It is pure and independent of env for the returned-value witnesses; the only
// env read is the documented operator override in step 1.
func resolveInKernelRadixBudgetTokens(cfg InKernelPlannerConfig, contextTokens int) int {
	if v, ok := os.LookupEnv("FAK_INKERNEL_RADIX_BUDGET"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 0 {
			return n
		}
	}
	if cfg.RadixBudgetTokens < 0 {
		return 0
	}
	if cfg.RadixBudgetTokens > 0 {
		return cfg.RadixBudgetTokens
	}
	if contextTokens > 0 {
		return contextTokens
	}
	return 0
}

func inKernelRadixEvictionPolicyFromEnv() radixkv.EvictionPolicy {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_NATIVE_KV_VICTIM_RULE"))) {
	case "cost-aware", "cost", "kvbm":
		return radixkv.EvictionCostAware
	default:
		return radixkv.EvictionLRU
	}
}
