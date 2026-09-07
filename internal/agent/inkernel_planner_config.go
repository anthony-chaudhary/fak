package agent

import (
	"fmt"
	"os"
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
	CPUOffloadExperts         bool
	QwenQ4KPrefillChunkTokens int
	Qwen35MetalGDNSequence    bool
	Q4KGateUpOutputSlab       bool
	DenseGPULayers            int
	CompactHistoryBudget      int
	ElideStaleReads           bool
	DeferColdTools            bool
}

// NewInKernelPlannerWithConfig is the explicit configuration constructor for native planning.
func NewInKernelPlannerWithConfig(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, cfg InKernelPlannerConfig) *InKernelPlanner {
	prefillChunkTokens, prefillChunkErr := resolveInKernelQwenQ4KPrefillChunkTokens(cfg.QwenQ4KPrefillChunkTokens)
	p := &InKernelPlanner{
		m:                            m,
		tok:                          tok,
		modelID:                      modelID,
		q4k:                          q4k,
		quant:                        true, // the served in-kernel path runs the Q8_0 forward (a quantized model)
		backend:                      backend,
		metal:                        metal,
		cpuOffloadExperts:            cfg.CPUOffloadExperts,
		denseGPULayers:               cfg.DenseGPULayers,
		maxNew:                       envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:                         envFloat("FAK_INKERNEL_TEMP", 0),
		seed:                         int64(envInt("FAK_INKERNEL_SEED", 0)),
		qwenQ4KPrefillChunkTokens:    prefillChunkTokens,
		qwenQ4KPrefillChunkConfigErr: prefillChunkErr,
		qwen35MetalGDNSequence:       cfg.Qwen35MetalGDNSequence,
		q4kGateUpOutputSlab:          cfg.Q4KGateUpOutputSlab,
		compactHistoryBudget:         cfg.CompactHistoryBudget,
		elideStaleReads:              cfg.ElideStaleReads,
		deferColdTools:               cfg.DeferColdTools,
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
			envInt("FAK_INKERNEL_RADIX_BUDGET", 0),
			envInt64("FAK_INKERNEL_RADIX_SNAPSHOT_BYTES", 0),
			envInt64("FAK_INKERNEL_RADIX_HOST_L2_BYTES", 0),
			inKernelRadixEvictionPolicyFromEnv(),
		)
		p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	}
	// The model-side KV-quarantine eviction bridge (#579) is OFF unless opted in, the same
	// default-off / fail-open posture as the ctxplan seam (FAK_CTXPLAN_SEAM). It runs over a
	// CPU model.Session, so like the radix tree it does not engage a device backend.
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_KVMMU"))) {
	case "on", "1", "true", "yes":
		p.kvSpanEvict = backend == nil
	}
	// Opt-in continuous-batch decode wiring (#401, L2). Default off keeps the serial
	// Session.Step loop; on routes decode through BatchSession.StepBatchActive (a batch of
	// one per request today, bit-identical to serial — see the batchDecode field note).
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKERNEL_BATCH"))) {
	case "on", "1", "true", "yes":
		p.batchDecode = true
	}
	return p
}

// RuntimeConfig returns the explicit native settings fixed at planner construction.
// It is a readback seam for gateway/operator reachability receipts; changing the returned
// value cannot mutate a running planner.
func (p *InKernelPlanner) RuntimeConfig() InKernelPlannerConfig {
	return InKernelPlannerConfig{
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

func inKernelRadixEvictionPolicyFromEnv() radixkv.EvictionPolicy {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_NATIVE_KV_VICTIM_RULE"))) {
	case "cost-aware", "cost", "kvbm":
		return radixkv.EvictionCostAware
	default:
		return radixkv.EvictionLRU
	}
}
