package agent

import (
	"errors"
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
	// DeepSeekV41ChatProfile opts into a pinned authenticated text contract.
	// Empty preserves legacy rendering; unknown values refuse prompt admission.
	DeepSeekV41ChatProfile string

	// WarmDiskModelIdentity binds restart snapshots to the loaded weight artifact.
	// Empty declines persistence because a display model ID is not a weight identity.
	WarmDiskModelIdentity string
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
	KVPrecision            model.KVPrecision
	CPUOffloadExperts      bool
	RequireDeviceExecution bool
	// DeviceOnlyBypass is the admitted bring-up bypass token. When set and
	// RequireDeviceExecution is false, one native session at a time runs host
	// stages; every other session stays device-only.
	DeviceOnlyBypass          string
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
	// V41EngramLookAhead opts serial Complete/CompleteStream forwards into bounded
	// Engram look-ahead. False preserves existing execution. Enabled requests
	// require BatchDecode=false, an attached V4.1 model and no speculation/EP.
	// Source eligibility and lifetime admission remain independent; cancellation
	// joins noninterruptible reads and is not a wall-time guarantee.
	V41EngramLookAhead bool
	// IncrementalContext opts this planner into the composed system-context
	// registry seam (inkernel_incremental.go): independently refreshable typed
	// context sources whose changes drive incremental context invalidation. It is
	// false by default, leaving the historical planner byte-for-byte inert (a nil
	// registry is the no-op identity). Enabling it is a construction-time seam;
	// sources are installed with SetIncrementalContext.
	IncrementalContext bool
}

// NewInKernelPlannerWithConfig is the explicit configuration constructor for native planning.
func NewInKernelPlannerWithConfig(m *model.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, cfg InKernelPlannerConfig) *InKernelPlanner {
	prefillChunkTokens, prefillChunkErr := resolveInKernelQwenQ4KPrefillChunkTokens(cfg.QwenQ4KPrefillChunkTokens)
	prefillChunkExplicit := cfg.QwenQ4KPrefillChunkTokens != 0
	seed, seedExplicit := inKernelSeedFromEnv()
	p := &InKernelPlanner{
		m:                            m,
		tok:                          tok,
		modelID:                      modelID,
		q4k:                          q4k,
		quant:                        true, // the served in-kernel path runs the Q8_0 forward (a quantized model)
		backend:                      backend,
		metal:                        metal,
		cpuOffloadExperts:            cfg.CPUOffloadExperts,
		requireDeviceExecution:       cfg.RequireDeviceExecution,
		deviceOnlyBypass:             cfg.DeviceOnlyBypass,
		kvPrecision:                  cfg.KVPrecision,
		contextTokens:                cfg.ContextTokens,
		denseGPULayers:               cfg.DenseGPULayers,
		maxNew:                       envInt("FAK_INKERNEL_MAX_TOKENS", 256),
		temp:                         envFloat("FAK_INKERNEL_TEMP", 0),
		seed:                         seed,
		seedExplicit:                 seedExplicit,
		qwenQ4KPrefillChunkTokens:    prefillChunkTokens,
		qwenQ4KPrefillChunkExplicit:  prefillChunkExplicit,
		qwenQ4KPrefillChunkConfigErr: prefillChunkErr,
		qwen35MetalGDNSequence:       cfg.Qwen35MetalGDNSequence,
		q4kGateUpOutputSlab:          cfg.Q4KGateUpOutputSlab,
		compactHistoryBudget:         cfg.CompactHistoryBudget,
		elideStaleReads:              cfg.ElideStaleReads,
		deferColdTools:               cfg.DeferColdTools,
		batchDecode:                  cfg.BatchDecode,
		v41EngramLookAhead:           cfg.V41EngramLookAhead,
		warmDiskModelIdentity:        strings.TrimSpace(cfg.WarmDiskModelIdentity),
		deepSeekV41ChatProfile:       cfg.DeepSeekV41ChatProfile,
	}
	if cfg.IncrementalContext {
		p.incrementalContext = EmptyIncrementalContext()
	}
	if m != nil {
		p.kvPrecision, p.kvPrecisionFallback = pickDeviceKVTier(m.Cfg, backend, cfg.KVPrecision)
		if p.kvPrecisionFallback != "" {
			fmt.Fprintln(os.Stderr, "fak inkernel:", p.kvPrecisionFallback)
		}
	}
	if backend == nil && metal {
		m.PrepareMetalResidency(q4k)
	}
	// fak#13567: snapshot the resident byte split now, before any request can decode, so a
	// /healthz probe never walks the store maps concurrently with decode-time store writes.
	p.WeightResidency()
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
		hostL2 := resolveInKernelRadixHostL2Bytes(os.LookupEnv, compute.HostSystemMemoryInfo)
		if hostL2.Err != nil {
			fmt.Fprintln(os.Stderr, hostL2.Err)
		}
		fmt.Fprintf(os.Stderr, "fak inkernel: radix host-DRAM L2 KV reserve=%d bytes source=%s\n", hostL2.Bytes, hostL2.Source)
		p.radixHostL2 = hostL2
		p.tree = radixkv.NewWithTierBudgetsAndEvictionPolicy(
			resolveInKernelRadixBudgetTokens(cfg, contextTokensForRadixBudget(cfg, m)),
			envInt64("FAK_INKERNEL_RADIX_SNAPSHOT_BYTES", 0),
			hostL2.Bytes,
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
		p.armRadixLocalSSD(m, backend, modelID)
		p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
		p.pagedPrefix = inKernelPagedPrefixLedgerFor(m, backend, p.tree)
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

// inKernelSeedFromEnv reads FAK_INKERNEL_SEED with envInt's parse rules and reports
// whether it was set: only a parsed value makes seedless sampling reproducible.
func inKernelSeedFromEnv() (int64, bool) {
	if v := os.Getenv("FAK_INKERNEL_SEED"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n, true
		}
	}
	return 0, false
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
		WarmDiskModelIdentity:     p.warmDiskModelIdentity,
		DeepSeekV41ChatProfile:    p.deepSeekV41ChatProfile,
		V41EngramLookAhead:        p.v41EngramLookAhead,
		ContextTokens:             p.contextTokens,
		CPUCacheBytes:             cpuBytes,
		KVPrecision:               p.kvPrecision,
		CPUOffloadExperts:         p.cpuOffloadExperts,
		RequireDeviceExecution:    p.requireDeviceExecution,
		DeviceOnlyBypass:          p.deviceOnlyBypass,
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
	if m == nil {
		return backend == nil
	}
	// V4.1 is the ONE architecture whose bare *KVCache is deliberately incomplete (it omits
	// the committed token history and per-layer temporal state), so the legacy bare-KV
	// predicate stays FALSE for it and it can never be admitted through that gate. It earns
	// planner reuse through the TWO complete-snapshot capabilities that the warming program
	// landed: the HOST route (#13338, HostCompletePrefixSnapshotSupported — a device-less
	// PrefixSnapshot carries the full v41ForwardSnapshot continuation state) and the DEVICE
	// route (#13334/#13338, InKernelBackendPrefixReuseSupportedFor — only the qualified
	// backend identity that witnessed the snapshot contract). Both are consulted with the
	// concrete route in hand, and both fail closed on an unqualified backend. This leaf
	// (#13335) is the planner-side consumer; it does not weaken the bare-KV refusal.
	if m.Cfg.IsDeepSeekV41() {
		if backend == nil {
			return m.Cfg.HostCompletePrefixSnapshotSupported()
		}
		return m.Cfg.InKernelBackendPrefixReuseSupportedFor(backend)
	}
	if !m.Cfg.KVPrefixReuseSupported() {
		return false
	}
	if backend == nil {
		return true
	}
	return m.Cfg.InKernelBackendPrefixReuseSupported()
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

// DefaultNativeOutputReserveTokens is the output reserve the native serve/turnkey
// path holds back from a resolved context window before spending any of it on
// resident history. It matches the 32K reserve the ctxplan envelopes use
// (internal/ctxplan/envelope.go), so the two budget layers agree on one number.
const DefaultNativeOutputReserveTokens = 32000

// NativeCompactResidentSharePercent is the share of the post-reserve window that
// may stay resident as compactible history. It encodes the long-context-defaults
// doctrine that "the model's advertised context window is a hard cap, not a
// target": the raw window is never the resident line, but a 1M-declared native
// model must not be flattened to the 48K/96K provider-shaped defaults either. At
// a 150K window this leaves ~70K resident, which is what makes a ~150k agent turn
// usable rather than a 400 wall.
const NativeCompactResidentSharePercent = 60

// DeriveCompactHistoryBudget is the single authority for the compaction shed-line
// on the NATIVE (in-kernel) path, given a resolved context window. It exists
// because InKernelPlannerConfig.CompactHistoryBudget previously had no production
// setter: both `fak serve` (serveNativePlannerConfigWithContext) and turnkey
// `fak up` (newTurnkeyInKernelPlanner) constructed the planner without it, so
// ApplyPromptShrink short-circuited on `compactBudget <= 0` and an over-window
// transcript hit refuseContextLength at full size -> HTTP 400
// context_length_exceeded with no second chance.
//
// Precedence and shape:
//
//   - requested > 0 wins verbatim (an explicit operator budget is never second-guessed).
//   - window <= 0 returns 0: no resolved window means no honest shed-line, and 0
//     preserves the historical no-compaction default rather than inventing a bound.
//   - a window at or below the reserve yields 0 as well; there is no resident
//     slack to spend, and a positive budget there would shed the live task itself.
//   - otherwise: (window - reserve) * share%, floored at
//     DefaultNativeOutputReserveTokens so a small-window model still gets a usable
//     line instead of a degenerate one.
//
// It is pure: same window in, same budget out, with no env or clock read, so the
// serve planner, the turnkey planner, and their tests all agree by construction.
func DeriveCompactHistoryBudget(windowTokens, requested int) int {
	if requested > 0 {
		return requested
	}
	if windowTokens <= DefaultNativeOutputReserveTokens {
		return 0
	}
	usable := windowTokens - DefaultNativeOutputReserveTokens
	budget := usable * NativeCompactResidentSharePercent / 100
	if budget < DefaultNativeOutputReserveTokens {
		return DefaultNativeOutputReserveTokens
	}
	return budget
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

// Host-DRAM L2 KV reserve defaults (perf-parity-20261029 umem TICKET-02): with no operator
// byte count, the radix tree may retain evicted prefixes in host DRAM up to this fraction of
// MemAvailable at planner construction instead of dropping them for a cold prefill.
const (
	inKernelRadixHostL2BytesEnv    = "FAK_INKERNEL_RADIX_HOST_L2_BYTES"
	inKernelRadixHostL2FractionEnv = "FAK_INKERNEL_RADIX_HOST_L2_FRACTION"
	inKernelRadixHostL2FractionDef = 0.15
	inKernelRadixHostL2FractionMin = 0.10
	inKernelRadixHostL2FractionMax = 0.20
	// inKernelRadixHostL2Margin is the MemAvailable the reserve never takes, matching serve's
	// host-margin floor for the expert ring (serveExpertRingHostMarginFloor).
	inKernelRadixHostL2Margin int64 = 6 << 30
)

// Sources of a resolved host L2 reserve, printed at planner construction.
const (
	InKernelRadixHostL2SourceEnv     = "env"
	InKernelRadixHostL2SourceDefault = "default 15% MemAvailable"
	InKernelRadixHostL2SourceUnknown = "unknown host memory"
	InKernelRadixHostL2SourceRefused = "refused"
)

// ErrInKernelRadixHostL2Config marks a host L2 env value the resolver refused; the reserve then
// fails closed to 0 (L2 off), today's behaviour.
var ErrInKernelRadixHostL2Config = errors.New("fak inkernel: invalid radix host L2 KV reserve")

// InKernelRadixHostL2 is the resolved host-DRAM L2 KV reserve and where it came from.
type InKernelRadixHostL2 struct {
	Bytes    int64
	Source   string
	Fraction float64
	Err      error
}

// resolveInKernelRadixHostL2Bytes sizes the radix tree's host-DRAM L2 tier:
//
//  1. FAK_INKERNEL_RADIX_HOST_L2_BYTES set (any value, including 0 = the OFF arm) is honoured
//     exactly; an unparseable or negative value refuses and fails closed to 0.
//  2. Otherwise, with host memory known, FAK_INKERNEL_RADIX_HOST_L2_FRACTION (default 0.15, in
//     [0.10, 0.20]; out of range refuses and fails closed to 0) x MemAvailable, clamped to
//     MemAvailable - inKernelRadixHostL2Margin and rounded down to MiB.
//  3. Unknown host memory: 0, today's default.
//
// The reserve is a ceiling, not an allocation: the tier holds bytes only for evicted prefixes.
func resolveInKernelRadixHostL2Bytes(lookupEnv func(string) (string, bool), hostMemoryInfo func() (int64, int64, bool)) InKernelRadixHostL2 {
	if lookupEnv != nil {
		if v, ok := lookupEnv(inKernelRadixHostL2BytesEnv); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil || n < 0 {
				return InKernelRadixHostL2{Source: InKernelRadixHostL2SourceRefused, Err: fmt.Errorf("%w: %s=%q: want a non-negative byte count", ErrInKernelRadixHostL2Config, inKernelRadixHostL2BytesEnv, v)}
			}
			return InKernelRadixHostL2{Bytes: n, Source: InKernelRadixHostL2SourceEnv}
		}
	}
	fraction := inKernelRadixHostL2FractionDef
	if lookupEnv != nil {
		if v, ok := lookupEnv(inKernelRadixHostL2FractionEnv); ok && strings.TrimSpace(v) != "" {
			f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil || !(f >= inKernelRadixHostL2FractionMin && f <= inKernelRadixHostL2FractionMax) {
				return InKernelRadixHostL2{Source: InKernelRadixHostL2SourceRefused, Err: fmt.Errorf("%w: %s=%q: want a fraction in [%g, %g]", ErrInKernelRadixHostL2Config, inKernelRadixHostL2FractionEnv, v, inKernelRadixHostL2FractionMin, inKernelRadixHostL2FractionMax)}
			}
			fraction = f
		}
	}
	if hostMemoryInfo == nil {
		return InKernelRadixHostL2{Source: InKernelRadixHostL2SourceUnknown}
	}
	_, avail, known := hostMemoryInfo()
	if !known || avail <= 0 {
		return InKernelRadixHostL2{Source: InKernelRadixHostL2SourceUnknown}
	}
	reserve := min(int64(fraction*float64(avail)), avail-inKernelRadixHostL2Margin)
	reserve = max(reserve, 0) &^ (1<<20 - 1)
	source := InKernelRadixHostL2SourceDefault
	if fraction != inKernelRadixHostL2FractionDef {
		source = fmt.Sprintf("%s=%g x MemAvailable", inKernelRadixHostL2FractionEnv, fraction)
	}
	return InKernelRadixHostL2{Bytes: reserve, Source: source, Fraction: fraction}
}

// RadixHostL2 reports the host-DRAM L2 KV reserve this planner resolved at construction; the
// zero value when the radix tree is off.
func (p *InKernelPlanner) RadixHostL2() InKernelRadixHostL2 {
	return p.radixHostL2
}

func inKernelRadixEvictionPolicyFromEnv() radixkv.EvictionPolicy {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_NATIVE_KV_VICTIM_RULE"))) {
	case "cost-aware", "cost", "kvbm":
		return radixkv.EvictionCostAware
	default:
		return radixkv.EvictionLRU
	}
}
