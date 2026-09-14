package compute

// ContextSizeConfig is the model geometry the context auto-sizer needs to turn a
// context-token count into a memory plan: the KV-store layout, the per-token HAL
// scratch geometry, and the model's declared full context window. It is the
// compute-level projection of a model Config — the serve boot path
// (cmd/fak/serve.go) and the in-kernel per-request planner
// (internal/agent/inkernel_planner.go) both map their model config into this and
// size their context plan through AutoSizeContextPlan, so the two cannot disagree on
// the same (model, host) inputs (#1049). Before this seam the boot path sized KV from
// MaxPositionEmbeddings while the per-request path sized from prompt+new, so boot could
// refuse at full ctx where a request would have fit; now both build the plan here.
type ContextSizeConfig struct {
	KV           KVConfig
	SessionState MemoryPlan // fixed context-independent state (for example a recurrent mixer)
	Scratch      TransformerScratchConfig
	MaxContext   int // model's declared full window (MaxPositionEmbeddings); <=0 = unknown
}

// PerContextMemoryPlan builds the per-context memory demands — the KV store sized to
// `tokens` cached positions plus the per-token HAL transient scratch — that both fit
// call sites share. Weight demands are arm-specific (device-lean / cpu-offload / f32 /
// resident) and stay caller-side; this owns only the context-sized portion. tokens <= 0
// omits the KV demand (scratch still applies), matching the fail-open behavior both call
// sites had before they delegated here.
func (c ContextSizeConfig) PerContextMemoryPlan(tokens int) MemoryPlan {
	var plan MemoryPlan
	if tokens > 0 {
		plan = append(plan, EstimateKVStoreMemoryPlan(c.KV, tokens)...)
	}
	plan = append(plan, c.SessionState...)
	return append(plan, EstimateHALTransientMemoryPlan(c.Scratch)...)
}

// AutoSizeContextPlan is the single context auto-sizer every fit call site delegates to
// (#1049): given the model geometry, the weights/fixed demands the context shares memory
// with, the available memory ceiling, and an optional context-token override, it returns
// the context-token count to serve and the per-context memory plan sized to it.
//
// Token policy: a non-negative `override` is an explicit request — the serve boot path
// passes the operator's --context-budget-tokens, the per-request planner passes its exact
// prompt+new count. When the memory ceiling is known and the model declares a window, an
// override LARGER than the largest context that provably fits is clamped DOWN to that
// fitted bound (#13036) instead of being taken verbatim into a plan the load-time fit check
// must fatally refuse; an override that fits stays verbatim. A negative `override` means
// "not set": fall back to the model's declared full window (MaxContext).
//
// `weights` and `avail` are the inputs #1046's auto-fit-to-host policy reads to derive
// the largest context that fits when no override is set; they are threaded through every
// call site now so that policy can land in THIS one function without re-touching them.
// avail <= 0 means "unknown" (fail open to the full declared window).
func AutoSizeContextPlan(cfg ContextSizeConfig, weights MemoryPlan, avail int64, override int) (tokens int, plan MemoryPlan) {
	tokens = cfg.contextTokens(override, weights, avail)
	return tokens, cfg.PerContextMemoryPlan(tokens)
}

// MinAutoContextTokens is the floor the #1046 auto-fit clamps a derived context to (capped at
// MaxContext for a model whose full window is already below it). When the box is so small that
// weights + scratch leave room for fewer than this many cached tokens, the auto-sizer still
// returns this floor rather than 0 — so the resulting plan keeps a usefully small KV demand and
// the LOAD-TIME fit check (RefuseMemoryPlanIfTooBig*) stays the single place a genuinely-too-small
// box is refused with a typed FitTooBig, instead of this sizer silently picking a zero context.
const MinAutoContextTokens = 512

// contextTokens applies the auto-sizer's token policy. An explicit override is taken
// verbatim when it fits (or the ceiling is unknown — fail-open; unknown capacity never
// rewrites an operator's request), and CLAMPED to the #1046 largest-fitting derivation
// when it would exceed it (#13036): never emit an over-maximal KV plan the load-time fit
// check must fatally refuse. A negative override falls back to the full declared window,
// or — when the memory ceiling is known — to that same largest-fitting derivation (the
// #13025 exported wrapper LargestFittingContextTokens shares it), so the unset-override
// path and the exported form cannot drift.
//
// Weights-overflow note: when the box cannot even hold the weights, largestFittingContext
// returns MinAutoContextTokens, and an explicit override is clamped to it. That is the
// intended policy — the sizer only lowers the context so the LOAD-TIME fit check
// (RefuseMemoryPlanIfTooBig*) stays the single place a genuinely-too-small box is refused
// with a typed FitTooBig — an over-maximal KV demand is never emitted ahead of it.
func (c ContextSizeConfig) contextTokens(override int, weights MemoryPlan, avail int64) int {
	if override < 0 {
		// #1046: no explicit budget. A negative (unset) override falls back to the full
		// declared window; with a known ceiling it derives the largest context that fits
		// (the exact continue the #13025 exported wrapper shares — they cannot drift).
		if c.MaxContext <= 0 {
			return 0
		}
		if avail <= 0 {
			return c.MaxContext // ceiling unprobeable → full declared window (historical behavior)
		}
		return c.largestFittingContext(weights, avail)
	}
	// Explicit override: verbatim when the ceiling is unknown (fail-open) or the model
	// declares no window to bound the derivation against.
	if avail <= 0 || c.MaxContext <= 0 {
		return override
	}
	// #13036: an explicitly requested window larger than the largest context that fits is
	// clamped DOWN to that fitted bound; anything at or below it is honored verbatim
	// (the clamp only ever shrinks — a small explicit request is never expanded).
	fit := c.largestFittingContext(weights, avail)
	if override <= fit {
		return override
	}
	return fit
}

// largestFittingContext returns the largest context-token count whose resident KV store plus
// per-token HAL scratch fits `avail` once the fixed `weights` already in that pool are
// subtracted — the #1046 auto-fit-to-host derivation that replaces the old "size against the
// full MaxContext window and refuse" fallback. `avail` is the headroom-adjusted budget the
// matching load-time fit check uses, and KV + scratch are device-pool demands, so only the
// device-scoped weights compete with them (a cpu-offload serve's host-resident experts do not —
// see weights.DeviceTotal). The result is clamped to [MinAutoContextTokens, MaxContext]: at the
// ceiling it is the full window (nothing to shrink); at the floor the plan stays small and the
// load-time fit check refuses a box too small to hold even that. Fail-open: KV geometry it
// cannot size yields the full window, never a refusal here.
func (c ContextSizeConfig) largestFittingContext(weights MemoryPlan, avail int64) int {
	perToken := EstimateKVStoreBytes(c.KV, 1)
	if perToken <= 0 {
		return c.MaxContext // cannot size KV → fail open to the full window
	}
	scratch := EstimateHALTransientMemoryPlan(c.Scratch).Total()
	fixed := c.SessionState.DeviceTotal()
	fit := (avail - weights.DeviceTotal() - fixed - scratch) / perToken
	floor := int64(MinAutoContextTokens)
	if floor > int64(c.MaxContext) {
		floor = int64(c.MaxContext) // a model whose full window is below the floor cannot exceed it
	}
	if fit < floor {
		return int(floor)
	}
	if fit >= int64(c.MaxContext) {
		return c.MaxContext
	}
	return int(fit)
}

// LargestFittingContextTokens is the exported form of the #1046 largest-fitting derivation
// (#13025), for refusal sites that must name "the maximum context that WOULD fit" without
// contradicting an unset-override AutoSizeContextPlan plan built from the same weights and
// budget. It carries the unset-override branch's fail-open verbatim: an unprobeable ceiling
// (avail <= 0) yields the full declared window — the fail-open contract — NOT a floor clamp,
// and a model with no declared window yields 0. A known ceiling routes through
// largestFittingContext, so the exported answer and the auto-sizer's are one derivation.
func LargestFittingContextTokens(cfg ContextSizeConfig, weights MemoryPlan, avail int64) int {
	if cfg.MaxContext <= 0 {
		return 0
	}
	if avail <= 0 {
		return cfg.MaxContext // ceiling unprobeable → full declared window (never a floor clamp)
	}
	return cfg.largestFittingContext(weights, avail)
}
