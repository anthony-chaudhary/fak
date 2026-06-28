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
	KV         KVConfig
	Scratch    TransformerScratchConfig
	MaxContext int // model's declared full window (MaxPositionEmbeddings); <=0 = unknown
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
	return append(plan, EstimateHALTransientMemoryPlan(c.Scratch)...)
}

// AutoSizeContextPlan is the single context auto-sizer every fit call site delegates to
// (#1049): given the model geometry, the weights/fixed demands the context shares memory
// with, the available memory ceiling, and an optional context-token override, it returns
// the context-token count to serve and the per-context memory plan sized to it.
//
// Token policy: a non-negative `override` is an explicit request and is used verbatim —
// the serve boot path passes the operator's --context-budget-tokens, the per-request
// planner passes its exact prompt+new count. A negative `override` means "not set": fall
// back to the model's declared full window (MaxContext).
//
// `weights` and `avail` are the inputs #1046's auto-fit-to-host policy reads to derive
// the largest context that fits when no override is set; they are threaded through every
// call site now so that policy can land in THIS one function without re-touching them.
// avail <= 0 means "unknown" (fail open to the full declared window).
func AutoSizeContextPlan(cfg ContextSizeConfig, weights MemoryPlan, avail int64, override int) (tokens int, plan MemoryPlan) {
	tokens = cfg.contextTokens(override, weights, avail)
	return tokens, cfg.PerContextMemoryPlan(tokens)
}

// contextTokens applies the auto-sizer's token policy. A non-negative override is taken
// verbatim; a negative override falls back to the full declared window. weights/avail are
// forwarded so the #1046 auto-fit derivation (largest context that fits avail − weights −
// scratch − headroom) lands here when no override is set; until then the full window is
// the historical behavior.
func (c ContextSizeConfig) contextTokens(override int, weights MemoryPlan, avail int64) int {
	if override >= 0 {
		return override // explicit operator/request count wins
	}
	if c.MaxContext <= 0 {
		return 0
	}
	// #1046 seam: when avail is known, derive the largest fitting context here using
	// weights + scratch + headroom. Unset today → the full declared window, unchanged.
	_ = weights
	_ = avail
	return c.MaxContext
}
