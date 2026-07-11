// Package guardvars holds the canonical wire shapes shared by the /debug/vars PRODUCER
// (internal/gateway) and the `fak info` CONSUMER (cmd/fak) for the blocks both sides once
// hand-copied field-for-field.
//
// Before this leaf, each of these blocks existed twice: once as an (unexported) emit struct in
// internal/gateway and once as a decode struct in cmd/fak. The two were kept "field-for-field"
// by hand, so a new field on the producer silently dropped on the floor at the consumer until a
// human noticed. Defining each shape ONCE here and aliasing both sides to it (see
// gateway/debug.go and cmd/fak/info.go) makes that drift a compile-time impossibility: there is
// one definition, and adding a field updates both surfaces at once.
//
// Scope note: only the blocks that are a TRUE field-for-field mirror live here. The pane
// deliberately decodes lean SUBSETS of some other blocks (e.g. it reads 3 of the gateway's 9
// Gateway fields), and those stay consumer-local by design — a shared type would force the pane
// to carry fields it never renders. This leaf is for the shapes both sides agree on in full.
//
// The struct tags carry the PRODUCER's `,omitempty` options. omitempty affects marshaling only
// (the producer's emit), never unmarshaling (the consumer's decode), so one definition serves
// both roles correctly. This package imports nothing: it is pure data.
package guardvars

// SessionVars is one /debug/vars sessions row: the main agent and any sub-agents it spawned,
// with the remaining budget axes and live wall-clock the `fak info` agents pane renders. A
// non-empty ParentTrace marks a sub-agent; Generation is its spawn depth. The budget fields are
// what REMAINS of the seeded allotment (0 usually means "never seeded"). LastTool/SpawnCount/
// InflightSeconds/IdleSeconds are the live-status activity cell (#2627): the last ADMITTED tool
// name (payload-free), the admitted subagent-spawn count, and the in-flight-OR-idle age of the
// trace — a row carries InflightSeconds or IdleSeconds, never both.
type SessionVars struct {
	TraceID           string `json:"trace_id"`
	Run               string `json:"run"`
	ParentTrace       string `json:"parent_trace,omitempty"`
	Generation        int    `json:"generation,omitempty"`
	Priority          int    `json:"priority,omitempty"`
	TurnsLeft         int    `json:"turns_left"`
	TokensLeft        int    `json:"tokens_left"`
	ContextTokensLeft int    `json:"context_tokens_left,omitempty"`
	ElapsedSeconds    int64  `json:"elapsed_seconds,omitempty"`
	Assumptions       int    `json:"assumptions,omitempty"`
	LastTool          string `json:"last_tool,omitempty"`
	SpawnCount        int    `json:"spawn_count,omitempty"`
	InflightSeconds   int64  `json:"inflight_seconds,omitempty"`
	IdleSeconds       int64  `json:"idle_seconds,omitempty"`
}

// CacheAttributionVars is the /debug/vars owner-split block (#1849): the same provider-vs-fak
// savings split the guard-exit banner prints, so an operator watching a session sees the same
// numbers, not a provider-only "saved X" that conflates the provider's default cache with
// fak-authored savings. Every token-equiv value is the same input-token currency as the vcache
// block; FakVDSOAvoidedCalls is a separate avoided-call counter (skipped engine calls, not
// prompt tokens).
type CacheAttributionVars struct {
	ProviderTokenEquiv float64 `json:"provider_token_equiv"` // net: read rebate minus write premium
	FakTokenEquiv      float64 `json:"fak_token_equiv"`      // compaction shed + in-kernel KV-prefix reuse
	TotalTokenEquiv    float64 `json:"total_token_equiv"`

	ProviderPromptCacheReadTokenEquiv         float64 `json:"provider_prompt_cache_read_token_equiv"`
	ProviderPromptCacheWritePremiumTokenEquiv float64 `json:"provider_prompt_cache_write_premium_token_equiv"` // negative until reads repay writes
	FakCompactionShedTokens                   uint64  `json:"fak_compaction_shed_tokens"`
	// FakCompactionCacheReadTokens is the OBSERVED provider cache_read at this session's compaction
	// fires — the warm witness FakTokenEquiv prices the shed on (min(shed, this) prices at the read
	// marginal, the cold remainder at full input). Carried so the `fak info` cache tab can EXPLAIN
	// why the shed's token-equiv is below its raw count on a warm session, not just report the net.
	FakCompactionCacheReadTokens uint64 `json:"fak_compaction_cache_read_tokens,omitempty"`
	FakKVPrefixReusedTokens      uint64 `json:"fak_kv_prefix_reused_tokens"`
	FakVDSOAvoidedCalls          uint64 `json:"fak_vdso_avoided_calls"` // avoided engine calls, NOT a token-equiv
}

// ManagedCacheVars is the /debug/vars managed-cache 1h TTL-upgrade POSTURE (#2190). Active is the
// resolved lever state independent of whether any head was eligible; Inert names the
// ACTIVE-but-inert misconfiguration signal (lever ON but every head refused, so a long idle
// session pays the 5m re-write the operator opted out of). Upgraded and Reasons mirror the
// AdjudicationSummary ttl-upgrade fields the durable ledger and the /metrics counter carry.
// Reasons is refusal-only (the "upgraded" outcome lives in Upgraded).
type ManagedCacheVars struct {
	Active   bool              `json:"active"`
	Inert    bool              `json:"inert"`
	Upgraded uint64            `json:"upgraded"`
	Reasons  map[string]uint64 `json:"reasons,omitempty"`
}
