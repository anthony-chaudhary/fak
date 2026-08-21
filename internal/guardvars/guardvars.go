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

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// SessionVars is one /debug/vars sessions row: every session running through the gateway, with
// the remaining budget axes and live wall-clock the `fak info` agents pane renders.
//
// The row carries TWO different lineages, and conflating them misreports the fleet. ParentTrace
// is CONTINUATION lineage — internal/session writes it only from Table.Recontinue, "the trace
// this session was re-continued FROM" — so a non-empty ParentTrace marks the same agent after a
// hidden context reset (or a relay leg handoff), and Generation counts those re-continuations,
// NOT spawn depth. SpawnCount is the sub-agent axis: the admitted subagent-spawn count this
// trace issued. It is parent-side by construction; no field here identifies a row as somebody
// else's child, and the pane may not invent one from the continuation fields.
//
// The budget fields are what REMAINS of the seeded allotment (0 usually means "never seeded").
// LastTool/SpawnCount/InflightSeconds/IdleSeconds are the live-status activity cell (#2627): the
// last ADMITTED tool name (payload-free), the admitted subagent-spawn count, and the
// in-flight-OR-idle age of the trace — a row carries InflightSeconds or IdleSeconds, never both.
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
// TokenSavingsVars is the privacy-safe aggregate receipt block emitted by /debug/vars.
type TokenSavingsVars struct {
	NativeMCPFilter TokenSavingLever `json:"native_mcp_filter"`
	StaleReadElide  TokenSavingLever `json:"stale_read_elide"`
	ColdToolDefer   TokenSavingLever `json:"cold_tool_defer"`
	ModelRouting    TokenSavingLever `json:"model_routing"`
}

type TokenSavingLever struct {
	Fired          uint64  `json:"fired"`
	Units          uint64  `json:"units"`
	SavedBytes     uint64  `json:"saved_bytes,omitempty"`
	SavedTokens    uint64  `json:"saved_tokens,omitempty"`
	Evidence       string  `json:"evidence,omitempty"`
	Unavailable    string  `json:"unavailable_reason,omitempty"`
	Baseline       string  `json:"baseline,omitempty"`
	Fingerprint    string  `json:"compatibility_fingerprint,omitempty"`
	ModeledTokens  float64 `json:"modeled_input_tokens_delta,omitempty"`
	ModeledCalls   float64 `json:"modeled_model_calls_delta,omitempty"`
	ModeledSeconds float64 `json:"modeled_latency_seconds_delta,omitempty"`
}

type CacheAttributionVars struct {
	ProviderTokenEquiv float64 `json:"provider_token_equiv"` // net: read rebate minus write premium
	FakTokenEquiv      float64 `json:"fak_token_equiv"`      // compaction shed + in-kernel KV-prefix reuse
	TotalTokenEquiv    float64 `json:"total_token_equiv"`

	ProviderPromptCacheReadTokenEquiv         float64 `json:"provider_prompt_cache_read_token_equiv"`
	ProviderPromptCacheWritePremiumTokenEquiv float64 `json:"provider_prompt_cache_write_premium_token_equiv"` // negative until reads repay writes
	CacheCreationTokensHeadOnly               uint64  `json:"cache_creation_tokens_head_only,omitempty"`
	CacheCreationTokensMessagePrefix          uint64  `json:"cache_creation_tokens_message_prefix,omitempty"`
	FakCompactionShedTokens                   uint64  `json:"fak_compaction_shed_tokens"`
	// FakCompactionCacheReadTokens is the OBSERVED provider cache_read at this session's compaction
	// fires — the warm witness FakTokenEquiv prices the shed on (min(shed, this) prices at the read
	// marginal, the cold remainder at full input). Carried so the `fak info` cache tab can EXPLAIN
	// why the shed's token-equiv is below its raw count on a warm session, not just report the net.
	FakCompactionCacheReadTokens uint64 `json:"fak_compaction_cache_read_tokens,omitempty"`
	FakKVPrefixReusedTokens      uint64 `json:"fak_kv_prefix_reused_tokens"`
	FakVDSOAvoidedCalls          uint64 `json:"fak_vdso_avoided_calls"` // avoided engine calls, NOT a token-equiv
	FakResponseMemoCalls         uint64 `json:"fak_response_memo_calls"`
	FakInlineServedCalls         uint64 `json:"fak_inline_served_calls"`
	// FakDeferCold* surface the cold-tool-DEFER shed (#3647/#3232, the --defer-cold-tools lever) —
	// a THIRD shed mechanism distinct from the compaction shed above. FakDeferColdTurns/Count witness
	// how many turns fak deferred the cold tool tail and how many custom defs it marked defer_loading;
	// FakDeferColdToolNames names WHICH tools were deferred. Unlike the token-equiv fields above this
	// is NOT priced in tokens — defer shrinks no request bytes; the reduction is provider-side (only
	// the hot core loads into context) and OBSERVED in the usage relay — so the `fak info` Cache tab
	// renders it as its own informational line, never a token-savings bar. Absent when the lever is off.
	FakDeferColdTurns     uint64   `json:"fak_defer_cold_turns,omitempty"`
	FakDeferColdCount     uint64   `json:"fak_defer_cold_count,omitempty"`
	FakDeferColdToolNames []string `json:"fak_defer_cold_tool_names,omitempty"`
	// FakDeferFinding carries the #3621 live-watchdog verdict: FindingDeferEnabledButInert when
	// the cold-tool-defer lever was ARMED across enough eligible turns and deferred nothing on
	// any of them — the silent-identity failure mode the FakDeferCold* counters above cannot
	// express on their own (they are a numerator; a flat zero reads the same whether the lever
	// was off or on-and-inert). FakDeferStandDownTurns is that missing denominator: eligible
	// turns on which the transform ran and stood down. Both empty/absent on a healthy defer
	// session, on a lever-off session, and on a wire the transform never runs on, so the
	// finding field's PRESENCE is itself the alarm.
	FakDeferStandDownTurns uint64 `json:"fak_defer_stand_down_turns,omitempty"`
	FakDeferFinding        string `json:"fak_defer_finding,omitempty"`
}

// FindingDeferEnabledButInert is the CacheAttributionVars.FakDeferFinding value for the #3621
// watchdog: the cold-tool-defer lever armed, at least the eligible-turn floor accrued, zero cold
// definitions ever deferred. The guard exit banner prints the same token so the live pane and the
// exit artifact agree.
const FindingDeferEnabledButInert = "DEFER_ENABLED_BUT_INERT"

// WireOpenAIResponses is the resolved-upstream provider string for the OpenAI Responses
// (codex) wire (gateway.Config.Provider / guardManagedCachePosture.provider). That wire has
// no cache_control grammar and thus no 1h-TTL upgrade lever: the provider prefix cache is
// automatic and fak's managed-cache lever is the stable prompt_cache_key the outbound adapter
// pins. So on this wire an ACTIVE managed-cache posture with zero 1h upgrades is EXPECTED, not
// the #2190 ACTIVE-but-inert misconfiguration — the posture surfaces (Inert / "ACTIVE but
// inert" / POSTURE_MISMATCH) key on this string to stay wire-aware, mirroring
// guardManagedCachePosture.bannerLine's `provider == "openai-responses"` branch.
const WireOpenAIResponses = "openai-responses"

// ManagedCacheVars is the /debug/vars managed-cache 1h TTL-upgrade POSTURE (#2190). Active is the
// resolved lever state independent of whether any head was eligible; Inert names the
// ACTIVE-but-inert misconfiguration signal (lever ON but every head refused, so a long idle
// session pays the 5m re-write the operator opted out of). Upgraded and Reasons mirror the
// AdjudicationSummary ttl-upgrade fields the durable ledger and the /metrics counter carry.
// Reasons is refusal-only (the "upgraded" outcome lives in Upgraded).
//
// Wire is the resolved upstream provider (gateway.Config.Provider) the posture was formed on.
// The 1h-TTL upgrade lever Inert keys on is Anthropic-only; on the OpenAI Responses wire
// (WireOpenAIResponses) the real lever is the pinned prompt_cache_key, so a zero-upgrade ACTIVE
// session there is NOT inert. Empty (the historical default) preserves the Anthropic reading so
// existing callers and captured fixtures are unaffected. omitempty keeps the block byte-stable on
// the Anthropic wire.
type ManagedCacheVars struct {
	Active   bool              `json:"active"`
	Inert    bool              `json:"inert"`
	Upgraded uint64            `json:"upgraded"`
	Reasons  map[string]uint64 `json:"reasons,omitempty"`
	Wire     string            `json:"wire,omitempty"`
	// Finding carries the #3620 live-watchdog verdict: FindingUpgradeNeverFired when an
	// ACTIVE session on a wire that HAS the 1h-TTL lever accrued enough upgrade-eligible
	// turns with every one refused and zero "upgraded" outcomes — the lever's payoff never
	// arrived even though the posture claims it. Empty on a healthy, passive, short, or
	// lever-less session, so the field's presence is itself the alarm.
	Finding string `json:"finding,omitempty"`
}

// FindingUpgradeNeverFired is the ManagedCacheVars.Finding value for the #3620 watchdog:
// posture=ACTIVE, at least the attempt floor accrued, zero upgraded outcomes observed. The
// guard exit banner prints the same token so the live pane and the exit artifact agree.
const FindingUpgradeNeverFired = "UPGRADE_NEVER_FIRED"

// WireHasNo1hTTLLever reports whether the resolved wire lacks the Anthropic 1h-TTL upgrade
// lever — true only on the OpenAI Responses (codex) wire, where the managed-cache lever is the
// pinned prompt_cache_key and a zero 1h-upgrade count is expected rather than inert. Empty wire
// (the historical default) returns false, preserving Anthropic-wire behavior.
func (m ManagedCacheVars) WireHasNo1hTTLLever() bool {
	return m.Wire == WireOpenAIResponses
}

// The stable tokens for the three largest prompt-shrink levers (#5493). They are named here,
// beside the wire shapes, because TWO surfaces must agree on them byte-for-byte or a log scrape
// silently matches one and misses the other: the `fak serve` / `fak guard` startup admission
// (cmd/fak/shrink_lever_wire.go) and the live /debug/vars block below. A literal in each place
// would let them drift with nothing to catch it.
const (
	// ShrinkLeverCompactHistoryBudget — --compact-history-budget, stood down off the
	// passthrough by gateway.compactAnthropicRawWithReason.
	ShrinkLeverCompactHistoryBudget = "compact_history_budget"
	// ShrinkLeverElideStaleReads — --elide-stale-reads, stood down off the passthrough by
	// gateway.maybeElideStaleReads.
	ShrinkLeverElideStaleReads = "elide_stale_reads"
	// ShrinkLeverDeferColdTools — --defer-cold-tools, stood down off the passthrough by
	// gateway.maybeDeferColdTools.
	ShrinkLeverDeferColdTools = "defer_cold_tools"
)

// FindingShrinkLeverInertOnWire is the ShrinkLeverVars.Finding value — and the token the
// startup admission prints — for the #5493 condition: a prompt-shrink lever is configured ON
// and the wire this gateway actually built cannot run it, so the prompt forwards UNSHRUNK.
const FindingShrinkLeverInertOnWire = "SHRINK_LEVER_INERT_ON_WIRE"

// ShrinkLeverVars is the /debug/vars prompt-shrink-lever posture block (#5493): which of the
// three levers are configured ON, and which of those the LIVE wire can actually run.
//
// It exists because "configured" and "live" are different facts that every other surface
// conflates. The three levers are each gated, inside the gateway, on the Anthropic passthrough
// decision, so on any other wire all three stand down to identity — and all three ship
// default-ON. The per-turn counters next door in CacheAttributionVars cannot express that:
// FakDeferColdTurns and its DEFER_ENABLED_BUT_INERT watchdog only ever accrue PAST the
// eligibility gate, so on a wire the transform never runs they are flat zero, which reads
// exactly like a lever that was left off. This block is the missing denominator at the wire
// level: it names the levers whose absence would otherwise be invisible.
//
// The startup admission already refuses an EXPLICITLY enabled inert lever and prints a notice
// for a default-on one. This is the surface a long-running process, an A/B harness, or a
// scraped `/debug/vars` can read AFTER boot, when the startup line has scrolled away.
type ShrinkLeverVars struct {
	// WireRunsLevers is the gateway's own passthrough predicate (Server.anthropicPassthrough):
	// true only when this process fronts a single Anthropic-provider HTTP upstream, which is
	// the one wire on which req.Raw is forwarded verbatim and the three transforms can bite.
	WireRunsLevers bool `json:"wire_runs_levers"`
	// Wire is the resolved upstream provider (gateway.Config.Provider) this posture formed on.
	// The PROVIDER only, never the base URL — an operator-supplied upstream URL can carry an
	// embedded credential and can name a host that has no business on a debug surface.
	Wire string `json:"wire,omitempty"`
	// LiveOnWire lists the tokens of levers configured ON that this wire can run; InertOnWire
	// lists those configured ON that it cannot. Together they are the whole configured-ON set,
	// so a reader never has to guess whether an absent lever was off or merely unreported.
	//
	// "Live" here means REACHABLE — configured on, on a wire whose gate admits it. Whether a
	// reachable lever actually FIRED on a given turn is a separate, finer fact carried by the
	// CacheAttributionVars counters; this block deliberately does not claim it.
	LiveOnWire  []string `json:"live_on_wire,omitempty"`
	InertOnWire []string `json:"inert_on_wire,omitempty"`
	// DualLocalRouting warns that WireRunsLevers is a per-PROCESS answer to a per-TURN
	// question. In dual mode (an Anthropic upstream alongside in-kernel weights) a request
	// naming a locally-served model is not a passthrough, so it gets none of the three even
	// though this block reports the wire as running them. Present only when such a planner is
	// wired, so the caveat appears exactly where it applies — and that is also the deployment
	// most likely to be A/B'd local-vs-remote, where mistaking the two is most expensive.
	DualLocalRouting bool `json:"dual_local_routing,omitempty"`
	// Finding is FindingShrinkLeverInertOnWire when InertOnWire is non-empty, and empty
	// otherwise, so the field's PRESENCE is itself the alarm — the same convention
	// ManagedCacheVars.Finding and CacheAttributionVars.FakDeferFinding follow.
	Finding string `json:"finding,omitempty"`
}

// ObservationSchemaV1 is the first compatibility contract for typed observations.
const ObservationSchemaV1 = "fak-observation/1"

// Availability states whether data was measured and how a consumer must interpret it.
type Availability string

const (
	AvailabilityObserved      Availability = "OBSERVED"
	AvailabilityEmpty         Availability = "EMPTY"
	AvailabilityUnavailable   Availability = "UNAVAILABLE"
	AvailabilityStale         Availability = "STALE"
	AvailabilityNotApplicable Availability = "NOT_APPLICABLE"
)

// ObservationEnvelope carries observation provenance without conflating absent data with zero.
// Data remains raw JSON so each observation family can retain its own versioned payload.
type ObservationEnvelope struct {
	Schema       string          `json:"schema"`
	Source       string          `json:"source"`
	ObservedAt   string          `json:"observed_at,omitempty"`
	Revision     string          `json:"revision,omitempty"`
	Provenance   string          `json:"provenance"`
	Availability Availability    `json:"availability"`
	Data         json.RawMessage `json:"data,omitempty"`
	Reason       string          `json:"reason,omitempty"`
}

// Validate enforces the state laws. Unknown schema and availability values fail explicitly.
func (o ObservationEnvelope) Validate() error {
	if o.Schema != ObservationSchemaV1 {
		return fmt.Errorf("unsupported observation schema %q; expected %q", o.Schema, ObservationSchemaV1)
	}
	if strings.TrimSpace(o.Source) == "" || strings.TrimSpace(o.Provenance) == "" {
		return errors.New("observation source and provenance are required")
	}
	hasData := len(bytes.TrimSpace(o.Data)) > 0 && string(bytes.TrimSpace(o.Data)) != "null"
	hasReason := strings.TrimSpace(o.Reason) != ""
	switch o.Availability {
	case AvailabilityObserved:
		if !hasData || hasReason {
			return errors.New("OBSERVED requires data and forbids reason")
		}
	case AvailabilityEmpty:
		if hasData || hasReason {
			return errors.New("EMPTY forbids data and reason")
		}
	case AvailabilityUnavailable, AvailabilityStale:
		if hasData || !hasReason {
			return fmt.Errorf("%s requires reason and forbids data", o.Availability)
		}
	case AvailabilityNotApplicable:
		if hasData || !hasReason {
			return errors.New("NOT_APPLICABLE requires reason and forbids data")
		}
	default:
		return fmt.Errorf("unknown observation availability %q", o.Availability)
	}
	if o.ObservedAt == "" && o.Revision == "" {
		return errors.New("observation requires observed_at or revision")
	}
	return nil
}
