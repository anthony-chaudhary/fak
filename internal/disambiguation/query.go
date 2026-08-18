package disambiguation

import (
	"errors"
	"fmt"
)

// QuerySchemaVersion identifies the canonical-identity query response contract.
const QuerySchemaVersion = "fak-disambiguation-query/1"

// PublicIndexVersion identifies the immutable public seed set used by this reader.
// The later generator may replace the seed while retaining this read contract.
const PublicIndexVersion = "public-seed/1"

// ErrCanonicalTermNotFound reports that an exact canonical-term lookup missed.
var ErrCanonicalTermNotFound = errors.New("canonical term not found")

// ErrScopeRequired reports that a token has multiple scoped owners and cannot
// be resolved safely without an exact scope qualifier.
var ErrScopeRequired = errors.New("scope required for overloaded term")

// QueryResponse is the versioned, machine-readable result of a lookup. Entry
// always exposes the canonical owner. MatchedAlias is populated only when an
// exact declared alias selected that owner and preserves the caller's spelling.
type QueryResponse struct {
	Schema       string `json:"schema"`
	IndexVersion string `json:"index_version"`
	MatchedAlias string `json:"matched_alias,omitempty"`
	Entry        Entry  `json:"entry"`
}

// Query performs an exact, case-sensitive lookup of a canonical term. It stays
// canonical-only so callers that require canonical ownership cannot silently
// broaden their lookup to aliases.
func Query(canonicalTerm string) (QueryResponse, error) {
	entry, ok, ambiguous := publicIndex.queryCanonical(canonicalTerm)
	if ambiguous {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrScopeRequired, canonicalTerm)
	}
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, canonicalTerm)
	}
	return queryResponse(entry, ""), nil
}

// Resolve performs an exact, case-sensitive lookup across canonical terms and
// declared aliases. The returned entry always carries the canonical identity;
// MatchedAlias records the exact alias used and is empty for canonical input.
func Resolve(term string) (QueryResponse, error) {
	entry, matchedAlias, ok := publicIndex.resolve(term)
	if publicIndex.ambiguous(term) {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrScopeRequired, term)
	}
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q", ErrCanonicalTermNotFound, term)
	}
	return queryResponse(entry, matchedAlias), nil
}

// QueryScoped performs an exact canonical lookup constrained by the required
// scope qualifier. The entry returns the stored scope unchanged.
func QueryScoped(canonicalTerm string, scope Scope) (QueryResponse, error) {
	return resolveScoped(canonicalTerm, scope, false)
}

// ResolveScoped performs an exact canonical-or-alias lookup constrained by the
// required scope qualifier. The entry returns the stored scope unchanged.
func ResolveScoped(term string, scope Scope) (QueryResponse, error) {
	return resolveScoped(term, scope, true)
}

func resolveScoped(term string, scope Scope, allowAlias bool) (QueryResponse, error) {
	if err := requireText("scope.kind", scope.Kind); err != nil {
		return QueryResponse{}, err
	}
	if err := requireText("scope.value", scope.Value); err != nil {
		return QueryResponse{}, err
	}
	var entry Entry
	var matchedAlias string
	var ok bool
	if allowAlias {
		entry, matchedAlias, ok = publicIndex.resolveScoped(term, scope)
	} else {
		for _, candidate := range publicIndex.canonical[term] {
			if candidate.Scope == scope {
				entry, ok = cloneEntry(candidate), true
				break
			}
		}
	}
	if !ok {
		return QueryResponse{}, fmt.Errorf("%w: %q in scope %s=%q", ErrCanonicalTermNotFound, term, scope.Kind, scope.Value)
	}
	return queryResponse(entry, matchedAlias), nil
}

func queryResponse(entry Entry, matchedAlias string) QueryResponse {
	return QueryResponse{
		Schema:       QuerySchemaVersion,
		IndexVersion: PublicIndexVersion,
		MatchedAlias: matchedAlias,
		Entry:        entry,
	}
}

var publicEntries = []Entry{
	{
		Schema: EntrySchemaVersion,
		Identity: Identity{
			CanonicalTerm: "agent kernel",
			Aliases:       []string{"fused agent kernel"},
		},
		Definition: "The fak management boundary that governs model traffic, tool effects, context, and recovery.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "compute kernel",
			Explanation:         "An arithmetic routine executed by a processor; it does not govern an agent's tool effects.",
			RequiredPair:        boolPointer(true),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope: Scope{Kind: "product", Value: "fak"},
		Owner: Owner{Leaf: "kernel", Lane: "kernel"},
		Sources: []SourceWitness{{
			Kind: "document", Locator: "README.md#how-it-works", Revision: "692e4b57d0",
			CheckedAt: "2026-08-11T00:00:00Z", Probe: "fak-disambiguation-seed",
		}},
		Freshness: Freshness{
			Verdict:    "fresh",
			ReasonCode: "SOURCE_CURRENT",
			CheckedAt:  "2026-08-11T00:00:00Z",
			Probe:      "public-seed/1",
		},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "compute kernel", Aliases: []string{}},
		Definition: "An arithmetic routine executed by a processor.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "agent kernel",
			Explanation:         "The fak management boundary governs agent behavior; it is not a processor arithmetic routine.",
			RequiredPair:        boolPointer(true),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "computing", Value: "processor"},
		Owner:     Owner{Leaf: "kernel", Lane: "kernel"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "README.md#how-it-works", Revision: "692e4b57d0", CheckedAt: "2026-08-11T00:00:00Z", Probe: "fak-disambiguation-seed"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "kernel", Aliases: []string{}},
		Definition: "The internal/disambiguation Go package that validates and queries public terminology records.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "fak CLI kernel",
			Explanation:         "The package API is not the fak command-line product surface.",
			RequiredPair:        boolPointer(false),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "package", Value: "internal/disambiguation"},
		Owner:     Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "internal/disambiguation/README.md", Revision: "public-seed/1", CheckedAt: "2026-08-11T00:00:00Z", Probe: "fak-disambiguation-seed"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "kernel", Aliases: []string{}},
		Definition: "The fak command-line product surface for operating the agent kernel.",
		Contrasts: []Contrast{{
			CanonicalTerm:       "disambiguation package",
			Explanation:         "The command-line product surface is not the internal Go package API.",
			RequiredPair:        boolPointer(false),
			ForbiddenConflation: boolPointer(true),
		}},
		Scope:     Scope{Kind: "cli", Value: "fak"},
		Owner:     Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "document", Locator: "README.md#how-it-works", Revision: "public-seed/1", CheckedAt: "2026-08-11T00:00:00Z", Probe: "fak-disambiguation-seed"}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"},
		Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "fak CLI kernel", Aliases: []string{}},
		Definition: "The fak command-line product surface, named as a contrast target for the package-scoped kernel entry.",
		Contrasts:  []Contrast{{CanonicalTerm: "kernel", Explanation: "The CLI surface and package-scoped kernel are distinct.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "cli", Value: "fak"}, Owner: Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "go-source", Locator: "cmd/fak/main.go", Revision: "public-seed/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-seed", Reference: &PublicReference{Kind: ReferenceKindCLIVerb, Name: "disambiguation"}}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"}, Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "disambiguation package", Aliases: []string{}},
		Definition: "The internal/disambiguation package, named as a contrast target for the CLI-scoped kernel entry.",
		Contrasts:  []Contrast{{CanonicalTerm: "kernel", Explanation: "The package and CLI-scoped kernel are distinct.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "package", Value: "internal/disambiguation"}, Owner: Owner{Leaf: "disambiguation", Lane: "canon"},
		Sources:   []SourceWitness{{Kind: "go-source", Locator: "internal/disambiguation/query.go", Revision: "public-seed/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-seed", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Query"}}},
		Freshness: Freshness{Verdict: "fresh", ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-11T00:00:00Z", Probe: "public-seed/1"}, Lifecycle: Lifecycle{Class: "current", Rollout: "on"},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "agent session", Aliases: []string{"session"}},
		Definition: "A durable, addressable agent execution record carrying drive state and pointers without storing the provider transcript.",
		Contrasts:  []Contrast{{CanonicalTerm: "session resume", Explanation: "The session is the durable execution identity; resume is one transition that re-admits a paused session.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "internal/session"}, Owner: Owner{Leaf: "session", Lane: "session"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/session/descriptor.go", Revision: "session-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-session-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Descriptor"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "session-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "session resume", Aliases: []string{"resume"}},
		Definition: "The paused-to-running boundary that re-admits an existing session using warm KV when available or a safe cold re-prefill.",
		Contrasts: []Contrast{
			{CanonicalTerm: "agent session", Explanation: "Resume changes the run state of an existing session; it is not the session identity or transcript.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "session recovery", Explanation: "Resume continues a valid paused session; recovery repairs or reroutes state that cannot safely continue as-is.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "runtime", Value: "internal/session"}, Owner: Owner{Leaf: "session", Lane: "session"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/session/resume.go", Revision: "session-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-session-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "ResumeMode"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "session-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "session recovery", Aliases: []string{"recovery"}},
		Definition: "A bounded repair or reroute response when persisted or cumulative session state cannot safely continue unchanged.",
		Contrasts: []Contrast{
			{CanonicalTerm: "session resume", Explanation: "Recovery responds to corrupt or over-envelope state; resume merely re-admits a valid paused session.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "recovery checkpoint", Explanation: "Recovery is the repair action; a recovery checkpoint is the structured continuation state handed to that action.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "runtime", Value: "internal/session"}, Owner: Owner{Leaf: "session", Lane: "session"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/session/quarantine.go", Revision: "session-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-session-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "RecoveryEvent"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "session-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "context compaction", Aliases: []string{"compaction"}},
		Definition: "A context-window event that replaces prior history so resident input falls while cumulative usage and transcript bytes may continue rising.",
		Contrasts:  []Contrast{{CanonicalTerm: "recovery checkpoint", Explanation: "Compaction reduces resident model context; a recovery checkpoint preserves typed continuation state for rerouting or repair.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "codex-context"}, Owner: Owner{Leaf: "session", Lane: "session"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/session/compactaudit.go", Revision: "session-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-session-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "CompactSessionReport"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "session-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "recovery checkpoint", Aliases: []string{"checkpoint"}},
		Definition: "A typed snapshot of goal, pending turn, continuation, generation, and state revision emitted when session recovery is requested.",
		Contrasts: []Contrast{
			{CanonicalTerm: "context compaction", Explanation: "The checkpoint preserves control-plane continuation state; compaction rewrites model-visible history to reduce resident context.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "session recovery", Explanation: "The checkpoint is evidence and continuation data for recovery, not the repair or reroute action itself.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "runtime", Value: "internal/session"}, Owner: Owner{Leaf: "session", Lane: "session"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/session/cumulative_envelope.go", Revision: "session-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-session-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "SessionRecoveryCheckpoint"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "session-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "tool-result cache", Aliases: []string{"vDSO cache"}},
		Definition: "A fak-owned cache of completed tool-call results keyed by tool, argument hash, principal when isolated, and world-version epochs.",
		Contrasts: []Contrast{
			{CanonicalTerm: "model KV cache", Explanation: "The tool-result cache stores tool outputs and invalidates on modeled world changes; the KV cache stores per-token attention tensors.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "radix prefix cache", Explanation: "The tool-result cache looks up effect-safe tool calls; the radix cache longest-prefix-matches token sequences to reusable model snapshots.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "provider prompt cache", Explanation: "The tool-result cache is kernel-owned and directly invalidated by fak epochs; provider prompt-cache entries are externally owned and observed through usage accounting.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "cache", Value: "tool-results"}, Owner: Owner{Leaf: "vdso", Lane: "vdso"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/vdso/vdso.go", Revision: "cache-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-cache-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "VDSO"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "cache-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "model KV cache", Aliases: []string{"KV cache", "KVCache"}},
		Definition: "Kernel-owned per-layer attention key/value tensors indexed by token position and invalidated or rewritten when the model sequence changes.",
		Contrasts: []Contrast{
			{CanonicalTerm: "tool-result cache", Explanation: "The KV cache contains model attention state, not completed tool outputs or world-versioned effects.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "radix prefix cache", Explanation: "A KV cache is one sequence's live attention state; the radix cache indexes token prefixes and references reusable snapshots across requests.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "provider prompt cache", Explanation: "The model KV cache is directly owned and mutable inside fak; provider prompt caching is an upstream reuse service exposed as billed token axes.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "cache", Value: "model-attention"}, Owner: Owner{Leaf: "model", Lane: "model"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/model/kvcache.go", Revision: "cache-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-cache-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "KVCache"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "cache-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "radix prefix cache", Aliases: []string{"radix cache", "prefix cache"}},
		Definition: "A fak-owned radix tree that longest-prefix-matches namespaced token sequences to reusable KV snapshots under token and byte budgets.",
		Contrasts: []Contrast{
			{CanonicalTerm: "tool-result cache", Explanation: "The radix cache keys token-prefix paths and snapshot residency; the tool-result cache keys tool calls plus effect epochs.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "model KV cache", Explanation: "The radix cache is a multi-prefix lookup and residency index; a model KV cache is the live tensor state for one sequence.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "provider prompt cache", Explanation: "The radix cache is namespace-scoped, budgeted, and evicted by fak; provider prompt caching is controlled upstream and reported through cache read/write usage.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "cache", Value: "radix-prefix-snapshots"}, Owner: Owner{Leaf: "radixkv", Lane: "radixkv"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/radixkv/radixkv.go", Revision: "cache-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-cache-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Tree"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "cache-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "provider prompt cache", Aliases: []string{"provider cache", "prompt cache"}},
		Definition: "An upstream provider-owned prompt-prefix reuse service observed as cache-read and cache-creation token accounting with provider TTL and pricing rules.",
		Contrasts: []Contrast{
			{CanonicalTerm: "tool-result cache", Explanation: "Provider prompt caching reuses model input prefixes outside fak; the tool-result cache locally serves completed tool outputs under effect-aware invalidation.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "model KV cache", Explanation: "Provider prompt cache state is not directly addressable tensor memory in fak and cannot be edited like the kernel-owned KV cache.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "radix prefix cache", Explanation: "Provider cache lifetime and identity are upstream contracts; the radix prefix cache is fak-owned, namespace-keyed, and explicitly budgeted and evicted.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "cache", Value: "provider-prompt-prefix"}, Owner: Owner{Leaf: "gateway", Lane: "gateway"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/gateway/cache_pricing.go", Revision: "cache-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-cache-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "CacheUsage"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "cache-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "ABI refusal reason", Aliases: []string{"POLICY_BLOCK", "refusal reason"}},
		Definition: "A closed trainable ReasonCode explaining why an adjudication refused a call; POLICY_BLOCK means an explicit policy rule denied it.",
		Contrasts: []Contrast{
			{CanonicalTerm: "policy posture verdict", Explanation: "A refusal reason explains a tool-call decision; a policy posture verdict is the ALLOW/DENY result of folding organization amendment authority.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "hook gate class", Explanation: "ReasonCode labels adjudication semantics; a hook gate class declares which tree surface a hook may mutate.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "DOS decision kind", Explanation: "A refusal reason is a closed ABI label; a DOS decision kind classifies an operator queue row and its resolver lifecycle.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "vocabulary", Value: "abi-reason"}, Owner: Owner{Leaf: "abi", Lane: "abi"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/abi/reasons.go", Revision: "reason-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-reason-source", Reference: &PublicReference{Kind: ReferenceKindReasonCode, Name: "POLICY_BLOCK"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "reason-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "policy posture verdict", Aliases: []string{"organization amendment verdict"}},
		Definition: "The ALLOW or DENY result of folding compiled, environment, and organization authority over a policy amendment.",
		Contrasts: []Contrast{
			{CanonicalTerm: "ABI refusal reason", Explanation: "The posture verdict is the decision result; POLICY_BLOCK is one explanatory reason attached to a separate tool-call refusal vocabulary.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "hook gate class", Explanation: "ALLOW/DENY describes policy authority, not whether a hook lands the tree or operates in a worktree twin.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "DOS decision kind", Explanation: "A posture verdict resolves one policy fold; a DOS decision kind categorizes persistent arbitration or operator work.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "vocabulary", Value: "policy-verdict"}, Owner: Owner{Leaf: "policy", Lane: "policy"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/policy/orgprecedence.go", Revision: "reason-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-reason-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "OrgAmendVerdict"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "reason-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "hook gate class", Aliases: []string{"LANDS_TREE"}},
		Definition: "A hook-runner classification declaring whether a gate mutates the index/worktree, intentionally uses a worktree, or is a tree-twin checker.",
		Contrasts: []Contrast{
			{CanonicalTerm: "ABI refusal reason", Explanation: "LANDS_TREE declares hook execution scope; it is not a reason why a model tool call was refused.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "policy posture verdict", Explanation: "A hook class schedules and isolates gate execution; it does not grant or deny organization policy authority.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "DOS decision kind", Explanation: "A hook class is local runner metadata; a DOS decision kind is a persisted arbitration or resolver category.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "vocabulary", Value: "hook-gate-class"}, Owner: Owner{Leaf: "hooks", Lane: "hooks"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/hooks/gatescope.go", Revision: "reason-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-reason-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "ClassLandsTree"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "reason-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "DOS decision kind", Aliases: []string{"ARBITER_REFUSE"}},
		Definition: "A persistent DOS row category identifying arbitration refusal work whose resolution depends on the current lane-lease state.",
		Contrasts: []Contrast{
			{CanonicalTerm: "ABI refusal reason", Explanation: "ARBITER_REFUSE classifies a DOS decision row; it is not a ReasonCode in the kernel adjudication ABI.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "policy posture verdict", Explanation: "A DOS decision kind drives resolver lifecycle and history, not an ALLOW/DENY policy amendment result.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)},
			{CanonicalTerm: "hook gate class", Explanation: "A DOS decision kind persists arbitration state; a hook gate class controls execution isolation for a checker.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)},
		},
		Scope: Scope{Kind: "vocabulary", Value: "dos-decision-kind"}, Owner: Owner{Leaf: "dosdecision", Lane: "dos"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/dosdecision/revalidate.go", Revision: "reason-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-reason-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "KindArbiterRefuse"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "reason-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "runtime", Aliases: []string{"agent application runtime"}},
		Definition: "The host-side agent application loop that turns model completions into tool calls and final answers through the Planner seam.",
		Contrasts:  []Contrast{{CanonicalTerm: "agent kernel", Explanation: "The agent application runtime drives the task loop; the agent kernel mediates its model and tool effects at the enforcement boundary.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "agent-application"}, Owner: Owner{Leaf: "agent", Lane: "agent"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/agent/chat.go", Revision: "runtime-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-runtime-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Planner"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "runtime-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "runtime", Aliases: []string{"gateway serving runtime"}},
		Definition: "The configured gateway server that exposes HTTP or MCP transport, authentication, routing, kernel mediation, and observability.",
		Contrasts:  []Contrast{{CanonicalTerm: "fak CLI kernel", Explanation: "The gateway runtime is a long-lived transport server; the fak CLI kernel is the command surface used to configure and launch it.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "gateway-serving"}, Owner: Owner{Leaf: "gateway", Lane: "gateway"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/gateway/gateway.go", Revision: "runtime-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-runtime-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Server"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "runtime-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "runtime", Aliases: []string{"guard enforcement runtime"}},
		Definition: "The wrapper process that launches a guest command under fak policy, hook, capability, and stop-gate enforcement.",
		Contrasts:  []Contrast{{CanonicalTerm: "agent session", Explanation: "The guard runtime enforces a launched process; an agent session is the durable execution identity and state pointers that may outlive one wrapper process.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "guard-enforcement"}, Owner: Owner{Leaf: "guard", Lane: "guard"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "cmd/fak/guard.go", Revision: "runtime-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-runtime-source", Reference: &PublicReference{Kind: ReferenceKindCLIVerb, Name: "guard"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "runtime-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "runtime", Aliases: []string{"model serving runtime"}},
		Definition: "The model-completion implementation behind an engine driver, such as an on-device llama.cpp or Ollama adapter that generates text for one turn.",
		Contrasts:  []Contrast{{CanonicalTerm: "model KV cache", Explanation: "The model-serving runtime executes completion; the model KV cache is attention state owned or reused during that execution.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "model-serving"}, Owner: Owner{Leaf: "engine", Lane: "engine"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/engine/on_device.go", Revision: "runtime-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-runtime-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "OnDeviceRuntime"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "runtime-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "runtime", Aliases: []string{"worker execution runtime"}},
		Definition: "The dispatch worker process that selects a backend, optionally wraps it with fak guard, and executes one lane-scoped work packet.",
		Contrasts:  []Contrast{{CanonicalTerm: "DOS decision kind", Explanation: "The worker runtime executes admitted work; a DOS decision kind classifies persisted arbitration or resolver state about whether work may proceed.", RequiredPair: boolPointer(false), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "runtime", Value: "worker-execution"}, Owner: Owner{Leaf: "dispatchworker", Lane: "dispatch"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "cmd/dispatchworker/main.go", Revision: "runtime-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-runtime-source", Reference: &PublicReference{Kind: ReferenceKindCLIVerb, Name: "dispatchworker"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "runtime-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "dispatch worker", Aliases: []string{"worker process"}},
		Definition: "One executing worker record with structured issue, lane, backend, and witnessed-result fields; its free-form output is untrusted narration.",
		Contrasts:  []Contrast{{CanonicalTerm: "account seat", Explanation: "A worker is one execution process; a seat is provider-account capacity that may host multiple worker sessions.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "worker"}, Owner: Owner{Leaf: "dispatchaudit", Lane: "dispatch"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/dispatchaudit/dispatchaudit.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Worker"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "account seat", Aliases: []string{"seat"}},
		Definition: "A provider account-capacity slot with availability, session cap, leased slots, free slots, and bound worker IDs.",
		Contrasts:  []Contrast{{CanonicalTerm: "dispatch worker", Explanation: "A seat supplies bounded account capacity; it is not the worker process consuming one slot.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "account-seat"}, Owner: Owner{Leaf: "fleetaccounts", Lane: "accounts"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/fleetaccounts/resolve.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Seat"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "dispatch lane", Aliases: []string{"lane"}},
		Definition: "A named taxonomy partition that maps a work request to a canonical file-tree region and concurrency policy.",
		Contrasts:  []Contrast{{CanonicalTerm: "lane lease", Explanation: "The lane names the work partition; a lease is the time-bounded ownership claim that currently holds it or its tree.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "lane"}, Owner: Owner{Leaf: "laneadmit", Lane: "dos"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/laneadmit/laneadmit.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Request"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "lane lease", Aliases: []string{"lease"}},
		Definition: "A live ownership claim carrying lease ID, lane or tree, holder identity, and read-only posture for collision admission.",
		Contrasts:  []Contrast{{CanonicalTerm: "dispatch lane", Explanation: "A lease is an active holder claim; the lane is the durable taxonomy partition it may claim.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "lease"}, Owner: Owner{Leaf: "laneadmit", Lane: "dos"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/laneadmit/laneadmit.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Lease"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "compute fleet", Aliases: []string{"fleet"}},
		Definition: "A transport-agnostic roster of uniquely identified controllable machines whose live reports are folded by the public fleet core.",
		Contrasts:  []Contrast{{CanonicalTerm: "dispatch wave", Explanation: "A fleet is the available machine roster; a wave is one concurrency-safe batch of work selected for launch.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "fleet"}, Owner: Owner{Leaf: "fleet", Lane: "fleet"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/fleet/roster.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Roster"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "dispatch wave", Aliases: []string{"wave"}},
		Definition: "An indexed, bounded batch of dispatch members with a shared step budget and explicit lease regions or whole-lane claims.",
		Contrasts:  []Contrast{{CanonicalTerm: "compute fleet", Explanation: "A wave is a selected launch batch; the fleet is the machine population on which batches may execute.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "wave"}, Owner: Owner{Leaf: "issuecohort", Lane: "dispatch"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/issuecohort/issuecohort.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "Wave"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "dispatch loop", Aliases: []string{"loop"}},
		Definition: "A durable recurring dispatch state machine identified by loop ID and measured through admitted, refused, started, ended, and witnessed runs.",
		Contrasts:  []Contrast{{CanonicalTerm: "fleet supervisor", Explanation: "A loop owns recurring execution state for one cadence; a supervisor observes multiple witnessed surfaces and decides interventions.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "loop"}, Owner: Owner{Leaf: "loopmgr", Lane: "loop"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/loopmgr/loopmgr.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "LoopSnapshot"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
	{
		Schema:     EntrySchemaVersion,
		Identity:   Identity{CanonicalTerm: "fleet supervisor", Aliases: []string{"supervisor"}},
		Definition: "A decision layer whose input is witnessed liveness, worker verdicts, escalations, and leases; missing witnesses cause escalation rather than inference.",
		Contrasts:  []Contrast{{CanonicalTerm: "dispatch loop", Explanation: "The supervisor reasons over witnessed fleet state; a loop is one recurring execution state machine it may observe or act on.", RequiredPair: boolPointer(true), ForbiddenConflation: boolPointer(true)}},
		Scope:      Scope{Kind: "dispatch", Value: "supervisor"}, Owner: Owner{Leaf: "supervisoragent", Lane: "fleet"},
		Sources:   []SourceWitness{{Kind: SourceKindGoSource, Locator: "internal/supervisoragent/input.go", Revision: "fleet-source/1", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fak-disambiguation-fleet-source", Reference: &PublicReference{Kind: ReferenceKindGoSymbol, Name: "SupervisorInput"}}},
		Freshness: Freshness{Verdict: FreshnessFresh, ReasonCode: "SOURCE_CURRENT", CheckedAt: "2026-08-17T00:00:00Z", Probe: "fleet-source/1"}, Lifecycle: Lifecycle{Class: LifecycleCurrent, Rollout: RolloutOn},
	},
}

var publicIndex = mustNewIndex(publicEntries)

func mustNewIndex(entries []Entry) *Index {
	index, err := NewIndex(entries)
	if err != nil {
		panic(fmt.Sprintf("invalid public disambiguation index: %v", err))
	}
	return index
}

func cloneEntry(entry Entry) Entry {
	entry.Identity.Aliases = append([]string(nil), entry.Identity.Aliases...)
	entry.Contrasts = append([]Contrast(nil), entry.Contrasts...)
	entry.Sources = append([]SourceWitness(nil), entry.Sources...)
	return entry
}

func boolPointer(value bool) *bool { return &value }
