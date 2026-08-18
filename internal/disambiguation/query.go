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
