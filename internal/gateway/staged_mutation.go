package gateway

// staged_mutation.go — issue #2849: the cache-aware STATE-MUTATION contract,
// offered once as a kernel primitive instead of hand-rolled per command.
//
// Hermes learned that a slash command mutating system-prompt state (install a
// skill, toggle a tool, reload memory) must be CACHE-AWARE: changing the stable
// prefix mid-session bursts the provider's prompt cache, so the whole prefix has
// to be re-written (a cache_creation cost) on the very next turn. The discipline
// Hermes settled on is: DEFAULT to DEFERRED invalidation — the change is queued
// and takes effect at the next session boundary, where the prefix cold-starts
// anyway, so the running session's cache is preserved and the change costs
// nothing extra — with an opt-in `--now` for immediate invalidation. Hermes
// leaves that discipline to convention (`AGENTS.md`; canonical
// `/skills install --now`), re-implemented by hand in each command.
//
// fak offers it ONCE. Any harness mutation of the stable prefix goes through
// [StagedMutationQueue]: [StagedMutationQueue.Mutate] defers by default and only
// an explicit `now` (the `--now` opt-in) applies immediately — and that path
// CHARGES the measured cache-rebuild cost, priced by the same C1 cache-pricing
// model (cache_pricing.go) the guard exit summary values every other
// cache_creation token with. So harnesses inherit cache-safe state mutation
// instead of hand-rolling it, and every `--now` carries a witnessed cost.
//
// This is a pure, deterministic primitive: state in (the queued mutations, the
// price model), decision out. No clock, no I/O, no network — the same inputs
// always yield the same result and the same dollar charge.

// MutationTiming names WHEN a staged mutation applies relative to the running
// session — the one bit the cache-aware contract turns on.
type MutationTiming int

const (
	// ApplyDeferred is the DEFAULT: the change is queued and applies at the next
	// session boundary. The running session's cached stable prefix is preserved
	// byte-for-byte, so a deferred mutation induces NO extra cache-rebuild cost —
	// the next session cold-starts its prefix regardless of whether a change was
	// staged, so deferring the change to that boundary is precisely what makes it
	// free.
	ApplyDeferred MutationTiming = iota
	// ApplyNow is the opt-in `--now` path: the change applies to the LIVE session
	// immediately, bursting the cached prefix. The rebuild re-writes the new
	// prefix to cache, and that cache_creation is charged — the only path in this
	// primitive that induces a cost.
	ApplyNow
)

// String renders the timing as the wire token an operator surface prints.
func (t MutationTiming) String() string {
	if t == ApplyNow {
		return "now"
	}
	return "deferred"
}

// StagedMutation describes one change to the stable prefix (the cached
// system-prompt state a harness command would mutate). It is plain data: the
// command that staged it, the token length of the NEW stable prefix the change
// installs (what a `--now` rebuild must re-write to cache), and the cache tier
// that prefix will be written under.
type StagedMutation struct {
	// Name identifies the harness command that staged the change (e.g.
	// "skills install"), so a queued or applied mutation is legible in a report.
	Name string
	// NewPrefixTokens is the token length of the stable prefix this mutation
	// installs — the prefix that must be re-written to cache when it applies
	// immediately. It is the harness's own measurement (or the gateway
	// tokenizer's); this primitive prices it, it does not author it. A negative
	// value is clamped to 0 so a mis-measuring harness can never produce a
	// negative charge.
	NewPrefixTokens int
	// WriteTTL is the cache tier the rebuilt prefix is written under. An unset
	// tier prices at the 5-minute default (see CacheTTL.WriteMultiplier), the
	// same conservative fallback cache_pricing.go applies to any unattributed
	// write.
	WriteTTL CacheTTL
}

// MutationResult reports how a staged mutation resolved under the cache-aware
// contract. For a DEFERRED mutation it records that the change was queued (not
// yet live) with no induced cost; for a `--now` mutation it records that the
// change is live and carries the measured cache_creation the rebuild induced.
type MutationResult struct {
	// Name echoes the mutation's command name.
	Name string
	// Timing is the path taken: ApplyDeferred (queued) or ApplyNow (immediate).
	Timing MutationTiming
	// Applied is true once the change is LIVE — immediately for a `--now`
	// mutation, or after [StagedMutationQueue.ApplyAtBoundary] drains a deferred
	// one. A freshly deferred mutation reports false: it is queued, not applied.
	Applied bool
	// CacheCreationTokens is the cache WRITE (prefix re-write) the change
	// induced. Nonzero ONLY on the `--now` path; a deferred mutation induces no
	// extra write (the next session's cold prefix is not this change's cost), so
	// it reports 0.
	CacheCreationTokens int
	// RebuildCostUSD is the measured dollar cost of the `--now` cache rebuild,
	// priced by C1 (CachePricing.CostUSD over the induced cache_creation). 0 for
	// a deferred mutation — the witnessed price of choosing immediacy.
	RebuildCostUSD float64
}

// StagedMutationQueue is the kernel primitive: it holds the DEFERRED mutations
// waiting for the next session boundary and prices the `--now` rebuilds against
// a C1 cache-price model. It is not safe for concurrent use — a session drives
// its own queue on one goroutine, the same ownership the gateway's per-session
// state already assumes.
type StagedMutationQueue struct {
	// pricing is the C1 model the `--now` rebuild cost is charged against. It is
	// the model for whatever tier the harness runs under (Opus 4.8, Fable 5, …);
	// the caller supplies it via NewStagedMutationQueue, exactly as the rest of
	// cache_pricing.go takes its base price as a parameter.
	pricing CachePricing
	// pending is the FIFO of deferred mutations awaiting the session boundary.
	pending []StagedMutation
}

// NewStagedMutationQueue builds a queue that prices `--now` rebuilds against the
// supplied C1 model. Pass the CachePricing for the tier the harness runs under
// (e.g. DefaultCachePricing("anthropic", "claude") for the guarded Opus path).
func NewStagedMutationQueue(pricing CachePricing) *StagedMutationQueue {
	return &StagedMutationQueue{pricing: pricing}
}

// Mutate routes a stable-prefix change through the cache-aware contract. When
// now is false (the DEFAULT), the change is queued for the next session boundary
// and the result reports it as deferred, not-yet-applied, with no induced cost.
// When now is true (the `--now` opt-in), the change applies to the live session
// immediately and the result carries the measured cache_creation the rebuild
// induced, priced by C1 — the witnessed cost of bursting the cache now instead
// of waiting for the boundary.
func (q *StagedMutationQueue) Mutate(m StagedMutation, now bool) MutationResult {
	if !now {
		q.pending = append(q.pending, m)
		return MutationResult{Name: m.Name, Timing: ApplyDeferred, Applied: false}
	}
	tokens, cost := q.rebuildCost(m.NewPrefixTokens, m.WriteTTL)
	return MutationResult{
		Name:                m.Name,
		Timing:              ApplyNow,
		Applied:             true,
		CacheCreationTokens: tokens,
		RebuildCostUSD:      cost,
	}
}

// PendingCount is the number of deferred mutations waiting for the next session
// boundary — the queue depth an operator surface can render.
func (q *StagedMutationQueue) PendingCount() int { return len(q.pending) }

// ApplyAtBoundary applies every queued deferred mutation, in the order it was
// staged, and empties the queue. This is the session-boundary event the deferred
// contract targets: the new session cold-starts its prefix regardless, so each
// applied mutation reports Applied=true with NO induced cost — deferring the
// change to this boundary is what made it free. Returns the applied results
// (empty, never nil, when nothing was queued).
func (q *StagedMutationQueue) ApplyAtBoundary() []MutationResult {
	out := make([]MutationResult, 0, len(q.pending))
	for _, m := range q.pending {
		out = append(out, MutationResult{Name: m.Name, Timing: ApplyDeferred, Applied: true})
	}
	q.pending = nil
	return out
}

// rebuildCost prices the cache_creation a `--now` rebuild of an n-token prefix
// induces, via C1 (CachePricing.CostUSD): the new prefix is WRITTEN to cache at
// the tier's write multiplier — exactly the cost cache_pricing.go books for any
// cache_creation token — so a `--now` charge is byte-identical to how the
// session's own cache writes are valued. A negative token count is clamped to 0.
func (q *StagedMutationQueue) rebuildCost(prefixTokens int, ttl CacheTTL) (int, float64) {
	if prefixTokens < 0 {
		prefixTokens = 0
	}
	usage := CacheUsage{CacheCreationTokens: prefixTokens, WriteTTL: ttl}
	return prefixTokens, q.pricing.CostUSD(usage)
}

// HarnessMutation is the small adapter an EXTERNAL harness implements to opt a
// stable-prefix-changing command into the cache-aware contract without
// re-deriving the deferred/`--now` discipline by hand. It exposes only what the
// primitive needs to defer or price the change: the command's name, the token
// length of the stable prefix it would install, and the cache tier for the
// rebuild. StageHarnessMutation adapts it to a StagedMutation.
type HarnessMutation interface {
	// MutationName identifies the command (e.g. "skills install").
	MutationName() string
	// StablePrefixTokens is the token length of the stable prefix the command
	// would install — what a `--now` rebuild must re-write to cache.
	StablePrefixTokens() int
	// CacheTier is the cache tier the rebuilt prefix is written under.
	CacheTier() CacheTTL
}

// StageHarnessMutation routes an external harness's stable-prefix change through
// the same cache-aware contract as a native one: deferred when now is false,
// applied-and-charged when now is true. It is the one-line opt-in for a harness
// that already knows its prefix size — read the adapter, build the mutation,
// hand it to the queue.
func StageHarnessMutation(q *StagedMutationQueue, h HarnessMutation, now bool) MutationResult {
	return q.Mutate(StagedMutation{
		Name:            h.MutationName(),
		NewPrefixTokens: h.StablePrefixTokens(),
		WriteTTL:        h.CacheTier(),
	}, now)
}
