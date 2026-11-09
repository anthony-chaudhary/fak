package cacheobs

// capability.go — CAPABILITY PROBE + METRIC ALIAS TABLE for foreign-server scrapes.
//
// Borrow (study-repo, clean-room conceptual adaptation of
// anthony-chaudhary/Benchmark @b60f2f05e23ec38ae4f50ca19717c2f615ed419a; MIT-declared with no
// root LICENSE → ADAPT verdict, no source copied): §B.14 "capability detection over value
// inference" — a metric-registry probe whose consumers gate on capability instead of guessing
// from silent zeros (AIBrix-only fields read 0 on vanilla servers; hybrid builds leave
// evict/load-back counters flat-zero) — and §C.4 "pinned-version metric contracts": a floating
// upstream version silently absorbed a metric rename (TPOT → inter_token_latency_seconds) and
// produced hundreds of all-zero snapshots while the sibling latency stayed correct; the fix is
// alias tuples that keep old and new spellings resolving to one canonical.
//
// The fak seam this closes: every counter in this package is one THIS process books. When the
// observation target becomes a FOREIGN server's wire — a scraped set of metric family names —
// two failure modes appear that the owned-counter surface never has:
//
//  1. A capability the server does not EMIT is indistinguishable from a capability that emits
//     zeros. A consumer reading values infers "idle" where the truth is "absent", and gates
//     nothing off it.
//  2. An upstream rename moves the wire name out from under a fixed reader, and the miss shows
//     up later as a long run of silent zeros, not as an error at the point of drift.
//
// This file adds the two-part contract:
//
//   - Capabilities + Probe: explicit availability booleans derived ONLY from metric-name
//     PRESENCE in a caller-supplied name set. Absence and zero are distinguishable because the
//     probe never sees values: a name that is present but reads zero still declares its
//     capability; a missing name leaves it false whatever a value-infering consumer would
//     guess. Consumers gate on the bools, never on silent zeros.
//   - The metric alias table: a data literal mapping canonical names to their upstream wire
//     spellings, resolved in both directions (ResolveCanonical, AliasesOf). Renaming an
//     upstream metric is a table edit — append a row — not a code change; TestMetricAliasRenameIsATableEdit
//     walks that path. An unregistered name fails closed with a typed ErrUnknownMetric
//     (errors.Is), so a drift surfaces at lookup time, not as a downstream zero.
//
// No scrape/HTTP wiring in this leaf: the probe consumes a caller-supplied name set. This file
// complements metricspec.go without modifying it: the spec registry declares how to fold events
// this package books; this file declares what a foreign wire may or may not offer, and under
// which spellings.
//
// Route: clean-room conceptual adaptation (ADAPT); behavior contracts from the pinned study
// note §B–§C; no bytes vendored.

import (
	"errors"
	"fmt"
	"sort"
)

// ErrUnknownMetric is the sentinel returned when a metric name resolves to no registered
// canonical metric. Callers match it with errors.Is; the concrete *UnknownMetricError carries
// the offending name for diagnostics.
var ErrUnknownMetric = errors.New("cacheobs: unknown metric")

// UnknownMetricError is the closed-contract error for an unregistered metric name. It wraps
// the ErrUnknownMetric sentinel via Is so callers can branch on the class without coupling to
// prose.
type UnknownMetricError struct {
	Name string
}

// Error renders the refusal naming the offending spelling.
func (e *UnknownMetricError) Error() string {
	return fmt.Sprintf("cacheobs: unknown metric %q: register the spelling in the metric alias table", e.Name)
}

// Is reports identity against the ErrUnknownMetric sentinel — the closed contract callers
// assert with errors.Is.
func (e *UnknownMetricError) Is(target error) bool { return target == ErrUnknownMetric }

// aliasRow is one DATA row of the metric alias table: a canonical metric name and the
// alternate upstream wire spellings that refer to the same quantity. Rows carry strings only
// — no functions, no computed fields — so renaming an upstream metric is an edit to the table
// literal, never a change to resolution logic. Multiple rows may share a canonical (that is
// how a rename is appended); validateAliasTable pins the invariants that keep resolution
// unique regardless.
type aliasRow struct {
	canonical string
	aliases   []string
}

// metricAliasTable is the alias table itself: a data literal, seeded from the wire spellings
// documented at the pinned Benchmark tree. Canonical names use this package's bare snake_case
// vocabulary; aliases are the upstream wire spellings observed on the foreign server.
//
// The inter_token_latency_seconds row encodes the documented rename (§C.4): the pre-rename
// spelling and the post-rename spelling both resolve to one canonical, so a scrape taken
// before OR after the rename joins to the same field.
var metricAliasTable = []aliasRow{
	{canonical: "cached_tokens_total", aliases: []string{"sglang:cached_tokens_total"}},
	{canonical: "prompt_tokens_total", aliases: []string{"sglang:prompt_tokens_total"}},
	{canonical: "evicted_tokens_total", aliases: []string{"sglang:evicted_tokens_total"}},
	{canonical: "load_back_tokens_total", aliases: []string{"sglang:load_back_tokens_total"}},
	{canonical: "backuped_tokens_total", aliases: []string{"sglang:backuped_tokens_total"}},
	{canonical: "prefetched_tokens_total", aliases: []string{"sglang:prefetched_tokens_total"}},
	{
		canonical: "inter_token_latency_seconds",
		aliases:   []string{"sglang:inter_token_latency_seconds", "sglang:time_per_output_token_seconds"},
	},
}

// Capabilities is the typed availability result of a capability probe: one explicit boolean
// per capability, each derived ONLY from metric-name presence in a scraped name set. A true
// bool means "the wire offers this metric under some registered spelling"; it says nothing
// about the metric's VALUE. That is the point: absence and zero stay distinguishable, so a
// consumer can gate on capability instead of inferring from silent zeros.
type Capabilities struct {
	// CachedTokens: the base cache-hit token counter is offered on the wire.
	CachedTokens bool
	// PromptTokens: the prompt token counter is offered on the wire.
	PromptTokens bool
	// Eviction: the host-pool eviction counter is offered on the wire.
	Eviction bool
	// LoadBack: the host→device load-back counter is offered on the wire.
	LoadBack bool
	// Prefetch: the storage-tier prefetch counter is offered on the wire.
	Prefetch bool
	// Backup: the storage-tier backup counter is offered on the wire.
	Backup bool
	// InterTokenLatency: the inter-token latency metric is offered under any registered
	// spelling (pre- or post-rename).
	InterTokenLatency bool
}

// HierarchicalCache reports whether a host (L2) tier is detectable from counter presence:
// eviction or load-back counters on the wire (§B.4 layer detection). A server without either
// counter cannot evidence a host tier, however large its DRAM.
func (c Capabilities) HierarchicalCache() bool { return c.Eviction || c.LoadBack }

// L3Storage reports whether a storage (L3) tier is detectable from counter presence:
// prefetch or backup counters on the wire (§B.4). These are the names that only appear when
// a storage backend is actually configured.
func (c Capabilities) L3Storage() bool { return c.Prefetch || c.Backup }

// HybridCounterEmission reports whether BOTH the eviction and load-back counters are emitted
// together — the signal that distinguishes a build that genuinely drives the host pool from
// one whose hybrid counters sit flat-zero by construction. Presence of the pair, not their
// values, is what is asserted here.
func (c Capabilities) HybridCounterEmission() bool { return c.Eviction && c.LoadBack }

// Probe derives the capability availability from the set of metric names observed on the
// wire. Presence-based ONLY: the input is a name set, so a name that is present but reads
// zero still declares its capability, and a missing name leaves its capability false
// regardless of what a value-infering consumer would compute from a silent zero. Observed
// names are matched through the alias table, so an upstream variant spelling declares the
// same capability as its canonical form; names the table does not register are ignored —
// they witness no capability, and the probe stays total (it never errors on a scrape).
// A nil or empty set yields the zero Capabilities: everything absent.
func Probe(observed map[string]bool) Capabilities {
	present := make(map[string]bool)
	for name := range observed {
		if canonical, err := ResolveCanonical(name); err == nil {
			present[canonical] = true
		}
	}
	return Capabilities{
		CachedTokens:      present["cached_tokens_total"],
		PromptTokens:      present["prompt_tokens_total"],
		Eviction:          present["evicted_tokens_total"],
		LoadBack:          present["load_back_tokens_total"],
		Backup:            present["backuped_tokens_total"],
		Prefetch:          present["prefetched_tokens_total"],
		InterTokenLatency: present["inter_token_latency_seconds"],
	}
}

// ResolveCanonical returns the canonical metric name for any REGISTERED spelling — the
// canonical itself or any of its aliases, across all rows sharing that canonical. Matching is
// exact and case-sensitive (wire names are identifiers, not prose). An unregistered name
// returns an error satisfying errors.Is(err, ErrUnknownMetric): drift fails closed at lookup
// time instead of surfacing later as a run of silent zeros.
func ResolveCanonical(name string) (string, error) {
	for _, row := range metricAliasTable {
		if name == row.canonical {
			return row.canonical, nil
		}
		for _, alias := range row.aliases {
			if name == alias {
				return row.canonical, nil
			}
		}
	}
	return "", &UnknownMetricError{Name: name}
}

// AliasesOf returns every registered spelling of a canonical metric — the canonical itself
// plus all aliases contributed by every row naming it — sorted for deterministic output. This
// is the reverse direction of ResolveCanonical: given the canonical, enumerate the wire
// spellings a scraper must recognize. An unregistered canonical returns an error satisfying
// errors.Is(err, ErrUnknownMetric).
func AliasesOf(canonical string) ([]string, error) {
	var out []string
	found := false
	for _, row := range metricAliasTable {
		if row.canonical != canonical {
			continue
		}
		found = true
		out = append(out, row.canonical)
		out = append(out, row.aliases...)
	}
	if !found {
		return nil, &UnknownMetricError{Name: canonical}
	}
	// Deduplicate (a canonical repeated across rows must not list twice) and sort.
	seen := make(map[string]bool, len(out))
	deduped := out[:0]
	for _, s := range out {
		if !seen[s] {
			seen[s] = true
			deduped = append(deduped, s)
		}
	}
	sort.Strings(deduped)
	return deduped, nil
}

// validateAliasTable checks the data-literal invariants that keep resolution UNIQUE no matter
// how the table grows: an alias must not equal its own row's canonical, and no spelling may
// claim two different canonicals (whether as canonical or alias). Duplicate canonicals across
// rows are permitted — that is the append-a-row rename workflow. Returns nil when the table is
// consistent.
func validateAliasTable(table []aliasRow) error {
	owner := make(map[string]string) // spelling -> canonical that claims it
	for _, row := range table {
		for _, spelling := range append([]string{row.canonical}, row.aliases...) {
			if spelling == "" {
				return fmt.Errorf("alias table: empty spelling in row %q", row.canonical)
			}
			if owner[spelling] != "" && owner[spelling] != row.canonical {
				return fmt.Errorf("alias table: spelling %q claimed by both %q and %q", spelling, owner[spelling], row.canonical)
			}
			owner[spelling] = row.canonical
		}
		for _, alias := range row.aliases {
			if alias == row.canonical {
				return fmt.Errorf("alias table: alias %q equals its own canonical", alias)
			}
		}
	}
	return nil
}
