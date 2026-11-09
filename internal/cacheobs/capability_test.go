package cacheobs

import (
	"errors"
	"reflect"
	"testing"
)

// allCanonicalNames lists every seeded canonical once — the "server offers everything" input.
func allCanonicalNames() map[string]bool {
	set := make(map[string]bool)
	for _, row := range metricAliasTable {
		set[row.canonical] = true
	}
	return set
}

// TestCapabilityProbeDerivesBoolFromPresence pins the core contract: each capability bool is
// true exactly when its witness metric is present in the scraped name set — nothing else turns
// it on. Absent witnesses stay false; the derived helpers follow from the base bools alone.
func TestCapabilityProbeDerivesBoolFromPresence(t *testing.T) {
	// Nil and empty sets: everything absent, including the derived signals.
	if c := Probe(nil); c != (Capabilities{}) {
		t.Fatalf("nil name set must yield all-absent capabilities, got %+v", c)
	}
	if c := Probe(map[string]bool{}); c != (Capabilities{}) {
		t.Fatalf("empty name set must yield all-absent capabilities, got %+v", c)
	}

	// Full set under canonical spellings: every capability declared.
	full := Probe(allCanonicalNames())
	want := Capabilities{
		CachedTokens:      true,
		PromptTokens:      true,
		Eviction:          true,
		LoadBack:          true,
		Backup:            true,
		Prefetch:          true,
		InterTokenLatency: true,
	}
	if full != want {
		t.Fatalf("full name set must declare every capability, got %+v", full)
	}
	if !full.HierarchicalCache() || !full.L3Storage() || !full.HybridCounterEmission() {
		t.Fatalf("full set must satisfy every derived signal, got %+v", full)
	}

	// Partial sets: only the witnessed capability (and its derived closure) is true.
	partial := Probe(map[string]bool{"prefetched_tokens_total": true})
	if partial != (Capabilities{Prefetch: true}) {
		t.Fatalf("prefetch-only set must declare only Prefetch, got %+v", partial)
	}
	if !partial.L3Storage() || partial.HierarchicalCache() || partial.HybridCounterEmission() {
		t.Fatalf("prefetch-only set must imply L3Storage and nothing else, got %+v", partial)
	}

	loadBackOnly := Probe(map[string]bool{"load_back_tokens_total": true})
	if loadBackOnly != (Capabilities{LoadBack: true}) {
		t.Fatalf("load-back-only set must declare only LoadBack, got %+v", loadBackOnly)
	}
	if !loadBackOnly.HierarchicalCache() || loadBackOnly.HybridCounterEmission() {
		t.Fatalf("one counter evidences the tier but not hybrid emission, got %+v", loadBackOnly)
	}
}

// TestCapabilityAliasSpellingDeclaresSameCapability proves the alias table feeds the probe: an
// upstream wire spelling declares exactly the capability its canonical form declares, and the
// documented rename pair (pre- and post-rename spellings) both gate the same capability. Names
// the table does not register witness nothing.
func TestCapabilityAliasSpellingDeclaresSameCapability(t *testing.T) {
	variants := map[string]bool{
		"sglang:cached_tokens_total":         true,
		"sglang:prompt_tokens_total":         true,
		"sglang:evicted_tokens_total":        true,
		"sglang:load_back_tokens_total":      true,
		"sglang:backuped_tokens_total":       true,
		"sglang:prefetched_tokens_total":     true,
		"sglang:inter_token_latency_seconds": true,
	}
	fromVariants := Probe(variants)
	fromCanonical := Probe(allCanonicalNames())
	if fromVariants != fromCanonical {
		t.Fatalf("upstream variant spellings must declare the same capabilities as canonical forms:\n variants=%+v\n canonical=%+v", fromVariants, fromCanonical)
	}

	// The documented rename: both spellings gate InterTokenLatency identically.
	pre := Probe(map[string]bool{"sglang:time_per_output_token_seconds": true})
	post := Probe(map[string]bool{"sglang:inter_token_latency_seconds": true})
	if pre != (Capabilities{InterTokenLatency: true}) || post != pre {
		t.Fatalf("pre- and post-rename spellings must declare InterTokenLatency identically, got pre=%+v post=%+v", pre, post)
	}

	// An unregistered name witnesses no capability — the probe stays total and silent.
	if c := Probe(map[string]bool{"some_unregistered_wire_name": true}); c != (Capabilities{}) {
		t.Fatalf("unregistered names must witness nothing, got %+v", c)
	}
}

// TestCapabilityAbsenceAndZeroAreDistinguishable pins why the probe is presence-only: the input
// type carries NO values, so "the metric exists and reads zero" and "the metric does not exist"
// are different inputs with different results. A consumer gating on the bool can never confuse a
// silent zero with an absent capability — the confusion is unrepresentable here.
func TestCapabilityAbsenceAndZeroAreDistinguishable(t *testing.T) {
	// Present (whatever its value would be) -> capability declared.
	present := Probe(map[string]bool{"cached_tokens_total": true})
	if !present.CachedTokens {
		t.Fatalf("a present witness name must declare its capability even if it reads zero, got %+v", present)
	}
	// Absent -> capability false; the zero-value struct IS the all-absent baseline.
	absent := Probe(map[string]bool{"prompt_tokens_total": true})
	if absent.CachedTokens {
		t.Fatalf("an absent witness name must leave its capability false, got %+v", absent)
	}
	if absent != (Capabilities{PromptTokens: true}) {
		t.Fatalf("result must be a pure function of name-set membership, got %+v", absent)
	}
}

// TestMetricAliasResolvesCanonicalFromAnyRegisteredName walks every seeded row in the forward
// direction: the canonical resolves to itself, and every registered alias resolves to that
// canonical.
func TestMetricAliasResolvesCanonicalFromAnyRegisteredName(t *testing.T) {
	for _, row := range metricAliasTable {
		got, err := ResolveCanonical(row.canonical)
		if err != nil || got != row.canonical {
			t.Fatalf("canonical %q must resolve to itself, got %q, err %v", row.canonical, got, err)
		}
		for _, alias := range row.aliases {
			got, err := ResolveCanonical(alias)
			if err != nil {
				t.Fatalf("alias %q must resolve without error, got %v", alias, err)
			}
			if got != row.canonical {
				t.Fatalf("alias %q must resolve to canonical %q, got %q", alias, row.canonical, got)
			}
		}
	}
}

// TestMetricAliasReverseDirectionListsAllSpellings walks the reverse direction: given a
// canonical, enumerate every registered spelling (canonical plus all aliases, sorted), and
// round-trip each spelling back through ResolveCanonical to the same canonical.
func TestMetricAliasReverseDirectionListsAllSpellings(t *testing.T) {
	const canonical = "inter_token_latency_seconds"
	spellings, err := AliasesOf(canonical)
	if err != nil {
		t.Fatalf("AliasesOf(%q) must succeed, got %v", canonical, err)
	}
	want := []string{
		"inter_token_latency_seconds",
		"sglang:inter_token_latency_seconds",
		"sglang:time_per_output_token_seconds",
	}
	if !reflect.DeepEqual(spellings, want) {
		t.Fatalf("AliasesOf(%q) = %v, want %v (sorted, canonical included)", canonical, spellings, want)
	}
	for _, s := range spellings {
		back, err := ResolveCanonical(s)
		if err != nil || back != canonical {
			t.Fatalf("round trip: %q must resolve back to %q, got %q, err %v", s, canonical, back, err)
		}
	}

	// A single-alias row lists exactly canonical + alias.
	single, err := AliasesOf("cached_tokens_total")
	if err != nil {
		t.Fatalf("AliasesOf(cached_tokens_total) must succeed, got %v", err)
	}
	if !reflect.DeepEqual(single, []string{"cached_tokens_total", "sglang:cached_tokens_total"}) {
		t.Fatalf("single-alias row must list canonical + alias sorted, got %v", single)
	}
}

// TestMetricAliasUnknownNameFailsTyped pins the closed contract: an unregistered name fails
// with the ErrUnknownMetric sentinel in BOTH directions. Tests assert the sentinel via
// errors.Is, never the prose.
func TestMetricAliasUnknownNameFailsTyped(t *testing.T) {
	if _, err := ResolveCanonical("no_such_metric"); err == nil {
		t.Fatal("unregistered name must fail resolution")
	} else if !errors.Is(err, ErrUnknownMetric) {
		t.Fatalf("unregistered name must fail with ErrUnknownMetric, got %T: %v", err, err)
	}
	if _, err := ResolveCanonical(""); err == nil || !errors.Is(err, ErrUnknownMetric) {
		t.Fatalf("empty name must fail with ErrUnknownMetric, got %v", err)
	}
	if _, err := AliasesOf("no_such_metric"); err == nil {
		t.Fatal("unregistered canonical must fail the reverse lookup")
	} else if !errors.Is(err, ErrUnknownMetric) {
		t.Fatalf("reverse lookup of unregistered canonical must fail with ErrUnknownMetric, got %T: %v", err, err)
	}
}

// TestMetricAliasRenameIsATableEdit is the acceptance proof that renaming an upstream metric is
// a TABLE EDIT, not a code change: append one row carrying the new wire spelling for an
// existing canonical, and resolution works through it in both directions — and the capability
// probe gates on it — with zero changes to any resolver or probe logic.
func TestMetricAliasRenameIsATableEdit(t *testing.T) {
	prev := metricAliasTable
	t.Cleanup(func() { metricAliasTable = prev })
	metricAliasTable = append(metricAliasTable, aliasRow{
		canonical: "inter_token_latency_seconds",
		aliases:   []string{"sglang:new_tpot_spelling"},
	})

	// Forward: the appended alias resolves through the unchanged resolver.
	got, err := ResolveCanonical("sglang:new_tpot_spelling")
	if err != nil || got != "inter_token_latency_seconds" {
		t.Fatalf("appended alias must resolve to the existing canonical, got %q, err %v", got, err)
	}
	// Reverse: the appended alias now enumerates for the canonical.
	spellings, err := AliasesOf("inter_token_latency_seconds")
	if err != nil {
		t.Fatalf("AliasesOf after table edit must succeed, got %v", err)
	}
	found := false
	for _, s := range spellings {
		if s == "sglang:new_tpot_spelling" {
			found = true
		}
	}
	if !found {
		t.Fatalf("reverse direction must list the appended alias, got %v", spellings)
	}
	// Capability gating follows the table edit too.
	if c := Probe(map[string]bool{"sglang:new_tpot_spelling": true}); !c.InterTokenLatency {
		t.Fatalf("probe must gate InterTokenLatency on the appended alias, got %+v", c)
	}
}

// TestMetricAliasTableConsistency pins the data-literal invariants: the seeded table is
// consistent, and the validator bites on planted violations (one spelling claimed by two
// rival canonicals; an alias equal to its own canonical).
func TestMetricAliasTableConsistency(t *testing.T) {
	if err := validateAliasTable(metricAliasTable); err != nil {
		t.Fatalf("seeded alias table must be consistent, got %v", err)
	}

	prev := metricAliasTable
	t.Cleanup(func() { metricAliasTable = prev })

	// Violation 1: one alias claimed under two different canonicals.
	metricAliasTable = append(append([]aliasRow{}, prev...),
		aliasRow{canonical: "rival_canonical_a", aliases: []string{"contested_spelling"}},
		aliasRow{canonical: "rival_canonical_b", aliases: []string{"contested_spelling"}},
	)
	if err := validateAliasTable(metricAliasTable); err == nil {
		t.Fatal("a spelling claimed by two rival canonicals must fail validation")
	}

	// Violation 2: an alias equal to its own canonical.
	metricAliasTable = append(append([]aliasRow{}, prev...),
		aliasRow{canonical: "self_aliasing", aliases: []string{"self_aliasing"}},
	)
	if err := validateAliasTable(metricAliasTable); err == nil {
		t.Fatal("an alias equal to its own canonical must fail validation")
	}
}
