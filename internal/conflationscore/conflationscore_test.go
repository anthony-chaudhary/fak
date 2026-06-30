package conflationscore

import (
	"strings"
	"testing"
)

// These mirror tools/conflation_scorecard_test.py 1:1 -- each KPI on a DEFECT fixture (the
// conflation it must catch) and a CLEAN fixture (the honest phrasing it must pass). The
// fixtures are the rendered fact-strings, exactly what the KPIs read. A clean live tree
// cannot exercise the defect branches, so these are the proof the port catches what the
// Python oracle catches.

func TestProvenanceLabeled_UnlabeledExternalIsDebt(t *testing.T) {
	surfaces := map[string][]string{"x/metrics.go": {
		"Prompt tokens cache_read_input_tokens served by the provider across turns."}}
	k := kpiProvenanceLabeled(surfaces)
	if len(k.Defects) == 0 {
		t.Error("an external value with no OBSERVED qualifier must be debt")
	}
	if k.Score >= 100.0 {
		t.Errorf("score should drop below 100 on a defect, got %v", k.Score)
	}
}

func TestProvenanceLabeled_LabeledExternalIsClean(t *testing.T) {
	surfaces := map[string][]string{"x/metrics.go": {
		"OBSERVED (provider-reported, relayed verbatim): cache_read_input_tokens on compacted turns."}}
	k := kpiProvenanceLabeled(surfaces)
	if len(k.Defects) != 0 {
		t.Errorf("an OBSERVED-labeled external value is honest, got defects %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestProvenanceLabeled_HonestProviderSidePhrasingRecognized(t *testing.T) {
	// The phrasing already in the tree ("provider-side reuse, distinct from the local caches")
	// is genuinely disambiguated and must NOT be flagged.
	surfaces := map[string][]string{"x/metrics.go": {
		"Prompt tokens the provider served from its prompt cache (cache_read). This is " +
			"provider-side reuse -- distinct from the local fak caches."}}
	k := kpiProvenanceLabeled(surfaces)
	if len(k.Defects) != 0 {
		t.Errorf("honest provider-side phrasing must not be debt, got %v", k.Defects)
	}
}

func TestNoFalseAttribution_BlamingFakIsDebt(t *testing.T) {
	surfaces := map[string][]string{"x/metrics.go": {
		"cache_read on the most recent turn. Cratering to 0 while fires climb means the cache broke."}}
	k := kpiNoFalseAttribution(surfaces)
	if len(k.Defects) == 0 {
		t.Error("'the cache broke' attributed to fak with no qualifier is debt")
	}
	if k.Score != 0.0 {
		t.Errorf("attribution defect score=%v want 0", k.Score)
	}
}

func TestNoFalseAttribution_DisambiguatedIsClean(t *testing.T) {
	// Describing the failure to PREVENT it (not asserting it) is honest.
	surfaces := map[string][]string{"x/metrics.go": {
		"If cache_read craters the prefix was still byte-identical, so the provider missed for a " +
			"reason fak does NOT control. Reading the crater as 'the cache broke' is the conflation " +
			"this split prevents."}}
	k := kpiNoFalseAttribution(surfaces)
	if len(k.Defects) != 0 {
		t.Errorf("disambiguated attribution is honest, got %v", k.Defects)
	}
	if k.Score != 100.0 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestFaultSignalIsolated_MixedWithoutFaultIsSoft(t *testing.T) {
	surfaces := map[string][]string{"x/metrics.go": {
		"WITNESSED fak authored shed tokens.",
		"OBSERVED provider cache_read_input_tokens relayed."}}
	k := kpiFaultSignalIsolated(surfaces)
	if len(k.Soft) == 0 {
		t.Error("a mixed family naming no single fak-fault signal is a soft nudge")
	}
	if len(k.Defects) != 0 {
		t.Errorf("fault isolation is SOFT, never HARD debt, got defects %v", k.Defects)
	}
}

func TestFaultSignalIsolated_MixedWithNamedFaultIsClean(t *testing.T) {
	surfaces := map[string][]string{"x/metrics.go": {
		"WITNESSED fak authored shed; byte-identical prefix.",
		"OBSERVED provider cache_read_input_tokens; only bail_reason prefix_mismatch>0 is fak's bug."}}
	k := kpiFaultSignalIsolated(surfaces)
	if len(k.Soft) != 0 {
		t.Errorf("a named fault signal clears the soft nudge, got %v", k.Soft)
	}
}

func TestCacheHeadlineProvenance_BareCacheWinIsDebt(t *testing.T) {
	surfaces := map[string][]cacheHeadlineLine{"docs/x.md": {
		{Line: 12, Text: "The frozen trajectory cache win is obvious."},
	}}
	k := kpiCacheHeadlineProvenance(surfaces)
	if len(k.Defects) == 0 {
		t.Fatal("bare cache win headline must be provenance debt")
	}
	if !strings.Contains(k.Defects[0], "docs/x.md:12") {
		t.Fatalf("defect should include a stable file:line, got %v", k.Defects)
	}
}

func TestCacheHeadlineProvenance_ObservedWithoutOwnerIsDebt(t *testing.T) {
	surfaces := map[string][]cacheHeadlineLine{"docs/x.md": {
		{Line: 7, Text: "OBSERVED cache win across the session."},
	}}
	k := kpiCacheHeadlineProvenance(surfaces)
	if len(k.Defects) == 0 {
		t.Fatal("cache headline with provenance but no owner/plane must be debt")
	}
	if !strings.Contains(k.Defects[0], "owner/plane/provenance") {
		t.Fatalf("defect should name the missing owner/plane/provenance contract, got %v", k.Defects)
	}
}

func TestCacheHeadlineProvenance_LabeledProviderCacheHitIsClean(t *testing.T) {
	surfaces := map[string][]cacheHeadlineLine{"docs/x.md": {
		{Line: 3, Text: "OBSERVED provider cache-hit 0.99, WITNESSED ctxplan S/N 0.30."},
		{Line: 4, Text: "Same OBSERVED provider-cache cost/latency win, no trust claim."},
	}}
	k := kpiCacheHeadlineProvenance(surfaces)
	if len(k.Defects) != 0 {
		t.Fatalf("labeled cache headlines should be clean, got %v", k.Defects)
	}
}

func TestExtractCacheHeadlineLinesSkipsCodeFence(t *testing.T) {
	src := "```text\n99% cache-hit\n```\n\n# OBSERVED provider cache-hit 99%\n"
	lines := extractCacheHeadlineLines(src)
	if len(lines) != 1 {
		t.Fatalf("lines=%v, want one non-fenced cache headline", lines)
	}
	if lines[0].Line != 5 || !strings.Contains(lines[0].Text, "OBSERVED provider") {
		t.Fatalf("line=%+v, want markdown headline outside fence", lines[0])
	}
}

func TestExtractHelpAndSummaryStrings(t *testing.T) {
	src := `writeCounter(b, "n", "OBSERVED provider cache_read_input_tokens relayed", x)` + "\n" +
		`fmt.Fprintf(&b, "fak guard: compaction -- %d fired", n)` + "\n"
	got := ExtractHelpStrings(src)
	if !containsSubstr(got, "cache_read_input_tokens") {
		t.Errorf("did not extract the writeCounter help string: %v", got)
	}
	if !containsSubstr(got, "fak guard: compaction") {
		t.Errorf("did not extract the Fprintf summary string: %v", got)
	}
}

func TestBuildEnvelopeShape(t *testing.T) {
	// Build over the repo root (resolved relative to this test's package dir) must carry the
	// full control-pane envelope keys the fold reads.
	p := Build("../..") // internal/conflationscore -> repo root
	if p.Schema != Schema {
		t.Errorf("schema=%q want %q", p.Schema, Schema)
	}
	for _, key := range []string{DebtKey, "grade", "score", "surfaces", "external_values_seen", "cache_headline_claims_seen"} {
		if _, ok := p.Corpus[key]; !ok {
			t.Errorf("corpus missing key %q: %v", key, p.Corpus)
		}
	}
	if p.Verdict == "" || p.Finding == "" || p.NextAction == "" {
		t.Error("envelope prose fields must be populated")
	}
}

func TestLiveTreeFloorIsZero(t *testing.T) {
	// The regression sentinel: the real reporting surfaces must stay provenance-honest.
	p := Build("../..")
	if got := p.Corpus[DebtKey]; got != CleanFloor {
		t.Errorf("conflation debt rose above %d: %v (%s)", CleanFloor, got, p.Reason)
	}
	if !p.OK {
		t.Errorf("disciplined tree should be ok, reason=%q", p.Reason)
	}
}

func containsSubstr(ss []string, sub string) bool {
	for _, s := range ss {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
