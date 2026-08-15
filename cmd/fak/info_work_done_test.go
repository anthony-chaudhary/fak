package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func workDoneFixture() guardInfoVars {
	var v guardInfoVars
	if err := json.Unmarshal([]byte(`{
		"vcache":{"saved_token_equiv":184000},
		"cache_attribution":{"total_token_equiv":184000,"provider_token_equiv":112000,"fak_token_equiv":72000,"fak_compaction_shed_tokens":72000,"fak_vdso_avoided_calls":27,"fak_response_memo_calls":19,"fak_inline_served_calls":8}
	}`), &v); err != nil {
		panic(err)
	}
	v.Adjudication = &gateway.AdjudicationSummary{E2ELatencySumSeconds: 38, E2ELatencyCount: 27}
	v.WorkDone = ptrGuardInfoWorkDone(guardInfoWorkDoneFromVars(v))
	return v
}

func TestGuardInfoWorkDoneCapturedRender(t *testing.T) {
	v := workDoneFixture()
	roomy := strings.Join(guardInfoWorkDoneRows(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 120), guardPanelFull), "\n")
	for _, want := range []string{
		"vs direct provider path r1 · observed session",
		"+184k input tok avoided · 27 model calls avoided · 38s wait avoided",
		"provider ~112k tok + fak ~72k tok + fak 27 calls · Cache tab for ablation",
	} {
		if !strings.Contains(roomy, want) {
			t.Fatalf("captured roomy render missing %q:\n%s", want, roomy)
		}
	}
	mini := strings.Join(guardInfoWorkDoneRows(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 42), guardPanelMini), "\n")
	if !strings.Contains(mini, "+184k input tok avoided · 27 model calls avoided") {
		t.Fatalf("captured mini render lost outcomes: %s", mini)
	}
}

func TestGuardInfoWorkDoneTinyRenderKeepsOutcome(t *testing.T) {
	row := guardInfoVisualTinyRow(workDoneFixture())
	for _, want := range []string{"work vs direct provider", "+184k input tok avoided", "27 calls avoided"} {
		if !strings.Contains(row, want) {
			t.Fatalf("tiny render missing %q: %s", want, row)
		}
	}
}

func TestGuardInfoWorkDoneUnknownsAreNotZeroSavings(t *testing.T) {
	rows := strings.Join(guardInfoWorkDoneRows(newGuardInfoPanelCtx(guardInfoVars{}, newGuardInfoTrend(4), 100), guardPanelFull), "\n")
	if strings.Count(rows, "unavailable") < 3 || strings.Contains(rows, "0 model calls") {
		t.Fatalf("cold snapshot must identify unavailable evidence, not zero savings:\n%s", rows)
	}
}

func TestGuardInfoWorkDoneJSONUsesTheRenderedAccountingObject(t *testing.T) {
	v := workDoneFixture()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		WorkDone guardInfoWorkDone `json:"work_done"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.WorkDone.Schema != guardInfoWorkDoneSchema || doc.WorkDone.Baseline.ID != guardInfoWorkDoneBaselineID {
		t.Fatalf("JSON contract identity = %#v", doc.WorkDone)
	}
	if doc.WorkDone.Baseline.Revision != 1 || doc.WorkDone.Baseline.EffectiveUTC == "" || doc.WorkDone.Baseline.ConfigurationSHA256 == "" || doc.WorkDone.Baseline.ComparisonScope == "" {
		t.Fatalf("JSON baseline descriptor incomplete = %#v", doc.WorkDone.Baseline)
	}
	metrics := map[string]guardInfoWorkDoneMetric{
		"tokens": doc.WorkDone.Metrics.InputTokensAvoided, "calls": doc.WorkDone.Metrics.ModelCallsAvoided, "wait": doc.WorkDone.Metrics.WaitSecondsAvoided,
	}
	wantEvidence := map[string]string{"tokens": "observed", "calls": "witnessed", "wait": "modeled"}
	for name, metric := range metrics {
		if metric.BaselineID != doc.WorkDone.Baseline.ID {
			t.Fatalf("%s metric baseline = %q, want %q", name, metric.BaselineID, doc.WorkDone.Baseline.ID)
		}
		if metric.Evidence != wantEvidence[name] {
			t.Fatalf("%s metric evidence = %q, want %q", name, metric.Evidence, wantEvidence[name])
		}
	}
	if got := doc.WorkDone.Metrics.ModelCallsAvoided.Value; got != 27 {
		t.Fatalf("JSON calls = %v, want 27", got)
	}
	if got := doc.WorkDone.Metrics.WaitSecondsAvoided.Value; got != 38 {
		t.Fatalf("JSON wait = %v, want 38", got)
	}
	if len(doc.WorkDone.Sources) != 4 || doc.WorkDone.Sources[0].Schema != guardInfoWorkDoneSourceSchema {
		t.Fatalf("JSON source attribution = %#v", doc.WorkDone.Sources)
	}
	if !guardInfoWorkDoneReconciles(doc.WorkDone) {
		t.Fatalf("source effects do not reconcile to totals: %#v", doc.WorkDone)
	}
}

func TestGuardInfoWorkDoneBaselineDetailNamesBothArms(t *testing.T) {
	w := guardInfoWorkDoneFromVars(workDoneFixture())
	rows := strings.Join(guardInfoWorkDoneBaselineDetailRows(w), "\n")
	for _, want := range []string{
		"direct-provider/v1 · r1 · effective 2026-08-14",
		"same observed session",
		"provider cache and fak-local reuse enabled",
		"fak-local response reuse and inline serving disabled",
		"sha256:",
	} {
		if !strings.Contains(rows, want) {
			t.Fatalf("baseline detail missing %q:\n%s", want, rows)
		}
	}
}

func TestGuardInfoWorkDoneBaselineCompatibilityIsFingerprintBound(t *testing.T) {
	a := guardInfoDirectProviderBaseline()
	b := guardInfoDirectProviderBaseline()
	if !guardInfoWorkDoneBaselineCompatible(a, b) {
		t.Fatal("identical descriptors must be compatible")
	}
	b.ConfigurationSHA256 = "sha256:provider-baseline-shifted"
	if guardInfoWorkDoneBaselineCompatible(a, b) {
		t.Fatal("shifted configuration must not merge with old baseline")
	}

	tr := newGuardInfoTrend(4)
	v := workDoneFixture()
	tr.push(v)
	tr.baseline = b
	tr.push(v)
	if tr.baselineChanges != 1 || len(tr.saved) != 1 {
		t.Fatalf("baseline shift must restart trend: changes=%d saved=%v", tr.baselineChanges, tr.saved)
	}
	rows := strings.Join(guardInfoTrendsPanelRows(newGuardInfoPanelCtx(v, tr, 120), guardPanelFull), "\n")
	if !strings.Contains(rows, "base  changed ×1 · trend restarted at direct-provider/v1") {
		t.Fatalf("baseline compatibility event absent:\n%s", rows)
	}
}

func TestRenderInfoCacheViewExplainsBaseline(t *testing.T) {
	v := workDoneFixture()
	rows := strings.Join(renderInfoCacheView(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 140)), "\n")
	for _, want := range []string{"vs direct provider path r1", "fak    current session", "fak-local response reuse", "config sha256:"} {
		if !strings.Contains(rows, want) {
			t.Fatalf("Cache detail missing %q:\n%s", want, rows)
		}
	}
}

func TestGuardInfoWorkDoneSourcesAreProducerGroundedAndExclusive(t *testing.T) {
	w := guardInfoWorkDoneFromVars(workDoneFixture())
	want := map[string]guardInfoWorkDoneSource{}
	for _, source := range w.Sources {
		want[source.ID] = source
	}
	if got := want["provider_cache"]; got.Owner != "provider" || got.InputTokenEquiv != 112000 || got.ExclusivityGroup != "input_token_equiv_owner/v1" {
		t.Fatalf("provider source = %#v", got)
	}
	if got := want["context_reduction"]; got.Owner != "fak" || got.InputTokenEquiv != 72000 {
		t.Fatalf("context source = %#v", got)
	}
	if got := want["fak_response_reuse"]; got.Events != 19 || got.ModelCallsAvoided != 19 || got.Disposition != "served" {
		t.Fatalf("response memo source = %#v", got)
	}
	if got := want["inline_tool_local"]; got.Events != 8 || got.ModelCallsAvoided != 8 || got.Disposition != "served" {
		t.Fatalf("inline source = %#v", got)
	}
	if !guardInfoWorkDoneReconciles(w) {
		t.Fatalf("mixed-source fixture did not reconcile: %#v", w)
	}
}

func TestGuardInfoWorkDoneSourcesExposeUnknownAndColdPaths(t *testing.T) {
	var partial guardInfoVars
	if err := json.Unmarshal([]byte(`{"cache_attribution":{"fak_token_equiv":5,"fak_vdso_avoided_calls":3}}`), &partial); err != nil {
		t.Fatal(err)
	}
	partial.VCache = workDoneFixture().VCache
	partial.VCache.SavedTokenEquiv = 5
	w := guardInfoWorkDoneFromVars(partial)
	if len(w.Sources) != 2 || w.Sources[0].ID != "unknown" || w.Sources[1].ID != "unknown" || !guardInfoWorkDoneReconciles(w) {
		t.Fatalf("partial provenance = %#v", w.Sources)
	}

	var cold guardInfoVars
	if err := json.Unmarshal([]byte(`{"cache_attribution":{}}`), &cold); err != nil {
		t.Fatal(err)
	}
	cw := guardInfoWorkDoneFromVars(cold)
	if len(cw.Sources) != 1 || cw.Sources[0].ID != "cold_direct" {
		t.Fatalf("cold source = %#v", cw.Sources)
	}
	unknown := guardInfoWorkDoneFromVars(guardInfoVars{})
	if len(unknown.Sources) != 1 || unknown.Sources[0].ID != "unknown" {
		t.Fatalf("absent source = %#v", unknown.Sources)
	}
}

func TestRenderInfoCacheViewShowsSourceProvenanceHierarchy(t *testing.T) {
	v := workDoneFixture()
	rows := strings.Join(renderInfoCacheView(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 180)), "\n")
	for _, want := range []string{
		"sources (exclusive within each group; do not add across units)",
		"loaded   from provider prefix cache",
		"reduced  from context reduction",
		"served   from response memo",
		"served   from inline/tool local",
		"provider/observed", "fak/witnessed",
	} {
		if !strings.Contains(rows, want) {
			t.Fatalf("source render missing %q:\n%s", want, rows)
		}
	}
}

func TestGuardInfoWorkDoneSourceHierarchyNarrowCapture(t *testing.T) {
	v := workDoneFixture()
	rows := guardInfoWorkDoneSourceRows(guardInfoWorkDoneFromVars(v))
	captured := joinPaneRowsTUI(rows, 72, 0)
	for _, line := range strings.Split(captured, "\n") {
		if dispWidthTUI(line) > 72 {
			t.Fatalf("narrow source row wraps (%d cells): %q", dispWidthTUI(line), line)
		}
	}
	for _, want := range []string{"provider prefix cache", "context reduction", "response memo", "inline/tool local"} {
		if !strings.Contains(captured, want) {
			t.Fatalf("narrow capture lost %q:\n%s", want, captured)
		}
	}
}

func TestGuardInfoWorkDoneReconcilesStaleElisionWithoutCompactionOverlap(t *testing.T) {
	v := workDoneFixture()
	v.TokenSavings.StaleReadElide.Fired = 1
	v.TokenSavings.StaleReadElide.Units = 2
	v.TokenSavings.StaleReadElide.SavedBytes = 2400
	v.TokenSavings.StaleReadElide.SavedTokens = 600
	v.TokenSavings.NativeMCPFilter.Fired = 3
	v.TokenSavings.NativeMCPFilter.Units = 15
	v.TokenSavings.NativeMCPFilter.SavedBytes = 4000
	v.TokenSavings.NativeMCPFilter.SavedTokens = 1000

	w := guardInfoWorkDoneFromVars(v)
	var context guardInfoWorkDoneSource
	for _, source := range w.Sources {
		if source.ID == "context_reduction" {
			context = source
		}
	}
	wantContext := float64(v.CacheAttribution.FakCompactionShedTokens + 600 + 1000)
	if context.InputTokenEquiv != wantContext {
		t.Fatalf("context reduction=%v, want compaction+elision=%v", context.InputTokenEquiv, wantContext)
	}
	var reconciled float64
	for _, source := range w.Sources {
		reconciled += source.InputTokenEquiv
	}
	if reconciled != w.Metrics.InputTokensAvoided.Value {
		t.Fatalf("sources=%v do not reconcile to total=%v", reconciled, w.Metrics.InputTokensAvoided.Value)
	}
	rendered := strings.Join(guardInfoWorkDoneSourceRows(w), "\n")
	if !strings.Contains(rendered, "context reduction") || !strings.Contains(rendered, "~73.6k input") {
		t.Fatalf("captured render lost elision receipt:\n%s", rendered)
	}
	encoded, err := json.Marshal(guardInfoSessionWorkDoneQuery(v, time.Unix(0, 0)))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"id":"context_reduction"`) || !strings.Contains(string(encoded), `"input_token_equiv":73600`) {
		t.Fatalf("query lost elision receipt: %s", encoded)
	}
}
