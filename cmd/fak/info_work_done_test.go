package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func workDoneFixture() guardInfoVars {
	var v guardInfoVars
	if err := json.Unmarshal([]byte(`{
		"vcache":{"saved_token_equiv":184000},
		"cache_attribution":{"total_token_equiv":184000,"provider_token_equiv":112000,"fak_token_equiv":72000,"fak_vdso_avoided_calls":27}
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
		"provider cache ~112k tok + fak reuse ~72k tok · Cache tab for ablation",
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
	if len(doc.WorkDone.Sources) != 2 || doc.WorkDone.Sources[0].ExclusivityGroup != "cache_token_equiv_owner" {
		t.Fatalf("JSON source attribution = %#v", doc.WorkDone.Sources)
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
