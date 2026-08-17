package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

func TestHeadroomCompareNativeAndNoneJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runHeadroom(&out, &errb, []string{"compare", "--via", "none,native", "--json"})
	if code != 3 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report headroom.ComparisonReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.ArmsComplete || report.Complete || len(report.Arms) != 2 {
		t.Fatalf("report=%+v", report)
	}
}

func TestHeadroomCompareReportsUnavailableLLMLingua(t *testing.T) {
	var out, errb bytes.Buffer
	code := runHeadroom(&out, &errb, []string{"compare"})
	if code != 3 {
		t.Fatalf("code=%d, want 3; out=%s err=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "lingua") || !strings.Contains(out.String(), "status=error") {
		t.Fatalf("out=%s", out.String())
	}
}
func TestHeadroomCompareAppliesIndependentEvidence(t *testing.T) {
	metrics := headroom.LiveArmMetrics{
		TaskSuccess: 1, MetricFactRecall: 1, ProviderInputTokens: 100,
		TTFTMilliseconds: 10, RegrowthTaxTokens: 0, TotalCostUSD: 0.01,
	}
	evidence := headroom.LiveComparisonEvidence{
		Schema: "fak-headroom-live-evidence/1", Witness: "ledger://independent/run-1",
		WorkloadDigest: "sha256:abc", Model: "model-v1", Provider: "provider-v1",
		CacheState: "warm-prefix", Grader: "grader-v1",
		Arms: map[string]headroom.LiveArmMetrics{"none": metrics, headroom.NativeName: metrics},
	}
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/evidence.json"
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHeadroom(&out, &errb, []string{"compare", "--via", "none,native", "--evidence", path, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report headroom.ComparisonReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.Complete || report.LiveEvidence == nil || len(report.Pending) != 0 {
		t.Fatalf("report=%+v", report)
	}
}
