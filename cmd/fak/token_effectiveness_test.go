package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestTokenEffectivenessCoversEveryDefaultSaver(t *testing.T) {
	report := buildTokenEffectivenessReport(collectTokenDefaultsScorecard(repoRoot()))
	if !report.OK || report.Debt != 0 {
		t.Fatalf("report = %+v", report)
	}
	if len(report.Rows) != 9 {
		t.Fatalf("rows = %d, want source-derived 9", len(report.Rows))
	}
	for _, row := range report.Rows {
		if row.Key == "" || row.Default == "" || row.Configured == "" || row.Owner == "" || row.Mechanism == "" || row.EffectMetric == "" || row.WitnessKind == "" || row.Witness == "" || len(row.WitnessPaths) == 0 || row.Control == "" || row.Scope == "" || row.Observed != "captured" {
			t.Errorf("incomplete row: %+v", row)
		}
	}
}

func TestTokenEffectivenessCLIJSONAndMissingCoverage(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runTokenDefaultsScorecard(&out, &errOut, []string{"--effectiveness", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report tokenEffectivenessReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != "fak-token-effectiveness/1" || len(report.Rows) != 9 {
		t.Fatalf("report = %+v", report)
	}

	scorecard := collectTokenDefaultsScorecard(repoRoot())
	corpus := scorecard["corpus"].(map[string]any)
	rows := corpus["lever_status"].([]map[string]any)
	rows[0]["key"] = "new_saver_without_effect_witness"
	missing := buildTokenEffectivenessReport(scorecard)
	if missing.OK || missing.Debt != 1 || missing.Rows[0].Observed != "missing" {
		t.Fatalf("missing coverage escaped debt: %+v", missing)
	}
	if !strings.Contains(missing.Note, "not effectiveness") {
		t.Fatalf("note = %q", missing.Note)
	}
}

func TestTokenEffectivenessDoesNotOverclaimNoOpOrWireSavings(t *testing.T) {
	report := buildTokenEffectivenessReport(collectTokenDefaultsScorecard(repoRoot()))
	rows := map[string]tokenEffectivenessRow{}
	for _, row := range report.Rows {
		rows[row.Key] = row
	}
	if got := rows["ctxview"]; !strings.Contains(got.Scope, "no-op") || got.WitnessKind == "same-trace ablation" {
		t.Fatalf("ctxview must disclose the default no-op ablation, got %+v", got)
	}
	if got := rows["defercoldtools"]; !strings.Contains(got.Scope, "not an exact wire-byte saving claim") || !strings.Contains(got.EffectMetric, "resident tool-definition") {
		t.Fatalf("cold-tool deferral overclaimed byte savings: %+v", got)
	}
}
