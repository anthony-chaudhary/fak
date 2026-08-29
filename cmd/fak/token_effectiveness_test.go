package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

func TestTokenEffectivenessCoversEveryDefaultSaver(t *testing.T) {
	scorecard := collectTokenDefaultsScorecard(repoRoot())
	report := buildTokenEffectivenessReport(scorecard)
	if report.OK || report.Debt != 1 {
		t.Fatalf("report = %+v; want only default-on #3536 blocked as debt (headroom remains visible off+gated)", report)
	}
	sourceRows := scorecard["corpus"].(map[string]any)["lever_status"].([]map[string]any)
	if len(report.Rows) != len(sourceRows) {
		t.Fatalf("rows = %d, want one per source row %d", len(report.Rows), len(sourceRows))
	}
	for _, row := range report.Rows {
		if row.Key == "" || row.Default == "" || row.Configured == "" || row.Owner == "" || row.Mechanism == "" || row.EffectMetric == "" || row.WitnessKind == "" || row.Witness == "" || len(row.Paths) == 0 || row.Control == "" || row.Scope == "" || len(row.Provenance) == 0 {
			t.Errorf("incomplete row: %+v", row)
		}
	}
}

func TestTokenEffectivenessCLIJSONAndMissingCoverage(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runTokenDefaultsScorecard(&out, &errOut, []string{"--effectiveness", "--json"}); code != 1 {
		t.Fatalf("code=%d want debt exit 1; stderr=%s", code, errOut.String())
	}
	var report tokenEffectivenessReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	scorecard := collectTokenDefaultsScorecard(repoRoot())
	corpus := scorecard["corpus"].(map[string]any)
	rows := corpus["lever_status"].([]map[string]any)
	if report.Schema != "fak-token-effectiveness/2" || len(report.Rows) != len(rows) {
		t.Fatalf("report = %+v; source rows=%d", report, len(rows))
	}
	rows[len(rows)-2]["key"] = "new_saver_without_effect_witness"
	missing := buildTokenEffectivenessReport(scorecard)
	if missing.OK || missing.Debt != report.Debt+1 || missing.Rows[len(rows)-2].Observed != "missing" {
		t.Fatalf("missing coverage escaped debt: %+v", missing)
	}
	if !strings.Contains(missing.Note, "not effectiveness") {
		t.Fatalf("note = %q", missing.Note)
	}
}

func TestExecutableProofRejectsCommentsDecoysVacuityAndDeadAssertions(t *testing.T) {
	proof := tokenExecutableProof{Path: "fixture_test.go", Function: "TestWitness", Assertions: []tokenAssertionSpec{assertion("treatment", "control"), assertion("saved")}}
	cases := map[string]string{
		"comment-only": `package p; import "testing"; func TestWitness(t *testing.T) { // if treatment > control { t.Fatal("saved") }
}`,
		"decoy":        `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control, saved := 2, 1, 1; _, _, _ = treatment, control, saved; t.Log("ok") }`,
		"vacuous":      `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control, saved := 2, 1, 1; if treatment > control {}; if saved == 0 {} }`,
		"dead":         `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control, saved := 2, 1, 1; if false && treatment > control { t.Fatal("decoy") }; if false && saved == 0 { t.Fatal("decoy") } }`,
		"nested-dead":  `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control, saved := 2, 1, 1; if treatment <= control { if false { t.Fatal("dead nested") } }; if saved <= 0 { t.Fatal("saving") } }`,
		"const-dead":   `package p; import "testing"; func TestWitness(t *testing.T) { const dead = false; treatment, control, saved := 2, 1, 1; if treatment <= control { if dead { t.Fatal("dead const") } }; if saved <= 0 { t.Fatal("saving") } }`,
		"boolean-dead": `package p; import "testing"; func TestWitness(t *testing.T) { const dead = false; treatment, control, saved := 2, 1, 1; if treatment <= control { if (false || dead) && !true { t.Fatal("dead boolean") } }; if saved <= 0 { t.Fatal("saving") } }`,
		"effect-only":  `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control := 2, 1; if treatment <= control { t.Fatal("no effect") } }`,
	}
	base := tokenDefaultSources{root: repoRoot()}
	for name, source := range cases {
		t.Run(name, func(t *testing.T) {
			if base.withOverride(proof.Path, source).executableProof(proof) {
				t.Fatal("non-executable or incomplete sentinel was accepted")
			}
		})
	}
	good := `package p; import "testing"; func TestWitness(t *testing.T) { treatment, control, saved := 2, 1, 1; if treatment <= control { t.Fatal("no effect") }; if saved <= 0 { t.Fatal("no saving") } }`
	if !base.withOverride(proof.Path, good).executableProof(proof) {
		t.Fatal("executable treatment/control sentinel was rejected")
	}
}

func TestTokenGatePassIsStrictAndDuplicateSafe(t *testing.T) {
	valid := `{"schema":"fak-token-gate-pass/1","id":"#3536","status":"PASS"}`
	if !parseTokenGatePass(valid, deferColdToolsGate) {
		t.Fatal("valid exact gate receipt rejected")
	}
	invalid := []string{
		`{"schema":"fak-token-gate-pass/1","id":"#3536","id":"#3536","status":"PASS"}`,
		`{"schema":"fak-token-gate-pass/1","id":"#3536","status":"PASS"`,
		`{"schema":"fak-token-gate-pass/1","id":"#3536","status":"OPEN","note":"status PASS"}`,
		`{"schema":"fak-token-gate-pass/1","id":"#3536","status":"PASS","extra":true}`,
		`{"schema":"fak-token-gate-pass/1","id":"#9999","status":"PASS"}`,
	}
	for _, raw := range invalid {
		if parseTokenGatePass(raw, deferColdToolsGate) {
			t.Fatalf("invalid/duplicate/decoy receipt accepted: %s", raw)
		}
	}
	sources := tokenDefaultSources{root: repoRoot()}.withOverride(deferColdToolsGate.PassArtifact, valid)
	if blocker := tokenGateBlocker(sources, &deferColdToolsGate); blocker != "" {
		t.Fatalf("valid declaration + strict PASS stayed blocked: %s", blocker)
	}
	withoutPass := tokenDefaultSources{root: repoRoot()}
	if blocker := tokenGateBlocker(withoutPass, &deferColdToolsGate); !strings.Contains(blocker, "#3536 OPEN") {
		t.Fatalf("valid OPEN declaration without PASS did not block: %q", blocker)
	}
	invalidDeclarations := map[string]tokenGateDeclaration{
		"schema":      {Schema: "fak-token-required-gate/0", ID: "#3536", State: tokenGateOpen, PassArtifact: deferColdToolsGate.PassArtifact},
		"id":          {Schema: deferColdToolsGate.Schema, ID: "#9999", State: tokenGateOpen, PassArtifact: deferColdToolsGate.PassArtifact},
		"state":       {Schema: deferColdToolsGate.Schema, ID: "#3536", State: tokenGateState("PASS"), PassArtifact: deferColdToolsGate.PassArtifact},
		"path":        {Schema: deferColdToolsGate.Schema, ID: "#3536", State: tokenGateOpen, PassArtifact: "docs/notes/decoy.json"},
		"declaration": {},
	}
	for name, gate := range invalidDeclarations {
		t.Run("invalid-"+name, func(t *testing.T) {
			if blocker := tokenGateBlocker(sources, &gate); blocker != "invalid required-gate declaration" {
				t.Fatalf("invalid nonnil gate failed open: blocker=%q gate=%+v", blocker, gate)
			}
		})
	}
}

func TestTokenEffectivenessRegisteredFamilyAndGateAreExplicit(t *testing.T) {
	score := collectTokenDefaultsScorecardWithInputs(repoRoot(), loadTokenDefaultSources(repoRoot()), headroom.Names(), "noop")
	report := buildTokenEffectivenessReport(score)
	rows := map[string]tokenEffectivenessRow{}
	for _, row := range report.Rows {
		rows[row.Key] = row
	}
	if rows["headroomcompressor"].Observed != "missing" || !strings.Contains(rows["headroomcompressor"].Scope, "default-off") {
		t.Fatalf("headroom posture overclaimed: %+v", rows["headroomcompressor"])
	}
	if rows["defercoldtools"].Observed != "blocked" || !strings.Contains(rows["defercoldtools"].Blocker, "#3536 OPEN") {
		t.Fatalf("required gate hidden: %+v", rows["defercoldtools"])
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
