package mlpscore

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

type memorySnapshot map[string][]byte

func (s memorySnapshot) ReadFile(rel string) ([]byte, error) {
	b, ok := s[rel]
	if !ok {
		return nil, errors.New("not committed")
	}
	return append([]byte(nil), b...), nil
}

func (s memorySnapshot) Exists(rel string) bool {
	_, ok := s[rel]
	return ok
}

func testOpts() FoldOpts {
	return FoldOpts{Workspace: "/repo", Commit: "abcdef1234567890", GeneratedAt: "2026-07-10T00:00:00Z", Date: "2026-07-10"}
}

func completeSnapshot(t *testing.T) memorySnapshot {
	t.Helper()
	snapshot := memorySnapshot{}
	for _, spec := range criteria {
		manifest := WitnessManifest{Schema: WitnessSchema, Criterion: spec.key}
		for _, claimKey := range spec.requiredClaims {
			proof := "docs/mlp/proofs/" + spec.key + "-" + claimKey + ".md"
			manifest.Claims = append(manifest.Claims, WitnessClaim{
				Key: claimKey, Kind: "captured-run", Path: proof, Command: "fak proof " + claimKey,
			})
			snapshot[proof] = []byte("captured proof")
		}
		raw, err := json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
		snapshot[WitnessDir+"/"+spec.key+".json"] = raw
	}
	return snapshot
}

func TestGradeAbsentWitnessesIsNotYet(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	if s.Lovable || s.OK || s.MLPVerdict != "not-yet" || s.Verdict != "ACTION" {
		t.Fatalf("unexpected empty grade: %+v", s)
	}
	if s.Witnessed != 0 || s.Total != len(criteria) || s.Debt != len(criteria) {
		t.Fatalf("counts = witnessed %d total %d debt %d", s.Witnessed, s.Total, s.Debt)
	}
	for _, row := range s.Criteria {
		if row.Grade != GradeNotYet || len(row.Missing) == 0 {
			t.Fatalf("row should be an explained not-yet: %+v", row)
		}
		if !strings.HasPrefix(row.WitnessRef, WitnessDir+"/") {
			t.Fatalf("witness ref %q is outside %s", row.WitnessRef, WitnessDir)
		}
		if row.Evidence == nil {
			t.Fatalf("evidence for %s must encode as [], not null", row.Key)
		}
	}
}

func TestGradeCompleteCommittedWitnessesIsLovable(t *testing.T) {
	s := Grade(completeSnapshot(t), testOpts())
	if !s.Lovable || !s.OK || s.MLPVerdict != "lovable" || s.Verdict != "OK" {
		t.Fatalf("unexpected complete grade: %+v", s)
	}
	if s.Witnessed != s.Total || s.Debt != 0 {
		t.Fatalf("counts = witnessed %d total %d debt %d", s.Witnessed, s.Total, s.Debt)
	}
}

func TestManifestRequiresEveryNamedClaimAndCommittedProof(t *testing.T) {
	spec := criteria[0]
	manifest := WitnessManifest{
		Schema: WitnessSchema, Criterion: spec.key,
		Claims: []WitnessClaim{{
			Key: spec.requiredClaims[0], Kind: "test", Path: "cmd/fak/up_test.go", Command: "go test ./cmd/fak -run TestUp",
		}},
	}
	raw, _ := json.Marshal(manifest)
	snapshot := memorySnapshot{WitnessDir + "/" + spec.key + ".json": raw}
	s := Grade(snapshot, testOpts())
	row := s.Criteria[0]
	joined := strings.Join(row.Missing, " | ")
	for _, want := range []string{
		"proof is not committed: cmd/fak/up_test.go",
		"missing claim: " + spec.requiredClaims[1],
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in %q", want, joined)
		}
	}
	if row.Grade != GradeNotYet {
		t.Fatalf("incomplete manifest graded %q", row.Grade)
	}
}

func TestManifestRejectsWrongSchemaDuplicateClaimAndTraversal(t *testing.T) {
	spec := criteria[3]
	manifest := WitnessManifest{
		Schema: "claimed/1", Criterion: spec.key,
		Claims: []WitnessClaim{
			{Key: spec.requiredClaims[0], Kind: "captured-run", Path: "../outside.md", Command: "run it"},
			{Key: spec.requiredClaims[0], Kind: "captured-run", Path: "docs/other.md", Command: "run it"},
		},
	}
	raw, _ := json.Marshal(manifest)
	snapshot := memorySnapshot{WitnessDir + "/" + spec.key + ".json": raw, "docs/other.md": []byte("proof")}
	s := Grade(snapshot, testOpts())
	var row Criterion
	for _, candidate := range s.Criteria {
		if candidate.Key == spec.key {
			row = candidate
			break
		}
	}
	joined := strings.Join(row.Missing, " | ")
	if !strings.Contains(joined, "schema") || !strings.Contains(joined, "duplicate claim") {
		t.Fatalf("invalid manifest gaps = %q", joined)
	}
	if row.Grade != GradeNotYet {
		t.Fatalf("invalid manifest graded %q", row.Grade)
	}
}

func TestManifestRequiresReproductionCommand(t *testing.T) {
	spec := criteria[3]
	proof := "docs/mlp/proofs/init-agent.md"
	manifest := WitnessManifest{
		Schema: WitnessSchema, Criterion: spec.key,
		Claims: []WitnessClaim{{Key: spec.requiredClaims[0], Kind: "captured-run", Path: proof}},
	}
	raw, _ := json.Marshal(manifest)
	snapshot := memorySnapshot{
		WitnessDir + "/" + spec.key + ".json": raw,
		proof:                                 []byte("proof"),
	}
	s := Grade(snapshot, testOpts())
	for _, row := range s.Criteria {
		if row.Key == spec.key && !strings.Contains(strings.Join(row.Missing, " "), "no reproduction command") {
			t.Fatalf("missing reproduction command was not reported: %+v", row)
		}
	}
}

func TestJSONSchemaStable(t *testing.T) {
	raw, err := json.Marshal(Grade(memorySnapshot{}, testOpts()))
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Schema     string `json:"schema"`
		MLPVerdict string `json:"mlp_verdict"`
		Debt       int    `json:"mlp_debt"`
		Criteria   []struct {
			Key            string   `json:"key"`
			Grade          string   `json:"grade"`
			WitnessRef     string   `json:"witness_ref"`
			RequiredClaims []string `json:"required_claims"`
			Evidence       []any    `json:"evidence"`
		} `json:"criteria"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Schema != Schema || payload.MLPVerdict != "not-yet" || payload.Debt != len(criteria) {
		t.Fatalf("stable head drifted: %+v", payload)
	}
	if len(payload.Criteria) != len(criteria) {
		t.Fatalf("criteria = %d, want %d", len(payload.Criteria), len(criteria))
	}
	for i, spec := range criteria {
		row := payload.Criteria[i]
		if row.Key != spec.key || row.Grade != GradeNotYet || row.WitnessRef == "" || len(row.RequiredClaims) == 0 || row.Evidence == nil {
			t.Fatalf("criteria[%d] contract drifted: %+v", i, row)
		}
	}
}

func TestRenderMarkdownLinksWitnessesAndOwners(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	md := RenderMarkdown(s)
	for _, want := range []string{
		"MLP scorecard - first lovable cut",
		"**NOT YET**",
		"| Criterion | Workstream | Grade | Witness | Owners |",
		"](docs/mlp/witnesses/",
		"https://github.com/anthony-chaudhary/fak/issues/3420",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
	if RenderMarkdown(s) != md {
		t.Fatal("markdown render is not deterministic")
	}
}

func TestRenderHumanShowsBothVerdicts(t *testing.T) {
	if got := Render(Grade(memorySnapshot{}, testOpts())); !strings.Contains(got, "verdict: NOT YET") {
		t.Fatalf("not-yet render:\n%s", got)
	}
	if got := Render(Grade(completeSnapshot(t), testOpts())); !strings.Contains(got, "verdict: LOVABLE") {
		t.Fatalf("lovable render:\n%s", got)
	}
}

func TestWorkstreamsMatchIssueContract(t *testing.T) {
	got := Workstreams()
	want := []string{"B1", "B3", "C1/C3", "D2", "D5"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("workstreams = %v, want %v", got, want)
	}
}
