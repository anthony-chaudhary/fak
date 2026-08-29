package studyprio

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestBuildActualFiveClusterLedgerDeterministically(t *testing.T) {
	source := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	a, as, err := Build(BuildOptions{source})
	if err != nil {
		t.Fatal(err)
	}
	b, bs, err := Build(BuildOptions{source})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(as, bs) {
		t.Fatal("nondeterministic rebuild")
	}
	if as.SourceClusterCount != 5 || as.CandidateCount != 2 || as.QueueCount != 2 {
		t.Fatalf("counts=%+v", as)
	}
	if want := []string{"native-vllm-ir", "allocator-fragmentation"}; !reflect.DeepEqual(as.QueueCandidateIDs, want) {
		t.Fatalf("queue=%v want=%v", as.QueueCandidateIDs, want)
	}
	owners := map[string]string{}
	for _, c := range a.Candidates {
		for _, m := range c.SourceMappings {
			owners[m.ClusterID] = c.ID
		}
	}
	if len(owners) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture defines exactly five owner classes and requires all to be ranked
		t.Fatalf("source mapping count=%d", len(owners))
	}
	for _, id := range vllmIRClusters {
		if owners[id] != "native-vllm-ir" {
			t.Fatalf("%s owner=%s", id, owners[id])
		}
	}
	if owners[requiredSourceClusters[4]] != "allocator-fragmentation" {
		t.Fatal("allocator merged into IR")
	}
	if a.Sensitivity.CandidateID != "native-vllm-ir" || a.Sensitivity.BaselineScore != 68 || len(a.Sensitivity.Steps) != 3 {
		t.Fatalf("sensitivity=%+v", a.Sensitivity)
	}
	if a.Sensitivity.Steps[0].QueueFirst != "native-vllm-ir" || a.Sensitivity.Steps[1].QueueFirst != "native-vllm-ir" || a.Sensitivity.Steps[2].QueueFirst != "allocator-fragmentation" {
		t.Fatalf("sensitivity queue flip=%+v", a.Sensitivity.Steps)
	}
}

func TestValidateFilesDetectsArtifactDrift(t *testing.T) {
	source := filepath.Join("..", "..", "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	l, s, err := Build(BuildOptions{source})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	lp := filepath.Join(dir, "ledger.json")
	sp := filepath.Join(dir, "summary.json")
	write(t, lp, mustLedger(t, l))
	write(t, sp, mustSummary(t, s))
	opts := ValidateOptions{source, lp, sp}
	if err := ValidateFiles(opts); err != nil {
		t.Fatal(err)
	}
	l.Queue[0], l.Queue[1] = l.Queue[1], l.Queue[0]
	write(t, lp, mustLedger(t, l))
	if err := ValidateFiles(opts); !errors.Is(err, ErrInvalid) {
		t.Fatalf("tamper error=%v", err)
	}
}

func TestBuildRejectsSourceCoverageViolations(t *testing.T) {
	base := testSource()
	tests := []struct {
		name string
		edit func(*sourceLedger)
	}{
		{"missing", func(s *sourceLedger) { s.Joins = s.Joins[1:] }},
		{"extra", func(s *sourceLedger) { s.Joins = append(s.Joins, testJoin("other:body:work")) }},
		{"duplicate", func(s *sourceLedger) { s.Joins = append(s.Joins, s.Joins[0]) }},
		{"missing-checksum", func(s *sourceLedger) { s.Joins[0].MembersSHA256 = "" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := cloneSource(t, base)
			tt.edit(&s)
			if _, _, err := Build(BuildOptions{writeSource(t, s)}); !errors.Is(err, ErrInvalid) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestValidateRejectsGateScoreNativeAndMappingViolations(t *testing.T) {
	l, _, err := Build(BuildOptions{writeSource(t, testSource())})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*Ledger)
	}{
		{"failed-gate", func(x *Ledger) { x.Candidates[0].HardGates[0].Pass = false }},
		{"missing-gate", func(x *Ledger) { x.Candidates[0].HardGates = x.Candidates[0].HardGates[1:] }},
		{"score", func(x *Ledger) { x.Candidates[0].Score++ }},
		{"engine", func(x *Ledger) { x.Candidates[0].Witness.Engine = "llama.cpp" }},
		{"model", func(x *Ledger) { x.Candidates[0].Witness.Model = "Qwen3.6" }},
		{"fallback", func(x *Ledger) { x.Candidates[0].Execution.FallbackAllowed = true }},
		{"duplicate-source", func(x *Ledger) { x.Candidates[1].SourceMappings[0] = x.Candidates[0].SourceMappings[0] }},
		{"allocator-merge", func(x *Ledger) {
			for i := range x.Candidates {
				if x.Candidates[i].ID == "allocator-fragmentation" {
					x.Candidates[i].MergeJustification = "merge"
				}
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := cloneLedger(t, l)
			tt.edit(&x)
			if err := Validate(x); !errors.Is(err, ErrInvalid) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestQueueRejectsMissingDependenciesCyclesAndGateViolations(t *testing.T) {
	l, _, err := Build(BuildOptions{writeSource(t, testSource())})
	if err != nil {
		t.Fatal(err)
	}
	missing := cloneLedger(t, l).Candidates
	missing[0].Dependencies = []string{"missing"}
	if _, err := buildQueue(missing); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing error=%v", err)
	}
	cycle := cloneLedger(t, l).Candidates
	cycle[0].Dependencies = []string{cycle[1].ID}
	cycle[1].Dependencies = []string{cycle[0].ID}
	if _, err := buildQueue(cycle); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cycle error=%v", err)
	}
	gate := cloneLedger(t, l).Candidates
	gate[0].HardGates[0].Pass = false
	if _, err := buildQueue(gate); !errors.Is(err, ErrInvalid) {
		t.Fatalf("gate error=%v", err)
	}
}

func TestQueueDependencyAndDocumentedTieBreaks(t *testing.T) {
	l, _, err := Build(BuildOptions{writeSource(t, testSource())})
	if err != nil {
		t.Fatal(err)
	}
	deps := cloneLedger(t, l).Candidates
	deps[0].Dependencies = []string{deps[1].ID}
	deps[0].Score = 1000
	q, err := buildQueue(deps)
	if err != nil {
		t.Fatal(err)
	}
	if q[0].CandidateID != deps[1].ID {
		t.Fatalf("dependency not first: %+v", q)
	}
	tied := cloneLedger(t, l).Candidates
	tied[0].Score = tied[1].Score
	tied[0].Centrality = "Core"
	tied[1].Centrality = "Core"
	q, err = buildQueue(tied)
	if err != nil {
		t.Fatal(err)
	}
	want := tied[0].ID
	if tied[1].ID < want {
		want = tied[1].ID
	}
	if q[0].CandidateID != want {
		t.Fatalf("tie first=%s want=%s", q[0].CandidateID, want)
	}
}

func testSource() sourceLedger {
	s := sourceLedger{Schema: sourceSchema, Cutoff: "2026-08-27T04:36:00Z", SourceRevision: "fak@test"}
	for _, id := range requiredSourceClusters {
		s.Joins = append(s.Joins, testJoin(id))
	}
	return s
}
func testJoin(id string) sourceJoin {
	p := strings.Split(id, ":")
	return sourceJoin{ClusterID: id, Mechanism: p[0], Signal: strings.ReplaceAll(p[2], "-", " "), Rule: "mechanism." + p[0] + "." + p[1], Actionable: true, Disposition: "uncovered", Evidence: sourceEvidence{"sha256:evidence-" + id}, MembersSHA256: "sha256:members-" + id}
}
func writeSource(t *testing.T, s sourceLedger) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "source.json")
	write(t, p, b)
	return p
}
func write(t *testing.T, p string, b []byte) {
	t.Helper()
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
}
func mustLedger(t *testing.T, l Ledger) []byte {
	t.Helper()
	b, err := MarshalLedger(l)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func mustSummary(t *testing.T, s Summary) []byte {
	t.Helper()
	b, err := MarshalSummary(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func cloneLedger(t *testing.T, l Ledger) Ledger {
	t.Helper()
	b, _ := json.Marshal(l)
	var x Ledger
	if err := json.Unmarshal(b, &x); err != nil {
		t.Fatal(err)
	}
	return x
}
func cloneSource(t *testing.T, s sourceLedger) sourceLedger {
	t.Helper()
	b, _ := json.Marshal(s)
	var x sourceLedger
	if err := json.Unmarshal(b, &x); err != nil {
		t.Fatal(err)
	}
	return x
}
