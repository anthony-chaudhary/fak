package studylink

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func TestBuildCompleteCoverageUncoveredAndDeterministic(t *testing.T) {
	fx := writeFixture(t)
	one, summary, err := Build(fx.options())
	if err != nil {
		t.Fatal(err)
	}
	two, _, err := Build(fx.options())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one, two) {
		t.Fatal("build is not deterministic")
	}
	if len(one.Joins) != 6 || summary.Total != 6 || summary.Actionable != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture cardinalities are fixed evidence for complete join and ledger preservation
		t.Fatalf("coverage summary=%+v joins=%d", summary, len(one.Joins))
	}
	want := map[string]Disposition{
		"scheduling_batching:title:continuous-batching":                    Landed,
		"kv_cache:title:paged-attention":                                   OpenExact,
		"reliability_security:body:retry-storm":                            Partial,
		"speculative_decoding:title:dflash":                                Uncovered,
		"distributed_parallelism:title:tensor-parallel":                    Conflict,
		"explicit_non_candidate:disposition:release-metadata-noncandidate": Obsolete,
	}
	for _, join := range one.Joins {
		if join.Disposition != want[join.ClusterID] {
			t.Errorf("%s=%s want %s", join.ClusterID, join.Disposition, want[join.ClusterID])
		}
		if join.Evidence.Digest == "" || join.Evidence.Query == "" {
			t.Errorf("%s lacks reproducible evidence", join.ClusterID)
		}
		if join.Disposition == Uncovered && len(join.Artifacts) != 0 {
			t.Errorf("uncovered %s has artifacts", join.ClusterID)
		}
	}
	if summary.Counts[Uncovered] != 1 || summary.Counts[Conflict] != 1 || len(summary.ManualReview) < 2 {
		t.Fatalf("summary=%+v", summary)
	}
}

func TestBuildSeedJoinRejectsNonAffirmativeWitnessMode(t *testing.T) {
	seed := witnessSeed{Issue: 7, Mode: Uncovered}
	issues := map[int]ForgeRecord{7: {Number: 7, State: "open"}}
	join, _ := buildSeedJoin(Join{}, seed, issues, t.TempDir(), "revision")
	if join.Disposition != Conflict || !join.ManualReview || join.Confidence != "invalid-explicit-witness-mode" {
		t.Fatalf("non-affirmative witness mode did not fail closed: %+v", join)
	}
}

func TestValidatorRejectsBrokenIssueStatePathDuplicateAndMissingCoverage(t *testing.T) {
	fx := writeFixture(t)
	base, _, err := Build(fx.options())
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*Ledger){
		"missing issue id": func(l *Ledger) { j := joinByDisposition(l, OpenExact); j.Artifacts[0].ID = "999"; refresh(j) },
		"closed claimed open": func(l *Ledger) {
			j := joinByDisposition(l, OpenExact)
			j.Artifacts[0].State = "closed"
			for i := range l.CapturedIssues {
				if l.CapturedIssues[i].Number == 101 {
					l.CapturedIssues[i].State = "closed"
				}
			}
			refresh(j)
		},
		"broken repo path": func(l *Ledger) {
			j := joinByDisposition(l, Landed)
			for i := range j.Artifacts {
				if j.Artifacts[i].Kind == "repo_path" {
					j.Artifacts[i].Path = "internal/missing/nope.go"
					j.Artifacts[i].ID = j.Artifacts[i].Path
					break
				}
			}
			refresh(j)
		},
		"duplicate exact": func(l *Ledger) {
			src := joinByDisposition(l, OpenExact).Artifacts[0]
			dst := joinByDisposition(l, Landed)
			dst.Artifacts = append(dst.Artifacts, src)
			refresh(dst)
		},
		"missing cluster coverage": func(l *Ledger) { l.Joins = l.Joins[:len(l.Joins)-1] },
	}
	index, _, _ := readIndex(fx.index)
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			l := cloneLedger(t, base)
			mutate(&l)
			if !errors.Is(ValidateStructure(l, &index, fx.repo), ErrInvalid) {
				t.Fatal("invalid ledger admitted")
			}
		})
	}
}

func TestCheckedArtifactsCoverCompactIndex(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	ledgerPath := filepath.Join(root, "docs", "research", "vllm-fak-join-2026-08-27", "ledger.json")
	if _, err := os.Stat(ledgerPath); errors.Is(err, os.ErrNotExist) {
		t.Skip("checked ledger is generated later in the issue workflow")
	}
	ledger, err := ReadLedger(ledgerPath)
	if err != nil {
		t.Fatal(err)
	}
	index, _, err := readIndex(filepath.Join(root, "docs", "research", "vllm-classification-2026-08-26", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateStructure(ledger, &index, root); err != nil {
		t.Fatal(err)
	}
	if len(ledger.Joins) != 193 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture cardinalities are fixed evidence for complete join and ledger preservation
		t.Fatalf("checked ledger joins=%d want 193", len(ledger.Joins))
	}
	for _, join := range ledger.Joins {
		if join.Actionable && join.Disposition == "" {
			t.Fatalf("actionable cluster unclassified: %s", join.ClusterID)
		}
	}
	wantDisposition := map[string]Disposition{
		"apis_tool_calling_structured_output:body:guided-decoding": Landed,
		"kv_cache:title:paged-attention":                           Landed,
		"observability_operations:body:helm-chart":                 OpenExact,
		"kv_cache:title:prefix-cache":                              Partial,
		"distributed_parallelism:title:tensor-parallel":            Partial,
		"scheduling_batching:title:continuous-batching":            Conflict,
	}
	for clusterID, disposition := range wantDisposition {
		found := false
		for _, join := range ledger.Joins {
			if join.ClusterID == clusterID {
				found = true
				if join.Disposition != disposition {
					t.Fatalf("%s=%s want %s", clusterID, join.Disposition, disposition)
				}
				if (disposition == Partial || disposition == Conflict) && !join.ManualReview {
					t.Fatalf("%s must remain manual review", clusterID)
				}
			}
		}
		if !found {
			t.Fatalf("checked ledger lacks sample %s", clusterID)
		}
	}
	forgePath := filepath.Join(root, "_scratch", "vllm-fak-corpus", "fak-forge.json")
	if _, err := os.Stat(forgePath); err == nil {
		if err := ValidateFiles(ValidateOptions{
			LedgerPath:    ledgerPath,
			IndexPath:     filepath.Join(root, "docs", "research", "vllm-classification-2026-08-26", "index.json"),
			ForgePath:     forgePath,
			AdjacencyPath: filepath.Join(root, "docs", "research", "inventory", "vllm-related-system-adjacency-v1.json"),
			RepoRoot:      root,
		}); err != nil {
			t.Fatalf("full checked-artifact validation: %v", err)
		}
	}
}

type fixture struct{ repo, index, forge, adj string }

func (f fixture) options() BuildOptions {
	return BuildOptions{IndexPath: f.index, ForgePath: f.forge, AdjacencyPath: f.adj, RepoRoot: f.repo}
}

func writeFixture(t *testing.T) fixture {
	t.Helper()
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "internal/model/batch.go"), "package model\n")
	mustWrite(t, filepath.Join(root, "docs/ops/retry-storm.md"), "retry storm\n")
	run(t, root, "git", "init", "-q")
	run(t, root, "git", "config", "user.email", "test@example.com")
	run(t, root, "git", "config", "user.name", "Test")
	run(t, root, "git", "add", ".")
	run(t, root, "git", "commit", "-qm", "feat: fixture (#36)")
	clusters := []Cluster{
		{Key: "scheduling_batching:title:continuous-batching", Mechanism: "scheduling_batching", Rule: "mechanism.scheduling.title", Signal: "continuous batching", Actionable: true, Confidence: "medium", MembersChecksum: "sha256:a"},
		{Key: "kv_cache:title:paged-attention", Mechanism: "kv_cache", Rule: "mechanism.kv.title", Signal: "paged attention", Actionable: true, Confidence: "medium", MembersChecksum: "sha256:b"},
		{Key: "reliability_security:body:retry-storm", Mechanism: "reliability_security", Rule: "mechanism.reliability.body", Signal: "retry storm", Actionable: true, Confidence: "medium", MembersChecksum: "sha256:c"},
		{Key: "speculative_decoding:title:dflash", Mechanism: "speculative_decoding", Rule: "mechanism.spec.title", Signal: "dflash", Actionable: true, Confidence: "medium", MembersChecksum: "sha256:d"},
		{Key: "distributed_parallelism:title:tensor-parallel", Mechanism: "distributed_parallelism", Rule: "mechanism.distributed.title", Signal: "tensor parallel", Actionable: true, Confidence: "medium", MembersChecksum: "sha256:e"},
		{Key: "explicit_non_candidate:disposition:release-metadata-noncandidate", Mechanism: "explicit_non_candidate", Rule: "disposition.release", Signal: "release_metadata_noncandidate", Actionable: false, Confidence: "high", MembersChecksum: "sha256:f"},
	}
	index := CompactIndex{Schema: "fak-study-compact-index/1", ClustersChecksum: "sha256:clusters", Clusters: clusters}
	indexPath := filepath.Join(root, "index.json")
	writeJSON(t, indexPath, index)
	records := []ForgeRecord{
		{Source: "issues", Kind: "issue", Number: 36, Title: "feat: continuous batching scheduler", Body: "Implemented at internal/model/batch.go with tests.", State: "closed", URL: "https://example.test/36"},
		{Source: "issues", Kind: "issue", Number: 101, Title: "fix: paged attention eviction", Body: "Open design.", State: "open", URL: "https://example.test/101"},
		{Source: "issues", Kind: "issue", Number: 102, Title: "ops: retry handling", Body: "A retry storm needs manual analysis.", State: "open", URL: "https://example.test/102"},
		{Source: "issues", Kind: "issue", Number: 103, Title: "feat: tensor parallel runtime", Body: "Candidate one.", State: "open", URL: "https://example.test/103"},
		{Source: "issues", Kind: "issue", Number: 104, Title: "fix: tensor parallel witness", Body: "Candidate two.", State: "closed", URL: "https://example.test/104"},
	}
	forge := ForgeCorpus{Schema: "fak-studyforge-corpus/1", Receipt: ForgeReceipt{Schema: "fak-studyforge-receipt/1", Repository: "anthony-chaudhary/fak", Revision: "fixture-rev", Cutoff: "2026-08-27T04:36:00Z", Status: "complete", Sources: []ForgeReceiptSource{{Name: "issues", Status: "complete"}}}, Records: records}
	forgePath := filepath.Join(root, "forge.json")
	writeJSON(t, forgePath, forge)
	adj := map[string]any{"schema": "fak-study-adjacency/1", "id": "fixture-adjacency", "members": []any{map[string]any{"name": "vLLM"}}}
	adjPath := filepath.Join(root, "adj.json")
	writeJSON(t, adjPath, adj)
	return fixture{root, indexPath, forgePath, adjPath}
}

func joinByDisposition(l *Ledger, d Disposition) *Join {
	for i := range l.Joins {
		if l.Joins[i].Disposition == d {
			return &l.Joins[i]
		}
	}
	panic("missing disposition")
}
func refresh(j *Join) { j.Evidence.Digest = evidenceDigest(*j) }
func cloneLedger(t *testing.T, l Ledger) Ledger {
	t.Helper()
	b, _ := json.Marshal(l)
	var out Ledger
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}
func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	mustWrite(t, path, string(b))
}
func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
func run(t *testing.T, dir, name string, args ...string) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%s %v: %v: %s", name, args, err, out)
	}
}
