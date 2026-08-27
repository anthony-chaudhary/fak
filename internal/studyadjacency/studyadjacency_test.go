package studyadjacency

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestValidateAcceptsCompleteRequiredSet(t *testing.T) {
	if err := Validate(validManifest()); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestValidateRejectsRequiredFailures(t *testing.T) {
	tests := []struct {
		name   string
		want   string
		mutate func(*Manifest)
	}{
		{
			name: "wrong schema",
			want: "schema must be",
			mutate: func(manifest *Manifest) {
				manifest.Schema = "fak-study-adjacency/2"
			},
		},
		{
			name: "missing required core repository",
			want: `required core repository "llm-d/llm-d" is not declared`,
			mutate: func(manifest *Manifest) {
				manifest.DeclaredRepositories = manifest.DeclaredRepositories[:len(manifest.DeclaredRepositories)-1]
				manifest.Members = manifest.Members[:len(manifest.Members)-1]
			},
		},
		{
			name: "undeclared candidate repository link",
			want: "is not declared",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Candidates[0].RepositoryLinks = append(manifest.Members[0].Candidates[0].RepositoryLinks, Repository{Owner: "other", Repo: "runtime"})
			},
		},
		{
			name: "missing canonical owner",
			want: ".owner is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Repository.Owner = ""
			},
		},
		{
			name: "missing canonical repo",
			want: ".repo is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Repository.Repo = ""
			},
		},
		{
			name: "missing revision pin",
			want: ".revision must be a full 40-hex commit",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Pin.Revision = ""
			},
		},
		{
			name: "missing cutoff pin",
			want: ".cutoff is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Pin.Cutoff = ""
			},
		},
		{
			name: "missing observation pin",
			want: ".observed_at is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Pin.ObservedAt = ""
			},
		},
		{
			name: "duplicate member",
			want: "duplicates member repository",
			mutate: func(manifest *Manifest) {
				manifest.Members = append(manifest.Members, manifest.Members[0])
			},
		},
		{
			name: "duplicate candidate ID across repositories",
			want: "duplicates candidate from",
			mutate: func(manifest *Manifest) {
				manifest.Members[1].Candidates = []Candidate{candidateFor(manifest.Members[1].Repository, manifest.Members[0].Candidates[0].ID)}
			},
		},
		{
			name: "missing inclusion rationale",
			want: ".inclusion_rationale is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].InclusionRationale = ""
			},
		},
		{
			name: "missing decision relation",
			want: ".decision_relation is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].DecisionRelation = ""
			},
		},
		{
			name: "missing receipt status",
			want: ".status is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].SourceClassReceipts[0].Status = ""
			},
		},
		{
			name: "missing receipt notes",
			want: ".notes is required",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].SourceClassReceipts[0].Notes = ""
			},
		},
		{
			name: "complete receipt missing terminal receipt",
			want: ".terminal_receipt is required when status is complete",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].SourceClassReceipts[0].TerminalReceipt = ""
			},
		},
		{
			name: "candidate lacks linkage",
			want: "requires a vllm_mechanism_link or frontier_changing_contrast",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Candidates[0].VLLMMechanismLink = ""
				manifest.Members[0].Candidates[0].FrontierChangingContrast = ""
			},
		},
		{
			name: "declared member absent",
			want: "was not processed",
			mutate: func(manifest *Manifest) {
				manifest.Members = manifest.Members[:len(manifest.Members)-1]
			},
		},
		{
			name: "declared member not processed",
			want: ".processed must be true",
			mutate: func(manifest *Manifest) {
				manifest.Members[0].Processed = false
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest := validManifest()
			test.mutate(&manifest)
			err := Validate(manifest)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestReadIsStrictAndValidates(t *testing.T) {
	data, err := CanonicalJSON(validManifest())
	if err != nil {
		t.Fatal(err)
	}
	got, err := Read(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Read(valid) error = %v", err)
	}
	if got.Schema != Schema {
		t.Fatalf("Read(valid).Schema = %q", got.Schema)
	}

	withUnknown := bytes.Replace(data, []byte(`"title":`), []byte(`"unknown":true,"title":`), 1)
	if _, err := Read(bytes.NewReader(withUnknown)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("Read(unknown) error = %v", err)
	}
	if _, err := Read(strings.NewReader(string(data) + `{}`)); err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
		t.Fatalf("Read(trailing) error = %v", err)
	}
}

func TestRenderingAndCanonicalJSONAreDeterministic(t *testing.T) {
	manifest := validManifest()
	wantManifest := cloneManifest(t, manifest)

	firstMarkdown, err := RenderMarkdown(manifest)
	if err != nil {
		t.Fatal(err)
	}
	firstJSON, err := CanonicalJSON(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(manifest, wantManifest) {
		t.Fatal("rendering mutated the manifest")
	}

	reordered := cloneManifest(t, manifest)
	slices.Reverse(reordered.Scope.InclusionCriteria)
	slices.Reverse(reordered.Scope.ExclusionCriteria)
	slices.Reverse(reordered.DeclaredRepositories)
	slices.Reverse(reordered.Members)
	for i := range reordered.Members {
		slices.Reverse(reordered.Members[i].SourceClassReceipts)
		slices.Reverse(reordered.Members[i].Candidates)
		for j := range reordered.Members[i].Candidates {
			slices.Reverse(reordered.Members[i].Candidates[j].RepositoryLinks)
		}
	}

	secondMarkdown, err := RenderMarkdown(reordered)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := CanonicalJSON(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstMarkdown, secondMarkdown) {
		t.Fatalf("Markdown changed with input order\n--- first ---\n%s\n--- second ---\n%s", firstMarkdown, secondMarkdown)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("canonical JSON changed with input order\n--- first ---\n%s\n--- second ---\n%s", firstJSON, secondJSON)
	}
	if !bytes.Contains(firstMarkdown, []byte("## Processed members (6/6)")) {
		t.Fatalf("render does not prove processing completeness:\n%s", firstMarkdown)
	}
}

func TestCheckedInVLLMAdjacencyArtifacts(t *testing.T) {
	root := filepath.Join("..", "..")
	manifestPath := filepath.Join(root, "docs", "research", "inventory", "vllm-related-system-adjacency-v1.json")
	summaryPath := filepath.Join(root, "docs", "research", "inventory", "vllm-related-system-adjacency-v1.md")

	rawManifest, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Read(bytes.NewReader(rawManifest))
	if err != nil {
		t.Fatalf("Read(checked-in manifest) error = %v", err)
	}
	if got, want := len(manifest.DeclaredRepositories), 14; got != want {
		t.Fatalf("declared repositories = %d, want %d", got, want)
	}
	if got, want := len(manifest.Members), 14; got != want {
		t.Fatalf("processed members = %d, want %d", got, want)
	}
	candidates := 0
	for _, member := range manifest.Members {
		candidates += len(member.Candidates)
	}
	if candidates != 14 {
		t.Fatalf("candidates = %d, want 14", candidates)
	}
	if manifest.Anchor.Pin.Revision != "f18d0ba90d972a852a351c98be3f42b31372cfe4" ||
		manifest.Anchor.Pin.Cutoff != "2026-08-26T22:35:00Z" ||
		manifest.Anchor.NormalizedRecords != 53848 ||
		manifest.Anchor.SHA256 != "2a66d4876aee3811eb200c0884c6558a5f3ac86c90b6c7f8b92f45b85fe671b2" {
		t.Fatalf("anchor witness drifted: %+v", manifest.Anchor)
	}

	canonical, err := CanonicalJSON(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(rawManifest, canonical) {
		t.Fatal("checked-in manifest is not the canonical deterministic encoding")
	}
	rendered, err := RenderMarkdown(manifest)
	if err != nil {
		t.Fatal(err)
	}
	rawSummary, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(rawSummary, rendered) {
		t.Fatal("checked-in Markdown summary does not match deterministic rendering")
	}
}

func validManifest() Manifest {
	cutoff := "2026-08-26T22:35:00Z"
	observed := "2026-08-27T00:15:00Z"
	manifest := Manifest{
		Schema: Schema,
		ID:     "vllm-related-systems-2026-08-26",
		Title:  "vLLM related-system adjacency",
		Scope: Scope{
			BoundedMeaning:    "The named runtime peers plus criterion-admitted vLLM satellites.",
			InclusionCriteria: []string{"Named by issue #9275", "Changes a vLLM or FAK decision"},
			ExclusionCriteria: []string{"No mechanism-level decision linkage"},
		},
		Anchor: CorpusWitness{
			Repository:        Repository{Owner: "vllm-project", Repo: "vllm"},
			Pin:               Pin{Revision: strings.Repeat("a", 40), Cutoff: cutoff, ObservedAt: observed},
			NormalizedRecords: 53848,
			SHA256:            strings.Repeat("b", 64),
		},
	}
	revisions := []string{"c", "d", "e", "f", "1", "2"}
	for i, repository := range RequiredCoreRepositories {
		manifest.DeclaredRepositories = append(manifest.DeclaredRepositories, repository)
		member := Member{
			Name:               repository.Repo,
			Repository:         repository,
			Pin:                Pin{Revision: strings.Repeat(revisions[i], 40), Cutoff: cutoff, ObservedAt: observed},
			Processed:          true,
			InclusionRationale: "Named minimum with a serving mechanism relevant to the anchor.",
			DecisionRelation:   "Tests whether a vLLM scheduling choice transfers to FAK.",
			FreshnessNotes:     "Revision selected at the shared cutoff; metadata observed later.",
			PartialNotes:       "Forge corpus is partial except where terminal receipts say complete.",
			SourceClassReceipts: []SourceClassReceipt{
				{Class: "forge_metadata", Status: ReceiptComplete, TerminalReceipt: "scratch/receipt.json", Notes: "Terminal repository metadata receipt."},
				{Class: "runtime_source", Status: ReceiptPartial, Notes: "Mechanism-focused source slice only."},
			},
		}
		if i == 0 {
			member.Candidates = []Candidate{candidateFor(repository, "scheduler-contrast")}
		}
		manifest.Members = append(manifest.Members, member)
	}
	return manifest
}

func candidateFor(repository Repository, id string) Candidate {
	return Candidate{
		ID:                id,
		Title:             "Scheduler contrast",
		Rationale:         "Changes the admissible scheduler design frontier.",
		RepositoryLinks:   []Repository{repository},
		VLLMMechanismLink: "vllm:scheduler/continuous-batching",
	}
}

func cloneManifest(t *testing.T, manifest Manifest) Manifest {
	t.Helper()
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	var clone Manifest
	if err := json.Unmarshal(data, &clone); err != nil {
		t.Fatal(err)
	}
	return clone
}
