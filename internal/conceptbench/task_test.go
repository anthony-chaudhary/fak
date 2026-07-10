package conceptbench

import (
	"errors"
	"os/exec"
	"path/filepath"
	"testing"
)

// corpusDir is the on-disk task corpus, relative to this package directory.
const corpusDir = "../../experiments/agent-live/conceptbench/tasks"

func goodTask() Task {
	return Task{
		Schema:          TaskSchemaV1,
		ID:              "commit_stamp-0",
		Concept:         ConceptCommitStamp,
		Prompt:          "Ship the change with a stamped commit.",
		FixtureRef:      "fixtures/commit_stamp.fixture.json",
		ExpectedWitness: WitnessDosCommitAudit,
		Difficulty:      DifficultyEasy,
	}
}

func TestTaskValidate_Accepts(t *testing.T) {
	if err := goodTask().Validate(); err != nil {
		t.Fatalf("valid task rejected: %v", err)
	}
}

// A malformed task must be rejected with a typed *TaskError naming the exact
// failure mode (the issue's acceptance gate). The unknown-witness case is the
// load-bearing one: a task whose expected_witness names no known referee.
func TestTaskValidate_TypedRejections(t *testing.T) {
	cases := []struct {
		name string
		kind TaskErrorKind
		mut  func(*Task)
	}{
		{"unknown witness", ErrKindUnknownWitness, func(x *Task) { x.ExpectedWitness = "dos_make_it_pass" }},
		{"empty witness", ErrKindUnknownWitness, func(x *Task) { x.ExpectedWitness = "" }},
		{"witness grades other concept", ErrKindWitnessMismatch, func(x *Task) { x.ExpectedWitness = WitnessDosArbitrate }},
		{"unknown concept", ErrKindUnknownConcept, func(x *Task) { x.Concept = "vibes" }},
		{"bad schema", ErrKindSchema, func(x *Task) { x.Schema = "fak.conceptbench.task.v99" }},
		{"missing prompt", ErrKindMissingField, func(x *Task) { x.Prompt = "  " }},
		{"missing id", ErrKindMissingField, func(x *Task) { x.ID = "" }},
		{"missing fixture_ref", ErrKindMissingField, func(x *Task) { x.FixtureRef = "" }},
		{"bad difficulty", ErrKindBadDifficulty, func(x *Task) { x.Difficulty = "trivial" }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			task := goodTask()
			c.mut(&task)
			err := task.Validate()
			if err == nil {
				t.Fatalf("expected rejection, got nil")
			}
			var te *TaskError
			if !errors.As(err, &te) {
				t.Fatalf("want *TaskError, got %T: %v", err, err)
			}
			if te.Kind != c.kind {
				t.Fatalf("want kind %q, got %q (%v)", c.kind, te.Kind, err)
			}
		})
	}
}

// A pure-file + injected-context fixture must materialize byte-identically across
// two builds — the "golden" reproducibility proof (no git required, so it always
// runs).
func TestFixtureGolden_ByteIdentical_PureFile(t *testing.T) {
	r := FixtureRecipe{
		Schema: FixtureSchemaV1,
		ID:     "golden-pure",
		Files: []FixtureFile{
			{Path: "README.md", Content: "seed\n"},
			{Path: "src/a.go", Content: "package a\n"},
		},
		Inject: &InjectedContext{
			Verdict:      "STALE_BASE",
			RefusalToken: "OFF_TRUNK",
			Leases:       []Lease{{Lane: "bench", LaneKind: "cluster", Tree: []string{"internal/bench/**"}}},
		},
	}
	m1, err := r.Materialize(t.TempDir())
	if err != nil {
		t.Fatalf("build 1: %v", err)
	}
	m2, err := r.Materialize(t.TempDir())
	if err != nil {
		t.Fatalf("build 2: %v", err)
	}
	if m1.Digest() != m2.Digest() {
		t.Fatalf("fixture not byte-identical across builds:\n build1=%s\n build2=%s", m1.Digest(), m2.Digest())
	}
	if len(m1.Files) == 0 {
		t.Fatal("manifest has no files")
	}
}

// A git-backed fixture must also rebuild byte-identically: pinned identity +
// dates make the HEAD SHA deterministic. Skips only if git is unavailable.
func TestFixtureGolden_ByteIdentical_Git(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH; pure-file golden proof still covers the DoD")
	}
	r := FixtureRecipe{
		Schema: FixtureSchemaV1,
		ID:     "golden-git",
		Git: &FixtureGit{
			Commits: []FixtureCommit{{
				Subject:     "feat(x): seed the scratch repo",
				Files:       []FixtureFile{{Path: "x.txt", Content: "hello\n"}},
				AuthorName:  "conceptbench",
				AuthorEmail: "conceptbench@fak.invalid",
				Date:        "2026-01-01T00:00:00 +0000",
			}},
		},
	}
	m1, err := r.Materialize(t.TempDir())
	if err != nil {
		t.Fatalf("build 1: %v", err)
	}
	m2, err := r.Materialize(t.TempDir())
	if err != nil {
		t.Fatalf("build 2: %v", err)
	}
	if m1.HeadSHA == "" {
		t.Fatal("git fixture produced no HEAD SHA")
	}
	if m1.Digest() != m2.Digest() {
		t.Fatalf("git fixture not deterministic: head1=%s head2=%s", m1.HeadSHA, m2.HeadSHA)
	}
}

// The committed on-disk corpus must load, validate, and satisfy the DoD shape:
// >=12 tasks, >=2 per concept, every referenced fixture resolving and parsing.
func TestOnDiskCorpus_LoadsAndCovers(t *testing.T) {
	tasks, err := LoadCorpus(corpusDir)
	if err != nil {
		t.Fatalf("corpus failed to load: %v", err)
	}
	if len(tasks) < 12 {
		t.Fatalf("want >=12 tasks, got %d", len(tasks))
	}
	perConcept := map[Concept]int{}
	for _, task := range tasks {
		perConcept[task.Concept]++
	}
	for _, c := range Concepts() {
		if perConcept[c] < 2 {
			t.Errorf("concept %q has %d tasks, want >=2", c, perConcept[c])
		}
	}
}

// Every corpus fixture must materialize byte-identically across two builds —
// proving the whole corpus (not just the golden pair) is reproducible.
func TestOnDiskCorpus_FixturesRebuild(t *testing.T) {
	tasks, err := LoadCorpus(corpusDir)
	if err != nil {
		t.Fatalf("corpus failed to load: %v", err)
	}
	gitOK := true
	if _, err := exec.LookPath("git"); err != nil {
		gitOK = false
	}
	seen := map[string]bool{}
	ran := 0
	for _, task := range tasks {
		fx := filepath.Join(corpusDir, filepath.FromSlash(task.FixtureRef))
		if seen[fx] {
			continue
		}
		seen[fx] = true
		r, err := LoadFixture(fx)
		if err != nil {
			t.Fatalf("fixture %s: %v", task.FixtureRef, err)
		}
		if r.NeedsGit() && !gitOK {
			continue
		}
		m1, err := r.Materialize(t.TempDir())
		if err != nil {
			t.Fatalf("fixture %s build 1: %v", task.FixtureRef, err)
		}
		m2, err := r.Materialize(t.TempDir())
		if err != nil {
			t.Fatalf("fixture %s build 2: %v", task.FixtureRef, err)
		}
		if m1.Digest() != m2.Digest() {
			t.Fatalf("fixture %s not byte-identical across builds", task.FixtureRef)
		}
		ran++
	}
	if ran == 0 {
		t.Fatal("no fixture was rebuilt; corpus proves nothing")
	}
}
