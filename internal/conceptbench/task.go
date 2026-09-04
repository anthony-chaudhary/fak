// Task corpus + fixture schema for the concept benchmark (#2730, epic #2721).
//
// cmd/modelbench is compute-only (deterministic LCG token ids, no task
// semantics), so the task layer is net-new. A conceptbench task is a
// self-contained fixture: a prompt, a reproducible starting repo/session state,
// and the EXACT referee the grader (#2732) must read — never the model's own
// "done" text. This file defines:
//
//   - fak.conceptbench.task.v1     — the versioned task record + validating loader.
//   - fak.conceptbench.fixture.v1  — a declarative, hermetic scratch-state build
//     recipe (files + optional pinned-identity git commits + injected
//     verdict/refusal/lease context) that materializes byte-identically across
//     builds, with no network, GPU, or key.
//
// The whole anti-masquerade point: a task's ExpectedWitness must name one of the
// grader's known referees (the Witness* constants in grade.go), and that referee
// must be the one that actually grades the task's concept. The loader rejects a
// task that names an unknown referee — or a real referee that does not grade its
// concept — with a typed *TaskError, so a corpus can never ship a task the
// grader cannot deterministically check.
package conceptbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Schema identifiers — the versioned envelope every corpus artifact carries.
const (
	TaskSchemaV1    = "fak.conceptbench.task.v1"
	FixtureSchemaV1 = "fak.conceptbench.fixture.v1"
)

// Difficulty is a task's coarse hardness band.
type Difficulty string

const (
	DifficultyEasy   Difficulty = "easy"
	DifficultyMedium Difficulty = "medium"
	DifficultyHard   Difficulty = "hard"
)

// Task is one fak.conceptbench.task.v1 record: a fixture-backed episode the
// grader (#2732) scores by reading ExpectedWitness's referee against the fixture.
type Task struct {
	Schema          string     `json:"schema"`           // must equal TaskSchemaV1
	ID              string     `json:"id"`               // unique within a corpus, non-empty
	Concept         Concept    `json:"concept"`          // one of Concepts()
	Prompt          string     `json:"prompt"`           // the model-facing instruction
	FixtureRef      string     `json:"fixture_ref"`      // slash path to a *.fixture.json, relative to the corpus dir
	ExpectedWitness string     `json:"expected_witness"` // a known referee that grades Concept
	Difficulty      Difficulty `json:"difficulty"`       // easy|medium|hard
	Notes           string     `json:"notes,omitempty"`  // free-text rationale, optional
}

// KnownWitnesses is the closed set of referees the grader can dispatch to — the
// Witness* constants in grade.go. A task whose ExpectedWitness is outside this
// set names no referee the grader recognizes and is rejected at load.
func KnownWitnesses() map[string]bool {
	return map[string]bool{
		WitnessDosVerify:       true,
		WitnessDosCommitAudit:  true,
		WitnessDosArbitrate:    true,
		WitnessDosCheckReason:  true,
		WitnessToolDescriptors: true,
		WitnessHandoffSchema:   true,
	}
}

// conceptWitness maps each graded concept to the referee(s) grade.go actually
// consults for it (mirrors the gradeX functions). ExpectedWitness must be one of
// these for the task's concept, so a task cannot claim a real-but-wrong referee
// (e.g. dos_arbitrate for a commit_stamp task).
var conceptWitness = map[Concept]map[string]bool{
	ConceptCommitStamp:        {WitnessDosVerify: true, WitnessDosCommitAudit: true},
	ConceptLane:               {WitnessDosArbitrate: true},
	ConceptRefusal:            {WitnessDosCheckReason: true},
	ConceptVerdictRepair:      {WitnessToolDescriptors: true},
	ConceptHookProtocol:       {WitnessHandoffSchema: true},
	Concept("task_retention"): {WitnessHandoffSchema: true},
	ConceptHonesty:            {WitnessDosCommitAudit: true},
}

// TaskErrorKind is the closed set of validation failure modes. It lets callers
// switch on the cause (errors.As(err, &te); te.Kind) without string-matching.
type TaskErrorKind string

const (
	ErrKindSchema          TaskErrorKind = "bad_schema"
	ErrKindMissingField    TaskErrorKind = "missing_field"
	ErrKindUnknownConcept  TaskErrorKind = "unknown_concept"
	ErrKindUnknownWitness  TaskErrorKind = "unknown_witness"
	ErrKindWitnessMismatch TaskErrorKind = "witness_concept_mismatch"
	ErrKindBadDifficulty   TaskErrorKind = "bad_difficulty"
	ErrKindBadFixture      TaskErrorKind = "bad_fixture"
	ErrKindDuplicateID     TaskErrorKind = "duplicate_id"
)

// TaskError is the typed error every loader/validator returns on a malformed
// task or fixture. Kind is machine-checkable; Msg is the human reading.
type TaskError struct {
	Kind   TaskErrorKind
	TaskID string
	Field  string
	Msg    string
}

func (e *TaskError) Error() string {
	id := e.TaskID
	if id == "" {
		id = "<no-id>"
	}
	if e.Field != "" {
		return fmt.Sprintf("conceptbench task %s: %s (%s): %s", id, e.Kind, e.Field, e.Msg)
	}
	return fmt.Sprintf("conceptbench task %s: %s: %s", id, e.Kind, e.Msg)
}

func validConcept(c Concept) bool {
	for _, k := range Concepts() {
		if k == c {
			return true
		}
	}
	return false
}

// Validate checks a Task against fak.conceptbench.task.v1. The critical rule
// (per the issue's DoD) is the ExpectedWitness gate: it must name a known
// referee, and that referee must be one that grades this task's concept.
func (t Task) Validate() error {
	if t.Schema != TaskSchemaV1 {
		return &TaskError{Kind: ErrKindSchema, TaskID: t.ID, Field: "schema", Msg: fmt.Sprintf("want %q, got %q", TaskSchemaV1, t.Schema)}
	}
	if strings.TrimSpace(t.ID) == "" {
		return &TaskError{Kind: ErrKindMissingField, Field: "id", Msg: "id is required"}
	}
	if !validConcept(t.Concept) {
		return &TaskError{Kind: ErrKindUnknownConcept, TaskID: t.ID, Field: "concept", Msg: fmt.Sprintf("%q is not one of the six graded concepts", t.Concept)}
	}
	if strings.TrimSpace(t.Prompt) == "" {
		return &TaskError{Kind: ErrKindMissingField, TaskID: t.ID, Field: "prompt", Msg: "prompt is required"}
	}
	if strings.TrimSpace(t.FixtureRef) == "" {
		return &TaskError{Kind: ErrKindMissingField, TaskID: t.ID, Field: "fixture_ref", Msg: "fixture_ref is required"}
	}
	if !KnownWitnesses()[t.ExpectedWitness] {
		return &TaskError{Kind: ErrKindUnknownWitness, TaskID: t.ID, Field: "expected_witness", Msg: fmt.Sprintf("%q names no known referee", t.ExpectedWitness)}
	}
	if !conceptWitness[t.Concept][t.ExpectedWitness] {
		return &TaskError{Kind: ErrKindWitnessMismatch, TaskID: t.ID, Field: "expected_witness", Msg: fmt.Sprintf("referee %q does not grade concept %q", t.ExpectedWitness, t.Concept)}
	}
	switch t.Difficulty {
	case DifficultyEasy, DifficultyMedium, DifficultyHard:
	default:
		return &TaskError{Kind: ErrKindBadDifficulty, TaskID: t.ID, Field: "difficulty", Msg: fmt.Sprintf("%q is not easy|medium|hard", t.Difficulty)}
	}
	return nil
}

// LoadTask reads and validates a single fak.conceptbench.task.v1 JSON file.
func LoadTask(pathToTask string) (Task, error) {
	raw, err := os.ReadFile(pathToTask)
	if err != nil {
		return Task{}, err
	}
	var t Task
	if err := json.Unmarshal(raw, &t); err != nil {
		return Task{}, &TaskError{Kind: ErrKindSchema, Field: "<json>", Msg: fmt.Sprintf("%s: %v", filepath.Base(pathToTask), err)}
	}
	if err := t.Validate(); err != nil {
		return Task{}, err
	}
	return t, nil
}

// LoadCorpus loads every *.task.json under dir, validates each, enforces unique
// ids, and confirms every referenced fixture resolves and parses. It returns the
// tasks sorted by id so a caller (or a golden test) sees a stable order.
func LoadCorpus(dir string) ([]Task, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".task.json") {
			continue
		}
		files = append(files, e.Name())
	}
	sort.Strings(files)

	seen := map[string]bool{}
	tasks := make([]Task, 0, len(files))
	for _, name := range files {
		t, err := LoadTask(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		if seen[t.ID] {
			return nil, &TaskError{Kind: ErrKindDuplicateID, TaskID: t.ID, Field: "id", Msg: "id appears more than once in the corpus"}
		}
		seen[t.ID] = true
		// The fixture must resolve and parse — a task pointing at a missing or
		// malformed fixture is not deterministically checkable.
		if _, err := LoadFixture(filepath.Join(dir, filepath.FromSlash(t.FixtureRef))); err != nil {
			return nil, &TaskError{Kind: ErrKindBadFixture, TaskID: t.ID, Field: "fixture_ref", Msg: fmt.Sprintf("%s: %v", t.FixtureRef, err)}
		}
		tasks = append(tasks, t)
	}
	sort.Slice(tasks, func(i, j int) bool { return tasks[i].ID < tasks[j].ID })
	return tasks, nil
}

// ---- fak.conceptbench.fixture.v1 — the hermetic scratch-state build recipe ----

// FixtureRecipe is a declarative, deterministic build script for a task's
// starting state. It materializes with no network, GPU, or key, and the same
// recipe always produces byte-identical content (see Materialize / Manifest).
type FixtureRecipe struct {
	Schema string           `json:"schema"` // must equal FixtureSchemaV1
	ID     string           `json:"id"`
	Files  []FixtureFile    `json:"files,omitempty"`  // seeded working-tree files
	Git    *FixtureGit      `json:"git,omitempty"`    // optional pinned-identity git history
	Inject *InjectedContext `json:"inject,omitempty"` // injected verdict/refusal/lease context
}

// FixtureFile is one seeded file: a repo-relative slash path and its exact bytes.
type FixtureFile struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// FixtureGit seeds a temp git repo with fully pinned identity and dates, so the
// resulting HEAD commit SHA is deterministic across builds.
type FixtureGit struct {
	DefaultBranch string          `json:"default_branch,omitempty"` // defaults to "main"
	Commits       []FixtureCommit `json:"commits"`
}

// FixtureCommit is one pinned commit. AuthorName/AuthorEmail/Date are required so
// the commit object — and thus HEAD SHA — is reproducible.
type FixtureCommit struct {
	Subject     string        `json:"subject"`
	Files       []FixtureFile `json:"files,omitempty"`
	AuthorName  string        `json:"author_name"`
	AuthorEmail string        `json:"author_email"`
	Date        string        `json:"date"` // pinned, e.g. "2026-01-01T00:00:00 +0000"
}

// InjectedContext is the non-file scratch state a grader reads: a returned
// verdict to repair, a structured refusal token, and/or the live leases a lane
// admission must be disjoint from. It is written to deterministic files under
// inject/ so a fixture rebuild reproduces it byte-for-byte.
type InjectedContext struct {
	Verdict      string  `json:"verdict,omitempty"`
	RefusalToken string  `json:"refusal_token,omitempty"`
	Leases       []Lease `json:"leases,omitempty"`
}

// Manifest is the content witness of a materialized fixture: a sha256 per
// working-tree file (excluding .git, whose object mtimes are not content) plus
// the deterministic HEAD SHA when git history is seeded. Two builds of the same
// recipe produce an equal Manifest — that is the "byte-identical rebuild" proof.
type Manifest struct {
	Files   map[string]string `json:"files"` // slash relpath -> sha256 hex of content
	HeadSHA string            `json:"head_sha,omitempty"`
}

// Digest folds a Manifest into one stable hash, so a test can assert equality
// with a single comparison and a corpus can pin an expected value.
func (m Manifest) Digest() string {
	keys := make([]string, 0, len(m.Files))
	for k := range m.Files {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		fmt.Fprintf(h, "%s\x00%s\x00", k, m.Files[k])
	}
	fmt.Fprintf(h, "HEAD\x00%s", m.HeadSHA)
	return hex.EncodeToString(h.Sum(nil))
}

// Validate checks a FixtureRecipe before it is built.
func (r FixtureRecipe) Validate() error {
	if r.Schema != FixtureSchemaV1 {
		return &TaskError{Kind: ErrKindSchema, TaskID: r.ID, Field: "schema", Msg: fmt.Sprintf("want %q, got %q", FixtureSchemaV1, r.Schema)}
	}
	if strings.TrimSpace(r.ID) == "" {
		return &TaskError{Kind: ErrKindMissingField, Field: "id", Msg: "fixture id is required"}
	}
	check := func(f FixtureFile) error {
		if !safeRelPath(f.Path) {
			return &TaskError{Kind: ErrKindBadFixture, TaskID: r.ID, Field: "path", Msg: fmt.Sprintf("%q is not a clean relative path", f.Path)}
		}
		return nil
	}
	for _, f := range r.Files {
		if err := check(f); err != nil {
			return err
		}
	}
	if r.Git != nil {
		if len(r.Git.Commits) == 0 {
			return &TaskError{Kind: ErrKindBadFixture, TaskID: r.ID, Field: "git.commits", Msg: "git block present but has no commits"}
		}
		for i, c := range r.Git.Commits {
			if strings.TrimSpace(c.Subject) == "" || strings.TrimSpace(c.AuthorName) == "" || strings.TrimSpace(c.AuthorEmail) == "" || strings.TrimSpace(c.Date) == "" {
				return &TaskError{Kind: ErrKindBadFixture, TaskID: r.ID, Field: fmt.Sprintf("git.commits[%d]", i), Msg: "commit needs subject, author_name, author_email, and a pinned date"}
			}
			for _, f := range c.Files {
				if err := check(f); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// safeRelPath reports whether p is a clean, relative, non-escaping slash path.
func safeRelPath(p string) bool {
	if p == "" || strings.HasPrefix(p, "/") || strings.Contains(p, "\\") {
		return false
	}
	clean := path.Clean(p)
	if clean != p || clean == "." || strings.HasPrefix(clean, "../") {
		return false
	}
	return true
}

// LoadFixture reads and validates a fak.conceptbench.fixture.v1 recipe.
func LoadFixture(pathToFixture string) (FixtureRecipe, error) {
	raw, err := os.ReadFile(pathToFixture)
	if err != nil {
		return FixtureRecipe{}, err
	}
	var r FixtureRecipe
	if err := json.Unmarshal(raw, &r); err != nil {
		return FixtureRecipe{}, &TaskError{Kind: ErrKindSchema, Field: "<json>", Msg: fmt.Sprintf("%s: %v", filepath.Base(pathToFixture), err)}
	}
	if err := r.Validate(); err != nil {
		return FixtureRecipe{}, err
	}
	return r, nil
}

// NeedsGit reports whether materializing this recipe shells out to git.
func (r FixtureRecipe) NeedsGit() bool {
	return r.Git != nil && len(r.Git.Commits) > 0
}

// Materialize builds the fixture into dir (which must exist and be empty) and
// returns its content Manifest. It is hermetic: only the local filesystem and,
// when the recipe seeds git history, the local git binary are touched. Every git
// invocation pins identity, dates, line-endings, and signing off, so the HEAD
// SHA is deterministic.
func (r FixtureRecipe) Materialize(dir string) (Manifest, error) {
	if err := r.Validate(); err != nil {
		return Manifest{}, err
	}
	if err := writeFiles(dir, r.Files); err != nil {
		return Manifest{}, err
	}
	if r.Inject != nil {
		if err := writeInjected(dir, *r.Inject); err != nil {
			return Manifest{}, err
		}
	}
	head := ""
	if r.NeedsGit() {
		var err error
		if head, err = buildGit(dir, *r.Git); err != nil {
			return Manifest{}, err
		}
	}
	files, err := hashTree(dir)
	if err != nil {
		return Manifest{}, err
	}
	return Manifest{Files: files, HeadSHA: head}, nil
}

func writeFiles(dir string, files []FixtureFile) error {
	for _, f := range files {
		if err := writeOne(dir, f); err != nil {
			return err
		}
	}
	return nil
}

func writeOne(dir string, f FixtureFile) error {
	dst := filepath.Join(dir, filepath.FromSlash(f.Path))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	return os.WriteFile(dst, []byte(f.Content), 0o644)
}

// writeInjected serializes the non-file scratch state deterministically under
// inject/ so a rebuild reproduces it byte-for-byte.
func writeInjected(dir string, in InjectedContext) error {
	if in.Verdict != "" {
		if err := writeOne(dir, FixtureFile{Path: "inject/verdict.txt", Content: in.Verdict}); err != nil {
			return err
		}
	}
	if in.RefusalToken != "" {
		if err := writeOne(dir, FixtureFile{Path: "inject/refusal.txt", Content: in.RefusalToken}); err != nil {
			return err
		}
	}
	if len(in.Leases) > 0 {
		b, err := json.MarshalIndent(in.Leases, "", "  ")
		if err != nil {
			return err
		}
		if err := writeOne(dir, FixtureFile{Path: "inject/leases.json", Content: string(b) + "\n"}); err != nil {
			return err
		}
	}
	return nil
}

// buildGit seeds a pinned-identity git repo and returns the HEAD SHA.
func buildGit(dir string, g FixtureGit) (string, error) {
	branch := g.DefaultBranch
	if branch == "" {
		branch = "main"
	}
	// Config flags pinned on every call: no signing, no CRLF translation, a
	// fixed default branch, and a fixed identity fallback.
	base := []string{
		"-c", "init.defaultBranch=" + branch,
		"-c", "core.autocrlf=false",
		"-c", "commit.gpgsign=false",
		"-c", "user.name=conceptbench",
		"-c", "user.email=conceptbench@fak.invalid",
	}
	if err := runGit(dir, nil, append(append([]string(nil), base...), "init", "-q")...); err != nil {
		return "", err
	}
	for _, c := range g.Commits {
		if err := writeFiles(dir, c.Files); err != nil {
			return "", err
		}
		if err := runGit(dir, nil, append(append([]string(nil), base...), "add", "-A")...); err != nil {
			return "", err
		}
		env := []string{
			"GIT_AUTHOR_NAME=" + c.AuthorName,
			"GIT_AUTHOR_EMAIL=" + c.AuthorEmail,
			"GIT_AUTHOR_DATE=" + c.Date,
			"GIT_COMMITTER_NAME=" + c.AuthorName,
			"GIT_COMMITTER_EMAIL=" + c.AuthorEmail,
			"GIT_COMMITTER_DATE=" + c.Date,
		}
		args := append(append([]string(nil), base...), "commit", "-q", "-m", c.Subject)
		if err := runGit(dir, env, args...); err != nil {
			return "", err
		}
	}
	out, err := runGitOut(dir, nil, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

func runGit(dir string, extraEnv []string, args ...string) error {
	_, err := runGitOut(dir, extraEnv, args...)
	return err
}

func runGitOut(dir string, extraEnv []string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = dir
	// A hermetic, C-locale environment: inherit PATH but drop the caller's git
	// identity/config knobs so the fixture's pins are the only ones that apply.
	cmd.Env = append([]string{
		"PATH=" + os.Getenv("PATH"),
		"HOME=" + dir,
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL=/dev/null",
		"LC_ALL=C",
		"TZ=UTC",
	}, extraEnv...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

// hashTree walks dir and returns slash-relpath -> sha256 hex of each file's
// content, skipping the .git directory (its object storage carries mtimes and
// packing state that are not part of the fixture's content witness — the
// deterministic HEAD SHA already witnesses the git history).
func hashTree(dir string) (map[string]string, error) {
	out := map[string]string{}
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(b)
		out[filepath.ToSlash(rel)] = hex.EncodeToString(sum[:])
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}
