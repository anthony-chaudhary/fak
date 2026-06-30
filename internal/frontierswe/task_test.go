package frontierswe

import (
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// fixtureDir is the committed offline task tree (no network/Docker/model).
const fixtureDir = "testdata/frontierswe"

func TestLoadTask_GitToZig_ImplementationEnvelope(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "git-to-zig"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}

	if task.Name != "git-to-zig" {
		t.Errorf("Name = %q, want git-to-zig", task.Name)
	}
	if task.ScoringCategory != CategoryImplementation {
		t.Errorf("ScoringCategory = %q, want %q", task.ScoringCategory, CategoryImplementation)
	}
	if task.Version != "1.0" {
		t.Errorf("Version = %q, want 1.0", task.Version)
	}

	// The 20-hour agent budget (the wall-clock that dominates cost).
	if got := task.AgentTimeoutSec(); got != 72000.0 {
		t.Errorf("AgentTimeoutSec = %v, want 72000", got)
	}
	if got := task.VerifierTimeoutSec(); got != 86400.0 {
		t.Errorf("VerifierTimeoutSec = %v, want 86400", got)
	}

	// [metadata] (free-form, distinct from the scoring Category).
	if task.Metadata.Difficulty != "very_hard" {
		t.Errorf("Metadata.Difficulty = %q, want very_hard", task.Metadata.Difficulty)
	}
	if task.Metadata.Category != "c-to-zig" {
		t.Errorf("Metadata.Category = %q, want c-to-zig", task.Metadata.Category)
	}
	wantTags := []string{"zig", "c", "porting", "git", "systems-programming", "vcs"}
	if !reflect.DeepEqual(task.Metadata.Tags, wantTags) {
		t.Errorf("Metadata.Tags = %v, want %v", task.Metadata.Tags, wantTags)
	}

	// [environment] resource envelope read correctly.
	env := task.Environment
	if env.DockerImage != "ghcr.io/proximal-labs/frontier-swe/git-to-zig:v4" {
		t.Errorf("DockerImage = %q", env.DockerImage)
	}
	if env.BuildTimeoutSec != 1800.0 {
		t.Errorf("BuildTimeoutSec = %v, want 1800", env.BuildTimeoutSec)
	}
	if env.CPUs != 4 {
		t.Errorf("CPUs = %d, want 4", env.CPUs)
	}
	if env.MemoryMB != 16384 {
		t.Errorf("MemoryMB = %d, want 16384", env.MemoryMB)
	}
	if env.StorageMB != 30720 {
		t.Errorf("StorageMB = %d, want 30720", env.StorageMB)
	}
	if env.GPUs != 0 {
		t.Errorf("GPUs = %d, want 0", env.GPUs)
	}
	if env.AllowInternet {
		t.Errorf("AllowInternet = true, want false")
	}
	if len(env.MCPServers) != 0 {
		t.Errorf("MCPServers = %v, want empty", env.MCPServers)
	}

	// job.yaml fold.
	if !reflect.DeepEqual(task.Job.Agents, []string{"claude-code", "codex"}) {
		t.Errorf("Job.Agents = %v", task.Job.Agents)
	}
	if task.Job.NAttempts != 3 {
		t.Errorf("Job.NAttempts = %d, want 3", task.Job.NAttempts)
	}
	if task.Job.NConcurrentTrial != 2 {
		t.Errorf("Job.NConcurrentTrial = %d, want 2", task.Job.NConcurrentTrial)
	}
	if !reflect.DeepEqual(task.Job.Artifacts, []string{"solution.patch", "agent.log", "reward.json"}) {
		t.Errorf("Job.Artifacts = %v", task.Job.Artifacts)
	}

	// oracle.yaml fold.
	if task.Oracle.Command != "python -m verifier.run --task git-to-zig" {
		t.Errorf("Oracle.Command = %q", task.Oracle.Command)
	}
	if task.Oracle.RewardKey != "fraction_tests_passed" {
		t.Errorf("Oracle.RewardKey = %q, want fraction_tests_passed", task.Oracle.RewardKey)
	}
}

func TestLoadTask_Performance_GPUEnvelope(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "granite-mamba2-inference-optimization"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	if task.ScoringCategory != CategoryPerformance {
		t.Errorf("ScoringCategory = %q, want %q", task.ScoringCategory, CategoryPerformance)
	}
	if got := task.AgentTimeoutSec(); got != 72000.0 {
		t.Errorf("AgentTimeoutSec = %v, want 72000", got)
	}
	// A performance task gets a GPU and a large memory envelope.
	if task.Environment.GPUs != 1 {
		t.Errorf("GPUs = %d, want 1", task.Environment.GPUs)
	}
	if task.Environment.CPUs != 8 {
		t.Errorf("CPUs = %d, want 8", task.Environment.CPUs)
	}
	if task.Environment.MemoryMB != 65536 {
		t.Errorf("MemoryMB = %d, want 65536", task.Environment.MemoryMB)
	}
	if task.Environment.AllowInternet {
		t.Errorf("AllowInternet = true, want false")
	}
}

func TestLoadTask_MLResearch_Envelope(t *testing.T) {
	task, err := LoadTask(filepath.Join(fixtureDir, "frogsgame-rl"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	if task.ScoringCategory != CategoryMLResearch {
		t.Errorf("ScoringCategory = %q, want %q", task.ScoringCategory, CategoryMLResearch)
	}
	// This ML-research task runs against a hosted inference API, so it permits
	// internet and carries no local GPU.
	if !task.Environment.AllowInternet {
		t.Errorf("AllowInternet = false, want true")
	}
	if task.Environment.GPUs != 0 {
		t.Errorf("GPUs = %d, want 0", task.Environment.GPUs)
	}
	if got := task.AgentTimeoutSec(); got != 28800.0 {
		t.Errorf("AgentTimeoutSec = %v, want 28800", got)
	}
}

func TestLoadTask_MissingTOML(t *testing.T) {
	if _, err := LoadTask(filepath.Join(fixtureDir, "does-not-exist")); err == nil {
		t.Fatal("LoadTask of a missing dir: want error, got nil")
	}
}

func TestLoadTask_OptionalFilesAbsent(t *testing.T) {
	// granite has no job.yaml / oracle.yaml — the fold must stay zero-valued,
	// not error.
	task, err := LoadTask(filepath.Join(fixtureDir, "granite-mamba2-inference-optimization"))
	if err != nil {
		t.Fatalf("LoadTask: %v", err)
	}
	if !reflect.DeepEqual(task.Job, Job{}) {
		t.Errorf("Job = %+v, want zero value", task.Job)
	}
	if !reflect.DeepEqual(task.Oracle, Oracle{}) {
		t.Errorf("Oracle = %+v, want zero value", task.Oracle)
	}
}

// wantImplementation / wantPerformance / wantMLResearch replicate the upstream
// scripts/score_from_reward.py lists verbatim. They are the INDEPENDENT witness:
// the catalog in catalog.go must agree with these exactly, so a typo or drift in
// either place fails the build.
var (
	wantImplementation = []string{
		"git-to-zig",
		"dart-style-haskell",
		"lua-native-compiler",
		"postgres-sqlite-wire-adapter",
		"modular-stack-wan21",
	}
	wantPerformance = []string{
		"libexpat-to-x86asm",
		"ffmpeg-swscale-rewrite",
		"pyright-type-checking-optimization",
		"granite-mamba2-inference-optimization",
		"notebook-compression",
		"revideo-perf-opt",
		"cranelift-codegen-opt",
		"dependent-type-checker",
		"inference-system-optimization",
	}
	wantMLResearch = []string{
		"pcqm4mv2-autoresearch",
		"frogsgame-rl",
		"optimizer-design",
	}
)

func TestCatalog_MatchesScoreFromReward(t *testing.T) {
	// Total must be exactly 17.
	if got := len(Catalog()); got != 17 {
		t.Fatalf("catalog size = %d, want 17", got)
	}

	checkList := func(name string, want []string, cat Category) {
		for _, n := range want {
			got, ok := CategoryOf(n)
			if !ok {
				t.Errorf("%s: task %q missing from catalog", name, n)
				continue
			}
			if got != cat {
				t.Errorf("%s: task %q category = %q, want %q", name, n, got, cat)
			}
		}
		// The catalog must contain NO extra task in this category.
		gotNames := TaskNamesByCategory(cat)
		wantSorted := append([]string(nil), want...)
		gotSorted := append([]string(nil), gotNames...)
		sort.Strings(wantSorted)
		sort.Strings(gotSorted)
		if !reflect.DeepEqual(gotSorted, wantSorted) {
			t.Errorf("%s: TaskNamesByCategory(%q) = %v, want %v", name, cat, gotSorted, wantSorted)
		}
	}

	checkList("IMPLEMENTATION", wantImplementation, CategoryImplementation)
	checkList("PERFORMANCE", wantPerformance, CategoryPerformance)
	checkList("ML_RESEARCH", wantMLResearch, CategoryMLResearch)

	// And the union must equal the full TaskNames() set (no orphan entries).
	union := map[string]bool{}
	for _, n := range wantImplementation {
		union[n] = true
	}
	for _, n := range wantPerformance {
		union[n] = true
	}
	for _, n := range wantMLResearch {
		union[n] = true
	}
	for _, n := range TaskNames() {
		if !union[n] {
			t.Errorf("catalog has unexpected task %q not in score_from_reward.py lists", n)
		}
	}
	if len(union) != 17 {
		t.Errorf("witness lists union = %d names, want 17", len(union))
	}
}

func TestCategory_Valid(t *testing.T) {
	for _, c := range []Category{CategoryImplementation, CategoryPerformance, CategoryMLResearch} {
		if !c.Valid() {
			t.Errorf("%q.Valid() = false, want true", c)
		}
	}
	if Category("bogus").Valid() {
		t.Error("bogus category reported valid")
	}
	if got := Category("").String(); got != "unknown" {
		t.Errorf("empty Category String() = %q, want unknown", got)
	}
}

func TestCategoryOf_Unknown(t *testing.T) {
	if c, ok := CategoryOf("not-a-real-task"); ok || c != "" {
		t.Errorf("CategoryOf(unknown) = (%q, %v), want (\"\", false)", c, ok)
	}
}
