package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// repoRootTasksDir is the committed FrontierSWE task fixtures, addressed relative
// to the cmd/fak test working directory (go test runs with cwd = the package
// dir). The fixtures live at the repo-root testdata/frontierswe so a real `fak
// frontierswe describe` run from the repo root finds them by the default path —
// the same repo-root testdata contract `fak swebench describe` uses.
const repoRootTasksDir = "../../testdata/frontierswe"

// TestFrontiersweDescribeOfflineListsAllTasks is the load-bearing acceptance:
// `fak frontierswe describe` exits 0 with no flags (offline, no model/GPU/Docker),
// emits clean JSON on stdout, lists all 17 catalog tasks with their scoring gate
// class, and carries correct budgets/resources for the tasks that have a committed
// task.toml fixture.
func TestFrontiersweDescribeOfflineExitsZeroAndListsTasks(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// No args: must default to describe and run fully offline.
	code := runFrontierswe(&stdout, &stderr, []string{"describe"})
	if code != 0 {
		t.Fatalf("frontierswe describe exit = %d, want 0 (offline must succeed)\nstderr:\n%s", code, stderr.String())
	}

	var d FrontierDescribe
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("stdout is not the describe JSON: %v\nstdout:\n%s", err, stdout.String())
	}

	// All 17 catalog tasks are listed (the catalog is the source of truth, so the
	// count is independent of how many per-task fixtures are committed).
	wantNames := frontierswe.TaskNames()
	if len(wantNames) != 17 {
		t.Fatalf("catalog has %d tasks, expected 17 — fixture/test drift", len(wantNames))
	}
	if d.TaskCount != 17 || len(d.Tasks) != 17 {
		t.Fatalf("describe listed %d tasks (TaskCount=%d), want 17", len(d.Tasks), d.TaskCount)
	}
	got := map[string]FrontierTaskRow{}
	for _, row := range d.Tasks {
		got[row.Name] = row
	}
	for _, name := range wantNames {
		row, ok := got[name]
		if !ok {
			t.Errorf("task %q missing from describe output", name)
			continue
		}
		// Every task has a gate class resolved from the catalog.
		cat, _ := frontierswe.CategoryOf(name)
		if row.GateClass != cat.String() {
			t.Errorf("%s: gate class = %q, want %q", name, row.GateClass, cat.String())
		}
	}

	// The headline cost is the canonical 20-hour per-task agent budget.
	if d.HeadlineHours != 20 {
		t.Errorf("headline budget = %.2fh, want 20h", d.HeadlineHours)
	}

	// The gate-class histogram matches the catalog families: 5 implementation,
	// 9 performance, 3 ml_research.
	wantGates := map[string]int{"implementation": 5, "performance": 9, "ml_research": 3}
	for class, want := range wantGates {
		if d.GateClasses[class] != want {
			t.Errorf("gate class %q = %d, want %d", class, d.GateClasses[class], want)
		}
	}

	// The offline-fallback notice is announced on stderr (the swebench contract).
	if !strings.Contains(stderr.String(), "committed fixtures") {
		t.Errorf("stderr missing the offline-fallback announcement; got:\n%s", stderr.String())
	}
}

// TestFrontiersweDescribeBudgetsAndResources checks the per-task budget/resource
// overlay against the committed task.toml fixtures: the 20h budget, the gate
// class, and the [environment] envelope (cpus/memory_mb/gpus/allow_internet) must
// match the upstream values for the tasks that have a fixture.
func TestFrontiersweDescribeBudgetsAndResources(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// Point --tasks at the committed repo-root fixtures (the default path is
	// resolved against cwd, which is the repo root for a real run but the cmd/fak
	// package dir under `go test`).
	if code := runFrontierswe(&stdout, &stderr, []string{"describe", "--json", "--tasks", repoRootTasksDir}); code != 0 {
		t.Fatalf("frontierswe describe --json exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	var d FrontierDescribe
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("stdout not JSON: %v", err)
	}
	if d.FixtureCount < 3 {
		t.Fatalf("fixture overlay found %d task.toml fixtures, want >=3 (git-to-zig, granite, frogsgame-rl)", d.FixtureCount)
	}
	rows := map[string]FrontierTaskRow{}
	for _, r := range d.Tasks {
		rows[r.Name] = r
	}

	// git-to-zig: implementation, very_hard, 20h budget, 4 cpu / 16 GB / 0 gpu /
	// no internet.
	z := rows["git-to-zig"]
	if !z.HasFixture {
		t.Fatal("git-to-zig has a committed fixture; HasFixture should be true")
	}
	if z.GateClass != "implementation" {
		t.Errorf("git-to-zig gate = %q, want implementation", z.GateClass)
	}
	if z.Difficulty != "very_hard" {
		t.Errorf("git-to-zig difficulty = %q, want very_hard", z.Difficulty)
	}
	if z.AgentTimeoutS != 72000 {
		t.Errorf("git-to-zig agent budget = %ds, want 72000 (20h)", z.AgentTimeoutS)
	}
	if z.AgentTimeoutHr != 20 {
		t.Errorf("git-to-zig agent budget = %.2fh, want 20h", z.AgentTimeoutHr)
	}
	if z.CPUs != 4 || z.MemoryMB != 16384 || z.GPUs != 0 || z.AllowInternet {
		t.Errorf("git-to-zig env = {cpus:%d mem:%d gpus:%d net:%v}, want {4 16384 0 false}",
			z.CPUs, z.MemoryMB, z.GPUs, z.AllowInternet)
	}

	// granite-mamba2-inference-optimization: performance, GPU envelope (1 gpu /
	// 8 cpu / 64 GB), no internet.
	g := rows["granite-mamba2-inference-optimization"]
	if g.GateClass != "performance" {
		t.Errorf("granite gate = %q, want performance", g.GateClass)
	}
	if g.GPUs != 1 || g.CPUs != 8 || g.MemoryMB != 65536 || g.AllowInternet {
		t.Errorf("granite env = {cpus:%d mem:%d gpus:%d net:%v}, want {8 65536 1 false}",
			g.CPUs, g.MemoryMB, g.GPUs, g.AllowInternet)
	}

	// frogsgame-rl: ml_research, hosted-API task (allow_internet true, no local
	// GPU), with the shorter 8h agent budget.
	f := rows["frogsgame-rl"]
	if f.GateClass != "ml_research" {
		t.Errorf("frogsgame-rl gate = %q, want ml_research", f.GateClass)
	}
	if !f.AllowInternet || f.GPUs != 0 {
		t.Errorf("frogsgame-rl env = {gpus:%d net:%v}, want {0 true}", f.GPUs, f.AllowInternet)
	}
	if f.AgentTimeoutS != 28800 {
		t.Errorf("frogsgame-rl agent budget = %ds, want 28800 (8h)", f.AgentTimeoutS)
	}

	// --json suppresses the human table on stderr but keeps the offline notice.
	if strings.Contains(stderr.String(), "== fak frontierswe describe ==") {
		t.Errorf("--json should suppress the human table; stderr:\n%s", stderr.String())
	}
}

// TestFrontiersweUnknownSubcommand confirms an unknown subcommand fails with the
// usage code (2), not a silent success.
func TestFrontiersweUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runFrontierswe(&stdout, &stderr, []string{"nope"}); code != 2 {
		t.Fatalf("unknown subcommand exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Errorf("stderr missing the unknown-subcommand notice; got:\n%s", stderr.String())
	}
}

// TestFrontiersweBenchcatalogRowResolves confirms the registry row is wired so the
// surface appears in `fak benchmarks list` and `fak benchmarks describe
// frontierswe` resolves it: a KindVerb / NeedNone / LevelE2E offline row.
func TestFrontiersweBenchcatalogRowResolves(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchmarks(&stdout, &stderr, []string{"describe", "frontierswe"}); code != 0 {
		t.Fatalf("benchmarks describe frontierswe exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "frontierswe") {
		t.Errorf("benchmarks describe output missing the name; got:\n%s", out)
	}
	if !strings.Contains(out, "fak frontierswe describe") {
		t.Errorf("benchmarks describe missing the run command; got:\n%s", out)
	}
}
