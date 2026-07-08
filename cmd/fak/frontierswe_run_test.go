package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// TestFrontiersweRunWritesArtifacts is the C9 (#1715) end-to-end acceptance: a
// `fak frontierswe run --preds-only` drives >=1 turn against a mocked environment
// and writes the submission artifact + the fak.frontierswe.run.v1 meta + the
// per-turn TTS trace, then honestly gates the real environment.
func TestFrontiersweRunWritesArtifacts(t *testing.T) {
	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"run", "--json", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--preds-only", "--turns", "4", "--output", out,
	})
	if code != 0 {
		t.Fatalf("run exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}

	var r frontierswe.RunResult
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("stdout is not run JSON: %v\n%s", err, stdout.String())
	}
	if r.Meta.Schema != frontierswe.RunSchema || r.Meta.Task != "git-to-zig" {
		t.Fatalf("unexpected run identity: %+v", r.Meta)
	}
	if r.Meta.Turns < 1 {
		t.Fatalf("run drove %d turns, want >=1", r.Meta.Turns)
	}
	if r.Meta.BudgetSec != 72000 {
		t.Fatalf("budget = %d, want the 20h 72000s", r.Meta.BudgetSec)
	}
	if !r.Meta.PredsOnly {
		t.Fatalf("--preds-only not recorded: %+v", r.Meta)
	}

	// The three required emissions exist on disk.
	for _, name := range []string{"meta.json", "tts-trace.json", "job.yaml"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Errorf("missing output %s: %v", name, err)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "submission", "SUBMISSION.md")); err != nil {
		t.Errorf("missing submission artifact: %v", err)
	}

	// meta.json round-trips to the same schema.
	mb, err := os.ReadFile(filepath.Join(out, "meta.json"))
	if err != nil {
		t.Fatalf("read meta.json: %v", err)
	}
	var meta frontierswe.RunMeta
	if err := json.Unmarshal(mb, &meta); err != nil || meta.Schema != frontierswe.RunSchema {
		t.Fatalf("meta.json bad: err=%v schema=%q", err, meta.Schema)
	}

	// tts-trace.json carries the per-turn series with the C8 reuse fold.
	tb, err := os.ReadFile(filepath.Join(out, "tts-trace.json"))
	if err != nil {
		t.Fatalf("read tts-trace.json: %v", err)
	}
	var trace frontierswe.TTSTrace
	if err := json.Unmarshal(tb, &trace); err != nil {
		t.Fatalf("tts-trace.json bad: %v", err)
	}
	if len(trace.Points) != r.Meta.Turns {
		t.Fatalf("trace has %d points, want %d (turn count)", len(trace.Points), r.Meta.Turns)
	}
	if !trace.CacheSeries.CacheBit {
		t.Fatalf("TTS trace C8 reuse series did not bite: %+v", trace.CacheSeries)
	}

	// The job.yaml artifact list (from the git-to-zig fixture) was collected.
	wantArtifacts := map[string]bool{"solution.patch": false, "agent.log": false, "reward.json": false}
	for _, a := range r.Artifacts {
		if _, ok := wantArtifacts[a.Name]; ok {
			wantArtifacts[a.Name] = a.Collected
		}
	}
	for name, collected := range wantArtifacts {
		if !collected {
			t.Errorf("job.yaml artifact %q not collected", name)
		}
	}
}

// TestFrontiersweRunFoldsJobYAMLArtifacts witnesses genuine job.yaml artifact-list
// folding through the cmd path (not the fallback): a task whose job.yaml declares
// the canonical cranelift artifact trio has exactly those collected.
func TestFrontiersweRunFoldsJobYAMLArtifacts(t *testing.T) {
	tasksDir := t.TempDir()
	taskDir := filepath.Join(tasksDir, "cranelift-codegen-opt")
	if err := os.MkdirAll(taskDir, 0o755); err != nil {
		t.Fatal(err)
	}
	toml := "version = \"1.0\"\n[agent]\ntimeout_sec = 72000.0\n[environment]\nallow_internet = false\n"
	job := "agents:\n  - claude-code\nn_concurrent_trials: 1\nartifacts:\n  - /app/wasmtime\n  - /logs/agent\n  - /logs/verifier\n"
	if err := os.WriteFile(filepath.Join(taskDir, "task.toml"), []byte(toml), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(taskDir, "job.yaml"), []byte(job), 0o644); err != nil {
		t.Fatal(err)
	}

	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"run", "--json", "--tasks", tasksDir, "--task", "cranelift-codegen-opt",
		"--preds-only", "--turns", "2", "--output", out,
	})
	if code != 0 {
		t.Fatalf("run exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	var r frontierswe.RunResult
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("stdout is not run JSON: %v", err)
	}
	got := map[string]bool{}
	for _, a := range r.Artifacts {
		got[a.Name] = a.Collected
	}
	for _, want := range []string{"/app/wasmtime", "/logs/agent", "/logs/verifier"} {
		if !got[want] {
			t.Errorf("job.yaml artifact %q not folded+collected; got %+v", want, r.Artifacts)
		}
	}
	// The submission target is the modified source tree the task grades.
	if r.Submission.Target != "/app/cranelift-codegen-opt" {
		t.Errorf("submission target = %q, want /app/cranelift-codegen-opt", r.Submission.Target)
	}
}

// TestFrontiersweRunHonestGate proves the run gates honestly where Docker is
// unavailable: the summary prints the exact remote command instead of faking a
// live run.
func TestFrontiersweRunHonestGate(t *testing.T) {
	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"run", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--preds-only", "--turns", "2", "--output", out,
	})
	if code != 0 {
		t.Fatalf("run exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	s := stderr.String()
	if !strings.Contains(s, "mocked        : true") {
		t.Errorf("summary should mark the run mocked:\n%s", s)
	}
	// Where Docker is absent the gate must print the remote command; where present
	// it need not. Assert only that a non-runnable gate surfaces the command.
	if strings.Contains(s, "runnable=false") && !strings.Contains(s, "docker run") {
		t.Errorf("gated run must print the remote docker command:\n%s", s)
	}
}

// TestFrontiersweRunRefusesExternalNoInternetGateway keeps the no-internet
// boundary a hard refusal, not a mock papered over.
func TestFrontiersweRunRefusesExternalNoInternetGateway(t *testing.T) {
	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"run", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--gateway", "https://gateway.example.com/v1", "--output", out,
	})
	if code != 1 {
		t.Fatalf("external gateway under allow_internet=false: exit = %d, want 1\nstderr:\n%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "REFUSED") {
		t.Errorf("expected an explicit REFUSED, got:\n%s", stderr.String())
	}
}
