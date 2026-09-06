package loopmgr

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestExecutableJobPersistence verifies that:
// 1. Executable jobs retain exact argv, workdir, timeout, and state across persistence cycles.
// 2. Legacy schedule-only registry payloads (no execution field or null) load cleanly and remain valid.
// 3. Malformed execution specifications (relative paths, empty argv, invalid timeouts) are rejected.
func TestExecutableJobPersistence(t *testing.T) {
	t.Run("retention_of_exact_execution_spec_and_state", func(t *testing.T) {
		tempDir := t.TempDir()
		path := filepath.Join(tempDir, "registry.json")
		workDirA := filepath.Join(tempDir, "work-a")
		workDirB := filepath.Join(tempDir, "work-b")
		now := regClock()

		reg := Registry{Jobs: map[string]Job{}}

		jobA := Job{
			Schedule: Schedule{
				JobID:           "task-worker/run",
				IntervalSeconds: 300,
				MissedRun:       MissedSkip,
				JitterSeconds:   15,
			},
			State: JobArmed,
			Execution: &ExecutionSpec{
				Argv:           []string{"fak-worker", "--queue", "high", "--batch=10"},
				WorkDir:        workDirA,
				TimeoutSeconds: 600,
			},
		}

		jobB := Job{
			Schedule: Schedule{
				JobID:           "task-cleanup/daily",
				IntervalSeconds: 86400,
				MissedRun:       MissedCatchUp,
			},
			State: JobDisabled,
			Execution: &ExecutionSpec{
				Argv:           []string{"/usr/bin/env", "sh", "-c", "echo cleaning"},
				WorkDir:        workDirB,
				TimeoutSeconds: 3600,
			},
		}

		mustPut(t, &reg, jobA, now)
		mustPut(t, &reg, jobB, now)

		if err := SaveRegistry(path, reg); err != nil {
			t.Fatalf("SaveRegistry: %v", err)
		}

		reloaded, err := LoadRegistry(path)
		if err != nil {
			t.Fatalf("LoadRegistry: %v", err)
		}

		gotA, ok := reloaded.Get("task-worker/run")
		if !ok {
			t.Fatalf("job task-worker/run not found in reloaded registry")
		}
		if gotA.State != JobArmed {
			t.Fatalf("jobA state = %q, want %q", gotA.State, JobArmed)
		}
		if !gotA.Executable() {
			t.Fatalf("expected jobA to be Executable()")
		}
		if gotA.Execution == nil {
			t.Fatalf("jobA Execution is nil")
		}
		if !reflect.DeepEqual(gotA.Execution.Argv, []string{"fak-worker", "--queue", "high", "--batch=10"}) {
			t.Fatalf("jobA argv mismatch: got %#v", gotA.Execution.Argv)
		}
		if gotA.Execution.WorkDir != workDirA {
			t.Fatalf("jobA WorkDir = %q, want %q", gotA.Execution.WorkDir, workDirA)
		}
		if gotA.Execution.TimeoutSeconds != 600 {
			t.Fatalf("jobA TimeoutSeconds = %d, want 600", gotA.Execution.TimeoutSeconds)
		}

		gotB, ok := reloaded.Get("task-cleanup/daily")
		if !ok {
			t.Fatalf("job task-cleanup/daily not found in reloaded registry")
		}
		if gotB.State != JobDisabled {
			t.Fatalf("jobB state = %q, want %q", gotB.State, JobDisabled)
		}
		if !gotB.Executable() {
			t.Fatalf("expected jobB to be Executable()")
		}
		if gotB.Execution == nil {
			t.Fatalf("jobB Execution is nil")
		}
		if !reflect.DeepEqual(gotB.Execution.Argv, []string{"/usr/bin/env", "sh", "-c", "echo cleaning"}) {
			t.Fatalf("jobB argv mismatch: got %#v", gotB.Execution.Argv)
		}
		if gotB.Execution.WorkDir != workDirB {
			t.Fatalf("jobB WorkDir = %q, want %q", gotB.Execution.WorkDir, workDirB)
		}
		if gotB.Execution.TimeoutSeconds != 3600 {
			t.Fatalf("jobB TimeoutSeconds = %d, want 3600", gotB.Execution.TimeoutSeconds)
		}

		// Ensure ArmedJobs respects armed state even for executable jobs
		armed := reloaded.ArmedJobs()
		if len(armed) != 1 || armed[0].JobID() != "task-worker/run" {
			t.Fatalf("reloaded ArmedJobs = %+v, want only task-worker/run", jobIDs(armed))
		}
	})

	t.Run("successful_loading_of_legacy_schedule_only_payload", func(t *testing.T) {
		// Legacy registry JSON payload with no "execution" field on job1,
		// and explicit "execution": null on job2.
		legacyJSON := `{
  "schema": "fak.loop-registry.v1",
  "jobs": {
    "legacy-scheduler/tick": {
      "schedule": {
        "job_id": "legacy-scheduler/tick",
        "interval_seconds": 60,
        "missed_run": "skip"
      },
      "state": "armed"
    },
    "legacy-scheduler/report": {
      "schedule": {
        "job_id": "legacy-scheduler/report",
        "interval_seconds": 3600,
        "missed_run": "catch-up"
      },
      "state": "disabled",
      "execution": null
    }
  }
}`
		path := filepath.Join(t.TempDir(), "legacy-registry.json")
		if err := os.WriteFile(path, []byte(legacyJSON), 0o644); err != nil {
			t.Fatalf("write legacy json: %v", err)
		}

		reloaded, err := LoadRegistry(path)
		if err != nil {
			t.Fatalf("LoadRegistry for legacy format failed: %v", err)
		}

		j1, ok := reloaded.Get("legacy-scheduler/tick")
		if !ok {
			t.Fatalf("missing legacy-scheduler/tick")
		}
		if j1.Execution != nil {
			t.Fatalf("expected nil Execution for legacy job, got %#v", j1.Execution)
		}
		if j1.Executable() {
			t.Fatalf("legacy job without execution spec must not report Executable()")
		}
		if j1.State != JobArmed {
			t.Fatalf("legacy job state = %q, want armed", j1.State)
		}

		j2, ok := reloaded.Get("legacy-scheduler/report")
		if !ok {
			t.Fatalf("missing legacy-scheduler/report")
		}
		if j2.Execution != nil {
			t.Fatalf("expected nil Execution for null execution job, got %#v", j2.Execution)
		}
		if j2.Executable() {
			t.Fatalf("job with null execution must not report Executable()")
		}
		if j2.State != JobDisabled {
			t.Fatalf("legacy job 2 state = %q, want disabled", j2.State)
		}
	})

	t.Run("rejection_of_malformed_execution_data", func(t *testing.T) {
		tempDir := t.TempDir()
		absDir := filepath.Join(tempDir, "abs-work")
		now := regClock()

		testCases := []struct {
			name     string
			spec     ExecutionSpec
			jsonSpec string
		}{
			{
				name: "relative_work_dir",
				spec: ExecutionSpec{
					Argv:           []string{"run.sh"},
					WorkDir:        "relative/path/to/dir",
					TimeoutSeconds: 60,
				},
				jsonSpec: `{"argv": ["run.sh"], "work_dir": "relative/path/to/dir", "timeout_seconds": 60}`,
			},
			{
				name: "empty_work_dir",
				spec: ExecutionSpec{
					Argv:           []string{"run.sh"},
					WorkDir:        "",
					TimeoutSeconds: 60,
				},
				jsonSpec: `{"argv": ["run.sh"], "work_dir": "", "timeout_seconds": 60}`,
			},
			{
				name: "empty_argv",
				spec: ExecutionSpec{
					Argv:           []string{},
					WorkDir:        absDir,
					TimeoutSeconds: 60,
				},
				jsonSpec: `{"argv": [], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": 60}`,
			},
			{
				name: "empty_first_argv_element",
				spec: ExecutionSpec{
					Argv:           []string{""},
					WorkDir:        absDir,
					TimeoutSeconds: 60,
				},
				jsonSpec: `{"argv": [""], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": 60}`,
			},
			{
				name: "whitespace_first_argv_element",
				spec: ExecutionSpec{
					Argv:           []string{"   "},
					WorkDir:        absDir,
					TimeoutSeconds: 60,
				},
				jsonSpec: `{"argv": ["   "], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": 60}`,
			},
			{
				name: "zero_timeout_seconds",
				spec: ExecutionSpec{
					Argv:           []string{"run.sh"},
					WorkDir:        absDir,
					TimeoutSeconds: 0,
				},
				jsonSpec: `{"argv": ["run.sh"], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": 0}`,
			},
			{
				name: "negative_timeout_seconds",
				spec: ExecutionSpec{
					Argv:           []string{"run.sh"},
					WorkDir:        absDir,
					TimeoutSeconds: -30,
				},
				jsonSpec: `{"argv": ["run.sh"], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": -30}`,
			},
			{
				name: "excessive_timeout_seconds",
				spec: ExecutionSpec{
					Argv:           []string{"run.sh"},
					WorkDir:        absDir,
					TimeoutSeconds: MaxExecutionTimeoutSeconds + 1,
				},
				jsonSpec: `{"argv": ["run.sh"], "work_dir": ` + quoteJSON(absDir) + `, "timeout_seconds": 86401}`,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// 1. Direct validation must fail
				if err := tc.spec.Validate(); err == nil {
					t.Fatalf("expected ExecutionSpec.Validate() to reject %s, got nil", tc.name)
				}

				// 2. Put must reject the malformed job
				reg := Registry{Jobs: map[string]Job{}}
				job := Job{
					Schedule: Schedule{
						JobID:           "test/job",
						IntervalSeconds: 60,
						MissedRun:       MissedSkip,
					},
					State:     JobArmed,
					Execution: &tc.spec,
				}
				if err := reg.Put(job, now); err == nil {
					t.Fatalf("expected Registry.Put to reject malformed execution spec %s, got nil", tc.name)
				}

				// 3. LoadRegistry must reject a persisted file with malformed execution spec
				rawJSON := `{"schema": "fak.loop-registry.v1", "jobs": {"test/job": {"schedule": {"job_id": "test/job", "interval_seconds": 60, "missed_run": "skip"}, "state": "armed", "execution": ` + tc.jsonSpec + `}}}`
				badFile := filepath.Join(t.TempDir(), "bad.json")
				if err := os.WriteFile(badFile, []byte(rawJSON), 0o644); err != nil {
					t.Fatalf("failed to write test file: %v", err)
				}
				if _, err := LoadRegistry(badFile); err == nil {
					t.Fatalf("expected LoadRegistry to reject malformed execution json for %s, got nil", tc.name)
				}

				// 4. SaveRegistry must reject a registry constructed with malformed execution spec
				directReg := Registry{
					Jobs: map[string]Job{
						"test/job": job,
					},
				}
				if err := SaveRegistry(filepath.Join(t.TempDir(), "should-fail.json"), directReg); err == nil {
					t.Fatalf("expected SaveRegistry to reject malformed execution spec %s, got nil", tc.name)
				}
			})
		}
	})
}

func TestExecutionSpecClone(t *testing.T) {
	var nilSpec *ExecutionSpec
	if nilSpec.Clone() != nil {
		t.Fatalf("expected nil clone for nil ExecutionSpec")
	}

	orig := &ExecutionSpec{
		Argv:           []string{"cmd", "arg1"},
		WorkDir:        filepath.Clean(os.TempDir()),
		TimeoutSeconds: 42,
	}
	cp := orig.Clone()
	if cp == orig {
		t.Fatalf("Clone did not create a new pointer")
	}
	if !reflect.DeepEqual(cp, orig) {
		t.Fatalf("Clone mismatch: got %#v, want %#v", cp, orig)
	}

	// Mutating cp.Argv must not mutate orig.Argv
	cp.Argv[0] = "mutated"
	if orig.Argv[0] == "mutated" {
		t.Fatalf("Clone did not deep copy Argv slice")
	}
}

func quoteJSON(s string) string {
	b, _ := os.ReadFile(s)
	_ = b
	// Escape backslashes for JSON string formatting on Windows
	escaped := ""
	for _, c := range s {
		if c == '\\' {
			escaped += "\\\\"
		} else if c == '"' {
			escaped += "\\\""
		} else {
			escaped += string(c)
		}
	}
	return `"` + escaped + `"`
}
