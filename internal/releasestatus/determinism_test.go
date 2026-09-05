package releasestatus

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"sync"
	"testing"
)

// TestDeterminismActionableCIBaseRedDiagnosis verifies that releasestatus evaluation
// for Actionable CI base-red diagnosis is strictly deterministic: given identical
// inputs, running the evaluation twice yields deeply equal and byte-identical outputs.
func TestDeterminismActionableCIBaseRedDiagnosis(t *testing.T) {
	logBytes, err := os.ReadFile("testdata/ci-fast-go-test-failed.log")
	if err != nil {
		t.Fatalf("failed reading testdata log: %v", err)
	}
	capturedLog := string(logBytes)

	run := CIRun{
		DatabaseID: 33047582124,
		HeadSHA:    "c222c119b4f12d780e61030cefabadf508eaa265",
		Conclusion: "failure",
		URL:        "https://github.com/anthony-chaudhary/fak/actions/runs/33047582124",
	}
	workflow := "ci-fast.yml"
	failedSteps := []string{"go test ./... (no -race)", "vet", "checkout"}

	// 1. Evaluate twice sequentially and assert deep equality.
	diag1 := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
	diag2 := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)

	if !reflect.DeepEqual(diag1, diag2) {
		t.Fatalf("DiagnoseCIFailure diverged between identical runs:\n run1: %#v\n run2: %#v", diag1, diag2)
	}

	// Non-vacuity assertions on the base-red diagnosis.
	if diag1.Status != "diagnosed" || diag1.Kind != "go_test_failure" {
		t.Fatalf("unexpected diagnosis status/kind: status=%q, kind=%q", diag1.Status, diag1.Kind)
	}
	if diag1.Summary.PackageCount != 12 || diag1.Summary.WorkUnitCount != 12 {
		t.Fatalf("expected 12 work units, got %d", diag1.Summary.WorkUnitCount)
	}
	if diag1.Primary == nil || diag1.Primary.Package != "github.com/anthony-chaudhary/fak/cmd/fak" {
		t.Fatalf("unexpected primary cause: %#v", diag1.Primary)
	}

	// Assert JSON byte-level identity.
	b1, err := json.Marshal(diag1)
	if err != nil {
		t.Fatalf("marshal diag1: %v", err)
	}
	b2, err := json.Marshal(diag2)
	if err != nil {
		t.Fatalf("marshal diag2: %v", err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("DiagnoseCIFailure JSON serialization diverged:\n b1: %s\n b2: %s", b1, b2)
	}

	// 2. Evaluate full release status Fold twice with the CI_BASE_RED diagnosis.
	makeFacts := func(d CIDiagnosis) Facts {
		return Facts{
			Root:    "/repo",
			HeadSHA: "c222c119b4f12d780e61030cefabadf508eaa265",
			Branch:  "main",
			Decision: Decision{
				Decision: "hold",
				Reason:   "main CI is failing",
				Blockers: []string{"CI_BASE_RED"},
			},
			RollingTags:          []string{"v1.0.0", "v1.1.0"},
			LastTag:              "v1.1.0",
			CommitsSinceTag:      12,
			FilesTouchedSinceTag: 8,
			Dirty: []DirtyEntry{
				{Path: "README.md", Untracked: false},
			},
			CIDiagnosis: d,
			Cadence: CadenceText{
				Present: true,
				Path:    ".github/workflows/release-cadence.yml",
				Text:    "name: release-cadence\n",
			},
			Stable: []StableTag{
				{
					Tag:           "stable/2026-05-stable",
					SHA:           "cccc2222dddd",
					EvidenceFound: true,
					Evidence: map[string]string{
						"candidate_sha":      "cccc2222dddd",
						"underlying_version": "v1.0.0",
						"codename":           "2026-05-stable",
					},
					UnderlyingExists: true,
					UnderlyingTagSHA: "cccc2222dddd",
					Version:          "v1.0.0",
				},
			},
			StableWindowDays:  3,
			CandidateAgeDays:  1.5,
			CandidateAgeKnown: true,
			SuggestedCodename: "2026-06-stable",
			CandidateSHA:      "c222c119b4f12d780e61030cefabadf508eaa265",
			BranchRegime: FoldBranchRegime(BranchRegimeFacts{
				DevelopmentBranch: "main",
				ReleaseBranch:     "main",
			}),
		}
	}

	status1 := Fold(makeFacts(diag1))
	status2 := Fold(makeFacts(diag2))

	if !reflect.DeepEqual(status1, status2) {
		t.Fatalf("Fold status diverged between identical runs:\n status1: %#v\n status2: %#v", status1, status2)
	}
	if status1.NextAction.Kind != "fix_ci" {
		t.Fatalf("expected next_action.kind 'fix_ci', got %q", status1.NextAction.Kind)
	}

	sb1, err := json.Marshal(status1)
	if err != nil {
		t.Fatalf("marshal status1: %v", err)
	}
	sb2, err := json.Marshal(status2)
	if err != nil {
		t.Fatalf("marshal status2: %v", err)
	}
	if !bytes.Equal(sb1, sb2) {
		t.Fatalf("Fold status JSON diverged:\n sb1: %s\n sb2: %s", sb1, sb2)
	}

	render1 := Render(status1)
	render2 := Render(status2)
	if render1 != render2 {
		t.Fatalf("Render output diverged:\n r1: %s\n r2: %s", render1, render2)
	}

	triageLine1 := AttentionTriageLine(status1)
	triageLine2 := AttentionTriageLine(status2)
	if triageLine1 != triageLine2 {
		t.Fatalf("AttentionTriageLine diverged: %q vs %q", triageLine1, triageLine2)
	}
}

// TestDeterminismBaseRedDiagnosisVariations verifies determinism across various
// base-red failure categories: billing admission, non-Go step failures, and timeouts.
func TestDeterminismBaseRedDiagnosisVariations(t *testing.T) {
	cases := []struct {
		name       string
		workflow   string
		run        CIRun
		steps      []string
		log        string
		wantAction string
		wantKind   string
	}{
		{
			name:       "billing_wall",
			workflow:   "ci.yml",
			run:        CIRun{DatabaseID: 101, Conclusion: "failure"},
			steps:      []string{"build"},
			log:        "The job was not started because recent account payments have failed. Check Billing & plans.",
			wantAction: "fix_ci_billing",
			wantKind:   "billing",
		},
		{
			name:       "step_only_failure",
			workflow:   "ci.yml",
			run:        CIRun{DatabaseID: 102, Conclusion: "failure"},
			steps:      []string{"gofmt", "vet"},
			log:        "",
			wantAction: "fix_gofmt",
			wantKind:   "gofmt",
		},
		{
			name:       "timed_out",
			workflow:   "ci.yml",
			run:        CIRun{DatabaseID: 103, Conclusion: "timed_out"},
			steps:      []string{"go test ./..."},
			log:        "",
			wantAction: "retry_ci",
			wantKind:   "timed_out",
		},
		{
			name:       "startup_failure",
			workflow:   "ci.yml",
			run:        CIRun{DatabaseID: 104, Conclusion: "startup_failure"},
			steps:      nil,
			log:        "",
			wantAction: "retry_ci",
			wantKind:   "startup_failure",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := DiagnoseCIFailure(tc.workflow, tc.run, tc.steps, tc.log)
			second := DiagnoseCIFailure(tc.workflow, tc.run, tc.steps, tc.log)

			if !reflect.DeepEqual(first, second) {
				t.Fatalf("case %s diverged between runs:\n first: %#v\n second: %#v", tc.name, first, second)
			}
			if first.Action != tc.wantAction || first.Kind != tc.wantKind {
				t.Fatalf("case %s: action=%q kind=%q, want action=%q kind=%q", tc.name, first.Action, first.Kind, tc.wantAction, tc.wantKind)
			}

			fb1, _ := json.Marshal(first)
			fb2, _ := json.Marshal(second)
			if !bytes.Equal(fb1, fb2) {
				t.Fatalf("case %s JSON diverged:\n fb1: %s\n fb2: %s", tc.name, fb1, fb2)
			}
		})
	}
}

// TestDeterminismBaseRedDiagnosisConcurrentRaceWitness serves as a concurrent
// execution and race witness: multiple goroutines concurrently evaluate the same
// captured CI base-red failure and assert deep equality against a reference result.
func TestDeterminismBaseRedDiagnosisConcurrentRaceWitness(t *testing.T) {
	logBytes, err := os.ReadFile("testdata/ci-fast-go-test-failed.log")
	if err != nil {
		t.Fatalf("failed reading testdata log: %v", err)
	}
	capturedLog := string(logBytes)

	run := CIRun{
		DatabaseID: 33047582124,
		HeadSHA:    "c222c119b4f12d780e61030cefabadf508eaa265",
		Conclusion: "failure",
	}
	workflow := "ci-fast.yml"
	failedSteps := []string{"go test ./... (no -race)", "vet"}

	refDiag := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
	refStatus := Fold(Facts{
		Root:    "/repo",
		HeadSHA: "c222c119b4f12d780e61030cefabadf508eaa265",
		Branch:  "main",
		Decision: Decision{
			Decision: "hold",
			Blockers: []string{"CI_BASE_RED"},
		},
		CIDiagnosis: refDiag,
	})

	const goroutines = 32
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines*2)

	start := make(chan struct{})
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start

			gotDiag := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
			if !reflect.DeepEqual(gotDiag, refDiag) {
				t.Errorf("concurrent DiagnoseCIFailure diverged from reference")
				return
			}

			gotStatus := Fold(Facts{
				Root:    "/repo",
				HeadSHA: "c222c119b4f12d780e61030cefabadf508eaa265",
				Branch:  "main",
				Decision: Decision{
					Decision: "hold",
					Blockers: []string{"CI_BASE_RED"},
				},
				CIDiagnosis: gotDiag,
			})
			if !reflect.DeepEqual(gotStatus, refStatus) {
				t.Errorf("concurrent Fold status diverged from reference")
				return
			}
		}()
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}

// TestDeterminismBaseRedDiagnosisRepeatIterations tests for map-iteration or
// ordering instability across 100 sequential evaluations.
func TestDeterminismBaseRedDiagnosisRepeatIterations(t *testing.T) {
	logBytes, err := os.ReadFile("testdata/ci-fast-go-test-failed.log")
	if err != nil {
		t.Fatalf("failed reading testdata log: %v", err)
	}
	capturedLog := string(logBytes)

	run := CIRun{DatabaseID: 33047582124, Conclusion: "failure"}
	workflow := "ci-fast.yml"
	steps := []string{"checkout", "vet", "go test ./... (no -race)", "gofmt", "build"}

	ref := DiagnoseCIFailure(workflow, run, steps, capturedLog)
	refJSON, err := json.Marshal(ref)
	if err != nil {
		t.Fatalf("marshal reference: %v", err)
	}

	for i := 0; i < 100; i++ {
		got := DiagnoseCIFailure(workflow, run, steps, capturedLog)
		if !reflect.DeepEqual(got, ref) {
			t.Fatalf("iteration %d: DiagnoseCIFailure output diverged", i)
		}
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("iteration %d: marshal: %v", i, err)
		}
		if !bytes.Equal(gotJSON, refJSON) {
			t.Fatalf("iteration %d: JSON output diverged", i)
		}
	}
}
