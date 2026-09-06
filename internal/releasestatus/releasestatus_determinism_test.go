package releasestatus

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"sync"
	"testing"
)

// TestReleaseStatusDeterminism validates that CI base-red failure parsing, diagnosis,
// and status folding are strictly deterministic and safe under concurrent execution.
// It ensures that identical inputs yield deeply equal and byte-identical outputs,
// with no map ordering instability, no unseeded random values, and no data races.
func TestReleaseStatusDeterminism(t *testing.T) {
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

	// 1. Sequential run twice on identical input: assert DeepEqual and byte-identical JSON.
	t.Run("SequentialIdenticalOutputs", func(t *testing.T) {
		units1 := ParseGoTestWorkUnits(capturedLog)
		units2 := ParseGoTestWorkUnits(capturedLog)
		if !reflect.DeepEqual(units1, units2) {
			t.Fatalf("ParseGoTestWorkUnits diverged between runs:\n units1: %#v\n units2: %#v", units1, units2)
		}
		uBytes1, err := json.Marshal(units1)
		if err != nil {
			t.Fatalf("marshal units1: %v", err)
		}
		uBytes2, err := json.Marshal(units2)
		if err != nil {
			t.Fatalf("marshal units2: %v", err)
		}
		if !bytes.Equal(uBytes1, uBytes2) {
			t.Fatalf("ParseGoTestWorkUnits JSON diverged:\n b1: %s\n b2: %s", uBytes1, uBytes2)
		}

		diag1 := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
		diag2 := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
		if !reflect.DeepEqual(diag1, diag2) {
			t.Fatalf("DiagnoseCIFailure diverged between runs:\n diag1: %#v\n diag2: %#v", diag1, diag2)
		}
		dBytes1, err := json.Marshal(diag1)
		if err != nil {
			t.Fatalf("marshal diag1: %v", err)
		}
		dBytes2, err := json.Marshal(diag2)
		if err != nil {
			t.Fatalf("marshal diag2: %v", err)
		}
		if !bytes.Equal(dBytes1, dBytes2) {
			t.Fatalf("DiagnoseCIFailure JSON diverged:\n b1: %s\n b2: %s", dBytes1, dBytes2)
		}

		makeFacts := func(d CIDiagnosis) Facts {
			return Facts{
				Root:    "/repo",
				HeadSHA: "c222c119b4f12d780e61030cefabadf508eaa265",
				Branch:  "main",
				Decision: Decision{
					Decision: "hold",
					Reason:   "main CI is red",
					Blockers: []string{"CI_BASE_RED"},
				},
				RollingTags:          []string{"v1.0.0", "v1.1.0"},
				LastTag:              "v1.1.0",
				CommitsSinceTag:      10,
				FilesTouchedSinceTag: 5,
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
				CandidateAgeDays:  2.0,
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
			t.Fatalf("Fold status diverged:\n s1: %#v\n s2: %#v", status1, status2)
		}

		sBytes1, err := json.Marshal(status1)
		if err != nil {
			t.Fatalf("marshal status1: %v", err)
		}
		sBytes2, err := json.Marshal(status2)
		if err != nil {
			t.Fatalf("marshal status2: %v", err)
		}
		if !bytes.Equal(sBytes1, sBytes2) {
			t.Fatalf("Fold status JSON diverged:\n b1: %s\n b2: %s", sBytes1, sBytes2)
		}

		if r1, r2 := Render(status1), Render(status2); r1 != r2 {
			t.Fatalf("Render output diverged:\n r1: %s\n r2: %s", r1, r2)
		}

		if t1, t2 := AttentionTriageLine(status1), AttentionTriageLine(status2); t1 != t2 {
			t.Fatalf("AttentionTriageLine diverged: %q vs %q", t1, t2)
		}
	})

	// 2. Iteration loop verifying absence of map order instability.
	t.Run("RepeatIterationsStability", func(t *testing.T) {
		refDiag := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
		refJSON, err := json.Marshal(refDiag)
		if err != nil {
			t.Fatalf("marshal refDiag: %v", err)
		}

		for i := 0; i < 50; i++ {
			iterDiag := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
			if !reflect.DeepEqual(iterDiag, refDiag) {
				t.Fatalf("iteration %d: DiagnoseCIFailure diverged from reference", i)
			}
			iterJSON, err := json.Marshal(iterDiag)
			if err != nil {
				t.Fatalf("iteration %d: marshal: %v", i, err)
			}
			if !bytes.Equal(iterJSON, refJSON) {
				t.Fatalf("iteration %d: JSON output diverged", i)
			}
		}
	})

	// 3. Concurrent race witness across multiple goroutines.
	t.Run("ConcurrentRaceWitness", func(t *testing.T) {
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

		const workers = 32
		var wg sync.WaitGroup
		errCh := make(chan error, workers*3)
		start := make(chan struct{})

		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start

				gotUnits := ParseGoTestWorkUnits(capturedLog)
				if !reflect.DeepEqual(gotUnits, refDiag.WorkUnits) {
					t.Errorf("concurrent ParseGoTestWorkUnits diverged")
					return
				}

				gotDiag := DiagnoseCIFailure(workflow, run, failedSteps, capturedLog)
				if !reflect.DeepEqual(gotDiag, refDiag) {
					t.Errorf("concurrent DiagnoseCIFailure diverged")
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
					t.Errorf("concurrent Fold diverged")
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
	})
}
