package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestWorkerLandSymptomPreflight(t *testing.T) {
	cases := []struct {
		name    string
		paths   []string
		refused bool
	}{
		{"unscoped defers to witness", nil, false},
		{"empty slice defers to witness", []string{}, false},
		{"go test file", []string{"pkg/calc.go", "pkg/calc_test.go"}, false},
		{"go test file alone", []string{"pkg/calc_test.go"}, false},
		{"python test_ prefix", []string{"tools/x.py", "tools/test_x.py"}, false},
		{"python _test suffix", []string{"tools/x_test.py"}, false},
		{"windows separators", []string{`pkg\calc_test.go`}, false},
		{"non-test go only", []string{"pkg/calc.go"}, true},
		{"docs and go", []string{"README.md", "pkg/calc.go"}, true},
		{"testdata go test file", []string{"pkg/testdata/fixture_test.go"}, true},
		{"testdata python file", []string{"pkg/testdata/test_fixture.py"}, true},
		{"top-level testdata", []string{"testdata/x_test.go", "pkg/calc.go"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, refused := workerLandSymptomPreflight(tc.paths)
			if refused != tc.refused {
				t.Fatalf("refused=%v want %v (res=%+v)", refused, tc.refused, res)
			}
			if !refused {
				if res.Code != "" || res.SymptomVerdict != "" {
					t.Fatalf("non-refusal carried a verdict: code=%q verdict=%q", res.Code, res.SymptomVerdict)
				}
				return
			}
			if res.OK || res.Code != "SYMPTOM_UNWITNESSED" || res.SymptomVerdict != "SYMPTOM_NO_TEST" {
				t.Fatalf("ok=%v code=%q verdict=%q", res.OK, res.Code, res.SymptomVerdict)
			}
			if !strings.HasPrefix(res.Detail, "SYMPTOM_NO_TEST:") {
				t.Fatalf("detail lacks SYMPTOM_NO_TEST: token prefix: %q", res.Detail)
			}
		})
	}
}

func TestWorktreeWorkerLandPrepareFixWithoutTestPathRefusedBeforeBuild(t *testing.T) {
	t.Setenv("GOFLAGS", "-tags=preparedcli")
	repo, wt, base, _ := newPreparedCLIWorkerFixture(t, true)
	before := worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main")

	originalBuild, originalSymptom := worktreeWorkerPreparedGoBuildVerify, worktreeWorkerPreparedSymptomVerify
	t.Cleanup(func() {
		worktreeWorkerPreparedGoBuildVerify, worktreeWorkerPreparedSymptomVerify = originalBuild, originalSymptom
	})
	buildCalls, symptomCalls := 0, 0
	worktreeWorkerPreparedGoBuildVerify = func(string) (bool, string) { buildCalls++; return true, "built" }
	worktreeWorkerPreparedSymptomVerify = func(string, string, []string, time.Duration) workerworktree.Result {
		symptomCalls++
		return workerworktree.Result{OK: true}
	}

	res, receipt, code, stderr := runPreparedCLI(t, preparedCLIArgs("prepare", repo, wt, base, []string{"pkg/calc.go"}))
	if code != 1 || res.OK {
		t.Fatalf("code=%d ok=%v want code 1 refusal; stderr=%s", code, res.OK, stderr)
	}
	if res.Code != "SYMPTOM_UNWITNESSED" || res.SymptomVerdict != "SYMPTOM_NO_TEST" {
		t.Fatalf("code=%q verdict=%q want SYMPTOM_UNWITNESSED/SYMPTOM_NO_TEST", res.Code, res.SymptomVerdict)
	}
	if receipt != nil {
		t.Fatalf("refused prepare minted a receipt: %+v", receipt)
	}
	if buildCalls != 0 || symptomCalls != 0 {
		t.Fatalf("preflight ran seams: build=%d symptom=%d", buildCalls, symptomCalls)
	}
	if after := worktreeWorkerTestGit(t, repo, "rev-parse", "refs/heads/main"); after != before {
		t.Fatalf("refused prepare moved trunk: %s -> %s", before, after)
	}
}
