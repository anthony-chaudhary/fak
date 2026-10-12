package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// A run the symptom budget killed can leave a detail that also names a zero match; the
// budget expiry must win so the operator raises --symptom-timeout instead of chasing a
// selector that was never actually evaluated.
func TestSymptomRefusalDetailClassifiesBudgetExpiryAsTimeout(t *testing.T) {
	for _, tc := range []struct {
		detail string
		want   string
	}{
		{"timed out before a trustworthy verdict (symptom budget expired: context deadline exceeded); discarded result: parent symptom selector matched no executed test", "SYMPTOM_TIMEOUT"},
		{"parent selected symptom test timed out (symptom budget expired: context deadline exceeded)", "SYMPTOM_TIMEOUT"},
		{"parent symptom selector matched no executed test", "SYMPTOM_NO_MATCH"},
		{"parent selected symptom test did not build", "SYMPTOM_PARENT_BUILD"},
	} {
		if got := symptomRefusalDetail("abstained", tc.detail); !strings.HasPrefix(got, tc.want+":") {
			t.Errorf("symptomRefusalDetail(%q) = %q, want subtype %s", tc.detail, got, tc.want)
		}
	}
}

func TestWorktreeWorkerLandRejectsNonPositiveSymptomTimeout(t *testing.T) {
	for _, value := range []string{"0s", "-5m"} {
		var stdout, stderr bytes.Buffer
		res, code := runWorktreeWorkerLand(&stdout, &stderr, []string{"--worktree", t.TempDir(), "--symptom-timeout", value})
		if code != 2 || res.OK || res.Code != "SYMPTOM_TIMEOUT_INVALID" {
			t.Fatalf("--symptom-timeout %s: result=%+v code=%d stderr=%s", value, res, code, stderr.String())
		}
	}
}

func TestWorktreeWorkerLandPrepareThreadsSymptomTimeout(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want time.Duration
	}{
		{"default budget unchanged", nil, defaultWorkerLandSymptomTimeout},
		{"operator budget", []string{"--symptom-timeout", "37m"}, 37 * time.Minute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo, wt, base, paths := newPreparedCLIWorkerFixture(t, true)
			originalBuild, originalSymptom := worktreeWorkerPreparedGoBuildVerify, worktreeWorkerPreparedSymptomVerify
			var got []time.Duration
			worktreeWorkerPreparedGoBuildVerify = func(string) (bool, string) { return true, "built" }
			worktreeWorkerPreparedSymptomVerify = func(_ string, _ string, _ []string, timeout time.Duration) workerworktree.Result {
				got = append(got, timeout)
				return workerworktree.Result{OK: true}
			}
			t.Cleanup(func() {
				worktreeWorkerPreparedGoBuildVerify, worktreeWorkerPreparedSymptomVerify = originalBuild, originalSymptom
			})
			res, receipt, code, stderr := runPreparedCLI(t, append(preparedCLIArgs("prepare", repo, wt, base, paths), tc.args...))
			if code != 0 || !res.OK || receipt == nil {
				t.Fatalf("fix prepare result=%+v receipt=%+v code=%d stderr=%s", res, receipt, code, stderr)
			}
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("symptom verifier budgets=%v, want [%s]", got, tc.want)
			}
		})
	}
}

// Only a refutation refuses a fix(*) land; every abstention lands with a closed
// SYMPTOM_ABSTAIN_* advisory token.
// fak-test:runtime fast est=1ms lane=default
func TestWorkerLandSymptomVerdictRefusesOnlyRefutations(t *testing.T) {
	for _, tc := range []struct {
		outcome abi.WitnessOutcome
		detail  string
		wantOK  bool
		want    string
	}{
		{abi.WitnessConfirmed, "selected symptom test failed at parent and passed at candidate", true, symptomVerdictConfirmed},
		{abi.WitnessAbstain, "timed out before a trustworthy verdict (symptom budget expired: context deadline exceeded); discarded result: parent selected symptom test did not build", true, symptomVerdictAbstainTimeout},
		{abi.WitnessAbstain, "parent selected symptom test timed out", true, symptomVerdictAbstainTimeout},
		{abi.WitnessAbstain, "parent selected symptom test did not build: every selected test names fix-introduced Abs", true, symptomVerdictAbstainParentBuild},
		{abi.WitnessAbstain, "parent changed-test package did not build", true, symptomVerdictAbstainParentBuild},
		{abi.WitnessAbstain, "invalid symptom test selector", true, symptomVerdictAbstainSelection},
		{abi.WitnessAbstain, "candidate scratch worktree failed", true, symptomVerdictAbstainOther},
		{abi.WitnessRefuted, "candidate selected symptom test failed", false, "SYMPTOM_CANDIDATE_FAILED"},
		{abi.WitnessRefuted, "selected symptom test passed at parent", false, "SYMPTOM_PARENT_PASSED"},
		{abi.WitnessRefuted, "no changed test file", false, "SYMPTOM_NO_TEST"},
	} {
		res := workerLandSymptomVerdict(tc.outcome, tc.detail, time.Minute)
		if res.OK != tc.wantOK || res.SymptomVerdict != tc.want {
			t.Errorf("%v %q: ok=%v verdict=%q, want ok=%v verdict=%q", tc.outcome, tc.detail, res.OK, res.SymptomVerdict, tc.wantOK, tc.want)
		}
		if !tc.wantOK && res.Code != "SYMPTOM_UNWITNESSED" {
			t.Errorf("%q: code=%q, want SYMPTOM_UNWITNESSED", tc.detail, res.Code)
		}
		if tc.wantOK && tc.outcome == abi.WitnessAbstain && !strings.HasPrefix(res.SymptomAdvisory, tc.want+":") {
			t.Errorf("%q: advisory=%q, want %s prefix", tc.detail, res.SymptomAdvisory, tc.want)
		}
	}
}
