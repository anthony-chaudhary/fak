package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestSymptomSubtypeCandidateBuild(t *testing.T) {
	cases := []struct{ detail, want string }{
		{"candidate selected symptom test did not build", "SYMPTOM_CANDIDATE_BUILD"},
		{"candidate selected symptom test failed", "SYMPTOM_CANDIDATE_FAILED"},
		{"parent selected symptom test did not build", "SYMPTOM_PARENT_BUILD"},
		{"no changed test file", "SYMPTOM_NO_TEST"},
	}
	for _, tc := range cases {
		if got := symptomSubtype(tc.detail); got != tc.want {
			t.Errorf("symptomSubtype(%q) = %q, want %q", tc.detail, got, tc.want)
		}
	}
}

func TestWorkerLandSymptomVerdictCandidateBuildRefuses(t *testing.T) {
	res := workerLandSymptomVerdict(abi.WitnessRefuted, "candidate selected symptom test did not build", time.Minute)
	if res.OK || res.Code != "SYMPTOM_UNWITNESSED" || res.SymptomVerdict != "SYMPTOM_CANDIDATE_BUILD" {
		t.Fatalf("ok=%v code=%q verdict=%q", res.OK, res.Code, res.SymptomVerdict)
	}
	if res.SymptomAdvisory != "" || strings.HasPrefix(res.SymptomVerdict, "SYMPTOM_ABSTAIN_") {
		t.Fatalf("refusal treated as abstention: verdict=%q advisory set=%v", res.SymptomVerdict, res.SymptomAdvisory != "")
	}
}
