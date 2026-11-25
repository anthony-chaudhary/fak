package main

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1ms
func TestSymptomRegressionSubtypeMapsBeforePassedAtParent(t *testing.T) {
	cases := []struct{ detail, want string }{
		{"candidate selected symptom test regressed: TestOld failed at candidate on rerun and passed at parent", "SYMPTOM_CANDIDATE_REGRESSION"},
		{"candidate selected symptom test regressed: TestNew failed at candidate on rerun and is absent at parent", "SYMPTOM_CANDIDATE_REGRESSION"},
		{"candidate selected symptom test failed: TestB gave no explicit verdict at parent", "SYMPTOM_CANDIDATE_FAILED"},
	}
	for _, tc := range cases {
		if got := symptomSubtype(tc.detail); got != tc.want {
			t.Errorf("symptomSubtype(%q) = %q, want %q", tc.detail, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestSymptomRegressionVerdictRefusesAndTolerantConfirmAdvises(t *testing.T) {
	const regressed = "candidate selected symptom test regressed: TestOld failed at candidate on rerun and passed at parent"
	refused := workerLandSymptomVerdict(abi.WitnessRefuted, regressed, time.Minute)
	if refused.OK || refused.SymptomVerdict != "SYMPTOM_CANDIDATE_REGRESSION" {
		t.Errorf("regressed refutation = OK %v verdict %q, want refused SYMPTOM_CANDIDATE_REGRESSION", refused.OK, refused.SymptomVerdict)
	}

	const tolerated = "selected symptom test failed at parent and passed at candidate (tolerated at candidate: flaky on rerun TestA)"
	confirmed := workerLandSymptomVerdict(abi.WitnessConfirmed, tolerated, time.Minute)
	if !confirmed.OK || confirmed.SymptomVerdict != "SYMPTOM_CONFIRMED" {
		t.Errorf("tolerant confirm = OK %v verdict %q, want OK SYMPTOM_CONFIRMED", confirmed.OK, confirmed.SymptomVerdict)
	}
	if !strings.HasPrefix(confirmed.SymptomAdvisory, "SYMPTOM_CONFIRMED_TOLERANT:") {
		t.Errorf("tolerant confirm advisory = %q, want SYMPTOM_CONFIRMED_TOLERANT prefix", confirmed.SymptomAdvisory)
	}

	plain := workerLandSymptomVerdict(abi.WitnessConfirmed, "selected symptom test failed at parent and passed at candidate", time.Minute)
	if plain.SymptomAdvisory != "" {
		t.Errorf("plain confirm advisory = %q, want none", plain.SymptomAdvisory)
	}
}
