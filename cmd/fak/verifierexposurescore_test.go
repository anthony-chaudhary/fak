package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/verifierexposure"
)

func TestVerifierExposureScoreJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runVerifierExposureScore(&out, &errOut, []string{"--workspace", repoRoot(), "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got verifierexposure.Report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Schema != verifierexposure.Schema || got.GateCount < 6 || len(got.Worklist) != got.GateCount {
		t.Fatalf("report=%+v", got)
	}
	if got.VerifierExposureDebt != 0 {
		t.Fatalf("verifier exposure debt=%d, want zero; worklist=%+v", got.VerifierExposureDebt, got.Worklist)
	}
}
