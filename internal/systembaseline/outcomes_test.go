package systembaseline

import (
	"encoding/json"
	"testing"
	"time"
)

func TestOutcomeCountsReadout(t *testing.T) {
	cleanPolicy := DefaultPolicy()
	clean := Build(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, cleanPolicy, 0, false)

	refusalPolicy := DefaultPolicy()
	refusalPolicy.MaximumNonSUTCPUPercent = 1
	refusal := Build(quietFixture(100e6), fixture(500e6, 0), 10, time.Second, refusalPolicy, 0, false)
	if refusal.Verdict != VerdictInvestigate {
		t.Fatalf("refusal fixture verdict = %q", refusal.Verdict)
	}

	invalid := clean
	invalid.Digest = "sha256:corrupt"

	counts := CountOutcomes([]Report{clean, refusal, invalid})
	want := OutcomeCounts{Success: 1, Refusal: 1, Error: 1}
	if counts != want {
		t.Fatalf("CountOutcomes() = %+v, want %+v", counts, want)
	}
	readout, err := json.Marshal(counts)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(readout), `{"success":1,"refusal":1,"error":1}`; got != want {
		t.Fatalf("readout = %s, want %s", got, want)
	}
	t.Logf("system-baseline outcomes: %s", readout)
}

func TestOutcomeCountsClassifiesValidInvalidVerdictAsError(t *testing.T) {
	invalid := Build(nil, nil, 0, time.Second, DefaultPolicy(), 1, false)
	if err := invalid.Validate(); err != nil {
		t.Fatalf("invalid-verdict fixture should be structurally valid: %v", err)
	}
	if got := CountOutcomes([]Report{invalid}); got != (OutcomeCounts{Error: 1}) {
		t.Fatalf("CountOutcomes(invalid) = %+v", got)
	}
}
