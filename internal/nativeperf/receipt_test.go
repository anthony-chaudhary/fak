package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func validReceipt(t *testing.T, role, lever string) ExperimentReceipt {
	t.Helper()
	r, err := BaselineTemplate(ActiveGraph(), lever)
	if err != nil {
		t.Fatal(err)
	}
	r.Role, r.Revision, r.Machine.ScrubbedID = role, "rev-123", "lab-node-class-a"
	r.Memory = MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
	r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
	r.ProfilerArtifacts = []ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
	for i := range r.Repetitions {
		r.Repetitions[i] = Repetition{EndToEndMilliseconds: 100, TokensPerSecond: 3 + float64(i)/10}
	}
	if role == RoleCandidate {
		r.ChangedAxes = []string{"lever:" + lever}
	}
	return r
}

func TestMetalAndCUDAReceiptsRoundTrip(t *testing.T) {
	for _, lever := range []string{"metal.command-buffer-amortization", "cuda.q8_1-activation-quant"} {
		r := validReceipt(t, RoleBaseline, lever)
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		got, err := DecodeReceipt(data)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateReceipt(ActiveGraph(), got); err != nil {
			t.Fatal(err)
		}
		if got.EnvelopeID != r.EnvelopeID || got.ChangedLeverID != lever {
			t.Fatalf("round trip drift: %+v", got)
		}
	}
}

func TestValidateReceiptFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ExperimentReceipt)
		want string
	}{
		{"envelope drift", func(r *ExperimentReceipt) { r.EnvelopeID = "other" }, "does not belong"},
		{"native identity", func(r *ExperimentReceipt) { r.Execution.Engine = "other" }, "must name fak-native"},
		{"fallback", func(r *ExperimentReceipt) { r.Execution.FallbackCount = 1 }, "fallback count"},
		{"missing repetition", func(r *ExperimentReceipt) { r.Repetitions = nil }, "repetition count"},
		{"private host", func(r *ExperimentReceipt) { r.Machine.ScrubbedID = "user@host" }, "private path/host"},
		{"multi axis", func(r *ExperimentReceipt) { r.ChangedAxes = []string{"lever:" + r.ChangedLeverID, "batch"} }, "exactly the candidate lever"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
			tt.edit(&r)
			err := ValidateReceipt(ActiveGraph(), r)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %q", err, tt.want)
			}
		})
	}
}

func TestCompareReceiptsDeterministicAndRejectsDrift(t *testing.T) {
	b := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	c := validReceipt(t, RoleCandidate, "metal.command-buffer-amortization")
	c.Revision = "rev-456"
	for i := range c.Repetitions {
		c.Repetitions[i].TokensPerSecond++
	}
	first, err := CompareReceipts(ActiveGraph(), b, c)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompareReceipts(ActiveGraph(), b, c)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || math.Abs(first.DeltaTokensPerS-1) > 1e-12 {
		t.Fatalf("comparison drift: %+v %+v", first, second)
	}
	c.Controls.Batch = 2
	if _, err := CompareReceipts(ActiveGraph(), b, c); err == nil || !strings.Contains(err.Error(), "undeclared control axis drifted") {
		t.Fatalf("drift err=%v", err)
	}
}
