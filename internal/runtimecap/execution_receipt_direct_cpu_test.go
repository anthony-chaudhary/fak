package runtimecap

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestDirectCPUReceiptPreservesSelectedEnvelopeModel(t *testing.T) {
	t.Parallel()
	report := Report{
		Schema: Schema, GOOS: "linux", GOARCH: "amd64", BinaryRunnable: true,
		SupportedCPUEnvelopes: []CPUEnvelope{
			{ID: "other-envelope", Model: "other-model"},
			{ID: "selected-envelope", Model: "fixture-qwen3.8"},
		},
		ModelExecution: Execution{
			Runnable: true, Engine: "fak-native", Backend: "cpu-ref",
			Mode: executionModeStandard, CPUEnvelope: "selected-envelope",
			PayloadCompatibility: payloadCompatibilitySupported,
		},
	}
	for _, tc := range []struct {
		name, envelope, model string
		valid                 bool
	}{
		{"selected", "selected-envelope", "fixture-qwen3.8", true},
		{"missing", "unlisted-envelope", EvidenceUnknown, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report.ModelExecution.CPUEnvelope = tc.envelope
			r := ExecutionModeReceiptFromReport(report)
			if r.Valid != tc.valid || r.Identity.Model != tc.model || r.Mode != ExecutionModeOfflineModelBacked || r.OperatingEnvelope != tc.envelope {
				t.Fatalf("direct CPU identity projection = %+v", r)
			}
			wantHealth := ExecutionHealthOffline
			if !tc.valid {
				wantHealth = ExecutionHealthRefused
				if r.ValidationReason == nil || r.ValidationReason.Code != "model_identity_unwitnessed" {
					t.Fatalf("missing envelope must remain unverified: %+v", r)
				}
			} else if r.ValidationReason != nil {
				t.Fatalf("selected envelope unexpectedly refused: %+v", r)
			}
			if r.Identity.Engine != "fak-native" || r.Identity.Backend != "cpu-ref" || r.Health != wantHealth || r.Status.Health != wantHealth || r.Audit.Health != wantHealth {
				t.Fatalf("execution identity or health changed: %+v", r)
			}
			if r.Witness.Status != EvidenceObserved || r.Witness.Source != "runtime-capabilities" || r.Witness.Certification != "pre-payload-observation" || r.Status.Evidence != EvidenceUnwitnessed || r.Audit.Evidence != EvidenceUnwitnessed {
				t.Fatalf("projection overstated execution evidence: %+v", r)
			}
		})
	}
}
