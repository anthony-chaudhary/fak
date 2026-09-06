package cachevalue

import (
	"math"
	"strings"
	"testing"
)

func TestValidatePacket_ValidFixturePassesAllGates(t *testing.T) {
	pkt := NewValidSubagentWorkloadPacket()
	if err := ValidatePacket(pkt); err != nil {
		t.Fatalf("expected valid fixture to pass ValidatePacket, got error: %v", err)
	}

	if err := ValidateFrozenWorkload(pkt); err != nil {
		t.Fatalf("expected valid fixture to pass ValidateFrozenWorkload, got error: %v", err)
	}

	digest := HashPacket(pkt)
	if len(digest) != 64 {
		t.Fatalf("expected 64-char sha256 digest, got %q (len %d)", digest, len(digest))
	}
	if pkt.Digest != digest {
		t.Fatalf("expected packet digest %q to match HashPacket result %q", pkt.Digest, digest)
	}
}

func TestValidatePacket_Sub80PctAttainmentFailure(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(pkt *SubagentWorkloadPacket)
		wantErr string
	}{
		{
			name: "attained_bandwidth_below_target",
			mutate: func(pkt *SubagentWorkloadPacket) {
				// Theoretical peak is 273.056 GB/s, 80% is 218.4448 GB/s. 200 GB/s is ~73.2%.
				pkt.MeasuredRun.AttainedBandwidthGBps = 200.0
				pkt.MeasuredRun.AttainmentRatio = 200.0 / pkt.HardwareSpec.PeakBandwidthGBps
			},
			wantErr: "below 80% threshold",
		},
		{
			name: "passed_80pct_flag_false",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.MeasuredRun.Passed80Pct = false
			},
			wantErr: "Passed80Pct",
		},
		{
			name: "attainment_ratio_below_floor",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.MeasuredRun.AttainmentRatio = 0.79
			},
			wantErr: "attainment ratio",
		},
		{
			name: "zero_attained_bandwidth",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.MeasuredRun.AttainedBandwidthGBps = 0.0
				pkt.MeasuredRun.AttainmentRatio = 0.0
			},
			wantErr: "below 80% threshold",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pkt := NewValidSubagentWorkloadPacket()
			tc.mutate(pkt)
			err := ValidatePacket(pkt)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePacket_CosineParityFailure(t *testing.T) {
	tests := []struct {
		name    string
		parity  float64
		wantErr string
	}{
		{
			name:    "just_below_threshold",
			parity:  0.999899,
			wantErr: "cosine parity 0.999899 below required threshold 0.999900",
		},
		{
			name:    "significantly_degraded",
			parity:  0.950000,
			wantErr: "cosine parity",
		},
		{
			name:    "nan_value",
			parity:  math.NaN(),
			wantErr: "cosine parity is NaN",
		},
		{
			name:    "negative_value",
			parity:  -0.5,
			wantErr: "cosine parity",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pkt := NewValidSubagentWorkloadPacket()
			pkt.MeasuredRun.CosineParity = tc.parity
			err := ValidatePacket(pkt)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePacket_MissingDenominatorAndZeroValuesRejection(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(pkt *SubagentWorkloadPacket)
		wantErr string
	}{
		{
			name: "zero_hardware_peak_bandwidth",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.PeakBandwidthGBps = 0.0
			},
			wantErr: "hardware peak bandwidth must be greater than zero",
		},
		{
			name: "negative_hardware_peak_bandwidth",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.PeakBandwidthGBps = -10.0
			},
			wantErr: "hardware peak bandwidth must be greater than zero",
		},
		{
			name: "zero_hardware_target_80pct_bandwidth",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.Target80PctBandwidthGBps = 0.0
			},
			wantErr: "hardware target 80% bandwidth must be greater than zero",
		},
		{
			name: "zero_hardware_bus_width",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.BusWidth = 0
			},
			wantErr: "hardware bus width must be greater than zero",
		},
		{
			name: "zero_hardware_clock_mhz",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.ClockMHz = 0.0
			},
			wantErr: "hardware clock MHz must be greater than zero",
		},
		{
			name: "empty_hardware_arch",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.HardwareSpec.Arch = "   "
			},
			wantErr: "hardware arch is required",
		},
		{
			name: "empty_model_name",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.ModelSpec.Name = ""
			},
			wantErr: "model name is required",
		},
		{
			name: "zero_model_param_count",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.ModelSpec.ParamCount = 0
			},
			wantErr: "model parameter count must be greater than zero",
		},
		{
			name: "empty_model_quant_format",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.ModelSpec.QuantFormat = ""
			},
			wantErr: "model quant format is required",
		},
		{
			name: "zero_model_active_weight_bytes",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.ModelSpec.ActiveWeightBytes = 0
			},
			wantErr: "model active weight bytes must be greater than zero",
		},
		{
			name: "empty_workload_subagents",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.WorkloadSpec.Subagents = nil
			},
			wantErr: "workload subagents list must not be empty",
		},
		{
			name: "zero_workload_shared_prefix_tokens",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.WorkloadSpec.SharedPrefixTokens = 0
			},
			wantErr: "workload shared prefix tokens must be greater than zero",
		},
		{
			name: "zero_workload_private_suffix_tokens",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.WorkloadSpec.PrivateSuffixTokens = 0
			},
			wantErr: "workload private suffix tokens must be greater than zero",
		},
		{
			name: "zero_workload_total_turns",
			mutate: func(pkt *SubagentWorkloadPacket) {
				pkt.WorkloadSpec.TotalTurns = 0
			},
			wantErr: "workload total turns must be greater than zero",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pkt := NewValidSubagentWorkloadPacket()
			tc.mutate(pkt)
			err := ValidatePacket(pkt)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}

	// Test nil packet explicitly
	if err := ValidatePacket(nil); err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("expected nil packet error, got: %v", err)
	}
}

func TestValidatePacket_SchemaVersionMismatchRejection(t *testing.T) {
	tests := []struct {
		name    string
		schema  string
		wantErr string
	}{
		{
			name:    "v2_schema",
			schema:  "fak.subagent_workload/v2",
			wantErr: `schema version mismatch: got "fak.subagent_workload/v2", want "fak.subagent_workload/v1"`,
		},
		{
			name:    "empty_schema",
			schema:  "",
			wantErr: `schema version mismatch: got "", want "fak.subagent_workload/v1"`,
		},
		{
			name:    "wrong_namespace",
			schema:  "other.workload/v1",
			wantErr: `schema version mismatch`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pkt := NewValidSubagentWorkloadPacket()
			pkt.SchemaVersion = tc.schema
			err := ValidatePacket(pkt)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidatePacket_SimulatedRunRejectionWithoutProofTokens(t *testing.T) {
	pkt := NewValidSubagentWorkloadPacket()
	pkt.MeasuredRun.Simulated = true
	pkt.MeasuredRun.ProofToken = ""
	pkt.MeasuredRun.ProofTokens = nil

	if err := ValidatePacket(pkt); err == nil || !strings.Contains(err.Error(), "missing verified proof tokens") {
		t.Fatalf("expected simulated run without proof tokens to fail, got: %v", err)
	}

	// With single proof token, it should succeed
	pkt.MeasuredRun.ProofToken = "strix-sim-proof-7718ac"
	if err := ValidatePacket(pkt); err != nil {
		t.Fatalf("expected simulated run with valid proof token to succeed, got: %v", err)
	}

	// With proof tokens slice, it should succeed
	pkt.MeasuredRun.ProofToken = ""
	pkt.MeasuredRun.ProofTokens = []string{"proof-token-1"}
	if err := ValidatePacket(pkt); err != nil {
		t.Fatalf("expected simulated run with proof tokens slice to succeed, got: %v", err)
	}
}

func TestValidatePacket_NegativeDurationAndCountersRejection(t *testing.T) {
	// Negative duration
	pkt := NewValidSubagentWorkloadPacket()
	pkt.MeasuredRun.DurationMS = -0.5
	if err := ValidatePacket(pkt); err == nil || !strings.Contains(err.Error(), "negative duration") {
		t.Fatalf("expected negative duration rejection, got: %v", err)
	}

	// Negative prefill avoided
	pkt = NewValidSubagentWorkloadPacket()
	pkt.MeasuredRun.TotalPrefillAvoided = -1
	if err := ValidatePacket(pkt); err == nil || !strings.Contains(err.Error(), "total prefill avoided cannot be negative") {
		t.Fatalf("expected negative prefill avoided rejection, got: %v", err)
	}

	// Negative reprefill count
	pkt = NewValidSubagentWorkloadPacket()
	pkt.MeasuredRun.ReprefillCount = -1
	if err := ValidatePacket(pkt); err == nil || !strings.Contains(err.Error(), "reprefill count cannot be negative") {
		t.Fatalf("expected negative reprefill count rejection, got: %v", err)
	}
}

func TestHashPacket_DeterminismAndIdempotence(t *testing.T) {
	pkt := NewValidSubagentWorkloadPacket()
	h1 := HashPacket(pkt)
	h2 := HashPacket(pkt)
	if h1 != h2 {
		t.Fatalf("HashPacket is non-deterministic: %q != %q", h1, h2)
	}

	// Setting pkt.Digest does not alter hash calculation
	pkt.Digest = "pre-existing-digest-value"
	h3 := HashPacket(pkt)
	if h1 != h3 {
		t.Fatalf("HashPacket altered by pkt.Digest presence: %q != %q", h1, h3)
	}

	// Mutating payload alters the hash
	pkt.MeasuredRun.AttainedBandwidthGBps += 1.0
	h4 := HashPacket(pkt)
	if h1 == h4 {
		t.Fatalf("HashPacket failed to reflect payload mutation: %q == %q", h1, h4)
	}

	// Nil packet returns empty string
	if HashPacket(nil) != "" {
		t.Fatalf("expected empty string for nil packet")
	}
}

func TestFrozenModelSpec_SupportedModelsAndQuants(t *testing.T) {
	models := []string{ModelQwen38Coder35B, ModelQwen38Coder27B, ModelQwen38Coder14B}
	quants := []string{QuantFormatQ4KM, QuantFormatQ3KM}

	for _, m := range models {
		for _, q := range quants {
			spec, err := FrozenModelSpec(m, q)
			if err != nil {
				t.Fatalf("expected model %s quant %s to be valid, got: %v", m, q, err)
			}
			if spec.Name != m || spec.QuantFormat != q || spec.ParamCount <= 0 || spec.ActiveWeightBytes <= 0 {
				t.Fatalf("invalid spec for %s / %s: %+v", m, q, spec)
			}
		}
	}

	// Unsupported quant
	if _, err := FrozenModelSpec(ModelQwen38Coder27B, "Q8_0"); err == nil {
		t.Fatal("expected error for unsupported quant Q8_0")
	}

	// Unknown model
	if _, err := FrozenModelSpec("Unknown-Model-70B", QuantFormatQ4KM); err == nil {
		t.Fatal("expected error for unknown model")
	}
}

func TestValidateFrozenWorkload_ContextBoundaries(t *testing.T) {
	pkt := NewValidSubagentWorkloadPacket()

	// Valid bounds pass
	if err := ValidateFrozenWorkload(pkt); err != nil {
		t.Fatalf("expected valid bounds to pass, got: %v", err)
	}

	// Shared prefix below minimum (32768)
	pkt.WorkloadSpec.SharedPrefixTokens = 16384
	if err := ValidateFrozenWorkload(pkt); err == nil || !strings.Contains(err.Error(), "shared prefix tokens 16384 outside frozen range") {
		t.Fatalf("expected error for shared prefix below minimum, got: %v", err)
	}

	// Shared prefix above maximum (65536)
	pkt.WorkloadSpec.SharedPrefixTokens = 70000
	if err := ValidateFrozenWorkload(pkt); err == nil || !strings.Contains(err.Error(), "outside frozen range") {
		t.Fatalf("expected error for shared prefix above maximum, got: %v", err)
	}

	// Reset shared prefix, test private suffix below minimum (100)
	pkt.WorkloadSpec.SharedPrefixTokens = 32768
	pkt.WorkloadSpec.PrivateSuffixTokens = 50
	if err := ValidateFrozenWorkload(pkt); err == nil || !strings.Contains(err.Error(), "private suffix tokens 50 outside frozen range") {
		t.Fatalf("expected error for private suffix below minimum, got: %v", err)
	}

	// Private suffix above maximum (500)
	pkt.WorkloadSpec.PrivateSuffixTokens = 600
	if err := ValidateFrozenWorkload(pkt); err == nil || !strings.Contains(err.Error(), "outside frozen range") {
		t.Fatalf("expected error for private suffix above maximum, got: %v", err)
	}
}
