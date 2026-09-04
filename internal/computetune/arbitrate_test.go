package computetune

import (
	"testing"
)

func TestRecomputeArbitratorStrix(t *testing.T) {
	profile := DefaultStrixHaloProfile()
	if err := profile.Validate(); err != nil {
		t.Fatalf("DefaultStrixHaloProfile validation failed: %v", err)
	}

	arbitrator := NewStorageComputeArbitrator(profile)

	// Scenario 1: Short prompt / low reuse (1,000 tokens with 1 expected reuse).
	// Recompute should be favored because flash wear and SSD write overhead outweigh
	// recomputing a modest number of tokens in high-bandwidth unified memory.
	dec1 := arbitrator.Decide(1000, 1.0)
	if dec1.Verdict != VerdictRecompute {
		t.Fatalf("expected VerdictRecompute for 1,000 tokens with 1 expected reuse, got %v (offload=%.4f, recompute=%.4f)",
			dec1.Verdict, dec1.OffloadCost, dec1.RecomputeCost)
	}
	if dec1.OffloadCost < dec1.RecomputeCost {
		t.Fatalf("expected OffloadCost (%.4f) >= RecomputeCost (%.4f) for VerdictRecompute",
			dec1.OffloadCost, dec1.RecomputeCost)
	}
	if dec1.Reason == "" {
		t.Fatal("expected non-empty reason for decision")
	}

	// Scenario 2: Deep-context prefix with high reuse (32,000 tokens with 5 expected reuses).
	// Offload should be favored because recomputing 32k tokens across 5 passes is
	// computationally prohibitive compared to reading the cached KV from NVMe once per reuse.
	dec2 := arbitrator.Decide(32000, 5.0)
	if dec2.Verdict != VerdictOffload {
		t.Fatalf("expected VerdictOffload for 32,000 tokens with 5 expected reuses, got %v (offload=%.4f, recompute=%.4f)",
			dec2.Verdict, dec2.OffloadCost, dec2.RecomputeCost)
	}
	if dec2.OffloadCost >= dec2.RecomputeCost {
		t.Fatalf("expected OffloadCost (%.4f) < RecomputeCost (%.4f) for VerdictOffload",
			dec2.OffloadCost, dec2.RecomputeCost)
	}
	if dec2.Reason == "" {
		t.Fatal("expected non-empty reason for decision")
	}
}

func TestRecomputeArbitratorEdgeCases(t *testing.T) {
	profile := DefaultStrixHaloProfile()
	arbitrator := NewStorageComputeArbitrator(profile)

	tests := []struct {
		name        string
		tokens      int
		reuse       float64
		wantVerdict ArbitratorVerdict
		wantZero    bool
	}{
		{
			name:        "zero tokens",
			tokens:      0,
			reuse:       1.0,
			wantVerdict: VerdictRecompute,
			wantZero:    true,
		},
		{
			name:        "negative tokens",
			tokens:      -500,
			reuse:       2.0,
			wantVerdict: VerdictRecompute,
			wantZero:    true,
		},
		{
			name:        "zero reuse",
			tokens:      4000,
			reuse:       0.0,
			wantVerdict: VerdictRecompute,
			wantZero:    false,
		},
		{
			name:        "negative reuse",
			tokens:      8000,
			reuse:       -1.5,
			wantVerdict: VerdictRecompute,
			wantZero:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dec := arbitrator.Decide(tt.tokens, tt.reuse)
			if dec.Verdict != tt.wantVerdict {
				t.Fatalf("got verdict %v, want %v", dec.Verdict, tt.wantVerdict)
			}
			if tt.wantZero {
				if dec.RecomputeCost != 0 || dec.OffloadCost != 0 {
					t.Fatalf("expected zero costs, got recompute=%.4f, offload=%.4f",
						dec.RecomputeCost, dec.OffloadCost)
				}
			} else if tt.reuse <= 0 {
				if dec.RecomputeCost != 0 {
					t.Fatalf("expected zero recompute cost for reuse <= 0, got %.4f", dec.RecomputeCost)
				}
				if dec.OffloadCost <= 0 {
					t.Fatalf("expected positive offload cost (write+wear), got %.4f", dec.OffloadCost)
				}
			}
		})
	}
}

func TestRecomputeArbitratorParametricSweep(t *testing.T) {
	profile := DefaultStrixHaloProfile()
	arbitrator := NewStorageComputeArbitrator(profile)

	tokenCounts := []int{500, 1000, 2000, 4000, 8000, 16000, 32000}
	reuseMultipliers := []float64{0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0}

	for _, tokens := range tokenCounts {
		var prevAdvantage float64 // RecomputeCost - OffloadCost (higher means offload is more advantageous)
		for i, reuse := range reuseMultipliers {
			dec := arbitrator.Decide(tokens, reuse)
			advantage := dec.RecomputeCost - dec.OffloadCost

			if i > 0 && advantage <= prevAdvantage {
				t.Fatalf("expected offload advantage to monotonically increase with reuse for %d tokens (at reuse=%.1f: %.4f <= %.4f)",
					tokens, reuse, advantage, prevAdvantage)
			}
			prevAdvantage = advantage

			// Consistency check: verdict must match cost inequality
			if dec.OffloadCost < dec.RecomputeCost && dec.Verdict != VerdictOffload {
				t.Fatalf("verdict %v inconsistent with offload < recompute (%.4f < %.4f)",
					dec.Verdict, dec.OffloadCost, dec.RecomputeCost)
			}
			if dec.OffloadCost >= dec.RecomputeCost && dec.Verdict != VerdictRecompute {
				t.Fatalf("verdict %v inconsistent with offload >= recompute (%.4f >= %.4f)",
					dec.Verdict, dec.OffloadCost, dec.RecomputeCost)
			}
		}
	}
}

func TestRecomputeArbitratorProfiles(t *testing.T) {
	halo := DefaultStrixHaloProfile()
	point := DefaultStrixPointProfile()

	if err := halo.Validate(); err != nil {
		t.Fatalf("halo validation: %v", err)
	}
	if err := point.Validate(); err != nil {
		t.Fatalf("point validation: %v", err)
	}

	haloArb := NewStorageComputeArbitrator(halo)
	pointArb := NewStorageComputeArbitrator(point)

	// Since Strix Point has lower prefill tok/s (160 vs 300), recomputing is
	// relatively more expensive on Point, so it should favor offload earlier.
	tokens := 16000
	reuse := 3.0
	haloDec := haloArb.Decide(tokens, reuse)
	pointDec := pointArb.Decide(tokens, reuse)

	if pointDec.RecomputeCost <= haloDec.RecomputeCost {
		t.Fatalf("expected Strix Point recompute cost (%.4f) > Strix Halo recompute cost (%.4f)",
			pointDec.RecomputeCost, haloDec.RecomputeCost)
	}
}

func TestPlatformProfileValidation(t *testing.T) {
	tests := []struct {
		name    string
		profile PlatformProfile
		wantErr bool
	}{
		{
			name:    "valid halo profile",
			profile: DefaultStrixHaloProfile(),
			wantErr: false,
		},
		{
			name: "missing name",
			profile: PlatformProfile{
				UMAPrefillTokS: 300,
				NVMeWriteGBps:  1.5,
				NVMeReadGBps:   3.5,
				BytesPerToken:  1024,
			},
			wantErr: true,
		},
		{
			name: "invalid prefill tok/s",
			profile: PlatformProfile{
				Name:           "BadPrefill",
				UMAPrefillTokS: 0,
				NVMeWriteGBps:  1.5,
				NVMeReadGBps:   3.5,
				BytesPerToken:  1024,
			},
			wantErr: true,
		},
		{
			name: "invalid NVMe write BW",
			profile: PlatformProfile{
				Name:           "BadWrite",
				UMAPrefillTokS: 300,
				NVMeWriteGBps:  0,
				NVMeReadGBps:   3.5,
				BytesPerToken:  1024,
			},
			wantErr: true,
		},
		{
			name: "invalid NVMe read BW",
			profile: PlatformProfile{
				Name:           "BadRead",
				UMAPrefillTokS: 300,
				NVMeWriteGBps:  1.5,
				NVMeReadGBps:   -1,
				BytesPerToken:  1024,
			},
			wantErr: true,
		},
		{
			name: "invalid bytes per token",
			profile: PlatformProfile{
				Name:           "BadBytes",
				UMAPrefillTokS: 300,
				NVMeWriteGBps:  1.5,
				NVMeReadGBps:   3.5,
				BytesPerToken:  0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.profile.Validate()
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestStorageComputeArbitratorDecideConsistency(t *testing.T) {
	profile := DefaultStrixHaloProfile()
	arbitrator := NewStorageComputeArbitrator(profile)

	if arbitrator.Profile().Name != profile.Name {
		t.Fatalf("expected profile name %s, got %s", profile.Name, arbitrator.Profile().Name)
	}

	methodDec := arbitrator.Decide(10000, 3.5)
	funcDec := Decide(10000, 3.5, profile)

	if methodDec.Verdict != funcDec.Verdict {
		t.Fatalf("method verdict %v != func verdict %v", methodDec.Verdict, funcDec.Verdict)
	}
	if methodDec.RecomputeCost != funcDec.RecomputeCost {
		t.Fatalf("method recompute %.4f != func recompute %.4f", methodDec.RecomputeCost, funcDec.RecomputeCost)
	}
	if methodDec.OffloadCost != funcDec.OffloadCost {
		t.Fatalf("method offload %.4f != func offload %.4f", methodDec.OffloadCost, funcDec.OffloadCost)
	}
}
