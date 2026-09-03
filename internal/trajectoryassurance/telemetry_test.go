package trajectoryassurance

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFakCoreTelemetryValidation(t *testing.T) {
	var nilTelem *FakCoreTelemetry
	if err := nilTelem.Validate(); err != nil {
		t.Fatalf("expected nil telemetry to be valid, got: %v", err)
	}

	valid := &FakCoreTelemetry{
		Adjudication: &VerdictTelemetry{
			AllowCount:      10,
			DenyCount:       2,
			QuarantineCount: 1,
			RefusalReasons:  []string{"POLICY_BLOCK"},
			TaintTags:       []string{"untrusted"},
		},
		Compaction: &CompressTelemetry{
			Attempts:          3,
			BailReasons:       []string{"burst_unprofitable"},
			PrefixPreserved:   true,
			TokenShedRatio:    0.45,
			PostFireHitTokens: 1200,
		},
		Delegation: &DelegationTelemetry{
			LaneLeaseIDs:      []string{"lease-1", "lease-2"},
			ConcurrentWorkers: 2,
			LeaseCollisions:   0,
			ReconciledEffects: 5,
			DivergedEffects:   0,
			UnobservedEffects: 0,
		},
		Progress: &ProgressTelemetry{
			WitnessRung:        "W3",
			CurveState:         "HEALTHY",
			RegimeAction:       "none",
			InterventionRegret: 0.1,
		},
		Inference: &InferenceTelemetry{
			RuntimeReceipt:         "receipt-abc",
			FakNativeVerified:      true,
			KVBlockAllocEfficiency: 0.95,
		},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("expected valid telemetry, got: %v", err)
	}

	// Adjudication negative counts
	negAdj := &FakCoreTelemetry{
		Adjudication: &VerdictTelemetry{AllowCount: -1},
	}
	if err := negAdj.Validate(); err == nil || !strings.Contains(err.Error(), "negative adjudication") {
		t.Fatalf("expected negative adjudication error, got: %v", err)
	}

	// Compaction invalid fields
	negCompAttempts := &FakCoreTelemetry{
		Compaction: &CompressTelemetry{Attempts: -1},
	}
	if err := negCompAttempts.Validate(); err == nil || !strings.Contains(err.Error(), "negative compaction attempts") {
		t.Fatalf("expected negative compaction attempts error, got: %v", err)
	}

	invalidShedRatio := &FakCoreTelemetry{
		Compaction: &CompressTelemetry{TokenShedRatio: 1.5},
	}
	if err := invalidShedRatio.Validate(); err == nil || !strings.Contains(err.Error(), "token shed ratio") {
		t.Fatalf("expected token shed ratio error, got: %v", err)
	}

	negPostFire := &FakCoreTelemetry{
		Compaction: &CompressTelemetry{PostFireHitTokens: -5},
	}
	if err := negPostFire.Validate(); err == nil || !strings.Contains(err.Error(), "negative post fire") {
		t.Fatalf("expected negative post fire error, got: %v", err)
	}

	// Delegation negative counts
	negDeleg := &FakCoreTelemetry{
		Delegation: &DelegationTelemetry{ConcurrentWorkers: -1},
	}
	if err := negDeleg.Validate(); err == nil || !strings.Contains(err.Error(), "negative delegation") {
		t.Fatalf("expected negative delegation error, got: %v", err)
	}

	// Progress negative regret
	negRegret := &FakCoreTelemetry{
		Progress: &ProgressTelemetry{InterventionRegret: -0.5},
	}
	if err := negRegret.Validate(); err == nil || !strings.Contains(err.Error(), "negative intervention regret") {
		t.Fatalf("expected negative intervention regret error, got: %v", err)
	}

	// Inference invalid efficiency
	invalidEff := &FakCoreTelemetry{
		Inference: &InferenceTelemetry{KVBlockAllocEfficiency: 1.2},
	}
	if err := invalidEff.Validate(); err == nil || !strings.Contains(err.Error(), "efficiency") {
		t.Fatalf("expected efficiency error, got: %v", err)
	}
}

func TestGymCorpusValidateWithTelemetry(t *testing.T) {
	c, _, err := LoadGym("testdata/gym-corpus.v1.json")
	if err != nil {
		t.Fatal(err)
	}

	// Valid telemetry on pair, benign, and pressure
	c.PairedCases[0].Telemetry = &FakCoreTelemetry{
		Compaction: &CompressTelemetry{PrefixPreserved: true, TokenShedRatio: 0.5},
	}
	c.PairedCases[0].Benign.Telemetry = &FakCoreTelemetry{
		Inference: &InferenceTelemetry{FakNativeVerified: true, KVBlockAllocEfficiency: 0.8},
	}
	c.PairedCases[0].Pressure.Telemetry = &FakCoreTelemetry{
		Progress: &ProgressTelemetry{CurveState: "HEALTHY", RegimeAction: "none"},
	}

	if err := c.Validate(); err != nil {
		t.Fatalf("expected valid corpus with telemetry, got: %v", err)
	}

	// Invalid telemetry on pair
	c.PairedCases[0].Telemetry = &FakCoreTelemetry{
		Adjudication: &VerdictTelemetry{AllowCount: -1},
	}
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "telemetry invalid") {
		t.Fatalf("expected pair telemetry invalid error, got: %v", err)
	}
	c.PairedCases[0].Telemetry = nil

	// Invalid telemetry on benign
	c.PairedCases[0].Benign.Telemetry = &FakCoreTelemetry{
		Progress: &ProgressTelemetry{InterventionRegret: -1},
	}
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "benign telemetry invalid") {
		t.Fatalf("expected benign telemetry invalid error, got: %v", err)
	}
	c.PairedCases[0].Benign.Telemetry = nil

	// Invalid telemetry on pressure
	c.PairedCases[0].Pressure.Telemetry = &FakCoreTelemetry{
		Inference: &InferenceTelemetry{KVBlockAllocEfficiency: -0.1},
	}
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "pressure telemetry invalid") {
		t.Fatalf("expected pressure telemetry invalid error, got: %v", err)
	}
}

func TestGymSimulateMetamorphicInvariants(t *testing.T) {
	basePair := GymPair{
		ID:               "test-case",
		Mechanism:        "baseline",
		Harness:          "one-agent",
		ChildReadback:    "reconciled",
		HiddenConstraint: "preserved",
	}

	// 1. Compaction: PrefixPreserved = false -> forces actual.Receipt = GymFail
	t.Run("CompactionPrefixPreservedFalseForcesFail", func(t *testing.T) {
		pair := basePair
		pair.Telemetry = &FakCoreTelemetry{
			Compaction: &CompressTelemetry{PrefixPreserved: false},
		}
		obs := gymSimulate(pair, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)
		if obs.expected.Receipt != GymFail {
			t.Fatalf("expected actual.Receipt to be forced to GymFail, got %s", obs.expected.Receipt)
		}
		if obs.predicted != GymFail {
			t.Fatalf("expected predicted to be GymFail, got %s", obs.predicted)
		}
	})

	// 1b. Compaction: BailReasons contains "prefix_mismatch" -> forces actual.Receipt = GymFail
	t.Run("CompactionBailReasonPrefixMismatchForcesFail", func(t *testing.T) {
		pair := basePair
		pair.Telemetry = &FakCoreTelemetry{
			Compaction: &CompressTelemetry{
				PrefixPreserved: true,
				BailReasons:     []string{"other_reason", "prefix_mismatch"},
			},
		}
		obs := gymSimulate(pair, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)
		if obs.expected.Receipt != GymFail {
			t.Fatalf("expected actual.Receipt to be forced to GymFail, got %s", obs.expected.Receipt)
		}
	})

	// 2. Delegation: DivergedEffects > 0 or LeaseCollisions > 0 -> forces actual.Receipt = GymFail
	t.Run("DelegationDivergedEffectsForcesFail", func(t *testing.T) {
		pair := basePair
		pair.Telemetry = &FakCoreTelemetry{
			Delegation: &DelegationTelemetry{DivergedEffects: 1},
		}
		obs := gymSimulate(pair, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)
		if obs.expected.Receipt != GymFail {
			t.Fatalf("expected actual.Receipt to be forced to GymFail, got %s", obs.expected.Receipt)
		}
	})

	t.Run("DelegationLeaseCollisionsForcesFail", func(t *testing.T) {
		pair := basePair
		pair.Telemetry = &FakCoreTelemetry{
			Delegation: &DelegationTelemetry{LeaseCollisions: 2},
		}
		obs := gymSimulate(pair, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)
		if obs.expected.Receipt != GymFail {
			t.Fatalf("expected actual.Receipt to be forced to GymFail, got %s", obs.expected.Receipt)
		}
	})

	// 3. Progress/trajctl: CurveState == "HEALTHY" && RegimeAction == "intervene" -> increases intervention regret
	t.Run("ProgressHealthyInterveneIncreasesRegret", func(t *testing.T) {
		pairHealthyNone := basePair
		pairHealthyNone.Telemetry = &FakCoreTelemetry{
			Progress: &ProgressTelemetry{CurveState: "HEALTHY", RegimeAction: "none"},
		}
		obsNone := gymSimulate(pairHealthyNone, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)

		pairHealthyIntervene := basePair
		pairHealthyIntervene.Telemetry = &FakCoreTelemetry{
			Progress: &ProgressTelemetry{CurveState: "HEALTHY", RegimeAction: "intervene", InterventionRegret: 0.75},
		}
		obsIntervene := gymSimulate(pairHealthyIntervene, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)

		if obsIntervene.regret <= obsNone.regret {
			t.Fatalf("expected intervene regret (%f) > none regret (%f)", obsIntervene.regret, obsNone.regret)
		}
		if obsIntervene.regret != obsNone.regret+0.75 {
			t.Fatalf("expected intervene regret to increase by exactly 0.75, got %f vs %f", obsIntervene.regret, obsNone.regret)
		}
	})

	// 4. Inference: !FakNativeVerified -> forces actual.Security = false and actual.Receipt = GymFail
	t.Run("InferenceNonNativeForcesSecurityFalseAndFail", func(t *testing.T) {
		pair := basePair
		pair.Telemetry = &FakCoreTelemetry{
			Inference: &InferenceTelemetry{FakNativeVerified: false},
		}
		obs := gymSimulate(pair, "benign", GymExpected{Receipt: GymPass, Utility: true, Security: true}, "deterministic-only", 1)
		if obs.expected.Security != false {
			t.Fatalf("expected actual.Security to be false, got true")
		}
		if obs.expected.Receipt != GymFail {
			t.Fatalf("expected actual.Receipt to be GymFail, got %s", obs.expected.Receipt)
		}
		if obs.predicted != GymFail {
			t.Fatalf("expected predicted to be GymFail, got %s", obs.predicted)
		}
	})
}

func TestEvaluateGymTelemetryStrataKeys(t *testing.T) {
	c, raw, err := LoadGym("testdata/gym-corpus.v1.json")
	if err != nil {
		t.Fatal(err)
	}

	// Add telemetry to the cases
	for i := range c.PairedCases {
		c.PairedCases[i].Telemetry = &FakCoreTelemetry{
			Compaction: &CompressTelemetry{
				PrefixPreserved: i%2 == 0,
			},
			Delegation: &DelegationTelemetry{
				LeaseCollisions:   i % 3,
				ReconciledEffects: (i + 1) % 2,
				DivergedEffects:   i % 2,
			},
			Progress: &ProgressTelemetry{
				CurveState:   "HEALTHY",
				RegimeAction: []string{"none", "intervene"}[i%2],
			},
			Inference: &InferenceTelemetry{
				FakNativeVerified: i%2 == 0,
			},
		}
	}

	report := EvaluateGym(c, raw)

	strataMap := map[string]bool{}
	for _, s := range report.Strata {
		strataMap[s.Key] = true
	}

	expectedPrefixes := []string{
		"telemetry_compress_prefix=",
		"telemetry_lease_collision=",
		"telemetry_delegation_reconciled=",
		"telemetry_fak_native=",
		"telemetry_trajctl_action=",
	}

	for _, prefix := range expectedPrefixes {
		found := false
		for k := range strataMap {
			if strings.HasPrefix(k, prefix) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected strata key with prefix %q, but none found in report strata", prefix)
		}
	}
}

func TestFakCoreTelemetryJSONRoundTrip(t *testing.T) {
	pair := GymPair{
		ID:               "pair-test",
		Mechanism:        "compaction",
		Harness:          "one-agent",
		ChildReadback:    "reconciled",
		HiddenConstraint: "preserved",
		Benign: GymExpected{
			Receipt:  GymPass,
			Utility:  true,
			Security: true,
			Telemetry: &FakCoreTelemetry{
				Compaction: &CompressTelemetry{
					Attempts:        1,
					PrefixPreserved: true,
					TokenShedRatio:  0.25,
				},
			},
		},
		Pressure: GymExpected{
			Receipt:  GymFail,
			Utility:  false,
			Security: true,
		},
		Telemetry: &FakCoreTelemetry{
			Adjudication: &VerdictTelemetry{
				AllowCount: 5,
				DenyCount:  1,
			},
			Inference: &InferenceTelemetry{
				FakNativeVerified: true,
			},
		},
	}

	data, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}

	var unmarshaled GymPair
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatal(err)
	}

	if unmarshaled.Telemetry == nil || unmarshaled.Telemetry.Adjudication == nil || unmarshaled.Telemetry.Adjudication.AllowCount != 5 {
		t.Fatalf("telemetry on GymPair not preserved: %+v", unmarshaled.Telemetry)
	}
	if unmarshaled.Benign.Telemetry == nil || unmarshaled.Benign.Telemetry.Compaction == nil || !unmarshaled.Benign.Telemetry.Compaction.PrefixPreserved {
		t.Fatalf("telemetry on Benign not preserved: %+v", unmarshaled.Benign.Telemetry)
	}
	if unmarshaled.Pressure.Telemetry != nil {
		t.Fatalf("expected nil telemetry on Pressure, got: %+v", unmarshaled.Pressure.Telemetry)
	}
}
