package ablate

import "testing"

// TestChildEnvHonorsReaderArmDialects is the behavioral witness for #6362:
// each arm must speak the value vocabulary consumed by its production reader.
func TestChildEnvHonorsReaderArmDialects(t *testing.T) {
	tests := []struct {
		feature string
		off     string
		on      string
	}{
		{FeatureNormgate, "FAK_NORMGATE=off", "FAK_NORMGATE="},
		{FeatureRadix, "FAK_INKERNEL_RADIX=off", "FAK_INKERNEL_RADIX="},
		{FeatureCompressor, "FAK_COMPRESSOR=off", "FAK_COMPRESSOR="},
		{FeatureIFC, "FAK_IFC=off", "FAK_IFC="},
		{FeatureGitgate, "FAK_GITGATE=off", "FAK_GITGATE="},
		{FeatureCtxplanSeam, "FAK_CTXPLAN_SEAM=off", "FAK_CTXPLAN_SEAM=on"},
		{FeatureWireScreen, "FAK_WIRE_SCREEN=", "FAK_WIRE_SCREEN=heuristic"},
		{FeatureWireRedact, "FAK_WIRE_REDACT=", "FAK_WIRE_REDACT=pii"},
	}
	for _, tt := range tests {
		t.Run(tt.feature, func(t *testing.T) {
			off := FeatureConfig{EnvFeatures: map[string]string{tt.feature: "off"}}.childEnv()
			if len(off) != 1 || off[0] != tt.off {
				t.Fatalf("OFF arm child env = %v, want [%s]", off, tt.off)
			}
			on := FeatureConfig{EnvFeatures: map[string]string{tt.feature: "on"}}.childEnv()
			if len(on) != 1 || on[0] != tt.on {
				t.Fatalf("ON arm child env = %v, want [%s]", on, tt.on)
			}
		})
	}
}

// TestEveryEnvConceptArmContractFlips is the registry-wide conformance witness.
// A newly registered env concept cannot silently fall back to a guessed dialect.
func TestEveryEnvConceptArmContractFlips(t *testing.T) {
	for _, feature := range KnownFeatures() {
		concept, ok := registeredConcept(feature)
		if !ok || concept.EnvVar == "" {
			continue
		}
		if concept.EnvArms == nil || concept.EnvArms.Enabled == nil {
			t.Fatalf("env concept %q has no arm contract", feature)
		}
		if concept.EnvArms.Enabled(concept.EnvArms.Off) {
			t.Errorf("%s OFF value %q enables its reader", feature, concept.EnvArms.Off)
		}
		if !concept.EnvArms.Enabled(concept.EnvArms.On) {
			t.Errorf("%s ON value %q disables its reader", feature, concept.EnvArms.On)
		}
	}
}

func TestRegisterRejectsInvalidEnvArmContract(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Register accepted env arm values that do not flip the reader")
		}
	}()
	Register(Concept{
		Token:   "invalid_arm_contract_test",
		EnvVar:  "FAK_INVALID_ARM_CONTRACT_TEST",
		EnvArms: &EnvArmContract{On: "1", Off: "0", Enabled: func(string) bool { return true }},
	})
}
