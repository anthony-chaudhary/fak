package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func TestWorkProfilesCanonicalizeAndRemainSafe(t *testing.T) {
	for _, intensity := range []string{"low", "medium", "high"} {
		got := DescribeWorkProfile("ponytail:" + intensity)
		if !got.Known || !got.Applied || got.Profile != "ponytail:native:"+intensity {
			t.Fatalf("ponytail:%s = %+v", intensity, got)
		}
		for _, required := range []string{"security", "correct", "tests", "evidence"} {
			if !strings.Contains(strings.ToLower(got.Segment), required) {
				t.Errorf("%s segment omits safety carve-out %q", intensity, required)
			}
		}
		if got.Witness == "" {
			t.Errorf("%s has no fragment digest", intensity)
		}
	}
}

func TestWorkProfilePonytailBypassesNoCodeChangeOnExplicitRequest(t *testing.T) {
	got := DescribeWorkProfile("ponytail:medium")
	want := "When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation."
	if !strings.Contains(got.Segment, want) {
		t.Fatalf("ponytail:medium missing explicit request bypass guidance: %q", got.Segment)
	}
}

func TestWorkProfileStandardAndUnknownAreOff(t *testing.T) {
	for _, name := range []string{"", "standard", "off"} {
		got := DescribeWorkProfile(name)
		if !got.Known || got.Applied || got.Profile != WorkProfileStandard || got.Segment != "" {
			t.Errorf("DescribeWorkProfile(%q) = %+v", name, got)
		}
	}
	for _, name := range []string{"ponytail:original:high", "ponytail:max", "caveman:native:high"} {
		got := DescribeWorkProfile(name)
		if got.Known || got.Applied || got.Profile != WorkProfileStandard || got.Segment != "" {
			t.Errorf("unsupported %q did not fail safe: %+v", name, got)
		}
	}
}

func TestWorkProfileIntensityIsMonotonicAndStable(t *testing.T) {
	low := DescribeWorkProfile("ponytail:low")
	medium := DescribeWorkProfile("ponytail:medium")
	high := DescribeWorkProfile("ponytail:high")
	if !(len(low.Segment) < len(medium.Segment) && len(medium.Segment) < len(high.Segment)) {
		t.Fatalf("intensity did not add guidance: low=%d medium=%d high=%d", len(low.Segment), len(medium.Segment), len(high.Segment))
	}
	again := DescribeWorkProfile("ponytail:high")
	if high.Segment != again.Segment || high.Witness != again.Witness {
		t.Fatalf("work profile is not byte stable: %+v vs %+v", high, again)
	}
}

func TestWorkProfileFromEnvDefaultsToPonytailMedium(t *testing.T) {
	for _, getenv := range []func(string) string{nil, func(string) string { return "" }} {
		got := WorkProfileFromEnv(getenv)
		if got.Profile != WorkProfilePonytailNativeMed || !got.Applied {
			t.Fatalf("WorkProfileFromEnv default = %+v", got)
		}
	}
}

func TestWorkProfileFromEnvCanExplicitlyDisableDefault(t *testing.T) {
	got := WorkProfileFromEnv(func(string) string { return "standard" })
	if got.Profile != WorkProfileStandard || got.Applied {
		t.Fatalf("WorkProfileFromEnv standard = %+v", got)
	}
}

func TestWorkProfileFromEnvReadsSeparateKnob(t *testing.T) {
	got := WorkProfileFromEnv(func(key string) string {
		if key != WorkProfileEnvVar {
			t.Fatalf("read %q, want %q", key, WorkProfileEnvVar)
		}
		return "ponytail:medium"
	})
	if got.Profile != WorkProfilePonytailNativeMed || !got.Applied {
		t.Fatalf("WorkProfileFromEnv = %+v", got)
	}
}

func TestHeadlessWorkProfilesCanonicalizeAndRemainSafe(t *testing.T) {
	for _, intensity := range []string{"low", "medium", "high"} {
		got := DescribeWorkProfile("ponytail:headless:" + intensity)
		if !got.Known || !got.Applied || got.Profile != "ponytail:headless:"+intensity {
			t.Fatalf("ponytail:headless:%s = %+v", intensity, got)
		}
		if got.Family != "ponytail" || got.Implementation != "headless" || got.Intensity != intensity {
			t.Fatalf("metadata mismatch for %s: family=%q impl=%q intensity=%q", intensity, got.Family, got.Implementation, got.Intensity)
		}
		if !strings.Contains(got.Segment, AutonomousActionBiasDirective) {
			t.Errorf("%s segment omits autonomous action bias directive: %q", intensity, got.Segment)
		}
		if !strings.Contains(got.Segment, AmbiguityResolutionDirective) {
			t.Errorf("%s segment omits ambiguity resolution directive: %q", intensity, got.Segment)
		}
		if !strings.Contains(got.Segment, TestBreadthCalibrationDirective) {
			t.Errorf("%s segment omits test breadth calibration directive: %q", intensity, got.Segment)
		}
		for _, required := range []string{
			"autonomous action-bias",
			"unattended mode is active",
			"never pause to ask clarifying or conversational questions",
			"prioritize minimal reversible action and execution",
			"ambiguity resolution",
			"when faced with ambiguity",
			"choose the smallest reversible action",
			"inspect/verify",
			"stalling, debating trade-offs endlessly, or halting",
			"test breadth calibration",
			"focus test authoring on concise, high-signal, atomic reproduction tests",
			"1-3 focused assertions proving the defect or contract",
			"sprawling, redundant, over-engineered test matrices",
			"burn output token limits or corrupt the working tree",
			"security", "correct", "tests", "evidence",
		} {
			if !strings.Contains(strings.ToLower(got.Segment), required) {
				t.Errorf("%s segment omits required directive/carve-out %q", intensity, required)
			}
		}
		if got.Witness == "" {
			t.Errorf("%s has no fragment digest", intensity)
		}
		expectedSum := sha256.Sum256([]byte(got.Segment))
		expectedWitness := "sha256:" + hex.EncodeToString(expectedSum[:])
		if got.Witness != expectedWitness {
			t.Errorf("%s witness mismatch: got %q, want %q", intensity, got.Witness, expectedWitness)
		}
	}
}

func TestHeadlessWorkProfileMediumActionBiasAndWitnessHash(t *testing.T) {
	got := DescribeWorkProfile("ponytail:headless:medium")
	if !got.Known || !got.Applied {
		t.Fatalf("ponytail:headless:medium failed to apply: %+v", got)
	}
	if got.Profile != "ponytail:headless:medium" {
		t.Fatalf("profile = %q, want ponytail:headless:medium", got.Profile)
	}

	// Verify unambiguous action-bias segment injection
	wantDirectives := []string{
		AutonomousActionBiasDirective,
		"Autonomous action-bias: unattended mode is active.",
		"Never pause to ask clarifying or conversational questions when running headless",
		"prioritize minimal reversible action and execution.",
		AmbiguityResolutionDirective,
		"Ambiguity resolution: when faced with ambiguity",
		"choose the smallest reversible action, inspect/verify",
		"rather than stalling, debating trade-offs endlessly, or halting.",
		TestBreadthCalibrationDirective,
		"Test breadth calibration: focus test authoring on concise, high-signal, atomic reproduction tests",
		"1-3 focused assertions proving the defect or contract",
		"rather than generating sprawling, redundant, over-engineered test matrices",
		"burn output token limits or corrupt the working tree.",
		"When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.",
	}
	for _, want := range wantDirectives {
		if !strings.Contains(got.Segment, want) {
			t.Errorf("missing directive substring %q in segment:\n%s", want, got.Segment)
		}
	}

	// Verify cryptographic witness hash
	sum := sha256.Sum256([]byte(got.Segment))
	expectedWitness := "sha256:" + hex.EncodeToString(sum[:])
	const pinnedWitness = "sha256:e18fcaef925476ced4a2533ae2718a0895d4ddb092e3dda39c7e11ed319d6535"
	if got.Witness != expectedWitness || got.Witness != pinnedWitness {
		t.Fatalf("witness = %q, want %q (pinned: %q)", got.Witness, expectedWitness, pinnedWitness)
	}
	if !strings.HasPrefix(got.Witness, "sha256:") || len(got.Witness) != 71 {
		t.Fatalf("witness format invalid: %q", got.Witness)
	}
}

func TestAmbiguityResolutionDirectiveAndWitnessHash(t *testing.T) {
	if AmbiguityResolutionDirective == "" {
		t.Fatal("AmbiguityResolutionDirective is empty")
	}
	if AmbiguityResolutionDirective != AmbiguityResolutionRule {
		t.Fatalf("AmbiguityResolutionRule alias mismatch: %q != %q", AmbiguityResolutionRule, AmbiguityResolutionDirective)
	}
	if AmbiguityResolutionDirective != AmbiguityResolutionContract {
		t.Fatalf("AmbiguityResolutionContract alias mismatch: %q != %q", AmbiguityResolutionContract, AmbiguityResolutionDirective)
	}

	for _, intensity := range []string{"low", "medium", "high"} {
		profileName := "ponytail:headless:" + intensity
		got := DescribeWorkProfile(profileName)
		if !got.Known || !got.Applied {
			t.Fatalf("%s failed to resolve: %+v", profileName, got)
		}
		if !strings.Contains(got.Segment, AmbiguityResolutionDirective) {
			t.Errorf("%s missing AmbiguityResolutionDirective: %q", profileName, got.Segment)
		}
		expectedSum := sha256.Sum256([]byte(got.Segment))
		expectedWitness := "sha256:" + hex.EncodeToString(expectedSum[:])
		if got.Witness != expectedWitness {
			t.Errorf("%s witness mismatch: got %q, want %q", profileName, got.Witness, expectedWitness)
		}
	}

	// Explicitly verify pinned witness for ponytail:headless:medium
	gotMed := DescribeWorkProfile("ponytail:headless:medium")
	const expectedMediumWitness = "sha256:e18fcaef925476ced4a2533ae2718a0895d4ddb092e3dda39c7e11ed319d6535"
	if gotMed.Witness != expectedMediumWitness {
		t.Fatalf("ponytail:headless:medium witness = %q, want pinned %q", gotMed.Witness, expectedMediumWitness)
	}
}

func TestTestBreadthCalibrationDirectiveAndWitnessHash(t *testing.T) {
	if TestBreadthCalibrationDirective == "" {
		t.Fatal("TestBreadthCalibrationDirective is empty")
	}
	if TestBreadthCalibrationDirective != TestBreadthCalibrationRule {
		t.Fatalf("TestBreadthCalibrationRule alias mismatch: %q != %q", TestBreadthCalibrationRule, TestBreadthCalibrationDirective)
	}
	if TestBreadthCalibrationDirective != TestBreadthCalibrationContract {
		t.Fatalf("TestBreadthCalibrationContract alias mismatch: %q != %q", TestBreadthCalibrationContract, TestBreadthCalibrationDirective)
	}

	for _, intensity := range []string{"low", "medium", "high"} {
		profileName := "ponytail:headless:" + intensity
		got := DescribeWorkProfile(profileName)
		if !got.Known || !got.Applied {
			t.Fatalf("%s failed to resolve: %+v", profileName, got)
		}
		if !strings.Contains(got.Segment, TestBreadthCalibrationDirective) {
			t.Errorf("%s missing TestBreadthCalibrationDirective: %q", profileName, got.Segment)
		}
		expectedSum := sha256.Sum256([]byte(got.Segment))
		expectedWitness := "sha256:" + hex.EncodeToString(expectedSum[:])
		if got.Witness != expectedWitness {
			t.Errorf("%s witness mismatch: got %q, want %q", profileName, got.Witness, expectedWitness)
		}
	}

	// Explicitly verify pinned witness for ponytail:headless:medium
	gotMed := DescribeWorkProfile("ponytail:headless:medium")
	const expectedMediumWitness = "sha256:e18fcaef925476ced4a2533ae2718a0895d4ddb092e3dda39c7e11ed319d6535"
	if gotMed.Witness != expectedMediumWitness {
		t.Fatalf("ponytail:headless:medium witness = %q, want pinned %q", gotMed.Witness, expectedMediumWitness)
	}
}

func TestHeadlessWorkProfileIntensityIsMonotonicAndStable(t *testing.T) {
	low := DescribeWorkProfile("ponytail:headless:low")
	medium := DescribeWorkProfile("ponytail:headless:medium")
	high := DescribeWorkProfile("ponytail:headless:high")
	if !(len(low.Segment) < len(medium.Segment) && len(medium.Segment) < len(high.Segment)) {
		t.Fatalf("headless intensity did not add guidance monotonically: low=%d medium=%d high=%d", len(low.Segment), len(medium.Segment), len(high.Segment))
	}
	again := DescribeWorkProfile("ponytail:headless:medium")
	if medium.Segment != again.Segment || medium.Witness != again.Witness {
		t.Fatalf("headless work profile is not byte stable: %+v vs %+v", medium, again)
	}
}

func TestHeadlessWorkProfileAliases(t *testing.T) {
	canonical := DescribeWorkProfile("ponytail:headless:medium")
	for _, alias := range []string{"headless:medium", "headless", "ponytail:headless"} {
		got := DescribeWorkProfile(alias)
		if got.Profile != canonical.Profile || got.Segment != canonical.Segment || got.Witness != canonical.Witness {
			t.Errorf("alias %q did not resolve to canonical ponytail:headless:medium: got %+v", alias, got)
		}
	}
	low := DescribeWorkProfile("ponytail:headless:low")
	if got := DescribeWorkProfile("headless:low"); got.Profile != low.Profile || got.Witness != low.Witness {
		t.Errorf("headless:low alias did not match canonical: got %+v, want %+v", got, low)
	}
	high := DescribeWorkProfile("ponytail:headless:high")
	if got := DescribeWorkProfile("headless:high"); got.Profile != high.Profile || got.Witness != high.Witness {
		t.Errorf("headless:high alias did not match canonical: got %+v, want %+v", got, high)
	}
}
