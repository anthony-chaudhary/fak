package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
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

func TestWorkProfilePonytailHeadlessAutonomousActionBias(t *testing.T) {
	for _, name := range []string{"ponytail:headless", "ponytail:headless:med", "ponytail:headless:medium"} {
		got := DescribeWorkProfile(name)
		if !got.Known || !got.Applied {
			t.Fatalf("DescribeWorkProfile(%q) not applied: %+v", name, got)
		}
		if got.Profile != WorkProfilePonytailHeadlessMed {
			t.Fatalf("DescribeWorkProfile(%q).Profile = %q, want %q", name, got.Profile, WorkProfilePonytailHeadlessMed)
		}
		if got.Family != "ponytail" || got.Implementation != "headless" || got.Intensity != "medium" {
			t.Fatalf("DescribeWorkProfile(%q) unexpected readout fields: %+v", name, got)
		}
		wantDirective := AutonomousActionBiasDirective
		if !strings.Contains(got.Segment, wantDirective) {
			t.Fatalf("DescribeWorkProfile(%q) missing autonomous action bias directive %q: %s", name, wantDirective, got.Segment)
		}
		expectedSum := sha256.Sum256([]byte(got.Segment))
		expectedWitness := "sha256:" + hex.EncodeToString(expectedSum[:])
		if got.Witness != expectedWitness {
			t.Fatalf("DescribeWorkProfile(%q) witness mismatch: got %q, want %q", name, got.Witness, expectedWitness)
		}
	}
}

func TestWorkProfileTestBreadthCalibration(t *testing.T) {
	for _, name := range []string{
		"ponytail:medium",
		WorkProfilePonytailHeadlessMed,
	} {
		got := DescribeWorkProfile(name)
		for _, phrase := range []string{
			"single atomic reproduction tests",
			"prohibit sprawling 20-test suites when 1 witness suffices",
			"prevent token exhaustion",
		} {
			if !strings.Contains(got.Segment, phrase) {
				t.Fatalf("profile %q missing test breadth calibration phrase %q: %q", name, phrase, got.Segment)
			}
		}
	}
}

func TestWorkProfileAmbiguityResolutionLadder(t *testing.T) {
	want := AmbiguityResolutionDirective
	for _, profile := range []string{
		"ponytail:low", "ponytail:medium", "ponytail:high",
		WorkProfilePonytailHeadlessLow, WorkProfilePonytailHeadlessMed, WorkProfilePonytailHeadlessHigh,
	} {
		got := DescribeWorkProfile(profile)
		if !got.Known || !got.Applied {
			t.Fatalf("DescribeWorkProfile(%q) not applied: %+v", profile, got)
		}
		if !strings.Contains(got.Segment, want) {
			t.Fatalf("DescribeWorkProfile(%q) missing ambiguity resolution fragment %q: %s", profile, want, got.Segment)
		}
	}
}

func TestWorkProfileHeadlessIntensityIsMonotonicAndStable(t *testing.T) {
	low := DescribeWorkProfile(WorkProfilePonytailHeadlessLow)
	med := DescribeWorkProfile(WorkProfilePonytailHeadlessMed)
	high := DescribeWorkProfile(WorkProfilePonytailHeadlessHigh)
	if !low.Known || !med.Known || !high.Known {
		t.Fatalf("headless profiles not recognized: low=%v med=%v high=%v", low.Known, med.Known, high.Known)
	}
	if !(len(low.Segment) < len(med.Segment) && len(med.Segment) < len(high.Segment)) {
		t.Fatalf("headless intensity did not add guidance: low=%d med=%d high=%d", len(low.Segment), len(med.Segment), len(high.Segment))
	}
	for _, prof := range []WorkProfileReadout{low, med, high} {
		for _, required := range []string{"security", "correct", "tests", "evidence"} {
			if !strings.Contains(strings.ToLower(prof.Segment), required) {
				t.Errorf("%s segment omits safety carve-out %q", prof.Profile, required)
			}
		}
	}
	again := DescribeWorkProfile(WorkProfilePonytailHeadlessMed)
	if med.Segment != again.Segment || med.Witness != again.Witness {
		t.Fatalf("headless work profile is not byte stable: %+v vs %+v", med, again)
	}
}

func TestWorkProfileNamesIncludesHeadless(t *testing.T) {
	names := WorkProfileNames()
	for _, want := range []string{
		"ponytail:headless",
		WorkProfilePonytailHeadlessLow,
		WorkProfilePonytailHeadlessMed,
		WorkProfilePonytailHeadlessHigh,
	} {
		if !slices.Contains(names, want) {
			t.Errorf("WorkProfileNames() missing %q", want)
		}
	}
}

func TestWorkProfileDivideAndConquerDirective(t *testing.T) {
	if DivideAndConquerDirective != DivideAndConquerFragment {
		t.Fatalf("DivideAndConquerDirective != DivideAndConquerFragment")
	}
	for _, name := range []string{
		"ponytail:medium", "ponytail:high",
		WorkProfilePonytailHeadlessLow, WorkProfilePonytailHeadlessMed, WorkProfilePonytailHeadlessHigh,
	} {
		got := DescribeWorkProfile(name)
		if !got.Known || !got.Applied {
			t.Fatalf("DescribeWorkProfile(%q) not applied: %+v", name, got)
		}
		if !strings.Contains(got.Segment, DivideAndConquerDirective) {
			t.Fatalf("profile %q missing DivideAndConquerDirective: %q", name, got.Segment)
		}
	}
}

func TestWorkProfileAutonomousGitSyncPushDirective(t *testing.T) {
	if AutonomousGitSyncPushDirective != AutonomousGitSyncPushFragment {
		t.Fatalf("AutonomousGitSyncPushDirective != AutonomousGitSyncPushFragment")
	}
	for _, name := range []string{
		"ponytail:medium", "ponytail:high",
		WorkProfilePonytailHeadlessLow, WorkProfilePonytailHeadlessMed, WorkProfilePonytailHeadlessHigh,
	} {
		got := DescribeWorkProfile(name)
		if !got.Known || !got.Applied {
			t.Fatalf("DescribeWorkProfile(%q) not applied: %+v", name, got)
		}
		if !strings.Contains(got.Segment, AutonomousGitSyncPushDirective) {
			t.Fatalf("profile %q missing AutonomousGitSyncPushDirective: %q", name, got.Segment)
		}
	}
}
