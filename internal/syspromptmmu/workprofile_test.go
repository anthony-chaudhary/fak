package syspromptmmu

import (
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
