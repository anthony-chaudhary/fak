//go:build windows

package main

import (
	"regexp"
	"testing"
)

// vulkanVersionRe matches the "x.y.z" major.minor.patch rendering the probe emits.
var vulkanVersionRe = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)

// TestProbeVulkanLoaderFactsMapsResult exercises the live Windows probe seam and
// asserts the fact it maps. It must tolerate a host WITH or WITHOUT a Vulkan
// runtime: when Available is false the Kind must be one of the five known typed
// kinds (never a crash or an empty kind), and when Available the Version must
// parse as x.y.z. Absence of Vulkan is never a test failure.
func TestProbeVulkanLoaderFactsMapsResult(t *testing.T) {
	f := probeVulkanLoaderFacts()
	if f == nil {
		t.Fatal("probeVulkanLoaderFacts returned nil on windows")
	}
	if !f.Applicable {
		t.Error("windows probe fact should be Applicable")
	}

	knownKinds := map[string]bool{
		"loader":            true,
		"symbol":            true,
		"call":              true,
		"zero-version":      true,
		"malformed-version": true,
	}

	if f.Available {
		if !vulkanVersionRe.MatchString(f.Version) {
			t.Errorf("Available=true but Version %q is not x.y.z", f.Version)
		}
		if f.Kind != "" {
			t.Errorf("Available=true should carry no error kind, got %q", f.Kind)
		}
	} else {
		if !knownKinds[f.Kind] {
			t.Errorf("Available=false but Kind %q is not one of the five known kinds", f.Kind)
		}
		if f.Version != "" {
			t.Errorf("Available=false should carry no version, got %q", f.Version)
		}
		if f.Detail == "" {
			t.Error("Available=false should carry an operator-facing Detail")
		}
	}
}
