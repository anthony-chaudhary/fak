package compute

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// metal_guard_test.go — build-tag-free witness for the FAK_METAL_REQUIRE_DEVICE fail-loud
// guard (GG#12783). Metal device tests skip silently today, so a silent Metal→CPU
// degradation passes every cosine parity gate. This file mirrors the Vulkan twin
// (vulkan_test.go:34-44) with no OS/arch constraints: the decision core is a pure
// function over the metalgemm stub, so the guard contract is exercised on every host —
// including windows/amd64 CI where metal_test.go itself does not compile.

// metalAvailabilityVerdict names why no Metal backend is registered, mirroring the
// Vulkan guard's two-state reason: compiled-out of this binary vs built-but-unusable.
func metalAvailabilityVerdict() string {
	return metalAvailabilityVerdictFor(metalgemm.Compiled(), metalgemm.Available(), metalgemm.MPSAvailable())
}

const (
	metalSkipNoDevice    = "metal backend not registered (no reachable Metal device)"
	metalSkipMPSNotOptIn = "metal device present but MPS not opted in (set FAK_METAL_MPS=1)"
)

// metalAvailabilityVerdictFor is pure so every state is testable on any host. The backend
// registers only with MPS, which is opt-in because probing it can fault headless, so a
// present device without MPS must not be reported as "no device".
func metalAvailabilityVerdictFor(compiled, device, mps bool) string {
	switch {
	case !compiled:
		return "built without darwin/arm64 cgo"
	case !device:
		return "no usable Metal device"
	case !mps:
		return metalSkipMPSNotOptIn
	}
	return "metal init failed"
}

func metalSkipReasonFor(device, mps bool) string {
	if device && !mps {
		return metalSkipMPSNotOptIn
	}
	return metalSkipNoDevice
}

func metalSkipReason() string {
	return metalSkipReasonFor(metalgemm.Available(), metalgemm.MPSAvailable())
}

// metalGuardVerdict is the pure decision core of metalOrSkip — the Metal twin of the vk(t)
// fail-loud guard: with FAK_METAL_REQUIRE_DEVICE=1 an unregistered device is a hard
// failure naming the availability verdict instead of a silent skip a CPU-route regression
// could hide behind; FAK_METAL_EXPECT_DEVICE tier-matches a registered device's label.
// Returned fatalReason != "" means t.Fatal; else skipReason != "" means t.Skip.
func metalGuardVerdict(registered bool, tier string) (fatalReason, skipReason string) {
	if registered {
		if expected := os.Getenv("FAK_METAL_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(tier), strings.ToLower(expected)) {
			return "Metal device " + tier + " does not match required device " + expected, ""
		}
		return "", ""
	}
	if os.Getenv("FAK_METAL_REQUIRE_DEVICE") == "1" {
		return "required Metal device is not registered: " + metalAvailabilityVerdict(), ""
	}
	return "", metalSkipReason()
}

// TestMetalGuardRequireDeviceFailsLoudOnMetalLessHost pins the fail-loud rung on the
// stub build: with FAK_METAL_REQUIRE_DEVICE=1 and no registered device the guard must
// produce a fatal naming the two-state availability reason ("built without darwin/arm64
// cgo" here; "no usable Metal device" on a darwin/arm64 host without a device).
func TestMetalGuardRequireDeviceFailsLoudOnMetalLessHost(t *testing.T) {
	if _, registered := Lookup("metal"); registered {
		t.Skip("metal backend registered on this host; fail-loud path is vacuous here")
	}
	t.Setenv("FAK_METAL_REQUIRE_DEVICE", "1")
	fatal, skip := metalGuardVerdict(false, "")
	if fatal == "" || skip != "" {
		t.Fatalf("guard with FAK_METAL_REQUIRE_DEVICE=1 on unregistered device = (%q, %q), want fatal naming the reason, no skip", fatal, skip)
	}
	if !strings.Contains(fatal, "required Metal device is not registered") ||
		(!strings.Contains(fatal, "built without darwin/arm64 cgo") && !strings.Contains(fatal, "no usable Metal device") && !strings.Contains(fatal, "MPS not opted in")) {
		t.Fatalf("fatal %q must name the two-state availability reason (compiled-out OR no device)", fatal)
	}
	// Pin that the fatal's named reason matches the compiled/unavailable state. A
	// darwin/arm64 cgo build with metalgemm present but no reachable device is a
	// legitimate "no usable Metal device" host: the backend stays unregistered even
	// though it compiled, so Compiled() alone must not be assumed false.
	if metalgemm.Compiled() {
		if !strings.Contains(fatal, "no usable Metal device") && !strings.Contains(fatal, "MPS not opted in") && !strings.Contains(fatal, "metal init failed") {
			t.Fatalf("compiled Metal host with no registered backend named reason %q, want the device-unavailable state", fatal)
		}
	} else if !strings.Contains(fatal, "built without darwin/arm64 cgo") {
		t.Fatalf("uncompiled Metal build named reason %q, want the compiled-out state", fatal)
	}
}

// TestMetalGuardUnsetEnvPreservesSkip pins the unset-env byte-compatibility rung: without
// FAK_METAL_REQUIRE_DEVICE an unregistered device must still produce the original skip
// message verbatim, so CPU-only CI sees zero test churn.
func TestMetalGuardUnsetEnvPreservesSkip(t *testing.T) {
	if _, registered := Lookup("metal"); registered {
		t.Skip("metal backend registered on this host; skip path is vacuous here")
	}
	t.Setenv("FAK_METAL_REQUIRE_DEVICE", "")
	fatal, skip := metalGuardVerdict(false, "")
	if want := metalSkipReason(); skip != want || fatal != "" {
		t.Fatalf("guard unset on unregistered device = (%q, %q), want skip %q, no fatal", fatal, skip, want)
	}
}

func TestMetalAvailabilityVerdictStates(t *testing.T) {
	cases := []struct {
		compiled, device, mps bool
		verdict, skip         string
	}{
		{false, false, false, "built without darwin/arm64 cgo", metalSkipNoDevice},
		{true, false, false, "no usable Metal device", metalSkipNoDevice},
		{true, true, false, metalSkipMPSNotOptIn, metalSkipMPSNotOptIn},
		{true, true, true, "metal init failed", metalSkipNoDevice},
	}
	for _, c := range cases {
		if got := metalAvailabilityVerdictFor(c.compiled, c.device, c.mps); got != c.verdict {
			t.Errorf("verdict(compiled=%v device=%v mps=%v) = %q, want %q", c.compiled, c.device, c.mps, got, c.verdict)
		}
		if got := metalSkipReasonFor(c.device, c.mps); got != c.skip {
			t.Errorf("skip(device=%v mps=%v) = %q, want %q", c.device, c.mps, got, c.skip)
		}
	}
}

// TestMetalGuardExpectDeviceTierMismatch pins the tier-match rung: with a registered
// device whose tier label does not contain the FAK_METAL_EXPECT_DEVICE value, the guard
// must fail naming the expected tier. Device-gated: runs only where Metal is registered.
func TestMetalGuardExpectDeviceTierMismatch(t *testing.T) {
	if _, registered := Lookup("metal"); !registered {
		t.Skip("no metal backend registered on this host; tier-match path is device-gated")
	}
	t.Setenv("FAK_METAL_REQUIRE_DEVICE", "1")
	t.Setenv("FAK_METAL_EXPECT_DEVICE", "DefinitelyNotThisTierLabel")
	fatal, skip := metalGuardVerdict(true, Pick("metal").Tier())
	if fatal == "" || skip != "" || !strings.Contains(fatal, "DefinitelyNotThisTierLabel") {
		t.Fatalf("guard with mismatched FAK_METAL_EXPECT_DEVICE = (%q, %q), want fatal naming the expected tier", fatal, skip)
	}
}

// TestMetalGuardExpectDeviceTierMatchARB pins the tier-match pass-through rung with
// stable expectations: a substring match must survive case differences, and an
// empty FAK_METAL_EXPECT_DEVICE must never fatal a registered device.
func TestMetalGuardExpectDeviceTierMatchARB(t *testing.T) {
	if _, registered := Lookup("metal"); !registered {
		t.Skip("no metal backend registered on this host; tier-match path is device-gated")
	}
	t.Setenv("FAK_METAL_REQUIRE_DEVICE", "1")
	t.Setenv("FAK_METAL_EXPECT_DEVICE", strings.ToUpper(Pick("metal").Tier()))
	if fatal, skip := metalGuardVerdict(true, Pick("metal").Tier()); fatal != "" || skip != "" {
		t.Fatalf("case-insensitive tier match = (%q, %q), want pass-through", fatal, skip)
	}
	t.Setenv("FAK_METAL_EXPECT_DEVICE", "")
	if fatal, skip := metalGuardVerdict(true, Pick("metal").Tier()); fatal != "" || skip != "" {
		t.Fatalf("empty FAK_METAL_EXPECT_DEVICE = (%q, %q), want pass-through", fatal, skip)
	}
}
