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
	if !metalgemm.Compiled() {
		return "built without darwin/arm64 cgo"
	}
	if !metalgemm.MPSAvailable() {
		return "no usable Metal device"
	}
	return "metal init failed"
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
	return "", "metal backend not registered (no reachable Metal device)"
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
		(!strings.Contains(fatal, "built without darwin/arm64 cgo") && !strings.Contains(fatal, "no usable Metal device")) {
		t.Fatalf("fatal %q must name the two-state availability reason (compiled-out OR no device)", fatal)
	}
	if metalgemm.Compiled() {
		t.Fatalf("metalgemm.Compiled() = true with no registered metal backend; availability verdict taxonomy drifted")
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
	if skip != "metal backend not registered (no reachable Metal device)" || fatal != "" {
		t.Fatalf("guard unset on unregistered device = (%q, %q), want the original skip verbatim, no fatal", fatal, skip)
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
