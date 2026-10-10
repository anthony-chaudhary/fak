//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
	"time"
)

const vulkanProfileFixtureChildEnv = "FAK_TEST_VULKAN_PROFILE_FIXTURE_CHILD"

// runVulkanProfileFixture returns true after the parent has run the exact fixture
// in a fresh process. Native dispatch profiling is a C++ static startup setting;
// changing Go's environment after registration cannot enable its real counters.
func runVulkanProfileFixture(t *testing.T) bool {
	t.Helper()
	if os.Getenv(vulkanProfileFixtureChildEnv) == t.Name() {
		if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
			t.Fatal("Vulkan profile child must start with dispatch profiling enabled")
		}
		return false
	}
	// Preserve the original device/identity gate and ordinary device-less skip.
	// Once eligible, the child must actually execute, never convert to a skip.
	vk(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.v", "-test.run=^"+regexp.QuoteMeta(t.Name())+"$")
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if key != vulkanProfileFixtureChildEnv && key != "FAK_VULKAN_DISPATCH_PROFILE" && key != "FAK_VULKAN_REQUIRE_DEVICE" {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	cmd.Env = append(cmd.Env, vulkanProfileFixtureChildEnv+"="+t.Name(),
		"FAK_VULKAN_DISPATCH_PROFILE=1", "FAK_VULKAN_REQUIRE_DEVICE=1")
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("Vulkan profile fixture timed out: %v\n%s", ctx.Err(), out)
	}
	if err != nil {
		t.Fatalf("Vulkan profile fixture failed: %v\n%s", err, out)
	}
	if strings.Contains(string(out), "--- SKIP:") ||
		!strings.Contains(string(out), "--- PASS: "+t.Name()+" (") {
		t.Fatalf("Vulkan profile fixture did not execute every required case:\n%s", out)
	}
	// Preserve the child's physical-counter receipt and case results in -v output.
	t.Logf("Vulkan profile fixture output:\n%s", out)
	return true
}
