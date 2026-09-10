//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"
)

const vulkanReadCheckedChildEnv = "FAK_TEST_VULKAN_READ_CHECKED_CHILD"

func TestVulkanReadFailsClosed(t *testing.T) {
	mode := os.Getenv(vulkanReadCheckedChildEnv)
	if mode == "" {
		for _, childMode := range []string{"sticky-submission", "d2h-staging"} {
			t.Run(childMode, func(t *testing.T) {
				runVulkanReadCheckedChild(t, childMode)
			})
		}
		return
	}

	switch mode {
	case "sticky-submission":
		testVulkanReadStickySubmissionFailure(t)
	case "d2h-staging":
		testVulkanReadD2HStagingFailure(t)
	default:
		t.Fatalf("unknown %s mode %q", vulkanReadCheckedChildEnv, mode)
	}
}

func runVulkanReadCheckedChild(t *testing.T, mode string) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.v", "-test.run=^TestVulkanReadFailsClosed$")
	cmd.Env = append(os.Environ(), vulkanReadCheckedChildEnv+"="+mode)
	out, err := cmd.CombinedOutput()
	if strings.Contains(string(out), "--- SKIP: TestVulkanReadFailsClosed") {
		t.Skipf("Vulkan checked-read subprocess skipped without a device:\n%s", out)
	}
	if err != nil {
		t.Fatalf("Vulkan checked-read subprocess %q failed: %v\n%s", mode, err, out)
	}
}

func testVulkanReadStickySubmissionFailure(t *testing.T) {
	v := vk(t)
	values := []float32{3.25, -7.5, 11, 0.125}
	d := v.Upload(NewF32(Default(), []int{len(values)}, values), F32)

	if got := v.Read(d); !slices.Equal(got, values) {
		v.Free(d)
		t.Fatalf("initial device read = %v, want %v", got, values)
	}

	v.VulkanDebugSetRestoreFailureAfterSubmits(0)
	_, _, restoreErr := v.VulkanRestoreImmutableResidencyGroup(
		context.Background(),
		[]VulkanImmutableResidencySource{{Binding: "checked-read/submission", Bytes: []byte{1, 2, 3, 4}}},
		VulkanRestoreLimits{MaxBatchBytes: 4, MaxBatchEntries: 1},
	)
	v.VulkanDebugSetRestoreFailureAfterSubmits(-1)
	if restoreErr == nil {
		v.Free(d)
		t.Fatal("native submission failure injection did not fail the restore submit")
	}

	be := requireVulkanReadPanic(t, func() { _ = v.Read(d) })
	requireVulkanReadError(t, be, VulkanClassDeviceLost, ErrVulkanDeviceLost)
	// The injected device failure is intentionally sticky. This child process owns
	// the poisoned Vulkan context and exits without attempting further operations.
}

func testVulkanReadD2HStagingFailure(t *testing.T) {
	v := vk(t)
	values := []float32{9.5, -2.25, 6.75, 14}
	d := v.Upload(NewF32(Default(), []int{len(values)}, values), F32)
	defer v.Free(d)
	defer v.VulkanDebugSetD2HStagingFailureOnce(false)

	if got := v.Read(d); !slices.Equal(got, values) {
		t.Fatalf("initial device read = %v, want %v", got, values)
	}

	v.VulkanDebugSetD2HStagingFailureOnce(true)
	probe := v.Upload(NewF32(Default(), []int{1}, []float32{21}), F32)
	v.Free(probe)
	be := requireVulkanReadPanic(t, func() { _ = v.Read(d) })
	requireVulkanReadError(t, be, VulkanClassAllocationFailed, ErrVulkanAllocationFailed)

	if got := v.Read(d); !slices.Equal(got, values) {
		t.Fatalf("read after one-shot D2H failure = %v, want %v", got, values)
	}
}

func requireVulkanReadPanic(t *testing.T, read func()) *BackendError {
	t.Helper()
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		read()
	}()
	if recovered == nil {
		t.Fatal("Vulkan Read returned a slice after an injected native failure; want typed panic")
	}
	be, ok := recovered.(*BackendError)
	if !ok {
		t.Fatalf("Vulkan Read panic type = %T (%v), want *BackendError", recovered, recovered)
	}
	return be
}

func requireVulkanReadError(t *testing.T, be *BackendError, class VulkanErrorClass, sentinel error) {
	t.Helper()
	if be.Backend != "vulkan" || be.Site != "Read" || be.Class != class || !errors.Is(be, sentinel) {
		t.Fatalf("Vulkan Read error = %+v, want backend=vulkan site=Read class=%s sentinel=%v", be, class, sentinel)
	}
}
