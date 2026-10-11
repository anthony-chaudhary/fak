package compute

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// Deferred source-only witness; neither compiler nor hardware qualification.
// fak-test:runtime fast est=2ms lane=default
func TestV41IndexerAdmissionFaultBeforeCapability(t *testing.T) {
	t.Parallel()
	for _, supported := range []bool{false, true} {
		for _, tc := range []struct {
			status int
			class  VulkanErrorClass
			cause  error
		}{
			{0, "", nil},
			{-4, VulkanClassDeviceLost, ErrVulkanDeviceLost},
			{-1, VulkanClassAllocationFailed, ErrVulkanAllocationFailed},
			{-2, VulkanClassAllocationFailed, ErrVulkanAllocationFailed},
			{-13, VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed},
			{3, VulkanClassExecutionFailed, ErrVulkanExecutionFailed},
		} {
			got, err := vulkanV41IndexerScoreAdmissionResult(tc.status, supported)
			if tc.status == 0 {
				if got != supported || err != nil {
					t.Fatalf("healthy capability=%v: got %v, %v", supported, got, err)
				}
				continue
			}
			var typed *BackendError
			if got || !errors.Is(err, tc.cause) || !errors.As(err, &typed) || typed.Class != tc.class ||
				typed.Backend != "vulkan" || typed.Site != "V41IndexerScoreAdmission" {
				t.Fatalf("fault status=%d capability=%v: got %v, %v", tc.status, supported, got, err)
			}
		}
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	section := func(source, begin, end string) string {
		t.Helper()
		start := strings.Index(source, begin)
		if start < 0 {
			t.Fatalf("missing section %q", begin)
		}
		body := source[start+len(begin):]
		stop := strings.Index(body, end)
		if stop < 0 {
			t.Fatalf("missing end %q after %q", end, begin)
		}
		return body[:stop]
	}
	adapter := section(read("vulkan_v41_indexer_score.go"),
		"func (v *vulkanBackend) V41IndexerScoreAdmission() (bool, error) {", "\n}\n")
	last := -1
	for _, token := range []string{
		"if v == nil", "vulkanMu.Lock()", "defer vulkanMu.Unlock()",
		"if status := int(C.fvk_submission_status()); status != 0 {",
		"return vulkanV41IndexerScoreAdmissionResult(status, false)",
		"return vulkanV41IndexerScoreAdmissionResult(0, C.fvk_have_v41_indexer_score() != 0)",
	} {
		next := strings.Index(adapter, token)
		if next <= last {
			t.Fatalf("missing or reordered admission step %q", token)
		}
		last = next
	}
	if strings.Count(adapter, "C.") != 2 || strings.Contains(adapter, "UnavailableReason") {
		t.Fatal("admission must only read native status/capability without reason parsing")
	}
	shim := read("vulkan_shim.cpp")
	// The status getter is sufficient only while native pending quarantine
	// implies nonzero sticky status. Every pending transition outside the
	// checked completion (pinned below) must set sticky status in its own block.
	checkedBody := section(shim, "bool v41EndSubmitWaitChecked(VkCommandBuffer cmd, bool submit) {", "\n}\n")
	outside := strings.Replace(shim, checkedBody, "", 1)
	lines := strings.Split(outside, "\n")
	pending := 0
	for i, line := range lines {
		if !strings.Contains(line, "g_v41SubmissionPendingFailure = true;") {
			continue
		}
		pending++
		paired := false
		for j := i; j >= 0 && j >= i-4; j-- {
			if strings.Contains(lines[j], "g_submissionStatus = ") && !strings.Contains(lines[j], "VK_SUCCESS;") {
				paired = true
				break
			}
		}
		if !paired {
			t.Fatalf("pending quarantine transition without sticky status in its block: %q", strings.TrimSpace(line))
		}
	}
	if pending < 2 ||
		strings.Count(shim, "g_v41SubmissionPendingFailure = false;") != 1 ||
		strings.Count(shim, "g_submissionStatus = VK_SUCCESS;") != 1 {
		t.Fatal("native quarantine/status transitions changed; re-audit admission")
	}
	for _, token := range []string{
		"int fvk_submission_status(void) { return (int)g_submissionStatus; }",
		"g_submissionStatus = allocated < VK_SUCCESS ? allocated : VK_ERROR_UNKNOWN;",
		"if (allocated == VK_ERROR_DEVICE_LOST) g_v41SubmissionPendingFailure = true;",
		"g_submissionStatus = VK_ERROR_DEVICE_LOST;\n            g_v41SubmissionPendingFailure = true;",
	} {
		if !strings.Contains(shim, token) {
			t.Fatalf("missing native quarantine/status invariant %q", token)
		}
	}
	checked := section(shim, "bool v41EndSubmitWaitChecked(VkCommandBuffer cmd, bool submit) {", "\n}\n")
	normalize := strings.Index(checked, "g_submissionStatus = r < VK_SUCCESS ? r : VK_ERROR_UNKNOWN;")
	if strings.Count(checked, "g_v41SubmissionPendingFailure = true;") != 2 ||
		normalize < strings.LastIndex(checked, "g_v41SubmissionPendingFailure = true;") ||
		strings.Count(checked, "return ") != 1 || !strings.HasSuffix(checked, "return r == VK_SUCCESS;") {
		t.Fatal("checked completion no longer records sticky status after every pending transition")
	}
	// Production capability is exactly the dispatch prerequisites: no env or
	// module-byte switch, and the same sticky-fault/float-control/pipeline gate
	// the checked dispatch repeats (Ref #13668).
	production := section(shim, "int fvk_have_v41_indexer_score(void) {", "\n}\n")
	if strings.Count(production, "return ") != 1 || !strings.HasSuffix(production, "return v41IndexerScorePrerequisites() ? 1 : 0;") ||
		strings.Contains(production, "getenv") {
		t.Fatal("production capability is no longer exactly the dispatch prerequisites")
	}
}
