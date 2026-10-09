package compute

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// This bounded source-ordering check does not execute Vulkan or establish a
// driver lifetime witness. Mutations below check that its gates fail closed.
func TestVulkanRestoreLifetimeSourceOrder(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	if err := validateVulkanRestoreLifetimeSource(source); err != nil {
		t.Fatal(err)
	}
	const submit = "int fvk_restore_submit(void) {"
	const checked = "const bool completed = v41EndSubmitWaitChecked(cmd, true);"
	const quarantine = "if (g_v41SubmissionPendingFailure) return (int)g_submissionStatus;"
	const failed = "if (!completed) return (int)g_submissionStatus;"
	for _, tc := range []struct {
		name, function, old, replacement string
	}{
		{"missing checked call", submit, checked, "const bool completed = true;"},
		{"missing quarantine return", submit, quarantine, ""},
		{"clear before quarantine", submit, quarantine + "\n    g_restore.cmd = VK_NULL_HANDLE;", "g_restore.cmd = VK_NULL_HANDLE;\n    " + quarantine},
		{"missing command clear", submit, "g_restore.cmd = VK_NULL_HANDLE;", ""},
		{"early safe failure return", submit, "g_restore.payloadBytes = 0;\n    " + failed, failed + "\n    g_restore.payloadBytes = 0;"},
		{"missing failure gate", submit, failed, ""},
		{"early success accounting", submit, failed + "\n    ++g_restore.successfulSubmits;", "++g_restore.successfulSubmits;\n    " + failed},
		{"missing entry snapshot", submit, "size_t submittedEntries = g_restore.entries;", ""},
		{"missing byte counter", submit, "checkedCounterAdd(g_h2dBytes, submittedBytes)", "true"},
		{"missing injection discard", submit, "restoreDiscardCommand();", ""},
		{"injection incorrectly quarantines", submit, "restoreDiscardCommand();", "g_v41SubmissionPendingFailure = true;\n        restoreDiscardCommand();"},
		{"missing submit entry gate", submit, "if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;", ""},
		{"missing begin entry gate", "int fvk_restore_begin(size_t max_bytes, size_t max_entries) {", "if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;", ""},
		{"missing add entry gate", "int fvk_restore_add(void* dst_handle, size_t dst_offset, const void* src, size_t bytes) {", "if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;", ""},
		{"discard loses quarantine", "void restoreDiscardCommand() {", "if (g_v41SubmissionPendingFailure) return;", ""},
		{"cleanup rejects safe sticky failure", "void restoreCleanup() {", "if (g_v41SubmissionPendingFailure) return;", "if (g_submissionStatus != VK_SUCCESS) return;"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body, err := vulkanRestoreLifetimeFunction(source, tc.function)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Count(body, tc.old) != 1 {
				t.Fatalf("negative control must replace exactly one %q", tc.old)
			}
			mutated := strings.Replace(body, tc.old, tc.replacement, 1)
			if err := validateVulkanRestoreLifetimeSource(strings.Replace(source, body, mutated, 1)); err == nil {
				t.Fatal("unsafe or incomplete source passed the ordering validator")
			}
		})
	}
}

func validateVulkanRestoreLifetimeSource(source string) error {
	if len(source) > 1<<20 {
		return fmt.Errorf("Vulkan source exceeds 1 MiB ordering-check bound")
	}
	// Every marker must exist exactly once in its function. Missing markers must
	// never turn strings.Index == -1 into an apparently valid ordering.
	ordered := func(body string, markers ...string) error {
		previous := -1
		for _, marker := range markers {
			if strings.Count(body, marker) != 1 {
				return fmt.Errorf("expected exactly one marker %q", marker)
			}
			at := strings.Index(body, marker)
			if at <= previous {
				return fmt.Errorf("out-of-order marker %q", marker)
			}
			previous = at
		}
		return nil
	}
	for _, signature := range []string{
		"int fvk_restore_begin(size_t max_bytes, size_t max_entries) {",
		"int fvk_restore_add(void* dst_handle, size_t dst_offset, const void* src, size_t bytes) {",
		"int fvk_restore_submit(void) {",
		"VkResult restoreBeginCommand() {",
	} {
		body, err := vulkanRestoreLifetimeFunction(source, signature)
		if err != nil {
			return err
		}
		gate := "if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;"
		if strings.HasPrefix(signature, "VkResult") {
			gate = "if (g_submissionStatus != VK_SUCCESS) return g_submissionStatus;"
		}
		if !strings.HasPrefix(strings.TrimSpace(strings.TrimPrefix(body, signature)), gate) {
			return fmt.Errorf("%s must reject sticky status before mutation", signature)
		}
	}
	submit, err := vulkanRestoreLifetimeFunction(source, "int fvk_restore_submit(void) {")
	if err != nil {
		return err
	}
	if err := ordered(submit,
		"g_restore.successfulSubmits >= g_restoreFailAfterSubmits",
		"restoreDiscardCommand();",
		"g_submissionStatus = VK_ERROR_DEVICE_LOST;",
		"return (int)VK_ERROR_DEVICE_LOST;",
		"VkCommandBuffer cmd = g_restore.cmd;",
		"const bool completed = v41EndSubmitWaitChecked(cmd, true);",
		"if (g_v41SubmissionPendingFailure) return (int)g_submissionStatus;",
		"g_restore.cmd = VK_NULL_HANDLE;",
		"size_t submittedBytes = g_restore.payloadBytes;",
		"size_t submittedEntries = g_restore.entries;",
		"g_restore.used = 0;",
		"g_restore.entries = 0;",
		"g_restore.payloadBytes = 0;",
		"if (!completed) return (int)g_submissionStatus;",
		"++g_restore.successfulSubmits;",
		"checkedCounterAdd(g_h2dCount, submittedEntries)",
		"checkedCounterAdd(g_h2dBytes, submittedBytes)",
		"dpOneShot(g_dp.oneShotH2D);",
		"return 0;"); err != nil {
		return err
	}
	if !strings.Contains(submit, "const bool completed = v41EndSubmitWaitChecked(cmd, true);\n    if (g_v41SubmissionPendingFailure) return (int)g_submissionStatus;") {
		return fmt.Errorf("uncertain completion must return immediately without touching restore state")
	}
	for _, forbidden := range []string{
		"vkEndCommandBuffer(", "vkResetFences(", "vkQueueSubmit(", "vkWaitForFences(", "vkFreeCommandBuffers(",
		"g_v41SubmissionPendingFailure =", "g_restore.successfulSubmits =", "g_restore =", "restoreCleanup(",
	} {
		if strings.Contains(submit, forbidden) {
			return fmt.Errorf("restore submit bypasses checked ownership or resets prior batches: %q", forbidden)
		}
	}
	// Safe pre-submit failures still discard/clean up; uncertain submissions retain
	// the command, mapped stage and payload until process exit.
	for _, check := range []struct {
		signature string
		markers   []string
	}{
		{"void restoreDiscardCommand() {", []string{"if (g_v41SubmissionPendingFailure) return;", "vkFreeCommandBuffers(", "g_restore.cmd = VK_NULL_HANDLE;", "g_restore.used = 0;", "g_restore.entries = 0;", "g_restore.payloadBytes = 0;"}},
		{"void restoreCleanup() {", []string{"if (g_v41SubmissionPendingFailure) return;", "restoreDiscardCommand();", "vkUnmapMemory(", "destroyBuffer(g_restore.stage);", "g_restore = RestoreTransaction{};"}},
	} {
		body, err := vulkanRestoreLifetimeFunction(source, check.signature)
		if err != nil {
			return err
		}
		if !strings.HasPrefix(strings.TrimSpace(strings.TrimPrefix(body, check.signature)), check.markers[0]) || strings.Contains(body, "g_submissionStatus != VK_SUCCESS") {
			return fmt.Errorf("%s must retain uncertain work but allow safe sticky cleanup", check.signature)
		}
		if err := ordered(body, check.markers...); err != nil {
			return err
		}
	}
	return nil
}

// Delimit only the named, conventionally formatted function, not a declaration
// or an unrelated later occurrence. This is intentionally not a C++ parser.
func vulkanRestoreLifetimeFunction(source, signature string) (string, error) {
	if strings.Count(source, signature) != 1 {
		return "", fmt.Errorf("expected exactly one function %q", signature)
	}
	start := strings.Index(source, signature)
	end := strings.Index(source[start:], "\n}\n")
	if end < 0 || end > 8<<10 {
		return "", fmt.Errorf("missing or oversized function %q", signature)
	}
	return source[start : start+end+3], nil
}
