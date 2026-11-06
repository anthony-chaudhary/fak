package compute

import (
	"os"
	"strings"
	"testing"
)

// This CPU source witness pins the V4.1 pipelined-cycle contract in the native
// shim without compiling cgo or executing Vulkan: a V4.1-checked batch cycles
// (its submit latches the quarantine synchronously, its completion latches at
// retire), an open step window stays serialized, and no resource of a retiring
// batch recycles before its checked result is known.
// fak-test:runtime fast est=5ms lane=default
func TestVulkanV41PipelineSourceContract(t *testing.T) {
	t.Parallel()
	src, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	cycle := cFuncBody(t, string(src), "void batchCycle() {")
	drain := cFuncBody(t, string(src), "void drainInflight() {")

	// The closed eligibility contract, verbatim: it keeps the timestamp-profile,
	// armed-hazard and open-step-window exclusions and no longer excludes
	// V4.1-checked batches.
	const wantEligible = "const bool eligible = g_pipelineFlush && !g_stepBatch && !g_tsOn &&\n        !g_batchHazardsArmed && g_submissionStatus == VK_SUCCESS;"
	if got := cStatement(cycle, "const bool eligible ="); got != wantEligible {
		t.Fatalf("batchCycle eligibility changed:\n%s\nwant:\n%s", got, wantEligible)
	}

	// A failed submit latches the pending failure synchronously, before anything
	// stashes; only checked batches take that path instead of the fatal abort.
	iFail := strings.Index(cycle, "sr != VK_SUCCESS")
	iLatch := strings.Index(cycle, "g_v41SubmissionPendingFailure = true")
	if iFail < 0 || iLatch < 0 || iFail > iLatch {
		t.Fatalf("batchCycle must latch the pending failure on a failed submit (fail at %d, latch at %d)", iFail, iLatch)
	}
	if iStash := strings.Index(cycle, "g_inflight.active = true"); iStash < iLatch {
		t.Fatal("the in-flight stash runs before the submit-side latch")
	}
	for _, stash := range []string{"g_inflight.v41 = g_batchHasV41", "g_inflight.hadWork = g_batchOps > 0"} {
		if !strings.Contains(cycle, stash) {
			t.Fatalf("batchCycle stopped stashing %q into the in-flight batch", stash)
		}
	}

	// At retire the checked bookkeeping precedes every recycle: the wait-side
	// latch, then the retain guard, then descriptor sets, command parking and
	// parked-buffer frees.
	pos := -1
	for _, step := range []struct{ what, anchor string }{
		{"wait-side latch", "g_v41SubmissionPendingFailure = true"},
		{"retain guard", "if (g_v41SubmissionPendingFailure) return;"},
		{"descriptor-set recycling", "g_descSetPool["},
		{"command parking", "g_batchIdleCmd = done.cmd"},
		{"parked-buffer frees", "fvk_free(b)"},
	} {
		at := strings.Index(drain, step.anchor)
		if at <= pos {
			t.Fatalf("drainInflight: %s (at %d) must come after the previous step (at %d)", step.what, at, pos)
		}
		pos = at
	}
}

// cFuncBody returns the brace-matched body of the C++ function defined by sig.
func cFuncBody(t *testing.T, src, sig string) string {
	t.Helper()
	start := strings.Index(src, sig)
	if start < 0 {
		t.Fatalf("anchor %q not found in vulkan_shim.cpp", sig)
	}
	open := strings.IndexByte(src[start:], '{')
	if open < 0 {
		t.Fatalf("no body opening after %q", sig)
	}
	depth := 0
	for i := start + open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start : i+1]
			}
		}
	}
	t.Fatalf("unbalanced braces after %q", sig)
	return ""
}

// cStatement returns the statement starting at anchor, up to and including its
// terminating semicolon.
func cStatement(body, anchor string) string {
	start := strings.Index(body, anchor)
	if start < 0 {
		return "<absent>"
	}
	end := strings.IndexByte(body[start:], ';')
	if end < 0 {
		return "<unterminated>"
	}
	return body[start : start+end+1]
}
