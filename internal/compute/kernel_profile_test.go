package compute

import (
	"context"
	"flag"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

func kernelProfileFixture() *KernelProfile {
	return &KernelProfile{
		Backend: "vulkan", Batches: 10, Ops: 300, Truncated: 0, UntimedCompute: 1, UntimedTransfer: 2, BarriersElided: 0,
		GPUNS: 9000, WallNS: 12000,
		Rows: []KernelRow{
			{Kernel: "q2k_matvec", GX: 5120, GY: 1, Calls: 100, NS: 6000},
			{Kernel: "rmsnorm", GX: 1, GY: 1, Calls: 100, NS: 1000},
			{Kernel: "d2d_copy", Calls: 100, NS: 2000},
		},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelProfileDeltaExactRows(t *testing.T) {
	before := kernelProfileFixture()
	after := kernelProfileFixture()
	after.Batches, after.Ops, after.UntimedCompute, after.UntimedTransfer = 14, 420, 1, 5
	after.GPUNS, after.WallNS = 13500, 18000
	after.Rows = []KernelRow{
		{Kernel: "d2d_copy", Calls: 100, NS: 2000},
		{Kernel: "q2k_matvec", GX: 5120, GY: 1, Calls: 160, NS: 9600},
		{Kernel: "rmsnorm", GX: 1, GY: 1, Calls: 140, NS: 1400},
		{Kernel: "argmax", GX: 1, GY: 1, Calls: 20, NS: 500},
	}
	got := KernelProfileDelta(before, after)
	want := &KernelProfile{
		Backend: "vulkan", Batches: 4, Ops: 120, UntimedTransfer: 3, GPUNS: 4500, WallNS: 6000,
		Rows: []KernelRow{
			{Kernel: "q2k_matvec", GX: 5120, GY: 1, Calls: 60, NS: 3600},
			{Kernel: "argmax", GX: 1, GY: 1, Calls: 20, NS: 500},
			{Kernel: "rmsnorm", GX: 1, GY: 1, Calls: 40, NS: 400},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("delta = %#v\nwant   %#v", got, want)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelProfileDeltaNilOnReset(t *testing.T) {
	cases := map[string]func(p *KernelProfile){
		"batches":          func(p *KernelProfile) { p.Batches-- },
		"ops":              func(p *KernelProfile) { p.Ops-- },
		"untimed_compute":  func(p *KernelProfile) { p.UntimedCompute-- },
		"untimed_transfer": func(p *KernelProfile) { p.UntimedTransfer-- },
		"gpu_ns":           func(p *KernelProfile) { p.GPUNS-- },
		"wall_ns":          func(p *KernelProfile) { p.WallNS-- },
		"row_calls":        func(p *KernelProfile) { p.Rows[0].Calls-- },
		"row_ns":           func(p *KernelProfile) { p.Rows[1].NS-- },
		"row_missing":      func(p *KernelProfile) { p.Rows = p.Rows[:2] },
		"row_nan":          func(p *KernelProfile) { p.Rows[0].NS = math.NaN() },
		"gpu_inf":          func(p *KernelProfile) { p.GPUNS = math.Inf(1) },
		"backend":          func(p *KernelProfile) { p.Backend = "cuda" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			after := kernelProfileFixture()
			mutate(after)
			if got := KernelProfileDelta(kernelProfileFixture(), after); got != nil {
				t.Fatalf("delta = %#v, want nil", got)
			}
		})
	}
	if KernelProfileDelta(nil, kernelProfileFixture()) != nil || KernelProfileDelta(kernelProfileFixture(), nil) != nil {
		t.Fatal("a missing side must give a nil delta")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelProfileAddSumsWindows(t *testing.T) {
	a := &KernelProfile{Backend: "vulkan", Batches: 2, Ops: 4, UntimedTransfer: 1, GPUNS: 100, WallNS: 150,
		Rows: []KernelRow{{Kernel: "q2k_matvec", GX: 8, GY: 1, Calls: 3, NS: 90}, {Kernel: "rope", GX: 1, GY: 1, Calls: 1, NS: 10}}}
	b := &KernelProfile{Backend: "vulkan", Batches: 1, Ops: 2, Truncated: 1, GPUNS: 60, WallNS: 70,
		Rows: []KernelRow{{Kernel: "q2k_matvec", GX: 8, GY: 1, Calls: 2, NS: 50}, {Kernel: "fill", Calls: 0, NS: 10}}}
	got := KernelProfileAdd(a, b)
	want := &KernelProfile{Backend: "vulkan", Batches: 3, Ops: 6, Truncated: 1, UntimedTransfer: 1, GPUNS: 160, WallNS: 220,
		Rows: []KernelRow{{Kernel: "q2k_matvec", GX: 8, GY: 1, Calls: 5, NS: 140}, {Kernel: "fill", Calls: 0, NS: 10}, {Kernel: "rope", GX: 1, GY: 1, Calls: 1, NS: 10}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sum = %#v\nwant %#v", got, want)
	}
	if KernelProfileAdd(a, nil) != nil || KernelProfileAdd(a, &KernelProfile{Backend: "cuda"}) != nil {
		t.Fatal("a missing or foreign window must give a nil sum")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKernelProfileSnapshotNilWithoutReader(t *testing.T) {
	if KernelProfileReaderRegistered() {
		t.Skip("a backend kernel profile reader is registered in this process")
	}
	if got := KernelProfileSnapshot(); got != nil {
		t.Fatalf("snapshot = %#v, want nil with no reader", got)
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = KernelProfileSnapshot() }); allocs != 0 {
		t.Fatalf("off-path snapshot allocs = %v, want 0", allocs)
	}
	unregister := RegisterKernelProfileReader("fake", func() (KernelProfile, bool) { return KernelProfile{Ops: 7}, true })
	got := KernelProfileSnapshot()
	if got == nil || got.Backend != "fake" || got.Ops != 7 {
		t.Fatalf("snapshot = %#v, want the fake reader's profile", got)
	}
	unusable := RegisterKernelProfileReader("fake", func() (KernelProfile, bool) { return KernelProfile{}, false })
	if KernelProfileSnapshot() != nil {
		t.Fatal("an unusable read must give a nil snapshot")
	}
	unregister()
	if !KernelProfileReaderRegistered() {
		t.Fatal("a stale unregister removed the newer reader")
	}
	unusable()
	if KernelProfileReaderRegistered() || KernelProfileSnapshot() != nil {
		t.Fatal("unregister left a reader behind")
	}
}

func BenchmarkKernelProfileSnapshotOff(b *testing.B) {
	if KernelProfileReaderRegistered() {
		b.Skip("a backend kernel profile reader is registered in this process")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if KernelProfileSnapshot() != nil {
			b.Fatal("snapshot without a reader")
		}
	}
}

func shimSource(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func shimKernelNames(t *testing.T, src string) []string {
	t.Helper()
	m := regexp.MustCompile(`(?s)const char\* const kKernelNames\[\] = \{(.*?)\};`).FindStringSubmatch(src)
	if m == nil {
		t.Fatal("kKernelNames table not found")
	}
	var names []string
	for _, q := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(m[1], -1) {
		names = append(names, q[1])
	}
	return names
}

// fak-test:runtime fast est=20ms lane=default
func TestKernelOpClassClosed(t *testing.T) {
	src := shimSource(t)
	names := shimKernelNames(t, src)
	if len(names) < 80 {
		t.Fatalf("parsed %d kernel names, want the full table", len(names))
	}
	pseudo := regexp.MustCompile(`case kTs[A-Za-z0-9]+: return "([a-z0-9_]+)";`).FindAllStringSubmatch(src, -1)
	// Vulkan's pseudo rows are one backend's inventory, not the union of all
	// timed transfer aliases. Keep both the backend set and the union exact.
	vulkanTransfers := map[string]bool{
		"d2d_copy": true, "h2d_upload": true, "d2d_gather": true, "fill": true, "readback": true,
	}
	seenVulkan := map[string]bool{}
	for _, p := range pseudo {
		name := p[1]
		if !vulkanTransfers[name] || seenVulkan[name] {
			t.Fatalf("unexpected or duplicate Vulkan transfer %q", name)
		}
		seenVulkan[name] = true
	}
	if len(seenVulkan) != len(vulkanTransfers) {
		t.Fatalf("Vulkan transfer inventory = %v, want %v", seenVulkan, vulkanTransfers)
	}
	expectedTransfers := map[string]bool{}
	for name := range vulkanTransfers {
		expectedTransfers[name] = true
	}
	for _, name := range []string{"sampling_candidates_fill", "sampling_candidates_retain"} {
		expectedTransfers[name] = true
	}
	if len(kernelTransferNames) != len(expectedTransfers) {
		t.Fatalf("cross-backend transfer inventory = %v, want %v", kernelTransferNames, expectedTransfers)
	}
	for name := range expectedTransfers {
		if !kernelTransferNames[name] || KernelOpClass(name) != KernelOpCopy {
			t.Errorf("expected transfer %s is missing or not classified copy", name)
		}
	}
	closed := map[string]bool{}
	for _, c := range KernelOpClasses {
		closed[c] = true
	}
	for _, p := range pseudo {
		if got := KernelOpClass(p[1]); got != KernelOpCopy {
			t.Errorf("pseudo kernel %s class = %s, want copy", p[1], got)
		}
	}
	for _, name := range names {
		if !closed[KernelOpClass(name)] {
			t.Errorf("kernel %s class %q outside the closed set", name, KernelOpClass(name))
		}
	}
	for name, want := range map[string]string{
		"q2k_matvec":                    KernelOpMatvec,
		"iq2xxs_matvec":                 KernelOpMatvec,
		"q4k_matmul":                    KernelOpMatmul,
		"rmsnorm_q2k_matmul2":           KernelOpMatmul,
		"swiglu_q4k_matmul_add":         KernelOpMatmul,
		"matmul_argmax_blocks":          KernelOpMatmul,
		"v41_q2k_token_panel":           KernelOpMatmul,
		"attention":                     KernelOpAttention,
		"qwen35_causal_attention_panel": KernelOpAttention,
		"qwen35_gdn_recurrent":          KernelOpGDN,
		"qwen35_gdn_q8_in_proj":         KernelOpMatvec,
		"v41_routed_gate_up_swiglu_id":  KernelOpMatvec,
		"v41_routed_down_sum":           KernelOpMatvec,
		"v41_q2k_matvec_id":             KernelOpMatvec,
		"v41_routed_swiglu_id":          KernelOpSwiGLU,
		"v41_routed_swiglu_panel_row":   KernelOpSwiGLU,
		"v41_routed_swiglu_panel_chunk": KernelOpSwiGLU,
		"v41_routed_down_panel_row":     KernelOpOther,
		"v41_routed_down_panel_chunk":   KernelOpOther,
		"glm_kda_wave32":                KernelOpGDN,
		"rmsnorm":                       KernelOpNorm,
		"v41_exact_rmsnorm":             KernelOpNorm,
		"rope":                          KernelOpRope,
		"qwen35_partial_rope_panel":     KernelOpRope,
		"swiglu":                        KernelOpSwiGLU,
		"argmax_pairs":                  KernelOpArgmax,
		"add":                           KernelOpOther,
		"sigmoid_mul":                   KernelOpOther,
		"readback":                      KernelOpCopy,
		"h2d_upload":                    KernelOpCopy,
		"splitkv":                       KernelOpOther,
		"lse_merge":                     KernelOpOther,
		"unknown":                       KernelOpOther,
		"":                              KernelOpOther,
	} {
		if got := KernelOpClass(name); got != want {
			t.Errorf("KernelOpClass(%q) = %s, want %s", name, got, want)
		}
	}
}

var (
	shimRecordedCommand = regexp.MustCompile(`vkCmd(?:Dispatch|CopyBuffer|FillBuffer)\(\s*g_batchCmd\b`)
	shimAnyCommand      = regexp.MustCompile(`vkCmd(?:Dispatch|CopyBuffer|FillBuffer)\(`)
	shimOneShotSubmit   = regexp.MustCompile(`(?:endSubmitWait|v41EndSubmitWaitChecked)\(cmd\b`)
)

// Every command recorded into a timed batch must be followed by its own
// timestamp, or its device time lands on the next timed op (or falls off the
// end of the batch). Every one-shot submit must be counted as untimed work.
// fak-test:runtime fast est=20ms lane=default
func TestTimestampProfileCoversBatchRecording(t *testing.T) {
	const timestampSearchLines = 4
	src := shimSource(t)
	recorded := shimRecordedCommand.FindAllStringIndex(src, -1)
	if len(recorded) < 8 {
		t.Fatalf("found %d batch-recorded commands, want the recorder's full set", len(recorded))
	}
	for _, loc := range recorded {
		rest := src[loc[1]:]
		end := len(rest)
		if next := shimAnyCommand.FindStringIndex(rest); next != nil {
			end = next[0]
		}
		window := rest[:end]
		if lines := strings.SplitN(window, "\n", timestampSearchLines+1); len(lines) > timestampSearchLines {
			window = strings.Join(lines[:timestampSearchLines], "\n")
		}
		if !strings.Contains(window, "tsAfterDispatch(") {
			line := strings.Count(src[:loc[0]], "\n") + 1
			t.Errorf("vulkan_shim.cpp:%d records into g_batchCmd without a following tsAfterDispatch", line)
		}
	}
	lines := strings.Split(src, "\n")
	submits := 0
	for i, line := range lines {
		if !shimOneShotSubmit.MatchString(line) {
			continue
		}
		submits++
		counted := strings.Contains(line, "// ts-untimed:") || (i > 0 && strings.Contains(lines[i-1], "// ts-untimed:"))
		for j := max(0, i-8); j < i; j++ {
			if strings.Contains(lines[j], "dpOneShot(") || strings.Contains(lines[j], "tsCountUntimed(") {
				counted = true
			}
		}
		if !counted {
			t.Errorf("vulkan_shim.cpp:%d one-shot submit is not counted as untimed work", i+1)
		}
	}
	if submits < 10 {
		t.Fatalf("found %d one-shot submits, want the shim's full set", submits)
	}
	if !regexp.MustCompile(`static inline void dpOneShot\([^)]*\) \{\n\s+if \(!timed\) tsCountUntimed\(`).MatchString(src) {
		t.Error("dpOneShot no longer counts untimed work first")
	}
	for _, format := range []string{
		`"fak-vulkan ts-profile batches=%llu ops/batch=%.1f gpu_ms/batch=%.3f wall_ms/batch=%.3f truncated=%llu mode=serialized_batches timing=bottom_of_pipe_intervals coverage=batched_commands_only\n"`,
		`"fak-vulkan ts-stage kernel=%s groups=%ux%u calls/batch=%.2f ms/batch=%.3f us/call=%.2f share=%.1f%%\n"`,
	} {
		if !strings.Contains(src, format) {
			t.Errorf("ts-stage stderr format changed: %s", format)
		}
	}
}

// shimUntimedComputeSites names the only functions whose one-shot compute
// stays untimed (counted in untimed_compute). Every other one-shot compute
// dispatch, the resident Qwen path's included, must carry a timestamp pair.
var shimUntimedComputeSites = map[string]bool{"fvk_v41_query_prefix_f32": true}

var (
	shimOneShotDispatch = regexp.MustCompile(`vkCmdDispatch\(cmd\b`)
	shimOneShotCopyFill = regexp.MustCompile(`vkCmd(?:CopyBuffer|FillBuffer)\(cmd\b`)
	shimFuncStart       = regexp.MustCompile(`^[A-Za-z][^;]*?(\w+)\(`)
)

// shimEnclosingFunc returns the top-level function holding line i: its name
// and the body's [start, end) line range.
func shimEnclosingFunc(lines []string, i int) (string, int, int) {
	start := i
	for start > 0 && !shimFuncStart.MatchString(lines[start]) {
		start--
	}
	end := i
	for end < len(lines) && lines[end] != "}" {
		end++
	}
	name := ""
	if m := shimFuncStart.FindStringSubmatch(lines[start]); m != nil {
		name = m[1]
	}
	return name, start, end
}

func shimFollowedWithin(lines []string, i, n int, want string) bool {
	for j := i + 1; j <= i+n && j < len(lines); j++ {
		if strings.Contains(lines[j], want) {
			return true
		}
	}
	return false
}

// A compute op submitted outside a batch must time itself: it opens a
// timestamp pair before its first command, stamps every command it records,
// and reads (or counts lost) the pair after its submit. Only the listed V4.1
// site may stay untimed, and it must count itself. The open refuses while a
// batch's slots are unread, and a one-shot is not a batch.
// fak-test:runtime fast est=20ms lane=default
func TestTimestampProfileTimesOneShotCompute(t *testing.T) {
	src := shimSource(t)
	lines := strings.Split(src, "\n")
	timed, untimed := 0, 0
	for i, line := range lines {
		if !shimOneShotDispatch.MatchString(line) {
			continue
		}
		name, start, end := shimEnclosingFunc(lines, i)
		body := strings.Join(lines[start:end], "\n")
		if shimUntimedComputeSites[name] {
			untimed++
			if !strings.Contains(body, "tsCountUntimed(true)") {
				t.Errorf("vulkan_shim.cpp:%d %s stays untimed without counting untimed_compute", i+1, name)
			}
			continue
		}
		timed++
		open := -1
		for j := start; j < end; j++ {
			if strings.Contains(lines[j], "const bool timed = tsOneShotBegin(cmd);") {
				open = j
				break
			}
		}
		if open < 0 {
			t.Errorf("vulkan_shim.cpp:%d %s submits one-shot compute without opening a timestamp pair", i+1, name)
			continue
		}
		if !shimFollowedWithin(lines, i, 2, "if (timed) tsAfterDispatch(cmd, (uint32_t)(&k - g_kern), groupsX, groupsY);") {
			t.Errorf("vulkan_shim.cpp:%d %s dispatch is not stamped", i+1, name)
		}
		if !strings.Contains(body, "dpOneShot(g_dp.oneShotCompute, timed);") {
			t.Errorf("%s does not tell dpOneShot the submit was timed", name)
		}
		submits := 0
		for j := start; j < end; j++ {
			if j < open && (shimOneShotDispatch.MatchString(lines[j]) || shimOneShotCopyFill.MatchString(lines[j])) {
				t.Errorf("vulkan_shim.cpp:%d %s records before the timestamp pair opens", j+1, name)
			}
			if shimOneShotCopyFill.MatchString(lines[j]) && !shimFollowedWithin(lines, j, 2, "if (timed) tsAfterDispatch(cmd, ") {
				t.Errorf("vulkan_shim.cpp:%d %s records an unstamped command", j+1, name)
			}
			if shimOneShotSubmit.MatchString(lines[j]) {
				submits++
				if !shimFollowedWithin(lines, j, 2, "if (timed) tsOneShotEnd(") {
					t.Errorf("vulkan_shim.cpp:%d %s submit does not read its timestamp pair", j+1, name)
				}
			}
		}
		if submits == 0 {
			t.Errorf("%s has no one-shot submit", name)
		}
	}
	if timed == 0 || untimed != len(shimUntimedComputeSites) {
		t.Fatalf("one-shot compute sites: timed=%d untimed=%d, want >=1 and %d", timed, untimed, len(shimUntimedComputeSites))
	}
	for name, re := range map[string]*regexp.Regexp{
		"open_refuses_pending_batch": regexp.MustCompile(`bool tsOneShotBegin\(VkCommandBuffer cmd\) \{\n\s+if \(!g_tsOn \|\| g_batching\) return false;\n\s+if \(!g_tsInitTried\) tsInit\(\);\n\s+if \(!g_tsUsable \|\| g_tsUsed != 0\) return false;\n\s+vkCmdResetQueryPool\(cmd, g_tsPool, 0, kTsOneShotSlots\);`),
		"one_shot_not_a_batch":       regexp.MustCompile(`if \(batch && \(\+\+g_tsBatches % g_tsEvery\) == 0\) tsReport\(\);`),
		"lost_one_shot_truncates":    regexp.MustCompile(`void tsOneShotEnd\(bool completed\) \{\n\s+if \(completed\) tsCollect\(false\);\n\s+else tsBatchLost\(\);\n\}`),
		"uncertain_v41_not_read":     regexp.MustCompile(`if \(timed\) tsOneShotEnd\(completed && !g_v41SubmissionPendingFailure\);`),
	} {
		if !re.MatchString(src) {
			t.Errorf("one-shot timestamp contract %s changed", name)
		}
	}
}

// A timed batch whose timestamps are never read must count its slots as
// truncated; otherwise its GPU time silently leaves the window while the
// profile still reads complete.
// fak-test:runtime fast est=20ms lane=default
func TestTimestampProfileCountsLostBatches(t *testing.T) {
	src := shimSource(t)
	for name, re := range map[string]*regexp.Regexp{
		"lost_counts_truncated": regexp.MustCompile(`void tsBatchLost\(\) \{\n\s+if \(g_tsOn && g_tsUsable && g_tsUsed >= 2\) g_tsTruncated \+= g_tsUsed - 1;\n\s+g_tsUsed = 0;\n\}`),
		"query_failure":         regexp.MustCompile(`VK_QUERY_RESULT_WAIT_BIT\) != VK_SUCCESS\) \{\n\s+tsBatchLost\(\);\n\s+return;`),
		"v41_not_collected":     regexp.MustCompile(`const bool completed = v41EndSubmitWaitChecked\(g_batchCmd, hadWork\);\n\s+if \(!completed \|\| !hadWork \|\| g_v41SubmissionPendingFailure\) tsBatchLost\(\);\n\s+if \(g_v41SubmissionPendingFailure\) return;[^\n]*\n\s+if \(completed && hadWork\) tsBatchCollect\(\);`),
		"empty_batch_freed":     regexp.MustCompile(`vkFreeCommandBuffers\(g_dev, g_cmdpool, 1, &g_batchCmd\);\n\s+tsBatchLost\(\);`),
	} {
		if !re.MatchString(src) {
			t.Errorf("shim lost-batch accounting %s is missing", name)
		}
	}
	// The asynchronous retire path is a third collector, not an unguarded
	// timestamp read. Its loss decision and quarantine return must precede
	// collection in the checked-batch block of the actual drain function.
	drain := cFuncBody(t, src, "void drainInflight() {")
	async := regexp.MustCompile(`if \(done\.v41\) \{\n(?:[ \t]*//[^\n]*\n)*[ \t]*if \(!completed \|\| !done\.hadWork \|\| g_v41SubmissionPendingFailure\) tsBatchLost\(\);\n[ \t]*if \(g_v41SubmissionPendingFailure\) return;[ \t]*(?://[^\n]*)?\n[ \t]*if \(completed && done\.hadWork\) tsBatchCollect\(\);\n[ \t]*\}`)
	if !async.MatchString(drain) || strings.Count(drain, "tsBatchCollect();") != 1 {
		t.Error("asynchronous checked retire must lose or conditionally collect timestamps after quarantine refusal")
	}
	for name, mutation := range map[string]string{
		"wrong-member":             strings.ReplaceAll(drain, "done.", "done_"),
		"missing-loss":             strings.Replace(drain, "if (!completed || !done.hadWork || g_v41SubmissionPendingFailure) tsBatchLost();", "", 1),
		"missing-refusal":          strings.Replace(drain, "if (g_v41SubmissionPendingFailure) return;", "", 1),
		"unconditional-collection": strings.Replace(drain, "if (completed && done.hadWork) tsBatchCollect();", "tsBatchCollect();", 1),
	} {
		if mutation == drain || async.MatchString(mutation) {
			t.Errorf("asynchronous timestamp contract accepted %s mutation", name)
		}
	}
	if got := strings.Count(src, "tsBatchCollect();"); got != 3 {
		t.Errorf("tsBatchCollect call sites = %d, want 3 (each with a lost-batch branch)", got)
	}
}

// cudaProfileSources returns cuda_kernels.cu and every file it textually
// includes (the cuda/*.cuh modules, tree_attention.cu, routed_expert_gemv.cu),
// with comments stripped, keyed by path relative to this package.
func cudaProfileSources(t *testing.T) map[string]string {
	t.Helper()
	include := regexp.MustCompile(`(?m)^#include "([^"]+)"`)
	comments := regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
	out := map[string]string{}
	var walk func(path string)
	walk = func(path string) {
		if _, seen := out[path]; seen {
			return
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		out[path] = comments.ReplaceAllString(string(raw), "")
		for _, m := range include.FindAllStringSubmatch(string(raw), -1) {
			if m[1] == "cuda_backend.h" {
				continue
			}
			walk(filepath.ToSlash(filepath.Clean(filepath.Join(filepath.Dir(path), m[1]))))
		}
	}
	walk("cuda_kernels.cu")
	return out
}

var (
	cudaTSWrap      = regexp.MustCompile(`FCUDA_TS\("([A-Za-z0-9_]+)",`)
	cudaLaunch      = regexp.MustCompile(`((?:[A-Za-z_][A-Za-z0-9_]*::)*)([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^<>;]*>)?\s*<<<`)
	cudaBLASCall    = regexp.MustCompile(`\b(cublasSgemm|cublasGemmEx|cublasLtMatmul)\(`)
	cudaAsyncCopy   = regexp.MustCompile(`\b(cudaMemcpyAsync|cudaMemsetAsync)\(`)
	cudaSyncCopy    = regexp.MustCompile(`\bcudaMemcpy(?:Peer)?\(`)
	cudaUntimedNote = regexp.MustCompile(`fcuda_ts_untimed_transfer\(\);[^\n]*\n\s*$`)
	cudaFuncStart   = regexp.MustCompile(`(?m)^(?:extern "C" |static )?[A-Za-z_][\w *&:<>,]*?\b(\w+)\([^;]*?\)\s*\{`)
)

// cudaLaunchTimingName preserves semantic profile rows for the exact source
// specializations that intentionally differ from their CUDA function names.
// Keep paths and template arguments explicit: a suffix rule would hide drift.
func cudaLaunchTimingName(path, launch, kernel string) string {
	launch = strings.Join(strings.Fields(strings.TrimSuffix(launch, "<<<")), "")
	aliases := map[string]string{
		"cuda/quant_matmul.cuh:k_f32_to_f16_vec4":         "k_f32_to_f16",
		"cuda/iq4xs_mmvq.cuh:fak_iq4xs::mul_mat_vec_q<1>": "k_iq4xs_mmvq_gemv",
		"cuda/iq4xs_mmvq.cuh:fak_iq4xs::mul_mat_vec_q<2>": "k_iq4xs_mmvq_gemv",
		"cuda/iq4xs_mmvq.cuh:fak_iq4xs::mul_mat_vec_q<3>": "k_iq4xs_mmvq_gemv",
		"cuda/paged_attention_vllm.cuh:kern":              "fak_paged_attention_v1",
		"cuda/paged_attention_vllm.cuh:kern2":             "fak_paged_attention_v2",
		"cuda/paged_attention_vllm.cuh:paged_attention_v2_reduce_kernel<float,HEAD_SIZE,NUM_THREADS,PARTITION_SIZE>": "fak_paged_attention_v2_reduce",
		"cuda/paged_attention_vllm.cuh:fak_vllm_pa::reshape_and_cache_run_kernel":                                    "fak_paged_kv_write",
		"cuda/paged_attention_vllm.cuh:fak_vllm_pa::reshape_and_cache_pages_kernel":                                  "fak_paged_kv_write_pages",
		"cuda/paged_attention_vllm.cuh:fak_vllm_pa::paged_gather_rows_kernel":                                        "fak_paged_kv_gather",
		"cuda/paged_attention_vllm.cuh:fak_paged_kv_copy_blocks_kernel":                                              "fak_paged_kv_copy_blocks",
		"cuda/q3k_mmvq.cuh:fak_q3k_mmvq::mul_mat_vec_q<1>":                                                           "k_q3k_mmvq_gemv",
		"cuda/q3k_mmvq.cuh:fak_q3k_mmvq::mul_mat_vec_q<2>":                                                           "k_q3k_mmvq_gemv",
		"cuda/q3k_mmvq.cuh:fak_q3k_mmvq::mul_mat_vec_q<3>":                                                           "k_q3k_mmvq_gemv",
		"cuda/q3k_mmvq.cuh:dequantize_block_q3_K":                                                                    "k_q3k_dequant",
		"cuda/q4k_mmvq.cuh:mul_mat_vec_q<1>":                                                                         "k_q4k_mmvq_gemv",
		"cuda/q4k_mmvq.cuh:mul_mat_vec_q<2>":                                                                         "k_q4k_mmvq_gemv",
		"cuda/q4k_mmvq.cuh:mul_mat_vec_q<3>":                                                                         "k_q4k_mmvq_gemv",
		"cuda/quant_matmul.cuh:fak_q4k_mmvq::mul_mat_vec_q<1,Dot>":                                                   "k_q56k_mmvq_gemv",
		"cuda/quant_matmul.cuh:fak_q4k_mmvq::mul_mat_vec_q<2,Dot>":                                                   "k_q56k_mmvq_gemv",
		"cuda/quant_matmul.cuh:fak_q4k_mmvq::mul_mat_vec_q<3,Dot>":                                                   "k_q56k_mmvq_gemv",
		"cuda/q8_mmq.cuh:mul_mat_q<J,fallback>":                                                                      "q8_mul_mat_q",
		"cuda/q8_mmq.cuh:mul_mat_q_stream_k_fixup<J,fallback>":                                                       "q8_mul_mat_q_stream_k_fixup",
		"cuda/q4k_shared_input.cuh:mul_mat_vec_q<1>":                                                                 "k_q4k_mmvq_gemv",
		"cuda/graph_quant_workspace.cuh:mul_mat_vec_q<1>":                                                            "k_q4k_mmvq_gemv",
		"cuda/q2k_mmq.cuh:mul_mat_q<J,fallback>":                                                                     "q2_mul_mat_q",
		"cuda/q2k_mmq.cuh:mul_mat_q_stream_k_fixup<J,fallback>":                                                      "q2_mul_mat_q_stream_k_fixup",
		"cuda/q3k_mmq.cuh:mul_mat_q<J,fallback>":                                                                     "q3_mul_mat_q",
		"cuda/q3k_mmq.cuh:mul_mat_q_stream_k_fixup<J,fallback>":                                                      "q3_mul_mat_q_stream_k_fixup",
		"cuda/q56k_mmq.cuh:quantize_mmq_q8_1<kind>":                                                                  "q56k_mmq_quantize",
		"cuda/q56k_mmq.cuh:mul_mat_q<kind,J,fallback>":                                                               "q56k_mmq_mul_mat_q",
		"cuda/q56k_mmq.cuh:mul_mat_q_stream_k_fixup<J,fallback>":                                                     "q56k_mmq_mul_mat_q_stream_k_fixup",
		"cuda/quant_matmul.cuh:k_q5k_gemm<4>":                                                                        "k_q5k_gemm_panel",
		"cuda/quant_matmul.cuh:k_q6k_gemm<4>":                                                                        "k_q6k_gemm_panel",
		"cuda/q4k_fp8lt.cuh:amax<true>":                                                                              "q4k_fp8lt_weight_amax",
		"cuda/q4k_fp8lt.cuh:amax<false>":                                                                             "q4k_fp8lt_activation_amax",
		"cuda/q4k_fp8lt.cuh:finish_scales":                                                                           "q4k_fp8lt_scales",
		"cuda/q4k_fp8lt.cuh:convert<true>":                                                                           "q4k_fp8lt_dequant",
		"cuda/q4k_fp8lt.cuh:convert<false>":                                                                          "q4k_fp8lt_quantize",
	}
	if name, ok := aliases[path+":"+launch]; ok {
		return name
	}
	return kernel
}

func cudaBLASTimingValid(path, call, name string) bool {
	semanticMatmul := path == "cuda/q4k_fp8lt.cuh" && call == "cublasLtMatmul" && name == "q4k_fp8lt_matmul"
	return (strings.HasPrefix(name, "cublas") || semanticMatmul) && KernelOpClass(name) == KernelOpMatmul
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDATimestampSemanticAliases(t *testing.T) {
	for _, tc := range []struct {
		path, call, row string
		want            bool
	}{
		{"cuda/q4k_fp8lt.cuh", "cublasLtMatmul", "q4k_fp8lt_matmul", true},
		{"cuda/other.cuh", "cublasLtMatmul", "q4k_fp8lt_matmul", false},
		{"cuda/q4k_fp8lt.cuh", "cublasSgemm", "q4k_fp8lt_matmul", false},
		{"cuda/q4k_fp8lt.cuh", "cublasLtMatmul", "unknown_matmul", false},
		{"cuda/q4k_fp8lt.cuh", "cublasLtMatmul", "", false},
		{"cuda/quant_matmul.cuh", "cublasGemmEx", "cublas_gemm_ex", true},
	} {
		if got := cudaBLASTimingValid(tc.path, tc.call, tc.row); got != tc.want {
			t.Errorf("BLAS timing (%s, %s, %q) accepted = %t, want %t", tc.path, tc.call, tc.row, got, tc.want)
		}
	}
	for _, tc := range []struct {
		name, path, src string
		want            bool
	}{
		{"f16 vector conversion", "cuda/quant_matmul.cuh", `FCUDA_TS("k_f32_to_f16", grid, k_f32_to_f16_vec4<<<grid, block>>>(x));`, true},
		{"f16 vector wrong source", "cuda/other.cuh", `FCUDA_TS("k_f32_to_f16", grid, k_f32_to_f16_vec4<<<grid, block>>>(x));`, false},
		{"iq4xs one column", "cuda/iq4xs_mmvq.cuh", `FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<1><<<grid, block>>>(x));`, true},
		{"iq4xs two columns", "cuda/iq4xs_mmvq.cuh", `FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<2><<<grid, block>>>(x));`, true},
		{"iq4xs three columns", "cuda/iq4xs_mmvq.cuh", `FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<3><<<grid, block>>>(x));`, true},
		{"iq4xs wrong specialization", "cuda/iq4xs_mmvq.cuh", `FCUDA_TS("k_iq4xs_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<4><<<grid, block>>>(x));`, false},
		{"iq4xs wrong row", "cuda/iq4xs_mmvq.cuh", `FCUDA_TS("k_q4k_mmvq_gemv", grid, fak_iq4xs::mul_mat_vec_q<1><<<grid, block>>>(x));`, false},
		{"iq4xs untimed", "cuda/iq4xs_mmvq.cuh", `fak_iq4xs::mul_mat_vec_q<1><<<grid, block>>>(x);`, false},
		{"paged multi-page writer", "cuda/paged_attention_vllm.cuh", `FCUDA_TS("fak_paged_kv_write_pages", tokens, fak_vllm_pa::reshape_and_cache_pages_kernel<<<tokens, threads>>>(x));`, true},
		{"paged COW copy", "cuda/paged_attention_vllm.cuh", `FCUDA_TS("fak_paged_kv_copy_blocks", planes, fak_paged_kv_copy_blocks_kernel<<<planes, 256, 0, g_stream>>>(x));`, true},
		{"paged COW copy wrong source", "cuda/other.cuh", `FCUDA_TS("fak_paged_kv_copy_blocks", planes, fak_paged_kv_copy_blocks_kernel<<<planes, 256, 0, g_stream>>>(x));`, false},
		{"paged COW copy wrong row", "cuda/paged_attention_vllm.cuh", `FCUDA_TS("fak_paged_kv_write", planes, fak_paged_kv_copy_blocks_kernel<<<planes, 256, 0, g_stream>>>(x));`, false},
		{"paged writer wrong path", "cuda/other.cuh", `FCUDA_TS("fak_paged_kv_write_pages", tokens, fak_vllm_pa::reshape_and_cache_pages_kernel<<<tokens, threads>>>(x));`, false},
		{"paged writer wrong row", "cuda/paged_attention_vllm.cuh", `FCUDA_TS("fak_paged_kv_write", tokens, fak_vllm_pa::reshape_and_cache_pages_kernel<<<tokens, threads>>>(x));`, false},
		{"q8 mul_mat_q", "cuda/q8_mmq.cuh", `FCUDA_TS("q8_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, true},
		{"q8 mul_mat_q_stream_k_fixup", "cuda/q8_mmq.cuh", `FCUDA_TS("q8_mul_mat_q_stream_k_fixup", grid, mul_mat_q_stream_k_fixup<J, fallback><<<grid, block>>>(x));`, true},
		{"q8 wrong source", "cuda/other.cuh", `FCUDA_TS("q8_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, false},
		{"shared q4", "cuda/q4k_shared_input.cuh", `FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block>>>(x));`, true},
		{"shared q4 wrong specialization", "cuda/q4k_shared_input.cuh", `FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<2><<<grid, block>>>(x));`, false},
		{"graph fixed q4", "cuda/graph_quant_workspace.cuh", `FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block>>>(x));`, true},
		{"graph fixed q4 wrong specialization", "cuda/graph_quant_workspace.cuh", `FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<2><<<grid, block>>>(x));`, false},
		{"graph fixed q4 wrong row", "cuda/graph_quant_workspace.cuh", `FCUDA_TS("k_q56k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block>>>(x));`, false},
		{"q2 mul_mat_q", "cuda/q2k_mmq.cuh", `FCUDA_TS("q2_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, true},
		{"q2 mul_mat_q_stream_k_fixup", "cuda/q2k_mmq.cuh", `FCUDA_TS("q2_mul_mat_q_stream_k_fixup", grid, mul_mat_q_stream_k_fixup<J, fallback><<<grid, block>>>(x));`, true},
		{"q2 wrong source", "cuda/other.cuh", `FCUDA_TS("q2_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, false},
		{"q2 wrong specialization", "cuda/q2k_mmq.cuh", `FCUDA_TS("q2_mul_mat_q", grid, mul_mat_q<1, fallback><<<grid, block>>>(x));`, false},
		{"q3 mul_mat_q", "cuda/q3k_mmq.cuh", `FCUDA_TS("q3_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, true},
		{"q3 mul_mat_q_stream_k_fixup", "cuda/q3k_mmq.cuh", `FCUDA_TS("q3_mul_mat_q_stream_k_fixup", grid, mul_mat_q_stream_k_fixup<J, fallback><<<grid, block>>>(x));`, true},
		{"q3 wrong source", "cuda/other.cuh", `FCUDA_TS("q3_mul_mat_q", grid, mul_mat_q<J, fallback><<<grid, block>>>(x));`, false},
		{"q3 wrong specialization", "cuda/q3k_mmq.cuh", `FCUDA_TS("q3_mul_mat_q", grid, mul_mat_q<1, fallback><<<grid, block>>>(x));`, false},
		{"q56k quantize", "cuda/q56k_mmq.cuh", `FCUDA_TS("q56k_mmq_quantize", qgrid, quantize_mmq_q8_1<kind><<<qgrid, block>>>(x));`, true},
		{"q56k mul_mat_q", "cuda/q56k_mmq.cuh", `FCUDA_TS("q56k_mmq_mul_mat_q", grid, mul_mat_q<kind, J, fallback><<<grid, block>>>(x));`, true},
		{"q56k stream_k_fixup", "cuda/q56k_mmq.cuh", `FCUDA_TS("q56k_mmq_mul_mat_q_stream_k_fixup", fgrid, mul_mat_q_stream_k_fixup<J, fallback><<<fgrid, block>>>(x));`, true},
		{"panel", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q5k_gemm_panel", grid, k_q5k_gemm<4><<<grid, block>>>(x));`, true},
		{"panel whitespace", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q6k_gemm_panel", grid, k_q6k_gemm < 4 > <<<grid, block>>>(x));`, true},
		{"ordinary template", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q5k_gemm", grid, k_q5k_gemm<1><<<grid, block>>>(x));`, true},
		{"fp8 conversion", "cuda/q4k_fp8lt.cuh", `FCUDA_TS("q4k_fp8lt_dequant", grid, convert<true><<<grid, block>>>(x));`, true},
		{"wrong specialization", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q5k_gemm_panel", grid, k_q5k_gemm<1><<<grid, block>>>(x));`, false},
		{"wrong source", "cuda/other.cuh", `FCUDA_TS("k_q5k_gemm_panel", grid, k_q5k_gemm<4><<<grid, block>>>(x));`, false},
		{"wrong row", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q6k_gemm_panel", grid, k_q5k_gemm<4><<<grid, block>>>(x));`, false},
		{"unknown alias", "cuda/quant_matmul.cuh", `FCUDA_TS("k_new_panel", grid, k_new<4><<<grid, block>>>(x));`, false},
		{"untimed", "cuda/quant_matmul.cuh", `k_q5k_gemm<4><<<grid, block>>>(x);`, false},
		{"previous wrapper", "cuda/quant_matmul.cuh", `FCUDA_TS("k_q5k_gemm_panel", grid, noop()); k_q5k_gemm<4><<<grid, block>>>(x);`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			matches := cudaLaunch.FindAllStringSubmatchIndex(tc.src, -1)
			if len(matches) != 1 {
				t.Fatalf("launch matches = %d, want 1", len(matches))
			}
			m := matches[0]
			want := cudaLaunchTimingName(tc.path, tc.src[m[0]:m[1]], tc.src[m[4]:m[5]])
			if got := cudaTSEnclosing(tc.src, m[0]) == want; got != tc.want {
				t.Fatalf("timing accepted = %t, want %t (expected row %q)", got, tc.want, want)
			}
		})
	}
}

// cudaTSEnclosing returns the FCUDA_TS name whose variadic body holds at.
// The name and grid arguments run before timing begins. Guard semicolons in
// the body do not end the macro; an unclosed or malformed wrapper proves nothing.
func cudaTSEnclosing(src string, at int) string {
	if at < 0 || at >= len(src) {
		return ""
	}
	var name string
	for _, m := range cudaTSWrap.FindAllStringSubmatchIndex(src[:at], -1) {
		body, end, ok := cudaTSBody(src, m[1])
		if ok && body <= at && at < end {
			name = src[m[2]:m[3]]
		}
	}
	return name
}

// cudaTSBody scans from the first comma to the matching macro close. Source
// comments are already stripped. This is a structural checker, not a C++ lexer.
func cudaTSBody(src string, start int) (body, end int, ok bool) {
	body = -1
	var stack []byte
	var quote byte
	for i := start; i < len(src); i++ {
		ch := src[i]
		if quote != 0 {
			if ch == '\\' {
				i++
			} else if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '(', '[', '{':
			stack = append(stack, ch)
		case ')', ']', '}':
			if len(stack) == 0 {
				return body, i, ch == ')' && body >= 0 && strings.TrimSpace(src[body:i]) != ""
			}
			open := stack[len(stack)-1]
			if !(open == '(' && ch == ')' || open == '[' && ch == ']' || open == '{' && ch == '}') {
				return 0, 0, false
			}
			stack = stack[:len(stack)-1]
		case ',':
			if body < 0 && len(stack) == 0 {
				if strings.TrimSpace(src[start:i]) == "" {
					return 0, 0, false
				}
				body = i + 1
			}
		case ';':
			if body < 0 && len(stack) == 0 {
				return 0, 0, false
			}
		}
	}
	return 0, 0, false
}

func cudaAsyncTimingValid(path, call, row, args string) bool {
	want, valid := cudaAsyncTimingName(call, args)
	if !valid {
		return false
	}
	if row == want {
		return true
	}
	return path == "cuda/sampling_candidates.cuh" &&
		(call == "cudaMemsetAsync" && want == "fill" && row == "sampling_candidates_fill" ||
			call == "cudaMemcpyAsync" && want == "d2d_copy" && row == "sampling_candidates_retain")
}

func cudaEnclosingFunc(src string, at int) string {
	var fn string
	for _, m := range cudaFuncStart.FindAllStringSubmatchIndex(src[:at], -1) {
		fn = src[m[2]:m[3]]
	}
	return fn
}

// cudaAsyncTimingName reads the actual transfer direction, not the enclosing
// filename. Unknown directions and unsupported argument syntax fail closed.
// Input starts after the call's opening parenthesis; source comments are stripped.
func cudaAsyncTimingName(call, src string) (string, bool) {
	const cudaMemsetAsyncArity, cudaMemcpyAsyncArity = 4, 5
	var args []string
	var stack []rune
	start := 0
	appendArg := func(end int) bool {
		arg := strings.TrimSpace(src[start:end])
		if arg == "" {
			return false
		}
		args = append(args, arg)
		return true
	}
	for i, ch := range src {
		switch ch {
		case '\'', '"', ';':
			return "", false
		case '(', '[', '{':
			stack = append(stack, ch)
		case ')', ']', '}':
			if len(stack) > 0 {
				open := stack[len(stack)-1]
				if !(open == '(' && ch == ')' || open == '[' && ch == ']' || open == '{' && ch == '}') {
					return "", false
				}
				stack = stack[:len(stack)-1]
				continue
			}
			if ch != ')' || !appendArg(i) {
				return "", false
			}
			if call == "cudaMemsetAsync" {
				return "fill", len(args) == cudaMemsetAsyncArity
			}
			if call != "cudaMemcpyAsync" || len(args) != cudaMemcpyAsyncArity {
				return "", false
			}
			name, ok := map[string]string{
				"cudaMemcpyDeviceToDevice": "d2d_copy",
				"cudaMemcpyHostToDevice":   "h2d_upload",
				"cudaMemcpyDeviceToHost":   "readback",
			}[args[3]]
			return name, ok
		case ',':
			if len(stack) == 0 {
				if !appendArg(i) {
					return "", false
				}
				start = i + 1
			}
		}
	}
	return "", false
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAAsyncTimestampDirections(t *testing.T) {
	for _, tc := range []struct {
		call, src, want string
		valid           bool
	}{
		{"cudaMemcpyAsync", "dst, src, bytes, cudaMemcpyDeviceToDevice, stream)", "d2d_copy", true},
		{"cudaMemcpyAsync", "(char *)dst + offsets[i], src[i], sizes[i], cudaMemcpyDeviceToHost, stream)", "readback", true},
		{"cudaMemcpyAsync", "dst, src, size(a, b), cudaMemcpyHostToDevice, stream)", "h2d_upload", true},
		{"cudaMemsetAsync", "dst, 0, bytes, stream)", "fill", true},
		{"cudaMemcpyAsync", "dst[), src, bytes, cudaMemcpyDeviceToHost, stream)", "", false},
		{"cudaMemcpyAsync", "dst, src, size(a, bytes, cudaMemcpyDeviceToHost, stream)", "", false},
		{"cudaMemcpyAsync", "dst,,bytes,cudaMemcpyDeviceToHost,stream)", "", false},
		{"cudaMemcpyAsync", "dst, src, bytes, cudaMemcpyDefault, stream)", "", false},
		{"cudaMemcpyAsync", "dst, src, bytes, direction, stream)", "", false},
		{"cudaMemcpyAsync", "dst, src, bytes, cudaMemcpyDeviceToHost)", "", false},
		{"cudaMemcpyAsync", "dst, src, bytes, cudaMemcpyHostToDevice, stream", "", false},
	} {
		got, valid := cudaAsyncTimingName(tc.call, tc.src)
		if got != tc.want || valid != tc.valid {
			t.Fatalf("%s(%s: got %q/%v, want %q/%v", tc.call, tc.src, got, valid, tc.want, tc.valid)
		}
	}
	for _, tc := range []struct {
		src  string
		want bool
	}{
		{`FCUDA_TS("h2d_upload", grid, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));`, true},
		{`FCUDA_TS("readback", grid, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));`, true},
		{`FCUDA_TS("d2d_copy", grid, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));`, false},
		{`cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);`, false},
		{`FCUDA_TS("h2d_upload", grid, noop()); cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);`, false},
	} {
		m := cudaAsyncCopy.FindStringSubmatchIndex(tc.src)
		if m == nil {
			t.Fatal("missing fixture transfer")
		}
		name, valid := cudaAsyncTimingName(tc.src[m[2]:m[3]], tc.src[m[1]:])
		if got := valid && cudaTSEnclosing(tc.src, m[0]) == name; got != tc.want {
			t.Fatalf("transfer timing accepted=%v want=%v: %s", got, tc.want, tc.src)
		}
	}

}

// fak-test:runtime fast est=1ms lane=default
func TestCUDATimestampGuardedBody(t *testing.T) {
	for _, tc := range []struct {
		name, src, want string
	}{
		{"guard", `FCUDA_TS("row", grid, if (closed()) return status(); target(););`, "row"},
		{"nested grid", `FCUDA_TS("row", dim3(f(1, 2), 0), if (closed()) return; target(););`, "row"},
		{"braces", `FCUDA_TS("row", grid, if (ready()) { int x[2] = {1, 2}; target(x[0]); });`, "row"},
		{"quoted delimiters", `FCUDA_TS("row", grid, log(";,])"); char ch = ')'; target(););`, "row"},
		{"escaped quote", `FCUDA_TS("row", grid, log("a\\\"),;"); target(););`, "row"},
		{"variadic comma", `FCUDA_TS("row", grid, target<<<grid, 256, 0, stream>>>(x););`, "row"},
		{"closed wrapper", `FCUDA_TS("row", grid, noop()); target();`, ""},
		{"unwrapped", `target();`, ""},
		{"grid is not timed", `FCUDA_TS("row", target(), noop());`, ""},
		{"nested grid is not timed", `FCUDA_TS("row", dim3(target(), 0), noop());`, ""},
		{"unclosed wrapper", `FCUDA_TS("row", grid, if (ready()) target();`, ""},
		{"mismatched brackets", `FCUDA_TS("row", grid, if (ready(]) target(););`, ""},
		{"unclosed quote", `FCUDA_TS("row", grid, target(); log("unterminated););`, ""},
		{"empty grid", `FCUDA_TS("row", , target());`, ""},
		{"no body", `FCUDA_TS("row", target());`, ""},
		{"latest wrapper", `FCUDA_TS("old", grid, noop()); FCUDA_TS("new", grid, if (ready()) return; target(););`, "new"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			at := strings.Index(tc.src, "target")
			if got := cudaTSEnclosing(tc.src, at); got != tc.want {
				t.Fatalf("row=%q want=%q: %s", got, tc.want, tc.src)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDATimestampSamplingTransferAliases(t *testing.T) {
	const path = "cuda/sampling_candidates.cuh"
	for _, tc := range []struct {
		path, call, row, args string
		want                  bool
	}{
		{path, "cudaMemsetAsync", "sampling_candidates_fill", "dst, 0, bytes, stream)", true},
		{path, "cudaMemcpyAsync", "sampling_candidates_retain", "dst, src, bytes, cudaMemcpyDeviceToDevice, stream)", true},
		{"cuda/other.cuh", "cudaMemsetAsync", "sampling_candidates_fill", "dst, 0, bytes, stream)", false},
		{"cuda/other.cuh", "cudaMemcpyAsync", "sampling_candidates_retain", "dst, src, bytes, cudaMemcpyDeviceToDevice, stream)", false},
		{path, "cudaMemcpyAsync", "sampling_candidates_retain", "dst, src, bytes, cudaMemcpyDeviceToHost, stream)", false},
		{path, "cudaMemcpyAsync", "sampling_candidates_fill", "dst, src, bytes, cudaMemcpyDeviceToDevice, stream)", false},
		{path, "cudaMemsetAsync", "sampling_candidates_retain", "dst, 0, bytes, stream)", false},
		{path, "cudaMemsetAsync", "sampling_candidates_unknown", "dst, 0, bytes, stream)", false},
		{path, "cudaMemcpyAsync", "sampling_candidates_retain", "dst, src, bytes, cudaMemcpyDefault, stream)", false},
		{path, "cudaMemsetAsync", "sampling_candidates_fill", "dst, 0, bytes)", false},
		{path, "cudaMemcpyAsync", "d2d_copy", "dst, src, bytes, cudaMemcpyDeviceToDevice, stream)", true},
	} {
		if got := cudaAsyncTimingValid(tc.path, tc.call, tc.row, tc.args); got != tc.want {
			t.Fatalf("%s %s row=%q valid=%t want=%t", tc.path, tc.call, tc.row, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestKernelOpClassSamplingRows(t *testing.T) {
	for name, want := range map[string]string{
		"sampling_candidates_fill": KernelOpCopy, "sampling_candidates_retain": KernelOpCopy,
		"k_sampling_candidates_histogram": KernelOpOther, "k_sampling_candidates_cutoff": KernelOpOther,
		"k_sampling_candidates_emit": KernelOpOther, "sampling_candidates_unknown": KernelOpOther,
		"sampling_candidates_fill_extra": KernelOpOther, "sampling_candidates_retain_extra": KernelOpOther,
	} {
		if got := KernelOpClass(name); got != want {
			t.Fatalf("KernelOpClass(%q)=%s want=%s", name, got, want)
		}
	}
}

// cudaUntimedTransferSites lists the async copies that run outside the event
// ring: a peer device's copy on the multi-GPU path (the profile keeps one device).
var cudaUntimedTransferSites = map[string]bool{
	"cuda/runtime_services.cuh:fcuda_d2d_on":   true,
	"cuda/async_upload.cuh:fcuda_upload_async": true,
}

// The checked uploader records whether a copy was attempted before draining
// failures. Its one untimed-transfer counter dominates that exact guarded copy;
// the intervening bookkeeping does not make the copy a g_stream timing row.
func cudaAsyncUploadAccountingSite(path, src string, off int) bool {
	if path != "cuda/async_upload.cuh" {
		return false
	}
	const begin = `extern "C" int fcuda_upload_async(`
	const end = `extern "C" int fcuda_upload_async_fence(`
	if strings.Count(src, begin) != 1 || strings.Count(src, end) != 1 {
		return false
	}
	start, stop := strings.Index(src, begin), strings.Index(src, end)
	if stop <= start || off < start || off >= stop {
		return false
	}
	body := src[start:stop]
	if strings.Count(body, "fcuda_ts_untimed_transfer(") != 1 || strings.Count(body, "cudaMemcpyAsync(") != 1 {
		return false
	}
	copy := regexp.MustCompile(`if \(!st\) st = fcuda_async_status\(cudaStreamWaitEvent\(g_copy_stream, g_copy_order, 0\), "order wait"\);\s+fcuda_ts_untimed_transfer\(\);\s+bool copy_attempted = false;\s+if \(!st\) \{\s+copy_attempted = true;\s+st = fcuda_async_status\((cudaMemcpyAsync)\(d, s\.host, n, cudaMemcpyHostToDevice, g_copy_stream\), "memcpy"\);\s+\}`)
	matches := copy.FindAllStringSubmatchIndex(body, -1)
	return len(matches) == 1 && start+matches[0][2] == off
}

// fak-test:runtime fast est=20ms lane=default
func TestCUDAAsyncUploadAccountingSourceContract(t *testing.T) {
	const path = "cuda/async_upload.cuh"
	src := cudaProfileSources(t)[path]
	check := func(s string, want bool) {
		t.Helper()
		off := strings.Index(s, "cudaMemcpyAsync(")
		if got := cudaAsyncUploadAccountingSite(path, s, off); got != want {
			t.Fatalf("checked async upload accounting accepted=%v want=%v", got, want)
		}
	}
	check(src, true)
	for _, mutation := range []struct{ old, new string }{
		{"fcuda_ts_untimed_transfer();", ""},
		{"fcuda_ts_untimed_transfer();", "fcuda_ts_untimed_transfer(); fcuda_ts_untimed_transfer();"},
		{"fcuda_ts_untimed_transfer();", "if (st) fcuda_ts_untimed_transfer();"},
		{"bool copy_attempted = false;", "return 1; bool copy_attempted = false;"},
		{"copy_attempted = true;", "copy_attempted = true; fcuda_async_init();"},
		{"if (!st) {", "if (st) {"},
		{"cudaMemcpyHostToDevice, g_copy_stream", "cudaMemcpyHostToDevice, g_stream"},
	} {
		if !strings.Contains(src, mutation.old) {
			t.Fatalf("mutation absent: %q", mutation.old)
		}
		check(strings.Replace(src, mutation.old, mutation.new, 1), false)
	}
	if cudaAsyncUploadAccountingSite("cuda/other.cuh", src, strings.Index(src, "cudaMemcpyAsync(")) ||
		cudaAsyncUploadAccountingSite(path, src, 0) {
		t.Fatal("foreign path or offset accepted")
	}
}

// These exact bodies are deliberately closed source contracts, not a general
// callback/dataflow exemption. Any changed control flow must be reviewed again.
func cudaTokenomosAccountingSite(srcs map[string]string, path string, off int, sync bool) bool {
	if path != "cuda/tokenomos_staging.h" {
		return false
	}
	normalize := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	region := func(s, begin, end string) string {
		i := strings.Index(s, begin)
		if i < 0 || strings.Count(s, begin) != 1 {
			return ""
		}
		s = s[i:]
		if end != "" {
			j := strings.Index(s, end)
			if j < 0 {
				return ""
			}
			s = s[:j]
		}
		return normalize(s)
	}
	staging := srcs[path]
	if region(staging, "  cudaError_t put(", "\n};") != normalize(tokenomosPutAccountingContract) ||
		region(staging, "static int fak_tokenomos_staging_copy(", "") != normalize(tokenomosCopyAccountingContract) ||
		region(srcs["cuda/model_candidates.cuh"], `extern "C" int fak_qwen35_embedding_gather_f32(`, `extern "C" int fak_qwen35_pointer_is_device`) != normalize(tokenomosCallerAccountingContract) {
		return false
	}
	call := "cudaMemcpyAsync("
	if sync {
		call = "cudaMemcpy("
	}
	return strings.Count(staging, call) == 1 && strings.Index(staging, call) == off
}

const tokenomosPutAccountingContract = `  cudaError_t put(void *device, const void *host, size_t bytes, void (*before_copy)() = nullptr) {
    if (terminal) return first_error;
    if (!open || !device || !host || bytes == 0 || bytes > capacity - at)
      return cudaErrorInvalidValue;
    auto *source = static_cast<unsigned char *>(arenas[current].host) + at;
    std::memcpy(source, host, bytes);

    if (before_copy) before_copy();
    cudaError_t error = cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return fail(error);


    at += ((bytes + alignment - 1) / alignment) * alignment;
    dirty = true;
    return cudaSuccess;
  }`

const tokenomosCopyAccountingContract = `static int fak_tokenomos_staging_copy(void *device, const void *host, size_t bytes, void (*before_copy)() = nullptr) {
  int admitted = fak_tokenomos_staging_admit();
  if (admitted != 0) return admitted;
  if (!device || !host || bytes == 0) return -1;
  if (bytes > fak_tokenomos_staging_pool::capacity) {


    if (before_copy) before_copy();
    cudaError_t error = cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    return error == cudaSuccess ? 0 : fak_tokenomos_staging_error(error);
  }
  cudaError_t error = g_tokenomos_staging.begin(g_stream);
  if (error == cudaSuccess) error = g_tokenomos_staging.put(device, host, bytes, before_copy);
  if (error == cudaSuccess) error = g_tokenomos_staging.seal();
  return error == cudaSuccess ? 0 : fak_tokenomos_staging_error(error);
}
`

const tokenomosCallerAccountingContract = `extern "C" int fak_qwen35_embedding_gather_f32(
    const float *dEmbedding, const int *hIDs, int *dIDs, float *dOut,
    int tokens, int hidden) {
  if (!dEmbedding || !hIDs || !dIDs || !dOut || tokens <= 0 || hidden <= 0) return -1;
  size_t idsBytes = (size_t)tokens * sizeof(int);
#ifdef FAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION

  int staged = fak_tokenomos_staging_copy(dIDs, hIDs, idsBytes, fcuda_ts_untimed_transfer);
  if (staged != 0) return staged;
#else
  cudaGetLastError();
  fcuda_ts_untimed_transfer();
  cudaError_t copy = cudaMemcpy(dIDs, hIDs, idsBytes, cudaMemcpyHostToDevice);
  if (copy != cudaSuccess) return 10000 + (int)copy;
#endif
  g_h2d_bytes += idsBytes;
  size_t n = (size_t)tokens * hidden;
  FCUDA_TS("k_qwen35_embedding_gather", (n + 255) / 256, k_qwen35_embedding_gather<<<(n + 255) / 256, 256, 0, g_stream>>>(
      dEmbedding, dIDs, dOut, tokens, hidden));
  cudaError_t launch = cudaGetLastError();
#ifdef FAK_CUDA_TOKENOMOS_STAGING_QUALIFICATION



  if (launch != cudaSuccess) return fak_tokenomos_staging_error(launch);
#endif
  return launch == cudaSuccess ? 0 : 20000 + (int)launch;
}

`

// fak-test:runtime fast est=10ms lane=default
func TestCUDATokenomosAccountingSourceContract(t *testing.T) {
	srcs := cudaProfileSources(t)
	path := "cuda/tokenomos_staging.h"
	check := func(m map[string]string, want bool) {
		t.Helper()
		for _, call := range []string{"cudaMemcpyAsync(", "cudaMemcpy("} {
			off := strings.Index(m[path], call)
			if got := cudaTokenomosAccountingSite(m, path, off, call == "cudaMemcpy("); got != want {
				t.Fatalf("%s accepted=%v want=%v", call, got, want)
			}
		}
	}
	check(srcs, true)
	for _, mutation := range []struct{ path, old, new string }{
		{path, "void (*before_copy)() = nullptr", "void (*before_copy)()"},
		{path, "if (before_copy) before_copy();", ""},
		{path, "if (before_copy) before_copy();", "if (before_copy) { before_copy(); before_copy(); }"},
		{path, "if (terminal) return first_error;", "if (before_copy) before_copy(); if (terminal) return first_error;"},
		{path, "cudaMemcpyHostToDevice, stream)", "cudaMemcpyHostToDevice, g_stream)"},
		{path, "bytes, before_copy)", "bytes)"},
		{"cuda/model_candidates.cuh", "idsBytes, fcuda_ts_untimed_transfer)", "idsBytes)"},
		{"cuda/model_candidates.cuh", "int staged =", "fcuda_ts_untimed_transfer(); int staged ="},
	} {
		m := make(map[string]string, len(srcs))
		for p, s := range srcs {
			m[p] = s
		}
		// ReplaceAll deliberately detects any extra callback, including an early one.
		if !strings.Contains(m[mutation.path], mutation.old) {
			t.Fatalf("mutation absent: %q", mutation.old)
		}
		m[mutation.path] = strings.ReplaceAll(m[mutation.path], mutation.old, mutation.new)
		check(m, false)
	}
	if cudaTokenomosAccountingSite(srcs, "cuda/other.h", 0, false) || cudaTokenomosAccountingSite(srcs, path, 0, false) {
		t.Fatal("foreign path/offset accepted")
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestCUDATimestampSourceCoversEveryLaunch(t *testing.T) {
	srcs := cudaProfileSources(t)
	for _, want := range []string{"cuda/kernel_profile.cuh", "tree_attention.cu", "routed_expert_gemv.cu", "cuda/q4k_mmq.cuh", "cuda/fattn_mma_hd256.cuh"} {
		if _, ok := srcs[want]; !ok {
			t.Fatalf("cuda_kernels.cu no longer includes %s", want)
		}
	}
	at := func(src string, off int) int { return strings.Count(src[:off], "\n") + 1 }
	launches, blas, async, sync := 0, 0, 0, 0
	for path, src := range srcs {
		if path == "cuda/kernel_profile.cuh" {
			continue
		}
		for _, m := range cudaLaunch.FindAllStringSubmatchIndex(src, -1) {
			launches++
			kernel := src[m[4]:m[5]]
			want := cudaLaunchTimingName(path, src[m[0]:m[1]], kernel)
			if got := cudaTSEnclosing(src, m[0]); got != want {
				t.Errorf("%s:%d launch of %s is timed as %q; wrap it in FCUDA_TS(%q, grid, ...)", path, at(src, m[0]), kernel, got, want)
			}
		}
		for _, m := range cudaBLASCall.FindAllStringSubmatchIndex(src, -1) {
			if strings.Count(src[strings.LastIndex(src[:m[0]], "\n")+1:m[0]], `"`)%2 == 1 {
				continue // a diagnostic string names the call
			}
			blas++
			got := cudaTSEnclosing(src, m[0])
			if !cudaBLASTimingValid(path, src[m[2]:m[3]], got) {
				t.Errorf("%s:%d %s is timed as %q; want a cublas* or exact semantic row that classifies as matmul", path, at(src, m[0]), src[m[2]:m[3]], got)
			}
		}
		for _, m := range cudaAsyncCopy.FindAllStringSubmatchIndex(src, -1) {
			async++
			call, got := src[m[2]:m[3]], cudaTSEnclosing(src, m[0])
			want, valid := cudaAsyncTimingName(call, src[m[1]:])
			if !valid {
				t.Errorf("%s:%d %s has no supported timestamp transfer direction", path, at(src, m[0]), call)
				continue
			}
			if cudaAsyncTimingValid(path, call, got, src[m[1]:]) || cudaTokenomosAccountingSite(srcs, path, m[0], false) {
				continue
			}
			line := src[:strings.LastIndex(src[:m[0]], "\n")+1]
			if got == "" && cudaUntimedTransferSites[path+":"+cudaEnclosingFunc(src, m[0])] &&
				(cudaUntimedNote.MatchString(line) || cudaAsyncUploadAccountingSite(path, src, m[0])) {
				continue
			}
			t.Errorf("%s:%d %s in %s is timed as %q, want %q (or a listed untimed transfer)", path, at(src, m[0]), call, cudaEnclosingFunc(src, m[0]), got, want)
		}
		for _, m := range cudaSyncCopy.FindAllStringIndex(src, -1) {
			sync++
			if !cudaUntimedNote.MatchString(src[:strings.LastIndex(src[:m[0]], "\n")+1]) && !cudaTokenomosAccountingSite(srcs, path, m[0], true) {
				t.Errorf("%s:%d synchronous %s is not counted by fcuda_ts_untimed_transfer()", path, at(src, m[0]), src[m[0]:m[1]-1])
			}
		}
	}
	if launches < 78 || blas < 7 || async < 5 || sync < 19 {
		t.Fatalf("scanned %d launches, %d cuBLAS calls, %d async and %d sync copies; want the full CUDA source", launches, blas, async, sync)
	}

	prof := srcs["cuda/kernel_profile.cuh"]
	for _, want := range []string{
		"getenv(\"FAK_CUDA_TIMESTAMP_PROFILE\");\n  return v && v[0] == '1' && v[1] == '\\0';",
		`getenv("FAK_CUDA_TIMESTAMP_PROFILE_EVERY")`,
		`"fak-cuda ts-profile batches=%llu ops/batch=%.1f gpu_ms/batch=%.3f wall_ms/batch=%.3f truncated=%llu timing=event_pairs graphs=disabled\n"`,
		`"fak-cuda ts-stage kernel=%s groups=%ux%u calls/batch=%.2f ms/batch=%.3f us/call=%.2f share=%.1f%%\n"`,
		"if (!g_ts_on) return -1;",
	} {
		if !strings.Contains(prof, want) {
			t.Errorf("kernel_profile.cuh lost %q", want)
		}
	}
	gosrc, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"if !graphEnabled || cudaKernelProfileOn {", "C.fcuda_ts_boundary()", `RegisterKernelProfileReader("cuda", readCUDAKernelProfile)`} {
		if !strings.Contains(string(gosrc), want) {
			t.Errorf("cuda.go lost %q", want)
		}
	}
}

// cudaKernelOtherAllowlist names the CUDA rows that stay in op class other:
// dtype conversions, dequantization and activation quantization, elementwise
// glue, gathers and status publication.
var cudaKernelOtherAllowlist = map[string]bool{
	// Candidate selection is integer histogram/cutoff/publication, not argmax.
	"k_sampling_candidates_histogram": true, "k_sampling_candidates_cutoff": true, "k_sampling_candidates_emit": true,
	// Graph position publication and indexed row copies match existing glue rows.
	"k_graph_once_position": true, "k_graph_once_copyrow": true,
	"k_kv_store_typed": true, "k_kv_load_typed": true, "k_mxfp8_cast_f32": true,
	// Blockscaled WKV scale-layout repack and finite-output status publication.
	"repack": true, "check_output": true,
	"q4k_fp8lt_weight_amax": true, "q4k_fp8lt_activation_amax": true, "q4k_fp8lt_scales": true,
	"q4k_fp8lt_dequant": true, "q4k_fp8lt_quantize": true,
	"k_add": true, "k_add_bias": true, "k_copyrow": true, "k_f32_to_f16": true, "k_f32_to_f16_T": true,
	"k_fp16_split": true, "k_nvfp4_lt_swizzle_scales": true, "k_q2k_dequant_split": true,
	"k_q4k_coeffs": true, "k_q4k_dequant_split": true, "k_q8_dequant_split": true,
	"k_q8_quant_act": true, "k_qwen35_embedding_gather": true, "k_qwen35_split_qg_panel": true,
	"k_qwen4exp_ple_hash_i64": true, "k_sigmoid_mul": true, "k_split_qwen35_qg": true,
	"k_v41_bf16_publish": true, "k_v41_bf16_publish_status": true, "quantize_mmq_q8_1_ds4": true,
	"k_v41_indexer_score_f32": true, "k_v41_indexer_score_tiled_f32": true,
	"fak_paged_kv_write": true, "fak_paged_kv_write_pages": true, "fak_paged_kv_gather": true, "paged_store_int_kernel": true,
	// V4.1 elementwise, pool-update and status-publish glue.
	"k_mhc": true, "k_v41_engram": true, "k_v41_pool": true, "v41c_status_kernel": true,
	"v41c_compressor_store_kernel": true, "v41c_compressor_update_pool_kernel": true,
	"quantize_q8_1": true, "k_q3k_dequant": true, "k_iq4xs_dequant_split": true,
	"quantize_mmq_q8_1_d2s6": true, "quantize_mmq_q8_1_d4": true, "q56k_mmq_quantize": true,
	// Engram row gathers/dequantization and poisoned-output status publication.
	"v41_eg_q5": true, "v41_eg_q2": true, "v41_eg_f32": true, "v41_eg_rows": true,
	"v41_eg_iq4nl": true, "v41_eg_iq4nl_row": true, "v41_eg_iq4nl_poison": true,
}

// fak-test:runtime fast est=20ms lane=default
func TestKernelOpClassCUDANames(t *testing.T) {
	names := map[string]bool{}
	for _, src := range cudaProfileSources(t) {
		for _, m := range cudaTSWrap.FindAllStringSubmatch(src, -1) {
			names[m[1]] = true
		}
	}
	if len(names) < 75 {
		t.Fatalf("found %d FCUDA_TS row names, want the full CUDA table", len(names))
	}
	for name := range names {
		got := KernelOpClass(name)
		if got == KernelOpOther && !cudaKernelOtherAllowlist[name] {
			t.Errorf("CUDA row %s classifies as other; add a token or list it", name)
		}
		if got != KernelOpOther && cudaKernelOtherAllowlist[name] {
			t.Errorf("CUDA row %s now classifies as %s; drop it from the other allowlist", name, got)
		}
	}
	for name := range cudaKernelOtherAllowlist {
		if !names[name] {
			t.Errorf("allowlisted %s is no longer a CUDA row", name)
		}
	}
	for name, want := range map[string]string{
		"k_q5k_gemm_panel": KernelOpMatmul, "k_q6k_gemm_panel": KernelOpMatmul, "q4k_fp8lt_matmul": KernelOpMatmul,
		"k_q4k_gemm": KernelOpMatmul, "k_q8_gemm_rowtile": KernelOpMatmul, "k_q2k_gemv": KernelOpMatvec, "k_q4k_mmvq_gemv": KernelOpMatvec, "k_q3k_mmvq_gemv": KernelOpMatvec,
		"k_iq4xs_gemv": KernelOpMatvec, "k_iq4xs_mmvq_gemv": KernelOpMatvec,
		"k_awq_gemv": KernelOpMatvec, "k_rex_gemv_mxfp4_e8m0": KernelOpMatvec, "k_rex_gemv_fp8_block32_e8m0": KernelOpMatvec,
		"q8_mul_mat_q": KernelOpMatmul, "q8_mul_mat_q_stream_k_fixup": KernelOpMatmul,
		"q2_mul_mat_q": KernelOpMatmul, "q2_mul_mat_q_stream_k_fixup": KernelOpMatmul,
		"q3_mul_mat_q": KernelOpMatmul, "q3_mul_mat_q_stream_k_fixup": KernelOpMatmul,
		"q56k_mmq_mul_mat_q": KernelOpMatmul, "q56k_mmq_mul_mat_q_stream_k_fixup": KernelOpMatmul,
		"mul_mat_q": KernelOpMatmul, "mul_mat_q_stream_k_fixup": KernelOpMatmul, "k_nvfp4_w4a16_f32": KernelOpMatmul,
		"cublas_sgemm": KernelOpMatmul, "cublas_gemm_ex": KernelOpMatmul, "cublaslt_matmul": KernelOpMatmul,
		"k_qwen38_fattn_mma_hd256": KernelOpAttention, "k_qwen38_fattn_stage_kv_f16": KernelOpAttention,
		"k_qwen38_causal_attention_panel_hd256_splitkv_partial": KernelOpAttention, "k_qwen38_splitkv_lse_merge_hd256": KernelOpAttention,
		"k_dsa_sparse_attend": KernelOpAttention, "k_dsa_index_topk": KernelOpAttention, "k_spec_verify_combine": KernelOpAttention,
		"k_tree_verify_attention": KernelOpAttention, "k_qwen35_gdn_fused_in_proj": KernelOpMatvec,
		"k_qwen35_gdn_recurrent_panel_cols": KernelOpGDN, "d2d_copy": KernelOpCopy, "fill": KernelOpCopy,
	} {
		if !names[name] {
			t.Errorf("%s is not a CUDA row", name)
		}
		if got := KernelOpClass(name); got != want {
			t.Errorf("KernelOpClass(%q) = %s, want %s", name, got, want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestKernelOpClassGraphOnceNames(t *testing.T) {
	source, err := os.ReadFile("cuda/graph_capture_once.cuh")
	if err != nil {
		t.Fatal(err)
	}
	rows := map[string]int{}
	for _, match := range cudaTSWrap.FindAllStringSubmatch(string(source), -1) {
		rows[match[1]]++
	}
	want := map[string]string{
		"k_graph_once_position": KernelOpOther,
		"k_graph_once_copyrow":  KernelOpOther,
		"k_rope":                KernelOpRope,
		"k_partial_rope_copy":   KernelOpRope,
		"k_flash_attention":     KernelOpAttention,
		"d2d_copy":              KernelOpCopy,
	}
	if len(rows) != len(want) {
		t.Fatalf("graph timestamp rows = %v, want exactly six named operations", rows)
	}
	for name, class := range want {
		wantCount := 1
		if name == "k_partial_rope_copy" {
			wantCount = 2 // separate Q and K launches share the same operation class
		}
		if rows[name] != wantCount {
			t.Errorf("graph row %s occurs %d times, want %d", name, rows[name], wantCount)
		}
		if got := KernelOpClass(name); got != class {
			t.Errorf("KernelOpClass(%q) = %s, want %s", name, got, class)
		}
		if cudaKernelOtherAllowlist[name] != (class == KernelOpOther) {
			t.Errorf("graph row %s has inconsistent Other registration", name)
		}
	}
}

var cudaProfileRecycleHost = flag.Bool("fak-cuda-profile-recycle-host", false, "run exact Recycle source against host timing-fault ownership controls; no CUDA arithmetic")

func cudaProfileRecycleMethod(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	start := strings.Index(src, "func (c *cudaBackend) Recycle() {")
	if start < 0 {
		t.Fatal("Recycle source missing")
	}
	end := strings.Index(src[start:], "\n// compactTransientLocked")
	if end < 0 {
		t.Fatal("Recycle source boundary missing")
	}
	return src[start : start+end]
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKernelProfileRecycleBoundaryContract(t *testing.T) {
	body := cudaProfileRecycleMethod(t)
	boundary := strings.Index(body, "C.fcuda_ts_boundary()")
	loop := strings.Index(body, "for _, b := range c.transient")
	if boundary < 0 || loop <= boundary {
		t.Fatal("missing timing boundary before recycling")
	}
	between := body[boundary:loop]
	if !strings.Contains(between, "if cudaQ56KMMVQReleaseBlocked() {\n\t\t\treturn\n\t\t}") {
		t.Fatal("timing boundary can detach roots after a newly terminal stream")
	}
	if strings.Count(body, "C.fcuda_ts_boundary()") != 1 || strings.Count(body, "cudaQ56KMMVQReleaseBlocked()") != 2 {
		t.Fatal("unexpected timing/shared-stop control flow")
	}
	if strings.Index(body, "C.fcuda_free(b.ptr)") < loop || strings.Index(body, "clear(c.transient)") < loop {
		t.Fatal("ownership mutation moved before the release loop")
	}
}

// Uses the complete unchanged production method. Only external native/owner
// calls are substituted. This proves Go control/root behavior, not that a void
// native free always retires or that actual asynchronous CUDA work completed.
// fak-test:runtime medium est=5s lane=optin
func TestCUDAKernelProfileRecycleBoundaryHost(t *testing.T) {
	if !*cudaProfileRecycleHost {
		t.Skip("requires -fak-cuda-profile-recycle-host; host controls only")
	}
	goexe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goexe += ".exe"
	}
	if _, err := os.Stat(goexe); err != nil {
		t.Fatal("required same-toolchain Go executable unavailable:", err)
	}
	path := filepath.Join(t.TempDir(), "main.go")
	program := cudaProfileRecycleFixture + "\n" + cudaProfileRecycleMethod(t) + "\n" + cudaProfileRecycleMain
	if err := os.WriteFile(path, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goexe, "run", path)
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local")
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil || err != nil {
		t.Fatalf("Recycle host controls failed: %v / %v\n%s", err, ctx.Err(), out)
	}
	if !strings.Contains(string(out), "Recycle boundary controls passed: 7") {
		t.Fatalf("missing complete host control receipt: %s", out)
	}
	t.Logf("%s", out)
}

const cudaProfileRecycleFixture = `package main
import("fmt";"runtime";"sync";"time";"unsafe")
type cudaBuf struct{ptr unsafe.Pointer;graphOncePinned bool}
type cudaBackend struct{transient []*cudaBuf}
var cudaMu sync.Mutex
var cudaKernelProfileOn bool
var terminal,queries,boundaries,frees,dependencies,drains,mode int
func cudaQ56KMMVQReleaseBlocked()bool{queries++;return terminal!=0}
func cudaQ4KAsyncReleaseBlockedLocked()bool{return false}
func cudaV41CompressorDrainLocked(*cudaBackend){drains++}
func cudaQ4KAsyncRequireDrainLocked(*cudaBackend){drains++}
func(*cudaBackend)graphOnceFreeDependencyLocked(*cudaBuf){dependencies++}
var C=struct{fcuda_ts_boundary func();fcuda_free func(unsafe.Pointer)}{
 func(){boundaries++;switch mode{case 1:terminal=73;case 2:panic("timing panic");case 3:runtime.Goexit()}},
 func(unsafe.Pointer){frees++;if terminal!=0{return}},
}
func require(ok bool,msg string){if !ok{panic(msg)}}
type outcome struct{returned bool;thrown any}
func invoke(c *cudaBackend)outcome{
 done:=make(chan outcome,1)
 go func(){returned:=false;defer func(){done<-outcome{returned,recover()}}();c.Recycle();returned=true}()
 select{case r:=<-done:return r;case <-time.After(2*time.Second):panic("Recycle unwind did not finish")}
}
`

const cudaProfileRecycleMain = `
func check(profile bool,which,prior int){
 terminal,queries,boundaries,frees,dependencies,drains,mode=prior,0,0,0,0,0,which
 cudaKernelProfileOn=profile
 backing:=[2]byte{1,2}
 ordinary:=&cudaBuf{ptr:unsafe.Pointer(&backing[0])}
 pinned:=&cudaBuf{ptr:unsafe.Pointer(&backing[1]),graphOncePinned:true}
 empty:=&cudaBuf{}
 c:=&cudaBackend{transient:[]*cudaBuf{ordinary,pinned,empty}}
 before:=append([]*cudaBuf(nil),c.transient...)
 capBefore:=cap(c.transient);baseBefore:=&c.transient[0]
 r:=invoke(c)
 require(cudaMu.TryLock(),"mutex stranded by timing fault");cudaMu.Unlock()
 stopped:=prior!=0 || profile&&which!=0
 if stopped{
  require(len(c.transient)==len(before)&&cap(c.transient)==capBefore&&&c.transient[0]==baseBefore,"transient roots/slice changed on stop")
  for i,b:=range before{require(c.transient[i]==b,"transient owner identity changed")}
  require(ordinary.ptr==unsafe.Pointer(&backing[0])&&pinned.ptr==unsafe.Pointer(&backing[1])&&empty.ptr==nil,"buffer pointer changed on stop")
  require(frees==0&&dependencies==0,"native free/dependency release reached after stop")
 }else{
  require(len(c.transient)==0&&ordinary.ptr==nil,"healthy transient not recycled")
  require(pinned.ptr==unsafe.Pointer(&backing[1])&&frees==1&&dependencies==1,"graph-pinned storage freed or wrong free count")
 }
 if prior!=0{require(terminal==prior&&boundaries==0&&drains==0,"preexisting failure overwritten or boundary reached")
 }else if profile{require(boundaries==1&&drains==2,"profile boundary/drain count")}else{require(boundaries==0&&drains==2,"profile-off behavior changed")}
 if profile&&prior==0&&which==1{require(terminal==73&&queries==2,"new timing failure not rechecked")}
 if profile&&prior==0&&which==2{require(!r.returned&&r.thrown=="timing panic","panic replaced or swallowed")
 }else if profile&&prior==0&&which==3{require(!r.returned&&r.thrown==nil,"Goexit replaced or swallowed")
 }else{require(r.returned&&r.thrown==nil,"normal return contract changed")}
 runtime.KeepAlive(backing)
}
func main(){
 check(true,1,0);check(true,0,0);check(false,1,0);check(true,1,19);
 check(true,2,0);check(true,3,0);check(false,0,19)
 fmt.Println("Recycle boundary controls passed: 7")
}
`
