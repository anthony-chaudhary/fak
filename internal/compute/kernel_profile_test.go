package compute

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
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
	if len(pseudo) != len(kernelTransferNames) {
		t.Fatalf("shim pseudo-kernel names = %d, Go transfer names = %d", len(pseudo), len(kernelTransferNames))
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
	if got := strings.Count(src, "tsBatchCollect();"); got != 2 {
		t.Errorf("tsBatchCollect call sites = %d, want 2 (each with a lost-batch branch)", got)
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
		"cuda/quant_matmul.cuh:k_q5k_gemm<4>": "k_q5k_gemm_panel",
		"cuda/quant_matmul.cuh:k_q6k_gemm<4>": "k_q6k_gemm_panel",
		"cuda/q4k_fp8lt.cuh:amax<true>":       "q4k_fp8lt_weight_amax",
		"cuda/q4k_fp8lt.cuh:amax<false>":      "q4k_fp8lt_activation_amax",
		"cuda/q4k_fp8lt.cuh:finish_scales":    "q4k_fp8lt_scales",
		"cuda/q4k_fp8lt.cuh:convert<true>":    "q4k_fp8lt_dequant",
		"cuda/q4k_fp8lt.cuh:convert<false>":   "q4k_fp8lt_quantize",
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

// cudaTSEnclosing returns the FCUDA_TS name whose argument list holds offset
// at, or "" when the call sits outside every FCUDA_TS statement.
func cudaTSEnclosing(src string, at int) string {
	var name string
	for _, m := range cudaTSWrap.FindAllStringSubmatchIndex(src[:at], -1) {
		if !strings.Contains(src[m[1]:at], ";") {
			name = src[m[2]:m[3]]
		} else {
			name = ""
		}
	}
	return name
}

func cudaEnclosingFunc(src string, at int) string {
	var fn string
	for _, m := range cudaFuncStart.FindAllStringSubmatchIndex(src[:at], -1) {
		fn = src[m[2]:m[3]]
	}
	return fn
}

// cudaUntimedTransferSites lists the async copies that run outside the event
// ring: a peer device's copy on the multi-GPU path (the profile keeps one device).
var cudaUntimedTransferSites = map[string]bool{
	"cuda/runtime_services.cuh:fcuda_d2d_on":   true,
	"cuda/async_upload.cuh:fcuda_upload_async": true,
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
			want := map[string]string{"cudaMemcpyAsync": "d2d_copy", "cudaMemsetAsync": "fill"}[call]
			if got == want {
				continue
			}
			line := src[:strings.LastIndex(src[:m[0]], "\n")+1]
			if got == "" && cudaUntimedNote.MatchString(line) && cudaUntimedTransferSites[path+":"+cudaEnclosingFunc(src, m[0])] {
				continue
			}
			t.Errorf("%s:%d %s in %s is timed as %q, want %q (or a listed untimed transfer)", path, at(src, m[0]), call, cudaEnclosingFunc(src, m[0]), got, want)
		}
		for _, m := range cudaSyncCopy.FindAllStringIndex(src, -1) {
			sync++
			if !cudaUntimedNote.MatchString(src[:strings.LastIndex(src[:m[0]], "\n")+1]) {
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
	"k_kv_store_typed": true, "k_kv_load_typed": true, "k_mxfp8_cast_f32": true,
	"q4k_fp8lt_weight_amax": true, "q4k_fp8lt_activation_amax": true, "q4k_fp8lt_scales": true,
	"q4k_fp8lt_dequant": true, "q4k_fp8lt_quantize": true,
	"k_add": true, "k_add_bias": true, "k_copyrow": true, "k_f32_to_f16": true, "k_f32_to_f16_T": true,
	"k_fp16_split": true, "k_nvfp4_lt_swizzle_scales": true, "k_q2k_dequant_transient": true,
	"k_q4k_coeffs": true, "k_q4k_dequant_transient": true, "k_q8_dequant_transient": true,
	"k_q8_quant_act": true, "k_qwen35_embedding_gather": true, "k_qwen35_split_qg_panel": true,
	"k_qwen4exp_ple_hash_i64": true, "k_sigmoid_mul": true, "k_split_qwen35_qg": true,
	"k_v41_bf16_publish": true, "k_v41_bf16_publish_status": true, "quantize_mmq_q8_1_ds4": true,
	"quantize_q8_1": true, "k_q3k_dequant": true,
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
		"k_awq_gemv": KernelOpMatvec, "k_rex_gemv_mxfp4_e8m0": KernelOpMatvec, "k_rex_gemv_fp8_block32_e8m0": KernelOpMatvec,
		"mul_mat_q": KernelOpMatmul, "mul_mat_q_stream_k_fixup": KernelOpMatmul, "k_nvfp4_w4a16_f32": KernelOpMatmul,
		"cublas_sgemm": KernelOpMatmul, "cublas_gemm_ex": KernelOpMatmul, "cublaslt_matmul": KernelOpMatmul,
		"k_qwen38_fattn_mma_hd256": KernelOpAttention, "k_qwen38_fattn_stage_kv_f16": KernelOpAttention,
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
