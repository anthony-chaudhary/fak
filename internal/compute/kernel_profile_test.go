package compute

import (
	"math"
	"os"
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
		if lines := strings.SplitN(window, "\n", 5); len(lines) == 5 {
			window = strings.Join(lines[:4], "\n")
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
	if !regexp.MustCompile(`static inline void dpOneShot\([^)]*\) \{\n\s+tsCountUntimed\(`).MatchString(src) {
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
