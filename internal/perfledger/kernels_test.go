package perfledger

import (
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

func kernelWindowFixture() KernelWindow {
	return KernelWindow{
		Backend: "vulkan", Steps: 10, GPUNS: 1e9, WindowWallNS: 1.25e9, UntimedTransfer: 3,
		Samples: []KernelSample{
			{Kernel: "rmsnorm", Op: "norm", GX: 1, GY: 1, Calls: 40, NS: 1e8},
			{Kernel: "q2k_matvec", Op: "matvec", GX: 5120, GY: 1, Calls: 20, NS: 5e8},
			{Kernel: "rope", Op: "rope", GX: 4, GY: 1, Calls: 10, NS: 4e7},
			{Kernel: "attention", Op: "attention", GX: 32, GY: 1, Calls: 10, NS: 3e8},
			{Kernel: "unused", Op: "other", GX: 1, GY: 1, Calls: 0, NS: 0},
			{Kernel: "d2d_copy", Op: "copy", GX: 0, GY: 0, Calls: 30, NS: 6e7},
		},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsSmallestPrefixReaching95(t *testing.T) {
	got := NewNativeKernels(kernelWindowFixture())
	want := &NativeKernels{
		Source: KernelSourceStructured, Backend: "vulkan", DecodeSteps: 10,
		GPUMSPerToken: 100, WallMSPerToken: 125, DeviceBusy: 0.8, SelectedCoverage: 0.96,
		UntimedTransfer: 3, Complete: true, Pipelining: KernelPipeliningDisabledByProfile,
		SelectedKernels: []SelectedKernel{
			{Kernel: "q2k_matvec", Op: "matvec", Groups: "5120x1", CallsPerToken: 2, GPUMSPerToken: 50, USPerCall: 25000, Share: 0.5},
			{Kernel: "attention", Op: "attention", Groups: "32x1", CallsPerToken: 1, GPUMSPerToken: 30, USPerCall: 30000, Share: 0.3},
			{Kernel: "rmsnorm", Op: "norm", Groups: "1x1", CallsPerToken: 4, GPUMSPerToken: 10, USPerCall: 2500, Share: 0.1},
			{Kernel: "d2d_copy", Op: "copy", Groups: "0x0", CallsPerToken: 3, GPUMSPerToken: 6, USPerCall: 2000, Share: 0.06},
			{Kernel: KernelOther, Op: KernelOther, Groups: "", CallsPerToken: 1, GPUMSPerToken: 4, USPerCall: 4000, Share: 0.04},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("kernels = %+v\nwant      %+v", got, want)
	}
	var sum float64
	for _, k := range got.SelectedKernels {
		sum += k.Share
	}
	if math.Abs(sum-1) > 1e-7 {
		t.Fatalf("shares sum to %v, want 1", sum)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsCapLeavesCoverageShort(t *testing.T) {
	w := KernelWindow{Backend: "vulkan", Steps: 4, GPUNS: 3e7, WindowWallNS: 6e7}
	for i := 0; i < 30; i++ {
		w.Samples = append(w.Samples, KernelSample{Kernel: "k", Op: "other", GX: uint32(i + 1), GY: 1, Calls: 1, NS: 1e6})
	}
	got := NewNativeKernels(w)
	if got == nil || len(got.SelectedKernels) != MaxSelectedKernels+1 {
		t.Fatalf("selection = %+v, want %d kernels plus other", got, MaxSelectedKernels)
	}
	if got.SelectedCoverage != 0.8 || got.SelectedCoverage >= SelectedCoverageTarget {
		t.Fatalf("coverage = %v, want 0.8 below the target", got.SelectedCoverage)
	}
	if other := got.SelectedKernels[MaxSelectedKernels]; other.Kernel != KernelOther || other.Share != 0.2 {
		t.Fatalf("other row = %+v, want the six folded shapes at 0.2", other)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsRefusesUnmeasured(t *testing.T) {
	for name, mutate := range map[string]func(w *KernelWindow){
		"steps_zero":     func(w *KernelWindow) { w.Steps = 0 },
		"steps_negative": func(w *KernelWindow) { w.Steps = -1 },
		"gpu_zero":       func(w *KernelWindow) { w.GPUNS = 0 },
		"gpu_nan":        func(w *KernelWindow) { w.GPUNS = math.NaN() },
		"gpu_inf":        func(w *KernelWindow) { w.GPUNS = math.Inf(1) },
		"wall_zero":      func(w *KernelWindow) { w.WindowWallNS = 0 },
		"sample_nan":     func(w *KernelWindow) { w.Samples[0].NS = math.NaN() },
		"sample_inf":     func(w *KernelWindow) { w.Samples[1].NS = math.Inf(1) },
		"sample_neg":     func(w *KernelWindow) { w.Samples[2].NS = -1 },
		"no_calls": func(w *KernelWindow) {
			for i := range w.Samples {
				w.Samples[i].Calls = 0
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			w := kernelWindowFixture()
			mutate(&w)
			if got := NewNativeKernels(w); got != nil {
				t.Fatalf("kernels = %+v, want nil", got)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNewNativeKernelsCompleteGatesComputeOnly(t *testing.T) {
	for _, tc := range []struct {
		name     string
		mutate   func(w *KernelWindow)
		complete bool
	}{
		{"clean", func(*KernelWindow) {}, true},
		{"untimed_transfer_reported_only", func(w *KernelWindow) { w.UntimedTransfer = 9 }, true},
		{"untimed_compute", func(w *KernelWindow) { w.UntimedCompute = 1 }, false},
		{"truncated", func(w *KernelWindow) { w.Truncated = 1 }, false},
		{"barriers_elided", func(w *KernelWindow) { w.BarriersElided = 1 }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := kernelWindowFixture()
			tc.mutate(&w)
			if got := NewNativeKernels(w); got == nil || got.Complete != tc.complete {
				t.Fatalf("kernels = %+v, want complete=%v", got, tc.complete)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeKernelsWireKeysClosed(t *testing.T) {
	n := NewNativeTiming(448, 128, 850, 11760).WithKernels(NewNativeKernels(kernelWindowFixture()))
	raw, err := json.Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Kernels map[string]json.RawMessage `json:"kernels"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(wire.Kernels))
	for k := range wire.Kernels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	want := []string{"backend", "barriers_elided", "complete", "decode_steps", "device_busy", "gpu_ms_per_token", "pipelining",
		"selected_coverage", "selected_kernels", "source", "truncated", "untimed_compute", "untimed_transfer", "wall_ms_per_token"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("kernels keys = %v\nwant %v", keys, want)
	}
	var back NativeTiming
	if err := json.Unmarshal(raw, &back); err != nil || !reflect.DeepEqual(back.Kernels, n.Kernels) {
		t.Fatalf("round trip = %+v err=%v", back.Kernels, err)
	}
	if same := n.WithKernels(nil); same != n {
		t.Fatal("WithKernels(nil) must leave the timing unchanged")
	}
}

const goldenRecordWithoutKernels = `{"request_id":"req-1","trace_id":"gw-1","schema":"fak.gateway.perf-record.v1","unix_ms":1792200000000,"finish_reason":"stop","locality":"local","prompt_tokens":512,"completion_tokens":128,"cached_tokens":64,"cache_regime":"partial","e2e_ms":12000,"ttft_ms":900,"prefill_tps":568.89,"decode_tps":11.44,"native_timing":{"prefill_ms":850,"decode_ms":11760,"prefill_tokens":448,"decode_tokens":128,"prefill_tps":527.06,"decode_tps":10.8,"cpu_ms_per_token":16.7,"host_share":0.13},"model":"Qwen3.8-27B-UD-Q2_K_XL","client":"claude-code","gpu_busy_percent":77}`

const goldenCompactWithoutKernels = `PERF n=2 ttft p50=400ms p99=900ms (measured 2/2) | prefill p50=568.9 tok/s weighted=568.9 tok/s | decode p50=11.4 tok/s | e2e p50=3.00s p99=12.00s | cache=10.0% | gpu busy p50=77% (measured 1/2) | native phases=2 prefill p50=380ms/527.1 tok/s weighted=527.1 tok/s (measured 2/2) decode p50=2.87s/10.8 tok/s weighted=10.8 tok/s (measured 2/2) host share p50=0.130 cpu p50=16.70ms/tok (measured 1/2) | ttft p50 by regime: partial=900ms(n=1) cold=400ms(n=1) | by client: unknown=1 H=0%(c1) claude-code=1 H=11%(p1) | served by model=Qwen3.8-27B-UD-Q2_K_XL | slowest e2e=12.00s request_id=req-1`

const goldenSummaryWithoutKernels = `{"count":2,"ttft_measured":2,"ttft_p50_ms":400,"ttft_p99_ms":900,"prefill_tps_p50":568.89,"prefill_tps_weighted":568.89,"decode_tps_p50":11.44,"e2e_p50_ms":3000,"e2e_p99_ms":12000,"cache_hit_share":0.1,"native_timing":{"count":2,"prefill_measured":2,"decode_measured":2,"prefill_p50_ms":380,"decode_p50_ms":2870,"prefill_tps_p50":527.06,"decode_tps_p50":10.8,"prefill_tps_weighted":527.06,"decode_tps_weighted":10.8,"host_work_measured":1,"cpu_ms_per_token_p50":16.7,"host_share_p50":0.13},"errors":0,"by_regime":{"cold":{"count":1,"ttft_measured":1,"ttft_p50_ms":400,"ttft_p99_ms":400,"e2e_p50_ms":3000},"partial":{"count":1,"ttft_measured":1,"ttft_p50_ms":900,"ttft_p99_ms":900,"e2e_p50_ms":12000}},"by_client":{"claude-code":{"count":1,"cache_hit_share":0.1111,"by_regime":{"partial":1}},"unknown":{"count":1,"cache_hit_share":0,"by_regime":{"cold":1}}},"served_by":{"model":"Qwen3.8-27B-UD-Q2_K_XL"},"identities":1,"gpu_busy_measured":1,"gpu_busy_p50":77,"slowest":{"request_id":"req-1","e2e_ms":12000}}`

func goldenPerfRecords() []Record {
	busy := 77
	a := NewRecord(time.UnixMilli(1792200000000), "stop", "local", 512, 128, 64, 12*time.Second, 900*time.Millisecond)
	a.RequestID, a.TraceID, a.Model, a.Client = "req-1", "gw-1", "Qwen3.8-27B-UD-Q2_K_XL", "claude-code"
	a.GPUBusyPercent = &busy
	a.NativeTiming = NewNativeTiming(448, 128, 850, 11760).WithHostWork(16.7, true, 0.13, true)
	b := NewRecord(time.UnixMilli(1792200005000), "stop", "local", 64, 32, 0, 3*time.Second, 400*time.Millisecond)
	b.RequestID, b.Model = "req-2", "Qwen3.8-27B-UD-Q2_K_XL"
	b.NativeTiming = NewNativeTiming(64, 32, 380, 2870)
	return []Record{a, b}
}

// The goldens were written by the code before native kernels existed: an
// unprofiled row, summary and compact line must not change by one byte.
// fak-test:runtime fast est=10ms lane=default
func TestRecordJSONByteIdenticalWithoutKernels(t *testing.T) {
	var r Record
	if err := json.Unmarshal([]byte(goldenRecordWithoutKernels), &r); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != goldenRecordWithoutKernels {
		t.Fatalf("re-marshal changed the row:\n got %s\nwant %s", raw, goldenRecordWithoutKernels)
	}
	if fresh, _ := json.Marshal(goldenPerfRecords()[0]); string(fresh) != goldenRecordWithoutKernels {
		t.Fatalf("an unprofiled row changed:\n got %s\nwant %s", fresh, goldenRecordWithoutKernels)
	}
	if strings.Contains(string(raw), `"kernels"`) || r.Schema != "fak.gateway.perf-record.v1" || Schema != "fak.gateway.perf-record.v1" {
		t.Fatalf("schema=%q, kernels key present=%v", r.Schema, strings.Contains(string(raw), `"kernels"`))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRenderCompactGoldenWithoutKernels(t *testing.T) {
	rep := BuildReport(goldenPerfRecords(), 2, false, 0)
	if got := RenderCompact(rep); got != goldenCompactWithoutKernels {
		t.Fatalf("compact line changed:\n got %s\nwant %s", got, goldenCompactWithoutKernels)
	}
	raw, err := json.Marshal(rep.Summary)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != goldenSummaryWithoutKernels {
		t.Fatalf("summary changed:\n got %s\nwant %s", raw, goldenSummaryWithoutKernels)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestRenderCompactKernelSegment(t *testing.T) {
	recs := goldenPerfRecords()
	recs[0].NativeTiming = recs[0].NativeTiming.WithKernels(NewNativeKernels(kernelWindowFixture()))
	rep := BuildReport(recs, 2, false, 0)
	n := rep.Summary.NativeTiming
	if n == nil || n.KernelsMeasured != 1 || n.KernelsComplete != 1 || n.GPUMSPerTokenP50 != 100 || n.CoverageP50 != 0.96 || n.DeviceBusyP50 != 0.8 {
		t.Fatalf("summary = %+v, want one complete kernel row", n)
	}
	wantTop := []KernelShare{{Kernel: "q2k_matvec", Share: 0.5}, {Kernel: "attention", Share: 0.3}, {Kernel: "rmsnorm", Share: 0.1}}
	if !reflect.DeepEqual(n.TopKernels, wantTop) {
		t.Fatalf("top kernels = %+v, want %+v", n.TopKernels, wantTop)
	}
	const segment = " host share p50=0.130 cpu p50=16.70ms/tok (measured 1/2) kern gpu p50=100.00ms/tok cov=96% dev_busy=80% top=q2k_matvec:50%,attention:30%,rmsnorm:10% (measured 1/2 complete 1) |"
	if got := RenderCompact(rep); !strings.Contains(got, segment) || strings.Contains(got, "gpu busy p50=80%") {
		t.Fatalf("compact line missing the kernel segment:\n%s", got)
	}
}
