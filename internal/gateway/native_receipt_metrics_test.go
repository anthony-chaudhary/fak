package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestNativeReceiptMetricsRenderOnGatewayMetricsSurface(t *testing.T) {
	srv := newTestServer(t)
	if srv.nativeReceiptMetrics == nil {
		t.Fatal("New did not initialize native receipt metrics")
	}
	srv.nativeReceiptMetrics.Observe(&agent.NativeInferenceReceipt{
		PrefillSeconds: 0.2,
		TTFTSeconds:    0.25,
		DecodeSeconds:  0.4,
		Model:          "Qwen3.8-fixture",
		Engine:         "inkernel",
		Backend:        "cuda",
		ForwardPath:    "qwen-cuda-forward",
		CUDAImmutableWeightUploads: &agent.NativeCUDAImmutableWeightUploadDelta{
			Delta: agent.NativeCUDAImmutableWeightUploadCounters{TransferBytes: 4096},
		},
		Qwen35MetalStateIdentity: &model.Qwen35MetalStateIdentityReceipt{
			Available:        true,
			GDNStateD2HBytes: 128,
			GDNStateH2DBytes: 256,
		},
	}, time.Now())

	out := srv.renderMetrics()
	for _, want := range []string{
		`# TYPE fak_native_receipt_requests_total counter`,
		`fak_native_receipt_requests_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda"} 1`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="queue"} 0.05`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="prefill"} 0.2`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="decode"} 0.4`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",kind="kv"} 384`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",kind="transfer"} 4480`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestNativePhaseAccountingRendersOnGatewayMetricsSurface(t *testing.T) {
	srv := newTestServer(t)
	recorder := nativeperf.NewPhaseRecorder("inkernel", "cuda", "qwen3.8_cuda", 9*time.Millisecond)
	for _, interval := range []struct {
		phase, parent nativeperf.Phase
		start, end    time.Duration
		kind          nativeperf.WorkKind
	}{
		{nativeperf.PhaseQueueAdmission, "", 0, 2 * time.Millisecond, nativeperf.WorkWait},
		{nativeperf.PhaseHostUpload, nativeperf.PhasePrefill, 2 * time.Millisecond, 4 * time.Millisecond, nativeperf.WorkWait},
		{nativeperf.PhasePrefill, "", 2 * time.Millisecond, 6 * time.Millisecond, nativeperf.WorkActive},
		{nativeperf.PhaseKernel, nativeperf.PhasePrefill, 4 * time.Millisecond, 6 * time.Millisecond, nativeperf.WorkActive},
		{nativeperf.PhaseDecode, "", 6 * time.Millisecond, 8 * time.Millisecond, nativeperf.WorkActive},
		{nativeperf.PhaseOutput, "", 8 * time.Millisecond, 9 * time.Millisecond, nativeperf.WorkActive},
	} {
		if err := recorder.Add(interval.phase, interval.parent, interval.start, interval.end, interval.kind); err != nil {
			t.Fatal(err)
		}
	}
	receipt, err := recorder.Finalize(0)
	if err != nil {
		t.Fatal(err)
	}
	if !srv.nativeReceiptMetrics.ObservePhases(receipt, time.Now()) {
		t.Fatal("valid phase receipt rejected")
	}
	out := srv.renderMetrics()
	for _, want := range []string{
		`fak_native_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="queue_admission",kind="wait"} 0.002`,
		`fak_native_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="host_device_upload",kind="wait"} 0.002`,
		`fak_native_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",phase="kernel",kind="active"} 0.002`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}
}
