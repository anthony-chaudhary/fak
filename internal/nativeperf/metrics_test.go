package nativeperf

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestReceiptMetricsProjectsNativeReceipt(t *testing.T) {
	now := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	m := NewReceiptMetrics(time.Minute)
	if ok := m.Observe(fixtureNativeReceipt(), now); !ok {
		t.Fatal("Observe rejected supported fak-native receipt")
	}
	out := m.Prometheus(now.Add(10 * time.Second))
	for _, want := range []string{
		`fak_native_runtime_info{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",model="qwen3.8",planner="inkernel",owner="fak"} 1`,
		`fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device"} 1`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",phase="queue"} 0.2`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",phase="prefill"} 0.3`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",phase="decode"} 0.7`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",phase="kernel"} 0.025`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",kind="kv"} 300`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",kind="transfer"} 1000`,
		`fak_native_receipt_signal_supported{signal="kernel"} 1`,
		`fak_native_receipt_latest_age_seconds 10`,
		`fak_native_receipt_latest_stale 0`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestReceiptMetricsResetStaleUnsupportedAndLabelBudget(t *testing.T) {
	now := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	m := NewReceiptMetrics(time.Minute)
	absent := m.Prometheus(now)
	for _, want := range []string{
		`fak_native_receipt_latest_stale 1`,
		`fak_native_receipt_signal_supported{signal="queue"} 0`,
		`fak_native_receipt_signal_supported{signal="kernel"} 0`,
	} {
		if !strings.Contains(absent, want) {
			t.Fatalf("absent metrics missing %q\n%s", want, absent)
		}
	}

	unsupported := fixtureNativeReceipt()
	unsupported.Engine = "llama.cpp"
	unsupported.FallbackActive = true
	if ok := m.Observe(unsupported, now); ok {
		t.Fatal("fallback receipt accepted as fak-native evidence")
	}
	out := m.Prometheus(now.Add(2 * time.Minute))
	if !strings.Contains(out, `fak_native_receipt_unsupported_total 1`) || !strings.Contains(out, `fak_native_receipt_latest_stale 1`) {
		t.Fatalf("unsupported/stale metrics missing:\n%s", out)
	}
	if strings.Contains(out, `engine="llama.cpp"`) {
		t.Fatalf("fallback engine leaked into bounded labels:\n%s", out)
	}

	for i := 0; i < 100; i++ {
		r := fixtureNativeReceipt()
		r.Model = "request-or-artifact-" + strings.Repeat("x", i)
		r.Backend = "private-backend-" + strings.Repeat("y", i)
		r.ForwardPath = "/private/artifacts/" + strings.Repeat("z", i)
		m.Observe(r, now)
	}
	out = m.Prometheus(now)
	if got := strings.Count(out, "fak_native_receipt_requests_total{"); got != 1 {
		t.Fatalf("request series = %d, want 1 bounded series\n%s", got, out)
	}
	for _, forbidden := range []string{"request-or-artifact", "private-backend", "/private/artifacts/"} {
		if strings.Contains(out, forbidden) {
			t.Fatalf("unbounded receipt value %q leaked into metrics", forbidden)
		}
	}

	m.Reset()
	reset := m.Prometheus(now)
	if strings.Contains(reset, "fak_native_receipt_requests_total{") || !strings.Contains(reset, "fak_native_receipt_unsupported_total 0") || !strings.Contains(reset, "fak_native_receipt_latest_stale 1") {
		t.Fatalf("reset did not clear receipt state:\n%s", reset)
	}
}

func TestReceiptMetricsConcurrentObserveAndScrape(t *testing.T) {
	m := NewReceiptMetrics(time.Minute)
	now := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	const writers = 8
	const perWriter = 50
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < perWriter; j++ {
				m.Observe(fixtureNativeReceipt(), now)
				_ = m.Prometheus(now)
			}
		}()
	}
	wg.Wait()
	out := m.Prometheus(now)
	want := `fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device"} 400`
	if !strings.Contains(out, want) {
		t.Fatalf("concurrent total missing %q\n%s", want, out)
	}
}

func fixtureNativeReceipt() *model.NativeInferenceReceipt {
	return &model.NativeInferenceReceipt{
		TokenIDs:       []int{1, 2, 3},
		TokenLogprobs:  []float64{-0.1, -0.2, -0.3},
		PrefillSeconds: 0.3,
		TTFTSeconds:    0.5,
		DecodeSeconds:  0.7,
		Model:          "Qwen3.8-fixture",
		Engine:         "inkernel",
		Planner:        "inkernel",
		Owner:          "fak",
		Backend:        "metal",
		ForwardPath:    "qwen35-metal-gdn-preprojected-sequence",
		Q4K:            true,
		Qwen35MetalForwardSequence: &model.Qwen35MetalForwardSequenceReceipt{
			Available:         true,
			TimingAvailable:   true,
			GPUMilliseconds:   25,
			HostUploadBytes:   400,
			HostReadbackBytes: 300,
		},
		Qwen35MetalStateIdentity: &model.Qwen35MetalStateIdentityReceipt{
			Available:        true,
			GDNStateD2HBytes: 100,
			GDNStateH2DBytes: 200,
		},
	}
}

func TestReceiptMetricsSeparatesEvidenceHistory(t *testing.T) {
	now := time.Date(2026, 9, 7, 12, 0, 0, 0, time.UTC)
	m := NewReceiptMetrics(time.Minute)

	// Device evidence receipt: real Metal execution with hardware forward sequence.
	// Deliberately give it a model name containing "synthetic" to prove provenance is not inferred from model name.
	rDevice := fixtureNativeReceipt()
	rDevice.Model = "synthetic-override-ignored"
	if ok := m.Observe(rDevice, now); !ok {
		t.Fatal("Observe rejected device receipt")
	}

	// Synthetic evidence receipt: same engine, backend, and forward path, but lacking device execution evidence.
	// Model name looks like a real production model, proving provenance is inspected rather than inferring from model name.
	rSynthetic := fixtureNativeReceipt()
	rSynthetic.Model = "qwen3.8-production-honest"
	rSynthetic.Qwen35MetalForwardSequence = nil
	rSynthetic.Qwen35MetalStateIdentity = nil
	if ok := m.Observe(rSynthetic, now); !ok {
		t.Fatal("Observe rejected synthetic receipt")
	}

	// Measured evidence receipt: CPU execution.
	rMeasured := fixtureNativeReceipt()
	rMeasured.Backend = "cpu"
	rMeasured.ForwardPath = "cpu/reference"
	rMeasured.Qwen35MetalForwardSequence = nil
	rMeasured.Qwen35MetalStateIdentity = nil
	if ok := m.Observe(rMeasured, now); !ok {
		t.Fatal("Observe rejected measured receipt")
	}

	out := m.Prometheus(now)

	// Verify that history is partitioned into distinct time series by evidence_class
	// even when engine, backend, and forward_path are identical.
	wantDeviceReq := `fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device"} 1`
	wantSyntheticReq := `fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="synthetic"} 1`
	wantMeasuredReq := `fak_native_receipt_requests_total{engine="inkernel",backend="cpu",forward_path="other",evidence_class="measured"} 1`

	for _, want := range []string{wantDeviceReq, wantSyntheticReq, wantMeasuredReq} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing expected partitioned series %q\n%s", want, out)
		}
	}

	// Observe device receipt again and verify counts accumulate independently without pooling into synthetic history.
	if ok := m.Observe(rDevice, now); !ok {
		t.Fatal("second Observe rejected device receipt")
	}
	out = m.Prometheus(now)
	wantDeviceAccum := `fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device"} 2`
	if !strings.Contains(out, wantDeviceAccum) {
		t.Fatalf("device requests did not accumulate to 2: missing %q\n%s", wantDeviceAccum, out)
	}
	if !strings.Contains(out, wantSyntheticReq) {
		t.Fatalf("synthetic requests leaked or pooled with device: missing %q\n%s", wantSyntheticReq, out)
	}

	// Verify phase durations are also partitioned by evidence_class.
	wantDevicePrefill := `fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="device",phase="prefill"} 0.6`
	wantSyntheticPrefill := `fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",evidence_class="synthetic",phase="prefill"} 0.3`
	if !strings.Contains(out, wantDevicePrefill) || !strings.Contains(out, wantSyntheticPrefill) {
		t.Fatalf("phase seconds not partitioned by evidence_class:\nwant device: %s\nwant synthetic: %s\n%s", wantDevicePrefill, wantSyntheticPrefill, out)
	}

	// Verify phase accounting from ObservePhases also partitions across evidence classes.
	recorderDev := NewPhaseRecorder("inkernel", "cuda", "qwen_cuda", 10*time.Millisecond)
	_ = recorderDev.Add(PhaseKernel, "", 0, 5*time.Millisecond, WorkActive)
	_ = recorderDev.Add(PhaseDecode, "", 5*time.Millisecond, 10*time.Millisecond, WorkActive)
	pDev, err := recorderDev.Finalize(0)
	if err != nil {
		t.Fatalf("finalize dev: %v", err)
	}
	if ok := m.ObservePhases(pDev, now); !ok {
		t.Fatal("ObservePhases rejected device phase receipt")
	}

	recorderSynth := NewPhaseRecorder("inkernel", "cuda", "synthetic", 10*time.Millisecond)
	_ = recorderSynth.Add(PhaseDecode, "", 0, 10*time.Millisecond, WorkActive)
	pSynth, err := recorderSynth.Finalize(0)
	if err != nil {
		t.Fatalf("finalize synth: %v", err)
	}
	if ok := m.ObservePhases(pSynth, now); !ok {
		t.Fatal("ObservePhases rejected synthetic phase receipt")
	}

	out = m.Prometheus(now)
	wantDevicePhase := `fak_native_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="qwen_cuda",evidence_class="device",phase="kernel",kind="active"} 0.005`
	wantSynthPhase := `fak_native_phase_seconds_total{engine="inkernel",backend="cuda",forward_path="synthetic",evidence_class="synthetic",phase="decode",kind="active"} 0.01`
	if !strings.Contains(out, wantDevicePhase) || !strings.Contains(out, wantSynthPhase) {
		t.Fatalf("phase accounting not partitioned by evidence_class:\nwant dev: %s\nwant synth: %s\n%s", wantDevicePhase, wantSynthPhase, out)
	}
}
