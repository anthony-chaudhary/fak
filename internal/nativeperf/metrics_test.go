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
		`fak_native_runtime_info{engine="inkernel",backend="metal",forward_path="qwen_metal",model="qwen3.8",planner="inkernel",owner="fak"} 1`,
		`fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal"} 1`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",phase="queue"} 0.2`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",phase="prefill"} 0.3`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",phase="decode"} 0.7`,
		`fak_native_receipt_phase_seconds_total{engine="inkernel",backend="metal",forward_path="qwen_metal",phase="kernel"} 0.025`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="metal",forward_path="qwen_metal",kind="kv"} 300`,
		`fak_native_receipt_bytes_total{engine="inkernel",backend="metal",forward_path="qwen_metal",kind="transfer"} 1000`,
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
	want := `fak_native_receipt_requests_total{engine="inkernel",backend="metal",forward_path="qwen_metal"} 400`
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
