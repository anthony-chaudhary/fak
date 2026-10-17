package gateway

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

var hostWorkTestDispatches, hostWorkTestSubmitWait atomic.Uint64

// hostWorkPlanner stands in for the in-kernel decode loop: it opens a decode
// span on the request context and drives a fake device backend inside it.
type hostWorkPlanner struct{}

func (hostWorkPlanner) Model() string { return "host-work-model" }

func (hostWorkPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	span := nativeperf.StartDecodeSpan()
	// Spin past a Windows process-CPU tick (~15.6ms) so the window measures CPU.
	for deadline := time.Now().Add(40 * time.Millisecond); time.Now().Before(deadline); {
	}
	hostWorkTestDispatches.Add(26)
	hostWorkTestSubmitWait.Add(uint64(time.Millisecond))
	compute.CountHostFallback("host_work_test_stage")
	span.Finish(ctx, 3)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Model:        "host-work-model",
		Usage:        agent.Usage{PromptTokens: 64, CompletionTokens: 4, TotalTokens: 68},
		Timings:      agent.NewTimings(64, 0, 4, 0.05, 0.003),
		NativeDecode: &agent.NativeDecodeSummary{Path: agent.NativeDecodePathSerial},
	}, nil
}

// fak-test:runtime fast est=3s lane=default
func TestDeviceDispatchAndHostShareReachMetricsAndPerfRow(t *testing.T) {
	t.Cleanup(compute.RegisterDeviceWorkReader("host-work-test", func() compute.DeviceWork {
		return compute.DeviceWork{Dispatches: hostWorkTestDispatches.Load(), ReadbackBytes: 1 << 20, SubmitWaitNS: hostWorkTestSubmitWait.Load()}
	}))
	h := newPerfServedHarness(t, hostWorkPlanner{})
	h.chat("host-work-model", "decode", false)

	rep := h.recent()
	if len(rep.Records) != 1 {
		t.Fatalf("rows = %d, want 1", len(rep.Records))
	}
	n := rep.Records[0].NativeTiming
	if n == nil || n.HostShare == nil || n.CPUMSPerToken == nil {
		t.Fatalf("native timing = %+v, want measured host_share and cpu_ms_per_token", n)
	}

	resp, err := http.Get(h.ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	out := string(body)
	for _, want := range []string{
		`fak_compute_device_dispatches_total{backend="host-work-test"} `,
		`fak_compute_host_fallback_total{stage="host_work_test_stage"} `,
		"fak_compute_readback_bytes_total ",
		`phase="kernel",kind="wall"} 0.001` + "\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("/metrics missing %q", want)
		}
	}
}
