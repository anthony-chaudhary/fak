package gateway

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

var syncTestDevice struct {
	mu sync.Mutex
	w  compute.DeviceWork
}

// syncPlanner stands in for a profile-off decode loop on a backend that
// counts host syncs: 4 steps, 2ms device idle, 8 status flushes and 4 readbacks.
type syncPlanner struct{}

func (syncPlanner) Model() string { return "sync-model" }

func (syncPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	span := nativeperf.StartDecodeSpan()
	time.Sleep(5 * time.Millisecond)
	syncTestDevice.mu.Lock()
	syncTestDevice.w.Dispatches += 40
	syncTestDevice.w.DeviceIdleNS += uint64(2 * time.Millisecond)
	syncTestDevice.w.SyncSites = compute.DeviceSyncSitesAdd(syncTestDevice.w.SyncSites, []compute.DeviceSyncSite{
		{Site: compute.SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 8, WaitNS: uint64(800 * time.Microsecond)},
		{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, Syncs: 4, WaitNS: uint64(200 * time.Microsecond), D2HBytes: 16384},
	})
	syncTestDevice.mu.Unlock()
	span.Finish(ctx, 4)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Model:        "sync-model",
		Usage:        agent.Usage{PromptTokens: 64, CompletionTokens: 5, TotalTokens: 69},
		Timings:      agent.NewTimings(64, 0, 5, 0.05, 0.004),
		NativeDecode: &agent.NativeDecodeSummary{Path: agent.NativeDecodePathSerial},
	}, nil
}

// fak-test:runtime fast est=3s lane=default
func TestDeviceIdleAndSyncSitesReachPerfRow(t *testing.T) {
	syncTestDevice.mu.Lock()
	syncTestDevice.w = compute.DeviceWork{SyncObserved: true}
	syncTestDevice.mu.Unlock()
	t.Cleanup(compute.RegisterDeviceWorkReader("sync-test", func() compute.DeviceWork {
		syncTestDevice.mu.Lock()
		defer syncTestDevice.mu.Unlock()
		w := syncTestDevice.w
		w.SyncSites = append([]compute.DeviceSyncSite(nil), w.SyncSites...)
		return w
	}))
	h := newPerfServedHarness(t, syncPlanner{})
	h.chat("sync-model", "decode", false)

	rep := h.recent()
	if len(rep.Records) != 1 {
		t.Fatalf("rows = %d, want 1", len(rep.Records))
	}
	n := rep.Records[0].NativeTiming
	if n == nil || n.GPUIdleMSPerToken == nil || *n.GPUIdleMSPerToken != 0.5 || n.SyncsPerToken == nil || *n.SyncsPerToken != 3 {
		t.Fatalf("native timing = %+v, want 0.5 gpu idle ms/token and 3 syncs/token", n)
	}
	want := []perfledger.NativeSyncSite{
		{Site: compute.SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", SyncsPerToken: 2, WaitMSPerToken: 0.2},
		{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, SyncsPerToken: 1, WaitMSPerToken: 0.05, D2HBytesPerToken: 4096},
	}
	if !reflect.DeepEqual(n.SyncSites, want) {
		t.Fatalf("sync sites = %+v, want %+v", n.SyncSites, want)
	}
}
