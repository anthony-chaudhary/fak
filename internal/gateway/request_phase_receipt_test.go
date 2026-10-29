package gateway

import (
	"bytes"
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

var phaseTestDispatches, phaseTestSubmitWait atomic.Uint64

// phasePlanner stands in for a native turn with known engine phases: 8 ms
// prefill and 9 ms decode reported on the planner clock, inside ~20 ms of real
// work, and a decode window whose device submit-wait (kernel) is 2 ms.
type phasePlanner struct{}

func (phasePlanner) Model() string { return "phase-model" }

func (phasePlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(10 * time.Millisecond)
	span := nativeperf.StartDecodeSpan()
	time.Sleep(10 * time.Millisecond)
	phaseTestDispatches.Add(4)
	phaseTestSubmitWait.Add(uint64(2 * time.Millisecond))
	span.Finish(ctx, 4)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Model:        "phase-model",
		Usage:        agent.Usage{PromptTokens: 64, CompletionTokens: 5, TotalTokens: 69},
		Timings:      agent.NewTimings(64, 0, 5, 0.008, 0.009),
		NativeDecode: &agent.NativeDecodeSummary{Path: agent.NativeDecodePathSerial},
	}, nil
}

func getPerfPhases(t *testing.T, srv *Server) *httptest.ResponseRecorder {
	t.Helper()
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/perf/phases", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("phases status = %d: %s", rr.Code, rr.Body.String())
	}
	return rr
}

// fak-test:runtime fast est=2s lane=default
func TestRequestPhaseReceiptJoinsPerfRowAndReducesNative(t *testing.T) {
	t.Cleanup(compute.RegisterDeviceWorkReader("phase-test", func() compute.DeviceWork {
		return compute.DeviceWork{Dispatches: phaseTestDispatches.Load(), SubmitWaitNS: phaseTestSubmitWait.Load(), SubmitWaitObserved: true}
	}))
	h := newPerfServedHarness(t, phasePlanner{})
	h.chat("phase-model", "decode", false)

	rep := h.recent()
	if len(rep.Records) != 1 {
		t.Fatalf("rows = %d, want 1", len(rep.Records))
	}
	row := rep.Records[0]
	rr := getPerfPhases(t, h.srv)
	if got := rr.Header().Get("X-Fak-Phases-Refused"); got != "0" {
		t.Fatalf("refused = %q, want 0", got)
	}
	receipts, err := modelperfobs.ReadRequestPhaseReceipts(bytes.NewReader(rr.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 {
		t.Fatalf("phase receipts = %d, want 1", len(receipts))
	}
	env := receipts[0]
	if env.RequestID != row.RequestID || env.Model != row.Model {
		t.Fatalf("receipt identity %q/%q does not join row %q/%q", env.RequestID, env.Model, row.RequestID, row.Model)
	}
	if env.Receipt.Wall != 17*time.Millisecond {
		t.Fatalf("receipt wall = %s, want prefill+decode 17ms", env.Receipt.Wall)
	}
	if env.UnattributedMS == nil || math.Abs(*env.UnattributedMS+17-row.E2EMS) > 1e-3 {
		t.Fatalf("unattributed = %v, want e2e %g - 17", env.UnattributedMS, row.E2EMS)
	}
	exclusive := map[nativeperf.Phase]time.Duration{}
	for _, p := range env.Receipt.Phases {
		exclusive[p.Phase] = p.Exclusive.Wall
	}
	if exclusive[nativeperf.PhasePrefill] != 8*time.Millisecond || exclusive[nativeperf.PhaseKernel] != 2*time.Millisecond || exclusive[nativeperf.PhaseDecode] != 7*time.Millisecond {
		t.Fatalf("exclusive prefill/kernel/decode = %s/%s/%s, want 8ms/2ms/7ms", exclusive[nativeperf.PhasePrefill], exclusive[nativeperf.PhaseKernel], exclusive[nativeperf.PhaseDecode])
	}

	obs := modelperfobs.Observation{Schema: modelperfobs.Schema, RequestID: row.RequestID, Model: row.Model, Backend: env.Backend, Status: http.StatusOK, DurationMS: row.E2EMS}
	burndown, err := modelperfobs.ReduceTimeBurndown([]modelperfobs.Observation{obs}, receipts)
	if err != nil {
		t.Fatal(err)
	}
	if len(burndown.Groups) != 1 {
		t.Fatalf("groups = %d, want 1", len(burndown.Groups))
	}
	g := burndown.Groups[0]
	if g.Attribution != "native" || g.NativeRequests != 1 {
		t.Fatalf("attribution = %s native=%d, want native/1", g.Attribution, g.NativeRequests)
	}
	var sum float64
	for _, p := range g.NativeRank {
		sum += p.WallMS
	}
	if math.Abs(sum-g.NativeWallMS) > 1e-9 || g.NativeWallMS != 17 {
		t.Fatalf("native phases sum %g, native wall %g, want 17", sum, g.NativeWallMS)
	}
	if math.Abs(g.OutsideNativeMS-*env.UnattributedMS) > 1e-3 {
		t.Fatalf("outside native %g != unattributed %g", g.OutsideNativeMS, *env.UnattributedMS)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestRequestPhaseReceiptRefusesInvalidNativeRows(t *testing.T) {
	base := perfledger.Record{RequestID: "r1", Model: "m", E2EMS: 30, Engine: &perfledger.Engine{}, NativeTiming: &perfledger.NativeTiming{PrefillMS: 10, DecodeMS: 10}}
	if _, ok := buildRequestPhaseReceipt(base, nativeperf.PhaseReceipt{}, false); !ok {
		t.Fatal("valid native row refused")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*perfledger.Record)
	}{
		{"fallback prefill route", func(r *perfledger.Record) { r.PrefillRoute = &perfledger.PrefillRoute{FallbackActive: true} }},
		{"phases exceed wall", func(r *perfledger.Record) { r.E2EMS = 5 }},
		{"missing request id", func(r *perfledger.Record) { r.RequestID = "" }},
		{"no measured phase", func(r *perfledger.Record) { r.NativeTiming = &perfledger.NativeTiming{} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := base
			nt := *base.NativeTiming
			rec.NativeTiming = &nt
			tc.mutate(&rec)
			m := &gatewayMetrics{}
			m.recordRequestPhases(rec, nativeperf.PhaseReceipt{}, false)
			rows, _ := m.requestPhases.snapshot(0)
			if len(rows) != 0 || m.requestPhases.refused.Load() != 1 {
				t.Fatalf("rows=%d refused=%d, want omitted and counted", len(rows), m.requestPhases.refused.Load())
			}
		})
	}
	m := &gatewayMetrics{}
	m.recordRequestPhases(perfledger.Record{RequestID: "proxy", Model: "m", E2EMS: 10}, nativeperf.PhaseReceipt{}, false)
	if rows, _ := m.requestPhases.snapshot(0); len(rows) != 0 || m.requestPhases.refused.Load() != 0 {
		t.Fatal("non-native row produced or refused a receipt")
	}
}
