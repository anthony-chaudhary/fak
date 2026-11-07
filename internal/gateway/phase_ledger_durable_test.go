package gateway

import (
	"bytes"
	"math"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// phase_ledger_durable_test.go — the per-request native phase receipts persisted
// beside the default durable perf ledger. Before this unit the six-bucket
// same-run breakdown (CPU, copies, weight loading, kernels, sync, unexplained)
// was computable only from the in-memory ring served at /v1/fak/perf/phases,
// which dies with the process; the durable gateway-perf.jsonl carried latency
// axes only. This case serves one native turn through the real HTTP edge with
// the default ledger attached, then reads the phase receipts back OFF DISK from
// the .phases.jsonl sibling and reduces them into the closed campaign buckets.

// fak-test:runtime fast est=3s lane=default — synthetic planner on the CPU
// reference path; no weights, no network.
func TestPhaseLedgerDurableSiblingJoinsPerfRowAndReducesBuckets(t *testing.T) {
	t.Cleanup(compute.RegisterDeviceWorkReader("phase-durable", func() compute.DeviceWork {
		return compute.DeviceWork{Dispatches: phaseTestDispatches.Load(), SubmitWaitNS: phaseTestSubmitWait.Load(), SubmitWaitObserved: true}
	}))
	h := newPerfServedHarness(t, phasePlanner{})
	h.chat("phase-model", "decode", false)

	rep := h.recent()
	if len(rep.Records) != 1 {
		t.Fatalf("rows = %d, want 1", len(rep.Records))
	}
	row := rep.Records[0]

	if err := h.srv.ClosePhaseLedger(); err != nil {
		t.Fatalf("phase ledger close: %v", err)
	}
	sibling := PhaseLedgerSiblingPath(h.path)
	raw, err := os.ReadFile(sibling)
	if err != nil {
		t.Fatalf("read durable phase sibling %s: %v", sibling, err)
	}
	receipts, err := modelperfobs.ReadRequestPhaseReceipts(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("read phase receipts: %v", err)
	}
	if len(receipts) != 1 {
		t.Fatalf("durable phase receipts = %d, want 1", len(receipts))
	}
	env := receipts[0]
	if env.RequestID != row.RequestID || env.Model != row.Model {
		t.Fatalf("receipt identity %q/%q does not join perf row %q/%q", env.RequestID, env.Model, row.RequestID, row.Model)
	}
	exclusive := map[nativeperf.Phase]time.Duration{}
	for _, p := range env.Receipt.Phases {
		exclusive[p.Phase] = p.Exclusive.Wall
	}
	if exclusive[nativeperf.PhaseKernel] != 2*time.Millisecond {
		t.Fatalf("durable receipt kernel = %s, want the measured 2 ms decode submit-wait", exclusive[nativeperf.PhaseKernel])
	}

	// The same-run six-bucket breakdown now comes from default artifacts alone:
	// the durable perf row plus its phase sibling, no live scrape required.
	obs := modelperfobs.Observation{Schema: modelperfobs.Schema, RequestID: row.RequestID, Model: row.Model, Backend: env.Backend, Status: http.StatusOK, DurationMS: row.E2EMS}
	burndown, err := modelperfobs.ReduceTimeBurndownBuckets([]modelperfobs.Observation{obs}, receipts, modelperfobs.BurndownTargets{PrefillMS: 4, DecodeMS: 20}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(burndown.BucketGroups) != 1 {
		t.Fatalf("bucket groups = %d, want 1", len(burndown.BucketGroups))
	}
	g := burndown.BucketGroups[0]
	buckets := map[string]float64{}
	for _, b := range g.Buckets {
		buckets[b.Bucket] += b.TotalMS
	}
	for _, name := range modelperfobs.BurndownBuckets() {
		if _, ok := buckets[name]; !ok {
			t.Fatalf("bucket %q absent from durable reduction: %v", name, buckets)
		}
	}
	if buckets[modelperfobs.BucketGPUKernel] <= 0 {
		t.Fatalf("gpu_kernel = %v, want the measured kernel time booked", buckets[modelperfobs.BucketGPUKernel])
	}
	if buckets[modelperfobs.BucketCPUWork] <= 0 {
		t.Fatalf("cpu_work = %v, want admission/prefill/decode/output active work booked", buckets[modelperfobs.BucketCPUWork])
	}
	if math.Abs(g.BucketedMS-g.APIWallMS) > 1e-9 {
		t.Fatalf("bucketed %g ms != api wall %g ms: every millisecond must land in exactly one bucket", g.BucketedMS, g.APIWallMS)
	}
}
