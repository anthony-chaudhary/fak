package causalreceipt

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fixedTrace mirrors the bounded shape the recorder writes: two backends across
// three events, with real device durations, byte counts, and one failure.
func fixedTrace() computetrace.Artifact {
	base := time.Unix(1700000000, 0).UTC()
	return computetrace.Artifact{
		Schema: computetrace.Schema,
		Events: []computetrace.Event{
			{Sequence: 1, RunID: "run-1", RequestID: "req-1", Operation: "matmul", Kernel: "sgemm", Backend: "metal", Route: "metal_command_buffer", StartedAt: base, DurationNS: 100, DeviceDurationNS: 80, Status: "ok", BytesRead: 10, BytesWritten: 5},
			{Sequence: 2, RunID: "run-1", RequestID: "req-1", Operation: "matmul", Kernel: "sgemm", Backend: "metal", Route: "metal_command_buffer", StartedAt: base.Add(time.Second), DurationNS: 200, DeviceDurationNS: 120, Status: "ok", Bytes: 7},
			{Sequence: 3, RunID: "run-1", RequestID: "req-1", Operation: "rmsnorm", Kernel: "rmsnorm", Backend: "cpu-ref", Route: "scalar", StartedAt: base.Add(2 * time.Second), DurationNS: 50, DeviceDurationNS: 0, Status: "failed", BytesRead: 3},
		},
	}
}

func TestProduceValidatesAndAggregates(t *testing.T) {
	r, err := Produce(fixedTrace(), map[string]string{
		"work_id":       "w-1",
		"turn_id":       "t-1",
		"graph_id":      "g-1",
		"request_id":    "req-1",
		"model_session": "s-1",
		"launch_digest": "abc123",
	})
	if err != nil {
		t.Fatalf("Produce: %v", err)
	}
	if err := Validate(r); err != nil {
		t.Fatalf("produced receipt must validate: %v", err)
	}
	if r.Schema != Schema {
		t.Fatalf("schema = %q, want %q", r.Schema, Schema)
	}
	if r.IDs.Work != "w-1" || r.IDs.Turn != "t-1" || r.IDs.Graph != "g-1" || r.IDs.Request != "req-1" || r.IDs.ModelSession != "s-1" {
		t.Fatalf("ids not mapped from attributes: %+v", r.IDs)
	}

	if len(r.Phases) != 2 {
		t.Fatalf("phase count = %d, want 2 (one per engine/route)", len(r.Phases))
	}
	if r.Phases[0].Engine != "metal" || r.Phases[1].Engine != "cpu-ref" {
		t.Fatalf("engines = %q,%q want metal,cpu-ref (first-seen order)", r.Phases[0].Engine, r.Phases[1].Engine)
	}
	if r.Phases[0].Backend != "metal_command_buffer" || r.Phases[1].Backend != "scalar" {
		t.Fatalf("backends = %q,%q", r.Phases[0].Backend, r.Phases[1].Backend)
	}
	if r.Phases[0].RecomputeNS != 200 {
		t.Fatalf("metal RecomputeNS = %d, want 80+120=200", r.Phases[0].RecomputeNS)
	}
	if r.Phases[0].QueueNS != 0 || r.Phases[0].LoadNS != 0 || r.Phases[0].TransferNS != 0 || r.Phases[0].RecoveryNS != 0 || r.Phases[0].VerificationNS != 0 {
		t.Fatalf("non-Recompute axes must stay zero (no fabrication): %+v", r.Phases[0])
	}
	if r.Phases[0].Bytes != 22 {
		t.Fatalf("metal Bytes = %d, want 10+5+7=22", r.Phases[0].Bytes)
	}
	if r.Phases[1].RecomputeNS != 0 || r.Phases[1].Bytes != 3 {
		t.Fatalf("cpu-ref phase = %+v, want recompute 0 / bytes 3", r.Phases[1])
	}
	if r.Phases[0].Outcome != "completed" {
		t.Fatalf("metal outcome = %q, want completed", r.Phases[0].Outcome)
	}
	if r.Phases[1].Outcome != "failed" {
		t.Fatalf("cpu-ref outcome = %q, want failed", r.Phases[1].Outcome)
	}
	if r.Phases[0].Tokens != 0 || r.Phases[0].CacheReuseBytes != 0 {
		t.Fatalf("tokens/cache must stay zero when the trace lacks them: %+v", r.Phases[0])
	}

	m, err := DeriveMetrics(r)
	if err != nil {
		t.Fatalf("DeriveMetrics: %v", err)
	}
	if m.OverheadNS == 0 {
		t.Fatal("DeriveMetrics overhead must be non-zero")
	}
	var summed int64
	for _, p := range r.Phases {
		summed += p.QueueNS + p.LoadNS + p.TransferNS + p.RecomputeNS + p.RecoveryNS + p.VerificationNS
	}
	if m.OverheadNS != summed || m.OverheadNS != 200 {
		t.Fatalf("overhead = %d, want summed intervals %d == 200", m.OverheadNS, summed)
	}
	if m.PhaseCount != 2 || m.Bytes != 25 {
		t.Fatalf("metrics = %+v, want 2 phases / 25 bytes", m)
	}
}

func TestProduceRejectsForeignSchema(t *testing.T) {
	a := fixedTrace()
	a.Schema = "fak.compute_trace.future"
	if _, err := Produce(a, nil); err == nil {
		t.Fatal("foreign compute-trace schema must be rejected")
	}
}

func TestProduceNamesUnknownEngineWithoutFabricating(t *testing.T) {
	a := computetrace.Artifact{Schema: computetrace.Schema, Events: []computetrace.Event{{Sequence: 1, RequestID: "req-9", DurationNS: 10, Status: "ok"}}}
	r, err := Produce(a, nil)
	if err != nil {
		t.Fatalf("Produce: %v", err)
	}
	if len(r.Phases) != 1 || r.Phases[0].Engine != "unknown" {
		t.Fatalf("phase = %+v, want one engine=unknown phase", r.Phases)
	}
	if r.IDs.Request != "req-9" {
		t.Fatalf("request = %q, want req-9 sourced from the event", r.IDs.Request)
	}
	if err := Validate(r); err != nil {
		t.Fatalf("validate: %v", err)
	}
}

func TestProduceRecordsCodeVersionAndLaunchDigest(t *testing.T) {
	r, err := Produce(fixedTrace(), map[string]string{
		"work_id":       "w-1",
		"turn_id":       "t-1",
		"graph_id":      "g-1",
		"request_id":    "req-1",
		"code_version":  "v1+deadbeef",
		"launch_digest": "sha256:launch",
	})
	if err != nil {
		t.Fatalf("Produce: %v", err)
	}
	if err := Validate(r); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if got := r.ModuleVersions["internal/computetrace"]; got != "v1+deadbeef" {
		t.Fatalf("module versions = %v, want code version under internal/computetrace", r.ModuleVersions)
	}
	if got := r.Attributes["launch_context_digest"]; got != "sha256:launch" {
		t.Fatalf("attributes = %v, want launch_context_digest", r.Attributes)
	}
	if _, leaked := r.Attributes["launch_digest"]; leaked {
		t.Fatalf("raw launch_digest key must not survive: %v", r.Attributes)
	}
}
