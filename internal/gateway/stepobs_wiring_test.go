package gateway

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=2s — server construction + one in-process /metrics scrape;
// no model weights are loaded (the dual planner is selected from an empty Config model).

// newStepobsTestServer builds the smallest server that nativeEngineServing() accepts: a
// live proxy plus a loaded in-kernel model+tokenizer selects the dual planner, which is
// exactly the `fak serve` shape the fak_engine_* families render on.
func newStepobsTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{
		EngineID:      "test",
		Model:         "api-default",
		BaseURL:       "http://127.0.0.1:1/v1",
		Provider:      "openai",
		InKernelModel: &model.Model{},
		Tokenizer:     &tokenizer.Tokenizer{},
		LocalModelID:  "qwen2.5-coder:3b",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	if !srv.nativeEngineServing() {
		t.Fatalf("plannerKind = %q; this test needs a native engine to exercise the fak_engine_* render gate", plannerKind(srv.planner))
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func scrapeMetrics(t *testing.T, ts *httptest.Server) string {
	t.Helper()
	resp, err := ts.Client().Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("GET /metrics = %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func hasLine(body, line string) bool {
	for _, got := range strings.Split(body, "\n") {
		if got == line {
			return true
		}
	}
	return false
}

func hasPrefixLine(body, prefix string) bool {
	for _, got := range strings.Split(body, "\n") {
		if strings.HasPrefix(got, prefix) {
			return true
		}
	}
	return false
}

// TestServerConstructionAttachesTheStepObservers is the anti-fak_sched_preempt witness.
// The defect this change fixes is a family that has writers and no PRODUCTION
// attachment, so its board band can never light up. Presence of the recorder is not the
// claim — INVOKABILITY is: constructing a server, which is what every `fak serve` does,
// must install the kernel observer on the seam the compute backends already call.
func TestServerConstructionAttachesTheStepObservers(t *testing.T) {
	computetrace.SetObserver(nil)
	enginestep.SetPhaseObserver(nil)
	enginestep.SetStepObserver(nil)
	metrics.SetSpanObserver(nil)

	newStepobsTestServer(t)

	if !computetrace.Observed() {
		t.Fatal("gateway.New left the compute-trace seam unobserved: the kernel family would render ABSENT forever")
	}
	if !stepobs.Default.Attached() {
		t.Fatal("gateway.New did not attach the stepobs recorder")
	}

	// Prove the attachment is INVOKED, not merely present: feed the seam the compute
	// backends actually call and require the kernel family to move.
	before := stepobs.Default.Snapshot().KernelEvents
	computetrace.Record(computetrace.Event{
		Operation:   "matmul",
		Kernel:      "f32_matmul",
		Backend:     "cpu",
		TimerDomain: "host_monotonic",
		DurationNS:  int64(1500 * time.Microsecond),
	})
	if got := stepobs.Default.Snapshot().KernelEvents; got != before+1 {
		t.Fatalf("kernel events after one compute-trace record = %d, want %d", got, before+1)
	}

	// The sub-planner half needs the same INVOKED proof, and it is the half that was
	// unwitnessed. enginestep keeps its observer set private, so "an observer exists" is
	// not observable from here — the honest oracle is the same call internal/agent's
	// planner makes, on the same process recorder the planner feeds, moving the
	// planner-step registry. (internal/agent/planner_stepobs_wiring_test.go drives the
	// real InKernelPlanner.Complete that ends at this seam.)
	seatBefore := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindSeat)]
	enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Second)
	if got := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindSeat)]; got != seatBefore+1 {
		t.Fatalf("seat legs after one device_wait on the process recorder = %d, want %d; gateway.New left the sub-planner seam unwired", got, seatBefore+1)
	}
}

// TestComputeTraceObserverDoesNotChangeTraceSemantics pins that attaching an observer
// leaves the trace artifact path exactly as it was: Enabled() is still driven by the
// explicit recorder, so a metrics observer can never switch on per-GEMM device timing
// (or the model's activation-sample capture) behind the operator's back.
func TestComputeTraceObserverDoesNotChangeTraceSemantics(t *testing.T) {
	computetrace.SetObserver(nil)
	if computetrace.Observed() {
		t.Fatal("SetObserver(nil) left the seam observed")
	}
	computetrace.SetObserver(func(computetrace.Event) {})
	if computetrace.Enabled() {
		t.Fatal("attaching an observer flipped computetrace.Enabled(); the device backends would start measuring per GEMM")
	}
	rec, disable := computetrace.Enable(4, "run", "req")
	defer disable()
	if !computetrace.Enabled() {
		t.Fatal("the explicit recorder stopped enabling tracing")
	}
	if !rec.Enabled() {
		t.Fatal("the explicit recorder stopped being enabled")
	}
	// This test detaches a process-level seam; restore the production attachment so a
	// later test in this binary starts from the state `fak serve` gives it.
	t.Cleanup(attachStepObservation)
}

// TestNativeServeRendersKernelAndPlannerStepFamilies pins the /metrics surface: a native
// serve emits every declared family, and a proxy/mock serve emits NONE of them rather
// than a phantom idle engine.
func TestNativeServeRendersKernelAndPlannerStepFamilies(t *testing.T) {
	body := scrapeMetrics(t, newStepobsTestServer(t))
	for _, name := range stepobs.MetricFamilies {
		if !strings.Contains(body, "# TYPE "+name+" ") {
			t.Fatalf("family %q is missing from /metrics on a native serve", name)
		}
	}
	// The neighbour families must still be there: this change must not displace the
	// cycle namespace the new families sit inside.
	for _, name := range enginestep.MetricFamilies {
		if !strings.Contains(body, "# TYPE "+name+" ") {
			t.Fatalf("neighbour family %q disappeared from /metrics", name)
		}
	}
	// Both presence bits must be a well-formed 0 or 1 and must be PRESENT — a scrape
	// that cannot tell "absent" from "missing" is the defect this family set exists to
	// fix. (stepobs.Default is process-wide, so the recorded value itself is asserted
	// against a fresh recorder in internal/stepobs rather than here.)
	for _, name := range []string{stepobs.MetricKernelObserved, stepobs.MetricPlannerStepObserved} {
		bit := ""
		for _, line := range strings.Split(body, "\n") {
			if strings.HasPrefix(line, name+" ") {
				bit = strings.TrimSpace(strings.TrimPrefix(line, name))
			}
		}
		if bit != "0" && bit != "1" {
			t.Fatalf("%s rendered as %q, want a well-formed 0 or 1 presence bit", name, bit)
		}
	}
}

// TestMicroSpanLegsReachThePlannerStepFamily proves the microagent fan-out: a completed
// span leg recorded on a real MicroTracer — whose store is PER-INSTANCE, so this only
// works because the observer seam is process level — lands on the closed planner-step
// family the dashboard reads.
func TestMicroSpanLegsReachThePlannerStepFamily(t *testing.T) {
	newStepobsTestServer(t)
	tracer := metrics.NewMicroTracer()
	scope := tracer.Scope("micro-stepobs-1", metrics.MicroSpan{Kind: metrics.SpanSeat, Label: "device seat"})
	time.Sleep(2 * time.Millisecond)
	scope.End()

	before := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindSeat)]
	tracer.Scope("micro-stepobs-2", metrics.MicroSpan{Kind: metrics.SpanSeat}).End()
	after := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindSeat)]
	if after != before+1 {
		t.Fatalf("seat legs after one completed span = %d, want %d", after, before+1)
	}
	if !stepobs.Default.Snapshot().PlannerObserved {
		t.Fatal("planner_step_observed = false after a real span leg was fed")
	}
}

// TestDecodeStepsAndPhasesReachThePlannerStepFamily pins the native serving-loop fan-out:
// one decode forward and one device wait both land on the planner-step family, with the
// decode forward on the "step" kind and the wait on the "seat" kind.
func TestDecodeStepsAndPhasesReachThePlannerStepFamily(t *testing.T) {
	newStepobsTestServer(t)
	stepBefore := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindStep)]
	seatBefore := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindSeat)]
	admissionBefore := stepobs.Default.Snapshot().PlannerEvents[string(stepobs.StepKindAdmission)]

	rec := enginestep.New(8)
	rec.ObserveDecodeStep(enginestep.PathBatched, 4, 9*time.Millisecond)
	rec.ObservePhase(enginestep.PhaseDeviceWait, 3*time.Millisecond)
	rec.ObservePhase(enginestep.PhaseAdmissionWait, 2*time.Millisecond)
	rec.ObservePhase(enginestep.PhasePrefixLookup, 1*time.Millisecond)

	snap := stepobs.Default.Snapshot()
	if got := snap.PlannerEvents[string(stepobs.StepKindStep)]; got != stepBefore+1 {
		t.Fatalf("step legs = %d, want %d (one decode forward)", got, stepBefore+1)
	}
	if got := snap.PlannerEvents[string(stepobs.StepKindSeat)]; got != seatBefore+1 {
		t.Fatalf("seat legs = %d, want %d (one device wait)", got, seatBefore+1)
	}
	if got := snap.PlannerEvents[string(stepobs.StepKindAdmission)]; got != admissionBefore+1 {
		t.Fatalf("admission legs = %d, want %d (one admission wait)", got, admissionBefore+1)
	}
	// A phase with no closed-vocabulary counterpart must record nothing, not a default.
	if got := snap.PlannerEvents[string(stepobs.StepKindTool)]; got != 0 {
		t.Fatalf("tool legs = %d, want 0 (a prefix lookup is not a tool call)", got)
	}
}

// TestStepKindVocabularyTracksMicroSpanKind is the drift alarm. stepobs mirrors the
// MicroSpanKind vocabulary as plain strings so it does not have to import internal/metrics
// (which would be an import cycle in metrics' own test build). The mirror is only honest
// while it matches, so this binds every member to the constant itself, in both
// directions: a renamed or added MicroSpanKind fails here instead of silently landing in
// the kind-overflow bucket at runtime.
func TestStepKindVocabularyTracksMicroSpanKind(t *testing.T) {
	mirror := map[metrics.MicroSpanKind]stepobs.StepKind{
		metrics.SpanStep:      stepobs.StepKindStep,
		metrics.SpanTool:      stepobs.StepKindTool,
		metrics.SpanAdmission: stepobs.StepKindAdmission,
		metrics.SpanSeat:      stepobs.StepKindSeat,
		metrics.SpanVerdict:   stepobs.StepKindVerdict,
	}
	if len(mirror) != len(stepobs.StepKinds) {
		t.Fatalf("mirror covers %d kinds, stepobs.StepKinds has %d", len(mirror), len(stepobs.StepKinds))
	}
	for kind, want := range mirror {
		if string(kind) != string(want) {
			t.Fatalf("MicroSpanKind %q must mirror as stepobs kind %q, got %q", kind, want, want)
		}
		found := false
		for _, k := range stepobs.StepKinds {
			if k == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("mirrored kind %q is absent from stepobs.StepKinds", want)
		}
	}
	for _, k := range stepobs.StepKinds {
		covered := false
		for kind := range mirror {
			if string(kind) == string(k) {
				covered = true
				break
			}
		}
		if !covered {
			t.Fatalf("stepobs kind %q has no MicroSpanKind counterpart", k)
		}
	}
}

// TestNonNativeServeRendersNoEngineStepFamilies pins the fail-closed gate: a mock
// gateway must not claim an idle native engine, sub-seams included.
func TestNonNativeServeRendersNoEngineStepFamilies(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "api-default"})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	if srv.nativeEngineServing() {
		t.Fatalf("plannerKind = %q; this test needs a non-native gateway", plannerKind(srv.planner))
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	body := scrapeMetrics(t, ts)
	for _, name := range append(append([]string{}, stepobs.MetricFamilies...), enginestep.MetricFamilies...) {
		if strings.Contains(body, "# TYPE "+name+" ") {
			t.Fatalf("non-native gateway rendered native family %q", name)
		}
	}
	if hasLine(body, "fak_engine_kernel_observed 0") {
		t.Fatal("non-native gateway emitted a kernel observed bit")
	}
}
