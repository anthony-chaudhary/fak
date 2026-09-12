package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

func TestTurnCostMetricsRenderOnGatewayMetricsSurface(t *testing.T) {
	srv := newTestServer(t)
	if srv.turnCost == nil {
		t.Fatal("New did not initialize the turn-cost collector")
	}
	srv.turnCost.Observe(&turncost.TurnCostRecord{
		Streaming: false,
		Phases: map[turncost.Phase]float64{
			turncost.PhaseAdmission: 0.01,
			turncost.PhasePrefill:   0.2,
			turncost.PhaseDecode:    0.4,
			turncost.PhaseLedger:    0.005,
		},
	})
	out := srv.renderMetrics()
	for _, want := range []string{
		`# TYPE fak_turn_cost_turns_total counter`,
		`fak_turn_cost_turns_total{surface="buffered"} 1`,
		`fak_turn_cost_seconds_total{surface="buffered",phase="admission"} 0.01`,
		`fak_turn_cost_seconds_total{surface="buffered",phase="prefill"} 0.2`,
		`fak_turn_cost_seconds_total{surface="buffered",phase="decode"} 0.4`,
		`fak_turn_cost_seconds_total{surface="buffered",phase="ledger"} 0.005`,
		`fak_turn_cost_phase_samples_total{surface="buffered",phase="decode"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}
	if strings.Contains(out, `fak_turn_cost_seconds_total{surface="buffered",phase="stream"}`) {
		t.Fatalf("unmeasured phase stream must stay absent:\n%s", out)
	}
}

func TestTurnCostPhasePin(t *testing.T) {
	const delay = 25 * time.Millisecond
	const tol = 40 * time.Millisecond
	for _, phase := range turncost.Phases {
		t.Run(string(phase), func(t *testing.T) {
			rec := &turncost.TurnCostRecord{}
			timePhase(rec, phase, func() { time.Sleep(delay) })
			got, ok := rec.Phase(phase)
			if !ok {
				t.Fatalf("phase %q was not recorded", phase)
			}
			if d := time.Duration(got * float64(time.Second)); d < delay-tol/2 || d > delay+tol {
				t.Fatalf("phase %q measured %v, want ~%v (±%v)", phase, d, delay, tol)
			}
		})
	}
}

func TestTurnCostDisabledByteParity(t *testing.T) {
	body := []byte(`{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`)

	run := func(t *testing.T, enabled string) *httptest.ResponseRecorder {
		t.Helper()
		t.Setenv("FAK_TURN_COST", enabled)
		srv := newTestServer(t)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
		}
		return rr
	}

	off := run(t, "0")
	on := run(t, "1")

	normalize := func(raw []byte) []byte {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("unmarshal response: %v (%s)", err, raw)
		}
		delete(m, "id")
		delete(m, "created")
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}

	if !bytes.Equal(normalize(off.Body.Bytes()), normalize(on.Body.Bytes())) {
		t.Fatalf("response bodies differ between disabled and enabled cost exposure:\noff=%s\non =%s",
			off.Body.String(), on.Body.String())
	}
	if bytes.Contains(on.Body.Bytes(), []byte("turn_cost")) {
		t.Fatalf("plain turn must not embed turn_cost: %s", on.Body.String())
	}
	if !strings.Contains(on.Header().Get("Content-Type"), "json") {
		t.Fatalf("unexpected content type %q", on.Header().Get("Content-Type"))
	}
}

// slowPlanner delays its completion so a decoded turn contributes a measurable
// (nonzero) decode duration even under Windows' coarse monotonic clock.
type slowPlanner struct {
	comp  *agent.Completion
	delay time.Duration
}

func (p slowPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(p.delay)
	return p.comp, nil
}
func (slowPlanner) Model() string { return "slow" }

// slowStreamPlanner is a StreamingPlanner twin of slowPlanner: it delays, then emits
// one content delta through the sink so the live stream seam runs end-to-end.
type slowStreamPlanner struct {
	comp  *agent.Completion
	delay time.Duration
}

func (p slowStreamPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(p.delay)
	return p.comp, nil
}
func (p slowStreamPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(p.delay)
	if sink != nil {
		// A substantial streamed body makes the SSE serialization/flush span exceed
		// the OS clock resolution, so the stream phase is measured rather than absent.
		chunk := strings.Repeat("x", 4096)
		for i := 0; i < 512; i++ {
			if err := sink(chunk); err != nil {
				return nil, err
			}
		}
	}
	return p.comp, nil
}
func (slowStreamPlanner) StreamingSupported() bool { return true }
func (slowStreamPlanner) Model() string            { return "slow-stream" }

func TestTurnCostMetricsOnCompletedTurns(t *testing.T) {
	cases := []struct {
		name      string
		streaming bool
		body      string
		wantPhase string
	}{
		{"buffered", false, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`, "proxy_hop"},
		{"streamed", true, `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`, "stream"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv("FAK_TURN_COST", "1")
			srv := newTestServer(t)
			comp := &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}
			if c.streaming {
				srv.planner = slowStreamPlanner{delay: 30 * time.Millisecond, comp: comp}
			} else {
				srv.planner = slowPlanner{delay: 30 * time.Millisecond, comp: comp}
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(c.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
			}
			out := srv.renderMetrics()
			surface := "buffered"
			if c.streaming {
				surface = "stream"
			}
			if !strings.Contains(out, `fak_turn_cost_turns_total{surface="`+surface+`"} 1`) {
				t.Fatalf("%s turn not folded into /metrics:\n%s", surface, out)
			}
			want := `fak_turn_cost_seconds_total{surface="` + surface + `",phase="` + c.wantPhase + `"}`
			if !strings.Contains(out, want) {
				t.Fatalf("%s completed turn missing %q:\n%s", surface, want, out)
			}

			// The collector renders every phase the closed vocabulary can carry, on
			// both transport surfaces, once a completed turn reports it.
			full := newTestServer(t)
			full.turnCost.Observe(&turncost.TurnCostRecord{
				Streaming: c.streaming,
				Phases: map[turncost.Phase]float64{
					turncost.PhaseAdmission: 0.01,
					turncost.PhasePrefill:   0.02,
					turncost.PhaseDecode:    0.03,
					turncost.PhaseStream:    0.04,
					turncost.PhaseProxyHop:  0.05,
					turncost.PhaseLedger:    0.06,
				},
			})
			fullOut := full.renderMetrics()
			for _, phase := range turncost.Phases {
				want := `fak_turn_cost_seconds_total{surface="` + surface + `",phase="` + string(phase) + `"}`
				if !strings.Contains(fullOut, want) {
					t.Fatalf("%s /metrics missing %q after a completed turn:\n%s", surface, want, fullOut)
				}
			}
		})
	}
}

func TestTurnCostReceiptAndCompletionLifecycle(t *testing.T) {
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	t.Setenv("FAK_TURN_COST", "1")
	srv := newTestServer(t)
	srv.planner = slowPlanner{delay: 15 * time.Millisecond, comp: &agent.Completion{
		Model: "actual-model", Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop", NativeInference: &model.NativeInferenceReceipt{PrefillSeconds: 0.002, DecodeSeconds: 0.003},
	}}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-Trace-Id", "cost-lifecycle")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d: %s", rr.Code, rr.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Fak == nil || response.Fak.TurnCost == nil {
		t.Fatalf("missing receipt: %s", rr.Body.String())
	}
	rec := response.Fak.TurnCost
	if rec.Trace != "cost-lifecycle" || rec.Model != "actual-model" || rec.TotalSeconds <= 0 {
		t.Fatalf("serialized incomplete identity or total: %+v", rec)
	}
	if rec.Phases[turncost.PhasePrefill] != 0.002 || rec.Phases[turncost.PhaseDecode] != 0.003 {
		t.Fatalf("backend phase attribution: %+v", rec.Phases)
	}
	metrics := srv.renderMetrics()
	for _, want := range []string{`fak_turn_cost_turns_total{surface="buffered"} 1`, `fak_turn_cost_phase_samples_total{surface="buffered",phase="ledger"} 1`} {
		if !strings.Contains(metrics, want) {
			t.Fatalf("completion missing %s", want)
		}
	}
}

func TestTurnCostStreamingGate(t *testing.T) {
	for _, tc := range []struct {
		global, stream string
		want           bool
	}{{"1", "0", false}, {"0", "1", true}} {
		t.Run(tc.global+tc.stream, func(t *testing.T) {
			t.Setenv("FAK_TURN_COST", tc.global)
			t.Setenv("FAK_TURN_COST_STREAM", tc.stream)
			srv := newTestServer(t)
			srv.planner = slowStreamPlanner{comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}}
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
			rr := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				t.Fatalf("status=%d", rr.Code)
			}
			if got := strings.Contains(srv.renderMetrics(), "fak_turn_cost_turns_total"); got != tc.want {
				t.Fatalf("stream exposure = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestTurnCostNeverInventsBackendPhases(t *testing.T) {
	rec := &turncost.TurnCostRecord{}
	srv := &Server{}
	srv.recordBufferedTurnCost(servedSessionTurn{turnCost: rec}, &agent.Completion{NativeInference: &model.NativeInferenceReceipt{}}, time.Now().Add(-time.Second))
	if len(rec.Phases) != 0 {
		t.Fatalf("unmeasured backend phases fabricated: %+v", rec.Phases)
	}
	srv.recordBufferedTurnCost(servedSessionTurn{turnCost: rec}, &agent.Completion{}, time.Now().Add(-time.Millisecond))
	if rec.Phases[turncost.PhaseProxyHop] <= 0 {
		t.Fatal("direct planner hop was not timed")
	}
	if _, ok := rec.Phase(turncost.PhaseDecode); ok {
		t.Fatal("planner wall time mislabeled as decode")
	}
}
