package gateway

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

type engineSubstepsWire struct {
	Native   bool                    `json:"native"`
	Substeps *engineSubstepsSnapshot `json:"substeps"`
}

type engineSubstepsLatency struct {
	Count        uint64  `json:"count"`
	TotalSeconds float64 `json:"total_seconds"`
	MeanSeconds  float64 `json:"mean_seconds"`
	P50Seconds   float64 `json:"p50_seconds"`
	P95Seconds   float64 `json:"p95_seconds"`
	MaxSeconds   float64 `json:"max_seconds"`
}
type engineSubstepsKernel struct {
	Kernel      string `json:"kernel"`
	Backend     string `json:"backend"`
	TimerDomain string `json:"timer_domain"`
	Measured    bool   `json:"measured"`
	engineSubstepsLatency
}
type engineSubstepsSnapshot struct {
	RecorderAttached bool                             `json:"recorder_attached"`
	KernelObserved   bool                             `json:"kernel_observed"`
	PlannerObserved  bool                             `json:"planner_step_observed"`
	KernelEvents     uint64                           `json:"kernel_events"`
	KernelLatency    []engineSubstepsKernel           `json:"kernel_latency"`
	PlannerLatency   map[string]engineSubstepsLatency `json:"planner_step_latency"`
	PlannerEvents    map[string]uint64                `json:"planner_step_events"`
}

func readEngineSubsteps(t *testing.T, srv *Server) engineSubstepsWire {
	t.Helper()
	rec := getObservationEngine(t, srv, http.MethodGet, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("engine status=%d", rec.Code)
	}
	var wire engineSubstepsWire
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatal(err)
	}
	return wire
}

// fak-test:runtime fast est=1s lane=default
func TestObservationEngineSubstepsDefaultCPURequest(t *testing.T) {
	old := stepobs.Default
	stepobs.Default = stepobs.New()
	t.Cleanup(func() { stepobs.Default = old; stepobs.Attach() })
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")
	if computetrace.Enabled() {
		t.Fatal("fixture requires default tracing-disabled execution")
	}
	cfg := model.Config{HiddenSize: 32, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IntermediateSize: 64, VocabSize: 512, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	srv := newEngineStepNativeServer(t)
	m := model.NewSynthetic(cfg)
	m.Quantize()
	srv.planner = agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "native", false, nil, false)
	before := readEngineSubsteps(t, srv)
	if before.Substeps == nil || !before.Substeps.RecorderAttached || before.Substeps.KernelObserved || before.Substeps.PlannerObserved {
		t.Fatal("fresh native snapshot must be attached and honestly absent")
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
	req.RemoteAddr = "127.0.0.1:50001"
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("native CPU completion status=%d body=%s", rec.Code, rec.Body.String())
	}
	after := readEngineSubsteps(t, srv)
	if !after.Native || after.Substeps == nil {
		t.Fatal("native response lacks substeps")
	}
	s := after.Substeps
	if !s.KernelObserved || !s.PlannerObserved || s.KernelEvents == 0 || len(s.KernelLatency) == 0 || len(s.PlannerLatency) != 5 || s.PlannerLatency["step"].Count == 0 {
		t.Fatalf("native observation: attached=%t kernel_observed=%t kernel_events=%d kernel_series=%d planner_observed=%t planner_kinds=%d step_count=%d planner_events=%v", s.RecorderAttached, s.KernelObserved, s.KernelEvents, len(s.KernelLatency), s.PlannerObserved, len(s.PlannerLatency), s.PlannerLatency["step"].Count, s.PlannerEvents)
	}
	metrics := srv.renderMetrics()
	sample := func(name string) float64 {
		t.Helper()
		for _, line := range strings.Split(metrics, "\n") {
			if strings.HasPrefix(line, name+" ") {
				fields := strings.Fields(line)
				v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
				if err != nil {
					t.Fatal(err)
				}
				return v
			}
		}
		t.Fatalf("missing Prometheus sample %s", name)
		return 0
	}
	var count uint64
	for _, k := range s.KernelLatency {
		count += k.Count
		// A completed kernel may quantize to zero on Windows; count and Measured
		// prove observation without fabricating a minimum duration.
		if !k.Measured || k.TimerDomain != "host_monotonic" || k.Count == 0 || k.TotalSeconds < 0 || k.MeanSeconds < 0 || k.P50Seconds < 0 || k.P95Seconds < 0 || k.MaxSeconds < 0 || k.P50Seconds > k.P95Seconds || k.P95Seconds > k.MaxSeconds {
			t.Fatal("CPU kernel timing or attribution invalid")
		}
		labels := `{kernel="` + k.Kernel + `",backend="` + k.Backend + `",timer_domain="` + k.TimerDomain + `"}`
		if sample(stepobs.MetricKernelSeconds+"_count"+labels) != float64(k.Count) || math.Abs(sample(stepobs.MetricKernelSeconds+"_sum"+labels)-k.TotalSeconds) > 1e-9 {
			t.Fatal("kernel HTTP summary disagrees with metrics")
		}
	}
	if count != s.KernelEvents {
		t.Fatal("snapshot lost kernel calls")
	}
	for kind, latency := range s.PlannerLatency {
		if latency.Count != s.PlannerEvents[kind] || sample(stepobs.MetricPlannerStepSeconds+`_count{kind="`+kind+`"}`) != float64(latency.Count) || math.Abs(sample(stepobs.MetricPlannerStepSeconds+`_sum{kind="`+kind+`"}`)-latency.TotalSeconds) > 1e-9 {
			t.Fatal("planner HTTP summary disagrees with metrics")
		}
	}
	compact := getObservationEngine(t, srv, http.MethodGet, "format=compact").Body.String()
	if !strings.Contains(strings.ToLower(compact), "kernel") || !strings.Contains(compact, "STEP") || strings.Count(compact, "\n") != 1 {
		t.Fatalf("compact reader lacks kernel/planner summary: %q", compact)
	}
	if computetrace.Enabled() {
		t.Fatal("observation reader enabled expensive trace artifact collection")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestObservationEngineSubstepsProxyDoesNotLeakNativeHistory(t *testing.T) {
	srv := newEngineStepNativeServer(t)
	stepobs.Default.ObservePlannerStep(stepobs.StepKindStep, 1)
	for _, p := range []agent.Planner{&agent.MockPlanner{}, &agent.HTTPPlanner{BaseURL: "http://provider.invalid", ModelID: "proxy"}} {
		srv.planner = p
		wire := readEngineSubsteps(t, srv)
		if wire.Native || wire.Substeps != nil {
			t.Fatal("non-native planner leaks process-native telemetry")
		}
	}
}
