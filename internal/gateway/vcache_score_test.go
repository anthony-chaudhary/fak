package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func TestHandleFakVCacheScoreReturnsPlannedReportWhenIdle(t *testing.T) {
	restore := swapCacheObserver(cacheobs.New())
	defer restore()

	rec := httptest.NewRecorder()
	(&Server{}).handleFakVCacheScore(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/score", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.ActiveSource != "planned" {
		t.Fatalf("report schema/source=%q/%q, want planned vcache score", rep.Schema, rep.ActiveSource)
	}
	if rep.Planes.ProviderObserved.Available || rep.AgenticActivation.Active {
		t.Fatalf("idle API must not invent provider or fak-owned activation evidence: planes=%+v activation=%+v", rep.Planes, rep.AgenticActivation)
	}
}

func TestHandleFakVCacheScoreSeparatesProviderAndKernelEvidence(t *testing.T) {
	obs := cacheobs.New()
	restore := swapCacheObserver(obs)
	defer restore()
	obs.Observe(100, 95)

	m := newGatewayMetrics(time.Now())
	m.observeInference(86, 0, 1920, 0, "end_turn", time.Millisecond)
	s := &Server{metrics: m}

	rec := httptest.NewRecorder()
	s.handleFakVCacheScore(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/score", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if rep.ActiveSource != "telemetry" || rep.Observed == nil || !rep.Planes.ProviderObserved.Available {
		t.Fatalf("provider telemetry not surfaced as observed evidence: %+v", rep)
	}
	if !rep.AgenticActivation.Active || rep.AgenticActivation.KernelKVEvents != 1 || rep.AgenticActivation.Total != 1 {
		t.Fatalf("kernel activation not surfaced from cacheobs: %+v", rep.AgenticActivation)
	}
	kernel := rep.Planes.KernelWitnessed
	if !kernel.Available || kernel.Provenance != "WITNESSED" {
		t.Fatalf("kernel plane not surfaced as witnessed evidence: %+v", kernel)
	}
	if kernel.BaselineTokenEquiv != 100 || kernel.SavedTokenEquiv != 95 || kernel.CostTokenEquiv != 5 {
		t.Fatalf("kernel plane economics=%+v, want 95 saved / 100 baseline / 5 cost", kernel)
	}
	if rep.AgenticActivation.ProviderVCacheDecisions != 0 {
		t.Fatalf("provider cache counters must not become fak-authored provider decisions: %+v", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness.Facets.AgenticActivation != 20 {
		t.Fatalf("default-usefulness activation facet=%d, want fak-owned KV activation credit", rep.DefaultUsefulness.Facets.AgenticActivation)
	}
}

func TestHandleFakVCacheScoreReportsExternalEngineHitRate(t *testing.T) {
	restore := swapCacheObserver(cacheobs.New())
	defer restore()

	s := &Server{metrics: newGatewayMetrics(time.Now())}
	s.SetServingMetricsEmitters(staticServingEmitter{rows: []ServingMetricRow{
		{
			Labels: ServingMetricLabels{
				Worker: "gpu-1",
				Engine: "vllm",
				Model:  "m",
			},
			PrefixCacheHitRate: ServingGaugeValue(0.64),
		},
	}})

	rec := httptest.NewRecorder()
	s.handleFakVCacheScore(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/score", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	external := rep.Planes.ExternalEngineObserved
	if !external.Available || external.Provenance != "OBSERVED" {
		t.Fatalf("external plane=%+v, want observed external-engine evidence", external)
	}
	if external.HitRate != 0.64 {
		t.Fatalf("external hit rate=%g, want 0.64", external.HitRate)
	}
	if rep.Planes.ProviderObserved.Available || rep.Planes.KernelWitnessed.Available {
		t.Fatalf("external hit-rate evidence must not invent provider/kernel planes: %+v", rep.Planes)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue != 0 {
		t.Fatalf("external hit-rate-only evidence must not earn token-value credit: %+v", rep.DefaultUsefulness)
	}
}

func TestHandleFakVCacheScoreReportsContextCompactionEvidence(t *testing.T) {
	restore := swapCacheObserver(cacheobs.New())
	defer restore()

	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{Reason: agent.CompactReasonNone, Dropped: 3, ShedTokens: 800}, false)
	s := &Server{metrics: m, compactHistoryBudget: 1200}

	rec := httptest.NewRecorder()
	s.handleFakVCacheScore(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/vcache/score", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	context := rep.Planes.ContextWitnessed
	if !context.Available || context.Provenance != "WITNESSED" {
		t.Fatalf("context plane=%+v, want witnessed compaction evidence", context)
	}
	if context.SavedTokenEquiv != 800 || context.BaselineTokenEquiv != 2000 || context.CostTokenEquiv != 1200 {
		t.Fatalf("context economics=%+v, want 800 saved / 2000 baseline / 1200 resident cost", context)
	}
	if !rep.AgenticActivation.Active || rep.AgenticActivation.ContextEvents != 1 || rep.AgenticActivation.Total != 1 {
		t.Fatalf("context activation=%+v, want one context event only", rep.AgenticActivation)
	}
	if rep.Planes.ProviderObserved.Available || rep.Planes.KernelWitnessed.Available {
		t.Fatalf("context compaction evidence must not invent provider/kernel planes: %+v", rep.Planes)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue == 0 {
		t.Fatalf("default-usefulness should credit budgeted context value: %+v", rep.DefaultUsefulness)
	}
}

func TestHandleFakVCacheScoreRejectsNonGet(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakVCacheScore(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/vcache/score", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status=%d want 405", rec.Code)
	}
}

func swapCacheObserver(next *cacheobs.Observer) func() {
	prev := cacheobs.Default
	cacheobs.Default = next
	return func() { cacheobs.Default = prev }
}

type staticServingEmitter struct {
	rows []ServingMetricRow
}

func (e staticServingEmitter) SnapshotServingMetrics() []ServingMetricRow {
	return append([]ServingMetricRow(nil), e.rows...)
}
