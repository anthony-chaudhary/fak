package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// weightResidencyPlanner is a local-planner fake that reports resident weight bytes.
type weightResidencyPlanner struct {
	kvMemoryStatsPlanner
	wr agent.WeightResidency
	ok bool
}

func (p weightResidencyPlanner) WeightResidency() (agent.WeightResidency, bool) { return p.wr, p.ok }

func residentHealthzBody(t *testing.T, srv *Server) (int, map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz: %v (%s)", err, rec.Body.String())
	}
	return rec.Code, body
}

func TestHealthzResidentWeightsReported(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = weightResidencyPlanner{ok: true, wr: agent.WeightResidency{
		TotalResidentBytes:  1000,
		Q4KBytes:            600,
		Q6KEmbedBytes:       300,
		F32Bytes:            100,
		DecodeBytesPerToken: 900,
		LMHead:              "metal-q6k",
	}}
	code, body := residentHealthzBody(t, srv)
	if code != http.StatusOK || body["ok"] != true {
		t.Fatalf("resident_weights must not flip readiness: code=%d body=%v", code, body)
	}
	rw, ok := body["resident_weights"].(map[string]any)
	if !ok {
		t.Fatalf("resident_weights missing: %v", body)
	}
	for k, want := range map[string]any{
		"total_resident_bytes":   float64(1000),
		"q4k_bytes":              float64(600),
		"q6k_embed_bytes":        float64(300),
		"tied_embed_f32_bytes":   float64(0),
		"tied_head_q8_bytes":     float64(0),
		"decode_bytes_per_token": float64(900),
		"lm_head":                "metal-q6k",
	} {
		if rw[k] != want {
			t.Errorf("resident_weights[%q] = %v, want %v", k, rw[k], want)
		}
	}
}

func TestHealthzResidentWeightsAbsentWithoutReporter(t *testing.T) {
	srv := newTestServer(t)
	if _, body := residentHealthzBody(t, srv); body["resident_weights"] != nil {
		t.Fatalf("non-reporting planner must omit resident_weights: %v", body["resident_weights"])
	}
	srv.planner = weightResidencyPlanner{ok: false}
	if _, body := residentHealthzBody(t, srv); body["resident_weights"] != nil {
		t.Fatalf("reporter without a model must omit resident_weights: %v", body["resident_weights"])
	}
}
