package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type visionIngressProbePlanner struct {
	nativeTokenizeProbePlanner
	params agent.SampleParams
}

func (p *visionIngressProbePlanner) EncodePrompt(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (agent.PromptEncoding, error) {
	for _, opt := range opts {
		opt(&p.params)
	}
	return p.nativeTokenizeProbePlanner.EncodePrompt(ctx, messages, tools, opts...)
}

// fak-test:runtime fast est=20ms lane=default
func TestVisionIngressRequestOptionIsExplicit(t *testing.T) {
	req := ChatRequest{Fak: &FakRequestExt{MultimodalMode: string(model.MultimodalModeQuarantine)}}
	var params agent.SampleParams
	for _, opt := range nativeTokenizeSampleOpts(req) {
		opt(&params)
	}
	if params.MultimodalMode != model.MultimodalModeQuarantine {
		t.Fatalf("mode = %q, want explicit quarantine", params.MultimodalMode)
	}

	params = agent.SampleParams{}
	for _, opt := range nativeTokenizeSampleOpts(ChatRequest{}) {
		opt(&params)
	}
	if params.MultimodalMode != model.MultimodalModeDisabled {
		t.Fatalf("default mode = %q, want fail-closed disabled", params.MultimodalMode)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestVisionIngressNativeTokenizeRouteCarriesImageAndAdmission(t *testing.T) {
	planner := &visionIngressProbePlanner{}
	srv := newTestServer(t)
	srv.model = "test-model"
	srv.planner = planner
	body := `{"model":"test-model","messages":[{"role":"user","content":[{"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}}]}],"fak":{"multimodal_mode":"quarantine"}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/fak/tokenize", strings.NewReader(body))
	req.Header.Set("content-type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if planner.params.MultimodalMode != model.MultimodalModeQuarantine {
		t.Fatalf("planner mode = %q, want quarantine", planner.params.MultimodalMode)
	}
	if len(planner.messages) != 1 || planner.messages[0].Content != "inspect" || len(planner.messages[0].VisionImages) != 1 {
		t.Fatalf("planner messages = %+v, want text projection plus one native image carrier", planner.messages)
	}
}
