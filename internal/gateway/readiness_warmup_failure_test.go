package gateway

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type refusingWarmupPlanner struct{ err error }

func (p refusingWarmupPlanner) Model() string { return "refusing-warmup-planner" }

func (p refusingWarmupPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, p.err
}

func failedWarmupServer(t *testing.T, err error) *Server {
	t.Helper()
	srv := &Server{planner: refusingWarmupPlanner{err: err}}
	srv.ArmWarmupGate()
	if _, got := srv.RunWarmup(context.Background()); !errors.Is(got, err) {
		t.Fatalf("RunWarmup err = %v, want %v returned unchanged", got, err)
	}
	return srv
}

// TestRunWarmupFailureReportsWarmupFailed is the G14 regression: a backend warmup
// that fails closed must not leave /healthz reporting warmup_pending forever. The
// gate reports warmup_failed with a closed reason and a bounded message, stays not
// ready (503), and inference routes refuse with the warmup_failed code.
func TestRunWarmupFailureReportsWarmupFailed(t *testing.T) {
	refusal := &model.BackendForwardOperationError{
		Backend: "vulkan", Forward: "deepseek41", Path: "device-only", Stage: "attention",
		Cause: errors.New("host fallback refused"),
	}
	srv := failedWarmupServer(t, refusal)

	code, body := warmupHealthz(t, srv)
	if code != http.StatusServiceUnavailable || body["ok"] != false {
		t.Fatalf("failed warmup /healthz = %d ok=%v, want 503 ok=false", code, body["ok"])
	}
	if body["warmup_failed"] != true {
		t.Fatalf("failed warmup /healthz warmup_failed = %v, want true; body=%v", body["warmup_failed"], body)
	}
	if _, present := body["warmup_pending"]; present {
		t.Fatalf("failed warmup /healthz still reports warmup_pending")
	}
	failure, _ := body["warmup_failure"].(map[string]any)
	if failure["reason"] != "backend_refused" {
		t.Fatalf("warmup_failure.reason = %v, want backend_refused", failure["reason"])
	}
	msg, _ := failure["message"].(string)
	if !strings.Contains(msg, "failed closed") || len(msg) > forbiddenDetailMax+len("…") {
		t.Fatalf("warmup_failure.message = %q, want the bounded refusal text", msg)
	}

	rec := httptest.NewRecorder()
	if !srv.checkWarmupPending(rec) || rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), `"code":"warmup_failed"`) {
		t.Fatalf("inference gate after failed warmup = %d %s, want 503 warmup_failed", rec.Code, rec.Body.String())
	}

	srv.MarkWarmupComplete(0)
	if _, body := warmupHealthz(t, srv); body["warmup_failed"] != nil || body["warmup_failure"] != nil {
		t.Fatalf("completed warmup still reports a failure: %v", body)
	}
}

func TestRunWarmupFailureReasonIsClosedToken(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{context.Canceled, "canceled"},
		{model.ErrV4FlashAttentionUnimplemented, "backend_refused"},
		{&model.V41ForwardError{Layer: 3, Err: errors.New("stage not admitted")}, "backend_refused"},
		{model.ErrV41PrefillLatency, "backend_refused"},
		{errors.New("boom sk-ant-abcdefghijklmnop"), "backend_error"},
	}
	for _, c := range cases {
		_, body := warmupHealthz(t, failedWarmupServer(t, c.err))
		failure, _ := body["warmup_failure"].(map[string]any)
		if failure["reason"] != c.want {
			t.Errorf("warmup error %v: reason = %v, want %q", c.err, failure["reason"], c.want)
		}
		if msg, _ := failure["message"].(string); strings.Contains(msg, "sk-ant-") {
			t.Errorf("warmup_failure.message leaked a token-shaped secret: %q", msg)
		}
	}
}

// An unarmed serve stays byte-for-byte unaffected by a failed warmup call.
func TestRunWarmupFailureUnarmedGateSilent(t *testing.T) {
	srv := &Server{planner: refusingWarmupPlanner{err: errors.New("boom")}}
	_, _ = srv.RunWarmup(context.Background())
	if _, body := warmupHealthz(t, srv); body["warmup_failed"] != nil {
		t.Fatalf("unarmed gate reported warmup_failed: %v", body)
	}
}
