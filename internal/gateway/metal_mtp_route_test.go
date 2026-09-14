package gateway

// Issue #12331 source-only regression scope: a REMOTE request routed to a DualPlanner's
// proxy side must not inherit the local planner's Metal MTP speculative header when a
// server-level MetalMTPCoordinator is configured (SetMetalMTPCoordinator).

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestOpenAICompletionMetalMTPRoute drives the REAL HTTP chat-wire path (httptest over
// srv.Handler(), POST /v1/chat/completions): with a DualPlanner (remote-model proxy +
// local in-kernel planner) and a server-level Metal MTP coordinator configured, the
// LOCAL-routed request must advertise x-fak-speculative: mtp-metal while the
// REMOTE-routed request must not inherit it.
func TestOpenAICompletionMetalMTPRoute(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "4")

	m := model.NewSyntheticQwen38MTP()
	m.Quantize()

	localPlanner := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "local", false, nil, false)
	coord, err := model.NewMetalMTPCoordinator(nil, model.DefaultMetalMTPConfig())
	if err != nil {
		t.Fatalf("NewMetalMTPCoordinator: %v", err)
	}
	localPlanner.SetMetalMTPCoordinator(coord)
	t.Cleanup(func() {
		localPlanner.DisableMetalMTP()
		_ = coord.Close()
	})

	proxyPlanner := agent.NewMockPlanner("remote-model")
	dual, err := NewDualPlanner(proxyPlanner, localPlanner, "local")
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}

	srv := &Server{
		model:   "remote-model",
		planner: dual,
		logf:    func(format string, args ...any) { t.Logf(format, args...) },
		// k is the adjudication kernel the proxy turn's proposed tool calls fold
		// through; the mock's lead turn proposes get_user_details, which would
		// nil-panic on a bare Server literal (see kernel.go:157 adjudicatorsFor).
		k: kernel.New("test"),
	}
	srv.metrics = newGatewayMetrics(time.Now())
	// Server-level set — the regression's premise (the LOCAL planner's coordinator
	// mirrored onto the Server).
	srv.SetMetalMTPCoordinator(coord)

	// Local arm (guard, seam-level): the fully headless local round-trip cannot
	// complete here — the synthetic Qwen3.8 vocab (97) is smaller than the ChatML
	// byte-level tokenizer's 260+ symbol space, so a real /v1/chat/completions
	// decode panics out-of-vocab before the loop (verified, not the #12331 surface).
	// The routing guard asserts the seam the header stamp reads.
	if !srv.isMetalMTPActive("local") {
		t.Fatal("expected isMetalMTPActive('local') == true")
	}

	// Remote arm (the REAL HTTP RED baseline for #12331): the proxy-routed request,
	// served end-to-end by the MockPlanner proxy side, must NOT carry the local
	// planner's mtp-metal speculative header.
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	payload := `{"model":"remote-model","messages":[{"role":"user","content":"hi"}],"temperature":0}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("remote POST /v1/chat/completions: %v", err)
	}
	body, bodyErr := io.ReadAll(resp.Body)
	if bodyErr != nil {
		t.Fatalf("remote arm: read body: %v", bodyErr)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("remote arm: status = %d, want 200; body = %s", resp.StatusCode, string(body))
	}
	if got := resp.Header.Get(HeaderSpeculative); got != "" {
		t.Fatalf("remote must not inherit local Metal MTP header: %s = %q, want empty", HeaderSpeculative, got)
	}
}
