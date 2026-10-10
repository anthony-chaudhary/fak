package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type guidedRegexLimitPlanner struct {
	calls  int
	guided map[string]json.RawMessage
}

func (p *guidedRegexLimitPlanner) Model() string { return "recording" }
func (p *guidedRegexLimitPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.guided = sp.GuidedDecode
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}
func (p *guidedRegexLimitPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	c, err := p.Complete(ctx, msgs, tools, opts...)
	if err == nil && sink != nil {
		err = sink(c.Message.Content)
	}
	return c, err
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=4s lane=default
func TestGuidedRegexDecodedByteLimitsBeforeEP(t *testing.T) {
	deliveries := make(chan string, 128)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		deliveries <- string(raw)
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(follower.Close)
	for _, ep := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, key := range []string{"guided_regex", "regex"} {
				for _, encoding := range []string{"ascii", "unicode", "escaped", "escaped-ascii"} {
					for _, size := range []int{4096, 4097} {
						t.Run(fmt.Sprintf("ep=%t/stream=%t/%s/%s/%d", ep, stream, key, encoding, size), func(t *testing.T) {
							t.Setenv("FAK_EP_FANOUT_ADDRS", "")
							if ep {
								t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
							}
							encoded := strings.Repeat("x", 4096)
							if encoding == "unicode" {
								encoded = strings.Repeat("é", 2048)
							}
							if encoding == "escaped" {
								encoded = strings.Repeat(`\u00e9`, 2048)
							}
							if encoding == "escaped-ascii" {
								encoded = strings.Repeat(`\u0078`, 4096)
							}
							if size == 4097 {
								encoded += "a"
							}
							rawPattern := `"` + encoded + `"`
							body := fmt.Sprintf(` {"model":"m","messages":[{"role":"user","content":"hi"}],"stream":%t,"%s":%s} `, stream, key, rawPattern)
							srv := newTestServer(t)
							if srv.roster != nil {
								t.Fatal("expected rosterless EP fixture")
							}
							p := &guidedRegexLimitPlanner{}
							srv.planner = p
							rr := httptest.NewRecorder()
							srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
							if size == 4097 {
								requireStopLimitError(t, rr)
								if p.calls != 0 || len(deliveries) != 0 || !strings.Contains(rr.Body.String(), key) || !strings.Contains(rr.Body.String(), "4097 bytes") {
									t.Fatalf("oversize dispatched/misclassified: calls=%d deliveries=%d body=%s", p.calls, len(deliveries), rr.Body.String())
								}
								return
							}
							if rr.Code != http.StatusOK || p.calls != 1 || string(p.guided[key]) != rawPattern {
								t.Fatalf("accepted raw carrier changed: status=%d calls=%d guided=%s body=%s", rr.Code, p.calls, p.guided[key], rr.Body.String())
							}
							want := 0
							if ep {
								want = 1
							}
							if len(deliveries) != want {
								t.Fatalf("deliveries=%d want=%d", len(deliveries), want)
							}
							if ep && <-deliveries != body {
								t.Fatal("accepted EP body changed")
							}
						})
					}
				}
			}
		}
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=2s lane=default
func TestGuidedRegexLimitsPreserveLegacyShapesAndOtherGuidance(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, stream := range []bool{false, true} {
		for _, raw := range []string{`null`, `17`, `{}`, `["pattern"]`} {
			body := fmt.Sprintf(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":%t,"guided_regex":%s,"regex":%s,"guided_grammar":%q}`, stream, raw, raw, strings.Repeat("x", 4097))
			srv := newTestServer(t)
			p := &guidedRegexLimitPlanner{}
			srv.planner = p
			rr := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
			if rr.Code != http.StatusOK || p.calls != 1 || string(p.guided["guided_regex"]) != raw || string(p.guided["regex"]) != raw {
				t.Fatalf("legacy shape changed: status=%d guided=%v body=%s", rr.Code, p.guided, rr.Body.String())
			}
			var grammar string
			if err := json.Unmarshal(p.guided["guided_grammar"], &grammar); err != nil || grammar != strings.Repeat("x", 4097) {
				t.Fatal("unrelated guidance altered")
			}
		}
	}
}
