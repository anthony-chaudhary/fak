package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type stopLimitPlanner struct {
	calls int
	stop  []string
}

func (p *stopLimitPlanner) Model() string { return "recording" }

func (p *stopLimitPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.stop = append([]string(nil), sp.Stop...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (p *stopLimitPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	comp, err := p.Complete(ctx, messages, tools, opts...)
	if err == nil && sink != nil {
		err = sink(comp.Message.Content)
	}
	return comp, err
}

func stopLimitRequest(t *testing.T, wire string, stream bool, stop any) *http.Request {
	t.Helper()
	var path string
	body := map[string]any{"model": "m", "max_tokens": 16, "stream": stream}
	switch wire {
	case "chat":
		path = "/v1/chat/completions"
		body["messages"] = []map[string]string{{"role": "user", "content": "hi"}}
		body["stop"] = stop
	case "completions":
		path = "/v1/completions"
		body["prompt"] = "hi"
		body["stop"] = stop
	case "messages":
		path = "/v1/messages"
		body["messages"] = []map[string]string{{"role": "user", "content": "hi"}}
		body["stop_sequences"] = stop
	case "gemini":
		path = "/v1beta/models/m:generateContent"
		if stream {
			path = "/v1beta/models/m:streamGenerateContent"
		}
		body = map[string]any{
			"contents":         []any{map[string]any{"role": "user", "parts": []map[string]string{{"text": "hi"}}}},
			"generationConfig": map[string]any{"maxOutputTokens": 16, "stopSequences": stop},
		}
	default:
		t.Fatalf("unknown wire %q", wire)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(string(raw)))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func requireStopLimitError(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	if rr.Code != http.StatusBadRequest || strings.Contains(rr.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("expected pre-stream 400: status=%d headers=%v body=%s", rr.Code, rr.Header(), rr.Body.String())
	}
	var body struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Type != "invalid_request_error" || body.Error.Code != "parameter_out_of_range" {
		t.Fatalf("unexpected error envelope: %s", rr.Body.String())
	}
}

// fak-test:runtime medium est=2s lane=default
func TestStopCountFourWireBoundary(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, wire := range []string{"chat", "completions", "messages", "gemini"} {
		for _, stream := range []bool{false, true} {
			for _, n := range []int{0, 1, 32, 33} {
				t.Run(fmt.Sprintf("%s/stream=%t/count=%d", wire, stream, n), func(t *testing.T) {
					// Empty entries and duplicates still consume the cardinality budget;
					// Unicode, whitespace and boundary-length literal bytes must remain unchanged.
					stops := make([]string, n)
					for i := range stops {
						stops[i] = []string{"", "\u00a0終", " STOP ", strings.Repeat("x", 256)}[i%4]
					}
					srv := newTestServer(t)
					p := &stopLimitPlanner{}
					srv.planner = p
					rr := httptest.NewRecorder()
					srv.Handler().ServeHTTP(rr, stopLimitRequest(t, wire, stream, stops))
					if n == 33 {
						requireStopLimitError(t, rr)
						if p.calls != 0 {
							t.Fatalf("rejected request reached planner %d times", p.calls)
						}
						return
					}
					if rr.Code != http.StatusOK || p.calls != 1 || !slices.Equal(p.stop, stops) {
						t.Fatalf("accepted behavior changed: status=%d calls=%d stop=%q want=%q body=%s", rr.Code, p.calls, p.stop, stops, rr.Body.String())
					}
				})
			}
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestStopCountRejectsBeforeHTTPProxyAndNativeLoop(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	var calls atomic.Int32
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(up.Close)
	for _, wire := range []string{"chat", "completions", "messages", "gemini"} {
		for _, stream := range []bool{false, true} {
			srv := newTestServer(t)
			srv.planner = agent.NewHTTPPlanner(up.URL+"/v1", "m", "")
			rr := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rr, stopLimitRequest(t, wire, stream, make([]string, 33)))
			requireStopLimitError(t, rr)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("rejected requests reached HTTP proxy %d times", calls.Load())
	}
	for _, stream := range []bool{false, true} {
		srv := newTestServer(t)
		p := &stopLimitPlanner{}
		srv.planner, srv.native = p, true
		rr := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rr, stopLimitRequest(t, "messages", stream, make([]string, 33)))
		requireStopLimitError(t, rr)
		if p.calls != 0 {
			t.Fatal("native loop started for rejected request")
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestStopCountPreservesWireShapeCompatibility(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, wire := range []string{"chat", "completions", "messages", "gemini"} {
		for _, raw := range []string{`null`, `17`, `{}`, `["ok",17]`, `"  終  "`, "[" + strings.Repeat(`"ok",`, 32) + "17]"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", wire, raw, stream), func(t *testing.T) {
					srv := newTestServer(t)
					p := &stopLimitPlanner{}
					srv.planner = p
					rr := httptest.NewRecorder()
					srv.Handler().ServeHTTP(rr, stopLimitRequest(t, wire, stream, json.RawMessage(raw)))
					if (wire == "messages" || wire == "gemini") && raw != `null` {
						if rr.Code != http.StatusBadRequest || p.calls != 0 {
							t.Fatalf("typed wire accepted malformed stops: status=%d calls=%d", rr.Code, p.calls)
						}
						return
					}
					want := normalizeStop(json.RawMessage(raw))
					if rr.Code != http.StatusOK || p.calls != 1 || !slices.Equal(p.stop, want) {
						t.Fatalf("legacy stop shape changed: status=%d calls=%d got=%q want=%q", rr.Code, p.calls, p.stop, want)
					}
				})
			}
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestStopCountPrecedesRosterlessEPFanout(t *testing.T) {
	type delivery struct {
		body, path, follower string
	}
	deliveries := make(chan delivery, 16)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read follower body: %v", err)
		}
		deliveries <- delivery{string(raw), r.URL.Path, r.Header.Get(epFollowerHeader)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(follower.Close)
	t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
	for _, wire := range []string{"chat", "completions", "messages", "gemini"} {
		for _, stream := range []bool{false, true} {
			for _, n := range []int{33, 32} {
				t.Run(fmt.Sprintf("%s/stream=%t/count=%d", wire, stream, n), func(t *testing.T) {
					srv := newTestServer(t)
					if srv.roster != nil {
						t.Fatal("fixture must exercise the rosterless fanout route")
					}
					p := &stopLimitPlanner{}
					srv.planner = p
					req := stopLimitRequest(t, wire, stream, make([]string, n))
					raw, err := io.ReadAll(req.Body)
					if err != nil {
						t.Fatal(err)
					}
					// Formatting is intentionally noncanonical: fanout must retain the
					// exact original wire rather than remarshal the decoded request.
					original := " \n" + string(raw) + "\n "
					req.Body = io.NopCloser(strings.NewReader(original))
					rr := httptest.NewRecorder()
					srv.Handler().ServeHTTP(rr, req)
					// The handler's deferred EP join completes before these assertions.
					if n == 33 {
						requireStopLimitError(t, rr)
						if p.calls != 0 || len(deliveries) != 0 {
							t.Fatalf("rejected request dispatched: planner=%d followers=%d", p.calls, len(deliveries))
						}
						return
					}
					if rr.Code != http.StatusOK || p.calls != 1 || len(deliveries) != 1 {
						t.Fatalf("accepted fanout: status=%d planner=%d followers=%d body=%s", rr.Code, p.calls, len(deliveries), rr.Body.String())
					}
					got := <-deliveries
					wantPath := req.URL.Path
					if wire == "gemini" {
						wantPath = epRouteGeminiGenerateContent
					}
					if got.body != original || got.path != wantPath || got.follower != "1" {
						t.Fatalf("follower payload changed: %+v wantPath=%q", got, wantPath)
					}
				})
			}
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestStopLiteralByteBoundaryAcrossWiresAndEP(t *testing.T) {
	deliveries := make(chan string, 128)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read follower: %v", err)
		}
		deliveries <- string(raw)
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(follower.Close)
	for _, ep := range []bool{false, true} {
		for _, wire := range []string{"chat", "completions", "messages", "gemini"} {
			for _, stream := range []bool{false, true} {
				for _, unicode := range []bool{false, true} {
					for _, size := range []int{256, 257} {
						t.Run(fmt.Sprintf("ep=%t/%s/stream=%t/unicode=%t/bytes=%d", ep, wire, stream, unicode, size), func(t *testing.T) {
							t.Setenv("FAK_EP_FANOUT_ADDRS", "")
							if ep {
								t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
							}
							stop := strings.Repeat("x", 256)
							if unicode {
								stop = strings.Repeat("é", 128)
							}
							if size == 257 {
								stop += "a"
							}
							stops := []string{"", stop, stop, " END "}
							srv := newTestServer(t)
							if srv.roster != nil {
								t.Fatal("expected rosterless fixture")
							}
							p := &stopLimitPlanner{}
							srv.planner = p
							req := stopLimitRequest(t, wire, stream, stops)
							raw, err := io.ReadAll(req.Body)
							if err != nil {
								t.Fatal(err)
							}
							original := " \n" + string(raw) + "\n "
							req.Body = io.NopCloser(strings.NewReader(original))
							rr := httptest.NewRecorder()
							srv.Handler().ServeHTTP(rr, req)
							if size == 257 {
								requireStopLimitError(t, rr)
								if p.calls != 0 || len(deliveries) != 0 || !strings.Contains(rr.Body.String(), "257 bytes") {
									t.Fatalf("oversize dispatched or misclassified: planner=%d followers=%d body=%s", p.calls, len(deliveries), rr.Body.String())
								}
								return
							}
							if rr.Code != http.StatusOK || p.calls != 1 || !slices.Equal(p.stop, stops) {
								t.Fatalf("256-byte stop changed: status=%d calls=%d stop=%q body=%s", rr.Code, p.calls, p.stop, rr.Body.String())
							}
							wantDeliveries := 0
							if ep {
								wantDeliveries = 1
							}
							if len(deliveries) != wantDeliveries {
								t.Fatalf("follower deliveries=%d want=%d", len(deliveries), wantDeliveries)
							}
							if ep && <-deliveries != original {
								t.Fatal("accepted fanout changed original bytes")
							}
						})
					}
				}
			}
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestStopLiteralBytesUseDecodedOpenAIString(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, wire := range []string{"chat", "completions"} {
		for _, stream := range []bool{false, true} {
			for _, extra := range []string{"", "a"} {
				for _, array := range []bool{false, true} {
					raw := `"` + strings.Repeat(`\u00e9`, 128) + extra + `"`
					if array {
						raw = "[" + raw + "]"
					}
					srv := newTestServer(t)
					p := &stopLimitPlanner{}
					srv.planner = p
					rr := httptest.NewRecorder()
					srv.Handler().ServeHTTP(rr, stopLimitRequest(t, wire, stream, json.RawMessage(raw)))
					if extra != "" {
						requireStopLimitError(t, rr)
						if p.calls != 0 {
							t.Fatal("oversize bare/array stop reached planner")
						}
					} else if rr.Code != http.StatusOK || p.calls != 1 || !slices.Equal(p.stop, []string{strings.Repeat("é", 128)}) {
						t.Fatalf("JSON encoding length used instead of decoded bytes: status=%d stop=%q", rr.Code, p.stop)
					}
				}
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestStopLimitsCountErrorPrecedesLengthError(t *testing.T) {
	stops := make([]string, 33)
	stops[0] = strings.Repeat("x", 257)
	rr := httptest.NewRecorder()
	if !rejectStopLimits(rr, stops) || !strings.Contains(rr.Body.String(), "got 33") || strings.Contains(rr.Body.String(), "257 bytes") {
		t.Fatalf("count-first contract changed: %s", rr.Body.String())
	}
	requireStopLimitError(t, rr)
}
