package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

const completionsSamplingAll = `"seed":7,"logit_bias":{"42":-100,"7":2.5},"frequency_penalty":0.5,"presence_penalty":-1.25,"top_k":40,"min_p":0.05`

type completionsSamplingView struct {
	Seed             *int64
	LogitBias        map[int]float64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	TopK             *int
	ChatWireTopK     bool
	MinP             *float64
}

func completionsSamplingViewOf(sp agent.SampleParams) completionsSamplingView {
	return completionsSamplingView{
		Seed:             sp.Seed,
		LogitBias:        sp.LogitBias,
		FrequencyPenalty: sp.FrequencyPenalty,
		PresencePenalty:  sp.PresencePenalty,
		TopK:             sp.TopK,
		ChatWireTopK:     sp.ChatWireTopK,
		MinP:             sp.MinP,
	}
}

func (v completionsSamplingView) String() string {
	return fmt.Sprintf("seed=%s logit_bias=%v frequency_penalty=%s presence_penalty=%s top_k=%s chat_wire_top_k=%t min_p=%s",
		completionsSamplingShow(v.Seed), v.LogitBias, completionsSamplingShow(v.FrequencyPenalty), completionsSamplingShow(v.PresencePenalty),
		completionsSamplingShow(v.TopK), v.ChatWireTopK, completionsSamplingShow(v.MinP))
}

func completionsSamplingShow[T any](p *T) string {
	if p == nil {
		return "nil"
	}
	return fmt.Sprint(*p)
}

func completionsSamplingPtr[T any](v T) *T { return &v }

type completionsSamplingRecorder struct {
	mu    sync.Mutex
	calls int
	got   agent.SampleParams
}

func (p *completionsSamplingRecorder) Model() string { return "recording" }

func (p *completionsSamplingRecorder) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	p.calls++
	p.got = sp
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (p *completionsSamplingRecorder) snapshot() (int, agent.SampleParams) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.got
}

// fak-test:runtime fast est=80ms lane=default
func TestCompletionsSamplingCarriersReachPlanner(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	cases := []struct {
		name  string
		extra string
		want  completionsSamplingView
	}{
		{
			name:  "all",
			extra: completionsSamplingAll,
			want: completionsSamplingView{
				Seed:             completionsSamplingPtr(int64(7)),
				LogitBias:        map[int]float64{42: -100, 7: 2.5},
				FrequencyPenalty: completionsSamplingPtr(0.5),
				PresencePenalty:  completionsSamplingPtr(-1.25),
				TopK:             completionsSamplingPtr(40),
				ChatWireTopK:     true,
				MinP:             completionsSamplingPtr(0.05),
			},
		},
		{
			name:  "zero",
			extra: `"seed":0,"frequency_penalty":0,"presence_penalty":0,"top_k":0,"min_p":0`,
			want: completionsSamplingView{
				Seed:             completionsSamplingPtr(int64(0)),
				FrequencyPenalty: completionsSamplingPtr(0.0),
				PresencePenalty:  completionsSamplingPtr(0.0),
				TopK:             completionsSamplingPtr(0),
				ChatWireTopK:     true,
				MinP:             completionsSamplingPtr(0.0),
			},
		},
		{
			name:  "topk_disable",
			extra: `"top_k":-1`,
			want:  completionsSamplingView{TopK: completionsSamplingPtr(-1), ChatWireTopK: true},
		},
		{
			name:  "bounds",
			extra: `"frequency_penalty":2,"presence_penalty":-2`,
			want:  completionsSamplingView{FrequencyPenalty: completionsSamplingPtr(2.0), PresencePenalty: completionsSamplingPtr(-2.0)},
		},
		{name: "empty_bias", extra: `"logit_bias":{}`},
		{name: "none"},
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				srv := newTestServer(t)
				legacy := &completionsSamplingRecorder{}
				srv.planner = legacy
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, tc.extra))
				calls, got := legacy.snapshot()
				if rr.Code != http.StatusOK || calls != 1 {
					t.Fatalf("status=%d calls=%d body=%s", rr.Code, calls, rr.Body.String())
				}

				chat := &completionsSamplingRecorder{}
				srv.planner = chat
				crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("m", stream, tc.extra))
				chatCalls, chatGot := chat.snapshot()
				if crr.Code != http.StatusOK || chatCalls != 1 {
					t.Fatalf("chat precondition: status=%d calls=%d body=%s", crr.Code, chatCalls, crr.Body.String())
				}

				view, chatView := completionsSamplingViewOf(got), completionsSamplingViewOf(chatGot)
				if !reflect.DeepEqual(chatView, tc.want) {
					t.Fatalf("chat precondition: sampling %v, want %v", chatView, tc.want)
				}
				if !reflect.DeepEqual(view, tc.want) {
					t.Fatalf("completions sampling %v, want chat's %v", view, tc.want)
				}
			})
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestCompletionsSamplingCarriersIngressRefusedBeforePlanner(t *testing.T) {
	deliveries := make(chan string, 64)
	follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		deliveries <- string(raw)
		_, _ = io.WriteString(w, `{}`)
	}))
	t.Cleanup(follower.Close)
	cases := []struct {
		name, extra string
		code        string
	}{
		{"frequency_penalty_high", `"frequency_penalty":2.5`, codeSamplingParamOutOfRange},
		{"frequency_penalty_huge", `"frequency_penalty":1e39`, codeSamplingParamOutOfRange},
		{"presence_penalty_low", `"presence_penalty":-2.01`, codeSamplingParamOutOfRange},
		{"presence_penalty_huge", `"presence_penalty":-1e39`, codeSamplingParamOutOfRange},
		{"logit_bias_non_integer_key", `"logit_bias":{"x":1}`, ""},
		{"seed_overflow", `"seed":9223372036854775808`, ""},
		{"top_k_fractional", `"top_k":1.5`, ""},
	}
	for _, ep := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, tc := range cases {
				t.Run(fmt.Sprintf("ep=%t/stream=%t/%s", ep, stream, tc.name), func(t *testing.T) {
					t.Setenv("FAK_EP_FANOUT_ADDRS", "")
					if ep {
						t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
					}
					for len(deliveries) > 0 {
						<-deliveries
					}
					srv := newTestServer(t)
					p := &completionsSamplingRecorder{}
					srv.planner = p
					rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, tc.extra))
					calls, _ := p.snapshot()
					if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != tc.code {
						t.Fatalf("status=%d code=%q, want 400 %q", rr.Code, chatErrorCode(t, rr.Body.Bytes()), tc.code)
					}
					if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
						t.Fatalf("Content-Type=%q, want application/json", ct)
					}
					if calls != 0 || len(deliveries) != 0 {
						t.Fatalf("planner calls=%d follower deliveries=%d, want 0 and 0", calls, len(deliveries))
					}
					t.Setenv("FAK_EP_FANOUT_ADDRS", "")
					crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("m", stream, tc.extra))
					if crr.Code != rr.Code || chatErrorCode(t, crr.Body.Bytes()) != tc.code {
						t.Fatalf("chat parity: status=%d code=%q, want %d %q", crr.Code, chatErrorCode(t, crr.Body.Bytes()), rr.Code, tc.code)
					}
				})
			}
		}
	}
}

// fak-test:runtime fast est=60ms lane=default
func TestCompletionsSamplingCarriersSeedUnsupportedIs400BeforeSSE(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	var upstreamHits atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamHits.Add(1)
	}))
	t.Cleanup(upstream.Close)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			srv := newTestServer(t)
			planner, err := agent.NewProviderHTTPPlanner(string(agent.ProviderOpenAIResponses), upstream.URL, "m", "key")
			if err != nil {
				t.Fatal(err)
			}
			srv.planner = planner
			before := upstreamHits.Load()
			rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, `"seed":3`))
			if rr.Code != http.StatusBadRequest || chatErrorCode(t, rr.Body.Bytes()) != "sampling_seed_unsupported" {
				t.Fatalf("status=%d code=%q, want 400 sampling_seed_unsupported", rr.Code, chatErrorCode(t, rr.Body.Bytes()))
			}
			if ct := rr.Header().Get("Content-Type"); strings.HasPrefix(ct, "text/event-stream") {
				t.Fatalf("Content-Type=%q, want a non-SSE refusal", ct)
			}
			if hits := upstreamHits.Load() - before; hits != 0 {
				t.Fatalf("upstream hits=%d, want 0", hits)
			}
			crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("m", stream, `"seed":3`))
			if crr.Code != rr.Code || chatErrorCode(t, crr.Body.Bytes()) != "sampling_seed_unsupported" {
				t.Fatalf("chat parity: status=%d code=%q, want %d sampling_seed_unsupported", crr.Code, chatErrorCode(t, crr.Body.Bytes()), rr.Code)
			}
		})
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestCompletionsSamplingCarriersRideForwardsVerbatim(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	var mu sync.Mutex
	var forwarded map[string]json.RawMessage
	hits := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		mu.Lock()
		forwarded = body
		hits++
		mu.Unlock()
		if string(body["stream"]) == "true" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"model":"m","choices":[{"delta":{"role":"assistant","content":"ab"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"ab"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	t.Cleanup(upstream.Close)
	srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	type wire struct {
		Seed             *int64             `json:"seed"`
		LogitBias        map[string]float64 `json:"logit_bias"`
		FrequencyPenalty *float64           `json:"frequency_penalty"`
		PresencePenalty  *float64           `json:"presence_penalty"`
		TopK             *int               `json:"top_k"`
		MinP             *float64           `json:"min_p"`
	}
	keys := []string{"seed", "logit_bias", "frequency_penalty", "presence_penalty", "top_k", "min_p"}
	all := wire{
		Seed:             completionsSamplingPtr(int64(7)),
		LogitBias:        map[string]float64{"42": -100, "7": 2.5},
		FrequencyPenalty: completionsSamplingPtr(0.5),
		PresencePenalty:  completionsSamplingPtr(-1.25),
		TopK:             completionsSamplingPtr(40),
		MinP:             completionsSamplingPtr(0.05),
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			name  string
			extra string
			want  *wire
		}{
			{"all", completionsSamplingAll, &all},
			{"absent", "", nil},
		} {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tc.name), func(t *testing.T) {
				mu.Lock()
				before := hits
				mu.Unlock()
				rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("m", stream, tc.extra))
				if rr.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
				}
				mu.Lock()
				got, calls := forwarded, hits-before
				mu.Unlock()
				if calls != 1 {
					t.Fatalf("upstream hits=%d, want 1", calls)
				}
				if tc.want == nil {
					for _, key := range keys {
						if _, present := got[key]; present {
							t.Fatalf("upstream carries %s for a request that omitted it (body keys %v)", key, slices.Sorted(maps.Keys(got)))
						}
					}
					return
				}
				raw, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				var sent wire
				if err := json.Unmarshal(raw, &sent); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(sent, *tc.want) {
					t.Fatalf("upstream body %s lacks the client sampling fields verbatim", raw)
				}
			})
		}
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestCompletionsSamplingCarriersNativeMinPMatchesChat(t *testing.T) {
	srv := nativeReceiptServer(t)
	const extra = `"min_p":1.5`
	crr := completionsStructuredPost(t, srv, "/v1/chat/completions", completionsStructuredChatBody("synthetic-live", false, extra))
	chatCode := chatErrorCode(t, crr.Body.Bytes())
	if crr.Code != http.StatusBadRequest || chatCode != "sampling_mode_unsupported" {
		t.Fatalf("chat precondition: status=%d code=%q, want 400 sampling_mode_unsupported", crr.Code, chatCode)
	}
	rr := completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("synthetic-live", false, extra))
	if rr.Code != crr.Code || chatErrorCode(t, rr.Body.Bytes()) != chatCode {
		t.Fatalf("buffered status=%d code=%q, want chat %d %s", rr.Code, chatErrorCode(t, rr.Body.Bytes()), crr.Code, chatCode)
	}
	rr = completionsStructuredPost(t, srv, "/v1/completions", completionsStructuredBody("synthetic-live", true, extra))
	if rr.Code != http.StatusOK || !strings.HasPrefix(rr.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream status=%d Content-Type=%q, want 200 text/event-stream", rr.Code, rr.Header().Get("Content-Type"))
	}
	text, _, errCode, done := completionsStructuredReadStream(t, rr.Body.String())
	if errCode != chatCode || !done || text != "" {
		t.Fatalf("in-band error=%q done=%t text=%q, want %s, [DONE] and no text", errCode, done, text, chatCode)
	}
}
