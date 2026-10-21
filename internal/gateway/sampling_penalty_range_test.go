package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type samplingRangePlanner struct {
	mu    sync.Mutex
	calls int
	got   agent.SampleParams
}

func (p *samplingRangePlanner) record(opts []agent.SampleOpt) {
	var sp agent.SampleParams
	for _, opt := range opts {
		opt(&sp)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.got = sp
}

func (p *samplingRangePlanner) snapshot() (int, agent.SampleParams) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.got
}

func (p *samplingRangePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(opts)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (p *samplingRangePlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.record(opts)
	if sink != nil {
		if err := sink("ok"); err != nil {
			return nil, err
		}
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (*samplingRangePlanner) Model() string            { return "sampling-range" }
func (*samplingRangePlanner) StreamingSupported() bool { return true }

func postSamplingRangeChat(t *testing.T, url, fields string, stream bool) (int, string) {
	t.Helper()
	body := `{"model":"m",` + fields + `,"stream":` + strconv.FormatBool(stream) + `,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadRequest {
		return resp.StatusCode, ""
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("400 body is not an error envelope: %v", err)
	}
	return resp.StatusCode, env.Error.Code
}

// fak-test:runtime fast est=150ms lane=default
func TestChatCompletionsPenaltyOutOfRangeIsBadRequest(t *testing.T) {
	srv := newTestServer(t)
	planner := &samplingRangePlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, fields := range []string{
		`"presence_penalty":-1e39`,
		`"presence_penalty":2.5`,
		`"frequency_penalty":-2.01`,
		`"frequency_penalty":1e39`,
	} {
		for _, stream := range []bool{false, true} {
			status, code := postSamplingRangeChat(t, ts.URL, fields, stream)
			if status != http.StatusBadRequest || code != "sampling_param_out_of_range" {
				t.Errorf("%s stream=%t: status/code = %d/%q, want 400/%q", fields, stream, status, code, "sampling_param_out_of_range")
			}
		}
	}
	if calls, _ := planner.snapshot(); calls != 0 {
		t.Fatalf("out-of-range penalties reached the planner %d times, want 0", calls)
	}

	reached := 0
	for _, fields := range []string{
		`"presence_penalty":-2`,
		`"presence_penalty":2`,
		`"frequency_penalty":-2,"presence_penalty":0`,
		`"frequency_penalty":2`,
	} {
		for _, stream := range []bool{false, true} {
			if status, _ := postSamplingRangeChat(t, ts.URL, fields, stream); status != http.StatusOK {
				t.Errorf("%s stream=%t: status = %d, want 200", fields, stream, status)
			}
			reached++
			if calls, _ := planner.snapshot(); calls != reached {
				t.Fatalf("%s stream=%t: planner calls = %d, want %d", fields, stream, calls, reached)
			}
		}
	}
}

// fak-test:runtime fast est=60ms lane=default
func TestChatCompletionsExplicitTopPZeroReachesPlannerDistinctFromOmitted(t *testing.T) {
	srv := newTestServer(t)
	planner := &samplingRangePlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	for _, stream := range []bool{false, true} {
		if status, _ := postSamplingRangeChat(t, ts.URL, `"top_p":0`, stream); status != http.StatusOK {
			t.Fatalf("explicit top_p 0 stream=%t: status = %d, want 200", stream, status)
		}
		if _, got := planner.snapshot(); got.TopP == nil || *got.TopP != 0 {
			t.Fatalf("explicit top_p 0 stream=%t: planner top_p = %v, want explicit 0", stream, got.TopP)
		}
		if status, _ := postSamplingRangeChat(t, ts.URL, `"temperature":1`, stream); status != http.StatusOK {
			t.Fatalf("omitted top_p stream=%t: status = %d, want 200", stream, status)
		}
		if _, got := planner.snapshot(); got.TopP != nil {
			t.Fatalf("omitted top_p stream=%t: planner top_p = %v, want unset", stream, *got.TopP)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpeculativeNonFiniteLogitsClassification(t *testing.T) {
	status, code, message := upstreamErrorStatus(fmt.Errorf("verify round: %w", model.ErrSpeculativeNonFiniteLogits))
	if status != http.StatusInternalServerError || code != "native_sampler_nonfinite_logits" {
		t.Fatalf("classification %d/%s, want 500/native_sampler_nonfinite_logits", status, code)
	}
	if message == "" || strings.Contains(message, "verify round") {
		t.Fatalf("unsafe error detail: %q", message)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNativeSamplingModeUnsupportedIsBadRequest(t *testing.T) {
	status, code, message := upstreamErrorStatus(fmt.Errorf("lane detail: %w", model.ErrSamplingModeUnsupported))
	if status != http.StatusBadRequest || code != "sampling_mode_unsupported" {
		t.Fatalf("classification %d/%s, want 400/sampling_mode_unsupported", status, code)
	}
	if message == "" || strings.Contains(message, "lane detail") {
		t.Fatalf("unsafe error detail: %q", message)
	}
}
