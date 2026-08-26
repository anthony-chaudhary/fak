package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestChatDecodeTraceProductionSchemaAndEngine(t *testing.T) {
	srv := nativeReceiptServer(t)
	body := `{"model":"synthetic-live","messages":[{"role":"user","content":"trace"}],"max_tokens":4,"fak_decode_trace":true}`
	rr := postNativeReceipt(t, srv, body)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Fak == nil || resp.Fak.DecodeTrace == nil {
		t.Fatalf("response missing fak.decode_trace: %s", rr.Body.String())
	}
	trace := resp.Fak.DecodeTrace
	if trace.Schema != agent.NativeDecodeTraceSchema || trace.Engine != agent.NativeDecodeTraceEngine {
		t.Fatalf("trace provenance = schema %q engine %q", trace.Schema, trace.Engine)
	}
	if len(trace.Events) != resp.Usage.CompletionTokens || len(trace.Events) != 4 {
		t.Fatalf("events/completion_tokens = %d/%d, want 4/4", len(trace.Events), resp.Usage.CompletionTokens)
	}
	for i, event := range trace.Events {
		if event.TokenIndex != i+1 || event.ElapsedNS < 0 || (i > 0 && event.ElapsedNS < trace.Events[i-1].ElapsedNS) {
			t.Fatalf("event[%d] = %+v, want consecutive monotonic trace", i, event)
		}
	}
}

func TestChatDecodeTraceDefaultOffDoesNotExposePlannerTrace(t *testing.T) {
	srv := newTestServer(t)
	planner := &chatDecodeTraceCountingPlanner{native: true}
	srv.planner = planner
	rr := postNativeReceipt(t, srv, `{"messages":[{"role":"user","content":"x"}]}`)
	if rr.Code != http.StatusOK || planner.calls != 1 || bytes.Contains(rr.Body.Bytes(), []byte(`"decode_trace"`)) {
		t.Fatalf("status/calls/body = %d/%d/%s, want 200/1/no trace", rr.Code, planner.calls, rr.Body.String())
	}
}

type chatDecodeTraceCountingPlanner struct {
	native bool
	calls  int
}

func (p *chatDecodeTraceCountingPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		DecodeTrace: &agent.NativeDecodeTrace{
			Schema: agent.NativeDecodeTraceSchema,
			Engine: agent.NativeDecodeTraceEngine,
		},
	}, nil
}

func (*chatDecodeTraceCountingPlanner) Model() string { return "counting" }
func (p *chatDecodeTraceCountingPlanner) NativeDecodeTraceSupported() bool {
	return p.native
}

func TestChatDecodeTraceRejectsStreamAndProxyBeforeInference(t *testing.T) {
	cases := []struct {
		name   string
		native bool
		body   string
	}{
		{name: "stream", native: true, body: `{"messages":[{"role":"user","content":"x"}],"stream":true,"fak_decode_trace":true}`},
		{name: "proxy", native: false, body: `{"messages":[{"role":"user","content":"x"}],"fak_decode_trace":true}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			planner := &chatDecodeTraceCountingPlanner{native: tc.native}
			srv.planner = planner
			rr := postNativeReceipt(t, srv, tc.body)
			if rr.Code != http.StatusBadRequest || planner.calls != 0 || bytes.Contains(rr.Body.Bytes(), []byte(`"decode_trace"`)) {
				t.Fatalf("status/calls/body = %d/%d/%s, want 400/0/no trace", rr.Code, planner.calls, rr.Body.String())
			}
		})
	}
}

func TestChatDecodeTraceDualProxyRouteRejectsBeforeInference(t *testing.T) {
	srv := newTestServer(t)
	proxy := &chatDecodeTraceCountingPlanner{native: false}
	local := &chatDecodeTraceCountingPlanner{native: true}
	dual, err := NewDualPlanner(proxy, local, "local-model")
	if err != nil {
		t.Fatal(err)
	}
	srv.planner = dual
	rr := postNativeReceipt(t, srv, `{"model":"remote-model","messages":[{"role":"user","content":"x"}],"fak_decode_trace":true}`)
	if rr.Code != http.StatusBadRequest || proxy.calls != 0 || local.calls != 0 {
		t.Fatalf("status/proxy/local calls = %d/%d/%d, want 400/0/0", rr.Code, proxy.calls, local.calls)
	}
}
