package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestClientServiceTierParse(t *testing.T) {
	cases := []struct {
		raw     string
		present bool
		mode    modelroute.ServiceMode
		flex    bool
		echo    string
	}{
		{"", false, modelroute.ServiceModeStandard, false, ""},
		{"auto", true, modelroute.ServiceModeStandard, false, "default"},
		{"default", true, modelroute.ServiceModeStandard, false, "default"},
		{"standard", true, modelroute.ServiceModeStandard, false, "default"},
		{"scale", true, modelroute.ServiceModeStandard, false, "default"},
		{"priority", true, modelroute.ServiceModeFast, false, "priority"},
		{"fast", true, modelroute.ServiceModeFast, false, "priority"},
		{"Priority", true, modelroute.ServiceModeFast, false, "priority"},
		{"flex", true, modelroute.ServiceModeStandard, true, "default"},
	}
	for _, tc := range cases {
		got, err := parseClientServiceTier(tc.raw)
		if err != nil {
			t.Fatalf("%q: %v", tc.raw, err)
		}
		if got.Present != tc.present || got.Mode != tc.mode || got.RequestedFlex != tc.flex {
			t.Fatalf("%q: got %+v", tc.raw, got)
		}
		if e := got.echo(modelroute.ServiceModeUnknown); e != tc.echo {
			t.Fatalf("%q: echo = %q, want %q", tc.raw, e, tc.echo)
		}
	}
	for _, bad := range []string{"turbo", "premium", " ", "priority2"} {
		if _, err := parseClientServiceTier(bad); !errors.Is(err, errInvalidServiceTier) {
			t.Fatalf("%q: err = %v, want errInvalidServiceTier", bad, err)
		}
	}
}

func TestClientServiceTierEchoPrefersRealizedMode(t *testing.T) {
	fast, _ := parseClientServiceTier("priority")
	if got := fast.echo(modelroute.ServiceModeStandard); got != "default" {
		t.Fatalf("downgraded echo = %q", got)
	}
	std, _ := parseClientServiceTier("default")
	if got := std.echo(modelroute.ServiceModeFast); got != "priority" {
		t.Fatalf("upgraded echo = %q", got)
	}
}

// tierCapturePlanner records the service mode each turn was sampled with.
type tierCapturePlanner struct {
	mu    sync.Mutex
	modes []modelroute.ServiceMode
}

func (p *tierCapturePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&sp)
		}
	}
	p.mu.Lock()
	p.modes = append(p.modes, sp.ServiceTier)
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (*tierCapturePlanner) Model() string { return "tier-fixture" }

func (p *tierCapturePlanner) last() modelroute.ServiceMode {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.modes) == 0 {
		return ""
	}
	return p.modes[len(p.modes)-1]
}

func postTierJSON(t *testing.T, url string, body map[string]any) (int, map[string]json.RawMessage, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	_ = json.Unmarshal(out, &fields)
	return resp.StatusCode, fields, out
}

func newTierServer(t *testing.T) (*tierCapturePlanner, *httptest.Server) {
	t.Helper()
	planner := &tierCapturePlanner{}
	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return planner, ts
}

func TestClientServiceTierChatEchoAndForward(t *testing.T) {
	planner, ts := newTierServer(t)
	messages := []map[string]any{{"role": "user", "content": "hi"}}

	code, fields, raw := postTierJSON(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "messages": messages, "service_tier": "priority"})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if string(fields["service_tier"]) != `"priority"` || planner.last() != modelroute.ServiceModeFast {
		t.Fatalf("service_tier=%s forwarded=%q body=%s", fields["service_tier"], planner.last(), raw)
	}

	code, fields, raw = postTierJSON(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "messages": messages, "service_tier": "flex"})
	if code != http.StatusOK || string(fields["service_tier"]) != `"default"` || planner.last() != modelroute.ServiceModeStandard {
		t.Fatalf("flex: status=%d service_tier=%s forwarded=%q body=%s", code, fields["service_tier"], planner.last(), raw)
	}

	// Negative control: a request without the field gets no echo and no tier.
	code, fields, raw = postTierJSON(t, ts.URL+"/v1/chat/completions", map[string]any{"model": "m", "messages": messages})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if _, ok := fields["service_tier"]; ok || bytes.Contains(raw, []byte("service_tier")) || planner.last() != "" {
		t.Fatalf("absent request leaked tier: forwarded=%q body=%s", planner.last(), raw)
	}
}

func TestClientServiceTierChatRejectsUnknown(t *testing.T) {
	planner, ts := newTierServer(t)
	code, fields, raw := postTierJSON(t, ts.URL+"/v1/chat/completions", map[string]any{
		"model": "m", "messages": []map[string]any{{"role": "user", "content": "hi"}}, "service_tier": "turbo",
	})
	assertInvalidServiceTier(t, code, fields, raw)
	if len(planner.modes) != 0 {
		t.Fatalf("planner ran for a rejected tier: %v", planner.modes)
	}
}

func TestClientServiceTierResponsesEchoAndForward(t *testing.T) {
	planner, ts := newTierServer(t)
	code, fields, raw := postTierJSON(t, ts.URL+"/v1/responses", map[string]any{"model": "m", "input": "hi", "service_tier": "fast"})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if string(fields["service_tier"]) != `"priority"` || planner.last() != modelroute.ServiceModeFast {
		t.Fatalf("service_tier=%s forwarded=%q body=%s", fields["service_tier"], planner.last(), raw)
	}

	code, fields, raw = postTierJSON(t, ts.URL+"/v1/responses", map[string]any{"model": "m", "input": "hi"})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if _, ok := fields["service_tier"]; ok || bytes.Contains(raw, []byte("service_tier")) || planner.last() != "" {
		t.Fatalf("absent request leaked tier: forwarded=%q body=%s", planner.last(), raw)
	}

	code, fields, raw = postTierJSON(t, ts.URL+"/v1/responses", map[string]any{"model": "m", "input": "hi", "service_tier": "turbo"})
	assertInvalidServiceTier(t, code, fields, raw)
}

func assertInvalidServiceTier(t *testing.T, code int, fields map[string]json.RawMessage, raw []byte) {
	t.Helper()
	if code != http.StatusBadRequest {
		t.Fatalf("status = %d: %s", code, raw)
	}
	var errBody struct {
		Code string `json:"code"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(fields["error"], &errBody); err != nil || errBody.Code != invalidServiceTierCode {
		t.Fatalf("error envelope = %s (%v)", raw, err)
	}
}
