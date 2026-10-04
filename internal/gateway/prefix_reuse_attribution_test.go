package gateway

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// scopeCapturePlanner records the prefix-cache identity each planner call resolves and
// replays scripted in-kernel usage (prompt, matched) so the served path can attribute it.
type scopeCapturePlanner struct {
	mu     sync.Mutex
	scopes []capturedScope
	usage  [][2]int
}

type capturedScope struct {
	path  string
	owner radixkv.CacheIdentity
	ok    bool
}

func (p *scopeCapturePlanner) Model() string            { return "test-model" }
func (p *scopeCapturePlanner) StreamingSupported() bool { return true }

func (p *scopeCapturePlanner) record(ctx context.Context, path string) *agent.Completion {
	owner, ok := agent.PrefixCacheIdentityFromContext(ctx)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.scopes = append(p.scopes, capturedScope{path: path, owner: owner, ok: ok})
	prompt, matched := 100, 0
	if len(p.usage) > 0 {
		prompt, matched = p.usage[0][0], p.usage[0][1]
		p.usage = p.usage[1:]
	}
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: inKernelProducer, ModelID: "test-model", PromptTokens: int64(prompt), CachedTokens: int64(matched)})
	return &agent.Completion{
		Model:         "test-model",
		Message:       agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason:  "stop",
		ProviderCache: &entry,
		Usage: agent.Usage{PromptTokens: prompt, CompletionTokens: 10, TotalTokens: prompt + 10,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: matched}},
	}
}

func (p *scopeCapturePlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.record(ctx, "buffered"), nil
}

func (p *scopeCapturePlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	comp := p.record(ctx, "streamed")
	if sink != nil {
		if err := sink(comp.Message.Content); err != nil {
			return nil, err
		}
	}
	return comp, nil
}

var _ agent.StreamingPlanner = (*scopeCapturePlanner)(nil)

func principalTestServer(t *testing.T, planner agent.Planner, principal string) *httptest.Server {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = planner
	h := srv.Handler()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r.WithContext(WithPrincipal(r.Context(), principal)))
	}))
	t.Cleanup(ts.Close)
	return ts
}

func postPrefixReuse(t *testing.T, url, body string, headers map[string]string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBufferString(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST %s status=%d", url, resp.StatusCode)
	}
}

// TestStreamedAndBufferedTurnsShareOnePrefixCacheScope is the regression for the cache-scope
// split: a buffered and a streamed request from the same principal must resolve to the SAME
// tenant cache scope, on every wire, or a streamed coordinator and its buffered subagents
// land in disjoint radix namespaces and never share a prefix.
//
// fak-test:runtime fast est=2s
func TestStreamedAndBufferedTurnsShareOnePrefixCacheScope(t *testing.T) {
	planner := &scopeCapturePlanner{}
	ts := principalTestServer(t, planner, "tenant-a")
	user := `[{"role":"user","content":"probe"}]`
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", `{"model":"test-model","messages":`+user+`}`, nil)
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", `{"model":"test-model","stream":true,"messages":`+user+`}`, nil)
	postPrefixReuse(t, ts.URL+"/v1/messages", `{"model":"test-model","max_tokens":16,"stream":true,"messages":`+user+`}`, nil)

	planner.mu.Lock()
	defer planner.mu.Unlock()
	want := radixkv.CacheIdentity{Tenant: "tenant-a"}
	paths := map[string]bool{}
	for _, got := range planner.scopes {
		paths[got.path] = true
		if !got.ok || got.owner != want {
			t.Fatalf("%s planner call resolved scope=%+v ok=%v, want %+v", got.path, got.owner, got.ok, want)
		}
	}
	if !paths["buffered"] || !paths["streamed"] {
		t.Fatalf("planner paths exercised = %v, want both buffered and streamed", paths)
	}
}

// TestPrefixReuseAttributionSplitsSameAndCrossSession drives a coordinator and a subagent
// of one harness launch through the served path and asserts /metrics separates the
// subagent's reuse of the coordinator's prefix (cross_session) from the coordinator's reuse
// of its own earlier turn (same_session), and that the cross-agent ledger is fed.
//
// fak-test:runtime fast est=2s
func TestPrefixReuseAttributionSplitsSameAndCrossSession(t *testing.T) {
	planner := &scopeCapturePlanner{usage: [][2]int{
		{100, 0},   // coordinator turn 1: cold
		{80, 60},   // subagent first turn: 60 tokens it never prefilled itself
		{150, 110}, // coordinator turn 2: within its own 110-token footprint
	}}
	ts := principalTestServer(t, planner, "tenant-a")
	hdr := map[string]string{"X-Fak-Session-Id": "launch-1"}
	sys := `{"role":"system","content":"shared harness prompt"}`
	coord := `{"role":"user","content":"coordinate"}`
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", `{"model":"test-model","messages":[`+sys+`,`+coord+`]}`, hdr)
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", `{"model":"test-model","stream":true,"messages":[`+sys+`,{"role":"user","content":"subtask"}]}`, hdr)
	postPrefixReuse(t, ts.URL+"/v1/chat/completions", `{"model":"test-model","messages":[`+sys+`,`+coord+`,{"role":"assistant","content":"ok"},{"role":"user","content":"next"}]}`, hdr)

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	lines := map[string]bool{}
	for _, l := range strings.Split(string(body), "\n") {
		lines[strings.TrimSpace(l)] = true
	}
	for _, want := range []string{
		`fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin="same_session"} 110`,
		`fak_gateway_kv_prefix_reused_tokens_by_origin_total{origin="cross_session"} 60`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="same_session"} 1`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="cross_session"} 1`,
		`fak_gateway_kv_prefix_turns_by_origin_total{origin="cold"} 1`,
		`fak_gateway_kv_prefix_cross_agent_coordinators 1`,
		`fak_gateway_kv_prefix_cross_agent_subagent_turns 1`,
		`fak_gateway_kv_prefix_cross_agent_shared_tokens 60`,
		`fak_gateway_kv_prefix_cross_agent_prompt_tokens 80`,
	} {
		if !lines[want] {
			t.Errorf("/metrics missing sample %q", want)
		}
	}
}

// TestPrefixReuseAttributionClampsAndSkipsUnattributed pins the ledger contract: without a
// harness session id the origin split still counts, but no coordinator is invented.
//
// fak-test:runtime fast est=50ms
func TestPrefixReuseAttributionClampsAndSkipsUnattributed(t *testing.T) {
	var a prefixReuseAttribution
	if got := a.observe("", 7, 50, 999, 0); got != prefixReuseCrossSession {
		t.Fatalf("first turn with reuse origin=%v, want cross", got)
	}
	if got := a.observe("", 7, 0, 10, 0); got != prefixReuseCold {
		t.Fatalf("zero-prompt turn origin=%v, want cold", got)
	}
	snap := a.snapshot()
	if snap.crossTokens != 50 || snap.sameTokens != 0 || snap.coordinators != 0 {
		t.Fatalf("snapshot=%+v, want cross=50 (clamped) same=0 coordinators=0", snap)
	}
}
