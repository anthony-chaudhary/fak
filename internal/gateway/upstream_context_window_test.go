package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fakeLlamaUpstream serves llama-server-shaped /props and /slots and records the
// ?model= each probe carried.
type fakeLlamaUpstream struct {
	srv    *httptest.Server
	mu     sync.Mutex
	models []string
}

func newFakeLlamaUpstream(t *testing.T, props, slots string) *fakeLlamaUpstream {
	t.Helper()
	f := &fakeLlamaUpstream{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.models = append(f.models, r.URL.Path+"?"+r.URL.Query().Get("model"))
		f.mu.Unlock()
		body := ""
		switch r.URL.Path {
		case "/props":
			body = props
		case "/slots":
			body = slots
		}
		if body == "" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeLlamaUpstream) seen() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.models...)
}

type catalogWindows struct {
	data, models map[string]any
}

func catalogContextWindows(t *testing.T, s *Server) catalogWindows {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	var body struct {
		Data   []map[string]any `json:"data"`
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	out := catalogWindows{data: map[string]any{}, models: map[string]any{}}
	for _, row := range body.Data {
		out.data[row["id"].(string)] = row["context_window"]
	}
	for _, row := range body.Models {
		out.models[row["slug"].(string)] = row["context_window"]
	}
	return out
}

// TestUpstreamContextWindowAdvertisedPerModel: a proxy fronting several llama-server-class
// models probes each upstream with its own ?model=, keys the per-slot window by public
// model id, advertises it as context_window on both catalog shapes, and omits it where
// the upstream reports none.
//
// fak-test:runtime fast est=100ms
func TestUpstreamContextWindowAdvertisedPerModel(t *testing.T) {
	boot := newFakeLlamaUpstream(t, `{"total_slots":4,"default_generation_settings":{"n_ctx":32768}}`, "")
	routed := newFakeLlamaUpstream(t, `{"total_slots":2,"default_generation_settings":{}}`, `[{"id":0,"n_ctx":65536},{"id":1,"n_ctx":65536}]`)
	silent := newFakeLlamaUpstream(t, "", "")
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "routed", Kind: modelroute.KindLocal, BaseURL: routed.srv.URL + "/v1"},
			{ID: "silent", Kind: modelroute.KindLocal, BaseURL: silent.srv.URL + "/v1"},
		},
		Bindings: []modelroute.Binding{
			{Model: "model-b", Account: "routed", UpstreamModel: "wire-b"},
			{Model: "model-c", Account: "silent", UpstreamModel: "wire-c"},
		},
		Default: "routed",
	}
	srv := newTestServerWithConfig(t, Config{
		EngineID: "test", Model: "model-a", BaseURL: boot.srv.URL + "/v1", Provider: "openai",
		RouteAccounts: roster, LlamaSlotAffinity: true, VDSO: true,
	})
	if srv.upstreamWindows == nil {
		t.Fatal("upstream-context-window cache not armed for an opted-in proxy")
	}
	if _, ok := srv.loops.Get("upstream-context-window"); !ok {
		t.Fatal("upstream-context-window loop not registered")
	}
	before := catalogContextWindows(t, srv)
	if before.data["model-a"] != nil {
		t.Fatalf("window advertised before any probe: %v", before.data)
	}

	if err := srv.upstreamWindowTick(context.Background()); err != nil {
		t.Fatalf("tick: %v", err)
	}
	got := catalogContextWindows(t, srv)
	want := map[string]any{"model-a": float64(32768), "model-b": float64(65536), "model-c": nil}
	for id, w := range want {
		if got.data[id] != w || got.models[id] != w {
			t.Errorf("%s: data context_window=%v models context_window=%v, want %v", id, got.data[id], got.models[id], w)
		}
	}
	if s := boot.seen(); len(s) != 1 || s[0] != "/props?model-a" {
		t.Errorf("boot probes = %v, want one /props?model=model-a", s)
	}
	if s := routed.seen(); len(s) != 2 || s[0] != "/props?wire-b" || s[1] != "/slots?wire-b" {
		t.Errorf("routed probes = %v, want /props then /slots with model=wire-b", s)
	}

	// A miss backs off: the silent upstream is not re-polled on the next tick.
	_ = srv.upstreamWindowTick(context.Background())
	if n := len(silent.seen()); n != 2 {
		t.Errorf("silent upstream polled %d times after two ticks, want 2 (one /props + one /slots)", n)
	}
}

// TestUpstreamWindowCacheExpires: a window not refreshed within the max age reads unknown.
//
// fak-test:runtime fast est=1ms
func TestUpstreamWindowCacheExpires(t *testing.T) {
	now := time.Unix(1_000, 0)
	c := newUpstreamWindowCache()
	c.now = func() time.Time { return now }
	c.record("m", 16384)
	if c.get("m") != 16384 || c.get("other") != 0 {
		t.Fatalf("get = %d/%d, want 16384/0", c.get("m"), c.get("other"))
	}
	now = now.Add(upstreamWindowMaxAge + time.Second)
	if c.get("m") != 0 {
		t.Fatal("stale window still advertised")
	}
	var nilCache *upstreamWindowCache
	if nilCache.get("m") != 0 {
		t.Fatal("nil cache must read unknown")
	}
}

// TestUpstreamWindowNotArmedWithoutOptIn: a proxy without the llama-server discovery
// opt-in registers no probe loop, so generic upstreams never see a /props request.
//
// fak-test:runtime fast est=20ms
func TestUpstreamWindowNotArmedWithoutOptIn(t *testing.T) {
	up := newFakeLlamaUpstream(t, `{"default_generation_settings":{"n_ctx":4096}}`, "")
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "m", BaseURL: up.srv.URL + "/v1", Provider: "openai", VDSO: true})
	if srv.upstreamWindows != nil {
		t.Fatal("probe cache armed without opt-in")
	}
	if _, ok := srv.loops.Get("upstream-context-window"); ok {
		t.Fatal("probe loop registered without opt-in")
	}
}
