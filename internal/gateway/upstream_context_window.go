package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// upstream_context_window.go — learn the per-request context window of a proxied
// llama-server-class upstream so /v1/models can advertise it.
//
// A proxy gateway has no in-kernel model, so its catalog rows carried no window and a
// client could not tell a 32k node from a 128k one until a long prompt bounced. The
// upstream knows: llama-server's /props reports default_generation_settings.n_ctx, which
// is the PER-SLOT context (the --ctx-size divided across --parallel slots), i.e. the
// largest prompt+output one request can use. /slots reports the same n_ctx per slot and
// is the fallback.
//
// Rules: discovery runs on a supervised background loop, never on a request; values are
// keyed per public model id (a proxy may front several models, and llama-server's router
// mode takes ?model= on both routes); an entry not refreshed within
// upstreamWindowMaxAge is dropped, and an unknown window is omitted, never guessed.
// Probing follows the same opt-in as the llama-server slot hint (LlamaSlotAffinity, on
// by default in `fak serve`), so a generic client or a test fake upstream never sees an
// extra request.

const (
	upstreamWindowLoopInterval = 30 * time.Second
	upstreamWindowMaxAge       = 3 * time.Minute
	upstreamWindowRetryAfter   = 5 * time.Minute
	upstreamWindowProbeTimeout = 3 * time.Second
	upstreamWindowBodyCap      = 1 << 20
)

// upstreamWindowHTTPClient is the bounded discovery client; tests may replace it.
var upstreamWindowHTTPClient = &http.Client{Timeout: upstreamWindowProbeTimeout}

type upstreamWindowEntry struct {
	window   int
	observed time.Time
}

// upstreamWindowCache holds the discovered per-request window per public model id.
// All methods are nil-safe so a server without a probe target reads "unknown".
type upstreamWindowCache struct {
	mu        sync.Mutex
	byModel   map[string]upstreamWindowEntry
	nextProbe map[string]time.Time
	now       func() time.Time
}

func newUpstreamWindowCache() *upstreamWindowCache {
	return &upstreamWindowCache{
		byModel:   map[string]upstreamWindowEntry{},
		nextProbe: map[string]time.Time{},
		now:       time.Now,
	}
}

func (c *upstreamWindowCache) get(model string) int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.byModel[model]
	if !ok || c.now().Sub(e.observed) > upstreamWindowMaxAge {
		return 0
	}
	return e.window
}

func (c *upstreamWindowCache) due(model string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.now().Before(c.nextProbe[model])
}

// record stores a probe result: a positive window refreshes the entry and the next
// probe follows the loop cadence; a miss keeps any still-fresh entry and backs off so
// an upstream that never answers (vLLM, a hosted API) is not polled every tick.
func (c *upstreamWindowCache) record(model string, window int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	if window > 0 {
		c.byModel[model] = upstreamWindowEntry{window: window, observed: now}
		delete(c.nextProbe, model)
		return
	}
	c.nextProbe[model] = now.Add(upstreamWindowRetryAfter)
}

// upstreamWindowTarget is one proxied model the gateway can probe.
type upstreamWindowTarget struct {
	ModelID       string // public id the catalog advertises (the cache key)
	BaseURL       string // OpenAI-compatible base, e.g. http://host:8081/v1
	UpstreamModel string // wire model name sent as ?model=
	APIKey        string
}

// upstreamWindowTargets lists the proxied models whose upstream may answer /props: the
// boot planner's model (direct proxy or the proxy side of a dual planner) and every
// self-hosted roster binding. Each is keyed by its own public id.
func (s *Server) upstreamWindowTargets() []upstreamWindowTarget {
	if s == nil {
		return nil
	}
	var out []upstreamWindowTarget
	seen := map[string]bool{}
	add := func(t upstreamWindowTarget) {
		t.ModelID = strings.TrimSpace(t.ModelID)
		if t.ModelID == "" || strings.TrimSpace(t.BaseURL) == "" || seen[t.ModelID] {
			return
		}
		seen[t.ModelID] = true
		out = append(out, t)
	}
	var hp *agent.HTTPPlanner
	switch p := s.planner.(type) {
	case *agent.HTTPPlanner:
		hp = p
	case *DualPlanner:
		hp, _ = p.Proxy().(*agent.HTTPPlanner)
	}
	if hp != nil && (hp.LlamaSlotAffinity || s.llamaSlotAffinity) && hp.Provider == agent.ProviderOpenAI && hp.APIKeyFunc == nil {
		id := s.model
		if id == "" {
			id = hp.ModelID
		}
		add(upstreamWindowTarget{ModelID: id, BaseURL: hp.BaseURL, UpstreamModel: hp.ModelID, APIKey: hp.APIKey})
	}
	if s.roster != nil && s.llamaSlotAffinity {
		for _, binding := range s.roster.Bindings {
			target, err := s.roster.Resolve(binding.Model)
			if err != nil || !target.Zone().SelfHosted() {
				continue
			}
			switch target.Kind {
			case modelroute.KindLocal, modelroute.KindFleet:
			default:
				continue
			}
			key := ""
			if target.CredEnv != "" {
				key = os.Getenv(target.CredEnv)
			}
			add(upstreamWindowTarget{ModelID: binding.Model, BaseURL: target.BaseURL, UpstreamModel: target.UpstreamModel, APIKey: key})
		}
	}
	return out
}

// upstreamWindowFor is the advertised window for a proxied model id: 0 = unknown.
func (s *Server) upstreamWindowFor(model string) int {
	if s == nil {
		return 0
	}
	return s.upstreamWindows.get(model)
}

// upstreamWindowTick is the supervised loop body: probe each due target and record it.
func (s *Server) upstreamWindowTick(ctx context.Context) error {
	if s.upstreamWindows == nil {
		return nil
	}
	for _, t := range s.upstreamWindowTargets() {
		if ctx.Err() != nil {
			return nil
		}
		if !s.upstreamWindows.due(t.ModelID) {
			continue
		}
		s.upstreamWindows.record(t.ModelID, probeUpstreamContextWindow(ctx, t))
	}
	return nil
}

// probeUpstreamContextWindow returns the upstream's per-request window, or 0 when it
// does not report one: /props default_generation_settings.n_ctx first, then the
// smallest positive /slots n_ctx.
func probeUpstreamContextWindow(ctx context.Context, t upstreamWindowTarget) int {
	var props struct {
		DefaultGenerationSettings struct {
			SlotWindow int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if upstreamWindowGet(ctx, t, "/props", &props) && props.DefaultGenerationSettings.SlotWindow > 0 {
		return props.DefaultGenerationSettings.SlotWindow
	}
	var slots []struct {
		SlotWindow int `json:"n_ctx"`
	}
	if !upstreamWindowGet(ctx, t, "/slots", &slots) {
		return 0
	}
	window := 0
	for _, slot := range slots {
		if slot.SlotWindow > 0 && (window == 0 || slot.SlotWindow < window) {
			window = slot.SlotWindow
		}
	}
	return window
}

func upstreamWindowGet(ctx context.Context, t upstreamWindowTarget, route string, v any) bool {
	ctx, cancel := context.WithTimeout(ctx, upstreamWindowProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, upstreamRootURL(t.BaseURL, route, t.UpstreamModel), nil)
	if err != nil {
		return false
	}
	if t.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+t.APIKey)
	}
	resp, err := upstreamWindowHTTPClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	return json.NewDecoder(io.LimitReader(resp.Body, upstreamWindowBodyCap)).Decode(v) == nil
}

// upstreamRootURL maps an OpenAI-compatible base (http://h:8080/v1) to a llama-server
// root route, carrying the wire model as ?model= for router-mode servers.
func upstreamRootURL(base, route, model string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	base = strings.TrimSuffix(base, "/v1")
	u := base + route
	if model = strings.TrimSpace(model); model != "" {
		u += "?model=" + url.QueryEscape(model)
	}
	return u
}
