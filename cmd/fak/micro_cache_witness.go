package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"github.com/anthony-chaudhary/fak/internal/session"
)

type cacheSeatFlags []string

func (v *cacheSeatFlags) String() string     { return strings.Join(*v, ",") }
func (v *cacheSeatFlags) Set(s string) error { *v = append(*v, strings.TrimSpace(s)); return nil }

type microCacheArm struct {
	Affinity                 bool     `json:"affinity"`
	Calls                    int      `json:"calls"`
	SelectedSeats            []string `json:"selected_seats"`
	PromptTokens             int      `json:"prompt_tokens"`
	CachedPromptTokens       int      `json:"cached_prompt_tokens"`
	CacheCreationInputTokens int      `json:"cache_creation_input_tokens,omitempty"`
}
type microCacheWitness struct {
	Schema                string        `json:"schema"`
	Verdict               string        `json:"verdict"`
	Reason                string        `json:"reason"`
	DistinctSeatEndpoints int           `json:"distinct_seat_endpoints"`
	On                    microCacheArm `json:"affinity_on"`
	Off                   microCacheArm `json:"affinity_off"`
}

func runMicroCacheWitness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak micro collapse cache-witness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var seats cacheSeatFlags
	fs.Var(&seats, "gateway-seat", "provider seat as id=URL; repeat for distinct account/endpoints")
	model := fs.String("model", "", "model id")
	calls := fs.Int("calls", 3, "repeated-prefix calls per arm")
	prompt := fs.String("prompt", strings.Repeat("stable cache witness context ", 600)+"\nReturn exactly CACHE-WITNESS.", "identical repeated prompt")
	if !parseFlags(fs, argv) || *model == "" || *calls < 2 || len(seats) == 0 {
		fmt.Fprintln(stderr, "fak micro collapse cache-witness: require --model, --calls >=2, and --gateway-seat")
		return 2
	}
	parsed, distinct, err := parseCacheSeats(seats, *model)
	if err != nil {
		fmt.Fprintln(stderr, "fak micro collapse cache-witness:", err)
		return 2
	}
	r := microCacheWitness{Schema: "fak-micro-cache-affinity-witness/1", Verdict: "not-yet", DistinctSeatEndpoints: distinct}
	r.On, err = runMicroCacheArm(parsed, false, *calls, *prompt)
	if err == nil {
		r.Off, err = runMicroCacheArm(parsed, true, *calls, *prompt)
	}
	if err != nil {
		r.Reason = err.Error()
	} else if distinct < 2 {
		r.Reason = "need at least two distinct provider seat endpoints for an affinity ablation"
	} else if r.On.CachedPromptTokens == 0 && r.Off.CachedPromptTokens == 0 {
		r.Reason = "provider reported no cache-token witness; zero is not treated as proof of no cache"
	} else {
		r.Verdict = "ready"
		r.Reason = "provider cache counters captured for affinity-on and affinity-off arms"
	}
	_ = json.NewEncoder(stdout).Encode(r)
	if r.Verdict != "ready" {
		return 3
	}
	return 0
}

func parseCacheSeats(values []string, model string) ([]microagent.GatewaySeat, int, error) {
	out := make([]microagent.GatewaySeat, 0, len(values))
	unique := map[string]struct{}{}
	for _, raw := range values {
		id, endpoint, ok := strings.Cut(raw, "=")
		if !ok || strings.TrimSpace(id) == "" || strings.TrimSpace(endpoint) == "" {
			return nil, 0, fmt.Errorf("invalid --gateway-seat %q (want id=URL)", raw)
		}
		u, err := url.Parse(endpoint)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return nil, 0, fmt.Errorf("invalid seat URL %q", endpoint)
		}
		normalized := strings.TrimRight(endpoint, "/")
		unique[normalized] = struct{}{}
		out = append(out, microagent.GatewaySeat{ID: strings.TrimSpace(id), Gateway: agent.NewHTTPPlanner(gatewayBaseURL(normalized), model, defaultGatewayBearerToken()), Scheduler: microagent.NewScheduler(1)})
	}
	return out, len(unique), nil
}

func runMicroCacheArm(seats []microagent.GatewaySeat, disable bool, calls int, prompt string) (microCacheArm, error) {
	r := microCacheArm{Affinity: !disable, Calls: calls}
	router, err := microagent.NewSessionAffinityGateway(seats, session.NewTable(), microagent.AffinityGatewayConfig{DisableAffinity: disable, Observe: func(s microagent.AffinitySelection) { r.SelectedSeats = append(r.SelectedSeats, s.SeatID) }})
	if err != nil {
		return r, err
	}
	for i := 0; i < calls; i++ {
		ctx := microagent.WithTrace(context.Background(), "cache-witness")
		if !disable {
			ctx = microagent.WithAffinity(ctx, "stable-shared-prefix")
		}
		c, callErr := router.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: prompt}}, nil)
		if callErr != nil {
			return r, callErr
		}
		r.PromptTokens += c.Usage.PromptTokens
		r.CachedPromptTokens += c.Usage.CachedPromptTokens()
		r.CacheCreationInputTokens += c.Usage.CacheCreationInputTokens
	}
	return r, nil
}
