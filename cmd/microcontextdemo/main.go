// Command microcontextdemo is the minimal runnable spine for the micro-context
// research program. It drives many logical agent contexts over a bounded set of
// physical workers while one immutable base context remains installed in the
// controlled model seam.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type config struct {
	Contexts       int
	Workers        int
	Delay          time.Duration
	Selfcheck      bool
	Endpoint       string
	APIKey         string
	Model          string
	Provider       string
	Hardware       string
	RequestTimeout time.Duration
	PrefixMode     string
}

type report struct {
	Schema             string  `json:"schema"`
	Verdict            string  `json:"verdict"`
	LogicalShards      int     `json:"logical_shards"`
	PhysicalWorkers    int     `json:"physical_workers"`
	Completed          int     `json:"completed"`
	Failed             int     `json:"failed"`
	SharedBaseInstalls int64   `json:"shared_base_installs"`
	TurnCount          int64   `json:"turn_count"`
	PeakInFlight       int64   `json:"peak_in_flight"`
	ElapsedMS          int64   `json:"elapsed_ms"`
	ShardsPerSecond    float64 `json:"shards_per_second"`
	Scope              string  `json:"scope"`
	FirstFailure       string  `json:"first_failure,omitempty"`
	Mode               string  `json:"mode"`
	Endpoint           string  `json:"endpoint,omitempty"`
	Provider           string  `json:"provider,omitempty"`
	Model              string  `json:"model,omitempty"`
	Hardware           string  `json:"hardware,omitempty"`
	BaseFingerprint    string  `json:"base_fingerprint"`
	PromptTokens       int64   `json:"prompt_tokens,omitempty"`
	CompletionTokens   int64   `json:"completion_tokens,omitempty"`
	CachedPromptTokens int64   `json:"cached_prompt_tokens,omitempty"`
	UsageResponses     int     `json:"usage_responses,omitempty"`
	TTFTP50MS          float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP95MS          float64 `json:"ttft_p95_ms,omitempty"`
	LatencyP50MS       float64 `json:"latency_p50_ms,omitempty"`
	LatencyP95MS       float64 `json:"latency_p95_ms,omitempty"`
	PromptTokensPerSec float64 `json:"prompt_tokens_per_wall_second,omitempty"`
	DecodeTokensPerSec float64 `json:"decode_tokens_per_wall_second,omitempty"`
}

type sharedBase struct {
	instructions string
	fingerprint  string
}

func canonicalBaseInstructions() string {
	return strings.Repeat("You are one worker in a bounded micro-context fabric. Preserve task isolation and return a short non-empty answer. Stable shared setup material follows. ", 24) + " Context identity: 00000000."
}

func canonicalBaseFingerprint() string { return "microcontext-base-v1" }

type fakeEndpoint struct {
	base     *sharedBase
	delay    time.Duration
	calls    atomic.Int64
	inFlight atomic.Int64
	peak     atomic.Int64
	seenMu   sync.Mutex
	seen     map[string]struct{}
}

func newFakeEndpoint(base *sharedBase, delay time.Duration) *fakeEndpoint {
	return &fakeEndpoint{base: base, delay: delay, seen: make(map[string]struct{})}
}

func (g *fakeEndpoint) Model() string { return "microcontext-synthetic" }

func (g *fakeEndpoint) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	current := g.inFlight.Add(1)
	defer g.inFlight.Add(-1)
	for {
		old := g.peak.Load()
		if current <= old || g.peak.CompareAndSwap(old, current) {
			break
		}
	}
	g.calls.Add(1)
	if len(messages) != 1 || messages[0].Role != agent.RoleUser {
		return nil, fmt.Errorf("delta contract: got %d messages", len(messages))
	}
	id := messages[0].Content
	g.seenMu.Lock()
	if _, duplicate := g.seen[id]; duplicate {
		g.seenMu.Unlock()
		return nil, fmt.Errorf("duplicate context %q", id)
	}
	g.seen[id] = struct{}{}
	g.seenMu.Unlock()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(g.delay):
	}
	// The shared base is intentionally read here, at the kernel seam, rather
	// than copied into every logical agent transcript.
	if g.base.instructions == "" || g.base.fingerprint == "" {
		return nil, fmt.Errorf("shared base is not installed")
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done:" + id}}, nil
}

type shardAgent struct {
	id    string
	done  bool
	exact bool
}

func (a *shardAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	resp, err := gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: a.id}}, nil)
	if err != nil {
		return false, err
	}
	if resp == nil || strings.TrimSpace(resp.Message.Content) == "" {
		return false, fmt.Errorf("empty completion for %s", a.id)
	}
	if a.exact && resp.Message.Content != "done:"+a.id {
		return false, fmt.Errorf("bad completion for %s", a.id)
	}
	a.done = true
	return true, nil
}

func percentileMS(values []time.Duration, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	idx := int(float64(len(values)-1) * q)
	return float64(values[idx].Microseconds()) / 1000
}

func run(ctx context.Context, cfg config) (report, error) {
	if cfg.Contexts < 1 || cfg.Workers < 1 {
		return report{}, fmt.Errorf("contexts and workers must be positive")
	}
	base := &sharedBase{instructions: canonicalBaseInstructions(), fingerprint: canonicalBaseFingerprint()}
	var gw microagent.Gateway
	var live *openAIEndpoint
	mode := "synthetic"
	if cfg.Endpoint != "" {
		var err error
		live, err = newOpenAIEndpoint(cfg.Endpoint, cfg.APIKey, cfg.Model, base, cfg.RequestTimeout)
		if live != nil {
			live.prefixMode = cfg.PrefixMode
		}
		if err != nil {
			return report{}, err
		}
		gw = live
		mode = "openai-compatible"
	} else {
		gw = newFakeEndpoint(base, cfg.Delay)
	}
	host, err := microagent.NewHost(gw, microagent.Config{Workers: cfg.Workers, Queue: cfg.Contexts})
	if err != nil {
		return report{}, err
	}
	defer host.Close()
	start := time.Now()
	for i := 0; i < cfg.Contexts; i++ {
		id := "ctx-" + strconv.Itoa(i)
		if err := host.Spawn(id, &shardAgent{id: id, exact: live == nil}); err != nil {
			return report{}, fmt.Errorf("spawn %s: %w", id, err)
		}
	}
	if err := host.Drain(ctx); err != nil {
		return report{}, err
	}
	elapsed := time.Since(start)
	results := host.Reap()
	sort.Slice(results, func(i, j int) bool { return results[i].ID < results[j].ID })
	failed := 0
	var firstFailure string
	for _, result := range results {
		if !result.Done || result.Err != nil {
			failed++
			if firstFailure == "" && result.Err != nil {
				firstFailure = result.ID + ": " + result.Err.Error()
			}
		}
	}
	r := report{
		Schema: "fak-microcontext-spine/1", Verdict: "PASS", LogicalShards: cfg.Contexts,
		PhysicalWorkers: cfg.Workers, Completed: len(results) - failed, Failed: failed,
		SharedBaseInstalls: 1, ElapsedMS: elapsed.Milliseconds(), ShardsPerSecond: float64(cfg.Contexts) / elapsed.Seconds(),
		Mode: mode, Provider: cfg.Provider, Model: cfg.Model, Hardware: cfg.Hardware, BaseFingerprint: base.fingerprint, FirstFailure: firstFailure,
	}
	if live == nil {
		fake := gw.(*fakeEndpoint)
		r.TurnCount = fake.calls.Load()
		r.PeakInFlight = fake.peak.Load()
		r.Scope = "synthetic endpoint; proves bounded harness fan-out and shared-base semantics, not model tokens/sec"
		if fake.calls.Load() != int64(cfg.Contexts) || len(fake.seen) != cfg.Contexts || fake.peak.Load() > int64(cfg.Workers) {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("synthetic spine invariant failed")
		}
		if cfg.Selfcheck && cfg.Contexts > 1 && cfg.Workers > 1 && fake.peak.Load() < 2 {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("parallelism was not observed")
		}
	} else {
		stats := live.snapshot()
		r.Endpoint = cfg.Endpoint
		r.TurnCount = int64(len(stats.latencies))
		r.PeakInFlight = int64(cfg.Workers) // admission ceiling; endpoint-internal batching is not inferred
		r.PromptTokens = stats.promptTokens
		r.CompletionTokens = stats.completionTokens
		r.CachedPromptTokens = stats.cachedTokens
		r.UsageResponses = stats.usageResponses
		r.TTFTP50MS = percentileMS(stats.ttfts, .50)
		r.TTFTP95MS = percentileMS(stats.ttfts, .95)
		r.LatencyP50MS = percentileMS(stats.latencies, .50)
		r.LatencyP95MS = percentileMS(stats.latencies, .95)
		r.PromptTokensPerSec = float64(stats.promptTokens) / elapsed.Seconds()
		r.DecodeTokensPerSec = float64(stats.completionTokens) / elapsed.Seconds()
		r.Scope = "real streaming endpoint; token rates are aggregate observed usage divided by wall time, not server-internal kernel rates; critical-path latency is reported separately"
		if stats.usageResponses != cfg.Contexts || len(stats.ttfts) != cfg.Contexts {
			r.Verdict = "FAIL"
			return r, fmt.Errorf("live telemetry incomplete: usage=%d ttft=%d want=%d", stats.usageResponses, len(stats.ttfts), cfg.Contexts)
		}
	}
	if failed != 0 || len(results) != cfg.Contexts || r.TurnCount != int64(cfg.Contexts) {
		r.Verdict = "FAIL"
		return r, fmt.Errorf("spine invariant failed")
	}
	return r, nil
}
func main() {
	var cfg config
	var verifyPath string
	var abOutput string
	var verifyABPath string
	flag.IntVar(&cfg.Contexts, "contexts", 10000, "logical micro-contexts")
	flag.IntVar(&cfg.Workers, "workers", 64, "bounded physical worker slots")
	flag.DurationVar(&cfg.Delay, "synthetic-latency", 100*time.Microsecond, "synthetic endpoint latency per context")
	flag.BoolVar(&cfg.Selfcheck, "selfcheck", false, "enforce spine invariants")
	flag.StringVar(&cfg.Endpoint, "endpoint", "", "OpenAI-compatible endpoint root; empty uses the synthetic S0 endpoint")
	flag.StringVar(&cfg.APIKey, "api-key", "", "endpoint API key (prefer environment expansion by the caller)")
	flag.StringVar(&cfg.Model, "model", "", "live endpoint model id")
	flag.StringVar(&cfg.Provider, "provider", "", "provider provenance label")
	flag.StringVar(&cfg.Hardware, "hardware", "", "hardware provenance label")
	flag.DurationVar(&cfg.RequestTimeout, "request-timeout", 2*time.Minute, "per-request live endpoint timeout")
	flag.StringVar(&verifyPath, "verify", "", "verify a captured S1 JSON artifact and exit")
	flag.StringVar(&abOutput, "prefix-ab", "", "run the S2 prefix A/B and write JSON to this path (or - for stdout)")
	flag.StringVar(&verifyABPath, "verify-prefix-ab", "", "verify a captured S2 prefix A/B artifact and exit")
	flag.Parse()
	if verifyPath != "" {
		if err := verifyArtifact(verifyPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println("PASS: verified", verifyPath)
		return
	}
	if verifyABPath != "" {
		if err := verifyABArtifact(verifyABPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println("PASS: verified", verifyABPath)
		return
	}
	if abOutput != "" {
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()
		r, err := runAB(ctx, cfg)
		if writeErr := writeAB(abOutput, r); writeErr != nil {
			fmt.Fprintln(os.Stderr, writeErr)
			os.Exit(1)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	r, err := run(ctx, cfg)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
