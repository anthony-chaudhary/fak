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
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type config struct {
	Contexts  int
	Workers   int
	Delay     time.Duration
	Selfcheck bool
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
}

type sharedBase struct {
	instructions string
	fingerprint  string
}

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
	id   string
	done bool
}

func (a *shardAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	resp, err := gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: a.id}}, nil)
	if err != nil {
		return false, err
	}
	if resp == nil || resp.Message.Content != "done:"+a.id {
		return false, fmt.Errorf("bad completion for %s", a.id)
	}
	a.done = true
	return true, nil
}

func run(ctx context.Context, cfg config) (report, error) {
	if cfg.Contexts < 1 || cfg.Workers < 1 {
		return report{}, fmt.Errorf("contexts and workers must be positive")
	}
	base := &sharedBase{instructions: "one immutable agent base shared by every micro-context", fingerprint: "microcontext-base-v1"}
	gw := newFakeEndpoint(base, cfg.Delay)
	host, err := microagent.NewHost(gw, microagent.Config{Workers: cfg.Workers, Queue: cfg.Contexts})
	if err != nil {
		return report{}, err
	}
	defer host.Close()
	start := time.Now()
	for i := 0; i < cfg.Contexts; i++ {
		id := "ctx-" + strconv.Itoa(i)
		if err := host.Spawn(id, &shardAgent{id: id}); err != nil {
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
	for _, result := range results {
		if !result.Done || result.Err != nil {
			failed++
		}
	}
	r := report{
		Schema: "fak-microcontext-spine/1", Verdict: "PASS", LogicalShards: cfg.Contexts,
		PhysicalWorkers: cfg.Workers, Completed: len(results) - failed, Failed: failed,
		SharedBaseInstalls: 1, TurnCount: gw.calls.Load(), PeakInFlight: gw.peak.Load(),
		ElapsedMS: elapsed.Milliseconds(), ShardsPerSecond: float64(cfg.Contexts) / elapsed.Seconds(),
		Scope: "synthetic endpoint; proves bounded harness fan-out and shared-base semantics, not model tokens/sec",
	}
	if failed != 0 || len(results) != cfg.Contexts || gw.calls.Load() != int64(cfg.Contexts) || len(gw.seen) != cfg.Contexts || gw.peak.Load() > int64(cfg.Workers) {
		r.Verdict = "FAIL"
		return r, fmt.Errorf("spine invariant failed")
	}
	if cfg.Selfcheck && cfg.Contexts > 1 && cfg.Workers > 1 && gw.peak.Load() < 2 {
		r.Verdict = "FAIL"
		return r, fmt.Errorf("parallelism was not observed")
	}
	return r, nil
}

func main() {
	var cfg config
	flag.IntVar(&cfg.Contexts, "contexts", 10000, "logical micro-contexts")
	flag.IntVar(&cfg.Workers, "workers", 64, "bounded physical worker slots")
	flag.DurationVar(&cfg.Delay, "synthetic-latency", 100*time.Microsecond, "synthetic endpoint latency per context")
	flag.BoolVar(&cfg.Selfcheck, "selfcheck", false, "enforce spine invariants")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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
