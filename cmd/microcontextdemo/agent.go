package main

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

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
	id         string
	done       bool
	exact      bool
	maxRetries int
	attempts   int
}

func (a *shardAgent) Step(ctx context.Context, gw microagent.Gateway) (bool, error) {
	if a.done {
		return true, nil
	}
	var resp *agent.Completion
	var err error
	for {
		a.attempts++
		resp, err = gw.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: a.id}}, nil)
		if err == nil || a.attempts > a.maxRetries {
			break
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(time.Duration(a.attempts) * 100 * time.Millisecond):
		}
	}
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
