package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

type controlledSoakEvidence struct {
	canaryContexts     int
	canaryPassed       int
	baseRollbackCount  int
	queuePeakContexts  int
	hibernatedContexts int
	restoredContexts   int
}

type controlledSoakContext struct {
	ID              string `json:"id"`
	BaseFingerprint string `json:"base_fingerprint"`
	State           string `json:"state"`
}

func (a *controlledSoakContext) Step(context.Context, microagent.Gateway) (bool, error) {
	return true, nil
}

func (a *controlledSoakContext) Freeze() ([]byte, error) {
	return json.Marshal(a)
}

func (a *controlledSoakContext) Thaw(b []byte) error {
	return json.Unmarshal(b, a)
}

// runControlledSoakPreflight exercises the state transitions named by the S5
// contract before the measured 10k model-turn workload starts:
//
//   - one real endpoint canary runs under a candidate base, followed by an
//     observed rollback to the canonical base;
//   - every logical context outside the bounded resident-worker set is frozen
//     through HibernationStore and restored byte-identically.
//
// The canary's endpoint telemetry is reset before the measured workload, so it
// cannot inflate the 10k turn/token accounting.
func runControlledSoakPreflight(ctx context.Context, cfg config, live *openAIEndpoint, base *sharedBase) (controlledSoakEvidence, error) {
	if live == nil {
		return controlledSoakEvidence{}, errors.New("controlled soak requires a live endpoint")
	}
	if cfg.Contexts <= cfg.Workers {
		return controlledSoakEvidence{}, fmt.Errorf("controlled soak requires contexts (%d) above workers (%d)", cfg.Contexts, cfg.Workers)
	}

	evidence := controlledSoakEvidence{canaryContexts: 1, queuePeakContexts: cfg.Contexts}
	candidate := &sharedBase{
		instructions: base.instructions + " Controlled canary generation.",
		fingerprint:  base.fingerprint + "-canary",
	}
	live.base = candidate
	canary, err := live.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "controlled-canary"}}, nil)
	live.base = base
	evidence.baseRollbackCount = 1
	if err != nil {
		return evidence, fmt.Errorf("controlled canary: %w", err)
	}
	if canary == nil || canary.Message.Content == "" {
		return evidence, errors.New("controlled canary returned an empty completion")
	}
	if live.base != base || live.base.fingerprint != canonicalBaseFingerprint() {
		return evidence, errors.New("controlled base rollback did not restore the canonical base")
	}
	evidence.canaryPassed = 1
	live.resetStats()

	dir, err := os.MkdirTemp("", "fak-microcontext-soak-")
	if err != nil {
		return evidence, fmt.Errorf("controlled hibernation temp dir: %w", err)
	}
	defer os.RemoveAll(dir)
	store, err := microagent.NewHibernationStore(dir)
	if err != nil {
		return evidence, err
	}
	for i := cfg.Workers; i < cfg.Contexts; i++ {
		id := fmt.Sprintf("ctx-%d", i)
		a := &controlledSoakContext{ID: id, BaseFingerprint: base.fingerprint, State: "queued"}
		if _, err := store.Park(id, a); err != nil {
			return evidence, fmt.Errorf("hibernate %s: %w", id, err)
		}
		evidence.hibernatedContexts++
	}
	for i := cfg.Workers; i < cfg.Contexts; i++ {
		id := fmt.Sprintf("ctx-%d", i)
		a := &controlledSoakContext{}
		if err := store.Wake(id, a); err != nil {
			return evidence, fmt.Errorf("restore %s: %w", id, err)
		}
		if a.ID != id || a.BaseFingerprint != base.fingerprint || a.State != "queued" {
			return evidence, fmt.Errorf("restore %s changed context state", id)
		}
		evidence.restoredContexts++
	}
	return evidence, nil
}

// controlledSoakGateway injects two bounded, deterministic first-attempt faults
// at the real gateway seam. The agent retries each exactly once, so the final
// provider accounting remains one usage-bearing model turn per logical context.
type controlledSoakGateway struct {
	inner           *openAIEndpoint
	mu              sync.Mutex
	attempts        map[string]int
	retryInjected   atomic.Int64
	retryRecovered  atomic.Int64
	cancelInjected  atomic.Int64
	cancelRecovered atomic.Int64
	innerFailures   atomic.Int64
	innerRecovered  atomic.Int64
	innerFailed     map[string]int
}

func newControlledSoakGateway(inner *openAIEndpoint) *controlledSoakGateway {
	return &controlledSoakGateway{inner: inner, attempts: make(map[string]int), innerFailed: make(map[string]int)}
}

func (g *controlledSoakGateway) Model() string { return g.inner.Model() }

func (g *controlledSoakGateway) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	id := ""
	if len(messages) == 1 {
		id = messages[0].Content
	}
	g.mu.Lock()
	g.attempts[id]++
	attempt := g.attempts[id]
	g.mu.Unlock()
	if id == "ctx-0" && attempt == 1 {
		g.retryInjected.Add(1)
		return nil, errors.New("controlled transient overload")
	}
	if id == "ctx-1" && attempt == 1 {
		g.cancelInjected.Add(1)
		canceled, cancel := context.WithCancel(ctx)
		cancel()
		<-canceled.Done()
		return nil, canceled.Err()
	}
	out, err := g.inner.Complete(ctx, messages, tools, opts...)
	g.mu.Lock()
	if err != nil {
		g.innerFailures.Add(1)
		g.innerFailed[id]++
	} else if failed := g.innerFailed[id]; failed > 0 {
		g.innerRecovered.Add(int64(failed))
		delete(g.innerFailed, id)
	}
	g.mu.Unlock()
	if err == nil && id == "ctx-0" && attempt >= 2 {
		g.retryRecovered.CompareAndSwap(0, 1)
	}
	if err == nil && id == "ctx-1" && attempt >= 2 {
		g.cancelRecovered.CompareAndSwap(0, 1)
	}
	return out, err
}
