package servingsupervision

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ProxyRestartHook is invoked when the proxy process or ingress listener undergoes recovery.
// Invariant: Hook execution must re-initialize listeners without restarting backend model replicas.
// Guard: Hook failure marks the proxy PhaseFailed and aborts recovery.
type ProxyRestartHook func(ctx context.Context) error

// ProxySupervisor manages the ingress router and its reconstructible routing table.
// Invariant: Proxy operates in an independent failure domain; proxy crashes do not unload replica weights.
// Guard: Routing fails closed with ErrTrafficWithdrawn when proxy is not in PhaseReady, or ErrNoHealthyReplicas when no healthy replicas exist.
type ProxySupervisor struct {
	mu          sync.Mutex
	domainID    string
	proxyID     string
	generation  uint64
	phase       ServingPhase
	spec        ServingDomainSpec
	endpoint    string
	drainMgr    *DrainManager
	replicas    []*ReplicaSupervisor
	rrIdx       int
	restartHook ProxyRestartHook
	lastReceipt *ServingReceipt
	engine      string
}

// ProxyOption configures optional behavior on a ProxySupervisor.
// Invariant: Applied during constructor execution before drain manager initialization.
type ProxyOption func(*ProxySupervisor)

// WithProxyRestartHook attaches a callback for proxy re-initialization.
// Guard: Hook errors during restart cause proxy to transition to PhaseFailed.
func WithProxyRestartHook(fn ProxyRestartHook) ProxyOption {
	return func(p *ProxySupervisor) {
		p.restartHook = fn
	}
}

// WithProxyBackend sets the engine identity (default EngineNative).
// Invariant: Preserves FAK-native engine provenance in supervision receipts.
func WithProxyBackend(engine string) ProxyOption {
	return func(p *ProxySupervisor) {
		p.engine = engine
	}
}

// NewProxySupervisor creates a supervisor for the ingress proxy domain.
// Invariant: Non-positive DrainTimeout defaults to 5s; non-positive RestartBudget defaults to 5.
// Guard: Role is pinned strictly to RoleProxy; initial phase is PhaseStarting.
func NewProxySupervisor(spec ServingDomainSpec, proxyID string, endpoint string, opts ...ProxyOption) *ProxySupervisor {
	if spec.DrainTimeout <= 0 {
		spec.DrainTimeout = 5 * time.Second
	}
	if spec.RestartBudget <= 0 {
		spec.RestartBudget = 5
	}
	spec.Role = RoleProxy

	p := &ProxySupervisor{
		domainID:   spec.DomainID,
		proxyID:    proxyID,
		generation: 1,
		phase:      PhaseStarting,
		spec:       spec,
		endpoint:   endpoint,
		engine:     EngineNative,
	}

	for _, opt := range opts {
		opt(p)
	}

	p.drainMgr = NewDrainManager(spec.DomainID, proxyID, RoleProxy, spec.DrainTimeout, 1)
	p.drainMgr.SetModelBackend(p.engine)

	return p
}

// Start activates the proxy supervisor in PhaseReady.
// Invariant: Transitions proxy and internal drain manager to PhaseReady.
// Guard: Protected by internal mutex.
func (p *ProxySupervisor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.phase = PhaseReady
	p.drainMgr.SetPhase(PhaseReady)
	return nil
}

// Reconstruct populates the proxy routing table from healthy replicas without altering replica state.
// Invariant: Retains existing replica instances without changing their generation or loading state.
// Guard: Thread-safe copy protected by internal mutex.
func (p *ProxySupervisor) Reconstruct(ctx context.Context, replicas []*ReplicaSupervisor) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.replicas = make([]*ReplicaSupervisor, len(replicas))
	copy(p.replicas, replicas)
	p.phase = PhaseReady
	p.drainMgr.SetPhase(PhaseReady)
	return nil
}

// Restart reconstructs the proxy independently without tearing down healthy model replicas.
// Invariant: Advances proxy generation, executes bounded drain, runs restart hook, and adopts healthy replicas.
// Guard: Model replicas are never restarted or unloaded during proxy recovery.
func (p *ProxySupervisor) Restart(ctx context.Context, healthyReplicas []*ReplicaSupervisor) (*ServingReceipt, error) {
	p.mu.Lock()
	nextGen := p.generation + 1
	p.mu.Unlock()

	// Drain incoming requests bounded by drain timeout
	receipt, err := p.drainMgr.Drain(ctx, nil, ScopeLeafOnly, false, nextGen)
	if err != nil {
		return nil, fmt.Errorf("proxy drain failed: %w", err)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	p.phase = PhaseRecovering
	p.generation = nextGen

	if p.restartHook != nil {
		if hookErr := p.restartHook(ctx); hookErr != nil {
			p.phase = PhaseFailed
			p.drainMgr.SetPhase(PhaseFailed)
			return receipt, fmt.Errorf("proxy restart hook failed: %w", hookErr)
		}
	}

	// Reconstruct routing table from existing healthy replicas.
	// Healthy model replicas are NEVER unloaded or restarted!
	p.replicas = make([]*ReplicaSupervisor, len(healthyReplicas))
	copy(p.replicas, healthyReplicas)

	p.phase = PhaseReady
	p.drainMgr.Reset(p.generation)
	if receipt != nil {
		receipt.Engine = p.engine
	}
	p.lastReceipt = receipt

	return receipt, nil
}

// Route selects an active, healthy replica using round-robin scheduling.
// Invariant: Round-robin index advances only across currently healthy and unquarantined replicas.
// Guard: Returns ErrTrafficWithdrawn if proxy != PhaseReady; returns ErrNoHealthyReplicas if healthy pool is empty.
func (p *ProxySupervisor) Route() (*ReplicaSupervisor, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.phase != PhaseReady {
		return nil, ErrTrafficWithdrawn
	}

	healthy := make([]*ReplicaSupervisor, 0, len(p.replicas))
	for _, rep := range p.replicas {
		if rep.IsHealthy() {
			healthy = append(healthy, rep)
		}
	}

	if len(healthy) == 0 {
		return nil, ErrNoHealthyReplicas
	}

	chosen := healthy[p.rrIdx%len(healthy)]
	p.rrIdx++
	return chosen, nil
}

// Endpoint returns the stable ingress endpoint address.
// Invariant: Endpoint identity remains stable across proxy restarts.
// Guard: Thread-safe read protected by internal mutex.
func (p *ProxySupervisor) Endpoint() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.endpoint
}

// Phase returns the active serving phase of the proxy.
// Invariant: Thread-safe read of the active serving phase.
// Guard: Protected by internal mutex.
func (p *ProxySupervisor) Phase() ServingPhase {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.phase
}

// Generation returns the current proxy generation.
// Invariant: Increments monotonically on each proxy restart.
// Guard: Protected by internal mutex.
func (p *ProxySupervisor) Generation() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.generation
}

// Replicas returns the list of replicas registered with this proxy.
// Invariant: Returns shallow copy to prevent caller mutation of internal replica slice.
// Guard: Protected by internal mutex.
func (p *ProxySupervisor) Replicas() []*ReplicaSupervisor {
	p.mu.Lock()
	defer p.mu.Unlock()
	copied := make([]*ReplicaSupervisor, len(p.replicas))
	copy(copied, p.replicas)
	return copied
}

// LastReceipt returns the most recent supervision receipt for the proxy.
// Invariant: Returns nil if no drain or restart has taken place.
// Guard: Protected by internal mutex.
func (p *ProxySupervisor) LastReceipt() *ServingReceipt {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.lastReceipt
}

// DomainID returns the failure domain ID.
// Invariant: Immutable domain identifier.
func (p *ProxySupervisor) DomainID() string {
	return p.domainID
}

// ProxyID returns the member proxy ID.
// Invariant: Immutable proxy identifier.
func (p *ProxySupervisor) ProxyID() string {
	return p.proxyID
}
