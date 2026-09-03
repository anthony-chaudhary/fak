package servingsupervision

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ProxyRestartHook is invoked when the proxy process or ingress listener undergoes recovery.
type ProxyRestartHook func(ctx context.Context) error

// ProxySupervisor manages the ingress router and its reconstructible routing table.
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
type ProxyOption func(*ProxySupervisor)

// WithProxyRestartHook attaches a callback for proxy re-initialization.
func WithProxyRestartHook(fn ProxyRestartHook) ProxyOption {
	return func(p *ProxySupervisor) {
		p.restartHook = fn
	}
}

// WithProxyBackend sets the engine identity (default EngineNative).
func WithProxyBackend(engine string) ProxyOption {
	return func(p *ProxySupervisor) {
		p.engine = engine
	}
}

// NewProxySupervisor creates a supervisor for the ingress proxy domain.
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
func (p *ProxySupervisor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.phase = PhaseReady
	p.drainMgr.SetPhase(PhaseReady)
	return nil
}

// Reconstruct populates the proxy routing table from healthy replicas without altering replica state.
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
func (p *ProxySupervisor) Endpoint() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.endpoint
}

// Phase returns the active serving phase of the proxy.
func (p *ProxySupervisor) Phase() ServingPhase {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.phase
}

// Generation returns the current proxy generation.
func (p *ProxySupervisor) Generation() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.generation
}

// Replicas returns the list of replicas registered with this proxy.
func (p *ProxySupervisor) Replicas() []*ReplicaSupervisor {
	p.mu.Lock()
	defer p.mu.Unlock()
	copied := make([]*ReplicaSupervisor, len(p.replicas))
	copy(copied, p.replicas)
	return copied
}

// LastReceipt returns the most recent supervision receipt for the proxy.
func (p *ProxySupervisor) LastReceipt() *ServingReceipt {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.lastReceipt
}

// DomainID returns the failure domain ID.
func (p *ProxySupervisor) DomainID() string {
	return p.domainID
}

// ProxyID returns the member proxy ID.
func (p *ProxySupervisor) ProxyID() string {
	return p.proxyID
}
