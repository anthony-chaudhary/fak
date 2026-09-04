package servingsupervision

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DesiredServingState specifies the declarative serving workload.
// Invariant: Declares target replica count, model artifact, drain timeouts, and restart budgets.
// Guard: Serves as the single source of truth for controller reconciliation.
type DesiredServingState struct {
	DeploymentID  string        `json:"deployment_id"`
	ModelArtifact string        `json:"model_artifact"`
	ReplicaCount  int           `json:"replica_count"`
	DrainTimeout  time.Duration `json:"drain_timeout"`
	RestartBudget int           `json:"restart_budget"`
}

// ReconciliationReport captures the actions taken during controller reconciliation or restart.
// Invariant: Categorizes replicas into adopted, created, and removed slices.
// Guard: PreservedProxy indicates whether existing ingress proxy was non-destructively preserved.
type ReconciliationReport struct {
	AdoptedReplicas []string `json:"adopted_replicas"`
	CreatedReplicas []string `json:"created_replicas"`
	RemovedReplicas []string `json:"removed_replicas"`
	PreservedProxy  bool     `json:"preserved_proxy"`
}

// ControllerSupervisor owns state reconciliation from desired state and non-destructive adoption.
// Invariant: Controller crash/restart adopts existing healthy replicas without restarting or disrupting them.
// Guard: Desired replica count is maintained by creating missing replicas or draining excess ones.
type ControllerSupervisor struct {
	mu           sync.Mutex
	domainID     string
	controllerID string
	generation   uint64
	phase        ServingPhase
	spec         ServingDomainSpec
	desired      DesiredServingState
	topology     *ServingTopology
	replicas     map[string]*ReplicaSupervisor
	proxy        *ProxySupervisor
	drainMgr     *DrainManager
	lastReceipt  *ServingReceipt
	engine       string
}

// ControllerOption configures optional behavior on a ControllerSupervisor.
// Invariant: Applied during constructor execution.
type ControllerOption func(*ControllerSupervisor)

// WithControllerBackend sets the engine identity (default EngineNative).
// Invariant: Preserves FAK-native engine provenance in supervision receipts.
func WithControllerBackend(engine string) ControllerOption {
	return func(c *ControllerSupervisor) {
		c.engine = engine
	}
}

// NewControllerSupervisor builds a supervisor for the serving controller domain.
// Invariant: Non-positive DrainTimeout defaults to 5s; non-positive RestartBudget defaults to 5.
// Guard: Role is pinned strictly to RoleController; initial phase is PhaseStarting.
func NewControllerSupervisor(
	spec ServingDomainSpec,
	desired DesiredServingState,
	topology *ServingTopology,
	opts ...ControllerOption,
) *ControllerSupervisor {
	if spec.DrainTimeout <= 0 {
		spec.DrainTimeout = 5 * time.Second
	}
	if spec.RestartBudget <= 0 {
		spec.RestartBudget = 5
	}
	spec.Role = RoleController

	c := &ControllerSupervisor{
		domainID:     spec.DomainID,
		controllerID: spec.DomainID,
		generation:   1,
		phase:        PhaseStarting,
		spec:         spec,
		desired:      desired,
		topology:     topology,
		replicas:     make(map[string]*ReplicaSupervisor),
		engine:       EngineNative,
	}

	for _, opt := range opts {
		opt(c)
	}

	c.drainMgr = NewDrainManager(spec.DomainID, spec.DomainID, RoleController, spec.DrainTimeout, 1)
	c.drainMgr.SetModelBackend(c.engine)

	return c
}

// Start brings the controller into PhaseReady.
// Invariant: Transitions controller and internal drain manager to PhaseReady.
// Guard: Protected by internal mutex.
func (c *ControllerSupervisor) Start(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.phase = PhaseReady
	c.drainMgr.SetPhase(PhaseReady)
	return nil
}

// Reconcile reconciles running runtime components against the desired state.
// Invariant: Existing healthy replicas are adopted non-destructively without restarts.
// Guard: Deficits trigger creation of new replicas; excesses trigger graceful drain before removal.
func (c *ControllerSupervisor) Reconcile(
	ctx context.Context,
	existingReplicas []*ReplicaSupervisor,
	existingProxy *ProxySupervisor,
) (*ReconciliationReport, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	report := &ReconciliationReport{
		AdoptedReplicas: make([]string, 0),
		CreatedReplicas: make([]string, 0),
		RemovedReplicas: make([]string, 0),
	}

	replicaTable := make(map[string]*ReplicaSupervisor)

	// Step 1: Non-destructive adoption of existing healthy replicas
	for _, rep := range existingReplicas {
		if rep.IsHealthy() {
			replicaTable[rep.ReplicaID()] = rep
			report.AdoptedReplicas = append(report.AdoptedReplicas, rep.ReplicaID())
		}
	}

	// Step 2: Create additional replicas if desired count exceeds adopted healthy count
	if len(replicaTable) < c.desired.ReplicaCount {
		needed := c.desired.ReplicaCount - len(replicaTable)
		for i := 0; i < needed; i++ {
			replicaIndex := len(replicaTable)
			var repSpec ServingDomainSpec
			if c.topology != nil && replicaIndex < len(c.topology.Replicas) {
				repSpec = c.topology.Replicas[replicaIndex]
			} else {
				repSpec = ServingDomainSpec{
					DomainID:      fmt.Sprintf("%s-replica-%d", c.desired.DeploymentID, replicaIndex),
					ControllerID:  c.controllerID,
					DrainTimeout:  c.desired.DrainTimeout,
					RestartBudget: c.desired.RestartBudget,
					Role:          RoleReplica,
				}
			}

			newRep := NewReplicaSupervisor(repSpec, repSpec.DomainID, WithReplicaBackend(c.engine))
			if err := newRep.Start(ctx); err != nil {
				return nil, fmt.Errorf("start new replica %s: %w", repSpec.DomainID, err)
			}
			replicaTable[newRep.ReplicaID()] = newRep
			report.CreatedReplicas = append(report.CreatedReplicas, newRep.ReplicaID())
		}
	}

	// Step 3: Remove excess replicas if adopted count exceeds desired count
	if len(replicaTable) > c.desired.ReplicaCount {
		count := 0
		trimmedMap := make(map[string]*ReplicaSupervisor, c.desired.ReplicaCount)
		for id, rep := range replicaTable {
			if count < c.desired.ReplicaCount {
				trimmedMap[id] = rep
				count++
			} else {
				// Graceful drain before removal
				_, _ = rep.drainMgr.Drain(ctx, nil, ScopeNone, false, rep.Generation())
				report.RemovedReplicas = append(report.RemovedReplicas, id)
			}
		}
		replicaTable = trimmedMap
	}

	c.replicas = replicaTable

	// Step 4: Proxy adoption and update
	if existingProxy != nil {
		c.proxy = existingProxy
		report.PreservedProxy = true
	} else if c.topology != nil {
		c.proxy = NewProxySupervisor(c.topology.Proxy, c.topology.Proxy.DomainID, "default-endpoint", WithProxyBackend(c.engine))
		if err := c.proxy.Start(ctx); err != nil {
			return nil, fmt.Errorf("start proxy: %w", err)
		}
	}

	if c.proxy != nil {
		active := make([]*ReplicaSupervisor, 0, len(c.replicas))
		for _, rep := range c.replicas {
			if rep.IsHealthy() {
				active = append(active, rep)
			}
		}
		if err := c.proxy.Reconstruct(ctx, active); err != nil {
			return nil, fmt.Errorf("reconstruct proxy routes: %w", err)
		}
	}

	c.phase = PhaseReady
	c.drainMgr.SetPhase(PhaseReady)
	return report, nil
}

// Restart recovers the controller domain, advancing its generation and non-destructively adopting existing workers.
// Invariant: Drains controller domain, bumps generation, and adopts running replicas and proxy without worker teardown.
// Guard: Reconciles desired state; emits audit receipt witnessing controller recovery.
func (c *ControllerSupervisor) Restart(
	ctx context.Context,
	existingReplicas []*ReplicaSupervisor,
	existingProxy *ProxySupervisor,
) (*ServingReceipt, *ReconciliationReport, error) {
	c.mu.Lock()
	observed := c.generation
	nextGen := observed + 1
	c.phase = PhaseRecovering
	c.mu.Unlock()

	// Drain controller domain bounded by timeout
	receipt, err := c.drainMgr.Drain(ctx, nil, ScopeLeafOnly, false, nextGen)
	if err != nil {
		return nil, nil, fmt.Errorf("controller drain failed: %w", err)
	}

	report, reconErr := c.Reconcile(ctx, existingReplicas, existingProxy)
	if reconErr != nil {
		return receipt, nil, fmt.Errorf("controller reconciliation failed: %w", reconErr)
	}

	c.mu.Lock()
	c.generation = nextGen
	c.phase = PhaseReady
	c.drainMgr.Reset(c.generation)
	if receipt != nil {
		receipt.Engine = c.engine
	}
	c.lastReceipt = receipt
	c.mu.Unlock()

	return receipt, report, nil
}

// HealthyReplicas lists all currently healthy and unquarantined replicas.
// Invariant: Returns snapshot slice of replicas currently in PhaseReady and unquarantined.
// Guard: Thread-safe read protected by internal mutex.
func (c *ControllerSupervisor) HealthyReplicas() []*ReplicaSupervisor {
	c.mu.Lock()
	defer c.mu.Unlock()

	healthy := make([]*ReplicaSupervisor, 0, len(c.replicas))
	for _, rep := range c.replicas {
		if rep.IsHealthy() {
			healthy = append(healthy, rep)
		}
	}
	return healthy
}

// Replicas returns a snapshot map of all tracked replicas.
// Invariant: Returns shallow copy map of all registered replicas (healthy or otherwise).
// Guard: Thread-safe read protected by internal mutex.
func (c *ControllerSupervisor) Replicas() map[string]*ReplicaSupervisor {
	c.mu.Lock()
	defer c.mu.Unlock()

	copied := make(map[string]*ReplicaSupervisor, len(c.replicas))
	for k, v := range c.replicas {
		copied[k] = v
	}
	return copied
}

// Proxy returns the managed proxy supervisor.
// Invariant: Returns the active ingress proxy supervisor reference.
// Guard: Thread-safe read protected by internal mutex.
func (c *ControllerSupervisor) Proxy() *ProxySupervisor {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.proxy
}

// Phase returns the current serving phase of the controller.
// Invariant: Thread-safe read of the active controller phase.
// Guard: Protected by internal mutex.
func (c *ControllerSupervisor) Phase() ServingPhase {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.phase
}

// Generation returns the current generation counter.
// Invariant: Increments monotonically on controller restart.
// Guard: Protected by internal mutex.
func (c *ControllerSupervisor) Generation() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.generation
}

// LastReceipt returns the most recent supervision receipt for the controller.
// Invariant: Returns nil if no drain or restart action has occurred.
// Guard: Thread-safe read protected by internal mutex.
func (c *ControllerSupervisor) LastReceipt() *ServingReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastReceipt
}

// DomainID returns the failure domain ID.
// Invariant: Immutable domain identifier.
func (c *ControllerSupervisor) DomainID() string {
	return c.domainID
}

// ControllerID returns the controller identity.
// Invariant: Immutable controller identifier.
func (c *ControllerSupervisor) ControllerID() string {
	return c.controllerID
}
