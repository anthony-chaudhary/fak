package servingsupervision

import (
	"fmt"
	"time"
)

// ServingTopology declares the complete component hierarchy and failure domain boundaries.
type ServingTopology struct {
	DeploymentID string              `json:"deployment_id"`
	Controller   ServingDomainSpec   `json:"controller"`
	Proxy        ServingDomainSpec   `json:"proxy"`
	Router       *ServingDomainSpec  `json:"router,omitempty"`
	Replicas     []ServingDomainSpec `json:"replicas"`
	KVFabrics    []ServingDomainSpec `json:"kv_fabrics,omitempty"`
}

// TopologyOption customizes optional topology parameters.
type TopologyOption func(*ServingTopology)

// WithTopologyRouter configures an optional standalone router/scheduler domain.
func WithTopologyRouter(spec ServingDomainSpec) TopologyOption {
	return func(t *ServingTopology) {
		spec.Role = RoleRouter
		t.Router = &spec
	}
}

// WithTopologyKVFabric adds an explicitly coupled KV fabric failure domain.
func WithTopologyKVFabric(spec ServingDomainSpec) TopologyOption {
	return func(t *ServingTopology) {
		spec.Role = RoleKVFabric
		t.KVFabrics = append(t.KVFabrics, spec)
	}
}

// NewServingTopology creates and validates a topology ensuring separate failure domains.
func NewServingTopology(
	deploymentID string,
	controllerSpec ServingDomainSpec,
	proxySpec ServingDomainSpec,
	replicaSpecs []ServingDomainSpec,
	opts ...TopologyOption,
) (*ServingTopology, error) {
	controllerSpec.Role = RoleController
	proxySpec.Role = RoleProxy

	for i := range replicaSpecs {
		replicaSpecs[i].Role = RoleReplica
	}

	topology := &ServingTopology{
		DeploymentID: deploymentID,
		Controller:   controllerSpec,
		Proxy:        proxySpec,
		Replicas:     replicaSpecs,
	}

	for _, opt := range opts {
		opt(topology)
	}

	if err := topology.Validate(); err != nil {
		return nil, fmt.Errorf("validate topology: %w", err)
	}

	return topology, nil
}

// BuildDefaultTopology constructs a standard topology with separate failure domains for each component.
func BuildDefaultTopology(deploymentID string, replicaCount int, drainTimeout time.Duration, restartBudget int) (*ServingTopology, error) {
	if replicaCount <= 0 {
		return nil, fmt.Errorf("replica count must be greater than 0, got %d", replicaCount)
	}
	if drainTimeout <= 0 {
		drainTimeout = 5 * time.Second
	}
	if restartBudget <= 0 {
		restartBudget = 3
	}

	controllerID := fmt.Sprintf("%s-controller", deploymentID)
	controllerSpec := ServingDomainSpec{
		DomainID:      controllerID,
		ControllerID:  controllerID,
		DrainTimeout:  drainTimeout,
		RestartBudget: restartBudget,
		Role:          RoleController,
	}

	proxySpec := ServingDomainSpec{
		DomainID:      fmt.Sprintf("%s-proxy", deploymentID),
		ControllerID:  controllerID,
		DrainTimeout:  drainTimeout,
		RestartBudget: restartBudget,
		Role:          RoleProxy,
	}

	replicas := make([]ServingDomainSpec, replicaCount)
	for i := 0; i < replicaCount; i++ {
		replicas[i] = ServingDomainSpec{
			DomainID:      fmt.Sprintf("%s-replica-%d", deploymentID, i),
			ControllerID:  controllerID,
			DrainTimeout:  drainTimeout,
			RestartBudget: restartBudget,
			Role:          RoleReplica,
		}
	}

	return NewServingTopology(deploymentID, controllerSpec, proxySpec, replicas)
}

// Validate verifies that controller, proxy, and each replica inhabit strictly separate failure domains.
func (t *ServingTopology) Validate() error {
	if t.DeploymentID == "" {
		return fmt.Errorf("deployment id must not be empty")
	}

	if t.Controller.DomainID == "" {
		return fmt.Errorf("controller domain id must not be empty")
	}

	if t.Proxy.DomainID == "" {
		return fmt.Errorf("proxy domain id must not be empty")
	}

	if len(t.Replicas) == 0 {
		return fmt.Errorf("topology requires at least one replica domain")
	}

	seen := make(map[string]ServingRole)

	// Register Controller domain
	seen[t.Controller.DomainID] = RoleController

	// Validate Proxy separation
	if role, exists := seen[t.Proxy.DomainID]; exists {
		return fmt.Errorf("proxy domain %q collides with %s; must have separate failure domain", t.Proxy.DomainID, role)
	}
	seen[t.Proxy.DomainID] = RoleProxy

	// Validate Router separation if present
	if t.Router != nil {
		if t.Router.DomainID == "" {
			return fmt.Errorf("router domain id must not be empty")
		}
		if role, exists := seen[t.Router.DomainID]; exists {
			return fmt.Errorf("router domain %q collides with %s; must have separate failure domain", t.Router.DomainID, role)
		}
		seen[t.Router.DomainID] = RoleRouter
	}

	// Validate KVFabrics separation if present
	for _, kv := range t.KVFabrics {
		if kv.DomainID == "" {
			return fmt.Errorf("kv fabric domain id must not be empty")
		}
		if role, exists := seen[kv.DomainID]; exists {
			return fmt.Errorf("kv fabric domain %q collides with %s; must have separate failure domain", kv.DomainID, role)
		}
		seen[kv.DomainID] = RoleKVFabric
	}

	// Validate Replicas: each replica MUST have a distinct domain ID
	replicaDomainIDs := make(map[string]struct{}, len(t.Replicas))
	for i, rep := range t.Replicas {
		if rep.DomainID == "" {
			return fmt.Errorf("replica %d domain id must not be empty", i)
		}
		if role, exists := seen[rep.DomainID]; exists {
			return fmt.Errorf("replica domain %q collides with %s; each replica must have an independent failure domain", rep.DomainID, role)
		}
		if _, duplicate := replicaDomainIDs[rep.DomainID]; duplicate {
			return fmt.Errorf("duplicate replica domain %q; each replica must have an independent failure domain", rep.DomainID)
		}
		replicaDomainIDs[rep.DomainID] = struct{}{}
		seen[rep.DomainID] = RoleReplica
	}

	// Validate all domain specs for valid values and coupling rules
	allSpecs := t.Domains()
	for _, spec := range allSpecs {
		if spec.DrainTimeout < 0 {
			return fmt.Errorf("domain %q drain timeout cannot be negative", spec.DomainID)
		}
		if spec.RestartBudget < 0 {
			return fmt.Errorf("domain %q restart budget cannot be negative: %d", spec.DomainID, spec.RestartBudget)
		}

		for _, coupled := range spec.CoupledDomains {
			if coupled == spec.DomainID {
				return fmt.Errorf("domain %q cannot couple to itself", spec.DomainID)
			}
			targetRole, exists := seen[coupled]
			if !exists {
				return fmt.Errorf("domain %q couples to unknown domain %q", spec.DomainID, coupled)
			}
			// Sibling replicas cannot couple to each other: replicas must be isolated
			if spec.Role == RoleReplica && targetRole == RoleReplica {
				return fmt.Errorf("replica %q cannot couple to sibling replica %q; replicas must remain isolated", spec.DomainID, coupled)
			}
		}
	}

	return nil
}

// Domains lists all declared domain specifications in the topology.
func (t *ServingTopology) Domains() []ServingDomainSpec {
	specs := make([]ServingDomainSpec, 0, 2+len(t.Replicas)+len(t.KVFabrics)+1)
	specs = append(specs, t.Controller, t.Proxy)
	if t.Router != nil {
		specs = append(specs, *t.Router)
	}
	specs = append(specs, t.KVFabrics...)
	specs = append(specs, t.Replicas...)
	return specs
}

// Domain finds a domain specification by its DomainID.
func (t *ServingTopology) Domain(domainID string) (ServingDomainSpec, bool) {
	for _, spec := range t.Domains() {
		if spec.DomainID == domainID {
			return spec, true
		}
	}
	return ServingDomainSpec{}, false
}
