package routing

import (
	"fmt"
	"strings"
	"sync"
)

// PolicyRegistry is the named-policy manifest registry: the pure data map from
// a published PolicyRef (name + version) to the policy BODY the fabric routes
// with. The control plane publishes which named, versioned policy is current
// (DesiredState.Policy); the registry resolves the body, so a consumer routes
// with the published policy read live rather than passed by hand.
//
// Publication is an explicit Register: it validates the policy loud (a
// malformed manifest fails before it routes) and replaces only an explicit
// re-register of the same name+version. The registry never live-patches a
// body and never invents one: resolving an unknown name+version is a miss, and
// the caller falls back — the fabric answers with its built-in default rather
// than guessing.
//
// This closes the gap RouteChainWithPosture documented ("no manifest registry
// exists yet"): the registry IS the manifest registry, and it lives in the
// fabric because the policy manifest is fabric data — the control plane
// publishes the reference, the fabric resolves the body.

// policyKey is the registry's composite identity: a policy is (name, version).
type policyKey struct {
	name    string
	version string
}

// PolicyRegistry maps published policy references to policy bodies. It is safe
// for concurrent use; Register and Resolve are the only methods.
type PolicyRegistry struct {
	mu      sync.RWMutex
	entries map[policyKey]Policy
}

// NewPolicyRegistry returns an empty registry.
func NewPolicyRegistry() *PolicyRegistry {
	return &PolicyRegistry{entries: make(map[policyKey]Policy)}
}

// DefaultPolicyName is the built-in policy's registry name.
const DefaultPolicyName = "default"

var (
	defaultPolicyRegistryOnce sync.Once
	defaultPolicyRegistryInst *PolicyRegistry
)

// DefaultPolicyRegistry returns the process-wide registry pre-populated with
// the built-in named policy ("default" -> DefaultPolicy() at version
// routing-policy/1). Consumers publish additional named bodies by registering
// them here; resolution then sees them live on the next routing call, so a
// published policy takes effect without a restart.
//
// WARNING (coupled-failures TICKET-05): the default registry is process-wide
// shared fate — any in-process Register (including tests) changes every
// subsequent route that reads it live. A wave that must be insulated from a
// concurrent publish takes Snapshot() at spawn and routes against the copy;
// see resbroker.RouteChainWithPostureAndRegistry.
func DefaultPolicyRegistry() *PolicyRegistry {
	defaultPolicyRegistryOnce.Do(func() {
		r := NewPolicyRegistry()
		dp := DefaultPolicy()
		_ = r.Register(DefaultPolicyName, dp.Version, dp)
		defaultPolicyRegistryInst = r
	})
	return defaultPolicyRegistryInst
}

// Snapshot returns an isolated copy of the registry as of this call: every
// name+version body present now is carried over, and later Register calls on
// either registry (the source or the copy) do not affect the other. A spawn
// wave captures a snapshot so a concurrent publish — or a test registering a
// same-named body — cannot silently change the wave's routing outcome after
// admission. A nil receiver yields an empty registry, never a panic.
func (r *PolicyRegistry) Snapshot() *PolicyRegistry {
	out := NewPolicyRegistry()
	if r == nil {
		return out
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	for k, p := range r.entries {
		out.entries[k] = p
	}
	return out
}

// Register publishes one named policy body. The name is required; a version
// that is empty defaults to the policy's own Version. The policy is validated
// loud — a malformed manifest is a caller bug and must fail before it routes.
// Re-registering the same name+version overwrites (an operator's explicit
// update); a DIFFERENT version under the same name coexists, because a
// versioned policy set is the audited history, not a single slot.
func (r *PolicyRegistry) Register(name, version string, p Policy) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("routing: policy registry requires a name")
	}
	if version == "" {
		version = p.Version
	}
	if err := p.Validate(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries[policyKey{name: name, version: version}] = p
	return nil
}

// Resolve returns the policy body a published reference names, and whether one
// exists. A non-empty version must match exactly. An EMPTY version resolves
// the single entry registered under that name when exactly one exists —
// deterministic, never "latest wins" — and is a miss otherwise, so an
// ambiguous reference fails closed to the caller's fallback instead of
// silently picking one of several.
func (r *PolicyRegistry) Resolve(name, version string) (Policy, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if version != "" {
		p, ok := r.entries[policyKey{name: name, version: version}]
		return p, ok
	}
	seen := Policy{}
	count := 0
	for k, p := range r.entries {
		if k.name == name {
			seen = p
			count++
		}
	}
	if count != 1 {
		return Policy{}, false
	}
	return seen, true
}
