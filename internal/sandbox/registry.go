package sandbox

import (
	"context"
	"fmt"
	"sync"
)

// Registry maintains a thread-safe registry of sandbox Providers mapped by their unique names.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
}

// NewRegistry initializes an empty Registry.
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

// RegisterProvider adds or replaces a Provider in the registry.
func (r *Registry) RegisterProvider(p Provider) {
	if p == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[p.Name()] = p
}

// GetProvider retrieves a Provider by its registered name.
func (r *Registry) GetProvider(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[name]
	return p, ok
}

// ResolveTier finds an available Provider for the requested sandbox Tier.
// Returns an ErrSandboxUnavailable error if no suitable provider is registered or available.
func (r *Registry) ResolveTier(tier Tier) (Provider, error) {
	if !tier.Valid() {
		return nil, NewSandboxError(ErrSandboxUnavailable, fmt.Sprintf("invalid sandbox tier: %q", tier))
	}
	r.mu.RLock()
	defer r.mu.RUnlock()

	// 1. Prefer an available provider matching the requested tier
	for _, p := range r.providers {
		if p.Tier() == tier && p.Available() {
			return p, nil
		}
	}

	// 2. Check if a provider exists for the tier but is unavailable
	for _, p := range r.providers {
		if p.Tier() == tier {
			return nil, NewSandboxError(ErrSandboxUnavailable, fmt.Sprintf("sandbox provider %q for tier %q is unavailable", p.Name(), tier))
		}
	}

	return nil, NewSandboxError(ErrSandboxUnavailable, fmt.Sprintf("no registered provider for tier %q", tier))
}

// Execute validates the specification, resolves the provider for spec.Tier, instantiates
// a sandbox instance, and executes the requested command, ensuring instance teardown.
func (r *Registry) Execute(ctx context.Context, spec Spec, req ExecutionRequest) (ExecutionResult, error) {
	if err := spec.Validate(); err != nil {
		return ExecutionResult{}, err
	}
	provider, err := r.ResolveTier(spec.Tier)
	if err != nil {
		return ExecutionResult{}, err
	}
	inst, err := provider.Create(ctx, spec)
	if err != nil {
		return ExecutionResult{}, err
	}
	defer inst.Close()
	return inst.Execute(ctx, req)
}

// Global default registry singleton.
var (
	defaultReg     *Registry
	defaultRegOnce sync.Once
)

// DefaultRegistry returns the process-wide default Provider registry.
func DefaultRegistry() *Registry {
	defaultRegOnce.Do(func() {
		if defaultReg == nil {
			defaultReg = NewRegistry()
		}
		RegisterDefaultProviders()
	})
	return defaultReg
}

// RegisterDefaultProviders registers default sandbox providers (e.g. L1 native OS) into the default registry.
func RegisterDefaultProviders() {
	if defaultReg == nil {
		defaultReg = NewRegistry()
	}
	defaultReg.RegisterProvider(NewL1Provider())
}

// RegisterProvider registers a provider into the DefaultRegistry.
func RegisterProvider(p Provider) {
	DefaultRegistry().RegisterProvider(p)
}

// GetProvider retrieves a provider from the DefaultRegistry.
func GetProvider(name string) (Provider, bool) {
	return DefaultRegistry().GetProvider(name)
}

// ResolveTier resolves a provider for tier from the DefaultRegistry.
func ResolveTier(tier Tier) (Provider, error) {
	return DefaultRegistry().ResolveTier(tier)
}

// Execute executes a command via the DefaultRegistry.
func Execute(ctx context.Context, spec Spec, req ExecutionRequest) (ExecutionResult, error) {
	return DefaultRegistry().Execute(ctx, spec, req)
}
