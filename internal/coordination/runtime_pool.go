//go:build wip_coordination

package coordination

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// WorkerInstanceFactory instantiates a worker dispatcher for a role specification.
type WorkerInstanceFactory func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error)

type roleSlot struct {
	spec     harnesskit.WorkerSpec
	capacity int
	sem      chan struct{}

	mu     sync.Mutex
	active []harnesskit.WorkerDispatcher
}

// RuntimeWorkerPool implements harnesskit.WorkerPool managing concurrency semaphores
// and allocation lifecycles per role.
type RuntimeWorkerPool struct {
	mu      sync.RWMutex
	roles   map[string]*roleSlot
	factory WorkerInstanceFactory
}

var _ harnesskit.WorkerPool = (*RuntimeWorkerPool)(nil)

// NewRuntimeWorkerPool creates a new pool initialized from a manifest and instance factory.
func NewRuntimeWorkerPool(manifest harnesskit.CoordinationManifest, factory WorkerInstanceFactory) (*RuntimeWorkerPool, error) {
	if factory == nil {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "pool.new",
			Err:  errors.New("worker instance factory is required"),
		}
	}

	pool := &RuntimeWorkerPool{
		roles:   make(map[string]*roleSlot),
		factory: factory,
	}

	maxConcurrency := manifest.Manager.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = harnesskit.DefaultMaxConcurrency
	}

	for roleID, spec := range manifest.Workers {
		capLimit := maxConcurrency
		if spec.Metadata != nil {
			if v, ok := spec.Metadata["max_concurrency"]; ok {
				if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
					capLimit = parsed
				}
			} else if v, ok := spec.Metadata["concurrency"]; ok {
				if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
					capLimit = parsed
				}
			}
		}

		if err := pool.RegisterRole(spec, capLimit); err != nil {
			return nil, err
		}
		_ = roleID
	}

	return pool, nil
}

// RegisterRole registers a worker role spec and its semaphore capacity.
func (p *RuntimeWorkerPool) RegisterRole(spec harnesskit.WorkerSpec, capacity int) error {
	if err := spec.Validate(); err != nil {
		return &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "pool.register_role",
			Err:  err,
		}
	}

	if capacity <= 0 {
		if spec.Metadata != nil {
			if v, ok := spec.Metadata["max_concurrency"]; ok {
				if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
					capacity = parsed
				}
			} else if v, ok := spec.Metadata["concurrency"]; ok {
				if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
					capacity = parsed
				}
			}
		}
	}
	if capacity <= 0 {
		capacity = harnesskit.DefaultMaxConcurrency
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	p.roles[spec.RoleID] = &roleSlot{
		spec:     spec,
		capacity: capacity,
		sem:      make(chan struct{}, capacity),
		active:   make([]harnesskit.WorkerDispatcher, 0, capacity),
	}

	return nil
}

// Acquire acquires a worker dispatcher for the requested role, subject to role concurrency limits.
func (p *RuntimeWorkerPool) Acquire(ctx context.Context, roleID string) (harnesskit.WorkerDispatcher, error) {
	if err := ctx.Err(); err != nil {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeCanceled,
			Op:   "pool.acquire",
			Err:  err,
		}
	}

	p.mu.RLock()
	slot, exists := p.roles[roleID]
	p.mu.RUnlock()

	if !exists {
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "pool.acquire",
			Err:  fmt.Errorf("unknown role %q", roleID),
		}
	}

	// Acquire semaphore token
	select {
	case slot.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeCanceled,
			Op:   "pool.acquire",
			Err:  ctx.Err(),
		}
	}

	// Create worker instance via factory
	worker, err := p.factory(ctx, slot.spec)
	if err != nil {
		// Release acquired semaphore token on factory error
		<-slot.sem
		return nil, &harnesskit.Error{
			Code: harnesskit.CodeInternal,
			Op:   "pool.acquire",
			Err:  fmt.Errorf("failed to instantiate worker for role %q: %w", roleID, err),
		}
	}

	slot.mu.Lock()
	slot.active = append(slot.active, worker)
	slot.mu.Unlock()

	return worker, nil
}

// Release returns a worker dispatcher to the pool and frees its semaphore slot.
// Fails closed if the role is unknown or if the worker was not acquired or already released.
func (p *RuntimeWorkerPool) Release(ctx context.Context, roleID string, worker harnesskit.WorkerDispatcher) error {
	if worker == nil {
		return &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "pool.release",
			Err:  errors.New("cannot release nil worker"),
		}
	}

	p.mu.RLock()
	slot, exists := p.roles[roleID]
	p.mu.RUnlock()

	if !exists {
		return &harnesskit.Error{
			Code: harnesskit.CodeInvalid,
			Op:   "pool.release",
			Err:  fmt.Errorf("unknown role %q", roleID),
		}
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	idx := -1
	for i, act := range slot.active {
		if sameWorker(act, worker) {
			idx = i
			break
		}
	}

	if idx < 0 {
		return &harnesskit.Error{
			Code: harnesskit.CodeConflict,
			Op:   "pool.release",
			Err:  fmt.Errorf("extra release or unallocated worker for role %q", roleID),
		}
	}

	// Remove from active list
	slot.active = append(slot.active[:idx], slot.active[idx+1:]...)

	// Drain semaphore token
	select {
	case <-slot.sem:
	default:
		// Should not happen if bookkeeping is consistent, but fail closed if it does
		return &harnesskit.Error{
			Code: harnesskit.CodeInternal,
			Op:   "pool.release",
			Err:  fmt.Errorf("inconsistent semaphore state for role %q", roleID),
		}
	}

	return nil
}

// Available returns the number of available slots for a role. Returns 0 for unknown roles.
func (p *RuntimeWorkerPool) Available(roleID string) int {
	p.mu.RLock()
	slot, exists := p.roles[roleID]
	p.mu.RUnlock()

	if !exists {
		return 0
	}

	slot.mu.Lock()
	defer slot.mu.Unlock()

	avail := slot.capacity - len(slot.active)
	if avail < 0 {
		return 0
	}
	return avail
}

// Capacity returns the maximum concurrent slots configured for a role. Returns 0 for unknown roles.
func (p *RuntimeWorkerPool) Capacity(roleID string) int {
	p.mu.RLock()
	slot, exists := p.roles[roleID]
	p.mu.RUnlock()

	if !exists {
		return 0
	}

	return slot.capacity
}

func sameWorker(a, b harnesskit.WorkerDispatcher) bool {
	if a == nil || b == nil {
		return a == b
	}
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	if va.Kind() == reflect.Pointer && vb.Kind() == reflect.Pointer {
		return va.Pointer() == vb.Pointer()
	}
	if reflect.TypeOf(a).Comparable() && reflect.TypeOf(b).Comparable() {
		return a == b
	}
	return false
}
