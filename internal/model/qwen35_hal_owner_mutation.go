package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// mutableOwnerLocked returns an exclusive owner with owner.mu held. The
// containing qwen35HALState.mu must already be held. A failed pair clone leaves
// the shared owner and refcount unchanged.
func (l *qwen35HALLayerState) mutableOwnerLocked(backend compute.Backend, partial *qwen35HALLayerRelease) (*qwen35HALLayerOwner, error) {
	owner := l.ownerLocked(backend)
	if owner == nil {
		return nil, fmt.Errorf("model: missing Qwen3.5 recurrent state")
	}
	owner.mu.Lock()
	locked := owner
	defer func() {
		if locked != nil {
			locked.mu.Unlock()
		}
	}()
	if owner.refs == 1 {
		locked = nil // transfer the held lock to the mutation caller
		return owner, nil
	}
	cloner, ok := backend.(compute.TensorCloner)
	if !ok {
		return nil, fmt.Errorf("model: backend %T cannot clone Qwen3.5 recurrent state", backend)
	}
	// Publish each returned partial copy before another callback can interrupt.
	// The caller releases it after both this owner and its enclosing state unlock.
	partial.backend = backend
	conv, err := cloner.CloneTensor(owner.conv)
	partial.conv = conv
	if err != nil {
		return nil, fmt.Errorf("model: clone Qwen3.5 convolution state: %w", err)
	}
	recurrent, err := cloner.CloneTensor(owner.recurrent)
	partial.recurrent = recurrent
	if err != nil {
		return nil, fmt.Errorf("model: clone Qwen3.5 recurrent state: %w", err)
	}
	owner.refs--
	owner.mu.Unlock()
	locked = nil
	owner = &qwen35HALLayerOwner{refs: 1, backend: backend, conv: conv, recurrent: recurrent}
	owner.mu.Lock()
	l.owner = owner
	l.conv, l.recurrent = conv, recurrent
	*partial = qwen35HALLayerRelease{} // physical ownership moved to the new pair
	return owner, nil
}
