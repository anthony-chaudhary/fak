package speculative

import (
	"context"
	"errors"
	"fmt"
)

// Abort performs instantaneous resource teardown of the speculative branch:
//   - Destroys ForkedArena (b.ForkedArena.Destroy()), wiping the CoW upper directory in <5ms.
//   - Evicts speculative tokens from KVTree via b.KVTree.EvictPrefix(fullTokens), ensuring zero
//     speculative tokens remain in the prompt cache.
//   - Marks status as Aborted. Host disk and parent arena remain 100% untouched.
func (b *Branch) Abort(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Status == Aborted {
		return nil
	}
	if b.Status == Committed {
		return errors.New("speculative: cannot abort committed branch")
	}

	var errs []error

	// 1. Instantaneous resource teardown: destroy child CoW arena (<5ms)
	if b.ForkedArena != nil {
		if err := b.ForkedArena.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("speculative: failed to destroy forked arena: %w", err))
		}
	}

	// 2. KV eviction: evict speculative tokens from prompt cache
	if b.KVTree != nil && len(b.SpeculativeTokens) > 0 {
		treeLock := getTreeLock(b.KVTree)
		treeLock.Lock()
		b.KVTree.EvictPrefix(b.fullTokens)
		treeLock.Unlock()
	}

	b.Status = Aborted
	return errors.Join(errs...)
}
