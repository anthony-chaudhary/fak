package speculative

import (
	"context"
	"errors"
	"fmt"
)

// Commit promotes changes from b.ForkedArena back to b.ParentArena (or base workspace),
// retains speculative tokens in KVTree as the active trunk, tears down ephemeral branch arena,
// and marks status as Committed.
func (b *Branch) Commit(ctx context.Context, leaseOwner string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Status == Committed {
		return errors.New("speculative: branch already committed")
	}
	if b.Status == Aborted {
		return errors.New("speculative: cannot commit aborted branch")
	}
	if b.ForkedArena == nil {
		return errors.New("speculative: forked arena is nil")
	}

	// 1. Promote changes from forked arena back to parent arena and base workspace
	if err := b.ForkedArena.Promote(ctx); err != nil {
		return fmt.Errorf("speculative: failed to promote forked arena changes: %w", err)
	}

	// 2. Retain speculative tokens in KVTree as the active trunk
	if b.KVTree != nil {
		treeLock := getTreeLock(b.KVTree)
		treeLock.Lock()
		b.KVTree.MatchLen(b.fullTokens)
		treeLock.Unlock()
	}

	// 3. Teardown ephemeral branch arena
	if err := b.ForkedArena.Destroy(); err != nil {
		return fmt.Errorf("speculative: failed to destroy forked arena: %w", err)
	}

	// 4. Mark status as Committed
	b.Status = Committed
	b.LeaseOwner = leaseOwner
	return nil
}
