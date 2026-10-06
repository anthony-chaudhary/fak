package agent

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const inKernelSnapshotCheckpointTokens = 64

// inKernelSnapshotCheckpoint returns the deepest fixed block boundary strictly before
// the prompt and after the already restored prefix. A strict-before boundary preserves
// the full leaf for exact hits while creating a reusable ancestor for sibling suffixes.
func inKernelSnapshotCheckpoint(matched, promptTokens int) int {
	if promptTokens <= 1 {
		return 0
	}
	checkpoint := ((promptTokens - 1) / inKernelSnapshotCheckpointTokens) * inKernelSnapshotCheckpointTokens
	if checkpoint <= matched {
		return 0
	}
	return checkpoint
}

// inKernelAdaptiveSnapshotCheckpoint materializes a complete snapshot
// at the exact boundary of a structurally shared prefix. Recurrent
// snapshots cannot be synthesized by splitting a later leaf, so this repairs the
// boundary for the next sibling while retaining the historical one-checkpoint
// limit and strict-before-prompt fallback.
//
// The exact shared boundary earns the single checkpoint only when it is at least as
// deep as the grid fallback or saves at least one block over the restored prefix. A
// sub-block structural match (a chat-template header shared with an earlier request,
// a warmup leaf) must not displace the deepest grid checkpoint: on the first request
// of an agent loop that shallow boundary is the only one known, and spending the
// checkpoint there left the second turn restoring 5 of 1922 shared tokens (#12742).
// Adapted from oMLX's off-grid prefill-tail snapshots (Apache-2.0):
// https://github.com/jundot/omlx/blob/8288884d9b4f6db7b547633a94d36794c6b1d52d/omlx/scheduler.py
func inKernelAdaptiveSnapshotCheckpoint(matched, cacheable, promptTokens int) int {
	grid := inKernelSnapshotCheckpoint(matched, promptTokens)
	if cacheable > matched && cacheable < promptTokens &&
		(cacheable >= grid || cacheable-matched >= inKernelSnapshotCheckpointTokens) {
		return cacheable
	}
	return grid
}

// admitPrefillCheckpoint snapshots s (positioned at the end of prefix) and admits
// it. optional marks the shared-prefix boundary snapshot, an opportunistic extra:
// when the snapshot byte budget refuses it the request proceeds without it instead
// of failing, while the historical divergence/grid checkpoint keeps its contract.
func (p *InKernelPlanner) admitPrefillCheckpoint(ctx context.Context, s *model.Session, prefix []int, logits []float32, optional bool) error {
	snap, err := s.PrefixSnapshot()
	if err != nil {
		return err
	}
	if err := p.admitPrefixSnapshot(ctx, prefix, snap, logits); err != nil {
		snap.Close()
		if optional && errors.Is(err, radixkv.ErrSnapshotByteBudget) {
			return nil
		}
		return err
	}
	return nil
}

// admitPrefixSnapshot transfers snapshot ownership to the same scoped/unscoped tree
// used by lookup. On error ownership remains with the caller.
func (p *InKernelPlanner) admitPrefixSnapshot(ctx context.Context, ids []int, snap *model.PrefixSnapshot, logits []float32) error {
	if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped && p.scopedTree != nil {
		return p.scopedTree.AdmitPrivateSnapshot(owner, ids, snap, logits)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	b, matched := p.tree.Lookup(ids)
	leaf, err := p.tree.InsertSnapshot(b, ids[matched:], snap, logits)
	if leaf != nil {
		p.tree.Done(leaf)
	}
	return err
}

// admitGeneratedContinuation admits the exact state the decoder already owns after
// successful forwards. It deliberately does not Step the last emitted token merely to
// extend the cache: generation semantics and model work remain unchanged.
func (p *InKernelPlanner) admitGeneratedContinuation(ctx context.Context, s *model.Session, prompt, forwarded []int, logits []float32) {
	if p == nil || p.tree == nil || s == nil || len(forwarded) == 0 || ctx.Err() != nil {
		return
	}
	tokens := make([]int, 0, len(prompt)+len(forwarded))
	tokens = append(tokens, prompt...)
	tokens = append(tokens, forwarded...)
	if ctx.Err() != nil {
		return
	}
	if p.backend != nil {
		snap, err := s.PrefixSnapshot()
		if err != nil {
			return
		}
		if ctx.Err() != nil || snap.Tokens != len(tokens) {
			snap.Close()
			return
		}
		if err := p.admitPrefixSnapshot(ctx, tokens, snap, logits); err != nil {
			snap.Close()
		}
		return
	}
	if s.Cache == nil || s.Cache.Len() != len(tokens) {
		return
	}
	// Resident Qwen GDN decode owners advance the recurrent state without touching
	// the host Cache.linear the bare clone below copies. Synchronize first; when
	// the live state cannot be read, admit nothing rather than a stale prefix.
	if err := s.SyncQwen35ResidentGDNStateToHost(); err != nil {
		return
	}
	if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped && p.scopedTree != nil {
		_ = p.scopedTree.AdmitPrivate(owner, tokens, s.Cache, logits)
		return
	}
	p.mu.Lock()
	b, matched := p.tree.Lookup(tokens)
	leaf := p.tree.InsertCloneWithLogits(b, tokens[matched:], s.Cache, logits)
	p.tree.Done(leaf)
	p.mu.Unlock()
}

func (p *InKernelPlanner) sessionFromPrefixClone(prefix *model.KVCache) *model.Session {
	if p.backend != nil {
		s := p.m.NewBackendSession(p.backend)
		s.Cache = prefix
		return s
	}
	s := p.m.NewSession()
	s.Cache = prefix
	return s
}

func inKernelRefeedLastTokenForExactHit(s *model.Session, promptLen int) bool {
	if s == nil || s.Cache == nil || promptLen <= 0 || s.Cache.Len() < promptLen {
		return false
	}
	removed, err := s.Cache.TryEvict(promptLen-1, 1)
	return err == nil && removed == 1 && s.Cache.Len() == promptLen-1
}
