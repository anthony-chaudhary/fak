package agent

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"time"
)

// lookupOwnedNativePrefix is an ordinary-execution extraction, not candidate
// preparation. Named results intentionally retain partially assigned provenance
// on errors. callerErr aliases the caller's named error result so panic-time
// terminal telemetry observes the same error assignments as the original block.
// The caller owns page leases, cold creation, cleanup registration,
// session configuration and deadline checks at their existing outer boundaries.
// Existing warm-claim, locking, lookup metrics and exact-hit behavior is retained.
func (p *InKernelPlanner) lookupOwnedNativePrefix(ctx context.Context, ids []int, lookupStart time.Time, callerErr *error) (s *model.Session, cachedLogits []float32, cacheable, matched int, sourceTier radixkv.SnapshotTier, closeSession, skipExactDeviceL1Readmission bool, lookupLegacyM, lookupSnapshotM int, lookupSnapshotConsulted bool) {
	owner, scoped := prefixCacheIdentityFromContext(ctx)
	scopedLookup := scoped && p.scopedTree != nil
	var matchedKV *model.KVCache
	var matchedSnapshot *model.PrefixSnapshot
	var m int
	var tier radixkv.SnapshotTier
	var sourceScope radixkv.ShareScope
	if scopedLookup {
		cacheable, *callerErr = p.scopedTree.MatchLen(owner, ids)
		if *callerErr != nil {
			return
		}
		if p.backend != nil {
			matchedSnapshot, cachedLogits, m, sourceScope, tier, *callerErr = p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
			lookupSnapshotConsulted, lookupSnapshotM = true, m
		} else {
			matchedKV, cachedLogits, m, _, *callerErr = p.scopedTree.Lookup(owner, ids)
			if *callerErr == nil && p.warmHandoffScopeMatches(owner) {
				// CW-16 (#13336): the scoped lookup did not serve this request, but a
				// live startup warm may own a prepared prefix over the SAME scope. The
				// handoff adopts that prefix (reading it back through this same scoped
				// tree) and releases the startup claim, so the first real request owns
				// the continuation instead of re-prefilling it. A mismatch/expiry is a
				// truthful miss: the ordinary cold path below runs unchanged.
				if hand := p.demandWarmHandoff(ids); hand.HandedOff && hand.MatchedTokens > m {
					if hKV, hLogits, hMatched, _, hErr := p.scopedTree.Lookup(owner, ids); hErr == nil && hKV != nil {
						matchedKV, cachedLogits, m = hKV, hLogits, hMatched
					}
				}
			}
			lookupLegacyM = m
			if inKernelHostSnapshotReuse(p) && matchedKV == nil {
				// Recurrent hybrid on the host-session (Metal) seam: a mid-edge
				// split carries no KV because span eviction is unsupported, so the
				// only restorable boundary is a complete PrefixSnapshot admitted at
				// an adaptive block boundary. Consult that tier before falling open.
				var snap *model.PrefixSnapshot
				snap, cachedLogits, m, sourceScope, tier, *callerErr = p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
				lookupSnapshotConsulted, lookupSnapshotM = true, m
				if *callerErr == nil {
					matchedSnapshot = snap
				} else if snap != nil {
					snap.Close()
				}
			} else if inKernelHostSnapshotReuse(p) && *callerErr == nil && m < len(ids) && cacheable > m {
				// A KV-bearing match in one scope (an agent-private or fleet-promoted
				// prefix, a warm handoff) can be shallower than a complete snapshot
				// another scope admitted at a deeper checkpoint. Keep whichever
				// restorable state is deeper (#12742). The tiered lookup materializes
				// its best candidate, so it runs only when the cross-scope structural
				// match (an upper bound on any snapshot depth) is deeper than the KV.
				// A snapshot fault fails open to the KV already in hand unless the
				// request itself was cancelled.
				snap, snapLogits, snapM, snapScope, snapTier, snapErr := p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
				lookupSnapshotConsulted = true
				if snapErr == nil {
					lookupSnapshotM = snapM
				} else if ctxErr := ctx.Err(); ctxErr != nil {
					*callerErr = ctxErr
				}
				if snapErr == nil && snap != nil && snapM > m {
					matchedKV = nil
					matchedSnapshot, cachedLogits, m, sourceScope, tier = snap, snapLogits, snapM, snapScope, snapTier
				} else if snap != nil {
					snap.Close()
				}
			}
		}
	} else {
		p.mu.Lock()
		if p.backend != nil {
			b, snap, legacyMatched, lookupTier, lookupErr := p.tree.LookupSnapshotTieredContextLocked(ctx, ids, &p.mu)
			matchedSnapshot, m, *callerErr = snap, legacyMatched, lookupErr
			lookupSnapshotConsulted, lookupSnapshotM = true, legacyMatched
			if b != nil {
				cacheable = b.Plen()
			}
			tier = lookupTier
			if m >= len(ids) {
				cachedLogits = b.Logits()
			}
			p.tree.Done(b)
		} else {
			b, legacyMatched := p.tree.Lookup(ids)
			m = legacyMatched
			lookupLegacyM = legacyMatched
			if b != nil {
				cacheable = b.Plen()
			}
			if k := b.CloneKV(); k != nil {
				matchedKV = k
				if m >= len(ids) {
					cachedLogits = b.Logits()
				}
			}
			p.tree.Done(b)
			if inKernelHostSnapshotReuse(p) && matchedKV == nil {
				// See the scoped arm above: on a host-session recurrent hybrid a
				// mid-edge split owns no KV, so only an admitted complete snapshot
				// at a block boundary is restorable. The boundary node it returns
				// carries the structural Plen for cacheability; the snapshot length
				// is the realized, restorable prefix.
				var lookupTier radixkv.SnapshotTier
				site, snap, snapMatched, lookupTier, lookupErr := p.tree.LookupSnapshotTieredContextLocked(ctx, ids, &p.mu)
				if site != nil {
					if site.Plen() > cacheable {
						cacheable = site.Plen()
					}
					p.tree.Done(site)
				}
				m = snapMatched
				lookupSnapshotConsulted, lookupSnapshotM = true, snapMatched
				if lookupErr == nil {
					matchedSnapshot, cachedLogits, tier = snap, nil, lookupTier
				} else if snap != nil {
					snap.Close()
				}
			}
		}
		p.mu.Unlock()
	}
	if *callerErr != nil {
		return
	}
	// Cached logits describe the distribution after the exact cached prefix.
	// A partial hit must prefill its divergent suffix before sampling; scoped
	// snapshot lookup returns the ancestor's logits for observability, so do
	// not let those logits accidentally bypass the suffix forward.
	if m < len(ids) {
		cachedLogits = nil
	}
	// The lookup-side (cacheability) half of the #3390 split is the longest
	// structurally visible token prefix. m is narrower: the prefix backed by a
	// restorable KV/snapshot. Separate scoped visibility lookups can race only
	// toward a longer restorable prefix, so keep cacheable at least m. The realized
	// `matched` can only stay at or fall below this structural upper bound.
	if cacheable < m {
		cacheable = m
	}
	if matchedSnapshot != nil {
		s, *callerErr = p.restoreOwnedPrefixSnapshot(matchedSnapshot)
		if *callerErr != nil {
			return
		}
		closeSession, matched, sourceTier = true, m, tier
	} else if matchedKV != nil {
		s = p.sessionFromPrefixClone(matchedKV)
		closeSession = p.backend != nil
		matched = m
		sourceTier = radixkv.SnapshotTierDeviceL1
	}
	// Fully cached (an exact-duplicate transcript): the cached KV has the prefix but
	// decode still needs the last-token logits to sample the first generated token. New
	// leaves carry those logits; older/split leaves may not. When absent, refeed only
	// the final token: evicting the final cached row leaves an exact prefix of len(ids)-1,
	// and Prefill below recomputes that one row/logits. If the cache cannot truncate
	// exactly (for example a recurrent hybrid), fail open to the old full-prefill path.
	if s != nil && matched >= len(ids) && cachedLogits == nil {
		if inKernelRefeedLastTokenForExactHit(s, len(ids)) {
			matched = len(ids) - 1
		} else {
			if closeSession {
				s.Close()
			}
			s, matched, closeSession = nil, 0, false
			sourceTier = radixkv.SnapshotTierMiss
		}
	}
	// The unscoped hot tree already owns the full snapshot and lookup restored a
	// deep clone into this session. Cached logits mean no prompt state changed, so
	// capturing and admitting that same state again only performs a second clone.
	// Scoped hits from tenant or private scope similarly skip re-admission because
	// the tenant materialization already exists.
	skipExactDeviceL1Readmission = ((!scopedLookup) || (scopedLookup && (sourceScope == radixkv.ScopeTenant || sourceScope == radixkv.ScopeAgent))) &&
		matchedSnapshot != nil &&
		matched == len(ids) && cachedLogits != nil && sourceTier == radixkv.SnapshotTierDeviceL1
	enginestep.Default.ObservePhase(enginestep.PhasePrefixLookup, time.Since(lookupStart))
	enginestep.Default.ObservePrefixMatched(matched)
	return
}
