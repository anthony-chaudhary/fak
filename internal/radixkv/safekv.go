package radixkv

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ShareScope is the visibility of a cached prefix. Prefixes are admitted at a
// private scope and are never visible through a broader scope until Promote.
type ShareScope uint8

const (
	ScopeAgent ShareScope = iota
	ScopeTenant
	ScopeFleet
	ScopePrivate = ScopeTenant
)

var (
	ErrCacheIdentity = errors.New("radixkv: cache identity is incomplete")
	ErrCacheScope    = errors.New("radixkv: invalid cache share scope")
	ErrPrefixAbsent  = errors.New("radixkv: prefix is not cached at the requested scope")
)

// CacheIdentity is the non-secret ownership key for private prefix visibility.
// Tenant is required; Agent is required only for ScopeAgent.
type CacheIdentity struct {
	Tenant string
	Agent  string
}

// ScopedTree adds private-by-default visibility and explicit promotion to Tree.
// The mutex also supplies Tree's required external serialization.
type ScopedTree struct {
	lock sync.Locker
	tree *Tree
}

func NewScoped(budgetTokens int) *ScopedTree {
	return WrapScoped(New(budgetTokens))
}

// WrapScoped applies scoped visibility to an existing tree. The caller must
// route every access through the returned wrapper once wrapping begins.
func WrapScoped(tree *Tree) *ScopedTree {
	return WrapScopedWithLocker(tree, &sync.Mutex{})
}

// WrapScopedWithLocker shares serialization with other access paths over tree.
// This is required when a legacy single-user path and scoped requests coexist.
func WrapScopedWithLocker(tree *Tree, lock sync.Locker) *ScopedTree {
	if tree == nil {
		tree = New(0)
	}
	if lock == nil {
		lock = &sync.Mutex{}
	}
	return &ScopedTree{tree: tree, lock: lock}
}

// FlightGroup returns a PrefixFlightGroup synchronized with this ScopedTree.
func (s *ScopedTree) FlightGroup() *PrefixFlightGroup {
	if s == nil {
		return NewPrefixFlightGroup(nil)
	}
	return NewPrefixFlightGroupWithLocker(s.tree, s.lock)
}

// MatchLen returns the longest token prefix visible to owner across agent,
// tenant, and explicitly promoted fleet scopes. It reports structural cache
// identity only: callers must still use Lookup/LookupSnapshot to determine
// whether the matched prefix has a reusable KV payload.
func (s *ScopedTree) MatchLen(owner CacheIdentity, tokens []int) (int, error) {
	if strings.TrimSpace(owner.Tenant) == "" {
		return 0, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	best := 0
	for _, scope := range []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet} {
		if scope == ScopeAgent && strings.TrimSpace(owner.Agent) == "" {
			continue
		}
		ns, err := scopeNamespace(scope, owner)
		if err != nil {
			continue
		}
		if matched := s.tree.MatchLenNS(ns, tokens); matched > best {
			best = matched
		}
	}
	return best, nil
}

func scopeNamespace(scope ShareScope, owner CacheIdentity) (string, error) {
	tenant := strings.TrimSpace(owner.Tenant)
	agent := strings.TrimSpace(owner.Agent)
	switch scope {
	case ScopeAgent:
		if tenant == "" || agent == "" {
			return "", ErrCacheIdentity
		}
		return "private/tenant/" + tenant + "/agent/" + agent, nil
	case ScopeTenant:
		if tenant == "" {
			return "", ErrCacheIdentity
		}
		return "private/tenant/" + tenant, nil
	case ScopeFleet:
		return "shared/fleet", nil
	default:
		return "", ErrCacheScope
	}
}

// AdmitPrivate stores a prefix at tenant scope. This is the default admission
// path: another tenant cannot observe a match until an explicit Promote call.
func (s *ScopedTree) AdmitPrivate(owner CacheIdentity, tokens []int, kv *model.KVCache, logits []float32) error {
	return s.Admit(ScopeTenant, owner, tokens, kv, logits)
}

// Admit stores a prefix at an explicit scope. Direct ScopeFleet admission is
// rejected so broad visibility always has a private source and promotion event.
func (s *ScopedTree) Admit(scope ShareScope, owner CacheIdentity, tokens []int, kv *model.KVCache, logits []float32) error {
	if scope == ScopeFleet {
		return ErrCacheScope
	}
	ns, err := scopeNamespace(scope, owner)
	if err != nil {
		return err
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	boundary, matched := s.tree.LookupNS(ns, tokens)
	leaf := s.tree.InsertCloneWithLogits(boundary, tokens[matched:], kv, logits)
	s.tree.Done(leaf)
	return nil
}

// AdmitPrivateSnapshot stores a complete backend prefix at tenant scope.
func (s *ScopedTree) AdmitPrivateSnapshot(owner CacheIdentity, tokens []int, snap *model.PrefixSnapshot, logits []float32) error {
	ns, err := scopeNamespace(ScopeTenant, owner)
	if err != nil {
		return err
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	boundary, matched := s.tree.LookupNS(ns, tokens)
	leaf, err := s.tree.InsertSnapshot(boundary, tokens[matched:], snap, logits)
	if leaf != nil {
		s.tree.Done(leaf)
	}
	return err
}

// LookupSnapshot returns the longest visible independently owned backend prefix.
func (s *ScopedTree) LookupSnapshot(owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, error) {
	snap, logits, matched, scope, _, err := s.LookupSnapshotTiered(owner, tokens)
	return snap, logits, matched, scope, err
}

// LookupSnapshotTiered is LookupSnapshot with truthful physical source-tier
// attribution. It searches every visible hot scope before consulting host DRAM.
func (s *ScopedTree) LookupSnapshotTiered(owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, SnapshotTier, error) {
	return s.LookupSnapshotTieredContext(context.Background(), owner, tokens)
}

// LookupSnapshotTieredContext is the cancellable scoped L1→L2→L3 lookup.
func (s *ScopedTree) LookupSnapshotTieredContext(ctx context.Context, owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, SnapshotTier, error) {
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	checks := []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet}
	bestHotScope, bestHotMatched := ScopeTenant, 0
	bestHostScope, bestHostMatched := ScopeTenant, 0
	bestRemoteScope, bestRemoteMatched := ScopeTenant, 0
	bestRemoteNS := ""
	var bestHot, bestHost, bestRemote *node
	for _, scope := range checks {
		if scope == ScopeAgent && strings.TrimSpace(owner.Agent) == "" {
			continue
		}
		ns, err := scopeNamespace(scope, owner)
		if err != nil {
			continue
		}
		n, _ := s.tree.LookupNS(ns, tokens)
		for candidate := n; candidate != nil; candidate = candidate.parent {
			if candidate.snapshot != nil && candidate.plen > bestHotMatched {
				bestHot, bestHotMatched, bestHotScope = candidate, candidate.plen, scope
			}
			if candidate.hostSnapshot != nil && candidate.plen > bestHostMatched {
				bestHost, bestHostMatched, bestHostScope = candidate, candidate.plen, scope
			}
			if candidate.remoteSnapshot != nil && candidate.plen > bestRemoteMatched {
				bestRemote, bestRemoteMatched, bestRemoteScope, bestRemoteNS = candidate, candidate.plen, scope, ns
			}
		}
		if n != nil {
			s.tree.Done(n)
		}
	}
	if bestHot != nil {
		snap, err := bestHot.snapshot.Clone()
		if err != nil {
			s.tree.l1Faults++
			return nil, nil, bestHotMatched, bestHotScope, SnapshotTierDeviceL1, err
		}
		s.tree.l1Hits++
		s.tree.l1HitTokens += bestHotMatched
		return snap, bestHot.Logits(), bestHotMatched, bestHotScope, SnapshotTierDeviceL1, nil
	}
	s.tree.l1Misses++
	if !s.tree.HostL2Enabled() && !s.tree.RemoteSnapshotEnabled() {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, nil
	}
	if bestHost != nil {
		snap, err := bestHost.hostSnapshot.Restore()
		if err != nil {
			s.tree.l2Faults++
			return nil, nil, bestHostMatched, bestHostScope, SnapshotTierHostL2, err
		}
		s.tree.l2Hits++
		s.tree.l2HitTokens += bestHostMatched
		s.tree.l2RestoreBytes += bestHost.hostSnapshot.TransferBytes()
		return snap, bestHost.Logits(), bestHostMatched, bestHostScope, SnapshotTierHostL2, nil
	}
	if s.tree.HostL2Enabled() {
		s.tree.l2Misses++
	}
	if bestRemote != nil && s.tree.RemoteSnapshotEnabled() {
		snap, found, err := s.tree.restoreSnapshotFromRemote(ctx, bestRemoteNS, bestRemote)
		if err != nil {
			return nil, nil, bestRemoteMatched, bestRemoteScope, SnapshotTierRemoteL3, err
		}
		if found {
			return snap, bestRemote.Logits(), bestRemoteMatched, bestRemoteScope, SnapshotTierRemoteL3, nil
		}
	}
	if s.tree.RemoteSnapshotEnabled() {
		s.tree.l3Misses++
	}
	return nil, nil, 0, ScopeTenant, SnapshotTierMiss, nil
}

// Lookup returns the longest reusable prefix visible to owner. Agent-private is
// checked first, then tenant-private, then explicitly promoted fleet state.
func (s *ScopedTree) Lookup(owner CacheIdentity, tokens []int) (*model.KVCache, []float32, int, ShareScope, error) {
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, nil, 0, ScopeTenant, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	checks := []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet}
	bestScope, bestMatched := ScopeTenant, 0
	var bestKV *model.KVCache
	var bestLogits []float32
	for _, scope := range checks {
		if scope == ScopeAgent && strings.TrimSpace(owner.Agent) == "" {
			continue
		}
		ns, err := scopeNamespace(scope, owner)
		if err != nil {
			continue
		}
		node, matched := s.tree.LookupNS(ns, tokens)
		if node != nil {
			if matched > bestMatched && node.KV() != nil {
				bestMatched, bestScope = matched, scope
				bestKV, bestLogits = cloneKV(node.KV()), node.Logits()
			}
			s.tree.Done(node)
		}
	}
	return bestKV, bestLogits, bestMatched, bestScope, nil
}

// Promote copies an exact private prefix into fleet visibility. It is a
// separate call so callers can perform policy review asynchronously before the
// broader cache can influence lookup latency.
func (s *ScopedTree) Promote(from ShareScope, owner CacheIdentity, tokens []int) error {
	if from == ScopeFleet {
		return ErrCacheScope
	}
	sourceNS, err := scopeNamespace(from, owner)
	if err != nil {
		return err
	}
	fleetNS, _ := scopeNamespace(ScopeFleet, owner)
	s.lock.Lock()
	defer s.lock.Unlock()
	source, matched := s.tree.LookupNS(sourceNS, tokens)
	if source == nil || matched != len(tokens) {
		if source != nil {
			s.tree.Done(source)
		}
		return ErrPrefixAbsent
	}
	kv, logits := source.KV(), source.Logits()
	s.tree.Done(source)
	boundary, fleetMatched := s.tree.LookupNS(fleetNS, tokens)
	leaf := s.tree.InsertCloneWithLogits(boundary, tokens[fleetMatched:], kv, logits)
	s.tree.Done(leaf)
	return nil
}

// RevokeFleet removes a promoted prefix without touching any private copy.
func (s *ScopedTree) RevokeFleet(tokens []int) int {
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.tree.EvictPrefixNS("shared/fleet", tokens)
}

// AdmitRegime stores a prefix at an explicit scope fenced by decode regime.
func (s *ScopedTree) AdmitRegime(regime Regime, scope ShareScope, owner CacheIdentity, tokens []int, kv *model.KVCache, logits []float32) error {
	if !regime.Complete() {
		return ErrRegimeIncomplete
	}
	if scope == ScopeFleet {
		return ErrCacheScope
	}
	baseNS, err := scopeNamespace(scope, owner)
	if err != nil {
		return err
	}
	ns := regimeNamespace(baseNS, regime)
	s.lock.Lock()
	defer s.lock.Unlock()
	boundary, matched := s.tree.LookupNS(ns, tokens)
	leaf := s.tree.InsertCloneWithLogits(boundary, tokens[matched:], kv, logits)
	s.tree.Done(leaf)
	return nil
}

// AdmitPrivateRegime stores a prefix at tenant scope fenced by decode regime.
func (s *ScopedTree) AdmitPrivateRegime(regime Regime, owner CacheIdentity, tokens []int, kv *model.KVCache, logits []float32) error {
	return s.AdmitRegime(regime, ScopeTenant, owner, tokens, kv, logits)
}

// LookupRegime returns the longest reusable prefix visible to owner matching the specified decode regime.
func (s *ScopedTree) LookupRegime(regime Regime, owner CacheIdentity, tokens []int) (*model.KVCache, []float32, int, ShareScope, error) {
	if !regime.Complete() {
		return nil, nil, 0, ScopeTenant, ErrRegimeIncomplete
	}
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, nil, 0, ScopeTenant, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	checks := []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet}
	bestScope, bestMatched := ScopeTenant, 0
	var bestKV *model.KVCache
	var bestLogits []float32
	for _, scope := range checks {
		if scope == ScopeAgent && strings.TrimSpace(owner.Agent) == "" {
			continue
		}
		baseNS, err := scopeNamespace(scope, owner)
		if err != nil {
			continue
		}
		ns := regimeNamespace(baseNS, regime)
		node, matched := s.tree.LookupNS(ns, tokens)
		if node != nil {
			if matched > bestMatched && node.KV() != nil {
				bestMatched, bestScope = matched, scope
				bestKV, bestLogits = cloneKV(node.KV()), node.Logits()
			}
			s.tree.Done(node)
		}
	}
	return bestKV, bestLogits, bestMatched, bestScope, nil
}

// PromoteRegime copies an exact private prefix matching regime into fleet visibility.
func (s *ScopedTree) PromoteRegime(regime Regime, from ShareScope, owner CacheIdentity, tokens []int) error {
	if !regime.Complete() {
		return ErrRegimeIncomplete
	}
	if from == ScopeFleet {
		return ErrCacheScope
	}
	baseSrcNS, err := scopeNamespace(from, owner)
	if err != nil {
		return err
	}
	sourceNS := regimeNamespace(baseSrcNS, regime)
	baseFleetNS, _ := scopeNamespace(ScopeFleet, owner)
	fleetNS := regimeNamespace(baseFleetNS, regime)

	s.lock.Lock()
	defer s.lock.Unlock()
	source, matched := s.tree.LookupNS(sourceNS, tokens)
	if source == nil || matched != len(tokens) {
		if source != nil {
			s.tree.Done(source)
		}
		return ErrPrefixAbsent
	}
	kv, logits := source.KV(), source.Logits()
	s.tree.Done(source)
	boundary, fleetMatched := s.tree.LookupNS(fleetNS, tokens)
	leaf := s.tree.InsertCloneWithLogits(boundary, tokens[fleetMatched:], kv, logits)
	s.tree.Done(leaf)
	return nil
}

// RevokeFleetRegime removes a promoted prefix under regime without touching any private copy.
func (s *ScopedTree) RevokeFleetRegime(regime Regime, tokens []int) int {
	if !regime.Complete() {
		return 0
	}
	fleetNS := regimeNamespace("shared/fleet", regime)
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.tree.EvictPrefixNS(fleetNS, tokens)
}

// AdmitPrivateSnapshotRegime stores a complete backend prefix snapshot at tenant scope fenced by regime.
func (s *ScopedTree) AdmitPrivateSnapshotRegime(regime Regime, owner CacheIdentity, tokens []int, snap *model.PrefixSnapshot, logits []float32) error {
	if !regime.Complete() {
		return ErrRegimeIncomplete
	}
	baseNS, err := scopeNamespace(ScopeTenant, owner)
	if err != nil {
		return err
	}
	ns := regimeNamespace(baseNS, regime)
	s.lock.Lock()
	defer s.lock.Unlock()
	boundary, matched := s.tree.LookupNS(ns, tokens)
	leaf, err := s.tree.InsertSnapshot(boundary, tokens[matched:], snap, logits)
	if leaf != nil {
		s.tree.Done(leaf)
	}
	return err
}

// LookupSnapshotRegime returns the longest visible independently owned backend prefix under regime.
func (s *ScopedTree) LookupSnapshotRegime(regime Regime, owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, error) {
	snap, logits, matched, scope, _, err := s.LookupSnapshotTieredRegime(regime, owner, tokens)
	return snap, logits, matched, scope, err
}

// LookupSnapshotTieredRegime is LookupSnapshotRegime with truthful physical source-tier attribution.
func (s *ScopedTree) LookupSnapshotTieredRegime(regime Regime, owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, SnapshotTier, error) {
	return s.LookupSnapshotTieredContextRegime(context.Background(), regime, owner, tokens)
}

// LookupSnapshotTieredContextRegime is the cancellable scoped L1->L2->L3 lookup under regime.
func (s *ScopedTree) LookupSnapshotTieredContextRegime(ctx context.Context, regime Regime, owner CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, ShareScope, SnapshotTier, error) {
	if !regime.Complete() {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, ErrRegimeIncomplete
	}
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	checks := []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet}
	bestHotScope, bestHotMatched := ScopeTenant, 0
	bestHostScope, bestHostMatched := ScopeTenant, 0
	bestRemoteScope, bestRemoteMatched := ScopeTenant, 0
	bestRemoteNS := ""
	var bestHot, bestHost, bestRemote *node
	for _, scope := range checks {
		if scope == ScopeAgent && strings.TrimSpace(owner.Agent) == "" {
			continue
		}
		baseNS, err := scopeNamespace(scope, owner)
		if err != nil {
			continue
		}
		ns := regimeNamespace(baseNS, regime)
		n, _ := s.tree.LookupNS(ns, tokens)
		for candidate := n; candidate != nil; candidate = candidate.parent {
			if candidate.snapshot != nil && candidate.plen > bestHotMatched {
				bestHot, bestHotMatched, bestHotScope = candidate, candidate.plen, scope
			}
			if candidate.hostSnapshot != nil && candidate.plen > bestHostMatched {
				bestHost, bestHostMatched, bestHostScope = candidate, candidate.plen, scope
			}
			if candidate.remoteSnapshot != nil && candidate.plen > bestRemoteMatched {
				bestRemote, bestRemoteMatched, bestRemoteScope, bestRemoteNS = candidate, candidate.plen, scope, ns
			}
		}
		if n != nil {
			s.tree.Done(n)
		}
	}
	if bestHot != nil {
		snap, err := bestHot.snapshot.Clone()
		if err != nil {
			s.tree.l1Faults++
			return nil, nil, bestHotMatched, bestHotScope, SnapshotTierDeviceL1, err
		}
		s.tree.l1Hits++
		s.tree.l1HitTokens += bestHotMatched
		return snap, bestHot.Logits(), bestHotMatched, bestHotScope, SnapshotTierDeviceL1, nil
	}
	s.tree.l1Misses++
	if !s.tree.HostL2Enabled() && !s.tree.RemoteSnapshotEnabled() {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, nil
	}
	if bestHost != nil {
		snap, err := bestHost.hostSnapshot.Restore()
		if err != nil {
			s.tree.l2Faults++
			return nil, nil, bestHostMatched, bestHostScope, SnapshotTierHostL2, err
		}
		s.tree.l2Hits++
		s.tree.l2HitTokens += bestHostMatched
		s.tree.l2RestoreBytes += bestHost.hostSnapshot.TransferBytes()
		return snap, bestHost.Logits(), bestHostMatched, bestHostScope, SnapshotTierHostL2, nil
	}
	if s.tree.HostL2Enabled() {
		s.tree.l2Misses++
	}
	if bestRemote != nil && s.tree.RemoteSnapshotEnabled() {
		snap, found, err := s.tree.restoreSnapshotFromRemote(ctx, bestRemoteNS, bestRemote)
		if err != nil {
			return nil, nil, bestRemoteMatched, bestRemoteScope, SnapshotTierRemoteL3, err
		}
		if found {
			return snap, bestRemote.Logits(), bestRemoteMatched, bestRemoteScope, SnapshotTierRemoteL3, nil
		}
	}
	if s.tree.RemoteSnapshotEnabled() {
		s.tree.l3Misses++
	}
	return nil, nil, 0, ScopeTenant, SnapshotTierMiss, nil
}

func cloneKV(kv *model.KVCache) *model.KVCache {
	if kv == nil {
		return nil
	}
	return kv.Clone()
}

// ContinuationSoftPriority is the fixed soft-retention priority a scoped
// continuation applies. It is strictly below 60 so TierFromRetentionPriority
// maps it to Tier2IdleSubagent — never Tier0/Tier1 — which is what makes a
// continuation SOFT: it survives idle reclamation but is always reclaimable
// under budget pressure. It is never a pin.
const ContinuationSoftPriority = 45

// ContinuationSoftTTLTicks is the finite logical-tick window a scoped
// continuation retains its node for. It is measured in the tree's logical
// access clock, never wall-clock, so the reclaim verdict is deterministic and
// replayable. It must stay FINITE and positive: a RetainForever/0 TTL would be
// a hard pin in disguise.
const ContinuationSoftTTLTicks int64 = 64

var (
	ErrContinuationUnsupported = errors.New("radixkv: continuation retention is not supported here")
	ErrContinuationNotAdmitted = errors.New("radixkv: prefix was not privately admitted")
	ErrContinuationNodeGone    = errors.New("radixkv: continuation node is no longer resident")
	ErrContinuationGeneration  = errors.New("radixkv: continuation generation was replaced")
	ErrContinuationNotOwner    = errors.New("radixkv: continuation handle is not owner-scoped")
)

// ContinuationState is the typed status of a ContinuationHandle.
type ContinuationState uint8

const (
	// ContinuationDormant: created but not yet activated (no live policy applied).
	ContinuationDormant ContinuationState = iota
	// ContinuationActive: this handle currently holds the node's soft policy.
	ContinuationActive
	// ContinuationStale: the node is resident but its record generation was replaced.
	ContinuationStale
	// ContinuationEvicted: the node is no longer attached to the tree.
	ContinuationEvicted
	// ContinuationUnsupported: the handle is nil/foreign or the payload is gone.
	ContinuationUnsupported
)

func (s ContinuationState) String() string {
	switch s {
	case ContinuationDormant:
		return "dormant"
	case ContinuationActive:
		return "active"
	case ContinuationStale:
		return "stale"
	case ContinuationEvicted:
		return "evicted"
	default:
		return "unsupported"
	}
}

// ContinuationHandle is an owner-scoped reference to a resident reusable KV
// payload. It is DORMANT until Activate; it carries only a SOFT, finite-TTL
// retention policy and is never a pin.
//
// INVARIANTS:
//   - It holds NO tree lease. BeginPrivateContinuation calls Done immediately
//     after recording the node pointer, so holding a handle can never steal
//     eviction budget or ref-count-pin a prefix (the ticket forbids a ref pin).
//   - It is generation-bound only when the bound node owns a snapshot RECORD: the
//     generation is then that record's own incarnation (`recordGen`), and a
//     replaced record refuses typed (ErrContinuationGeneration). For a pure-KV
//     node (no record) the monotonic `gen` stamp (a fresh nextRecordSeq) is NOT
//     itself re-validated; the handle's identity is instead enforced by the
//     node's attachment (nodeAttached) plus its exact TOKEN identity: plen + a
//     mint-time edge-label copy compared against the node's current key. A radix
//     split re-parents the node and rewrites its edge label so the node now
//     spells a DIFFERENT prefix, and plen + the key copy are the read-only split
//     witnesses that detect this without mutating radixkv.go. A true split, a
//     longer path attached below the node (no children), an extension, or an
//     eviction refuses typed even though the pure-KV `gen` stamp is not itself
//     re-validated, so a stale continuation can never serve a payload it did not
//     admit.
//   - It is owner-scoped: Activate/Release reject a handle presented by a different
//     tenant/agent, so one owner cannot retain or release another's prefix.
type ContinuationHandle struct {
	tree   *Tree
	node   *node
	owner  CacheIdentity
	gen    uint64 // generation stamped at creation (recordGen when owned, else recordSeq)
	plen   int    // bound node's prefix length at mint time (split/replace witness)
	key    []int  // bound node's edge label at mint time (true-split witness)
	ttl    int64  // finite positive logical-tick retention window
	active bool
	// activation records the EXACT retention request this handle wrote when it
	// activated. Release compares it against the node's current retention: if a
	// later policy decision replaced it, release leaves that decision untouched.
	activation RetentionRequest
}

// Owner returns the tenant/agent this handle is scoped to.
func (h *ContinuationHandle) Owner() CacheIdentity {
	if h == nil {
		return CacheIdentity{}
	}
	return h.owner
}

// Generation returns the monotonic generation stamped at creation.
func (h *ContinuationHandle) Generation() uint64 {
	if h == nil {
		return 0
	}
	return h.gen
}

// Active reports whether this handle currently holds a live soft-retention
// policy on its node.
func (h *ContinuationHandle) Active() bool {
	return h != nil && h.active
}

// Dormant reports whether the handle was created but has not (yet) been
// activated.
func (h *ContinuationHandle) Dormant() bool {
	return h != nil && !h.active
}

// nextRecordSeq advances the tree's monotonic record-incarnation sequence and
// returns the fresh generation. Unlike mintRecord it does NOT create a snapshot
// record or mutate any node — it is only a generation stamp.
func (t *Tree) nextRecordSeq() uint64 {
	t.recordSeq++
	return t.recordSeq
}

// nodeHasReusablePayload reports whether n owns a complete reusable payload in
// any physical tier: a device K/V, or a complete snapshot resident in hot L1,
// host DRAM L2, or the remote L3 reference tier. It mirrors the tree's own
// record-liveness predicate so a continuation is admitted whenever a reusable
// payload is genuinely resident, not only when it is hot.
func nodeHasReusablePayload(n *node) bool {
	if n == nil {
		return false
	}
	return n.kv != nil || n.snapshot != nil || n.hostSnapshot != nil || n.remoteSnapshot != nil
}

// continuationSuperseded reports whether the handle's bound node no longer
// terminates the exact token path it was minted for. Two read-only witnesses
// cover the two ways that identity dies WITHOUT the pointer being detached:
//
//   - EDGE REWRITE (a true radix split): split() re-parents the node and replaces
//     its key with the suffix beyond the split point, so the node now spells a
//     DIFFERENT prefix. plen alone does not catch this — the full path length is
//     unchanged — so the mint-time key copy is compared.
//   - EXTENSION (a longer path attached below): the node was a terminal payload
//     but now has children, so its exact-terminal continuation was replaced by a
//     longer cached one.
//
// It never mutates the tree, creates a node, or allocates a payload: a stale
// handle is always a typed safe no-op.
func continuationSuperseded(h *ContinuationHandle) bool {
	if h == nil || h.node == nil {
		return true
	}
	if len(h.node.children) != 0 {
		return true
	}
	if len(h.node.key) != len(h.key) {
		return true
	}
	for i, tok := range h.key {
		if h.node.key[i] != tok {
			return true
		}
	}
	return false
}

// BeginPrivateContinuation mints a DORMANT, owner-scoped continuation handle for
// a prefix the caller has ALREADY privately admitted via AdmitPrivate /
// AdmitPrivateSnapshot.
//
// Generation binding is exact for a node owning a snapshot RECORD: the handle
// carries that record's incarnation (recordGen), and a replaced record refuses
// typed (ErrContinuationGeneration). A pure-KV node (no record) has no such
// incarnation, so its monotonic `gen` stamp is NOT re-validated; identity is
// enforced by the node's attachment plus its exact token identity (plen + a
// mint-time edge-label copy + terminal/no-children), so a split, extension, or
// eviction refuses typed.
//
// It requires the EXACT full token path to resolve to a node boundary carrying a
// reusable payload in any physical tier (device KV, or a complete snapshot in
// hot L1, host L2, or remote L3); a rejected admission or a foreign owner has no
// such node, so the node lookup IS the admission witness (we never trust a nil
// AdmitPrivate error alone). LookupNS leases the boundary (refs++); we copy what
// we need and call Done on EVERY path before returning, so the handle is not a
// ref-count pin and steals no eviction budget.
func (s *ScopedTree) BeginPrivateContinuation(owner CacheIdentity, tokens []int) (*ContinuationHandle, error) {
	if s == nil || s.tree == nil {
		return nil, ErrContinuationUnsupported
	}
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, ErrCacheIdentity
	}
	// The empty token path resolves to the namespace ROOT, which is not a privately
	// admitted prefix; refuse it rather than mint a handle on a structural root.
	if len(tokens) == 0 {
		return nil, ErrContinuationNotAdmitted
	}
	ns, err := scopeNamespace(ScopeTenant, owner)
	if err != nil {
		return nil, err
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	node, matched := s.tree.LookupNS(ns, tokens)
	if node == nil || matched != len(tokens) {
		if node != nil {
			s.tree.Done(node)
		}
		return nil, ErrContinuationNotAdmitted
	}
	// The node is leased by LookupNS. Release immediately: the handle must not be
	// a ref-count pin, yet we keep the *node pointer to re-validate on Activate.
	resident := nodeHasReusablePayload(node)
	// Derive the generation from the node's OWN record incarnation when it owns a
	// complete snapshot record. A fresh nextRecordSeq() here would collide with the
	// recordGen InsertSnapshot already minted, making every snapshot-backed handle
	// instantly stale (the activation guard compares recordGen != gen). For a
	// pure-KV node (hasRecord=false) a fresh monotonic sequence is the only
	// incarnation available.
	gen := node.recordGen
	if !node.hasRecord {
		gen = s.tree.nextRecordSeq()
	}
	plen := node.plen
	key := append([]int(nil), node.key...)
	s.tree.Done(node)
	if !resident {
		return nil, ErrContinuationUnsupported
	}
	return &ContinuationHandle{
		tree:  s.tree,
		node:  node,
		owner: owner,
		gen:   gen,
		plen:  plen,
		key:   key,
		ttl:   ContinuationSoftTTLTicks,
	}, nil
}

// ActivateContinuation applies the handle's soft retention exactly ONCE. It is
// idempotent: a second activation is a no-op that neither refreshes the
// generation nor extends the TTL window, so a caller cannot use repeated
// activation to pin a prefix indefinitely.
//
// It fails closed on a nil/foreign handle, an owner mismatch, a detached
// (evicted/split) node, a replaced generation, or a gone payload. The applied
// request is Priority ContinuationSoftPriority (<60 → Tier2IdleSubagent, always
// reclaimable) with a FINITE TTL; it is a soft policy, never a pin.
func (s *ScopedTree) ActivateContinuation(h *ContinuationHandle) error {
	if s == nil || s.tree == nil || h == nil || h.tree != s.tree {
		return ErrContinuationUnsupported
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	// Owner-scope guard: the handle must carry a complete scope identity. A
	// handle whose owner is empty is not owner-scoped and must not retain.
	if strings.TrimSpace(h.owner.Tenant) == "" {
		return ErrContinuationNotOwner
	}
	if !s.tree.nodeAttached(h.node) {
		return ErrContinuationNodeGone
	}
	if h.node.hasRecord && h.node.recordGen != h.gen {
		return ErrContinuationGeneration
	}
	// Split/replace witness: a radix split re-parents this node and rewrites its
	// key/plen so the pointer now spells a DIFFERENT prefix. A changed plen means
	// the handle's token identity is gone even though the pointer is still
	// attached; refuse rather than apply policy to the wrong node.
	if h.node.plen != h.plen {
		return ErrContinuationGeneration
	}
	// Supersession witness: a longer cached path was attached below this node, so
	// it no longer terminates the prefix the handle admitted. Refuse rather than
	// apply policy to a node whose exact-terminal identity was replaced.
	if continuationSuperseded(h) {
		return ErrContinuationGeneration
	}
	if !nodeHasReusablePayload(h.node) {
		return ErrContinuationUnsupported
	}
	if h.active {
		return nil
	}
	req := RetentionRequest{
		Priority: ContinuationSoftPriority,
		TTL:      h.ttl,
		Admitted: int64(s.tree.clock),
	}
	if err := s.tree.SetNodeRetention(h.node, req); err != nil {
		return err
	}
	h.activation = req
	h.active = true
	return nil
}

// ReleaseContinuation clears ONLY this handle's own policy. If a later policy
// decision has replaced the node's retention (different priority, or a different
// admission tick), release leaves that newer decision untouched — releasing an
// old handle must never erase a later decision. When the handle still owns the
// current policy it clears the node's retention entirely (no replacement
// descriptor, no RetainForever window) so release has an observable effect. A
// detached node returns ErrContinuationNodeGone as a typed, safe no-op; a
// replaced-generation handle returns ErrContinuationGeneration and clears no
// policy. A node is never recreated.
func (s *ScopedTree) ReleaseContinuation(h *ContinuationHandle) error {
	if s == nil || s.tree == nil || h == nil || h.tree != s.tree {
		return ErrContinuationUnsupported
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	if !h.active {
		return nil
	}
	if !s.tree.nodeAttached(h.node) {
		return ErrContinuationNodeGone
	}
	// Replaced-generation witness, same as Activate: the node still carries a
	// record but its record generation was replaced, so this handle no longer
	// names the recollection it admitted. Refuse rather than clear the policy of
	// a newer generation.
	if h.node.hasRecord && h.node.recordGen != h.gen {
		return ErrContinuationGeneration
	}
	// Split/replace witness, same as Activate: the pointer may still be attached
	// but now spells a different prefix. Never clear a policy on the wrong node.
	if h.node.plen != h.plen {
		return ErrContinuationGeneration
	}
	// Supersession witness, same as Activate: a longer path attached below the node
	// means this handle's exact-terminal identity was replaced. Release must be a
	// typed safe no-op (it must never clear a policy it no longer owns).
	if continuationSuperseded(h) {
		return ErrContinuationGeneration
	}
	if cur, ok := s.tree.NodeRetention(h.node); ok {
		// Only clear when the node still carries the EXACT request this handle
		// wrote; otherwise a later (stronger or refreshed) decision wins.
		if cur.Priority == h.activation.Priority && cur.Admitted == h.activation.Admitted {
			// Clear the soft policy outright (retention nil, tier unset) rather than
			// stamping a replacement descriptor: a release must leave NO retention
			// decision of its own, and must never write a RetainForever/0 window.
			s.tree.UnpinNode(h.node)
		}
	}
	h.active = false
	return nil
}

// ContinuationStatus classifies a handle without mutating it, for typed stale/
// evicted reporting that mirrors the error-returning operations.
func (s *ScopedTree) ContinuationStatus(h *ContinuationHandle) (ContinuationState, error) {
	if s == nil || s.tree == nil || h == nil || h.tree != s.tree {
		return ContinuationUnsupported, ErrContinuationUnsupported
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	if !s.tree.nodeAttached(h.node) {
		return ContinuationEvicted, ErrContinuationNodeGone
	}
	if h.node.hasRecord && h.node.recordGen != h.gen {
		return ContinuationStale, ErrContinuationGeneration
	}
	if h.node.plen != h.plen {
		return ContinuationStale, ErrContinuationGeneration
	}
	if continuationSuperseded(h) {
		return ContinuationStale, ErrContinuationGeneration
	}
	if !nodeHasReusablePayload(h.node) {
		return ContinuationUnsupported, ErrContinuationUnsupported
	}
	if h.active {
		return ContinuationActive, nil
	}
	return ContinuationDormant, nil
}
