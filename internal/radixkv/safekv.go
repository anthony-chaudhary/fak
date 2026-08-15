package radixkv

import (
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
	leaf := s.tree.InsertWithLogits(boundary, tokens[matched:], cloneKV(kv), logits)
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
	if strings.TrimSpace(owner.Tenant) == "" {
		return nil, nil, 0, ScopeTenant, SnapshotTierMiss, ErrCacheIdentity
	}
	s.lock.Lock()
	defer s.lock.Unlock()
	checks := []ShareScope{ScopeAgent, ScopeTenant, ScopeFleet}
	bestHotScope, bestHotMatched := ScopeTenant, 0
	bestHostScope, bestHostMatched := ScopeTenant, 0
	var bestHot, bestHost *node
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
	if !s.tree.HostL2Enabled() {
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
	s.tree.l2Misses++
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
			if matched > bestMatched {
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
	kv, logits := cloneKV(source.KV()), source.Logits()
	s.tree.Done(source)
	boundary, fleetMatched := s.tree.LookupNS(fleetNS, tokens)
	leaf := s.tree.InsertWithLogits(boundary, tokens[fleetMatched:], kv, logits)
	s.tree.Done(leaf)
	return nil
}

// RevokeFleet removes a promoted prefix without touching any private copy.
func (s *ScopedTree) RevokeFleet(tokens []int) int {
	s.lock.Lock()
	defer s.lock.Unlock()
	return s.tree.EvictPrefixNS("shared/fleet", tokens)
}

func cloneKV(kv *model.KVCache) *model.KVCache {
	if kv == nil {
		return nil
	}
	return kv.Clone()
}
