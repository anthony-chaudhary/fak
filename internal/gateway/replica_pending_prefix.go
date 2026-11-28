// Adapted from sgl-project/sglang sgl-router/src/state/kv_events/pending.rs:1-60
// Reimplemented cleanly in Go to compose with CacheAwarePolicy and PrefixResidencyIndex.

package gateway

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

const (
	// DefaultPendingPrefixMaxRecords bounds retained pending prefix stakes when
	// MaxRecords is left zero.
	DefaultPendingPrefixMaxRecords = 1 << 16

	// MaxPendingPrefixMaxRecords caps MaxRecords at SGLang's 1<<20 limit so
	// unbounded configurations cannot exhaust gateway memory.
	MaxPendingPrefixMaxRecords = 1 << 20
)

// PendingPrefixConfig configures PendingPrefixIndex. Zero TTL disables pending
// prefix tracking by default.
type PendingPrefixConfig struct {
	// TTL is how long a cold-prefix stake stays active after routing.
	// Zero disables pending prefix records (default off).
	TTL time.Duration

	// MaxFollowers caps how many subsequent requests may follow an active stake
	// after the initial request places it.
	MaxFollowers int

	// ConsultBelow restricts pending stake lookup to stake owners whose current
	// load is strictly below ConsultBelow (load < ConsultBelow). Zero leaves
	// pending consultation unconstrained by owner load.
	ConsultBelow int

	// MaxRecords bounds total retained pending stakes (FIFO/expiry evicted).
	// Zero defaults to DefaultPendingPrefixMaxRecords; values above
	// MaxPendingPrefixMaxRecords are clamped.
	MaxRecords int

	// Now supplies the clock source (defaults to time.Now).
	Now func() time.Time
}

// DefaultPendingPrefixConfig returns the standing safe defaults with pending
// tracking disabled (TTL=0).
func DefaultPendingPrefixConfig() PendingPrefixConfig {
	return PendingPrefixConfig{
		TTL:          0,
		MaxFollowers: 8,
		ConsultBelow: 16,
		MaxRecords:   DefaultPendingPrefixMaxRecords,
	}
}

type pendingStake struct {
	id        uint64
	worker    string
	prefix    []string
	key       string
	expiresAt time.Time
	followers int
}

type pendingTrieNode struct {
	parent   *pendingTrieNode
	segment  string
	children map[string]*pendingTrieNode
	stakes   map[uint64]*pendingStake
}

func newPendingTrieNode(parent *pendingTrieNode, segment string) *pendingTrieNode {
	return &pendingTrieNode{
		parent:   parent,
		segment:  segment,
		children: make(map[string]*pendingTrieNode),
		stakes:   make(map[uint64]*pendingStake),
	}
}

// PendingPrefixIndex tracks short-lived cold-prefix stakes per worker so a cold
// burst sharing a prefix converges onto one replica instead of scattering across
// idle workers before confirmed KV events arrive. Safe for concurrent use.
type PendingPrefixIndex struct {
	mu           sync.Mutex
	ttl          time.Duration
	maxFollowers int
	consultBelow int
	maxRecords   int
	now          func() time.Time

	nextID uint64
	root   *pendingTrieNode
	byKey  map[string]*pendingStake
	order  []*pendingStake
}

// NewPendingPrefixIndex validates cfg and returns a PendingPrefixIndex.
// Negative config parameters fail closed with an error; TTL=0 builds a valid
// disabled index.
func NewPendingPrefixIndex(cfg PendingPrefixConfig) (*PendingPrefixIndex, error) {
	if cfg.TTL < 0 {
		return nil, fmt.Errorf("gateway: pending prefix TTL must be >= 0, got %s", cfg.TTL)
	}
	if cfg.MaxFollowers < 0 {
		return nil, fmt.Errorf("gateway: pending prefix MaxFollowers must be >= 0, got %d", cfg.MaxFollowers)
	}
	if cfg.ConsultBelow < 0 {
		return nil, fmt.Errorf("gateway: pending prefix ConsultBelow must be >= 0, got %d", cfg.ConsultBelow)
	}
	if cfg.MaxRecords < 0 {
		return nil, fmt.Errorf("gateway: pending prefix MaxRecords must be >= 0, got %d", cfg.MaxRecords)
	}
	maxRec := cfg.MaxRecords
	if maxRec == 0 {
		maxRec = DefaultPendingPrefixMaxRecords
	}
	if maxRec > MaxPendingPrefixMaxRecords {
		maxRec = MaxPendingPrefixMaxRecords
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &PendingPrefixIndex{
		ttl:          cfg.TTL,
		maxFollowers: cfg.MaxFollowers,
		consultBelow: cfg.ConsultBelow,
		maxRecords:   maxRec,
		now:          now,
		root:         newPendingTrieNode(nil, ""),
		byKey:        make(map[string]*pendingStake),
	}, nil
}

// Enabled reports whether pending prefix tracking is active (TTL > 0).
func (p *PendingPrefixIndex) Enabled() bool {
	return p != nil && p.ttl > 0
}

// Len returns the number of currently unexpired pending stakes.
func (p *PendingPrefixIndex) Len() int {
	if !p.Enabled() {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.purgeExpiredLocked(p.clock())
	return len(p.byKey)
}

func (p *PendingPrefixIndex) clock() time.Time {
	if p == nil || p.now == nil {
		return time.Now()
	}
	return p.now()
}

func normalizePendingPrefix(prefix []string) []string {
	if len(prefix) == 0 {
		return nil
	}
	out := make([]string, 0, len(prefix))
	for _, seg := range prefix {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			break
		}
		out = append(out, seg)
	}
	return out
}

func pendingStakeKey(worker string, prefix []string) string {
	return worker + "\x1e" + residencyKey(prefix)
}

// Record places or refreshes a pending stake of prefix against worker for TTL.
// It is a no-op when pending tracking is disabled (TTL == 0) or inputs are empty.
func (p *PendingPrefixIndex) Record(worker string, prefix []string) {
	if !p.Enabled() {
		return
	}
	worker = strings.TrimSpace(worker)
	norm := normalizePendingPrefix(prefix)
	if worker == "" || len(norm) == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.recordLocked(worker, norm, p.clock())
}

func (p *PendingPrefixIndex) recordLocked(worker string, prefix []string, now time.Time) {
	p.purgeExpiredLocked(now)
	key := pendingStakeKey(worker, prefix)
	if existing, ok := p.byKey[key]; ok {
		existing.expiresAt = now.Add(p.ttl)
		existing.followers = 0
		return
	}
	for len(p.byKey) >= p.maxRecords && len(p.order) > 0 {
		oldest := p.order[0]
		p.order = p.order[1:]
		p.removeStakeLocked(oldest)
	}
	p.nextID++
	stake := &pendingStake{
		id:        p.nextID,
		worker:    worker,
		prefix:    append([]string(nil), prefix...),
		key:       key,
		expiresAt: now.Add(p.ttl),
	}
	p.byKey[key] = stake
	p.order = append(p.order, stake)

	cur := p.root
	for _, seg := range stake.prefix {
		child := cur.children[seg]
		if child == nil {
			child = newPendingTrieNode(cur, seg)
			cur.children[seg] = child
		}
		child.stakes[stake.id] = stake
		cur = child
	}
}

func (p *PendingPrefixIndex) purgeExpiredLocked(now time.Time) {
	if len(p.order) == 0 {
		return
	}
	kept := p.order[:0]
	for _, s := range p.order {
		if _, active := p.byKey[s.key]; !active {
			continue
		}
		if !now.Before(s.expiresAt) {
			p.removeStakeLocked(s)
			continue
		}
		kept = append(kept, s)
	}
	p.order = kept
}

func (p *PendingPrefixIndex) removeStakeLocked(stake *pendingStake) {
	if stake == nil {
		return
	}
	if cur, ok := p.byKey[stake.key]; !ok || cur != stake {
		return
	}
	delete(p.byKey, stake.key)

	cur := p.root
	var leaf *pendingTrieNode
	for _, seg := range stake.prefix {
		next := cur.children[seg]
		if next == nil {
			break
		}
		delete(next.stakes, stake.id)
		cur = next
		leaf = next
	}
	for n := leaf; n != nil && n != p.root; {
		if len(n.stakes) > 0 || len(n.children) > 0 {
			break
		}
		parent := n.parent
		if parent != nil {
			delete(parent.children, n.segment)
		}
		n = parent
	}
}

func (p *PendingPrefixIndex) stakeConsultableLocked(s *pendingStake, worker string, workerLoad int, now time.Time) bool {
	if s == nil || s.worker != worker {
		return false
	}
	if !now.Before(s.expiresAt) {
		return false
	}
	if s.followers >= p.maxFollowers {
		return false
	}
	if p.consultBelow > 0 && workerLoad >= p.consultBelow {
		return false
	}
	return true
}

// PendingOverlap returns the longest pending prefix match length held by worker
// for prefix, subject to TTL freshness, maxFollowers, and workerLoad < ConsultBelow.
// This lookup is read-only and does not consume a follower slot.
func (p *PendingPrefixIndex) PendingOverlap(worker string, prefix []string, workerLoad int) int {
	if !p.Enabled() {
		return 0
	}
	worker = strings.TrimSpace(worker)
	norm := normalizePendingPrefix(prefix)
	if worker == "" || len(norm) == 0 {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.clock()
	p.purgeExpiredLocked(now)
	ov, _ := p.bestStakeLocked(worker, norm, workerLoad, now)
	return ov
}

func (p *PendingPrefixIndex) bestStakeLocked(worker string, norm []string, workerLoad int, now time.Time) (int, *pendingStake) {
	cur := p.root
	bestOverlap := 0
	var bestStake *pendingStake
	for depth, seg := range norm {
		next := cur.children[seg]
		if next == nil {
			break
		}
		cur = next
		for _, stake := range cur.stakes {
			if p.stakeConsultableLocked(stake, worker, workerLoad, now) {
				bestOverlap = depth + 1
				bestStake = stake
			}
		}
	}
	return bestOverlap, bestStake
}

// CombinedOverlap returns max(confirmedOverlap, pendingOverlap) for worker on prefix.
func (p *PendingPrefixIndex) CombinedOverlap(confirmed ResidencyView, worker string, prefix []string, workerLoad int) int {
	conf := 0
	if confirmed != nil {
		conf = confirmed.Overlap(worker, prefix)
	}
	pend := p.PendingOverlap(worker, prefix, workerLoad)
	if pend > conf {
		return pend
	}
	return conf
}

func (p *PendingPrefixIndex) claimOrStakeLocked(worker string, norm []string, workerLoad int, now time.Time) {
	if !p.Enabled() || worker == "" || len(norm) == 0 {
		return
	}
	p.purgeExpiredLocked(now)
	if ov, stake := p.bestStakeLocked(worker, norm, workerLoad, now); ov > 0 && stake != nil {
		stake.followers++
		return
	}
	// Do not overwrite an active saturated stake on the same exact (worker, prefix)
	// before its TTL expires; only insert when no unexpired stake exists there.
	key := pendingStakeKey(worker, norm)
	if existing, ok := p.byKey[key]; ok && now.Before(existing.expiresAt) {
		return
	}
	p.recordLocked(worker, norm, now)
}

// PendingPrefixPolicy integrates PendingPrefixIndex with fak's existing
// CacheAwarePolicy / PrefixResidencyIndex path so cold same-prefix bursts
// converge onto a stake owner up to MaxFollowers while owner load < ConsultBelow,
// and fall back cleanly to confirmed residency + load balancing past SkewThreshold.
type PendingPrefixPolicy struct {
	mu        sync.Mutex
	confirmed ResidencyView
	skew      SkewThreshold
	pending   *PendingPrefixIndex
}

// NewPendingPrefixPolicy wraps an existing CacheAwarePolicy's residency index and
// skew guard with pending prefix tracking. When base is nil, a fresh
// PrefixResidencyIndex and DefaultSkewThreshold are used.
func NewPendingPrefixPolicy(base *CacheAwarePolicy, pending *PendingPrefixIndex) *PendingPrefixPolicy {
	var confirmed ResidencyView
	skew := DefaultSkewThreshold()
	if base != nil {
		confirmed = base.Index()
		skew = base.skew
	} else {
		confirmed = NewPrefixResidencyIndex(64)
	}
	return NewPendingPrefixPolicyWithView(confirmed, skew, pending)
}

// NewPendingPrefixPolicyWithView builds a PendingPrefixPolicy over any
// ResidencyView (for example PrefixResidencyIndex or PrefixResidencyScorer).
func NewPendingPrefixPolicyWithView(confirmed ResidencyView, skew SkewThreshold, pending *PendingPrefixIndex) *PendingPrefixPolicy {
	def := DefaultSkewThreshold()
	if skew.AbsLoad < 1 {
		skew.AbsLoad = def.AbsLoad
	}
	if skew.RelLoad < 1 {
		skew.RelLoad = def.RelLoad
	}
	return &PendingPrefixPolicy{
		confirmed: confirmed,
		skew:      skew,
		pending:   pending,
	}
}

// Pick implements PickPolicy over candidates using max(confirmed, pending) overlap
// and the dual abs+rel SkewThreshold balance fallback.
func (p *PendingPrefixPolicy) Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool) {
	if p == nil || len(candidates) == 0 {
		return PlannerReplica{}, false
	}
	if load == nil {
		load = func(string) int { return 0 }
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	norm := normalizePendingPrefix(prefix)
	key := replicaRendezvousKey(norm)

	// Lock pending once for consistent overlap + claim across the decision.
	var now time.Time
	if p.pending != nil && p.pending.Enabled() {
		p.pending.mu.Lock()
		defer p.pending.mu.Unlock()
		now = p.pending.clock()
		p.pending.purgeExpiredLocked(now)
	}

	type scored struct {
		replica PlannerReplica
		load    int
		overlap int
		score   float64
		hrw     uint64
	}
	items := make([]scored, 0, len(candidates))
	maxL, minL := 0, 0
	for _, c := range candidates {
		if c.Name == "" {
			continue
		}
		l := load(c.Name)
		if l < 0 {
			l = 0
		}
		if len(items) == 0 || l > maxL {
			maxL = l
		}
		if len(items) == 0 || l < minL {
			minL = l
		}
		ov := 0
		if p.confirmed != nil && len(prefix) > 0 {
			ov = p.confirmed.Overlap(c.Name, prefix)
		}
		if p.pending != nil && p.pending.Enabled() && len(norm) > 0 {
			if pov, _ := p.pending.bestStakeLocked(c.Name, norm, l, now); pov > ov {
				ov = pov
			}
		}
		items = append(items, scored{
			replica: c,
			load:    l,
			overlap: ov,
			score:   ScoreWithTierWeightedOverlap(float64(ov), l),
			hrw:     hrwScore(key, c.Name),
		})
	}
	if len(items) == 0 {
		return PlannerReplica{}, false
	}

	// Balance target: lowest load, ties broken by deterministic HRW score then name.
	bal := items[0]
	for _, it := range items[1:] {
		if it.load < bal.load || (it.load == bal.load && (it.hrw > bal.hrw || (it.hrw == bal.hrw && it.replica.Name < bal.replica.Name))) {
			bal = it
		}
	}

	chosen := bal
	skewed := maxL-minL >= p.skew.AbsLoad && float64(maxL) >= p.skew.RelLoad*float64(mathx.MaxInt(minL, 1))
	if !skewed {
		loc := items[0]
		for _, it := range items[1:] {
			if it.score > loc.score || (it.score == loc.score && (it.load < loc.load || (it.load == loc.load && (it.hrw > loc.hrw || (it.hrw == loc.hrw && it.replica.Name < loc.replica.Name))))) {
				loc = it
			}
		}
		if loc.score > bal.score {
			chosen = loc
		}
	}

	if p.pending != nil && p.pending.Enabled() && len(norm) > 0 {
		p.pending.claimOrStakeLocked(chosen.replica.Name, norm, chosen.load, now)
	}
	return chosen.replica, true
}
