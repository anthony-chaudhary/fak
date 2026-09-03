package radixkv

import (
	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// admission.go wires the pure cacheprice frequency sketch and compute value comparator
// into radixkv's real Insert / InsertSnapshot ownership path (#9311). The gate runs only
// when an insert would displace bounded token or snapshot residency. A no-pressure fill is
// unchanged, and a missing sketch fails open to the historical insert-always behavior
// unless the bounded tree cannot physically free enough unlocked capacity for the insert.
//
// Cost-aware trees enable this gate by default because they already selected the same
// value axis for eviction. SetAdmissionEnabled(false) is the immediate rollback: it drops
// the sketch and makes every later insert deterministic insert-always without changing the
// eviction strategy.
//
// Demand lookups feed a namespace-qualified full-prefix fingerprint into a fixed-size
// count-min sketch. Under pressure the candidate must first clear the frequency bar and
// then outrank every token/snapshot victim it would displace. A bypass returns the still-
// leased lookup boundary, so Done releases exactly the lease Lookup took; caller-owned KV
// remains untouched, and InsertSnapshot consumes/ closes its transferred snapshot on the
// successful bypass path.

const (
	admissionSketchWidth        = 1024
	admissionSketchDepth        = 4
	admissionFrequencyThreshold = 2
	admissionJournalCap         = 256
)

const (
	admissionReasonFrequencyBelowThreshold = "frequency_below_threshold"
	admissionReasonTelemetryAbsent         = "telemetry_absent"
	admissionReasonInsufficientCapacity    = "insufficient_evictable_capacity"
)

// admissionComparison is one pressured residency plane. Candidate geometry is kept
// separate per plane because token-tree accounting and complete snapshot bytes have
// different footprints, while the observed frequency is shared by the full prefix.
type admissionComparison struct {
	candidateTokens     int
	candidateBytes      int64
	candidateWriteBytes int64
	victim              compute.KVSpanStats
	forcedReason        string
}

// admissionRecord is one rejected candidate and its optional later recovery. The journal
// is bounded by admissionJournalCap; repeated rejects update the unresolved row instead of
// growing state, while recovery stamps that row so the reject→recover sequence remains
// inspectable rather than disappearing into aggregate counters.
type admissionRecord struct {
	keyHash       uint64
	firstRejected uint64
	lastRejected  uint64
	recoveredAt   uint64
	recoveryGap   uint64
	attempts      int
	frequency     int
	tokens        int
	reason        string
}

// SetAdmissionEnabled selects the live value/frequency admission gate. Disabling is the
// rollback/fallback path: frequency telemetry is discarded and future pressured inserts
// reproduce insert-always deterministically for feasible inserts. Lifetime counters and
// pending reject receipts remain visible; a later successful fallback insert can still
// reconcile a pending reject.
func (t *Tree) SetAdmissionEnabled(enabled bool) {
	if t.admissionEnabled == enabled {
		return
	}
	t.admissionEnabled = enabled
	t.admissionSketch = nil
}

// observeAdmission records one demand access before Lookup mutates/splits the tree. The
// full request path is used even on a miss, so a candidate rejected today accumulates heat
// and can earn admission on a later retry. Unbounded trees allocate no estimator state.
func (t *Tree) observeAdmission(root *node, tokens []int) {
	if !t.admissionEnabled || !t.admissionHasBound() || root == nil {
		return
	}
	if t.admissionSketch == nil {
		t.admissionSketch = cacheprice.NewFrequencySketch(admissionSketchWidth, admissionSketchDepth)
	}
	t.admissionSketch.Touch(admissionSketchKey(t.admissionKeyHash(root, tokens)))
	t.admissionObservations++
}

func (t *Tree) admissionHasBound() bool {
	_, tokenBound := t.resolveBudget()
	return tokenBound || t.maxSnapshotBytes > 0
}

// admissionKeyHash folds namespace identity plus the complete token path. Namespace is
// load-bearing here: heat in tenant A may not admit an identical-token candidate in tenant
// B, even though both namespaces share the global capacity budget.
func (t *Tree) admissionKeyHash(boundary *node, suffix []int) uint64 {
	if boundary == nil {
		return 0
	}
	root := boundary
	for root.parent != nil {
		root = root.parent
	}
	ns := ""
	if root != t.root {
		for candidateNS, candidateRoot := range t.nsRoots {
			if candidateRoot == root {
				ns = candidateNS
				break
			}
		}
	}
	h := fnv64Offset
	h = foldUint64(h, uint64(len(ns)))
	for i := 0; i < len(ns); i++ {
		h ^= uint64(ns[i])
		h *= fnv64Prime
	}
	return foldUint64(h, pathHash(boundary, suffix))
}

func foldUint64(h, value uint64) uint64 {
	for i := 0; i < 8; i++ {
		h ^= value & 0xff
		h *= fnv64Prime
		value >>= 8
	}
	return h
}

// admissionSketchKey renders a uint64 fingerprint as exactly eight bytes. FrequencySketch
// hashes opaque strings, so this avoids retaining variable-length prefixes or namespaces.
func admissionSketchKey(hash uint64) string {
	var key [8]byte
	for i := range key {
		key[i] = byte(hash >> (8 * i))
	}
	return string(key[:])
}

func (t *Tree) tokenAdmissionComparisons(boundary *node, suffix []int) []admissionComparison {
	if boundary == nil || len(suffix) == 0 {
		return nil
	}
	budget, evicts := t.resolveBudget()
	if !evicts || t.tokens+len(suffix) <= budget {
		return nil
	}
	candidate := admissionComparison{
		candidateTokens: len(suffix),
		candidateBytes:  int64(max(boundary.plen+len(suffix), 1)),
	}
	required := t.tokens + len(suffix) - budget
	victims, enough := t.prospectiveTokenVictims(required)
	if !enough {
		candidate.forcedReason = admissionReasonInsufficientCapacity
		return []admissionComparison{candidate}
	}
	comparisons := make([]admissionComparison, len(victims))
	for i, victim := range victims {
		comparisons[i] = candidate
		comparisons[i].victim = victim
	}
	return comparisons
}

// prospectiveTokenVictims simulates evictToBudget's upward leaf collapse without mutating
// the tree. It returns every existing victim the leased incoming leaf would force out, so
// admission compares against all displaced value rather than only the cheapest first leaf.
func (t *Tree) prospectiveTokenVictims(required int) ([]compute.KVSpanStats, bool) {
	if required <= 0 {
		return nil, true
	}
	remainingChildren := make(map[*node]int)
	removed := make(map[*node]bool)
	var nodes []*node
	t.forEachRoot(func(root *node) {
		var walk func(*node)
		walk = func(n *node) {
			remainingChildren[n] = len(n.children)
			for _, child := range n.children {
				nodes = append(nodes, child)
				walk(child)
			}
		}
		walk(root)
	})

	strategy := t.evictionStrategy()
	if prep, ok := strategy.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	freed := 0
	var victims []compute.KVSpanStats
	for freed < required {
		var best *node
		var bestKey victimKey
		for _, n := range nodes {
			if removed[n] || remainingChildren[n] != 0 || n.refs > 0 {
				continue
			}
			key := strategy.Priority(n)
			if best == nil || key.less(bestKey) {
				best, bestKey = n, key
			}
		}
		if best == nil {
			return victims, false
		}
		victims = append(victims, best.kvSpanStats())
		freed += len(best.key)
		removed[best] = true
		if best.parent != nil {
			remainingChildren[best.parent]--
		}
	}
	return victims, true
}

func (t *Tree) snapshotAdmissionComparisons(boundary *node, suffix []int, incoming int64) []admissionComparison {
	if !t.admissionEnabled || boundary == nil || incoming <= 0 || t.maxSnapshotBytes == 0 {
		return nil
	}
	var exclude *node
	var oldBytes int64
	if len(suffix) == 0 {
		exclude = boundary
		oldBytes = snapshotResidentBytes(boundary.snapshot, boundary.cachedLogits)
	}
	delta := incoming - oldBytes
	if delta <= 0 || t.snapshotBytes+delta <= t.maxSnapshotBytes {
		return nil
	}
	candidate := admissionComparison{
		candidateTokens:     boundary.plen + len(suffix),
		candidateBytes:      incoming,
		candidateWriteBytes: incoming,
	}
	required := t.snapshotBytes + delta - t.maxSnapshotBytes
	victims, enough := t.prospectiveSnapshotVictims(required, exclude)
	if !enough {
		candidate.forcedReason = admissionReasonInsufficientCapacity
		return []admissionComparison{candidate}
	}
	comparisons := make([]admissionComparison, len(victims))
	for i, victim := range victims {
		comparisons[i] = candidate
		comparisons[i].victim = victim
	}
	return comparisons
}

func (t *Tree) prospectiveSnapshotVictims(required int64, exclude *node) ([]compute.KVSpanStats, bool) {
	if required <= 0 {
		return nil, true
	}
	selected := make(map[*node]bool)
	freed := int64(0)
	var victims []compute.KVSpanStats
	for freed < required {
		var best *node
		var bestKey victimKey
		strategy := t.evictionStrategy()
		if prep, ok := strategy.(TreePreparer); ok {
			prep.PrepareTree(t)
		}
		var walk func(*node)
		walk = func(n *node) {
			if n != exclude && !selected[n] && n.refs == 0 && n.snapshot != nil {
				key := strategy.Priority(n)
				if best == nil || key.less(bestKey) {
					best, bestKey = n, key
				}
			}
			for _, child := range n.children {
				walk(child)
			}
		}
		t.forEachRoot(walk)
		if best == nil {
			return victims, false
		}
		victims = append(victims, snapshotKVSpanStats(best))
		freed += snapshotResidentBytes(best.snapshot, best.cachedLogits)
		selected[best] = true
	}
	return victims, true
}

func snapshotKVSpanStats(n *node) compute.KVSpanStats {
	if n == nil {
		return compute.KVSpanStats{}
	}
	return compute.KVSpanStats{
		Tokens:   n.plen,
		Bytes:    snapshotResidentBytes(n.snapshot, n.cachedLogits),
		Hits:     n.hits,
		LastUsed: n.lastUsed,
		Leased:   n.refs > 0,
	}
}

// admitCandidate applies frequency first, then the same value-of-keeping comparison used
// by cost-aware eviction. All comparisons must admit: a candidate cannot displace a
// valuable token victim merely because its snapshot victim was cold, or vice versa.
func (t *Tree) admitCandidate(keyHash uint64, comparisons []admissionComparison) bool {
	t.admissionCandidates++
	candidateTokens := 0
	var candidateBytes int64
	var candidateWriteBytes int64
	hotVictim := false
	for _, comparison := range comparisons {
		if comparison.candidateTokens > candidateTokens {
			candidateTokens = comparison.candidateTokens
		}
		if comparison.candidateBytes > candidateBytes {
			candidateBytes = comparison.candidateBytes
		}
		if comparison.candidateWriteBytes > candidateWriteBytes {
			candidateWriteBytes = comparison.candidateWriteBytes
		}
		if comparison.victim.Hits > 0 {
			hotVictim = true
		}
	}

	forcedReason := ""
	for _, comparison := range comparisons {
		if comparison.forcedReason != "" {
			forcedReason = comparison.forcedReason
			break
		}
	}
	if forcedReason != "" {
		frequency := 0
		if t.admissionSketch != nil {
			frequency = t.admissionSketch.Estimate(admissionSketchKey(keyHash))
		}
		t.rejectAdmission(keyHash, frequency, candidateTokens, candidateWriteBytes, forcedReason, hotVictim)
		return false
	}

	if !t.admissionEnabled || t.admissionSketch == nil {
		t.admissionAdmitted++
		t.admissionTelemetryFallbacks++
		t.lastAdmissionFrequency = 0
		t.lastAdmissionReason = admissionReasonTelemetryAbsent
		return true
	}

	frequency := t.admissionSketch.Estimate(admissionSketchKey(keyHash))
	t.lastAdmissionFrequency = frequency
	if !cacheprice.ShouldReadmit(frequency, admissionFrequencyThreshold, true, 0, -1) {
		t.rejectAdmission(keyHash, frequency, candidateTokens, candidateWriteBytes, admissionReasonFrequencyBelowThreshold, hotVictim)
		return false
	}

	reason := ""
	for _, comparison := range comparisons {
		candidate := compute.KVSpanStats{
			Tokens:   comparison.candidateTokens,
			Bytes:    comparison.candidateBytes,
			Hits:     max(frequency-1, 0),
			LastUsed: t.clock,
		}
		decision := compute.DecideKVAdmission(candidate, comparison.victim, 0)
		reason = string(decision.Reason)
		if decision.Verdict == compute.AdmitBypass {
			t.rejectAdmission(keyHash, frequency, candidateTokens, candidateWriteBytes, reason, hotVictim)
			return false
		}
	}

	t.admissionAdmitted++
	t.lastAdmissionReason = reason
	return true
}

func (t *Tree) rejectAdmission(keyHash uint64, frequency, tokens int, writeBytes int64, reason string, hotVictim bool) {
	t.admissionRejected++
	t.admissionRejectedTokens += tokens
	if writeBytes <= 0 {
		// Regular radix nodes have no snapshot byte receipt; use the existing
		// full-prefix token proxy so avoided write pressure remains observable.
		writeBytes = int64(tokens)
	}
	t.admissionRejectedBytes += writeBytes
	if hotVictim {
		t.admissionHotProtected++
	}
	t.lastAdmissionFrequency = frequency
	t.lastAdmissionReason = reason
	for i := len(t.admissionJournal) - 1; i >= 0; i-- {
		rec := &t.admissionJournal[i]
		if rec.keyHash != keyHash || rec.recoveredAt != 0 {
			continue
		}
		rec.lastRejected = t.clock
		rec.attempts++
		rec.frequency = frequency
		rec.tokens = tokens
		rec.reason = reason
		return
	}
	if len(t.admissionJournal) == admissionJournalCap {
		copy(t.admissionJournal, t.admissionJournal[1:])
		t.admissionJournal = t.admissionJournal[:admissionJournalCap-1]
		t.admissionJournalDropped++
	}
	t.admissionJournal = append(t.admissionJournal, admissionRecord{
		keyHash:       keyHash,
		firstRejected: t.clock,
		lastRejected:  t.clock,
		attempts:      1,
		frequency:     frequency,
		tokens:        tokens,
		reason:        reason,
	})
}

// noteAdmissionRecovery settles one pending rejection when the same namespace-qualified
// prefix is finally admitted. The row remains in the bounded journal with its recovery
// stamp and gap, so the recovery is witnessed rather than only counted.
func (t *Tree) noteAdmissionRecovery(keyHash uint64) {
	for i := len(t.admissionJournal) - 1; i >= 0; i-- {
		rec := &t.admissionJournal[i]
		if rec.keyHash != keyHash || rec.recoveredAt != 0 {
			continue
		}
		gap := t.clock - rec.firstRejected
		rec.recoveredAt = t.clock
		rec.recoveryGap = gap
		t.admissionRecoveries++
		t.admissionRecoveryGapLast = gap
		if gap > t.admissionRecoveryGapMax {
			t.admissionRecoveryGapMax = gap
		}
		return
	}
}

func (t *Tree) admissionJournalPending() int {
	pending := 0
	for i := range t.admissionJournal {
		if t.admissionJournal[i].recoveredAt == 0 {
			pending++
		}
	}
	return pending
}
