// Ported from github.com/maypok86/otter (policy.go, sketch.go, internal/xruntime/hasher.go
// @73e5472e8d8a6047589e48b37daaf952a390bc74), Apache-2.0.
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Alexey Mayshev and contributors.
// License: https://www.apache.org/licenses/LICENSE-2.0
// Modified for Fak: a standalone, single-threaded policy keyed by uint64 that owns its own
// queues and reports victims to the caller, instead of an intrusive policy driven by otter's
// node buffers and maintenance task. The liveness checks that only exist to reconcile
// out-of-order buffered writes are dropped because every call here is applied in order.

package radixkv

import (
	"errors"
	"math"
	"math/bits"
	"math/rand/v2"
)

// At this pin otter's adaptive policy is a window / probation / protected segmented LRU
// with a count-min frequency sketch for admission and a hill climber that resizes the
// window; there is no ghost queue. The S3FIFO names follow the requested row, the
// algorithm follows the source.
const (
	s3fifoPercentMain                 = 0.99
	s3fifoPercentMainProtected        = 0.80
	s3fifoHillClimberRestartThreshold = 0.05
	s3fifoHillClimberStepPercent      = 0.0625
	s3fifoHillClimberStepDecayRate    = 0.98
	// A candidate this popular may be admitted at random over a hotter victim, so an
	// attacker cannot pin the victim by flooding the candidate's sketch counters.
	s3fifoAdmitHashdosThreshold  = 6
	s3fifoQueueTransferThreshold = 1_000

	s3fifoResetMask = 0x7777777777777777
	s3fifoOneMask   = 0x1111111111111111
)

// S3FIFOQueue names the segment an entry is resident in.
type S3FIFOQueue uint8

const (
	S3FIFONotResident S3FIFOQueue = iota
	S3FIFOWindow
	S3FIFOProbation
	S3FIFOProtected
)

func (q S3FIFOQueue) String() string {
	switch q {
	case S3FIFOWindow:
		return "window"
	case S3FIFOProbation:
		return "probation"
	case S3FIFOProtected:
		return "protected"
	default:
		return "not-resident"
	}
}

// S3FIFOConfig configures an S3FIFOPolicy. Unweighted (the default), Capacity is an entry
// count and every Add weighs 1 whatever weight it passes. Weighted, Capacity is in the
// caller's weight units (bytes, blocks) and the frequency sketch is sized by the resident
// entry count instead, so a large byte capacity does not allocate a sketch that large.
// Seed and Rand default to random sources; tests pin them for determinism.
type S3FIFOConfig struct {
	Capacity uint64
	Weighted bool
	Seed     *uint64
	Rand     func() uint32
}

// ErrS3FIFOZeroCapacity refuses a policy that could never hold an entry.
var ErrS3FIFOZeroCapacity = errors.New("radixkv: s3fifo policy capacity must be > 0")

type s3fifoNode struct {
	key        uint64
	weight     uint64
	queue      S3FIFOQueue
	prev, next *s3fifoNode
}

// s3fifoDeque is ordered head (least recently used) to tail (most recently used).
type s3fifoDeque struct {
	head, tail *s3fifoNode
	n          int
}

func (d *s3fifoDeque) pushBack(n *s3fifoNode) {
	n.prev, n.next = d.tail, nil
	if d.tail == nil {
		d.head = n
	} else {
		d.tail.next = n
	}
	d.tail = n
	d.n++
}

func (d *s3fifoDeque) pushFront(n *s3fifoNode) {
	n.prev, n.next = nil, d.head
	if d.head == nil {
		d.tail = n
	} else {
		d.head.prev = n
	}
	d.head = n
	d.n++
}

func (d *s3fifoDeque) remove(n *s3fifoNode) {
	if n.prev == nil {
		d.head = n.next
	} else {
		n.prev.next = n.next
	}
	if n.next == nil {
		d.tail = n.prev
	} else {
		n.next.prev = n.prev
	}
	n.prev, n.next = nil, nil
	d.n--
}

func (d *s3fifoDeque) moveToBack(n *s3fifoNode) {
	if d.tail == n {
		return
	}
	d.remove(n)
	d.pushBack(n)
}

func (d *s3fifoDeque) moveToFront(n *s3fifoNode) {
	if d.head == n {
		return
	}
	d.remove(n)
	d.pushFront(n)
}

func (d *s3fifoDeque) popFront() *s3fifoNode {
	n := d.head
	if n != nil {
		d.remove(n)
	}
	return n
}

// s3fifoSketch is a 4-bit count-min sketch whose counters are halved every sampleSize
// additions, so popularity decays.
type s3fifoSketch struct {
	table       []uint64
	sampleSize  uint64
	blockMask   uint64
	size        uint64
	seed        uint64
	initialized bool
}

func (s *s3fifoSketch) ensureCapacity(maximumSize uint64) {
	if uint64(len(s.table)) >= maximumSize {
		return
	}
	s.initialized = true
	newSize := s3fifoRoundUpPowerOf2(maximumSize)
	if newSize < 8 {
		newSize = 8
	}
	s.table = make([]uint64, newSize)
	s.sampleSize = 10
	if maximumSize > math.MaxUint64/10 {
		s.sampleSize = math.MaxUint64
	} else if maximumSize != 0 {
		s.sampleSize = 10 * maximumSize
	}
	s.blockMask = (uint64(len(s.table)) >> 3) - 1
	s.size = 0
}

func (s *s3fifoSketch) frequency(k uint64) uint64 {
	if !s.initialized {
		return 0
	}
	frequency := uint64(math.MaxUint64)
	blockHash := s.hash(k)
	counterHash := s3fifoRehash(blockHash)
	block := (blockHash & s.blockMask) << 3
	for i := uint64(0); i < 4; i++ {
		h := counterHash >> (i << 3)
		index := (h >> 1) & 15
		offset := h & 1
		slot := block + offset + (i << 1)
		frequency = min(frequency, (s.table[slot]>>(index<<2))&0xf)
	}
	return frequency
}

func (s *s3fifoSketch) increment(k uint64) {
	if !s.initialized {
		return
	}
	blockHash := s.hash(k)
	counterHash := s3fifoRehash(blockHash)
	block := (blockHash & s.blockMask) << 3
	added := false
	for i := uint64(0); i < 4; i++ {
		h := counterHash >> (i << 3)
		added = s.incrementAt(block+(h&1)+(i<<1), (h>>1)&15) || added
	}
	if added {
		s.size++
		if s.size == s.sampleSize {
			s.reset()
		}
	}
}

func (s *s3fifoSketch) incrementAt(i, j uint64) bool {
	offset := j << 2
	mask := uint64(0xf) << offset
	if s.table[i]&mask != mask {
		s.table[i] += uint64(1) << offset
		return true
	}
	return false
}

func (s *s3fifoSketch) reset() {
	count := 0
	for i := range s.table {
		count += bits.OnesCount64(s.table[i] & s3fifoOneMask)
		s.table[i] = (s.table[i] >> 1) & s3fifoResetMask
	}
	s.size = (s.size - (uint64(count) >> 2)) >> 1
}

func (s *s3fifoSketch) hash(k uint64) uint64 {
	// Two multiply-xorshift rounds: one round lets keys differing by fixed XOR deltas collide
	// in the low bits whatever the seed is.
	hi, lo := bits.Mul64(k^s.seed, 0x2d358dccaa6c78a5)
	hi, lo = bits.Mul64(hi^lo, 0x8bb84b93962eacc9)
	return s3fifoSpread(hi ^ lo)
}

func s3fifoSpread(h uint64) uint64 {
	h ^= h >> 17
	h *= 0xed5ad4bb
	h ^= h >> 11
	h *= 0xac4c1b51
	h ^= h >> 15
	return h
}

func s3fifoRehash(h uint64) uint64 {
	h *= 0x31848bab
	h ^= h >> 14
	return h
}

func s3fifoRoundUpPowerOf2(x uint64) uint64 {
	if x <= 1 {
		return 1
	}
	return 1 << bits.Len64(x-1)
}

// S3FIFOPolicy is an opt-in, standalone adaptive admission/eviction policy over uint64
// keys. Despite the name it is otter's adaptive W-TinyLFU (window LRU, probation/protected
// SLRU, count-min admission, hill-climbed window), not a ghost-queue S3-FIFO. It is not
// safe for concurrent use. Writes (Add, Remove, SetCapacity) run the eviction and
// hill-climbing pass synchronously and return the evicted keys.
type S3FIFOPolicy struct {
	nodes     map[uint64]*s3fifoNode
	sketch    s3fifoSketch
	window    s3fifoDeque
	probation s3fifoDeque
	protected s3fifoDeque

	isWeighted                bool
	maximum                   uint64
	weightedSize              uint64
	windowMaximum             uint64
	windowWeightedSize        uint64
	mainProtectedMaximum      uint64
	mainProtectedWeightedSize uint64
	stepSize                  float64
	adjustment                int64
	hitsInSample              uint64
	missesInSample            uint64
	previousSampleHitRate     float64

	rand    func() uint32
	evicted []uint64
}

// NewS3FIFOPolicy builds a policy with the given capacity.
func NewS3FIFOPolicy(cfg S3FIFOConfig) (*S3FIFOPolicy, error) {
	if cfg.Capacity == 0 {
		return nil, ErrS3FIFOZeroCapacity
	}
	p := &S3FIFOPolicy{nodes: make(map[uint64]*s3fifoNode), rand: cfg.Rand, isWeighted: cfg.Weighted}
	if p.rand == nil {
		p.rand = rand.Uint32
	}
	if cfg.Seed != nil {
		p.sketch.seed = *cfg.Seed
	} else {
		p.sketch.seed = rand.Uint64()
	}
	p.setMaximumSize(cfg.Capacity)
	return p, nil
}

// Len is the number of resident entries.
func (p *S3FIFOPolicy) Len() int { return len(p.nodes) }

// WeightedSize is the total weight of resident entries.
func (p *S3FIFOPolicy) WeightedSize() uint64 { return p.weightedSize }

// Capacity is the maximum weighted size.
func (p *S3FIFOPolicy) Capacity() uint64 { return p.maximum }

// Contains reports residency without counting an access.
func (p *S3FIFOPolicy) Contains(key uint64) bool {
	_, ok := p.nodes[key]
	return ok
}

// QueueOf reports which segment holds key.
func (p *S3FIFOPolicy) QueueOf(key uint64) S3FIFOQueue {
	if n, ok := p.nodes[key]; ok {
		return n.queue
	}
	return S3FIFONotResident
}

// Frequency is the sketch's popularity estimate for key, 0..15.
func (p *S3FIFOPolicy) Frequency(key uint64) uint64 { return p.sketch.frequency(key) }

// Access records a read of key and reports whether it was resident. A miss changes
// nothing; the caller admits the key with Add.
func (p *S3FIFOPolicy) Access(key uint64) bool {
	n, ok := p.nodes[key]
	if !ok {
		return false
	}
	p.access(n)
	return true
}

// Add inserts key with weight, or updates the weight of a resident key and counts an
// access, then evicts down to capacity. A zero-weight entry is never evicted for size.
// An unweighted policy ignores weight and counts the entry as 1.
// The returned slice is only valid until the next write.
func (p *S3FIFOPolicy) Add(key uint64, weight uint32) []uint64 {
	p.evicted = p.evicted[:0]
	if !p.isWeighted {
		weight = 1
	}
	if n, ok := p.nodes[key]; ok {
		p.update(n, uint64(weight))
	} else {
		n = &s3fifoNode{key: key, weight: uint64(weight), queue: S3FIFOWindow}
		p.nodes[key] = n
		p.add(n)
	}
	p.maintain()
	return p.evicted
}

// Remove drops key without counting it as an eviction.
func (p *S3FIFOPolicy) Remove(key uint64) bool {
	n, ok := p.nodes[key]
	if !ok {
		return false
	}
	p.unlink(n)
	return true
}

// SetCapacity changes the maximum weighted size, restarts the hill climber, and evicts
// down to the new maximum. A zero capacity is ignored.
func (p *S3FIFOPolicy) SetCapacity(maximum uint64) []uint64 {
	p.evicted = p.evicted[:0]
	if maximum == 0 {
		return p.evicted
	}
	p.setMaximumSize(maximum)
	p.maintain()
	return p.evicted
}

func (p *S3FIFOPolicy) maintain() {
	p.evictNodes()
	p.climb()
}

func (p *S3FIFOPolicy) queueFor(q S3FIFOQueue) *s3fifoDeque {
	switch q {
	case S3FIFOWindow:
		return &p.window
	case S3FIFOProbation:
		return &p.probation
	default:
		return &p.protected
	}
}

func (p *S3FIFOPolicy) access(n *s3fifoNode) {
	p.sketch.increment(n.key)
	switch n.queue {
	case S3FIFOWindow:
		p.window.moveToBack(n)
	case S3FIFOProbation:
		p.reorderProbation(n)
	case S3FIFOProtected:
		p.protected.moveToBack(n)
	}
	p.hitsInSample++
}

func (p *S3FIFOPolicy) add(n *s3fifoNode) {
	p.adjustAccounted(n, 0, n.weight)
	if p.weightedSize >= p.maximum>>1 {
		// The sketch is sized lazily, once the policy is close to evicting.
		capacity := p.maximum
		if p.isWeighted {
			capacity = uint64(len(p.nodes))
		}
		p.sketch.ensureCapacity(capacity)
	}
	p.sketch.increment(n.key)
	p.missesInSample++

	// An entry too heavy for the window goes to its head so it leaves the window first.
	if n.weight > p.windowMaximum {
		p.window.pushFront(n)
	} else {
		p.window.pushBack(n)
	}
	if n.weight > p.maximum {
		p.evictNode(n)
	}
}

func (p *S3FIFOPolicy) update(n *s3fifoNode, weight uint64) {
	p.adjustAccounted(n, n.weight, weight)
	n.weight = weight
	switch n.queue {
	case S3FIFOWindow:
		switch {
		case weight > p.maximum:
			p.evictNode(n)
		case weight <= p.windowMaximum:
			p.access(n)
		default:
			p.window.moveToFront(n)
		}
	default:
		if weight <= p.maximum {
			p.access(n)
		} else {
			p.evictNode(n)
		}
	}
}

func (p *S3FIFOPolicy) evictNode(n *s3fifoNode) {
	p.unlink(n)
	p.evicted = append(p.evicted, n.key)
}

func (p *S3FIFOPolicy) unlink(n *s3fifoNode) {
	p.queueFor(n.queue).remove(n)
	p.adjustAccounted(n, n.weight, 0)
	n.queue = S3FIFONotResident
	delete(p.nodes, n.key)
}

func (p *S3FIFOPolicy) adjustAccounted(n *s3fifoNode, oldWeight, newWeight uint64) {
	p.weightedSize = p.weightedSize - oldWeight + newWeight
	switch n.queue {
	case S3FIFOWindow:
		p.windowWeightedSize = p.windowWeightedSize - oldWeight + newWeight
	case S3FIFOProtected:
		p.mainProtectedWeightedSize = p.mainProtectedWeightedSize - oldWeight + newWeight
	}
}

func (p *S3FIFOPolicy) setMaximumSize(maximum uint64) {
	if maximum == p.maximum {
		return
	}
	window := maximum - uint64(s3fifoPercentMain*float64(maximum))
	p.maximum = maximum
	p.windowMaximum = window
	p.mainProtectedMaximum = uint64(s3fifoPercentMainProtected * float64(maximum-window))

	// The climber's pending quota and last hit rate were measured for the old maximum.
	p.hitsInSample = 0
	p.missesInSample = 0
	p.previousSampleHitRate = 0
	p.adjustment = 0
	p.stepSize = -s3fifoHillClimberStepPercent * float64(maximum)

	if !p.isWeighted && p.weightedSize >= maximum>>1 {
		p.sketch.ensureCapacity(maximum)
	}
}

// reorderProbation promotes n to protected; protected overflow is demoted later by climb.
func (p *S3FIFOPolicy) reorderProbation(n *s3fifoNode) {
	if n.weight > p.mainProtectedMaximum {
		p.probation.moveToBack(n)
		return
	}
	p.mainProtectedWeightedSize += n.weight
	p.probation.remove(n)
	p.protected.pushBack(n)
	n.queue = S3FIFOProtected
}

func (p *S3FIFOPolicy) evictNodes() {
	candidate := p.evictFromWindow()
	p.evictFromMain(candidate)
}

func (p *S3FIFOPolicy) evictFromWindow() *s3fifoNode {
	var first *s3fifoNode
	n := p.window.head
	for p.windowWeightedSize > p.windowMaximum && n != nil {
		next := n.next
		if n.weight != 0 {
			p.window.remove(n)
			p.probation.pushBack(n)
			n.queue = S3FIFOProbation
			if first == nil {
				first = n
			}
			p.windowWeightedSize -= n.weight
		}
		n = next
	}
	return first
}

// evictFromMain duels the probation head (victim) against entries just demoted from the
// window (candidates); the sketch decides which one leaves.
func (p *S3FIFOPolicy) evictFromMain(candidate *s3fifoNode) {
	victimQueue := S3FIFOProbation
	candidateQueue := S3FIFOProbation
	victim := p.probation.head
	for p.weightedSize > p.maximum {
		if candidate == nil && candidateQueue == S3FIFOProbation {
			candidate = p.window.head
			candidateQueue = S3FIFOWindow
		}

		if candidate == nil && victim == nil {
			if victimQueue == S3FIFOProbation {
				victim = p.protected.head
				victimQueue = S3FIFOProtected
				continue
			} else if victimQueue == S3FIFOProtected {
				victim = p.window.head
				victimQueue = S3FIFOWindow
				continue
			}
			break
		}

		if victim != nil && victim.weight == 0 {
			victim = victim.next
			continue
		} else if candidate != nil && candidate.weight == 0 {
			candidate = candidate.next
			continue
		}

		if victim == nil {
			evict := candidate
			candidate = candidate.next
			p.evictNode(evict)
			continue
		} else if candidate == nil {
			evict := victim
			victim = victim.next
			p.evictNode(evict)
			continue
		}

		if candidate == victim {
			victim = victim.next
			p.evictNode(candidate)
			candidate = nil
			continue
		}

		if candidate.weight > p.maximum {
			evict := candidate
			candidate = candidate.next
			p.evictNode(evict)
			continue
		}

		if p.admit(candidate.key, victim.key) {
			evict := victim
			victim = victim.next
			p.evictNode(evict)
			candidate = candidate.next
		} else {
			evict := candidate
			candidate = candidate.next
			p.evictNode(evict)
		}
	}
}

func (p *S3FIFOPolicy) admit(candidateKey, victimKey uint64) bool {
	victimFreq := p.sketch.frequency(victimKey)
	incomingFreq := p.sketch.frequency(candidateKey)
	if incomingFreq > victimFreq {
		return true
	}
	if incomingFreq >= s3fifoAdmitHashdosThreshold {
		// Counters saturate at 15 and age to 7, so a warm floor keeps random admissions rare.
		return p.rand()&127 == 0
	}
	return false
}

func (p *S3FIFOPolicy) climb() {
	p.determineAdjustment()
	p.demoteFromMainProtected()
	switch {
	case p.adjustment > 0:
		p.increaseWindow()
	case p.adjustment < 0:
		p.decreaseWindow()
	}
}

func (p *S3FIFOPolicy) determineAdjustment() {
	if !p.sketch.initialized {
		p.previousSampleHitRate = 0
		p.missesInSample = 0
		p.hitsInSample = 0
		return
	}
	requestCount := p.hitsInSample + p.missesInSample
	if requestCount < p.sketch.sampleSize {
		return
	}
	if p.weightedSize < p.maximum>>1 {
		// Not evicting, so the hit rate says nothing about the window/main split.
		p.missesInSample = 0
		p.hitsInSample = 0
		return
	}

	hitRate := float64(p.hitsInSample) / float64(requestCount)
	hitRateChange := hitRate - p.previousSampleHitRate
	amount := p.stepSize
	if hitRateChange < 0 {
		amount = -p.stepSize
	}
	var nextStepSize float64
	if math.Abs(hitRateChange) >= s3fifoHillClimberRestartThreshold {
		k := float64(-1)
		if amount >= 0 {
			k = 1
		}
		nextStepSize = s3fifoHillClimberStepPercent * float64(p.maximum) * k
	} else {
		nextStepSize = s3fifoHillClimberStepDecayRate * amount
	}
	p.previousSampleHitRate = hitRate
	p.adjustment = int64(amount)
	p.stepSize = nextStepSize
	p.missesInSample = 0
	p.hitsInSample = 0
}

func (p *S3FIFOPolicy) demoteFromMainProtected() {
	size := p.mainProtectedWeightedSize
	for i := 0; i < s3fifoQueueTransferThreshold && size > p.mainProtectedMaximum; i++ {
		demoted := p.protected.popFront()
		if demoted == nil {
			break
		}
		demoted.queue = S3FIFOProbation
		p.probation.pushBack(demoted)
		size -= demoted.weight
	}
	p.mainProtectedWeightedSize = size
}

func (p *S3FIFOPolicy) increaseWindow() {
	if p.mainProtectedMaximum == 0 {
		return
	}
	quota := p.adjustment
	if p.mainProtectedMaximum < uint64(p.adjustment) {
		quota = int64(p.mainProtectedMaximum)
	}
	p.mainProtectedMaximum -= uint64(quota)
	p.windowMaximum += uint64(quota)
	p.demoteFromMainProtected()

	for i := 0; i < s3fifoQueueTransferThreshold; i++ {
		candidate := p.probation.head
		fromProbation := true
		if candidate == nil || quota < int64(candidate.weight) {
			candidate = p.protected.head
			fromProbation = false
		}
		if candidate == nil || quota < int64(candidate.weight) {
			break
		}
		quota -= int64(candidate.weight)
		if fromProbation {
			p.probation.remove(candidate)
		} else {
			p.mainProtectedWeightedSize -= candidate.weight
			p.protected.remove(candidate)
		}
		p.windowWeightedSize += candidate.weight
		p.window.pushBack(candidate)
		candidate.queue = S3FIFOWindow
	}

	p.mainProtectedMaximum += uint64(quota)
	p.windowMaximum -= uint64(quota)
	p.adjustment = quota
}

func (p *S3FIFOPolicy) decreaseWindow() {
	if p.windowMaximum <= 1 {
		return
	}
	quota := -p.adjustment
	windowMaximum := p.windowMaximum - 1
	if windowMaximum < uint64(-p.adjustment) {
		quota = int64(windowMaximum)
	}
	p.mainProtectedMaximum += uint64(quota)
	p.windowMaximum -= uint64(quota)

	for i := 0; i < s3fifoQueueTransferThreshold; i++ {
		candidate := p.window.head
		if candidate == nil || quota < int64(candidate.weight) {
			break
		}
		quota -= int64(candidate.weight)
		p.windowWeightedSize -= candidate.weight
		p.window.remove(candidate)
		p.probation.pushBack(candidate)
		candidate.queue = S3FIFOProbation
	}

	p.mainProtectedMaximum -= uint64(quota)
	p.windowMaximum += uint64(quota)
	p.adjustment = -quota
}
