package eviction

// WTinyLFU implements the Window-TinyLFU eviction policy.
// Window LRU (1%) -> admission filter -> Main segmented LRU (99%)
// Main = Probation (20% of main) + Protected (80% of main)
type WTinyLFU struct {
	capacity uint64
	sketch   *CountMinSketch

	window    lruList // 1% of capacity
	probation lruList // ~20% of main (99%)
	protected lruList // ~80% of main (99%)

	windowCap uint64
	probCap   uint64
	protCap   uint64

	pool      []lruEntry       // pre-allocated entry pool
	freeHead  int32            // head of free list (-1 = none)
	keyToNode map[uint64]int32 // keyHash -> pool index

	onEvict func(keyHash uint64, keyLen uint16) // callback when evicting
}

// listType identifies which LRU list an entry belongs to.
type listType uint8

const (
	listWindow listType = iota
	listProbation
	listProtected
)

type lruEntry struct {
	keyHash  uint64
	keyLen   uint16
	prev     int32
	next     int32
	listType listType
	inUse    bool
	freeNext int32 // next in free list
}

type lruList struct {
	head int32
	tail int32
	size uint64
}

// NewWTinyLFU creates a new W-TinyLFU eviction engine.
func NewWTinyLFU(capacity uint64, onEvict func(keyHash uint64, keyLen uint16)) *WTinyLFU {
	if capacity < 4 {
		capacity = 4
	}

	windowCap := capacity / 100
	if windowCap < 1 {
		windowCap = 1
	}
	mainCap := capacity - windowCap
	protCap := mainCap * 80 / 100
	probCap := mainCap - protCap

	poolSize := capacity + 1 // extra for temporary overflow
	pool := make([]lruEntry, poolSize)
	// Build free list
	for i := range pool {
		pool[i].freeNext = int32(i + 1)
		pool[i].prev = -1
		pool[i].next = -1
	}
	pool[len(pool)-1].freeNext = -1

	w := &WTinyLFU{
		capacity:  capacity,
		sketch:    NewCountMinSketch(capacity * 10),
		windowCap: windowCap,
		probCap:   probCap,
		protCap:   protCap,
		pool:      pool,
		freeHead:  0,
		keyToNode: make(map[uint64]int32, capacity),
		onEvict:   onEvict,
	}
	w.window.head = -1
	w.window.tail = -1
	w.probation.head = -1
	w.probation.tail = -1
	w.protected.head = -1
	w.protected.tail = -1
	return w
}

// Access records an access to a key. Returns true if the key was already tracked.
func (w *WTinyLFU) Access(keyHash uint64) bool {
	w.sketch.Increment(keyHash)

	nodeIdx, exists := w.keyToNode[keyHash]
	if !exists {
		return false
	}

	node := &w.pool[nodeIdx]
	switch node.listType {
	case listWindow:
		// Move to head of window
		w.moveToHead(&w.window, nodeIdx)
	case listProbation:
		// Promote to protected
		w.removeFrom(&w.probation, nodeIdx)
		node.listType = listProtected
		w.pushHead(&w.protected, nodeIdx)
		// If protected overflows, demote tail to probation
		for w.protected.size > w.protCap {
			w.demoteProtected()
		}
	case listProtected:
		// Move to head of protected
		w.moveToHead(&w.protected, nodeIdx)
	}

	return true
}

// Admit adds a key to the cache. May trigger eviction.
// Returns the list of evicted key hashes.
func (w *WTinyLFU) Admit(keyHash uint64, keyLen uint16) []uint64 {
	var evicted []uint64

	// Already tracked?
	if _, exists := w.keyToNode[keyHash]; exists {
		w.Access(keyHash)
		return nil
	}

	w.sketch.Increment(keyHash)

	// Allocate a pool entry
	nodeIdx := w.allocNode()
	if nodeIdx < 0 {
		// Pool exhausted — force eviction
		evHash := w.forceEvict()
		if evHash != 0 {
			evicted = append(evicted, evHash)
		}
		nodeIdx = w.allocNode()
		if nodeIdx < 0 {
			return evicted // shouldn't happen
		}
	}

	node := &w.pool[nodeIdx]
	node.keyHash = keyHash
	node.keyLen = keyLen
	node.listType = listWindow
	node.inUse = true
	w.keyToNode[keyHash] = nodeIdx

	w.pushHead(&w.window, nodeIdx)

	// Drain window overflow through admission filter
	for w.window.size > w.windowCap {
		evHash := w.drainWindow()
		if evHash != 0 {
			evicted = append(evicted, evHash)
		}
	}

	return evicted
}

// Remove explicitly removes a key from tracking.
func (w *WTinyLFU) Remove(keyHash uint64) {
	nodeIdx, exists := w.keyToNode[keyHash]
	if !exists {
		return
	}
	node := &w.pool[nodeIdx]
	switch node.listType {
	case listWindow:
		w.removeFrom(&w.window, nodeIdx)
	case listProbation:
		w.removeFrom(&w.probation, nodeIdx)
	case listProtected:
		w.removeFrom(&w.protected, nodeIdx)
	}
	delete(w.keyToNode, keyHash)
	w.freeNode(nodeIdx)
}

// Size returns the total number of tracked entries.
func (w *WTinyLFU) Size() uint64 {
	return w.window.size + w.probation.size + w.protected.size
}

// EvictOne force-evicts a single entry. Returns false if nothing to evict.
func (w *WTinyLFU) EvictOne() bool {
	return w.forceEvict() != 0
}

// drainWindow moves the window tail to probation (or evicts main tail).
func (w *WTinyLFU) drainWindow() uint64 {
	if w.window.tail < 0 {
		return 0
	}
	candidate := w.window.tail
	w.removeFrom(&w.window, candidate)

	// If main cache is full, run admission filter
	if w.probation.size+w.protected.size >= w.probCap+w.protCap {
		return w.admissionFilter(candidate)
	}

	// Main not full — admit directly to probation
	w.pool[candidate].listType = listProbation
	w.pushHead(&w.probation, candidate)
	return 0
}

// admissionFilter: compare candidate freq vs probation tail freq.
func (w *WTinyLFU) admissionFilter(candidateIdx int32) uint64 {
	candidate := &w.pool[candidateIdx]
	candidateFreq := w.sketch.Estimate(candidate.keyHash)

	// Find victim: probation tail
	victimIdx := w.probation.tail
	if victimIdx < 0 {
		// No probation entries, just admit candidate
		candidate.listType = listProbation
		w.pushHead(&w.probation, candidateIdx)
		return 0
	}

	victim := &w.pool[victimIdx]
	victimFreq := w.sketch.Estimate(victim.keyHash)

	if candidateFreq > victimFreq {
		// Candidate wins — evict victim, admit candidate
		evictedHash := victim.keyHash
		evictedKeyLen := victim.keyLen
		w.removeFrom(&w.probation, victimIdx)
		delete(w.keyToNode, evictedHash)
		w.freeNode(victimIdx)
		if w.onEvict != nil {
			w.onEvict(evictedHash, evictedKeyLen)
		}

		candidate.listType = listProbation
		w.pushHead(&w.probation, candidateIdx)
		return evictedHash
	}

	// Victim wins — evict candidate
	evictedHash := candidate.keyHash
	evictedKeyLen := candidate.keyLen
	delete(w.keyToNode, evictedHash)
	w.freeNode(candidateIdx)
	if w.onEvict != nil {
		w.onEvict(evictedHash, evictedKeyLen)
	}
	return evictedHash
}

func (w *WTinyLFU) demoteProtected() {
	if w.protected.tail < 0 {
		return
	}
	idx := w.protected.tail
	w.removeFrom(&w.protected, idx)
	w.pool[idx].listType = listProbation
	w.pushHead(&w.probation, idx)
}

func (w *WTinyLFU) forceEvict() uint64 {
	// Evict from probation first, then window
	if w.probation.tail >= 0 {
		idx := w.probation.tail
		hash := w.pool[idx].keyHash
		kLen := w.pool[idx].keyLen
		w.removeFrom(&w.probation, idx)
		delete(w.keyToNode, hash)
		w.freeNode(idx)
		if w.onEvict != nil {
			w.onEvict(hash, kLen)
		}
		return hash
	}
	if w.window.tail >= 0 {
		idx := w.window.tail
		hash := w.pool[idx].keyHash
		kLen := w.pool[idx].keyLen
		w.removeFrom(&w.window, idx)
		delete(w.keyToNode, hash)
		w.freeNode(idx)
		if w.onEvict != nil {
			w.onEvict(hash, kLen)
		}
		return hash
	}
	return 0
}

// Linked list operations (intrusive, pool-based)

func (w *WTinyLFU) pushHead(list *lruList, idx int32) {
	node := &w.pool[idx]
	node.prev = -1
	node.next = list.head
	if list.head >= 0 {
		w.pool[list.head].prev = idx
	}
	list.head = idx
	if list.tail < 0 {
		list.tail = idx
	}
	list.size++
}

func (w *WTinyLFU) removeFrom(list *lruList, idx int32) {
	node := &w.pool[idx]
	if node.prev >= 0 {
		w.pool[node.prev].next = node.next
	} else {
		list.head = node.next
	}
	if node.next >= 0 {
		w.pool[node.next].prev = node.prev
	} else {
		list.tail = node.prev
	}
	node.prev = -1
	node.next = -1
	list.size--
}

func (w *WTinyLFU) moveToHead(list *lruList, idx int32) {
	w.removeFrom(list, idx)
	w.pushHead(list, idx)
}

func (w *WTinyLFU) allocNode() int32 {
	if w.freeHead < 0 {
		return -1
	}
	idx := w.freeHead
	w.freeHead = w.pool[idx].freeNext
	w.pool[idx].freeNext = -1
	w.pool[idx].inUse = true
	return idx
}

func (w *WTinyLFU) freeNode(idx int32) {
	w.pool[idx].keyHash = 0
	w.pool[idx].keyLen = 0
	w.pool[idx].inUse = false
	w.pool[idx].prev = -1
	w.pool[idx].next = -1
	w.pool[idx].freeNext = w.freeHead
	w.freeHead = idx
}
