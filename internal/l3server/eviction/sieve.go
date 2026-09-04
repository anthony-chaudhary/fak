package eviction

// SIEVE implements the SIEVE eviction algorithm.
// Simple, scan-resistant. NSDI'24: "SIEVE is Simpler than LRU".
// Maintains a FIFO queue with a "hand" pointer and visited bits.
type SIEVE struct {
	capacity  uint64
	pool      []sieveEntry
	freeHead  int32
	head      int32 // queue head (newest)
	tail      int32 // queue tail (oldest)
	hand      int32 // eviction pointer
	size      uint64
	keyToNode map[uint64]int32
	onEvict   func(keyHash uint64, keyLen uint16)
}

type sieveEntry struct {
	keyHash  uint64
	keyLen   uint16
	prev     int32
	next     int32
	visited  bool
	inUse    bool
	freeNext int32
}

// NewSIEVE creates a new SIEVE eviction engine.
func NewSIEVE(capacity uint64, onEvict func(keyHash uint64, keyLen uint16)) *SIEVE {
	if capacity < 2 {
		capacity = 2
	}
	poolSize := capacity + 1
	pool := make([]sieveEntry, poolSize)
	for i := range pool {
		pool[i].freeNext = int32(i + 1)
		pool[i].prev = -1
		pool[i].next = -1
	}
	pool[len(pool)-1].freeNext = -1

	return &SIEVE{
		capacity:  capacity,
		pool:      pool,
		freeHead:  0,
		head:      -1,
		tail:      -1,
		hand:      -1,
		keyToNode: make(map[uint64]int32, capacity),
		onEvict:   onEvict,
	}
}

// Access marks a key as accessed (sets visited bit).
func (s *SIEVE) Access(keyHash uint64) bool {
	idx, exists := s.keyToNode[keyHash]
	if !exists {
		return false
	}
	s.pool[idx].visited = true
	return true
}

// Admit adds a key. Returns evicted key hashes.
func (s *SIEVE) Admit(keyHash uint64, keyLen uint16) []uint64 {
	if _, exists := s.keyToNode[keyHash]; exists {
		s.Access(keyHash)
		return nil
	}

	var evicted []uint64
	for s.size >= s.capacity {
		evHash := s.evict()
		if evHash != 0 {
			evicted = append(evicted, evHash)
		}
	}

	idx := s.sieveAllocNode()
	if idx < 0 {
		return evicted
	}
	s.pool[idx].keyHash = keyHash
	s.pool[idx].keyLen = keyLen
	s.pool[idx].visited = false
	s.pool[idx].inUse = true
	s.keyToNode[keyHash] = idx

	// Insert at head (newest)
	s.pool[idx].prev = -1
	s.pool[idx].next = s.head
	if s.head >= 0 {
		s.pool[s.head].prev = idx
	}
	s.head = idx
	if s.tail < 0 {
		s.tail = idx
	}
	s.size++

	return evicted
}

// Remove explicitly removes a key.
func (s *SIEVE) Remove(keyHash uint64) {
	idx, exists := s.keyToNode[keyHash]
	if !exists {
		return
	}
	if s.hand == idx {
		s.hand = s.pool[idx].prev
	}
	s.sieveRemoveNode(idx)
	delete(s.keyToNode, keyHash)
	s.sieveFreeNode(idx)
	s.size--
}

// Size returns the number of tracked entries.
func (s *SIEVE) Size() uint64 {
	return s.size
}

// EvictOne force-evicts a single entry. Returns false if nothing to evict.
func (s *SIEVE) EvictOne() bool {
	return s.evict() != 0
}

func (s *SIEVE) evict() uint64 {
	if s.size == 0 {
		return 0
	}

	// Start hand at tail if unset
	if s.hand < 0 {
		s.hand = s.tail
	}

	// Scan from hand toward head, looking for visited=false
	for {
		if s.hand < 0 {
			s.hand = s.tail
		}
		if s.hand < 0 {
			return 0
		}

		node := &s.pool[s.hand]
		if !node.visited {
			// Evict this one
			evicted := node.keyHash
			evictedKeyLen := node.keyLen
			prev := node.prev
			s.sieveRemoveNode(s.hand)
			delete(s.keyToNode, evicted)
			s.sieveFreeNode(s.hand)
			s.hand = prev
			s.size--
			if s.onEvict != nil {
				s.onEvict(evicted, evictedKeyLen)
			}
			return evicted
		}

		// Clear visited and advance hand
		node.visited = false
		s.hand = node.prev
	}
}

func (s *SIEVE) sieveRemoveNode(idx int32) {
	node := &s.pool[idx]
	if node.prev >= 0 {
		s.pool[node.prev].next = node.next
	} else {
		s.head = node.next
	}
	if node.next >= 0 {
		s.pool[node.next].prev = node.prev
	} else {
		s.tail = node.prev
	}
	node.prev = -1
	node.next = -1
}

func (s *SIEVE) sieveAllocNode() int32 {
	if s.freeHead < 0 {
		return -1
	}
	idx := s.freeHead
	s.freeHead = s.pool[idx].freeNext
	s.pool[idx].freeNext = -1
	return idx
}

func (s *SIEVE) sieveFreeNode(idx int32) {
	s.pool[idx].keyHash = 0
	s.pool[idx].keyLen = 0
	s.pool[idx].inUse = false
	s.pool[idx].visited = false
	s.pool[idx].prev = -1
	s.pool[idx].next = -1
	s.pool[idx].freeNext = s.freeHead
	s.freeHead = idx
}
