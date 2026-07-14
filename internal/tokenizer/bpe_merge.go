package tokenizer

// bpe applies the tokenizer's merge ranks to one byte-level / metaspace-encoded piece,
// returning the merged symbols. It reproduces the naive "find the single lowest-rank
// adjacent pair, apply it everywhere, repeat" rewrite (bpeNaive in bpe_merge_test.go)
// bit-for-bit, but incrementally: symbols live in a doubly linked list and a min-heap
// frontier yields the next lowest-rank adjacent pair, so an applied merge only invalidates
// and recomputes its two neighbor boundaries instead of re-scanning every pair each pass.
// That drops the O(n^2) full-rescan cliff a long metaspace chunk hits — Encode feeds the
// whole chunk to one bpe call (tokenizer.go), so a deep indentation / code run is exactly
// the quadratic worst case. Issue #4263 / epic #3236.
//
// Byte-exactness rests on real BPE merge tables being MONOTONE: a table only learns
// (x,y)->xy after both x and y already exist, so any pair created by an applied merge has a
// strictly higher rank than that merge. The lowest-rank pending pairs are therefore always
// the current-rank occurrences, which the heap drains left-to-right (position tie-break)
// exactly as the naive per-pass batch does. bpe_merge_test.go asserts the two paths are
// identical on every fixture, including a long metaspace run, so any table that broke the
// assumption would red the gate rather than silently diverge.
func (t *Tokenizer) bpe(encoded string) ([]string, error) {
	if len(encoded) == 0 {
		return nil, nil
	}

	// One node per rune, linked in order. cap by byte length (>= rune count).
	nodes := make([]bpeNode, 0, len(encoded))
	for _, r := range encoded {
		nodes = append(nodes, bpeNode{text: string(r)})
	}
	n := len(nodes)
	for i := range nodes {
		nodes[i].prev = i - 1
		nodes[i].next = i + 1
	}
	nodes[n-1].next = -1

	// push queues the merge whose LEFT symbol is node pos, if that boundary is a merge.
	h := make(mergeHeap, 0, n)
	push := func(pos int) {
		nx := nodes[pos].next
		if nx < 0 {
			return
		}
		if rank, ok := t.mergeRank[tokenPair{left: nodes[pos].text, right: nodes[nx].text}]; ok {
			h.push(mergeCandidate{rank: rank, pos: pos})
		}
	}
	for i := 0; i < n; i++ {
		push(i)
	}

	for len(h) > 0 {
		c := h.pop()
		left := &nodes[c.pos]
		if left.dead || left.next < 0 {
			continue // this node was absorbed, or lost its right neighbor
		}
		right := &nodes[left.next]
		rank, ok := t.mergeRank[tokenPair{left: left.text, right: right.text}]
		if !ok || rank != c.rank {
			continue // stale: the boundary changed since this candidate was queued
		}

		// Apply the merge: left absorbs right, right leaves the list.
		left.text += right.text
		right.dead = true
		left.next = right.next
		if right.next >= 0 {
			nodes[right.next].prev = c.pos
		}

		// Neighbor-only invalidation: only the two boundaries touching the merged node
		// can have changed. Their stale heap entries survive but are rejected by the
		// rank check above; the fresh best merge for each is queued here.
		if left.prev >= 0 {
			push(left.prev)
		}
		push(c.pos)
	}

	syms := make([]string, 0, n)
	for i := 0; i >= 0; i = nodes[i].next { // node 0 is never a right operand, so it heads the list
		syms = append(syms, nodes[i].text)
	}
	return syms, nil
}

// bpeNode is one symbol in the doubly linked list bpe merges in place. dead marks a node
// absorbed by its left neighbor; prev/next index into the node slice (-1 = none).
type bpeNode struct {
	text string
	prev int
	next int
	dead bool
}

// mergeCandidate is a pending "merge the pair whose left symbol is node pos" entry. rank is
// the merge rank captured when queued; a stale entry (its boundary since moved) is caught
// at pop time by recomputing the live pair's rank and comparing.
type mergeCandidate struct {
	rank int
	pos  int
}

// mergeHeap is a binary min-heap of mergeCandidate ordered by (rank, pos), so equal-rank
// pairs — which, because each rank maps to exactly one pair-text, are repeat occurrences of
// the same pair — drain left-to-right, matching the naive pass's left-to-right batch. It is
// hand-rolled (not container/heap) to avoid the interface{} boxing on every push/pop of
// this hot inner loop.
type mergeHeap []mergeCandidate

func (h mergeHeap) less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos
}

func (h *mergeHeap) push(c mergeCandidate) {
	*h = append(*h, c)
	hp := *h
	i := len(hp) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !hp.less(i, parent) {
			break
		}
		hp[i], hp[parent] = hp[parent], hp[i]
		i = parent
	}
}

func (h *mergeHeap) pop() mergeCandidate {
	hp := *h
	n := len(hp)
	top := hp[0]
	hp[0] = hp[n-1]
	hp = hp[:n-1]
	*h = hp
	i := 0
	for {
		l := 2*i + 1
		if l >= len(hp) {
			break
		}
		child := l
		if r := l + 1; r < len(hp) && hp.less(r, l) {
			child = r
		}
		if !hp.less(child, i) {
			break
		}
		hp[i], hp[child] = hp[child], hp[i]
		i = child
	}
	return top
}
