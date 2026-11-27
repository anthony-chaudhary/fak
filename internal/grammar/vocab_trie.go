package grammar

import (
	"bytes"
	"slices"
	"sort"
)

// vocabTrie is a sorted-vocabulary trie: token ids ordered by their bytes, so
// every prefix owns one contiguous range and a dead prefix rejects the whole
// range at once. Clean-room Go design after the sorted-vocab / LCP-rollback /
// subtree-reject technique in mlc-ai/xgrammar (Apache-2.0).
type vocabTrie struct {
	ids        []int    // token ids sorted by (bytes, id); empty tokens excluded
	pieces     [][]byte // pieces[i] is the bytes of ids[i]
	subtreeEnd []int    // subtreeEnd[i] = first j>i whose bytes lack prefix pieces[i]
	vocabSize  int
	maxLen     int
}

// newVocabTrie builds the trie over ids [0, vocabSize). The prefix test is
// bytes.HasPrefix, not a substring test: "ab","abc","acab" must not put "acab"
// inside the subtree of "ab".
func newVocabTrie(vocabSize int, tokenBytes func(int) []byte) *vocabTrie {
	t := &vocabTrie{vocabSize: vocabSize}
	for id := 0; id < vocabSize; id++ {
		if p := tokenBytes(id); len(p) > 0 {
			t.ids = append(t.ids, id)
			t.pieces = append(t.pieces, p)
		}
	}
	order := make([]int, len(t.ids))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		if c := bytes.Compare(t.pieces[a], t.pieces[b]); c != 0 {
			return c
		}
		return t.ids[a] - t.ids[b]
	})
	ids := make([]int, len(order))
	pieces := make([][]byte, len(order))
	for i, o := range order {
		ids[i], pieces[i] = t.ids[o], t.pieces[o]
		if len(pieces[i]) > t.maxLen {
			t.maxLen = len(pieces[i])
		}
	}
	t.ids, t.pieces = ids, pieces
	t.subtreeEnd = make([]int, len(ids))
	for i := range ids {
		t.subtreeEnd[i] = t.prefixEnd(i+1, pieces[i])
	}
	return t
}

// prefixEnd returns the first index j >= from whose piece lacks prefix p,
// assuming entries from `from` on with prefix p are contiguous (true in sorted
// order when from is just past p's own position or p's first extension).
func (t *vocabTrie) prefixEnd(from int, p []byte) int {
	return from + sort.Search(len(t.pieces)-from, func(k int) bool {
		return !bytes.HasPrefix(t.pieces[from+k], p)
	})
}

// walk visits every token in sorted order and calls accept(id) for each one
// whose every byte keeps the automaton alive. step receives the state at depth
// d (an opaque slot index) and writes the state after byte b into slot d+1,
// reporting whether it is still alive; slot 0 must hold the start state. The
// shared prefix with the previous token reuses slots, and a prefix that dies
// skips its whole subtree.
func (t *vocabTrie) walk(step func(depth int, b byte) bool, accept func(id int)) {
	var prev []byte
	valid := 0 // slots 1..valid hold alive states along prev
	for i := 0; i < len(t.ids); {
		piece := t.pieces[i]
		l := lcp(prev, piece)
		if l > valid {
			l = valid
		}
		dead := -1
		for d := l; d < len(piece); d++ {
			if !step(d, piece[d]) {
				dead = d
				break
			}
			valid = d + 1
		}
		if dead < 0 {
			valid = len(piece)
			prev = piece
			accept(t.ids[i])
			i++
			continue
		}
		valid = dead
		prev = piece[:dead]
		if dead == len(piece)-1 {
			i = t.subtreeEnd[i]
		} else {
			i = t.prefixEnd(i+1, piece[:dead+1])
		}
	}
}

func lcp(a, b []byte) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
