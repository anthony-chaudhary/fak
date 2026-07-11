package l3region

import (
	"bytes"
	"testing"
)

// chainChunks builds n small deterministic pairwise-distinct chunks.
func chainChunks(n int) [][]byte {
	chunks := make([][]byte, n)
	for i := range chunks {
		chunks[i] = []byte{byte(i), byte(i >> 8), 0xA5, byte(i * 7)}
	}
	return chunks
}

// TestPrefixChainIdenticalSequencesMatchFull: two chains over identical chunk
// sequences agree at every rung, so the off-tree prefix-hit length is the full
// sequence length.
func TestPrefixChainIdenticalSequencesMatchFull(t *testing.T) {
	chunks := chainChunks(8)
	a := PrefixChainKeys(chunks)
	b := PrefixChainKeys(chunks)
	if len(a) != len(chunks) {
		t.Fatalf("chain length = %d, want %d", len(a), len(chunks))
	}
	if got := PrefixChainMatchLen(a, b); got != len(chunks) {
		t.Fatalf("MatchLen(identical) = %d, want %d", got, len(chunks))
	}
}

// TestPrefixChainDivergenceAtK: for every k, sequences identical up to chunk k
// and differing AT chunk k share exactly k leading chained keys — and every key
// from k onward differs (the chain re-keys the whole downstream once the prefix
// diverges).
func TestPrefixChainDivergenceAtK(t *testing.T) {
	const n = 6
	base := chainChunks(n)
	for k := 0; k < n; k++ {
		other := make([][]byte, n)
		copy(other, base)
		other[k] = []byte{0xFF, 0xEE, byte(k)} // diverge at chunk k only
		a := PrefixChainKeys(base)
		b := PrefixChainKeys(other)
		if got := PrefixChainMatchLen(a, b); got != k {
			t.Fatalf("diverge at k=%d: MatchLen = %d, want %d", k, got, k)
		}
		for i := k; i < n; i++ {
			if a[i] == b[i] {
				t.Fatalf("diverge at k=%d: key[%d] equal across chains; downstream keys must re-key", k, i)
			}
		}
	}
}

// TestPrefixChainPrefixSensitivity: changing chunk 0 changes ALL downstream keys,
// so a shared SUFFIX under a different prefix yields match length 0 — the
// position-binding polarity the flat content-address key lacks.
func TestPrefixChainPrefixSensitivity(t *testing.T) {
	chunks := chainChunks(5)
	other := make([][]byte, len(chunks))
	copy(other, chunks)
	other[0] = []byte{0xDE, 0xAD} // chunks 1..4 stay a shared suffix
	a := PrefixChainKeys(chunks)
	b := PrefixChainKeys(other)
	if got := PrefixChainMatchLen(a, b); got != 0 {
		t.Fatalf("shared-suffix/different-prefix MatchLen = %d, want 0", got)
	}
	for i := range a {
		if a[i] == b[i] {
			t.Fatalf("key[%d] equal despite differing chunk 0; every downstream key must change", i)
		}
	}
}

// TestPrefixChainPositionBinding: identical chunk BYTES get distinct chained keys
// at every position (contrast: chunk()'s digest keys would dedup them to one), and
// the same chunk after prefix A vs prefix B gets distinct keys — the issue's
// position-binding proof. The first rung is also domain-separated from the flat
// content-address space (key[0] != digest(chunk_0)).
func TestPrefixChainPositionBinding(t *testing.T) {
	x := []byte{1, 2, 3}
	keys := PrefixChainKeys([][]byte{x, x, x})
	if keys[0] == keys[1] || keys[1] == keys[2] || keys[0] == keys[2] {
		t.Fatalf("identical chunk bytes deduped across positions: %v", keys)
	}
	if keys[0] == digest(x) {
		t.Fatalf("key[0] collides with the position-independent content address %s", digest(x))
	}
	afterA := PrefixChainKeys([][]byte{{0xAA}, x})
	afterB := PrefixChainKeys([][]byte{{0xBB}, x})
	if afterA[1] == afterB[1] {
		t.Fatalf("identical chunk after prefix A vs B got the SAME key %s; the chain must bind the prefix", afterA[1])
	}
}

// TestPrefixChainDeterministic: the same input yields byte-identical chains across
// calls (no randomized seed), which is what makes chains comparable across
// processes at all.
func TestPrefixChainDeterministic(t *testing.T) {
	chunks := chainChunks(7)
	a := PrefixChainKeys(chunks)
	b := PrefixChainKeys(chunks)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("key[%d] differs across calls: %s vs %s", i, a[i], b[i])
		}
	}
	r := payload(PageBytes + 100)
	ra := RegionPrefixChainKeys(r)
	rb := RegionPrefixChainKeys(r)
	for i := range ra {
		if ra[i] != rb[i] {
			t.Fatalf("region key[%d] differs across calls: %s vs %s", i, ra[i], rb[i])
		}
	}
}

// TestPrefixChainNilEmptySafe: nil/empty inputs are valid zero-value answers, not
// panics — a nil chunk sequence is a nil chain, and any match against an empty
// chain is 0.
func TestPrefixChainNilEmptySafe(t *testing.T) {
	if got := PrefixChainKeys(nil); got != nil {
		t.Fatalf("PrefixChainKeys(nil) = %v, want nil", got)
	}
	if got := PrefixChainKeys([][]byte{}); got != nil {
		t.Fatalf("PrefixChainKeys(empty) = %v, want nil", got)
	}
	if got := RegionPrefixChainKeys(nil); got != nil {
		t.Fatalf("RegionPrefixChainKeys(nil) = %v, want nil", got)
	}
	keys := PrefixChainKeys(chainChunks(3))
	if got := PrefixChainMatchLen(nil, nil); got != 0 {
		t.Fatalf("MatchLen(nil, nil) = %d, want 0", got)
	}
	if got := PrefixChainMatchLen(nil, keys); got != 0 {
		t.Fatalf("MatchLen(nil, keys) = %d, want 0", got)
	}
	if got := PrefixChainMatchLen(keys, keys[:1]); got != 1 {
		t.Fatalf("MatchLen(keys, keys[:1]) = %d, want 1 (shorter chain bounds the hit)", got)
	}
	// A nil chunk element folds as zero bytes, same as an empty element.
	withNil := PrefixChainKeys([][]byte{nil, {1}})
	withEmpty := PrefixChainKeys([][]byte{{}, {1}})
	if withNil[1] != withEmpty[1] {
		t.Fatalf("nil vs empty chunk element diverged: %s vs %s", withNil[1], withEmpty[1])
	}
}

// TestRegionPrefixChainExactPageHitLength: the region-level chains ride chunk()'s
// exact PageBytes boundaries, so comparing two regions' chains reports the exact
// number of leading PAGES they share — the off-tree prefix-hit length the flat
// tier could not previously answer from keys.
func TestRegionPrefixChainExactPageHitLength(t *testing.T) {
	shared := payload(PageBytes * 2) // two full shared pages
	tailA := bytes.Repeat([]byte{0x11}, PageBytes/2)
	tailB := bytes.Repeat([]byte{0x22}, PageBytes/2)
	regionA := append(append([]byte(nil), shared...), tailA...)
	regionB := append(append([]byte(nil), shared...), tailB...)

	a := RegionPrefixChainKeys(regionA)
	b := RegionPrefixChainKeys(regionB)
	if len(a) != 3 || len(b) != 3 {
		t.Fatalf("chain lengths = %d, %d; want 3 pages each (2 full + partial tail)", len(a), len(b))
	}
	if got := PrefixChainMatchLen(a, b); got != 2 {
		t.Fatalf("regions share exactly 2 leading pages but MatchLen = %d", got)
	}
	// Chain length matches chunk()'s page count for the same bytes (same boundaries).
	if _, keys := chunk(regionA); len(keys) != len(a) {
		t.Fatalf("chain has %d rungs but chunk() makes %d pages; boundaries must agree", len(a), len(keys))
	}
	// Identical regions match in full, including the partial tail page.
	if got := PrefixChainMatchLen(a, RegionPrefixChainKeys(regionA)); got != 3 {
		t.Fatalf("identical region MatchLen = %d, want 3", got)
	}
}
