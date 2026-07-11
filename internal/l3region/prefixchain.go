package l3region

import (
	"crypto/sha256"
	"encoding/hex"
)

// ---------------------------------------------------------------------------
// Prefix-hash-chain keying — exact OFF-TREE prefix-hit length by key comparison
// (#3378; parent epic #3366). Clean-room technique borrow from LMCache
// v1/token_database.py @ aaf7c0d3 (borrow id K1-prefix-hash-chain): technique
// only, original Go.
//
// POLARITY. chunk() keys a page by sha256 of its OWN bytes only, so identical
// pages at different positions dedup to one slot — position-INDEPENDENT keys,
// exactly what the flat L3 tier wants for storage dedup. But that polarity means
// the external tier structurally cannot answer "how many leading chunks of THIS
// sequence do you already hold?" from keys alone: a hit on a position-independent
// key says nothing about what preceded it. Today the exact prefix-hit length
// lives only in-process (radixkv walks raw token runs — MatchLen).
//
// THE CHAIN. A prefix-hash chain flips the polarity for the MATCH question
// without touching the storage keys: each chunk's chain key folds the previous
// chain key with the chunk's own content,
//
//	key[0] = H(seed      || chunk_0)
//	key[i] = H(key[i-1]  || chunk_i)        (H = sha256, hex-encoded)
//
// so key[i] commits to the ENTIRE prefix chunk_0..chunk_i. Two chained-key
// sequences are therefore equal at index i iff their first i+1 chunks are equal
// (up to sha256 collision), and the exact off-tree prefix-hit length is simply
// the count of leading equal keys — pure key comparison, no raw-token re-walk,
// no whole-tree structure shipped across the process boundary.
//
// FRAMING is unambiguous: the folded prev key is a fixed-length 64-char lowercase
// hex string (or, on the first rung, the seed — which is not hex-shaped, so the
// two rungs cannot alias), meaning prev||chunk cannot be re-sliced into a
// different (prev, chunk) pair. DETERMINISM: sha256 only, no randomized seed —
// the same chunk sequence yields the same chain in every process, which is what
// lets two processes compare chains at all. Chains are only comparable when
// built over the SAME chunk boundaries (e.g. both via RegionPrefixChainKeys'
// PageBytes pages).
// ---------------------------------------------------------------------------

// prefixChainSeed stands in for key[-1] on the first rung. It domain-separates
// the chain-key space from the flat content-address space: key[0] is
// sha256(seed||chunk_0), never sha256(chunk_0) = digest(chunk_0), so a chained
// key can never collide with (or be mistaken for) a position-independent page
// key in the same store.
const prefixChainSeed = "l3region/prefix-chain/v1"

// PrefixChainKeys derives the prefix-hash-chain keys for an ordered chunk
// sequence: keys[i] = sha256(keys[i-1] || chunks[i]) hex-encoded, with the fixed
// prefixChainSeed standing in for keys[-1] (see the banner above). keys[i]
// ENCODES the exact prefix chunks[0..i], so identical chunk bytes at different
// positions — or after different prefixes — get DISTINCT keys: the opposite
// polarity of chunk()'s position-independent content addresses. Pure and
// deterministic; a nil/empty sequence yields a nil chain. A nil chunk element
// folds as zero bytes (identical to an empty chunk).
func PrefixChainKeys(chunks [][]byte) []string {
	if len(chunks) == 0 {
		return nil
	}
	keys := make([]string, 0, len(chunks))
	prev := prefixChainSeed
	for _, c := range chunks {
		h := sha256.New()
		h.Write([]byte(prev))
		h.Write(c)
		prev = hex.EncodeToString(h.Sum(nil))
		keys = append(keys, prev)
	}
	return keys
}

// PrefixChainMatchLen returns the exact number of shared leading chunks between
// two prefix-key chains — the off-tree prefix-hit length in chunks. It stops at
// the first index where the chained keys differ; because each key commits to its
// whole prefix, leading key equality IS leading chunk-sequence equality (up to
// sha256 collision). Pure; nil/empty chains match 0. Only chains built over the
// same chunk boundaries are comparable.
func PrefixChainMatchLen(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// RegionPrefixChainKeys chunks a region into PageBytes pages on the SAME
// boundaries as chunk() and returns the pages' prefix-hash-chain keys, so two
// regions' chains are directly comparable with PrefixChainMatchLen: the result
// is the exact number of leading pages the regions share. A zero-length region
// is a nil chain.
func RegionPrefixChainKeys(b []byte) []string {
	if len(b) == 0 {
		return nil
	}
	chunks := make([][]byte, 0, (len(b)+PageBytes-1)/PageBytes)
	for off := 0; off < len(b); off += PageBytes {
		end := off + PageBytes
		if end > len(b) {
			end = len(b)
		}
		chunks = append(chunks, b[off:end])
	}
	return PrefixChainKeys(chunks)
}
