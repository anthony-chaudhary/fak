package simhash

import (
	"math"
	"sort"
	"strings"
)

// Dim is the embedding dimension. A few hundred buckets is enough resolution for
// near-duplicate detection over short queries and trajectory steps while keeping a
// Vector small (256 float32 = 1 KiB) and Cosine cheap. It is fixed so every Vector
// in an Index is comparable; do not vary it within one corpus.
const Dim = 256

// Vector is a fixed-dimension embedding. Embed returns an L2-normalized Vector, so
// Cosine reduces to a dot product; a hand-built Vector (an application's own
// embeddings) is normalized on Cosine's behalf, so callers need not pre-normalize.
type Vector = []float32

// Embed maps text to a deterministic, L2-normalized feature-hash Vector. The
// features are lower-cased word unigrams + bigrams and character 3-grams, each
// hashed (FNV-1a) into a signed bucket — the classic hashing-trick / random-sign
// sketch. Two texts that share many n-grams land on overlapping buckets with
// agreeing signs, so their Cosine is high even when the exact tokens differ.
//
// It is a pure function of the input bytes: same text -> same Vector, on any
// platform, with no model and no randomness. The empty string embeds to the zero
// Vector (Cosine with anything is 0).
func Embed(text string) Vector {
	v := make([]float32, Dim)
	lower := strings.ToLower(text)
	words := fields(lower)

	// Word unigrams and adjacent bigrams: the semantic backbone of a query.
	for i, w := range words {
		addFeature(v, "w1:"+w, 1.0)
		if i+1 < len(words) {
			addFeature(v, "w2:"+w+" "+words[i+1], 1.0)
		}
	}
	// Character 3-grams over the whole string: robustness to tokenization, typos,
	// and stemming differences that word features alone miss.
	for _, g := range charNGrams(lower, 3) {
		addFeature(v, "c3:"+g, 0.5)
	}
	normalize(v)
	return v
}

// Cosine is the cosine similarity of a and b in [-1, 1]. It returns 0 when either
// vector is the zero vector or a length mismatch makes them incomparable. Vectors
// from Embed are already normalized, so this is their dot product; an application's
// hand-built vectors are normalized here, so Cosine is correct for any input.
func Cosine(a, b Vector) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// Match is one TopK result: the stored entry's id and metadata and its cosine
// similarity to the query, in [-1, 1].
type Match struct {
	ID    string  `json:"id"`
	Meta  string  `json:"meta,omitempty"`
	Score float64 `json:"score"`
}

// entry is one stored vector in an Index.
type entry struct {
	id   string
	meta string
	vec  Vector
}

// Index is an in-memory, brute-force nearest-neighbor store: add (id, Vector, meta),
// then TopK(query, k) for the k most similar by cosine. Brute force is the right
// default here — a trajectory corpus is thousands, not millions, of rows, and a
// linear scan keeps the primitive dependency-free and deterministic. It is NOT safe
// for concurrent mutation; build it from one goroutine, then query (a corpus is
// loaded once and queried many times). The zero value is ready to use.
type Index struct {
	entries []entry
}

// Add stores a vector under id with optional metadata. A repeated id is appended,
// not replaced — the caller owns id uniqueness; TopK can legitimately return the
// same logical item twice if it was added twice.
func (ix *Index) Add(id string, v Vector, meta string) {
	ix.entries = append(ix.entries, entry{id: id, meta: meta, vec: v})
}

// AddText is the convenience that embeds text and adds it in one call.
func (ix *Index) AddText(id, text, meta string) { ix.Add(id, Embed(text), meta) }

// Len is the number of stored vectors.
func (ix *Index) Len() int { return len(ix.entries) }

// TopK returns the k entries most similar to q by cosine, descending. Ties break by
// id for a deterministic order. k <= 0 or k > Len returns all entries ranked. The
// query vector itself is never special-cased — if the query was also Added, it will
// appear with score ~1.0; a caller that wants "others like me" filters its own id.
func (ix *Index) TopK(q Vector, k int) []Match {
	out := make([]Match, 0, len(ix.entries))
	for _, e := range ix.entries {
		out = append(out, Match{ID: e.id, Meta: e.meta, Score: Cosine(q, e.vec)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ID < out[j].ID
	})
	if k > 0 && k < len(out) {
		out = out[:k]
	}
	return out
}

// addFeature hashes a feature string to a bucket and accumulates a signed weight —
// the hashing trick. A second hash bit picks the sign so collisions tend to cancel
// rather than always reinforce, which keeps the sketch unbiased.
func addFeature(v Vector, feature string, weight float32) {
	h := fnv1a(feature)
	bucket := h % uint64(Dim)
	if h&(1<<63) != 0 {
		weight = -weight
	}
	v[bucket] += weight
}

// normalize scales v to unit L2 length in place (a no-op on the zero vector), so
// Cosine over Embed outputs is a plain dot product.
func normalize(v Vector) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// fields splits on runs of non-alphanumeric runes — a deliberately simple,
// language-agnostic tokenizer matching contextq's own splitter, so simhash and the
// lexical ranker see the same word boundaries.
func fields(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
}

// charNGrams returns the sliding character n-grams of s (after collapsing
// whitespace to single spaces), the typo/stemming-robust feature family. A string
// shorter than n yields the whole string as one gram.
func charNGrams(s string, n int) []string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		if len(r) == 0 {
			return nil
		}
		return []string{string(r)}
	}
	out := make([]string, 0, len(r)-n+1)
	for i := 0; i+n <= len(r); i++ {
		out = append(out, string(r[i:i+n]))
	}
	return out
}

// fnv1a is the 64-bit FNV-1a hash used for both the bucket and the sign bit. A
// stable, well-distributed, allocation-free hash keeps Embed deterministic across
// builds and platforms.
func fnv1a(s string) uint64 {
	const off = 1469598103934665603
	const prime = 1099511628211
	h := uint64(off)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return h
}
