package simhash

import (
	"math"
	"math/bits"
	"sort"
	"strings"
)

// Dim is the fixed embedding dimension (256 float32 buckets) for vector comparability.
const Dim = 256

// Vector represents a fixed-dimension float32 embedding slice.
type Vector = []float32

// Embed maps text to a deterministic L2-normalized feature-hash Vector over word and char n-grams.
func Embed(text string) Vector {
	v := make([]float32, Dim)
	lower := strings.ToLower(text)
	words := fields(lower)

	for i, w := range words {
		addFeature(v, "w1:"+w, 1.0)
		if i+1 < len(words) {
			addFeature(v, "w2:"+w+" "+words[i+1], 1.0)
		}
	}
	for _, g := range charNGrams(lower, 3) {
		addFeature(v, "c3:"+g, 0.5)
	}
	normalize(v)
	return v
}

// Simhash computes the deterministic feature-hash vector for input text.
func Simhash(text string) Vector {
	return Embed(text)
}

// VectorSimhash generates the normalized feature-hash vector representation for text.
func VectorSimhash(text string) Vector {
	return Embed(text)
}

// HammingDistance calculates the bitwise difference count between two 64-bit fingerprint words.
func HammingDistance(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}

// Cosine computes the cosine similarity between two vectors in [-1, 1], returning 0 on mismatch or zero norm.
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

// Match contains a retrieved item ID, metadata, and similarity score from TopK search.
type Match struct {
	ID    string  `json:"id"`
	Meta  string  `json:"meta,omitempty"`
	Score float64 `json:"score"`
}

type entry struct {
	id   string
	meta string
	vec  Vector
}

// Index stores vectors in memory for nearest-neighbor similarity search.
type Index struct {
	entries []entry
}

// Add stores or replaces a vector under id with associated metadata.
func (ix *Index) Add(id string, v Vector, meta string) {
	for i := range ix.entries {
		if ix.entries[i].id == id {
			ix.entries[i] = entry{id: id, meta: meta, vec: v}
			return
		}
	}
	ix.entries = append(ix.entries, entry{id: id, meta: meta, vec: v})
}

// AddText embeds raw text and inserts the resulting vector into the index.
func (ix *Index) AddText(id, text, meta string) { ix.Add(id, Embed(text), meta) }

// Len returns the count of indexed vector entries.
func (ix *Index) Len() int { return len(ix.entries) }

// TopK retrieves the k most similar entries to query vector q ordered by descending cosine similarity.
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

func addFeature(v Vector, feature string, weight float32) {
	h := fnv1a(feature)
	bucket := h % uint64(Dim)
	if h&(1<<63) != 0 {
		weight = -weight
	}
	v[bucket] += weight
}

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

func fields(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
}

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
