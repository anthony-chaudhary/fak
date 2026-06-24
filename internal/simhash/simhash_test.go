package simhash

import (
	"math"
	"testing"
)

// TestEmbedDeterministic — the load-bearing property: the same text always embeds
// to the exact same vector. A non-deterministic embedder would make every
// downstream similarity report irreproducible.
func TestEmbedDeterministic(t *testing.T) {
	a := Embed("delete the production database now")
	b := Embed("delete the production database now")
	if len(a) != Dim || len(b) != Dim {
		t.Fatalf("dim = %d/%d, want %d", len(a), len(b), Dim)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at bucket %d: %v != %v", i, a[i], b[i])
		}
	}
}

// TestEmbedNormalized — Embed returns unit-length vectors, so Cosine is a dot
// product and self-similarity is exactly 1.
func TestEmbedNormalized(t *testing.T) {
	v := Embed("how do I rotate the signing key")
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Fatalf("not unit length: ||v||^2 = %v", sum)
	}
	if s := Cosine(v, v); math.Abs(s-1.0) > 1e-9 {
		t.Fatalf("self-cosine = %v, want 1", s)
	}
}

// TestEmptyEmbedsZero — the empty string is the zero vector and is similar to
// nothing. A gardening skill scoring an empty query must not get a spurious match.
func TestEmptyEmbedsZero(t *testing.T) {
	v := Embed("")
	for i, x := range v {
		if x != 0 {
			t.Fatalf("empty embed nonzero at %d: %v", i, x)
		}
	}
	if s := Cosine(v, Embed("anything")); s != 0 {
		t.Fatalf("cosine with zero vector = %v, want 0", s)
	}
}

// TestNearDuplicateRanksAboveUnrelated — the whole reason simhash exists: a
// paraphrase / near-duplicate must score higher than an unrelated query, even
// though it shares few exact tokens. This is what catches "bad query" clusters that
// lexical overlap misses.
func TestNearDuplicateRanksAboveUnrelated(t *testing.T) {
	base := Embed("please refund the customer's last payment")
	nearDup := Embed("refund the customers last payment please")
	unrelated := Embed("compile the kernel and run the benchmark suite")

	simDup := Cosine(base, nearDup)
	simUnrel := Cosine(base, unrelated)
	if simDup <= simUnrel {
		t.Fatalf("near-duplicate (%.3f) did not outrank unrelated (%.3f)", simDup, simUnrel)
	}
	if simDup < 0.5 {
		t.Fatalf("near-duplicate cosine too low: %.3f", simDup)
	}
}

// TestCosineMismatchAndZero — defensive: incomparable or zero vectors yield 0, never
// a panic or NaN.
func TestCosineMismatchAndZero(t *testing.T) {
	if s := Cosine([]float32{1, 0}, []float32{1, 0, 0}); s != 0 {
		t.Fatalf("length-mismatch cosine = %v, want 0", s)
	}
	if s := Cosine(nil, nil); s != 0 {
		t.Fatalf("nil cosine = %v, want 0", s)
	}
	if s := Cosine([]float32{0, 0}, []float32{1, 1}); s != 0 {
		t.Fatalf("zero-vector cosine = %v, want 0", s)
	}
}

// TestCosineHandBuiltNormalizes — an application's own (unnormalized) embeddings are
// normalized by Cosine, so the same direction at different magnitudes is identical.
func TestCosineHandBuiltNormalizes(t *testing.T) {
	a := []float32{3, 4} // |a| = 5
	b := []float32{6, 8} // |b| = 10, same direction
	if s := Cosine(a, b); math.Abs(s-1.0) > 1e-6 {
		t.Fatalf("same-direction cosine = %v, want 1", s)
	}
}

// TestIndexTopK — the gardening hot path: TopK returns the most similar stored
// query first, ranked descending, deterministically.
func TestIndexTopK(t *testing.T) {
	var ix Index
	ix.AddText("q1", "delete all rows from the users table", "")
	ix.AddText("q2", "drop every record in the users table", "")
	ix.AddText("q3", "what is the weather in Paris today", "")
	if ix.Len() != 3 {
		t.Fatalf("len = %d, want 3", ix.Len())
	}

	q := Embed("remove all entries from the users table")
	got := ix.TopK(q, 2)
	if len(got) != 2 {
		t.Fatalf("topk len = %d, want 2", len(got))
	}
	// q1/q2 are about the same destructive action; q3 is unrelated and must not be
	// in the top 2.
	for _, m := range got {
		if m.ID == "q3" {
			t.Fatalf("unrelated q3 ranked into top 2: %+v", got)
		}
	}
	if got[0].Score < got[1].Score {
		t.Fatalf("topk not descending: %+v", got)
	}
}

// TestTopKDeterministicTieBreak — equal scores break by id, so a corpus produces a
// stable ranking a downstream diff can rely on.
func TestTopKDeterministicTieBreak(t *testing.T) {
	var ix Index
	v := Embed("identical text")
	ix.Add("b", v, "")
	ix.Add("a", v, "")
	got := ix.TopK(v, 0) // k<=0 => all
	if len(got) != 2 || got[0].ID != "a" || got[1].ID != "b" {
		t.Fatalf("tie-break not id-ascending: %+v", got)
	}
}

// TestTopKAllWhenKExceedsLen — k larger than the corpus returns the whole corpus,
// ranked, rather than truncating or padding.
func TestTopKAllWhenKExceedsLen(t *testing.T) {
	var ix Index
	ix.AddText("only", "lonely entry", "")
	if got := ix.TopK(Embed("query"), 99); len(got) != 1 {
		t.Fatalf("topk len = %d, want 1", len(got))
	}
}
