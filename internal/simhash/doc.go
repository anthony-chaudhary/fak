// Package simhash is fak's REFERENCE vector-similarity primitive — the
// dependency-free embedding + cosine + top-k substrate the observability layer
// hands to anyone who wants to find near-duplicate queries, cluster trajectories,
// or flag outlier ("bad") queries WITHOUT fak choosing a semantic model for them.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; it imports nothing internal (stdlib only), so it
// sits with the other zero-import foundation taps (metrics, cacheobs).
//
// WHY A REFERENCE PRIMITIVE, NOT A MODEL. Everything in fak that "finds related
// context" today ranks by LEXICAL token overlap (internal/contextq). That is exact
// and cheap but blind to near-duplicates that share no surface tokens. A learned
// sentence embedder would fix that but would drag a model, a tokenizer, and a
// non-determinism surface into a foundation leaf — exactly what the layered DAG
// forbids. simhash threads the needle: a DETERMINISTIC, hashed n-gram feature
// embedder that is good enough to catch near-duplicate queries and outlier
// trajectories, and is explicitly a REFERENCE — an application that wants real
// semantic vectors swaps its own []float32 into the same Index shape and keeps the
// cosine/top-k machinery. fak ships the data plane and the seam; the semantic layer
// is built on top.
//
// WHAT YOU GET.
//
//   - Embed(text) Vector — a fixed-dimension, L2-normalized feature hash over
//     character and word n-grams. Deterministic: the same bytes always embed to the
//     same vector, on any platform, with no model and no RNG.
//   - Cosine(a, b) — the similarity of two vectors in [-1, 1] (0 when either is the
//     zero vector). Near-1 means near-duplicate text even when the surface tokens
//     differ.
//   - Index — an in-memory store of (id, Vector, meta) with TopK(query, k): the k
//     most similar entries by cosine, descending. This is the one call a gardening
//     skill makes to ask "what past queries look like this one?".
package simhash
