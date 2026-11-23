---
title: "The merge loop, by hand: a naive oracle and a fast twin"
description: "The BPE merge loop written out twice: the obvious O(n squared) version kept as a naive oracle, and the fast incremental twin that actually runs."
slug: the-merge-loop-by-hand
keywords:
  - BPE algorithm explained
  - byte pair encoding implementation
  - BPE merge ranks
  - tokenizer algorithm
  - reference implementation oracle
  - linked list BPE
  - min heap merge
  - tokenizer correctness
date: 2026-11-18
---

# The merge loop, by hand: a naive oracle and a fast twin

*You are on **Level 3 of 5** of the [tokenization ladder](README.md).*

> **In one breath:** The merge loop finds the adjacent pair with the lowest rank
> and applies it, over and over. This repository keeps two versions. One is slow
> and obvious. One is fast and subtle. A test proves they agree.

**One line:** `bpeNaive` re-scans every adjacent pair each pass and is O(n squared);
`bpe` keeps symbols in a doubly linked list with a min-heap frontier and only
recomputes the two boundaries a merge touched, and the two are asserted
bit-for-bit identical on every fixture, which is what makes the fast one
trustworthy rather than merely plausible.

> **Audience.** A working engineer who can read Go and is comfortable with a linked
> list and a heap. Prerequisites: [Level 2](level-2-the-three-stages.md) (the three
> stages; the merge stage takes a spelled string and returns symbols). By the end
> you will be able to read both implementations, explain why the fast one is
> allowed to skip work, and say exactly what assumption that permission rests on.

> **Short answer.** The merge rule is one sentence: *find the adjacent pair whose
> merge has the lowest rank, apply it everywhere, repeat.* The obvious way to
> implement that re-scans the whole symbol list every pass, which is quadratic. The
> fast way keeps the symbols in a linked list and a priority queue of candidate
> merges, so applying a merge only invalidates the two boundaries it touched. The
> fast way is only correct because real merge tables are **monotone**, and the
> repository states that assumption in the code and tests it.

## The rule, stated once

Everything on this page is an implementation of one sentence:

> Repeatedly find the adjacent pair of symbols whose merge has the **lowest rank**.
> Apply that merge everywhere it occurs. Repeat until no adjacent pair has a rank.

Two details in that sentence carry all the weight.

**"Lowest rank"** means the merge learned earliest in training, which means the
most common pair. Rank 0 is tried before rank 1.

**"Everywhere it occurs"** means the pass is a batch, not a single edit. If the
lowest-rank pair appears three times in the current symbol list, all three are
merged in that pass, left to right. This is not an optimization. It is part of the
definition, and it is what makes the result deterministic.

## The naive version: the oracle

The repository keeps the obvious implementation, and it is worth reading first
because it is the definition made executable. It lives in the test file, at
`internal/tokenizer/bpe_merge_test.go:14`:

```go
// bpeNaive is the reference O(n^2) merge loop bpe replaced (#4263): every pass re-scans
// all adjacent pairs for the single lowest-rank merge, applies every non-overlapping
// occurrence left-to-right, and rebuilds the slice. It is the exact pre-change body,
// kept here as the oracle the incremental bpe must match bit-for-bit.
func bpeNaive(mergeRank map[tokenPair]int, encoded string) []string {
	var syms []string
	for _, r := range encoded {
		syms = append(syms, string(r))
	}
	if len(syms) == 0 {
		return nil
	}
	for {
		bestRank := len(mergeRank)
		best := tokenPair{}
		found := false
		for i := 0; i+1 < len(syms); i++ {
			pair := tokenPair{left: syms[i], right: syms[i+1]}
			rank, ok := mergeRank[pair]
			if ok && rank < bestRank {
				bestRank = rank
				best = pair
				found = true
			}
		}
		if !found {
			return syms
		}
		next := syms[:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == best.left && syms[i+1] == best.right {
				next = append(next, best.left+best.right)
				i++
				continue
			}
			next = append(next, syms[i])
		}
		syms = next
	}
}
```

Read it in four moves.

**Move 1: start from single characters.** The first loop turns the string into a
list of one-character symbols. This is the "every character is its own token"
starting point from Level 1.

**Move 2: find the best pair.** The inner loop walks every adjacent pair, looks
each up in the rank table, and keeps the lowest rank it finds. `bestRank` starts at
`len(mergeRank)`, which is larger than any real rank, so the first pair with a rank
wins. If no pair has a rank, `found` stays false and the function returns.

**Move 3: apply it everywhere.** The second loop rebuilds the list. Whenever it
sees the winning pair, it appends the concatenation and skips one extra position.
That `i++` inside the loop is the "everywhere" part, and it is also what makes the
pass non-overlapping: after consuming a pair, the scan resumes *after* it.

**Move 4: repeat.** The outer `for` has no condition. It only exits through the
`return` in move 2.

That is the whole algorithm, and it is obviously correct. It is also obviously
slow, and Level 4 is about exactly how slow.

## The fast version: the same answer, less work

The implementation that actually runs is `bpe` at
`internal/tokenizer/bpe_merge.go:22`. Its doc comment states the contract and the
assumption in one paragraph, and it is worth reading before the code:

```go
// bpe applies the tokenizer's merge ranks to one byte-level / metaspace-encoded piece,
// returning the merged symbols. It reproduces the naive "find the single lowest-rank
// adjacent pair, apply it everywhere, repeat" rewrite (bpeNaive in bpe_merge_test.go)
// bit-for-bit, but incrementally: symbols live in a doubly linked list and a min-heap
// frontier yields the next lowest-rank adjacent pair, so an applied merge only invalidates
// and recomputes its two neighbor boundaries instead of re-scanning every pair each pass.
// That drops the O(n^2) full-rescan cliff a long metaspace chunk hits — Encode feeds the
// whole chunk to one bpe call (tokenizer.go), so a deep indentation / code run is exactly
// the quadratic worst case. Issue #4263 / epic #3236.
```

Three claims in there, and the rest of this page checks each one:

1. It produces the **same answer** as the naive version.
2. It does so **incrementally**, touching only the two boundaries a merge changed.
3. That is only sound because real merge tables are **monotone**.

### The data structure

The symbols live in a slice of nodes, linked into a list. The node is at
`internal/tokenizer/bpe_merge.go:100`:

```go
// bpeNode is one symbol in the doubly linked list bpe merges in place. dead marks a node
// absorbed by its left neighbor; prev/next index into the node slice (-1 = none).
type bpeNode struct {
	start int
	end   int
	prev  int
	next  int
	dead  bool
}
```

Two things about this are worth noticing, because both are deliberate.

**A symbol is a span, not a string.** `start` and `end` are offsets into the
original encoded string. A merge does not build a new string by concatenation. It
extends the left node's `end` to cover the right node's span. The comment in the
code says why: "Merges extend the left span instead of allocating a new
concatenated string." For a long input with many merges, that is a lot of
allocation avoided.

**Nodes are never removed.** A merged-away node is marked `dead` and unlinked, but
it stays in the slice. That keeps every index stable, which is what lets `prev` and
`next` be plain integers instead of pointers.

### The frontier

The second structure is a priority queue of pending merges. The entry is at
`internal/tokenizer/bpe_merge.go:111`:

```go
// mergeCandidate is a pending "merge the pair whose left symbol is node pos" entry. rank is
// the merge rank captured when queued; a stale entry (its boundary since moved) is caught
// at pop time by recomputing the live pair's rank and comparing.
type mergeCandidate struct {
	rank int
	pos  int
}
```

And the queue itself, at `internal/tokenizer/bpe_merge.go:121`:

```go
// mergeHeap is a binary min-heap of mergeCandidate ordered by (rank, pos), so equal-rank
// pairs — which, because each rank maps to exactly one pair-text, are repeat occurrences of
// the same pair — drain left-to-right, matching the naive pass's left-to-right batch. It is
// hand-rolled (not container/heap) to avoid the interface{} boxing on every push/pop of
// this hot inner loop.
type mergeHeap []mergeCandidate
```

The ordering is `(rank, pos)`, and the tie-break is the interesting half. Two
candidates with the same rank are two occurrences of the *same pair text*, because
each rank maps to exactly one pair. Ordering them by position makes them drain
left to right, which is exactly what the naive version's left-to-right batch does.
**The tie-break is not a detail. It is what makes the two implementations agree.**

### The loop

Now the main loop, at `internal/tokenizer/bpe_merge.go:62`:

```go
	for len(h) > 0 {
		c := h.pop()
		left := &nodes[c.pos]
		if left.dead || left.next < 0 {
			continue // this node was absorbed, or lost its right neighbor
		}
		right := &nodes[left.next]
		rank, ok := t.mergeRank[tokenPair{left: encoded[left.start:left.end], right: encoded[right.start:right.end]}]
		if !ok || rank != c.rank {
			continue // stale: the boundary changed since this candidate was queued
		}

		// Apply the merge: left absorbs right, right leaves the list.
		left.end = right.end
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
```

This is the heart of the page. Walk it.

**Pop the best candidate.** The heap hands back the lowest-rank pending merge. If
several share a rank, the leftmost.

**Reject it if it is stale.** This is the key move, and it is two checks. If the
node is `dead`, it was already absorbed by a neighbour, so this candidate is
obsolete. If the pair's *current* rank does not equal the rank recorded when the
candidate was queued, the boundary has changed since, so the candidate is obsolete
too. Both cases `continue` — the entry is simply dropped.

**Apply the merge.** The left node absorbs the right: `left.end = right.end`
extends the span, `right.dead = true` retires the node, and the two links are
rewired so the list skips over it.

**Queue only the two boundaries that changed.** This is the whole optimization.
Merging `left` and `right` can only have changed the pair `(left.prev, left)` and
the pair `(left, left.next)`. Every other adjacent pair in the list is untouched,
so its heap entry is still valid. Two `push` calls, and the loop continues.

Compare that to the naive version, which re-scans **every** pair on **every** pass.
That difference is the entire performance story, and Level 4 measures it.

### Why skipping work is allowed

Here is the part that deserves suspicion. The fast version does not re-scan. It
assumes that the pair it just popped is genuinely the lowest-rank pair remaining.
Why is that safe?

Because real merge tables are **monotone**. The doc comment states it at
`internal/tokenizer/bpe_merge.go:15`:

> Byte-exactness rests on real BPE merge tables being MONOTONE: a table only learns
> (x,y)->xy after both x and y already exist, so any pair created by an applied merge
> has a strictly higher rank than that merge. The lowest-rank pending pairs are
> therefore always the current-rank occurrences, which the heap drains left-to-right
> (position tie-break) exactly as the naive per-pass batch does.

Unpack that. A merge table is built by a training pass that only ever learns a new
rule from symbols that already exist. So when the table says `(x,y) -> xy` at rank
`r`, both `x` and `y` were already symbols at that point. That means any pair
*created* by applying merge `r` involves the new symbol `xy`, which did not exist
before rank `r`, so its own merge rule must have a rank **strictly greater** than
`r`.

The consequence: applying a merge can never create a pair with a *lower* rank than
the merge you just applied. So the lowest-rank pair in the list can only ever be
one that already existed. The heap's next pop is therefore always the right
answer, and the two `push` calls are enough.

**This is an assumption about the data, not a theorem about the code.** A merge
table that violated monotonicity would make the fast version silently wrong. The
repository handles that the honest way: it does not argue the assumption is
unnecessary, it tests that the assumption holds on every fixture it has, so a
table that broke it would red the gate rather than quietly diverge.

## The witness: how you know the two agree

The test is `TestBPEIncrementalMatchesNaive` at
`internal/tokenizer/bpe_merge_test.go:76`. Its body is short, and the shape is the
point:

```go
	for _, tc := range cases {
		tok := &Tokenizer{mergeRank: tc.mr}
		got, err := tok.bpe(tc.encoded)
		if err != nil {
			t.Fatalf("%s: bpe: %v", tc.name, err)
		}
		want := bpeNaive(tc.mr, tc.encoded)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: incremental bpe = %v, naive = %v", tc.name, got, want)
		}
	}
```

Run both implementations on the same input. Compare the results with
`reflect.DeepEqual`. If they differ, fail and print both. That is the entire
correctness argument, and it is a good one: the fast version is trusted because a
slow version that is obviously correct agrees with it, on every case anyone
thought to write down.

The cases are worth reading, because they show what the author was worried about:

```go
	cases := []struct {
		name    string
		mr      map[tokenPair]int
		encoded string
	}{
		{"empty", small.mergeRank, ""},
		{"single", small.mergeRank, "H"},
		{"hello", small.mergeRank, byteLevelEncode("Hello")},
		{"hello-world", small.mergeRank, byteLevelEncode("Hello world")},
		{"gemma-space-b", gemma.mergeRank, metaspaceEncode(" b b b")},
		{"doubling-1", doubling.mergeRank, strings.Repeat("▁", 1)},
		{"doubling-3", doubling.mergeRank, strings.Repeat("▁", 3)},
		{"doubling-7", doubling.mergeRank, strings.Repeat("▁", 7)},
		{"doubling-64", doubling.mergeRank, strings.Repeat("▁", 64)},
		// A 4-space (metaspace) indented code block repeated — the quadratic-cliff shape.
		{"indent-block", doubling.mergeRank, strings.Repeat(strings.Repeat("▁", 4)+"defoo", 200)},
	}
```

The first four are ordinary. The last five are the interesting ones, and they come
from a fixture built specifically to stress the assumption. It is at
`internal/tokenizer/bpe_merge_test.go:56`:

```go
// doublingMergeRank is a monotone merge table (each rule's output is only consumed by a
// strictly later rule, exactly as a real HF/GGML table) that progressively pairs the
// metaspace marker: ▁+▁ -> ▁▁, ▁▁+▁▁ -> ▁▁▁▁, and so on, plus a few letter merges. A run
// of k markers therefore collapses through ~log(k) rank levels with many occurrences per
// level — the metaspace-indentation shape #4263 targets, and the naive path's worst case.
func doublingMergeRank() map[tokenPair]int {
```

Read what that table does. It pairs the marker with itself, then pairs the result
with itself, and so on: `▁+▁ → ▁▁`, then `▁▁+▁▁ → ▁▁▁▁`, then `▁▁▁▁+▁▁▁▁ → ▁▁▁▁▁▁▁▁`.
Each level doubles the length, so a run of `k` markers collapses in about `log(k)`
levels, and **every level has many occurrences at once**. That is the shape that
makes the naive version's per-pass batch expensive, and it is the shape a deeply
indented code block produces.

The `doubling-64` case is 64 markers. The `indent-block` case is a four-space
indent followed by `defoo`, repeated 200 times. Both are real inputs, not
synthetic torture tests: `Encode` hands the whole chunk to one `bpe` call, so a
long indented run is exactly what a code-heavy prompt produces.

## What the two implementations disagree about, and what they do not

It is worth being precise about the claim, because it is narrower than "the fast
one is correct."

**What is proven:** on every fixture in the test, the fast version returns exactly
the symbols the naive version returns. The fixtures cover the empty string, a
single character, ordinary words, a metaspace family, and two long runs of the
worst-case shape.

**What is assumed:** that real merge tables are monotone. The test cannot prove
this for a table it has not seen. What it can do — and does — is include a fixture
whose whole purpose is to exercise the assumption, so a table that broke it would
be caught by the existing gate rather than by a user noticing bad output.

**What is not claimed:** that the fast version is correct for a *malformed* table.
If someone hand-wrote a merge table where a later rule produced a symbol that an
earlier rule consumed, the fast version could diverge. The code says so in the doc
comment rather than pretending otherwise.

That is the honest shape of a performance optimization: a fast path, a slow oracle,
a stated assumption, and a test that would catch the assumption breaking.

## See also

- [Level 4 — Why the merge loop is the hot path](level-4-why-the-merge-loop-is-the-hot-path.md):
  how much the naive version actually costs, and how the repository measures it.
- [Level 2 — The three stages](level-2-the-three-stages.md): where the merge stage
  sits in the pipeline.
- [The inference ladder](../inference/README.md) — what happens to the ids once
  they exist.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Level 4 — Why the merge loop is the hot path](level-4-why-the-merge-loop-is-the-hot-path.md)*
