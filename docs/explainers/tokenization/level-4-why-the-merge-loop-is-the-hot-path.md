---
title: "Why the merge loop is the hot path"
description: "The quadratic cliff in naive BPE, why a long indented code block is the worst case, and how the incremental frontier turns it near-linear. With the repository's own benchmarks and the caching layer above them. Level 4 of the fak tokenization ladder."
slug: why-the-merge-loop-is-the-hot-path
keywords:
  - BPE performance
  - tokenizer performance
  - quadratic algorithm
  - O(n log n) BPE
  - tokenizer benchmark
  - encode latency
  - tokenizer caching
  - prefill latency
date: 2026-11-18
---

# Why the merge loop is the hot path

*You are on **Level 4 of 5** of the [tokenization ladder](README.md).*

> **In one breath:** The naive merge loop re-scans every pair on every pass. That
> makes it quadratic in the length of the piece. A long indented code block is the
> worst case. The incremental version avoids the rescan and grows near-linearly.

**One line:** `bpeNaive` costs O(n squared) because each of its O(n) passes rescans
O(n) pairs, and `Encode` hands a whole pre-tokenized piece to one `bpe` call, so a
deeply indented code run is exactly the quadratic worst case; the incremental
frontier replaces the rescan with a heap pop and two pushes, giving ~O(n log n),
and `EncodeMemo` sits above both so a repeated prompt pays the cost once.

> **Audience.** A performance engineer, or anyone who has ever wondered why a
> prompt with a big code block feels slower than its token count suggests.
> Prerequisites: [Level 3](level-3-the-merge-loop-by-hand.md) (both
> implementations, and the monotone assumption that licenses the fast one). By the
> end you will be able to say where the time goes, why the input shape matters more
> than the input size, and how to measure it yourself.

> **Short answer.** The naive loop is quadratic: it makes one pass per merge, and
> each pass walks the whole symbol list. The fast loop is near-linear: it keeps a
> priority queue of candidate merges and only re-examines the two boundaries a
> merge touched. The input that separates them is not a long sentence. It is a long
> run of one repeated pattern, which is what indentation in code produces.

## Where the time actually goes

Start with the shape of the cost, because it is the whole point.

The naive loop at `internal/tokenizer/bpe_merge_test.go:14` has two nested loops.
The outer loop runs once per merge applied. The inner loop walks every adjacent
pair in the current symbol list. So if the input has `n` symbols and needs `m`
merges, the cost is roughly `n × m`.

For a piece of ordinary English, `m` is much smaller than `n`, because most merges
collapse several symbols at once and the list shrinks fast. That is why the naive
version survived as long as it did. It is not slow on the inputs people usually
test with.

The problem is the input where `m` is *not* much smaller than `n`. Then the cost is
`n × n`, and it is quadratic.

## The input that breaks it

The doc comment on the fast version names the shape at
`internal/tokenizer/bpe_merge.go:11`:

> That drops the O(n^2) full-rescan cliff a long metaspace chunk hits — Encode feeds
> the whole chunk to one bpe call (tokenizer.go), so a deep indentation / code run is
> exactly the quadratic worst case.

Two facts combine here, and both are worth stating separately.

**Fact one: the whole piece goes to one call.** The split stage from Level 2 cuts
the text into pieces, and each piece is merged independently. For a GPT-2-style
tokenizer, a piece is roughly a word. But for a **metaspace** tokenizer, the split
is much coarser, and a long run of indentation can arrive as a single piece. So the
`n` in the cost formula is not "the length of a word." It is "the length of the
piece," and a piece can be long.

**Fact two: indentation is a repeated pattern.** A four-space indent is the same
four characters over and over. A merge table that has learned to pair the
indentation marker with itself will collapse that run through about `log(k)` levels
for `k` markers, and **every level has many occurrences at once**. That is the
shape that makes the naive per-pass batch expensive: each pass has real work to do,
and there are many passes.

Put the two facts together and you get the cliff. A prompt with a deeply indented
code block is not just a long prompt. It is a prompt whose tokenizer input has the
one shape the naive loop handles worst.

## The fixture that reproduces it

The repository does not describe this shape in prose and hope you believe it. It
builds the shape and measures it. The fixture is `doublingMergeRank` at
`internal/tokenizer/bpe_merge_test.go:56`, and its doc comment says exactly what it
is for:

> `doublingMergeRank` is a monotone merge table (each rule's output is only consumed
> by a strictly later rule, exactly as a real HF/GGML table) that progressively pairs
> the metaspace marker: `▁+▁ -> ▁▁`, `▁▁+▁▁ -> ▁▁▁▁`, and so on, plus a few letter
> merges. A run of k markers therefore collapses through ~log(k) rank levels with
> many occurrences per level — the metaspace-indentation shape #4263 targets, and the
> naive path's worst case.

Read the doubling. Level 1 pairs single markers. Level 2 pairs the pairs. Level 3
pairs the quadruples. Each level doubles the run length, so `k` markers collapse in
about `log(k)` levels — and at each level, the number of occurrences is still large.
That is a table engineered to make the naive loop do maximum work per pass for
maximum number of passes.

The input generator is `benchInput` at `internal/tokenizer/bpe_merge_test.go:129`:

```go
// benchInput is a metaspace-indented code block repeated to reach the requested marker
// length, the long single-chunk input Encode hands bpe for a Gemma-style tokenizer.
func benchInput(markers int) string {
	block := strings.Repeat("▁", 4) + "defoo"
	var b strings.Builder
	for b.Len() < markers*3 { // ▁ is 3 bytes in UTF-8
		b.WriteString(block)
	}
	return b.String()
}
```

A four-marker indent, then `defoo`, repeated until the string is long enough. That
is a code block. Not a synthetic torture test — the thing you paste into a chat
window when you ask about a function.

## The measurement

The two benchmarks are at `internal/tokenizer/bpe_merge_test.go:141` and `:155`, and
the comment above them states the claim:

```go
// BenchmarkBPEIncremental vs BenchmarkBPENaive show the O(n^2) -> ~O(n log n) improvement:
// the naive ns/op grows ~quadratically with input length while the incremental frontier
// grows near-linearly. Run: go test ./internal/tokenizer -run=X -bench=BPE -benchmem.
func BenchmarkBPEIncremental(b *testing.B) {
	tok := &Tokenizer{mergeRank: doublingMergeRank()}
	for _, n := range []int{64, 256, 1024, 4096} {
		in := benchInput(n)
		b.Run(sizeLabel(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := tok.bpe(in); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
```

Four sizes: 64, 256, 1024, and 4096 markers. Each is 4× the previous, so the
sequence is a clean geometric ladder. That is deliberate, and it is the right way to
measure an exponent: if the cost is quadratic, each 4× step should cost about 16×
more. If it is near-linear, each step should cost about 4× more.

The naive benchmark is the same shape, calling `bpeNaive` instead. Both use the same
fixture and the same inputs, so the comparison is apples to apples.

**Run it yourself.** From the repository root:

```sh
go test ./internal/tokenizer -run=X -bench=BPE -benchmem
```

The `-run=X` is not a typo. It matches no test, which skips the test phase and runs
only the benchmarks. `-benchmem` adds allocation counts, which matter here because
the naive version rebuilds its symbol slice every pass while the fast version
extends spans in place.

**A caution about reading the numbers.** Benchmark results on a shared machine are
noisy, and this repository's own notes record a host where a peer process fleet
starved Go test binaries down to a single thread. If the two benchmarks come out
closer than the shape predicts, suspect the host before suspecting the code. The
*shape* — quadratic versus near-linear across a 4× ladder — is the robust signal.
The absolute nanoseconds are not.

## Why the fast version is near-linear

The mechanism is in Level 3, but the cost argument is worth restating in one place.

The naive loop, per pass: walk all `n` pairs, then rebuild the list. Cost per pass
`O(n)`. Number of passes `O(n)`. Total `O(n²)`.

The fast loop, per merge: pop one candidate from the heap, do a constant-time rank
check, apply the merge, and push at most two new candidates. Cost per merge
`O(log n)` for the heap operations. Number of merges `O(n)`. Total `O(n log n)`.

The difference is not a constant factor. It is the difference between "each merge
re-examines the whole list" and "each merge re-examines two boundaries." On a
4096-marker input, that is the difference between roughly 16 million pair
examinations and roughly 4096 heap operations.

There is a second saving that does not show up in the exponent but does show up in
`-benchmem`. The naive version allocates a new symbol slice every pass, and builds
new strings by concatenation. The fast version extends a span in place, as the
comment at `internal/tokenizer/bpe_merge.go:33` notes: "Merges extend the left span
instead of allocating a new concatenated string." Fewer allocations means less
garbage-collector pressure, which matters in a process that is also holding model
weights.

## The layer above: don't do it twice

Even a near-linear merge loop is work, and the same text gets encoded over and over.
A chat session re-sends the whole conversation on every turn. A system prompt is
identical across thousands of requests. So the repository caches encodings.

The cache is `EncodeMemo` at `internal/tokenizer/encode_memo.go:23`, and its entry
point is `Encode` at `internal/tokenizer/encode_memo.go:46`. The design is worth
knowing because it is deliberately bounded:

- It holds **immutable** encodings, so a cached result can be shared without
  copying.
- It is capped at **1 MiB and 1024 entries**, so a long-running process cannot grow
  it without limit.
- `canRetain` at `internal/tokenizer/encode_memo.go:121` decides whether a result is
  worth keeping, and `EncodeTransient` at `internal/tokenizer/encode_memo.go:50`
  exists for the case where it is not.

The layering is the point. The merge loop is the hot path *within* one encode. The
memo is what stops the same encode from happening twice. A performance engineer
looking at tokenization cost should check both, in that order: first, is the input
shape hitting the quadratic cliff; second, is the result being cached at all.

## What this rung is not claiming

Three honest limits.

**The benchmark is a microbenchmark.** It measures `bpe` in isolation, on a
synthetic input, with no model running. Real encode cost includes the split, the
spell, the lookup, and whatever else the process is doing. The microbenchmark
establishes the *shape* of the merge cost, not the end-to-end latency of a request.

**The numbers are host-dependent.** As above. The ladder's claim is about the
exponent, and the exponent is what the 4× ladder is designed to reveal.

**Near-linear is not linear.** `O(n log n)` still grows. A sufficiently long piece
is still expensive. The fix for that is not a better merge loop; it is not sending
one enormous piece, or caching the result. Both of those are decisions above this
layer.

## See also

- [Level 3 — The merge loop, by hand](level-3-the-merge-loop-by-hand.md): the two
  implementations, and the monotone assumption that makes the fast one correct.
- [Level 5 — The tokenizer frontier](level-5-the-tokenizer-frontier.md): how
  correctness is proven against real tokenizers, and what is still open.
- [The caching ladder](../caching/README.md) — the same cost reasoning applied to
  the key/value cache, one layer down the pipe.
- [The inference ladder](../inference/README.md) — where encode cost shows up as
  prefill latency.

---

*[Full ladder (README)](README.md) · next → [Level 5 — The tokenizer frontier](level-5-the-tokenizer-frontier.md)*
