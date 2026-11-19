---
title: "The three stages: split, spell, merge"
description: "The whole tokenization pipeline, end to end, with a worked example you can follow by hand: pre-tokenize the text, rewrite each piece as printable bytes, then merge by rank. Level 2 of the fak tokenization ladder."
slug: the-three-stages
keywords:
  - tokenization pipeline
  - pre-tokenizer explained
  - byte level encoding
  - BPE merge ranks
  - how does BPE work
  - tokenizer stages
  - text to token ids
  - tokenizer walkthrough
date: 2026-11-18
---

# The three stages: split, spell, merge

*You are on **Level 2 of 5** of the [tokenization ladder](README.md).*

> **In one breath:** Tokenization is three stages in a fixed order. First split
> the text into pieces. Then rewrite each piece as printable stand-ins for its
> bytes. Then merge neighbouring symbols by a rank table and look up the ids.

**One line:** The pipeline is `Encode → pre-tokenize (split) → byteLevelEncode or
metaspaceEncode (spell) → bpe (merge) → tokenToID (look up)`, and the three stages
are separable: the split decides *where* merges may cross, the spell makes every
byte representable, and the merge is the only stage with an algorithm in it.

> **Audience.** A new grad who can read Go or Python and knows what a function, a
> map, and an array are. Prerequisites: [Level 1](level-1-what-is-a-tokenizer.md)
> (a tokenizer is a learned dictionary plus a recipe; the recipe is an ordered list
> of merges). By the end you will be able to trace a string through all three
> stages by hand and name the function that does each one.

> **Short answer.** Three moves, in this order. **Split** the text into pieces with
> a rule that depends on the model family. **Spell** each piece by rewriting every
> byte as a printable character, so nothing is unrepresentable. **Merge** adjacent
> symbols using the learned rank table, lowest rank first. Then look each surviving
> symbol up in the vocabulary to get its number. Every move is one function in this
> repository.

## What does the whole pipeline look like?

Here is the entire machine on one screen. Read it top to bottom; each box is a
function you will meet below.

```text
  input text  "Hello world"
       │
       ▼
   Encode ─────────────────────────────────  internal/tokenizer/tokenizer.go:356
       │
       │  stage 1: SPLIT
       ▼
   preTokenizeByteLevel ───────────────────  internal/tokenizer/tokenizer.go:692
       │
       ▼
   pieces  ["Hello", " world"]
       │
       │  stage 2: SPELL
       ▼
   byteLevelEncode ────────────────────────  internal/tokenizer/tokenizer.go:679
       │
       ▼
   encoded  "Hello"  and  "Ġworld"
       │
       │  stage 3: MERGE
       ▼
   bpe ────────────────────────────────────  internal/tokenizer/bpe_merge.go:22
       │
       ▼
   symbols  ["Hello"]  and  ["Ġ", "world"]
       │
       │  look up
       ▼
   tokenToID ──────────────────────────────  internal/tokenizer/tokenizer.go:504
       │
       ▼
   ids  [9, 18]
```

Three stages, then a lookup. The rest of this page walks each one.

## Stage 1: split the text into pieces

The first stage is called **pre-tokenization**, and its job is to cut the text
into pieces that will be merged *independently*. A merge is never allowed to cross
a piece boundary.

Why does that matter? Because it is how the tokenizer keeps a space attached to
the word that follows it, and how it keeps digits separate from letters. If you
merged across the whole string with no boundaries, `"abc 123"` could collapse into
one token, and the model would lose the fact that there was a space and a number
in there.

The split rule is **not universal**. Different model families split differently,
and this is a real source of bugs. The repository keeps the whole set of rules in
one closed list, at `internal/tokenizer/pretok_dispatch.go:16`:

```go
const (
	// preTokByteLevel is the GPT-2 ByteLevel/Digits default (GPT-2, SmolLM2, bare-BPE).
	preTokByteLevel preTokKind = iota
	// preTokQwen is the explicit Split-regex family (Qwen2.5/Qwen3.6): one digit at a time.
	preTokQwen
	// preTokGLM4 is GLM-4/ChatGLM4: Qwen's split with digits grouped in runs of 1-3.
	preTokGLM4
	// preTokMetaspace is the SentencePiece metaspace BPE (Gemma 4): ▁-marked literal text.
	preTokMetaspace
	// preTokDeepSeek3 is llama.cpp's DeepSeek-V3/JoyAI three-regex sequence
	// (DeepSeek-V4.1-Flash GGUFs carry tokenizer.ggml.pre=joyai-llm).
	preTokDeepSeek3
)
```

Five families. The comment above that list explains why it is a closed list rather
than a function pointer per loader, and the reason is a bug worth reading:

> Keeping the dispatch in one exhaustive switch is the point: before this, the two
> loaders each carried their own inline dispatch and silently diverged — a GLM-4
> tokenizer loaded through the JSON path missed the GLM-4 branch the GGML path had
> and got the GPT-2 ByteLevel splitter instead, a wrong-tokens bug (#4265).

That is the shape of a tokenization bug. Nothing crashes. Nothing looks wrong. The
ids are simply different from what the model was trained on, and the model gets
quietly worse.

The default family, and the one this page uses, is the GPT-2 ByteLevel rule at
`internal/tokenizer/tokenizer.go:692`:

```go
// preTokenizeByteLevel implements the GPT-2 ByteLevel(+Digits) pre-tokenizer used
// by SmolLM2 and bare-BPE fixtures: a leading space attaches only to a following
// letter/symbol run (` ?\p{L}+`), digits are isolated.
func preTokenizeByteLevel(s string) []string {
```

Read the doc comment carefully, because it is the whole rule in one line. A space
attaches to the run of letters that follows it. Digits stand alone. So:

```text
   "Hello world"   →  ["Hello", " world"]
                        ▲        ▲
                        │        └─ the space came along with "world"
                        └─ no leading space, so nothing to attach

   "abc 123"       →  ["abc", " ", "123"]
                        ▲      ▲     ▲
                        │      │     └─ digits are isolated
                        │      └─ a space with no letter after it stands alone
                        └─ letters
```

That is why the tokenizer produces `" world"` and not `"world"`. The space is not
discarded. It is part of the piece.

## Stage 2: spell each piece as printable bytes

Now each piece has to become something the merge stage can work on. The merge
stage works on **bytes**, because bytes are the only thing that covers every
possible input.

But raw bytes are a problem. A byte is a number from 0 to 255, and most of those
numbers are not printable. You cannot reliably put them in a text file, and you
cannot reliably read them back. So the tokenizer rewrites every byte as a
**printable stand-in character**, using a fixed table that is one to one and
exactly reversible.

The function is `byteLevelEncode` at `internal/tokenizer/tokenizer.go:679`:

```go
func byteLevelEncode(s string) string {
	var b strings.Builder
	for _, bb := range []byte(s) {
		b.WriteRune(byteLevelEncodeRune[bb])
	}
	return b.String()
}
```

Three lines, and the whole idea is in them. Walk the string **as bytes**, not as
characters. For each byte, look up its stand-in character and write that instead.
The table is built once at `internal/tokenizer/tokenizer.go:662` by inverting the
decode table, so the two directions cannot drift apart:

```go
func makeByteLevelEncode() map[byte]rune {
	out := make(map[byte]rune, len(byteLevelDecode))
	for r, b := range byteLevelDecode {
		out[b] = r
	}
	return out
}
```

Building the encode table by inverting the decode table is a small thing that
prevents a whole class of bug. If someone edits one direction, the other follows
automatically. There is no second table to forget.

Here is what the spelling does to our pieces:

```text
   "Hello"   →  H  e  l  l  o        (all printable already, unchanged)
               └─ 5 bytes, 5 stand-ins

   " world"  →  Ġ  w  o  r  l  d
                ▲
                └─ the space byte (32) became the stand-in "Ġ"
```

So `" world"` is now the six-character string `"Ġworld"`. The `Ġ` is not a letter.
It is the printable spelling of a space.

**A second spelling exists.** Some model families do not use the byte table at all.
They use a different convention where a space becomes `▁` and the text is marked
up directly. That is the **metaspace** family, and its spelling function is
`metaspaceEncode` at `internal/tokenizer/tokenizer.go:62`:

```go
func metaspaceEncode(s string) string {
	return strings.ReplaceAll(s, " ", string(metaspaceRune))
}
```

One line, and it does the same job by a different route: replace every space with
the marker `▁`, declared at `internal/tokenizer/tokenizer.go:58`. The two spellings
are not interchangeable, and which one applies is decided by the same dispatch that
chose the split rule.

## Stage 3: merge by rank

Now the real algorithm. The input is a string of stand-in characters. The output is
a list of symbols, each of which is a chunk of that string.

The rule, in words:

> Repeatedly find the adjacent pair of symbols whose merge has the **lowest rank**.
> Apply that merge everywhere it occurs. Repeat until no adjacent pair has a rank.

The rank table is the learned recipe from Level 1. Rank 0 is the first merge the
training pass found, rank 1 the second, and so on. **Lower rank means learned
earlier, which means more common.** So "lowest rank first" means "most common
first."

Here is the whole thing on `"Hello"`, using the small merge table from the
repository's own test fixture at `internal/tokenizer/bpe_merge_test.go:77`:

```text
   start:   H  e  l  l  o

   ranks:   "H e"    → 0
            "He l"   → 1
            "Hel l"  → 2
            "Hell o" → 3

   ── pass 1 ────────────────────────────────────────────────
   scan every adjacent pair, find the lowest rank that exists:
        (H,e)=0  (e,l)=–  (l,l)=–  (l,o)=–
   lowest is rank 0: "H"+"e".  Apply it everywhere.
   →  He  l  l  o

   ── pass 2 ────────────────────────────────────────────────
   scan again:
        (He,l)=1  (l,l)=–  (l,o)=–
   lowest is rank 1: "He"+"l".  Apply it.
   →  Hel  l  o

   ── pass 3 ────────────────────────────────────────────────
        (Hel,l)=2  (l,o)=–
   lowest is rank 2: "Hel"+"l".  Apply it.
   →  Hell  o

   ── pass 4 ────────────────────────────────────────────────
        (Hell,o)=3
   lowest is rank 3: "Hell"+"o".  Apply it.
   →  Hello

   ── pass 5 ────────────────────────────────────────────────
   no adjacent pair has a rank.  Stop.
   result: ["Hello"]
```

One symbol. `"Hello"` is a single token in this fixture, because the fixture's
merge table was built to spell it out one letter at a time.

Now the same procedure on `"Ġworld"`, whose merges are `"w o"`, `"wo r"`,
`"wor l"`, `"worl d"`, and `"Ġ world"`:

```text
   start:   Ġ  w  o  r  l  d

   ── pass 1 ──  lowest rank present is "w"+"o"  →  Ġ  wo  r  l  d
   ── pass 2 ──  lowest rank present is "wo"+"r" →  Ġ  wor  l  d
   ── pass 3 ──  lowest rank present is "wor"+"l"→  Ġ  worl  d
   ── pass 4 ──  lowest rank present is "worl"+"d"→ Ġ  world
   ── pass 5 ──  lowest rank present is "Ġ"+"world" → Ġworld
   ── pass 6 ──  nothing left.  Stop.
   result: ["Ġworld"]
```

Also one symbol. So `"Hello world"` becomes two symbols, `["Hello", "Ġworld"]`,
and the lookup gives two ids.

**Notice what the rank order did.** In pass 5, the pair `"Ġ"+"world"` was available
from the very beginning — `Ġ` and `w` were adjacent at the start. But its rank is
the *highest* of the five, so it was tried *last*. If the ranks had been in a
different order, the result would be different. The order is the information.

## The lookup, and the end of the pipe

The symbols are now chunks of the spelled string. Each one is looked up in the
vocabulary to get its number. That happens in `appendOrdinary` at
`internal/tokenizer/tokenizer.go:504`:

```go
func (t *Tokenizer) appendOrdinary(ids []int, chunk string, added map[int]bool) ([]int, error) {
```

The relevant lines are the ones that turn a symbol into an id, and the error path
is worth reading because it is the honest one:

```go
		for _, sym := range syms {
			id, ok := t.tokenToID[sym]
			if !ok {
				return nil, fmt.Errorf("tokenizer: token %q not in vocab", sym)
			}
```

If a symbol is not in the vocabulary, the tokenizer **fails** rather than guessing.
That is the right call. A tokenizer that silently substituted something would
produce ids the model was never trained on, and the failure would surface much
later as a quality problem with no obvious cause.

## The reverse direction

`Decode` at `internal/tokenizer/tokenizer.go:557` runs the pipe backwards: id to
symbol, symbol to spelled bytes, spelled bytes back to real bytes through the
inverse table, and the bytes to text.

The property that matters is that the round trip is **exact**. Encode then Decode
returns the original bytes, including bytes that are not valid text. That is not a
nice-to-have. It is the contract, and the repository tests it directly.

## Where the pieces live

| Stage | Function | File |
|---|---|---|
| entry | `Encode` | `internal/tokenizer/tokenizer.go:356` |
| split | `preTokenizeByteLevel` | `internal/tokenizer/tokenizer.go:692` |
| split dispatch | `resolvePreTokenizer` | `internal/tokenizer/pretok_dispatch.go:34` |
| spell (bytes) | `byteLevelEncode` | `internal/tokenizer/tokenizer.go:679` |
| spell (metaspace) | `metaspaceEncode` | `internal/tokenizer/tokenizer.go:62` |
| merge | `bpe` | `internal/tokenizer/bpe_merge.go:22` |
| lookup | `appendOrdinary` | `internal/tokenizer/tokenizer.go:504` |
| reverse | `Decode` | `internal/tokenizer/tokenizer.go:557` |

Level 3 takes the merge stage and opens it up, because it is the only stage with a
real algorithm in it and the only one where two implementations have to agree.

## See also

- [Level 3 — The merge loop, by hand](level-3-the-merge-loop-by-hand.md): the merge
  stage in full, with the naive oracle and the fast implementation side by side.
- [Level 1 — What is a tokenizer?](level-1-what-is-a-tokenizer.md): the same
  pipeline with no code at all.
- [The inference ladder](../inference/README.md) — what happens to the ids once
  they exist.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Level 3 — The merge loop, by hand](level-3-the-merge-loop-by-hand.md)*
