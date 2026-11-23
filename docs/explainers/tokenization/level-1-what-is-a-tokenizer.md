---
title: "What is a tokenizer, and why does it count?"
description: "A newcomer's guide to the tokenizer: the program that turns your sentence into the numbers a model reads, and learns its chunks by counting."
slug: what-is-a-tokenizer
keywords:
  - what is a tokenizer
  - tokenizer explained simply
  - how does tokenization work
  - byte pair encoding for beginners
  - why do models use tokens
  - tokenizer vocabulary
  - text to numbers AI
  - tokenization for beginners
date: 2026-11-18
---

# What is a tokenizer, and why does it count?

*You are on **Level 1 of 5** of the [tokenization ladder](README.md).*

> **In one breath:** A tokenizer is the program that turns your text into whole
> numbers. It learns its chunks by counting which pieces of text appear together
> often. Common pieces get their own number. Rare pieces get built out of common
> ones.

**One line:** The tokenizer is a reversible map from a string of text to a list of
whole numbers, and the map is learned rather than designed: a training pass counts
adjacent symbol pairs across a corpus and records the most frequent merge first,
so the vocabulary is a frequency table wearing a lookup table's clothes.

> **Audience.** A total newcomer. You have used a chatbot. You have not written
> code, and you do not need to. By the end you will be able to say what a tokenizer
> does, why its chunks are not words, and why the whole thing is a counting
> exercise rather than a set of rules somebody wrote.

> **Short answer.** A tokenizer is a **dictionary plus a recipe**. The dictionary
> is a list of chunks of text, each with a number. The recipe says how to break a
> new sentence into those chunks. The surprising part is where the dictionary came
> from: nobody chose the chunks. A program counted which pairs of characters and
> pieces kept showing up next to each other, and wrote them down in order of how
> often they appeared. That ordered list *is* the tokenizer.

## Why does this step exist at all?

Because a model is arithmetic, and arithmetic does not take text.

A model is a very large pile of multiplications and additions. You can hand it
numbers. You cannot hand it the word "cat." So something has to stand between you
and the model and turn one into the other. That something is the tokenizer.

You could imagine the simplest possible version: give every character a number.
`c` is 1, `a` is 2, `t` is 3. Then "cat" is `[1, 2, 3]`. That works, and it is
reversible, and it is a real design that some systems use. So why does almost
nobody do it?

Two reasons, and they are the whole motivation for everything that follows.

**Reason one: sequences get long.** A model reads a fixed number of numbers at a
time. If every character is one number, then a page of text is thousands of
numbers, and you run out of room fast. If a common word is *one* number instead of
four, the same page fits in a quarter of the space.

**Reason two: characters carry too little meaning.** The letter `t` on its own
tells the model almost nothing. The chunk `the` tells it a great deal. Bigger
chunks mean each number the model reads is more informative, so the model has less
work to do to figure out what is going on.

So the goal is: **use chunks bigger than a character, but smaller than a whole
word, and pick the chunks so that common text uses few of them.** That is the
entire design brief. Everything else is how you satisfy it.

## So what is a token, really?

A **token** is one entry in the tokenizer's dictionary. It is a chunk of text, and
it has a number, called a **token id**.

Here is a sentence carried through the step:

```text
   "The cat sat on the mat."
            │
            │  the tokenizer breaks it into chunks
            ▼
   ┌──────┬───────┬───────┬───────┬───────┬───────┬──────┐
   │ The  │  cat  │  sat  │  on   │  the  │  mat  │  .   │
   └──────┴───────┴───────┴───────┴───────┴───────┴──────┘
            │
            │  each chunk is looked up and replaced by its number
            ▼
   ┌──────┬───────┬───────┬───────┬───────┬───────┬──────┐
   │ 464  │ 3797  │ 3332  │  319  │  262  │ 2603  │  13  │
   └──────┴───────┴───────┴───────┴───────┴───────┴──────┘
```

Those seven numbers are what the model receives. The text is gone.

**The numbers above are illustrative.** They are made up to look realistic. The
real numbers depend on the exact tokenizer a model ships with, and two different
models will number the same sentence differently. The *shape* of the answer — a
sentence becomes a short list of whole numbers — is the part that is always true.

Notice one detail: the chunks are `" cat"` and `" the"`, with the space attached to
the front. That is common. The tokenizer does not treat a space as a separator it
throws away. It folds the space into the following chunk. This is why a model can
be fussy about spacing in ways that look strange to a human.

## Why is a token not a word?

Because the chunks are chosen by **frequency**, not by grammar.

A token is a **frequently seen chunk of text**. Common words get their own token.
Rare or long words get chopped into pieces that are themselves common.

```text
   "the"            →  ┌──────┐
                       │ the  │            one token
                       └──────┘

   "unbelievable"   →  ┌──────┬──────────┬───────┐
                       │  un  │  believ  │  able │   three tokens
                       └──────┴──────────┴───────┘
```

Nothing in that picture knows what a word is. The tokenizer has no dictionary of
English and no grammar. It has a list of chunks and how often each appeared. `the`
appeared so often that it earned a single number. `unbelievable` is rare, so it
never earned one, and it gets spelled out of pieces that did.

This is why the same word can be one token in one model and three in another. It
depends entirely on what that model's training text happened to contain.

A rough rule of thumb for English is that one token is about three quarters of a
word. That is an average, not a rule, and it is badly wrong for code, for names,
and for languages other than English.

## Where does the dictionary come from?

This is the part that surprises people, so it is worth saying plainly.

**Nobody wrote the dictionary. A program counted it.**

Imagine you start with the dumbest possible tokenizer: every single character is
its own token. Now look at a huge pile of text and ask one question, over and
over:

> Which two tokens sit next to each other most often?

Suppose the answer is `t` followed by `h`. You write down a new token, `th`, and
from now on `th` counts as one thing. Then you ask again. Now maybe the answer is
`th` followed by `e`, so you write down `the`. Then you ask again, and again, and
again, thousands of times.

Each round you write down one new chunk, and you write them down **in the order
you found them**. That ordered list is the tokenizer's recipe. The first chunk you
found is the first merge it will try; the last is the last.

```text
   start:   t  h  e  r  e
            │
            │  round 1: "t"+"h" is the most common pair  →  learn "th"
            ▼
            th  e  r  e
            │
            │  round 2: "th"+"e" is now the most common pair  →  learn "the"
            ▼
            the  r  e
            │
            │  round 3: "r"+"e" is the most common pair  →  learn "re"
            ▼
            the  re
            │
            └─ no pair left worth learning. Done.
```

That is the whole learning procedure. It is called **byte pair encoding**, usually
shortened to **BPE**. The name is a little misleading, because it is not really
about bytes and it is not really about pairs. It is about counting.

Two things about it are worth holding on to:

- **The order is the information.** The list is not a set. Merge number 1 is tried
  before merge number 2, and that ordering is what makes the result deterministic.
- **It is greedy and it never reconsiders.** Once `th` is learned, it is learned.
  The procedure never goes back and unlearns it. That is a real limitation, and
  Level 5 comes back to it.

## How does the tokenizer use the recipe on a new sentence?

It runs the same loop, but now the order is fixed and it is not counting anything.

Take a new word the tokenizer has never seen, say `there`. It starts by breaking it
into single characters, then applies the learned merges **in order**, skipping any
that do not apply:

```text
   "there"  →  t  h  e  r  e
               │
               │  merge 1 is "t"+"h" → "th".  It applies.
               ▼
               th  e  r  e
               │
               │  merge 2 is "th"+"e" → "the".  It applies.
               ▼
               the  r  e
               │
               │  merge 3 is "r"+"e" → "re".  It applies.
               ▼
               the  re
               │
               └─ no more merges apply.  Result: ["the", "re"]
```

So `there` becomes two tokens, `the` and `re`, even though the tokenizer has never
seen the word `there` before. It did not need to. It only needed to know which
pieces are common.

That is the trick that makes this work at all. A tokenizer with a vocabulary of
about 128,000 chunks can spell essentially any text, including text it has never
seen, because it can always fall back to smaller and smaller pieces, down to a
single character if it has to.

## What happens to text that is not English?

It still works, and that is the point of the next idea.

The tokenizer does not start from letters. It starts from **bytes** — the raw
numbers a computer uses to store text. Every character in every language, every
emoji, every piece of punctuation, is some sequence of bytes. So the counting
procedure works on any text at all, in any script, without anyone writing a rule
for that script.

There is one wrinkle. A raw byte is a number from 0 to 255, and most of those
numbers are not printable characters. You cannot put them in a text file and read
them back reliably. So the tokenizer does a small piece of bookkeeping first: it
rewrites every byte as a **printable stand-in character**, chosen so that the
mapping is one to one and can be undone exactly.

```text
   the space character, as a byte, is 32
            │
            │  rewritten as a printable stand-in
            ▼
   the character "Ġ"   (this is the stand-in, not a real letter)

   so " the" is stored and counted as "Ġthe"
```

You will see these stand-ins if you ever look at a tokenizer's vocabulary file.
`Ġ` means a space. `▁` means a space in a different family of tokenizers. They look
like exotic letters. They are bookkeeping.

The important consequence: because the tokenizer works on bytes, it can never fail
to encode something. There is no input it cannot represent. That is what "lossless"
means here, and it is a property the repository tests directly.

## Where does fak come in?

Most of the time the tokenizer runs on someone else's computer and you never see
it. **fak runs it itself, in Go, in a single process on your own machine.** So the
thing you have just been reading about is a file you can open.

The two ends of the pipe are one line each:

- `internal/tokenizer/tokenizer.go:356` — `func (t *Tokenizer) Encode(text string) ([]int, error)`
  is the text-to-numbers step. You give it a sentence; it gives you the list.
- `internal/tokenizer/tokenizer.go:557` — `func (t *Tokenizer) Decode(ids []int) (string, error)`
  is the reverse. You give it the numbers; it gives you back the text.

And the counting procedure you just read about is a real function in this
repository, at `internal/tokenizer/bpe_merge.go:22`:

```go
func (t *Tokenizer) bpe(encoded string) ([]string, error) {
```

That is the merge loop. It is the heart of the whole page, and Level 3 walks
through it line by line.

One more thing worth knowing now, because it is unusual and it is a good sign. The
repository keeps **two** implementations of that loop. One is the fast one that
actually runs. The other is a deliberately slow, obvious version kept only for
testing, at `internal/tokenizer/bpe_merge_test.go:14`:

```go
func bpeNaive(mergeRank map[tokenPair]int, encoded string) []string {
```

The slow one is written the way you would write it if you did not care about
speed: scan everything, find the best pair, apply it, start over. The fast one has
to produce **exactly the same answer**, and there is a test that checks that on
every example the repository has. That pairing — a fast implementation and a slow
oracle that must agree — is the single most useful idea in this ladder, and Level 3
is where you learn to read it.

## See also

- [Level 2 — The three stages](level-2-the-three-stages.md): the same pipeline with
  the real function names, and an example you can follow by hand.
- [The inference ladder](../inference/README.md) — what happens *after* the ids
  exist. Its Level 1 is the plain-language companion to this page.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the words used
  here.
- [The learning path](../../../LEARNING-PATH.md) — course **FAK 522** covers this
  ground rigor-first.

---

*[Full ladder (README)](README.md) · next → [Level 2 — The three stages](level-2-the-three-stages.md)*
