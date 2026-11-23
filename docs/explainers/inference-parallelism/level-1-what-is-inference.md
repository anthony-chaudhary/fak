---
title: "What is inference? The one-screen version"
description: "A language model is a big pile of numbers. Inference is running those numbers over your input to predict the next token, one token at a time."
slug: what-is-inference
keywords:
  - what is inference
  - what is a forward pass
  - how does a language model work
  - next token prediction
  - what is a token
  - inference for beginners
  - fak inference ladder
date: 2026-11-18
---

# What is inference?

Inference is running a language model's numbers over your input to predict the next token, one token at a time, feeding each new token back in. This page assumes no ML background and uses one analogy and one worked example you can do by hand.

*You are on **Level 1 of 5** of the [fak inference ladder](README.md).*

> **Audience.** A reader with no machine-learning background at all — a new grad, a
> student, a backend engineer who has only ever called an API and never looked inside.
> No linear algebra assumed; we do the arithmetic by hand on numbers small enough to
> check on your fingers.

> **Short answer.** A language model is a very large pile of numbers called **weights**.
> **Inference** is the act of running those numbers over your input to produce a
> prediction of the next **token** (a chunk of text). The model does this one token at a
> time: it predicts a token, appends it to the input, and runs again — so the text grows
> one token per pass. That is the whole loop.

## What is a model, really?

Forget the word "intelligence" for a moment. A language model is a **file of numbers**.
Billions of them, arranged in tables. That is the entire artifact. There is no logic in
it, no rules, no database of facts — just numbers, and a fixed recipe for combining them
with your input.

The recipe is small and fixed. The numbers are huge and learned. When people say a model
was "trained," they mean those numbers were adjusted, over and over, until the recipe
started producing good predictions. Training is how the numbers got their values.
**Inference is what you do with them afterward** — you run the recipe, once, on your
input. Training is the factory; inference is the drive home.

## The analogy: the world's most well-read autocomplete

Your phone's keyboard suggests the next word. It does this by counting: after "good",
"morning" is common. A language model is the same idea, taken to an extreme.

- Instead of counting word pairs, it has read an enormous amount of text and compressed
  what it saw into those tables of numbers.
- Instead of suggesting one word, it **scores every token it knows** — all of them, at
  once — with a number that says "how likely is this to come next?"
- Then it picks the highest-scoring one, types it, and asks itself the question again
  with the new word included.

Carry that picture through this whole ladder. Everything we add later — layers,
parallelism, workers — is machinery for making that one scoring step faster or bigger.
The question never changes: *given everything so far, what comes next?*

## What is a token?

A **token** is a chunk of text — usually a word or a piece of a word. "inference" might
be one token; "unbelievable" might be three ("un", "believ", "able"). The model never
sees letters or words. It sees **token ids**: integers. A fixed table called the
tokenizer maps text to ids and back.

So the model's entire world is: a list of integers in, a score for every possible next
integer out.

## The loop, in one picture

```text
  prompt: "the cat sat"          ->  token ids: [ 41, 892, 330 ]   (ids illustrative)

  pass 1:  [41, 892, 330]              --Forward-->  scores  --pick-->  "on"   (id 271)
  pass 2:  [41, 892, 330, 271]         --Forward-->  scores  --pick-->  "the"  (id 55)
  pass 3:  [41, 892, 330, 271, 55]     --Forward-->  scores  --pick-->  "mat"  (id 903)
  pass 4:  [41, 892, 330, 271, 55, 903] --Forward-->  scores  --pick-->  "."    (id 13)
  ...
```

Read the left column downward: **the input grows by one token every pass.** The model
has no memory between passes. The only reason it "remembers" what it said is that we
hand the whole list back to it each time. That is why a long conversation costs more
than a short one — the list is longer, so there is more to run the numbers over.

In fak, one pass is a call to `Forward` (`internal/model/forward.go:483`). It takes the
list of token ids and returns, for every position, a score for every token in the
vocabulary.

## A worked example you can do by hand

Let us shrink the model until it fits on a napkin. Say the **hidden size is 4** (each
token is represented by 4 numbers) and the **vocabulary is 6** (the model knows only 6
tokens). Real models use thousands and tens of thousands; the shape of the arithmetic is
identical.

**Step 1 — turn the token id into numbers.** The model has an embedding table: one row
of 4 numbers per token id. This lookup is `embedBand` (`internal/model/partition.go:91`).

```text
  token id | embedding (4 numbers)
  ---------+------------------------
     0     | [ 0.1, -0.2,  0.3,  0.0]
     1     | [ 0.5,  0.1, -0.1,  0.2]
     2     | [-0.3,  0.4,  0.0,  0.1]
     3     | [ 0.2,  0.2, -0.4,  0.6]   <-- our input token is id 3
     4     | [ 0.0, -0.5,  0.2,  0.3]
     5     | [ 0.7,  0.0,  0.1, -0.2]
```

So the input token id 3 becomes the vector `x = [0.2, 0.2, -0.4, 0.6]`. That vector is
called the **hidden state**. It is the model's working memory for this token.

**Step 2 — run the layers.** The hidden state is passed through a stack of layers
(`forwardHiddenRows`, `internal/model/forward.go:508`). Each layer reads the 4 numbers
and writes back 4 numbers — it *shuffles* the values, mixing in information from the
other tokens, but it never changes the shape. What the layers actually do is
[Level 2](level-2-the-layer-loop.md). For now, suppose that after all the layers the
hidden state is:

```text
  x = [ 0.3, -0.1, 0.5, 0.2 ]        (still 4 numbers — the shape never changed)
```

**Step 3 — score every token.** This is the tail, `logitsFromHidden`
(`internal/model/forward.go:559`). It multiplies the hidden state by a **head matrix**
with one row per vocabulary entry (6 rows, 4 columns here). Each row is a "what does
this token look like?" template; the dot product measures how well our hidden state
matches it.

```text
  head matrix (6 rows x 4 cols)        x = [0.3, -0.1, 0.5, 0.2]
  ---------+-------------------------   score = row . x
   tok 0   | [ 0.1,  0.0, -0.2,  0.1]   ->  -0.05
   tok 1   | [ 0.0,  0.3,  0.1,  0.0]   ->   0.02
   tok 2   | [-0.1,  0.1,  0.2,  0.1]   ->   0.08
   tok 3   | [ 0.2, -0.1,  0.0,  0.3]   ->   0.13   <-- highest
   tok 4   | [ 0.0,  0.2, -0.1,  0.0]   ->  -0.07
   tok 5   | [ 0.4,  0.1,  0.1, -0.2]   ->   0.12
```

Check one by hand: token 3's row is `[0.2, -0.1, 0.0, 0.3]`, and
`0.2*0.3 + (-0.1)*(-0.1) + 0.0*0.5 + 0.3*0.2 = 0.06 + 0.01 + 0 + 0.06 = 0.13`. Good.

Those six numbers are the **logits** — one score per possible next token. The highest is
token 3 (0.13), so the model predicts **token 3**. Notice how close token 5 (0.12) is:
the winner is decided by the arithmetic, not by any hidden intent. (Real inference turns
these scores into probabilities and samples from them, but "pick the top one" is the
honest core.)

**Step 4 — append and repeat.** Token 3 is added to the input list, and we go back to
Step 1 with one more token. That is the loop from the picture above.

## Why this matters for the rest of the ladder

Everything from here on is about **Step 2** — the layers. Step 1 is a table lookup, Step
3 is one matrix multiply, but Step 2 is where nearly all the numbers and nearly all the
time live. It is also the part that is *shaped* in a way that lets us cut it up and run
it on several workers at once. That is the subject of Levels 3, 4, and 5.

## Try it

```bash
go test ./internal/model/ -run 'TestForwardBandTwoWayMatchesMonolithic' -count=1 -v
```

You should see `PASS`. This test builds a tiny model, runs the full forward pass, then
runs it again **split into two halves** and checks the two answers are bit-for-bit
identical — the first hint that the layer stack can be cut apart without changing the
result. [Level 2](level-2-the-layer-loop.md) explains what is being cut.

## See also

- [Level 2 — The layer loop](level-2-the-layer-loop.md): what actually happens between
  the embedding and the logits.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the words used here.
- [Context](../context.md) — why the growing input list is the thing that makes long
  sessions expensive.

---
*[Full ladder (README)](README.md) · next → [Level 2 — The layer loop](level-2-the-layer-loop.md)*
