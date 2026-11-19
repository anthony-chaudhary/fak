---
title: "What is a token, and what is a model?"
description: "A total newcomer's guide to what a language model actually receives when you type a sentence, and what it gives back. No math, no code background. Level 1 of the fak inference ladder."
slug: what-is-a-token
keywords:
  - what is a token
  - what is a token in AI
  - tokenization explained simply
  - how does a language model work
  - what is a language model
  - logits explained simply
  - how does a chatbot write text
  - LLM tokens for beginners
date: 2026-11-18
---

# What is a token, and what is a model?

*You are on **Level 1 of 5** of the [inference ladder](README.md).*

> **In one breath:** A model does not read your text. It reads a list of numbers
> called tokens. It then scores every token it could say next. It picks one and
> starts again.

**One line:** The model is a function from a sequence of token ids to a score for
every entry in its vocabulary; a tokenizer turns your text into those ids, and
generation is that function applied over and over, each time appending the token
it just chose.

> **Audience.** A total newcomer. You have used a chatbot. You have not written
> code, and you do not need to. By the end you will be able to say what the
> machine receives when you type, what it produces, and why repeating that one
> small step is enough to write a paragraph.

> **Short answer.** The model never sees your text. It sees a list of **tokens** —
> whole numbers, each standing for a chunk of text. It is not a store of
> knowledge; it is a **function**. You hand it a list of numbers, and it hands
> back a score for every single token it knows. You pick one, stick it on the end
> of the list, and run the function again. That loop is the whole thing.

## What does the machine actually receive when I type a sentence?

Not letters. Not words. **Tokens.**

A **token** is a chunk of text — sometimes a whole word, sometimes part of a
word, sometimes just a punctuation mark. Before the model sees anything, a
separate program called a **tokenizer** chops your sentence into these chunks and
replaces each chunk with a whole number, called a **token id**.

Here is a short sentence carried through that step:

```text
   "The cat sat on the mat."
            │
            │  step 1: split the text into pieces (tokens)
            ▼
   ┌──────┬───────┬───────┬───────┬───────┬───────┬──────┐
   │ The  │  cat  │  sat  │  on   │  the  │  mat  │  .   │
   └──────┴───────┴───────┴───────┴───────┴───────┴──────┘
            │
            │  step 2: look each piece up in the vocabulary
            ▼
   ┌──────┬───────┬───────┬───────┬───────┬───────┬──────┐
   │ 464  │ 3797  │ 3332  │  319  │  262  │ 2603  │  13  │
   └──────┴───────┴───────┴───────┴───────┴───────┴──────┘
```

Those seven numbers are what the model receives. Nothing else. The text is gone.

**The ids above are illustrative.** They are made up to look realistic. The real
ids depend on the exact tokenizer file a model ships with, and two different
models will number the same sentence differently. The *shape* of the answer — a
sentence becomes a short list of whole numbers — is the part that is always true.

Notice one detail in the picture: the pieces are `" cat"` and `" the"`, with the
space attached to the front. That is common. The tokenizer does not treat a space
as a separator it throws away; it folds the space into the following chunk. This
is why a model can be fussy about spacing in ways that look strange to a human.

## Is a token the same as a word?

No, and this is the single most useful thing to get straight.

A token is a **frequently seen chunk of text**, and the tokenizer decides the
chunks by counting. Common words get their own token. Rare or long words get
chopped into pieces that are themselves common.

```text
   "the"            →  ┌──────┐
                       │ the  │            one token
                       └──────┘

   "unbelievable"   →  ┌──────┬──────────┬───────┐
                       │  un  │  believ  │  able │   three tokens
                       └──────┴──────────┴───────┘
```

So a short, common word can be one token, and a long, unusual word can be three
or four. A rough rule of thumb for English is that one token is about three
quarters of a word — but that is an average, not a rule, and it is badly wrong
for code, for names, and for languages other than English.

Why does this matter to you? Because everything a model is measured in — how much
text it can hold at once, how much you pay, how fast it answers — is counted in
tokens, not in words or characters. When a service says "128,000 tokens of
context," it means 128,000 of those numbers, which is roughly a short book.

## What does the model do with those numbers?

It runs a **function**.

A function, in the plainest sense, is a machine with an input slot and an output
slot: you put something in, you get something out, and the same input always
gives the same output. A model is exactly that. Its input slot takes a list of
token ids. Its output slot gives back **one score for every token in its
vocabulary**.

The **vocabulary** is the full list of chunks the model knows — every token it is
allowed to produce. For a modern model that list is around **128,000 entries**.
So the output is not one number. It is about 128,000 numbers, one per possible
next token.

```text
   the model's output, for ONE next-token decision
   ┌──────────┬───────────────┬──────────────────────────────┐
   │ token id │  piece        │  score                       │
   ├──────────┼───────────────┼──────────────────────────────┤
   │    262   │  " the"       │   18.4   ← highest           │
   │   2603   │  " mat"       │   12.1                       │
   │    319   │  " on"        │    9.7                       │
   │   3797   │  " cat"       │    3.2                       │
   │    ...   │    ...        │    ...                       │
   │  91234   │  " zucchini"  │  -14.8   ← lowest            │
   └──────────┴───────────────┴──────────────────────────────┘
        about 128,000 rows — one for every token it knows
```

Those raw scores are called **logits**. A logit is just a number; bigger means
"more likely to be next," smaller means "less likely." They are not percentages
and they do not add up to anything in particular. They are a ranking.

Sit with how lopsided that is. The model is not choosing between a handful of
candidates. It is scoring every one of roughly 128,000 chunks of text, on every
single step, and then throwing almost all of them away.

## How does a score become an actual word?

Two steps: turn the scores into probabilities, then pick one.

**Step one: scores to probabilities.** The scores are squashed into a set of
numbers between 0 and 1 that add up to exactly 1 — a **probability** for each
token. A token with a much higher score than the rest gets a much bigger share.
This is a mechanical conversion; nothing is decided here. It just makes the
ranking comparable and summable.

**Step two: pick one token.** And here there are two honest choices.

- **Always take the top one.** Look down the list, take the highest score, done.
  This is called **greedy** decoding, and it is completely predictable: the same
  prompt always produces the same answer.
- **Roll dice weighted by the probabilities.** Do not always take the winner.
  Take a token at random, but in proportion to its probability — so a token with
  a 40% chance comes up about 40% of the time. This is called **sampling**.

Why would anyone want the dice? Because always taking the top token makes a model
that is correct and dull, and sometimes stuck. If the single best next token leads
down a bad path, greedy decoding has no way out — it committed at the first step
and never reconsiders. Rolling the dice lets the model take a slightly less
obvious word, which often leads somewhere better. It also makes the same question
produce different answers on different days, which is usually what you want from
something you are talking to, and usually not what you want from something you
are testing.

Both modes are real, and both are in this repository. The greedy pick is
`internal/model/kv.go:1652`; the dice are rolled in
`internal/agent/inkernel_sampling.go:14`.

## Then how does it write a whole paragraph?

By doing that one small thing over and over.

The chosen token is **appended to the input list**, and the whole function runs
again — now with one more number at the end. The new run produces a fresh set of
128,000 scores, a new token is picked, and it is appended too. Repeat.

```text
        prompt ids: [464, 3797, 3332, 319, 262, 2603, 13]
                          │
                          ▼
              ┌────────────────────────┐
        ┌────▶│  run the model         │
        │     │  ids in → scores out   │
        │     └───────────┬────────────┘
        │                 │  ~128,000 scores
        │                 ▼
        │     ┌────────────────────────┐
        │     │  pick one token        │
        │     │  (top score, or dice)  │
        │     └───────────┬────────────┘
        │                 │  one new id, say 262
        │                 ▼
        │     ┌────────────────────────┐
        │     │  append it to the ids  │
        │     └───────────┬────────────┘
        │                 │
        └─────────────────┘   repeat until a "stop" token or a length limit
```

That is the entire secret. A chatbot writing a paragraph is this loop running a
few hundred times, each pass adding one token. There is no plan being consulted
and no outline being followed. There is a list of numbers, a function, and a
loop.

The loop stops for one of two reasons: the model picks a special token that means
"end of message," or it hits a limit on how many tokens it is allowed to produce.

## Why does it feel like it "thinks"?

Because predicting the next token turns out to be a surprisingly demanding job.

To guess the next chunk of text well, across everything people write, you have to
absorb an enormous amount about how language works — grammar, facts, the shape of
an argument, the way a story usually goes. A model that is genuinely good at
"what comes next" has, as a side effect, picked up a great deal of that. So when
it answers a question, it is not retrieving an answer from a drawer. It is
producing the continuation that a very well-read predictor would produce.

That is a real and impressive thing. It is also exactly what it says on the tin:
next-token prediction. It is worth holding both halves of that at once. The
mechanism is small and completely understandable — you have just read all of it.
The behaviour that falls out of it is large, and nobody has a tidy explanation
for all of it. Level 1 does not need to resolve that tension. It only needs you
to know which half is the machine.

## Where does fak come in?

Most of the time, the loop above runs on someone else's computer, behind a web
service, and you never see it. **fak runs the loop itself, in Go, in a single
process on your own machine.** That means the thing you have just been reading
about is not a black box here — it is a file you can open.

The whole loop is one short function. Here it is, quoted in full:

```go
func (s *Session) Generate(prompt []int, n int) []int {
	if len(prompt) == 0 {
		return nil
	}
	defer s.BeginGPUKeepAlive()()
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		logits = s.Step(next)
	}
	return out
}
```

That is `internal/model/kv.go:1601`. Read it in plain English, line by line:

- `func (s *Session) Generate(prompt []int, n int) []int` — the function takes a
  **prompt**, which is a list of token ids, and a count `n` of how many tokens to
  produce. It returns a list of token ids.
- `if len(prompt) == 0 { return nil }` — an empty prompt produces nothing. A
  guard clause, not part of the idea.
- `defer s.BeginGPUKeepAlive()()` — housekeeping: keep the graphics card awake
  while we work. Ignore it for now.
- `logits := s.Prefill(prompt)` — **run the model on the whole prompt at once**
  and get back the scores for the first next token. This is the "ids in, scores
  out" box from the diagram.
- `out := make([]int, 0, n)` — an empty list to collect the tokens we produce.
- `for i := 0; i < n; i++ {` — **the loop.** Do the following `n` times.
- `next := argmaxF32(logits)` — **pick the top-scoring token.** This is the
  greedy choice. `argmax` means "the index of the largest value."
- `out = append(out, next)` — **append it** to the list we are building.
- `if s.M.Cfg.IsEOS(next) { break }` — if that token means "end of message," stop
  early.
- `logits = s.Step(next)` — **run the model again**, now with the new token on
  the end, and get the next set of scores. This is the arrow that loops back to
  the top of the diagram.
- `return out` — hand back the produced token ids.

The mysterious thing is a `for` loop. That is the payoff of Level 1.

Two more anchors, so you can see the other ends of the pipe:

- `internal/tokenizer/tokenizer.go:356` — `func (t *Tokenizer) Encode(text string) ([]int, error)`
  is the text-to-ids step. You give it a sentence; it gives you the list of
  numbers.
- `internal/tokenizer/tokenizer.go:557` — `func (t *Tokenizer) Decode(ids []int) (string, error)`
  is the reverse. You give it the ids the loop produced; it gives you back the
  text you actually read.

And the "pick the top score" step, in full, is `internal/model/kv.go:1652`:

```go
func argmaxF32(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
```

It walks the list of scores once, remembers the biggest one it has seen and where
it was, and returns that position. That is the whole of "the model chose a word."

So the full path, end to end, is: your text goes through `Encode` and becomes
ids; `Generate` runs the loop, calling `Prefill` once and `Step` repeatedly, and
picking with `argmaxF32` each time; the resulting ids go through `Decode` and
become text again. Everything between `Prefill` and `Step` — the part that
actually computes the scores — is what the rest of this ladder is about.

## See also

- [Level 2 — The forward pass, one token at a time](level-2-the-forward-pass.md):
  what happens *inside* the box, with a tiny example you can follow by hand.
- [The caching ladder](../caching/README.md) — the same five-rung pattern applied
  to fak's caching stack. This ladder is its sibling.
- [The tokenization ladder](../tokenization/README.md) — this page says the
  tokenizer decides the chunks by counting. That ladder shows the counting: the
  three stages, the merge loop, and the naive oracle the fast one must match.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the words used
  here.

---

*[Full ladder (README)](README.md) · next → [Level 2 — The forward pass, one token at a time](level-2-the-forward-pass.md)*
