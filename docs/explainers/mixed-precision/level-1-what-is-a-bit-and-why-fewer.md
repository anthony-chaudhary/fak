---
title: "What is a bit, and why would you use fewer?"
description: "A no-math explanation of why a model's numbers have a width, why that width costs memory and time, and why using fewer bits is the most-used lever in inference."
slug: what-is-a-bit-and-why-fewer
keywords:
  - what is a bit
  - what is quantization
  - why quantize a model
  - bits and bytes explained
  - model memory explained
  - quantization for beginners
  - fewer bits faster
  - fak mixed precision ladder
date: 2026-11-18
---

# What is a bit, and why would you use fewer?

*You are on **Level 1 of 5** of the [fak mixed-precision ladder](README.md).*

> **In one breath:** A model is a huge pile of numbers. Each number takes some
> space. Smaller numbers take less space and move faster. That is the whole
> idea.

**One line:** Every number in a model occupies a fixed number of bits, so choosing a narrower format reduces both how much memory the model needs and how long each generated token takes.

> **Audience.** You have used a language model. You have never thought about
> binary, floating point, or memory. This page assumes no math.

> **Short answer.** A model is numbers. Numbers have a **width** measured in
> bits. Width times count equals memory. And because generating text is limited
> by how fast numbers can be moved, width also equals time. Narrower numbers
> mean a smaller, faster model.

## What is a bit?

A bit is the smallest unit of information a computer has. It is either 0 or 1.
Nothing in between.

One bit is not very useful on its own. It can say "yes" or "no" and nothing
else. So bits are used in groups.

```text
  1 bit   ->  2 possible values      (0, 1)
  2 bits  ->  4 possible values      (00, 01, 10, 11)
  3 bits  ->  8 possible values
  4 bits  ->  16 possible values
  8 bits  ->  256 possible values
```

The pattern: each extra bit **doubles** the number of values you can represent.
That doubling is the whole reason width matters, and it is why the difference
between 4 bits and 8 bits is much bigger than it sounds.

## What is a byte?

Eight bits, grouped together. That is the standard unit for talking about
memory.

```text
  1 byte  =  8 bits  =  256 possible values
```

When someone says a model is "4-bit," they mean each number in it uses four
bits, which is **half a byte**. When they say "32-bit," they mean four bytes per
number.

## Why does a model have numbers at all?

Because a language model is a mathematical function, and mathematical functions
work on numbers.

The model learned millions of numbers during training. Those numbers are called
**parameters** or **weights**. They are what the model "knows." When you ask a
question, the model does arithmetic with those numbers and produces an answer.

```text
  a small model:   ~1,000,000,000 numbers
  a large model:  ~100,000,000,000 numbers
```

A billion numbers is already hard to picture. A hundred billion is harder. But
the arithmetic is simple: **size equals count times width.**

## Why does width cost memory?

Because each number has to be stored, and a wider number takes more room.

```text
  1,000,000,000 numbers  x  4 bytes each  =  4,000,000,000 bytes  =  4 GB
  1,000,000,000 numbers  x  2 bytes each  =  2,000,000,000 bytes  =  2 GB
  1,000,000,000 numbers  x  1 byte  each  =  1,000,000,000 bytes  =  1 GB
  1,000,000,000 numbers  x  0.5 bytes     =    500,000,000 bytes  =  0.5 GB
```

Same model. Same knowledge. Same answers, roughly. **Eight times less memory**
just by changing the width.

That is not a small optimization. It is the difference between a model that
needs a server and a model that runs on a laptop.

## Why does width cost time?

This is the part that surprises people, and it is worth sitting with.

You would guess that fewer bits means less arithmetic, so it is faster. That is
true but it is not the main effect. The main effect is **moving**.

To generate each token, the computer must read the model's numbers out of
memory. Not once per answer — **once per token**. Every single token, the whole
model gets read again.

```text
  generating token 1:  read all the numbers
  generating token 2:  read all the numbers
  generating token 3:  read all the numbers
  ...
```

So the time per token is basically:

```text
  time per token  =  how many bytes must be read  /  how fast bytes can move
```

Halve the bytes and you halve the time. The arithmetic got easier too, but
moving was always the bottleneck.

**The one sentence to remember: in text generation, bytes are time.**

## The analogy: a library with heavy books

Imagine you must read a library to answer each question. You read it, answer,
then read the whole library again for the next question.

Now imagine the books are printed on thinner paper. Same words, same knowledge,
half the weight. You can carry twice as many at once, and you can walk twice as
fast.

That is quantization. The information is (mostly) the same. The physical cost
drops.

## What is the catch?

If thinner paper were free, everyone would use the thinnest possible.

The catch: **a narrow number cannot represent as many distinct values.** Four
bits gives you 16 possible values. Thirty-two bits gives you about four billion.
So a narrow number has to **round** to the nearest value it can represent.

```text
  a value the model wants:   0.7231847...
  stored in 32 bits:         0.7231847...     (very close)
  stored in 4 bits:          0.75            (rounded to a nearby step)
```

Rounding introduces error. The question is whether the error matters.

Sometimes it does not matter at all. Sometimes it changes the answer
completely. **Which is which is the subject of the rest of this ladder**, and
the short version is: it depends on whether the model was close to a decision.

## Why is it called "mixed" precision?

Because you do not have to make one choice for the whole model.

Some parts of a model tolerate aggressive rounding. Others do not. So you can
store the tolerant parts in 4 bits and the sensitive parts in 16, and get most
of the memory saving without most of the damage.

```text
  +-------------------------------+
  |  sensitive part   ->  16 bits |   keep it accurate
  +-------------------------------+
  |  tolerant part    ->   4 bits |   compress it hard
  +-------------------------------+
  |  tolerant part    ->   4 bits |
  +-------------------------------+
```

That is **mixed** precision: different widths in different places, chosen on
purpose. Level 3 is about how you decide which is which.

## Check yourself

1. How many values can 4 bits represent? How many can 8 bits?
2. Why does halving the width halve the memory?
3. Why does halving the width also make generation faster?
4. What is the cost of using fewer bits?
5. What does the word "mixed" add to "precision"?

<details>
<summary>Answers</summary>

1. 4 bits gives 16 values. 8 bits gives 256. Each extra bit doubles the count.
2. Because size equals the number of values times the width of each one.
3. Because generating each token requires reading all the model's numbers out
   of memory, and the time is set by how many bytes must be moved.
4. A narrow number can represent fewer distinct values, so it must round, which
   introduces error.
5. It means you do not pick one width for the whole model. You use different
   widths in different places, based on which parts tolerate rounding.

</details>

## See also

- [Level 2 — Number formats, from fp32 to 4-bit](level-2-number-formats.md) — what the formats actually are.
- [The fak mixed-precision ladder](README.md) — all five rungs.
- [Level 4 of the inference ladder](../inference/level-4-prefill-vs-decode.md) — the same idea from the other side: why decode is memory-bound, and why fewer bytes per weight buys speed roughly proportionally.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Level 2 — Number formats, from fp32 to 4-bit](level-2-number-formats.md)*
