---
title: "What to quantize, and what to leave alone"
description: "What to quantize: why some numbers survive aggressive rounding and others break the model, and why the answer is decision margins, not magnitude."
slug: what-to-quantize
keywords:
  - quantization sensitivity
  - which layers to quantize
  - activation aware quantization
  - outlier channels
  - KV cache quantization
  - asymmetric quantization
  - quantization quality loss
  - mixed precision strategy
  - fak mixed precision ladder
date: 2026-11-18
---

# What to quantize, and what to leave alone

Quantization is storing a model's numbers in fewer bits, and not every number tolerates it equally. This page explains why some survive aggressive rounding and others break the model, why the answer is about decision margins rather than magnitude, and why Keys and Values differ.

*You are on **Level 3 of 5** of the [mixed-precision ladder](README.md).*

> **In one breath:** Rounding is only dangerous when the model was close to a
> decision. A confident prediction survives. A near-tie does not. So the
> question is which numbers decide.

**One line:** Quantization sensitivity is governed by how close a computation is to a decision boundary, not by how large the numbers are, which is why the same format can be free in one place and destructive in another.

> **Audience.** You are a practitioner. You have quantized a model and watched
> quality drop, or you are about to and want to avoid it. You know what a
> softmax is.

> **Short answer.** Two mechanisms decide sensitivity. **Decision margins**: a
> computation that is nearly a tie can be flipped by a tiny error. **Error
> amplification**: some operations multiply their inputs' errors rather than
> averaging them. Find the places where both are true, and leave those alone.

## The wrong question

The intuitive question is "which numbers are biggest?" Big numbers feel
important, so you would protect them.

That is the wrong question, and it leads to bad quantization decisions.

Here is why. Consider two logits, the raw scores a model produces before it
picks a token.

```text
  case A:  logits = [ 20.0,  -5.0 ]
  case B:  logits = [  8.70,  8.65 ]
```

Case A has the bigger numbers. Case B is the dangerous one.

```text
  case A, round the top logit by 0.1:
    before:  [ 20.0, -5.0 ]  ->  probs [ 1.0000, 0.0000 ]
    after:   [ 19.9, -5.0 ]  ->  probs [ 1.0000, 0.0000 ]
    the answer is unchanged

  case B, round the second logit by 0.1:
    before:  [ 8.70, 8.65 ]  ->  probs [ 0.512, 0.487 ]
    after:   [ 8.70, 8.75 ]  ->  probs [ 0.488, 0.512 ]
    the answer FLIPPED
```

Same size of error. In case A nothing happened. In case B the model chose a
different token, and every token after it is generated from a different history.

**Sensitivity is about the margin, not the magnitude.**

## Why does a flipped token matter so much?

Because generation is a chain, not a set of independent choices.

```text
  token 1  ->  becomes part of the input for token 2
  token 2  ->  becomes part of the input for token 3
  token 3  ->  ...
```

If token 1 changes, the input to token 2 changes, so token 2 may change, and so
on. A single flipped decision early in a sequence can send the whole answer down
a different path.

This is why quantization damage is often not "slightly worse text." It is
"confidently wrong," which is much harder to notice and much worse.

## The second mechanism: error amplification

Some operations **average** their inputs' errors. Others **multiply** them.

**Averaging operations are safe.** A weighted sum of many values, where the
weights are all moderate, tends to cancel noise. Errors point in different
directions and partly cancel.

```text
  output = 0.3*v1 + 0.3*v2 + 0.4*v3
  if each v has a small random error, the output's error is smaller than
  the largest input error, because the errors partly cancel
```

**Amplifying operations are dangerous.** An exponential is the clearest case.
A small change in the input becomes a large change in the output.

```text
  exp(8.70) = 6003
  exp(8.75) = 6336      <- a 0.05 input change, a 5.5% output change
```

Softmax is built on exponentials. So anything feeding a softmax is in the
amplifying category, and deserves more care than anything feeding a plain sum.

## Putting the two together

A number is sensitive when **both** conditions hold:

1. It feeds a decision that is close to a tie.
2. It feeds an operation that amplifies error.

```text
                    amplifies error?
                    no            yes
              +-------------+-------------+
   close to   |             |             |
   a tie?  no |   SAFE      |   watch     |
              |             |             |
              +-------------+-------------+
          yes |   watch     |  PROTECT    |
              |             |             |
              +-------------+-------------+
```

The bottom-right cell is where quantization damage comes from. Everything in
this ladder is a strategy for finding that cell and staying out of it.

## The KV cache: a worked case

The KV cache is the best example in the whole subject, because the two halves of
it land in different cells of that table.

Recall from the [inference ladder](../inference/level-3-inside-a-block.md)
that attention computes:

```text
  output  =  sum over j of  weight(Q, K_j) * V_j
```

The **weights** come from a softmax over Query-Key comparisons. The **output** is
a weighted sum of Values.

Now apply the table.

**Keys** feed the softmax. Softmax is exponential. So Key noise is amplified,
and it distorts *which tokens the model attends to*. That is a routing failure:
the model looks at the wrong part of the context. At long context, this loses
information badly.

```text
  Keys:    feed softmax  ->  amplified  ->  PROTECT
```

**Values** are combined by a weighted sum. That is a linear, averaging
operation. Value noise partly cancels.

```text
  Values:  feed a sum    ->  averaged   ->  compress hard
```

So the natural design is **asymmetric**: keep Keys at higher precision, compress
Values aggressively.

```text
  +------------------+------------------+
  |  Keys            |  Values          |
  |  8-bit or 16-bit |  4-bit           |
  |  (routing)       |  (content)       |
  +------------------+------------------+
```

fak documents exactly this reasoning in `internal/model/asymmetric_kv.go`, along
with the memory accounting: at a quarter-million token context, the file records
a 32-bit baseline of roughly 61 to 80 GB against an asymmetric 8/4 configuration
of roughly 15.6 to 20 GB.

**The honest caveat, again.** That file is a codec and a memory calculator. The
comment in `internal/model/kvquant.go` states that the asymmetric codec is not
wired into the decode path, and the live cache still stores 32-bit rows. The
reasoning is sound and the accounting is real; the wiring is not done. Level 5
grades this properly.

## The weights: outliers

The same logic applies to the model's weights, with a different mechanism.

Weight tensors often contain a few **outlier channels** — a small number of
dimensions whose values are much larger than the rest. If you use one scale for
the whole tensor, the outliers force a coarse scale and everything else loses
detail.

```text
  a weight row:  [0.01, 0.02, 0.01, 0.03, 12.4, 0.02, 0.01]
                                              ^^^^
                                              outlier
```

Two standard answers:

**Per-group scales.** Give each small group its own scale, so the outlier only
coarsens its own group. This is what most 4-bit formats do.

**Activation-aware scaling.** Notice that what matters is not the weight alone
but the weight times the activation it multiplies. A large weight that is always
multiplied by a tiny activation contributes little. So scale the weights to
protect the channels that actually matter, and compensate in the activations.

That second idea is **AWQ**, and fak implements it in `internal/model/awq.go`.
The format stores a per-channel scale and a 4-bit code, and dequantizes as
`scale[o] x (code - 8)`. The "activation-aware" part is the choice of scale: it
is picked to protect the channels with large activations, not the channels with
large weights.

## The practical checklist

When you quantize something and quality drops, work through this:

1. **Is the damage at a decision?** Look for near-ties in the logits. If the
   model is confidently wrong, a flipped decision is the likely cause.
2. **Does the quantized tensor feed a softmax?** If so, it is in the amplifying
   column and needs more bits.
3. **Are there outliers?** If a few channels dominate, per-group scales or
   activation-aware scaling will help more than raising the global bit width.
4. **Is it the weights or the cache?** They have different sizes, different
   growth rates, and different sensitivities. Fix the one that is actually
   hurting.
5. **Did you measure, or are you guessing?** A quality claim without a
   measurement is a hypothesis. Level 5 is about keeping those apart.

## Check yourself

1. Why is a near-tie more dangerous than a large number?
2. What are the two conditions that make a number sensitive?
3. Why do Keys need more precision than Values?
4. What is an outlier channel, and what are two ways to handle it?
5. What does "activation-aware" mean in AWQ?

<details>
<summary>Answers</summary>

1. Because a near-tie can be flipped by a small error, and a flipped decision
   changes every token generated afterward. A large number with a big margin
   survives the same error unchanged.
2. It feeds a decision that is close to a tie, and it feeds an operation that
   amplifies error rather than averaging it.
3. Keys feed the softmax over Query-Key comparisons, which is exponential and
   amplifies their noise, distorting which tokens the model attends to. Values
   are combined by a linear weighted sum, which averages noise out.
4. A small number of channels whose values are much larger than the rest. Handle
   them with per-group scales, or with activation-aware scaling.
5. The scale is chosen to protect the channels with large activations, rather
   than the channels with large weights, because what matters is the product of
   weight and activation.

</details>

## See also

- [Level 4 — The precision machinery in fak](level-4-the-precision-machinery.md) — how the runtime acts on this.
- [AWQ quantization](../awq-quantization.md) — activation-aware scaling, in full detail.
- [Level 5 of the inference ladder](../inference/level-5-at-scale.md) — the same argument from the other side: quantization as one of the things that changes at scale, with the KV cache's full-precision `Kraw` copy as the worked example.
- [The fak mixed-precision ladder](README.md) — all five rungs.

---

*[Full ladder (README)](README.md) · prev ← [Level 2 — Number formats, from fp32 to 4-bit](level-2-number-formats.md) · next → [Level 4 — The precision machinery in fak](level-4-the-precision-machinery.md)*
