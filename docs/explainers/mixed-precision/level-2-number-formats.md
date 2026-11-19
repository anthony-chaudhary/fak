---
title: "Number formats, from fp32 to 4-bit: the format zoo"
description: "What fp32, fp16, bf16, fp8, int8, and 4-bit actually are, what each can represent, and how a 4-bit number is packed and unpacked."
slug: number-formats
keywords:
  - fp32 vs fp16
  - what is bf16
  - what is fp8
  - e4m3 vs e5m2
  - int8 quantization
  - 4-bit quantization
  - block quantization
  - number formats explained
  - fak mixed precision ladder
date: 2026-11-18
---

# Number formats, from fp32 to 4-bit: the format zoo

*You are on **Level 2 of 5** of the [fak mixed-precision ladder](README.md).*

> **In one breath:** Numbers come in different widths. Some split their bits
> into an exponent and a fraction. Some are just integers. Each choice trades
> range against detail.

**One line:** A numeric format is a fixed budget of bits split between representing how large a number is and how precisely it is known, and the formats in an inference stack differ in how they spend that budget.

> **Audience.** You are a student. You know what a floating point number is, or
> you are willing to learn in two paragraphs. You are comfortable with powers of
> two.

> **Short answer.** Two families. **Floating point** splits its bits into a sign,
> an exponent (how big) and a fraction (how precise). **Integer** formats spend
> all their bits on a value in a fixed range, with a scale factor supplied
> separately. Narrowing the width forces a choice: keep the range, or keep the
> detail.

## The two families

Every format in this ladder belongs to one of two families.

**Floating point** can represent very large and very small numbers, because it
stores an exponent. It spends bits on three things:

```text
  [ sign ][ exponent ][ fraction ]
    1 bit    how big    how precise
```

**Integer** formats store a whole number in a fixed range. They cannot represent
0.001 or 1,000,000 directly. Instead, a **scale** is stored alongside, and the
real value is `scale x integer`.

```text
  stored:   [ 37 ]        (an integer)
  scale:    0.01
  real:     0.37
```

That is the trick that makes 4-bit and 8-bit quantization possible at all. You
are not storing the number. You are storing an index into a set of steps, plus
the step size.

## The floating point formats

| Format | Total bits | Sign | Exponent | Fraction | Notes |
|---|---|---|---|---|---|
| fp32 | 32 | 1 | 8 | 23 | The classic. Huge range, high precision. |
| fp16 | 16 | 1 | 5 | 10 | Half the size. Smaller range. |
| bf16 | 16 | 1 | 8 | 7 | Same range as fp32, less precision. |
| fp8 E4M3 | 8 | 1 | 4 | 3 | 4 exponent bits, 3 fraction bits. |
| fp8 E5M2 | 8 | 1 | 5 | 2 | 5 exponent bits, 2 fraction bits. |

The names encode the split. **E4M3** means 4 exponent bits and 3 mantissa
(fraction) bits. **E5M2** means 5 and 2.

### Why are there two 8-bit floats?

Because range and precision are a trade, and different jobs need different ends
of it.

**E5M2** has more exponent bits, so it covers a wider range of magnitudes, but
only 2 fraction bits, so it is coarse. It is the safer choice when values might
be large.

**E4M3** has fewer exponent bits, so a narrower range, but 3 fraction bits, so
it is finer. It is the better choice when values are known to be moderate and
you want accuracy.

There is a concrete consequence worth knowing. E4M3 has **no infinities** — the
bit pattern that would be infinity is reused for a large finite value. Its
maximum finite value is **448**. E5M2 does have infinities, and it is
bit-identical to the top 8 bits of a 16-bit float, which makes it cheap to
decode.

In fak, both decoders live in `internal/compute/fp8.go` and
`internal/mathx/fp8.go`. The E5M2 decoder is an exact identity — it just takes
the high byte. The E4M3 decoder has to reconstruct the value arithmetically,
because the bit layout does not line up with anything else.

## The integer formats

| Format | Bits | Values | How the real number is recovered |
|---|---|---|---|
| int8 | 8 | 256 | `scale x code` |
| 4-bit | 4 | 16 | `scale x code` |
| 2-bit | 2 | 4 | `scale x code` |

The **code** is the stored integer. The **scale** is stored separately, usually
one scale per small group of numbers rather than one per tensor.

### Why per-group scales?

Because a single scale for a whole tensor is a bad fit.

Imagine a tensor where most values are near zero and a few are huge. One scale
must cover the huge ones, so the near-zero values all round to the same code.
You have thrown away all the detail where most of the data lives.

```text
  one scale for everything:
    values:  [0.001, 0.002, 0.001, 900.0]
    scale:   900 / 7  (to cover the max)
    codes:   [0, 0, 0, 7]        <- the small values are all identical now

  one scale per group of 2:
    group 1: [0.001, 0.002]  scale 0.002/7  codes [3, 7]   <- detail kept
    group 2: [0.001, 900.0]  scale 900/7    codes [0, 7]
```

Per-group scales cost a little extra storage — you store one scale per group —
but they preserve far more detail. This is why real 4-bit formats quote a
bytes-per-parameter figure slightly above 0.5.

## A worked example: unpacking a 4-bit number

Let us do the whole thing concretely.

```text
  stored code:      5          (4 bits: 0101)
  group scale:      0.25
  group zero-point: 8          (the code that means "zero")

  real value  =  scale x (code - zero_point)
              =  0.25  x (5 - 8)
              =  0.25  x (-3)
              =  -0.75
```

Three numbers to recover one: the code, the scale, and the zero-point. The code
is 4 bits. The scale and zero-point are shared across the group.

This is exactly the shape of the AWQ format fak implements. In
`internal/model/awq.go` the dequantization is written as:

```text
  weight  =  scale[o] x (code - 8)
```

with a fixed zero-point of 8, which is the midpoint of the 0 to 15 code range.
The format works out to about **0.5625 bytes per parameter** — the 0.5 for the
codes plus the amortized cost of the scales.

## Why does the zero-point exist?

Because not every group of numbers is centred on zero.

If a group's values run from 2.0 to 5.0, a code of 0 should not mean 0.0. It
should mean 2.0. The zero-point records where the group's zero actually sits.

```text
  symmetric:   codes map to  -max ... +max     zero-point is 0
  asymmetric:  codes map to  min ... max       zero-point is wherever zero is
```

Symmetric is simpler and cheaper. Asymmetric fits skewed data better. AWQ uses
a fixed midpoint, which is the simple symmetric case.

## The format list in fak

fak declares its formats in `internal/compute/compute.go` as a `Dtype` type with
24 named constants, from `F32` through `Q4_K`, `IQ1_S`, and `NVFP4`. The
`Bytes()` method maps each to its width: 4 bytes for 32-bit float, 2 for 16-bit,
1 for 8-bit and below.

Twenty-four formats is a lot, and it is worth understanding why. Different
hardware has different fast paths. A format that is fast on one accelerator may
be unsupported on another. The list is long because the hardware is varied, not
because the formats are all equally good.

## Check yourself

1. What three things does a floating point number store?
2. What is the difference between E4M3 and E5M2, and when would you pick each?
3. How do you recover a real number from a 4-bit code?
4. Why use one scale per group instead of one per tensor?
5. What is the zero-point for, and when is it unnecessary?

<details>
<summary>Answers</summary>

1. A sign, an exponent (how large the number is), and a fraction (how precisely
   it is known).
2. E4M3 has 4 exponent bits and 3 fraction bits; E5M2 has 5 and 2. E5M2 covers
   a wider range, E4M3 is more precise. Pick E5M2 when values may be large,
   E4M3 when they are moderate and accuracy matters.
3. Multiply the code by the group scale, then subtract the zero-point:
   `scale x (code - zero_point)`.
4. Because one scale must cover the largest value in the tensor, which crushes
   all the small values into the same code. Per-group scales preserve detail
   where the data actually lives.
5. The zero-point records where the group's zero sits, so that a code of zero
   does not have to mean a value of zero. It is unnecessary when the data is
   symmetric around zero.

</details>

## See also

- [Level 3 — What to quantize, and what to leave alone](level-3-what-to-quantize.md) — which parts of a model tolerate this.
- [AWQ quantization](../awq-quantization.md) — the 4-bit format above, in full detail.
- [The fak mixed-precision ladder](README.md) — all five rungs.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · prev ← [Level 1 — What is a bit, and why fewer?](level-1-what-is-a-bit-and-why-fewer.md) · next → [Level 3 — What to quantize, and what to leave alone](level-3-what-to-quantize.md)*
