---
title: "What is an instruction set? A first look at the language your chip actually speaks"
description: "A beginner's guide to the instruction set: what an instruction is, why a processor has a fixed list of them, and the four-layer journey from your source file to a voltage. Written for students and new grads, grounded in fak's real code."
slug: level-1-what-is-an-instruction-set
keywords:
  - what is an instruction set
  - instruction set architecture for beginners
  - how does a CPU execute code
  - machine code explained
  - compiler explained simply
  - computer architecture for new grads
  - what is an ISA
date: 2026-11-18
---

# What is an instruction set?

> **In one breath:** A processor can only do a small fixed list of things. That list is
> called its instruction set. Your program is not on the list. So something must
> translate your program into it.

**One line:** the instruction set is the complete, frozen vocabulary of one chip — every
operation it can perform, and nothing else — and the entire software stack above it exists
to translate human intent into words from that vocabulary.

*You are on **Level 1 of 5** of the [fak hardware instruction-set ladder](README.md).*

> **Audience.** You have written a program. You have run it. You have never thought about
> what the chip did with it. That is the only prerequisite. No assembly, no hardware
> background, no C.
>
> **Short answer.** A processor is a machine that can do maybe a few hundred distinct
> things. Each thing is called an instruction. The full list of them is the instruction
> set. Your Python or Go or JavaScript is not on that list, so a chain of translators
> turns it into the list. This page is about that chain, and about why the list is
> different on every chip.

## Start with a question you can answer

Here is a program. You have written a thousand of these.

```python
total = 0
for n in [3, 5, 7]:
    total = total + n
print(total)
```

**Checkpoint:** before reading on, answer this in one sentence. What did the processor
actually *do* to make `15` appear on your screen?

Most people say something like "it added the numbers." That is true and it is also
useless, because it hides the interesting part. The processor did not add "the numbers."
It executed a specific sequence of a few dozen tiny operations, each of which it already
knew how to do before your program existed. Let us find out what those operations are.

## What is the smallest thing a processor can do?

A processor is a circuit. Circuits do not understand intent. They respond to electrical
signals. So the smallest thing a processor can do has to be expressible as a pattern of
signals, and that pattern has to be built into the circuit's wiring.

That is the whole constraint. **A processor can only do the things its wiring supports.**

Each supported thing is called an **instruction**. An instruction is a very small
operation. Not "add up a list." More like:

- "Copy the number in slot 3 into slot 7."
- "Add the number in slot 7 to the number in slot 9, and put the result in slot 9."
- "If the number in slot 9 is zero, jump to the instruction at position 40."

That is the scale. One instruction does one tiny thing. A program is thousands or
billions of these, in order.

> **The key idea.** An instruction is not a function call. It is not a statement. It is
> the smallest unit of work the hardware was built to perform, and it is fixed at
> manufacturing time.

## Why can't the chip just understand Python?

This is the question that unlocks everything else, so let us take it seriously.

Imagine you are designing a chip. You have a fixed budget of silicon. Every operation you
want the chip to be able to do natively costs area, power, and design effort. You cannot
support everything.

So you support a small, carefully chosen set. You support "add two integers" because
everything needs it. You support "load a value from memory" because you have to get data
somehow. You support a few dozen more. Then you stop, because the budget is spent.

Now: should "add up a list of numbers" be one of the supported operations? It is a very
common thing to want. But it is also *expressible* as a loop of "add two integers" plus
"compare" plus "jump." If you support those three, you get the loop for free, and you
spend your silicon budget on something else.

That trade is the entire reason instruction sets are small. **A chip supports a small
primitive vocabulary and lets software compose it.**

**Checkpoint:** why is `for n in [3, 5, 7]` not a single instruction? Answer in one
sentence before continuing.

## So what is an "instruction set," exactly?

The **instruction set** is the complete list of instructions a given processor supports,
together with the rules for how each one is written down and what it does.

The formal name is **instruction set architecture**, usually shortened to **ISA**. The
word "architecture" is doing real work there. The ISA is not the chip. It is the
*contract* the chip promises to honour. Two different chips can implement the same ISA
and be completely different pieces of silicon.

Here is the part that surprises people:

> **The instruction set is a contract, not a description.** It says what the chip will do.
> It says nothing about how fast, how much power, or how it is built.

A chip that implements the contract correctly is a valid implementation. A chip that is
ten times faster is also a valid implementation. The contract is the same. That gap
between "what it does" and "how fast it does it" is the subject of
[Rung 5](level-5-why-the-isa-is-not-the-whole-story.md), and it is the single most
important thing to understand about performance.

## The four-layer journey

Now we can answer the original question. Here is what happens between your source file
and a voltage.

```
   your source file          total = total + n
            │
            │  compiler
            ▼
   instructions              add  %eax, %ebx
            │
            │  assembler
            ▼
   machine code              01 D8
            │
            │  the processor's decode circuit
            ▼
   electrical signals        the adder's inputs go high
```

Four layers. Let us walk them.

### Layer 1: your source file

`total = total + n`. This is written for a human. It is deliberately ambiguous about
hardware. It does not say which slot holds `total`. It does not say whether the addition
is 32-bit or 64-bit. It does not say what happens on overflow.

That ambiguity is a feature. It is why the same line runs on a laptop and a phone.

### Layer 2: instructions

A **compiler** removes the ambiguity. It decides which slot holds `total`, picks a width,
and emits a specific instruction:

```
add  %eax, %ebx
```

This is **assembly language**. It is still text, still readable by a determined human, but
every line now names exactly one instruction from the chip's instruction set. The
ambiguity is gone.

### Layer 3: machine code

An **assembler** turns that text into numbers, because the chip does not read text:

```
01 D8
```

Two bytes. `01` means "add," `D8` encodes which two slots. This is **machine code**: the
instruction set written as the actual bit patterns the decode circuit expects.

### Layer 4: electrical signals

The processor's **decode circuit** reads those bytes and raises and lowers the right wires.
The adder's inputs go high. The result appears. No software is involved at this layer. It
is physics.

**Checkpoint:** which of the four layers is different on an ARM laptop versus an Intel
laptop? Answer before reading the next section.

## The answer to that checkpoint is the whole point

Layers 1 and 4 are the same everywhere. Your source file is portable. Physics is physics.

**Layers 2 and 3 are different on every chip family.** An ARM chip does not have `add
%eax, %ebx`. It has a different instruction set, with different names, different slot
names, and different bit patterns. The same source file compiles to completely different
instructions.

This is why the instruction set matters. It is the layer where portability ends.

## Let us look at a real one

fak is an agent kernel: a Go program that runs a language model's forward pass itself,
in-process. That forward pass is, at the bottom, an enormous number of multiply-and-add
operations. fak has to make those fast, on whatever chip it is running on.

So fak contains hand-written assembly for two different instruction sets. Here is the
same function, `qdot8asm`, written twice.

**For x86-64 chips, using the AVX2 instruction set** — from
[`internal/model/quant_amd64.s:29`](../../../internal/model/quant_amd64.s):

```asm
VPMOVSXBW (SI), Y0
VPMOVSXBW (DI), Y1
VPMADDWD  Y1, Y0, Y0
```

**For ARM64 chips, using the NEON instruction set** — from
[`internal/model/quant_arm64.s:39`](../../../internal/model/quant_arm64.s):

```asm
WORD $0x4e829404               // SDOT V4.4S, V0.16B, V2.16B
WORD $0x4e839424               // SDOT V4.4S, V1.16B, V3.16B
```

Look at what is different. Different instruction names. Different register names (`Y0`
versus `V0`). And in the ARM64 case, the instruction is written as a raw number, because
the Go assembler does not even have a name for it. The comment tells you what the number
means.

**Checkpoint:** these two snippets do the same arithmetic. Why are they not the same
text? Answer in one sentence.

That is the instruction set, made concrete. Same job, two vocabularies, because two
different chips.

## Why does fak bother?

A fair question. Why not write it once in Go and let the compiler handle it?

Because the compiler is not always good enough at the thing that matters most. The header
comment of the ARM64 file records a measured result
([`internal/model/quant_arm64.s:9`](../../../internal/model/quant_arm64.s)):

> The scalar int8 dot (`qdot8scalar`) is compute-bound — Go emits a per-byte sign-extend
> + scalar imul, so streaming 3.5× fewer weight bytes than f32 bought nothing on this arch
> (int8 measured 0.95× f32, i.e. no win) until the dot itself goes SIMD.

Read that carefully, because it is a small masterclass in engineering honesty. fak tried
to go faster by using smaller numbers. It streamed 3.5 times fewer bytes. And it got
**no speedup at all** — 0.95×, which is a rounding error, not a win.

Why? Because the bottleneck was not the bytes. It was the arithmetic. Go was emitting a
sign-extend and a multiply for every single byte, and that per-byte work dominated. Making
the data smaller did not help, because the data was never the problem.

The fix was to change the *instruction*. `SDOT` does sixteen multiply-accumulates in one
instruction. Once the arithmetic got cheap, the byte savings finally showed up.

**Checkpoint:** fak made the data 3.5× smaller and got 0.95× speed. What does that tell
you about where the time was going?

## The vocabulary you now have

| Term | What it means |
|---|---|
| **Instruction** | The smallest operation a processor can perform. Fixed at manufacturing time. |
| **Instruction set** | The complete list of instructions one processor supports. |
| **ISA** | Instruction set architecture. The same thing, with the emphasis on "this is a contract." |
| **Assembly language** | Instructions written as text, one instruction per line. |
| **Machine code** | Instructions written as the actual bit patterns the chip decodes. |
| **Compiler** | The program that turns your source into instructions. |
| **Assembler** | The program that turns instructions into machine code. |

## Try it

You can see all four layers on your own machine, right now, with no hardware and no
special tools.

**Step 1.** Write a tiny Go file.

```go
package main

import "fmt"

func main() {
	total := 0
	for _, n := range []int{3, 5, 7} {
		total = total + n
	}
	fmt.Println(total)
}
```

**Step 2.** Ask the Go compiler to show you layer 2 — the instructions.

```bash
go build -gcflags=-S main.go 2>&1 | head -40
```

**Expected output:** a wall of lines like `ADDQ $3, AX` and `CMPQ AX, $8`. Those are
instructions. You are looking at layer 2.

**Step 3.** Now change one thing. Add `-N` to disable optimizations:

```bash
go build -gcflags='-N -S' main.go 2>&1 | head -40
```

**You should see** many more instructions, and they will be simpler. The compiler stopped
combining and simplifying. Same source, same chip, different instruction sequence.

**Checkpoint:** you changed nothing about the program's meaning, and the instruction
sequence changed. What does that tell you about the relationship between source code and
instructions?

**Step 4.** If you are on an x86-64 machine, run `go env GOARCH`. Then, if you have an
ARM machine available, run it there. The value differs. That value is which instruction
set the compiler is targeting.

## What you should take away

1. A processor can only do what its wiring supports. That set of supported operations is
   the instruction set.
2. The instruction set is a **contract**, not a description. It says what, never how fast.
3. Your source code is not on the list. A compiler and an assembler put it there.
4. Layers 1 and 4 are portable. Layers 2 and 3 are not. That is why the instruction set
   is where portability ends.
5. Knowing the instruction set tells you what a chip *can* do. It does not tell you how
   fast it will do it. That is [Rung 5](level-5-why-the-isa-is-not-the-whole-story.md).

## See also

- [Rung 2 — One instruction, many numbers](level-2-simd-one-instruction-many-numbers.md) —
  what happens when one instruction operates on sixteen numbers at once, and why that is
  the single biggest lever in quantized inference.
- [What is a CUDA kernel?](../what-is-a-cuda-kernel.md) — the same journey, but ending on
  a graphics processor instead of a CPU.
- [Hardware portability](../hardware-portability.md) — how fak runs one forward pass on
  four different instruction sets without four different programs.
- [The fak/DOS glossary](../glossary.md) — one-line definitions of the vocabulary.

---

*[Full ladder (README)](README.md) · next → [Rung 2 — One instruction, many numbers](level-2-simd-one-instruction-many-numbers.md)*
