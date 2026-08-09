---
title: "Watch it: install to first DENY verdict — a recorded terminal cast (60 seconds)"
description: "A checked-in asciinema-style terminal recording of fak's 60-second proof: build the one static Go binary, run one preflight, and watch the default-deny capability gate refuse a dangerous tool call with verdict=DENY reason=DEFAULT_DENY — no API key, no model, no GPU. Plays in a browser or reads as an annotated transcript with a still frame, so you get the 'oh, that's it?' moment without running anything. Honest about being a recording; every line is real captured output."
slug: install-to-first-verdict-cast
keywords:
  - fak terminal cast
  - asciinema recording
  - default-deny capability gate
  - install to first verdict
  - fak preflight DENY
  - agent kernel demo
  - watch fak without running it
date: 2026-07-05
---

# Watch it: install to first DENY verdict

> **TL;DR:** a recorded terminal cast of fak's 60-second proof — `go build`, then one
> `fak preflight`, and the default-deny capability gate refuses a dangerous tool call with
> `verdict=DENY reason=DEFAULT_DENY`. **No API key, no model, no GPU.** Watch it in a
> browser or read the annotated transcript below; you never have to run anything to get the
> "oh, that's it?" moment.

This is dimension **C — Interactive & runnable demos** of the
[concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md). It
serves two concepts: **the one static Go binary, drop-in** and **the default-deny
capability gate + quarantine**. The runnable-command counterpart is the boundary proof in
the [README](../../../README.md#tool-call-controls); this page is the *watch-it-first*
version for a reader who has not cloned anything yet.

## This is a recording — read this first

Everything below is a **checked-in recording**, not a live terminal. The commands and their
output were **run on a real clean build and captured**, then annotated so the page reads
without audio. Nothing on this page executes when you open it. When you want to run it
yourself, the same commands are copy-pasteable from the [transcript](#annotated-transcript)
or from [`GETTING-STARTED.md`](../../../GETTING-STARTED.md) (Tier 0).

The cast file — [`install-to-first-verdict.cast`](install-to-first-verdict.cast) — is in
the standard [asciinema v2](https://docs.asciinema.org/manual/asciicast/v2/) JSON format, so
it plays in the asciinema web player or the `asciinema play` CLI:

```bash
asciinema play docs/adoption/casts/install-to-first-verdict.cast
```

## Still frame (the one screen that carries it)

The single moment worth remembering — the kernel refusing a dangerous call at the boundary,
with a structured reason, no model in the loop:

```console
$ ./fak preflight --tool refund_payment --args "{}"
verdict=DENY reason=DEFAULT_DENY by=monitor
#  ^ refund_payment is not on the allow-list, so the kernel fails closed.
```

## Annotated transcript

Read top to bottom. Lines beginning `$` are what was typed; the next line is the real
captured output; lines beginning `#` are annotation, not part of the run.

```console
# Step 1/3 — build the one static Go binary (Go 1.26+, two golang.org/x modules).
$ go build -o fak ./cmd/fak
$ ls -lh fak
-rwxr-xr-x  1 you  staff   44M  fak

# Step 2/3 — ask the kernel to preflight a dangerous tool call. Nothing runs;
# the capability floor decides at the boundary, structurally, with no model.
$ ./fak preflight --tool refund_payment --args "{}"
verdict=DENY reason=DEFAULT_DENY by=monitor
#  ^ refund_payment is not on the allow-list, so the kernel fails closed.
#    This is the first verdict: a DENY you can see in one command.

# Step 3/3 — contrast: an allow-listed read is admitted, same code path.
$ ./fak preflight --tool get_user_details --args "{}"
verdict=ALLOW reason=NONE by=monitor

# That's the whole boundary: default-deny, fail-closed, model-independent.
# Same binary next wraps your agent:  fak guard -- claude
```

## What just happened

- **One binary, zero setup.** `go build -o fak ./cmd/fak` produces a single static
  artifact — two `golang.org/x` modules in a 4-line `go.sum`, no Python, no CUDA. That one
  binary is the whole boundary.
- **The verdict is structural, not a guess.** `refund_payment` is denied because it is not
  on the allow-list — the floor fails closed. There is no classifier to fool and no model
  to prompt; the same `DENY` comes back offline, every time.
- **`DEFAULT_DENY` is a reason you can assert on.** Refusals carry a code from a closed
  vocabulary, so a fleet can act on the *why*, not just the *no*.

## Honest scope

- **A recording, labeled a recording.** No live execution happens on this page.
- **Assembly, not novelty.** A 29-claim prior-art audit scored **0/29 novel**; fak's
  contribution is the *assembly* of established primitives into one in-process gate.
- **Not a token engine.** fak fronts vLLM / SGLang / llama.cpp / a hosted provider; this
  demo proves the *boundary*, not throughput. No benchmark is claimed here.

## Go deeper

- Run it yourself, Tier 0: [`GETTING-STARTED.md`](../../../GETTING-STARTED.md)
- The runnable boundary proof in context: [README](../../../README.md#tool-call-controls)
- The printable one-pager: [concept card](../concept-card.md)
- The shareable card deck: [social storyboard](../social-storyboard.md)

## Verify

```
test -f docs/adoption/casts/install-to-first-verdict.cast   # the cast file exists
test -f docs/adoption/casts/README.md                       # this annotated page exists
fak score seo                                               # new doc does not red the SEO scorecard
```
