---
title: "Voices: adopter quotes & case notes (honest, empty until real)"
description: "A disciplined scaffold for real adopter testimonials and case notes about fak/DOS — a submission template, an explicit no-fabricated-testimonials rule, and an empty state that reads as an invitation ('be the first — here's how to share your story') instead of a placeholder quote. This page ships empty on purpose; every entry that lands here is a real, attributable, adopter-approved story, never an invented one."
slug: adopter-voices
keywords:
  - fak testimonials
  - adopter quotes
  - case studies
  - social proof
  - verify don't trust
  - honest testimonials
  - no fabricated reviews
  - community stories
date: 2026-07-05
---

# Voices: adopter quotes & case notes

> **TL;DR:** this is where real adopter stories go — and it is **empty on purpose**.
> Social proof is decisive only if it is real, so this page ships as a *scaffold*,
> not a wall of invented quotes. Below is the one rule (no fabricated testimonials),
> a submission template a real adopter fills in, and an honest empty state. When the
> first real story lands, it drops straight into the [Voices](#voices) section — until
> then, that section says *be the first*, not a fake.

This is dimension **E — Social proof & community** of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md).
It serves the *verify, don't trust (DOS)* concept turned on our own marketing: the
same kernel that refuses a tool call's self-report until git evidence corroborates it
refuses a testimonial until a real, attributable person stands behind it. A quote we
wrote ourselves would be exactly the un-witnessed claim the whole project exists to
reject.

## The one rule: no fabricated testimonials

**Every quote or case note on this page is real, attributable, and adopter-approved —
or it is not here.** No invented personas, no "a developer at a large company said",
no paraphrase dressed up as a quote, no aspirational testimonial written in advance
"to show the format". If we would not stand behind it in front of the person named,
it does not ship.

This is not just good manners; it is the concept:

- fak's whole pitch is **capability over recognition** and **witness over
  self-report**. A dashboard that invents its own social proof would refute the pitch
  on its own front page.
- The [adoption-signals dashboard](signals.md) already draws the line between
  `OBSERVED` (a number a third party controls) and `WITNESSED` (an artifact fak
  authored). A testimonial is the strongest `OBSERVED` signal there is — and the
  easiest to fake. So the bar here is higher, not lower: a name, a link, and consent.
- The [personas gallery](personas.md) pitches *for* personas (who fak is for) and is
  careful to note it holds **no fabricated quotes FROM them**. This page is the other
  half: the place a real quote FROM a real adopter would live, once one exists.

If you are tempted to seed this page to "prime the pump", don't. An honest empty state
converts better than a quote a reader can smell is manufactured — and it keeps the one
asset (credibility) that a fake quote spends permanently.

## Voices

<!--
  REAL ENTRIES ONLY. This section is intentionally empty until a real, named,
  consenting adopter has a story to tell. Do not add a placeholder or example quote
  here — the empty state below is the correct content until then. When a real entry
  lands, delete the invitation block and add it using the template further down.
-->

> **Be the first.** No one has shared a public story yet — that is honest, and it is
> an opening. If you have run fak in front of your agent, put it in front of a policy
> floor, or wired the DOS trust gate into your loop, your experience is the first
> real data point here. [Here's how to share your story.](#how-to-share-your-story)

*(This invitation stays until a real entry replaces it. It is not a testimonial and
does not claim adoption — it is an open door.)*

## How to share your story

Two honest paths, both adopter-driven:

1. **Open a short issue or PR** titled `voice: <your name / handle>` against this file,
   filling in the template below. A maintainer confirms attribution and consent, then
   merges it into [Voices](#voices).
2. **Say it in a public place** (an issue thread, a post, a talk) and link it here —
   a real, quotable, public statement you made, cited to its source, beats anything we
   could write for you.

Either way the entry carries **your** words with **your** name and a link back to the
source, so any reader can verify it is real. We will never edit your meaning; if a
trim is needed for length, we show you first.

### Submission template

Copy this block, fill it in, and open a PR. Every field is required except *Numbers*
(leave it out rather than invent one):

```markdown
### <Name or handle> — <role, org (optional), or "independent">

> <Your quote, in your own words. One to four sentences. What you were trying to do,
> and what fak/DOS actually did for you. Concrete beats superlative.>

- **Using fak for:** <e.g. fronting a Claude Code agent with a capability floor / a DOS trust gate on a dispatch loop>
- **Since:** <month + year, approximate is fine>
- **Numbers (optional, witnessed only):** <a number YOU measured, with how — omit if you have none; do not use ours as if they were yours>
- **Source:** <link to the public post / issue / talk, or "submitted directly, consent on file">
- **Consent:** <"OK to quote publicly with attribution">
```

### The rules a submission must pass

- **Real and attributable.** A name (or a stable public handle) and, ideally, a link.
  Anonymous "a happy user" text does not ship.
- **Your own words.** We do not write your quote and attribute it to you.
- **Your own numbers.** If you cite a metric, it is one *you* observed, stated with how
  you measured it. Do not borrow fak's benchmark numbers as if they were your result.
  (For fak's own figures, the honest headline is the **tuned ~4.1×**, never the naive
  ~60× — see [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) — and simulated
  numbers are labeled simulated.)
- **Consent to publish.** Explicit "OK to quote publicly with attribution".
- **No market claim.** A single real story is one data point, not evidence of broad
  adoption. This page never aggregates its entries into a "widely adopted" claim.

### Case notes (longer than a quote)

If your story is a paragraph or a short write-up rather than a one-liner, the same
rules apply — put the longer form under [`docs/adoption/stories/`](stories/) (where the
[witnessed bench-stories](stories/the-real-4x.md) already live) and add a one-line,
linked pointer in [Voices](#voices). Keep the distinction the repo keeps everywhere:
witnessed vs observed, real vs simulated.

## What this page is NOT

- **Not a testimonial wall we pre-fill.** It is empty until real stories exist, by
  design. The empty state is the feature, not a gap to paper over.
- **Not an adoption claim.** Even full, it is a set of individual voices, not proof
  that fak is widely used. See the [adoption-signals dashboard](signals.md) for the
  honest, labeled view of what the numbers do and do not prove.
- **Not a place for our own numbers dressed as a user's.** fak's witnessed figures
  live in [`CLAIMS.md`](../../CLAIMS.md) and [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md);
  a voice here carries the *adopter's* experience, in their words.

## Related

- [Adoption-signals dashboard](signals.md) — the honest, labeled view of adoption
  signals (`OBSERVED` vs `WITNESSED`) and what each does NOT prove.
- [Who is fak for? A persona gallery](personas.md) — pitches *for* personas; the
  companion that (like this page) refuses to fabricate quotes *from* them.
- [Good first popularization tasks](good-first-tasks.md) — small ways to help, if you
  want to contribute before you have a story to tell.
- [Bench-story: the 4× that's real](stories/the-real-4x.md) — the witnessed number a
  case note should cite honestly, never the naive multiplier.
- [Concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md) —
  the framing all 50 tickets share.
