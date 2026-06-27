# #scorecards Slack routing decision

Issue [#1003](https://github.com/anthony-chaudhary/fak/issues/1003) asks how the
~45 scorecard surfaces (the `*-score` family: code-quality, industry, persona,
agent-readiness, conflation, disambiguation, slop, guard-rsi, …) should reach the
fak scoreboard Slack workspace: one catch-all `#scorecards` channel, or a separate
channel per metric.

## Decision

Route every scorecard to one catch-all **`#scorecards`** channel by default, fed
by a single `FAK_SCORECARDS_CHANNEL`. Do not stand up ~45 per-metric channels up
front.

A metric **graduates** to its own channel only when it earns it. Until then it
posts to the catch-all.

Why:

- **Signal-to-noise of an empty fleet.** Most of the ~45 surfaces post on a
  /loop cadence or only when their debt number moves — a few lines a day each. Forty-five
  near-silent channels is worse discoverability, not better: a reader has to know which
  channel to watch, and the long tail stays empty. One channel keeps the whole
  scorecard story on a single timeline.
- **Discoverability beats partition at this volume.** The existing per-channel
  family (bench, dojo, capacity, steering, news, product, node-usage — each with
  its own `FAK_*_CHANNEL`) earned its split because each carries a distinct,
  sustained stream with its own audience. A scorecard's audience is the same person
  watching repo health; splitting it 45 ways fragments one audience across many
  rooms.
- **Alert fatigue is a volume problem, not a count problem.** Forty-five channels
  do not reduce the number of posts; they only scatter them. The fix for fatigue is
  to post on a real delta (debt changed, grade dropped), which the catch-all
  enforces by making every scorecard share one budget.

## The graduation rule

A scorecard moves out of `#scorecards` into its own `#<metric>-score` channel
(with its own `FAK_<METRIC>_CHANNEL`) when **both** hold:

1. **Sustained volume.** It posts often enough that it crowds the catch-all — as a
   rule of thumb, it is the dominant source in `#scorecards` over a sustained window
   (e.g. >10 posts/day for a week), not a one-off burst.
2. **A dedicated owner.** A specific person or loop owns acting on that metric and
   would watch a dedicated channel. A metric nobody owns does not need its own room.

Either condition alone is not enough: high volume with no owner is noise to move
off the catch-all by posting less often, not noise to give its own room; a keen
owner of a near-silent metric can watch `#scorecards` filtered by the metric name.

## Migration path

Graduating a metric is additive and reversible:

1. Create `#<metric>-score` in the scoreboard workspace; record its id as
   `FAK_<METRIC>_CHANNEL` alongside the rest of the channel family.
2. Point that one scorecard's post step at the new channel; leave every other
   scorecard on `FAK_SCORECARDS_CHANNEL`.
3. Keep a one-line pointer post in `#scorecards` when the graduated metric posts a
   notable delta, so the catch-all stays the index of record.

If a graduated channel goes quiet (the volume that justified it dries up), fold it
back: repoint the scorecard at `FAK_SCORECARDS_CHANNEL` and archive the room. The
catch-all is the home; a per-metric channel is a temporary lease the metric holds
only while it earns it.

## Status

This is the routing decision; the `FAK_SCORECARDS_CHANNEL` channel id and the
`fak scoreboard post` wiring are a follow-on and are **not yet wired** — no
scorecard posts to Slack on this scheme until that lands.
