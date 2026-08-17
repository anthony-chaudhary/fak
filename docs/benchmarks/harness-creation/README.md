# Harness-creation promotion study

Issues: #6911, #6809, executable evidence fold #6937.

This directory is the frozen, machine-readable study ledger for the phrases “create a
harness in 10 minutes” and “create your own harness this weekend.” Run:

```text
fak harness study creation --input docs/benchmarks/harness-creation/study.json
```

## Representative pack blueprints

Before timing, a participant may select one bounded need from `fak harness gallery list`
or state an equivalent need in one sentence. The gallery provides concrete readonly
support, coding workspace, cited research, and incident-operations examples with explicit
required and excluded capabilities, starter artifacts, ten-minute spines, weekend
extensions, and witnesses. Gallery selection does not count as independent authorship for
the weekend track; the participant must still implement the selected public-seam extension.
Independent operators begin with the [participant handoff](participant-handoff.md) and machine-check receipts against [the template](participant-receipt-template.json) before appending any study row.


## Paired parity question

The tuned Mastra calibration proves only that the alternative is runnable; it is not a
parity result. Before independent timing, the study freezes a matched non-inferiority
question: at least two unfamiliar builders each run both `fak` and `baseline` arms under
one random `pair_id`. The two-pair minimum is counterbalanced: one `fak-first` and one
`baseline-first`, with one frozen task digest and privacy-safe machine identity per pair,
and fresh directories and empty caches for each arm. Failures remain
in the denominator, and fak is supported only when
it has no fewer successful paired arms and the median elapsed-time ratio
`fak / baseline` is at most **1.25**. Missing arms remain visible as incomplete pairs.
This small denominator is a bounded creation-workflow witness, not statistical or product-
quality equivalence.

`fak harness study creation --input study.json` emits the parity denominator, arm
successes, median ratio when both arms succeed, and `supported`, `not_yet`, or `refuted`.
The checked-in study intentionally reports `not_yet` until independent paired receipts
are archived.

## Ten-minute task card

Every paired run uses [`paired-task-card.md`](paired-task-card.md), whose SHA-256 is frozen
as `protocol.task_digest` in `study.json`. Hash the exact checked-in UTF-8 bytes; a receipt
with another digest is inadmissible even when both arms agree with each other.

Prerequisite environment: a fresh machine or VM with Go installed, working network, no
fak checkout, no fak binary, and empty relevant Go build/module caches. Record exact OS,
architecture, Go version, cache state, and retrieval identity before opening this card.

The clock starts when the participant opens this task card. Within 600.0 elapsed seconds:

1. Retrieve the declared fak artifact.
2. State one bounded harness need in one sentence.
3. Generate the product outside a fak checkout.
4. Represent the need by editing only field values returned by `product.DefaultConfig` and, if needed, the body of `product.OfflineReply`; preserve those signatures.
5. Build the generated product and pass its offline `--selfcheck`.
6. Read `harness.lock.json` and record its exact upgrade command.

The clock stops only after all six outcomes exist. A timeout, command failure, abandoned
run, wrong-file edit, failed selfcheck, or missing upgrade command is a failed eligible
attempt and remains in the denominator. Work may continue after 600 seconds to diagnose
friction, but cannot become a ten-minute success retroactively.

Assistance is the task card and command `--help` output only. Every human hint or external
document opened after clock start is timestamped in the raw receipt and reported
separately. Participant identifiers are random privacy-safe slugs, never names or email
addresses.

## Weekend task card

Start from the ten-minute product and choose exactly one extension before timing: a skill
pack, provider/tool adapter, or branded UI over a public fak seam. The result must be
independently authored, pass the relevant conformance/selfcheck, preserve generated versus
user-owned boundaries, and record build, upgrade, and rollback commands. A maintainer demo
or a product assembled by the implementation team is calibration, not an eligible result.

## Raw receipt minimum

Each referenced receipt records: random run and participant slugs; participant class and
prior fak-internals familiarity; track; artifact and checksum/commit identity; OS, CPU,
Go version, network and cache state; monotonic start/stop and elapsed seconds; exact
commands and exits; files changed; rebuild count and duration; outcome; help requests;
privacy-safe transcript path; and, for weekend runs, independent authorship and conformance.

The baseline must be runnable, tuned, selected, and frozen before participant timing. The
frozen next-best alternative is [`create-mastra@1.25.0`](baselines/create-mastra-1.25.0/README.md);
its clean calibration is [archived here](receipts/windows-clean-mastra-calibration-1/README.md).
Baseline readiness makes future independent runs eligible but does not support either
claim. Maintainer calibration remains excluded from the independent denominator.

## Maintainer calibrations

Calibration receipts are archived under `receipts/` so friction and technical headroom
remain reproducible. They are always marked non-independent and cannot satisfy either
promotion threshold. In particular,
[`linux-clean-shape-calibration-4`](receipts/linux-clean-shape-calibration-4/README.md)
replays the natural customization that exposed #6940 against its shipped fix.
