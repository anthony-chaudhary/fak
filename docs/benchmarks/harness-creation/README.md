# Harness-creation promotion study

Issues: #6911, #6809, executable evidence fold #6937.

This directory is the frozen, machine-readable study ledger for the phrases “create a
harness in 10 minutes” and “create your own harness this weekend.” Run:

```text
fak harness study creation --input docs/benchmarks/harness-creation/study.json
```

## Ten-minute task card

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
checked-in placeholder deliberately keeps both promotional claims `not_yet`. Maintainer
warm calibration remains visible but is excluded from the independent denominator.

## Maintainer calibrations

Calibration receipts are archived under `receipts/` so friction and technical headroom
remain reproducible. They are always marked non-independent and cannot satisfy either
promotion threshold. In particular,
[`linux-clean-shape-calibration-4`](receipts/linux-clean-shape-calibration-4/README.md)
replays the natural customization that exposed #6940 against its shipped fix.
