# Dated notes: evidence and history

**Primary audience:** readers investigating why a decision was made, what an experiment observed, or how an earlier implementation behaved.
**Lifecycle:** research and historical evidence; each note's own status or replacement statement controls its use.
**Generation:** mixed; a note may describe `gen/now`, a later generation, or a retired path.
**Authority:** maintained guidance linked from the [documentation index](../index.md) and [implementation generation map](../generation/) takes precedence.
**Support:** use the [contributor route](../../CONTRIBUTING.md) for development help and the [operator guides](../operator/) for deployed operation.

**Next action:** open a dated note only after locating its current authority or replacement link; if neither is named, treat the note as evidence and verify its claims against maintained guidance or current code.

## What belongs here

`docs/notes/` preserves dated investigations, audits, measurements, decision context, repair records, and implementation snapshots. These records make the evidence trail inspectable. Their dates and historical detail provide provenance rather than an automatic promise of current behavior.

Use this directory when you need to answer one of these questions:

| Reader question | How to use a dated note | Current-authority check |
|---|---|---|
| Why was a design or process chosen? | Read the note for its assumptions, alternatives, and evidence at the stated date. | Follow its decision, contract, runbook, or implementation link and confirm that source is still maintained. |
| What did an experiment or audit observe? | Preserve the stated environment, baseline, and provenance when citing the result. | Look for a newer witness before applying the result to today's product. |
| How did an earlier implementation behave? | Use the note to reconstruct history or diagnose a regression. | Follow any superseding document or inspect current code and tests before acting. |
| What should I do now? | Leave the notes route and use maintained contributor or operator guidance. | Start from the [documentation index](../index.md), which routes by audience and lifecycle. |

## Authority and lifecycle

A dated note can remain useful after its implementation changes. Its lifecycle determines how it may be used:

- **Research:** a hypothesis, comparison, or experiment. It supports further validation and does not establish shipped behavior by itself.
- **Current evidence:** a dated witness explicitly linked by a maintained contract or guide. Its scope is the named environment, version, mode, and date.
- **Superseded:** a newer authority replaces its guidance. Keep the note for provenance and follow the replacement before acting.
- **Archived:** the work is retired or retained only as history. Archived material is not current guidance.

When evidence promotes a note's conclusion into maintained behavior, put the durable instruction in a current contract, guide, test, or runbook and link back to the note for rationale. When current behavior changes, add a clear replacement link rather than rewriting the historical observation. The [implementation generation map](../generation/) explains how generation and lifecycle differ.

## Mode and generation

Notes may describe offline proofs, live model serving, native Windows control-host behavior, Linux or CI execution, a particular backend, or a specific release. Apply a note only to the mode and generation it names. A date does not select a runtime mode, and a `gen/now` label does not make every historical instruction current.

If a note has no explicit status, mode, generation, or replacement link, use it as historical evidence. Verify an operational step in a maintained guide or current code before running it.

## Finding a note

The curated [Notes & research section](../../INDEX.md#notes--research-docsnotes) groups useful evidence by topic. For machine-oriented discovery across the repository, use [`llms.txt`](../../llms.txt). The separate [archive](../archive/) holds material already moved out of maintained routes.
