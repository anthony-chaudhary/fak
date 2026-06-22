---
title: "fak memory dream: offline cleanup of finished context"
description: "An offline pass that re-screens, repairs, and prunes a finished session's core image against today's gate without replaying the transcript or summarizing it."
---

# MEMORY-DREAM-CLEANUP-RESULTS - offline cleanup for finished context

> Companion to `RECALL-RESULTS.md` and `CDB-RESULTS.md`. Those lanes made a finished
> session durable and debuggable as a core image. This lane adds the "human dream"
> pass: while the session is asleep, the kernel re-checks and tidies its memory
> image without replaying the transcript or asking a model to summarize it.

## The claim

A finished session can be auto-cleaned as a page-table-over-CAS image: re-screen
resident pages against today's gate, strand pages whose external witness was later
refuted, repair sealed descriptors from metadata only, surface duplicate page aliases,
and prune unreferenced swap bytes. The cleaned image still loads through
`recall.Load`, so later page-ins keep the same witness gate and fresh content
re-screen.

This leans into FAK-specific features:

- **Context MMU:** cleanup is a content-admission pass, not a text summarizer.
- **Content-addressed pages:** duplicate tool results remain distinct page-table
  frames but share one CAS blob.
- **Witness revocation:** a page admitted under a refuted source is pre-sealed out
  of the resident recall set.
- **Durable recall:** the pass operates on the sleeping core image and writes a new
  image, leaving the original available for audit.

## What shipped

- `recall.Dream(ctx, dir, DreamOptions)` loads a persisted core image, verifies it,
  runs the cleanup ledger, and optionally writes a cleaned output image.
- `fak dream` exposes the pass:

```
fak dream --dir cdb-image --out-dir dream-image --out dream-report.json
fak dream --dry-run --dir cdb-image
```

If `--dir` does not exist, the command seeds a deterministic demo image with a
duplicate benign page, one page admitted under a witness that is then revoked, and
one already-sealed injection page.

## Witnesses

- `TestDreamTightenedReScreenSealsFormerlyResidentPage`: an old weak image that
  treated obfuscated injection as benign is cleaned into a sealed page; the safe
  descriptor does not carry the obfuscated bytes, and page-in refuses with
  `ErrSealed`.
- `TestDreamSealsRevokedWitnessOutOfTheResidentSet`: a refuted witness removes a
  formerly resident page from recall candidates before any query can fault it in.
- `TestDreamPrunesUnreferencedCASAndPreservesBenignBytes`: orphan CAS bytes are
  pruned, duplicate aliases are surfaced, and benign pages still resolve
  byte-identical.
- `TestDreamDryRunDoesNotWriteOutput`: dry-run mode reports without writing a new
  image.

## Honest residue

This is deterministic memory hygiene, not generated consolidation. It does not
write model summaries, merge semantic memories, or improve the detector itself. Its
value is that the cleaned output is a smaller and safer core image under the same
mechanical trust gate. A future higher-level "dream" tier can add extractive
decision summaries, but those summaries must themselves be admitted as pages and
re-screened like any other result.
