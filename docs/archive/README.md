# Archive: historical material and current replacements

**Primary audience:** readers who reached an archived document and need to find the maintained guidance that replaces it.
**Lifecycle:** archived; files here remain available as historical evidence and are outside maintained guidance.
**Generation:** retired or superseded material from any generation; the archive route itself is `gen/now` documentation.
**Authority:** the [documentation index](../index.md), [implementation generation map](../generation/), and replacement named by an archived file take precedence.
**Support:** use the [contributor route](../../CONTRIBUTING.md) for development help and the [operator guides](../operator/) for deployed operation.

**Next action:** open the archived file's replacement link and verify that the replacement covers your release and operating mode before following any instruction.

## How to use an archived file

Archived files preserve prior designs, retired instructions, and snapshots that remain useful for provenance, regression analysis, and decision history. Their value is the historical record. Current behavior comes from a maintained guide, contract, test, or implementation witness.

| What you found | What it establishes | Where to continue |
|---|---|---|
| A replacement or supersession link | The named destination is the current authority for the scope it states. | Open the replacement and check its release, backend, and mode. |
| A dated implementation snapshot | The repository or process looked that way at the stated date. | Compare it with current code and tests before drawing a present-tense conclusion. |
| A rejected or retired proposal | The option and its evidence were considered, then removed from active guidance. | Follow its decision record or current architecture route for the selected path. |
| No replacement link | The file is still historical evidence, not current instruction. | Start from the [documentation index](../index.md) or report the missing replacement through [CONTRIBUTING.md](../../CONTRIBUTING.md#reporting-issues). |

The current file in this directory, [`README-2026-06-25-before-fresh-start.md`](../notes/README-2026-06-25-before-fresh-start.md), is a snapshot of the repository front door before its 2026-06-25 refresh. Its maintained replacement is the root [`README.md`](../../README.md).

## Lifecycle and replacement contract

Material enters the archive when it is completed, superseded, rejected with evidence, or no longer has a current owner and witness path. Archiving changes authority, not history: preserve the original evidence and add the route to its replacement.

Each archived document should make these facts discoverable:

1. **Archived status and date** — when it left maintained guidance.
2. **Replacement** — the maintained document, implementation contract, or decision that now governs the same scope.
3. **Scope** — which release, mode, backend, or generation the historical record describes.
4. **Reason** — whether it was superseded, retired, completed, or rejected.

If no maintained replacement exists because the behavior was removed, link the retirement decision or current architecture boundary instead. Do not silently promote archived instructions back into use; move durable guidance into a maintained route and provide a new witness. The [generation map lifecycle](../generation/#lifecycle) explains promotion, demotion, and retirement.

## Mode and support boundary

An archived command can target a different platform, release, backend, or runtime mode from the one you use today. Treat its mode statement as historical scope, then select current instructions for your environment. The [dated-notes landing page](../notes/) covers evidence that remains under research or historical investigation without necessarily being retired into this archive.

For machine-oriented discovery, use [`llms.txt`](../../llms.txt). For current release guidance, use the [release index](../releases/).
