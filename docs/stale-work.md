# Stale-work review packets

`fak stale-work --json` is an advisory, read-only discovery pass over tracked Markdown and
claims. It emits schema `fak-stale-work/1`; it never edits, deletes, files, or closes anything.
Age contributes at most two points and **cannot** produce `review`: a semantic dependency commit
or an unpointed live version claim (the existing `docfreshrsi` detector) is required. Missing
evidence produces `abstain` (fail-to-abstain), not a guess.

```sh
fak stale-work --selfcheck
fak stale-work --json --limit 20
fak stale-work --path docs/operator-guide.md --json
```

The scanner uses Git's semantic commit history and the existing `docfreshrsi` version-claim
rules. Its dependency provenance is compatible with `modver` history; generated and historical
surfaces follow the exemptions used by `devfresh`; and `--state state.json` accepts the durable
dedupe/adjudication result of the `garden`/`issuepolicy` stage:

```json
{"open":{"docs/a.md":"#123"},"adjudicated":{"docs/b.md":"valid-through-v2"}}
```

Excerpts are whitespace-normalized, capped at 240 bytes, and hash-addressed. Dependency commits,
score components, dedupe key, issue state, proposed definition of done, and witness command are
included. Packet size is reported; no efficiency gain is claimed without a measured comparison
to the real `git log` + grep + full-file-read alternative.

## Two-stage contract

1. Scan and adjudicate. Historical, generated, vendored, already-open, and previously adjudicated
   entries are exempted/deduped. A reviewer validates executable/generated checks named by the
   packet's evidence and may file one dedicated issue per dedupe key using the supplied DoD.
2. Only that dedicated issue authorizes a fresh headless worker to edit or delete candidate
   content. The discovery packet itself is never authority to mutate a candidate.

The working-spine proof is captured in
[`docs/_witnesses/stale-work-live-2026-08-13.json`](_witnesses/stale-work-live-2026-08-13.json).
