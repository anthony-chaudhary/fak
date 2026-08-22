# Native-harness dynamic-instruction dogfood — 2026-08-22

**Verdict:** PASS. The shipped dynamic-instruction spine composed this repository's live issue and working-tree facts into two real instruction turns while preserving the kernel-owned stable prefix. The run surfaced no product defects, so no defect issues were filed.

- Issue: [#8046](https://github.com/anthony-chaudhary/fak/issues/8046)
- Shipped spine: `95404b4fe0d9998fee7bd8a6714d43990d034cd2`
- Live committed base: `56c32a0364e49a78af0436b122957825ff19bddc`
- Machine-readable receipt: [`native-harness-dynamic-instructions-dogfood-2026-08-22.json`](../_witnesses/native-harness-dynamic-instructions-dogfood-2026-08-22.json)
- Defect marker-key prefix: `fanout-harnesskit-dogfood-self-run`

## User outcome

For a native-harness operator, the public `pkg/harnesskit` contract and the kernel adapter can turn current repository state into attributable run-, thread-, and turn-scoped instructions. This run advances managed context and integrated operations without claiming model-quality or token savings.

## Failing-before evidence

Before the run, committed `HEAD` had no #8046 live-run artifact:

```text
git ls-tree -r --name-only HEAD docs/notes docs/_witnesses |
  rg -i '8046|native-harness.*dogfood|dynamic-instruction.*dogfood'
FAIL_BEFORE: no committed #8046 live-run readout or witness
exit 1
```

That absence was the issue's reproducible gap: the API spine existed, but no run against fak's own live work had been captured.

## Method

The run used a clean archive of committed `HEAD`, excluding every peer's uncommitted Go file, and called the shipped `internal/harnessinstructions.Resolve` seam. The one-off driver read only these live facts from the shared checkout and GitHub:

- repository `github.com/anthony-chaudhary/fak`, branch `main`, exact `HEAD`, and dirty-entry count;
- issue #8046 number, title, state, and public URL;
- this owner lifecycle's two declared output paths.

It provided three typed fragments on each turn:

| Fragment | Source | Lifetime | Residency | Live meaning |
|---|---|---|---|---|
| `issue-contract` | `github/issue-8046` | run | overlay | Current issue number, state, and title |
| `repository-state` | `git/live-main` | thread | overlay | Current branch, commit, dirty-entry count, and peer-WIP boundary |
| `turn-objective` | `issue-owner/live-phase` | turn | ephemeral tail | Capture first, then reconcile |

The run was bounded work: one existing spine, one live input set, two turns, one receipt, and one readout. It launched no child workers and changed no implementation paths.

## Observed result

| Check | Observation | Verdict |
|---|---|---|
| Live data composed | `main` at `56c32a0364e49a78af0436b122957825ff19bddc`; 68 dirty entries observed | PASS |
| Stable prefix preserved | Both turns: `blob-sha256:0eab07f1e7352ddc46ea4812761988a4da4bcb71238a3ee434c4c73a219b7169` | PASS |
| Dynamic turn applied | Full prompt digest changed from `sha256:a478b5760b64e7d6c59d6b0d2f293b392aee9df3df1044aad20a674eeac3b4ef` to `sha256:5a555076b1e106bf02845f59ed2d4159691dca4079c7c04c9dbafbd7eb3380ef` | PASS |
| Kernel prefix audit | `ok` on both turns | PASS |
| Typed provenance | Three inclusion decisions on each turn | PASS |

The receipt contains the exact realized prompt values, normalized fragments, content digests, inclusion decisions, byte/token estimates, public contract, and all five checks.

## Defect reconciliation

The receipt's `defects` array is empty. No failed invariant, malformed realization, authority crossing, missing provenance, or prefix-audit failure was observed. Therefore there are no new defect issue numbers to file under `fanout-harnesskit-dogfood-self-run`; filing a placeholder ticket would invent work the run did not surface.

This is deliberately a dogfood witness, not a performance or quality benchmark. The observed dirty-entry count describes the live shared checkout at capture time; it is not a health finding or an input to the composition verdict.
