# fak - normgate evasion-catch walkthrough

This example is the runnable face of the `normgate` claim in `CLAIMS.md`: a raw
context-MMU substring/regex screen misses simple obfuscations, while the registered
`normgate` + `ctxmmu` result-admission chain canonicalizes the result first and
quarantines the normalized threat.

The bundled `sample-battery.jsonl` is a small public representative battery, not the
private transcript-derived 24-row measurement. It demonstrates the same families behind
the shipped headline:

| corpus | raw ctxmmu | `ctxbench -chain` |
|---|---:|---:|
| public sample battery in this directory | 0/10 caught | 10/10 caught |
| recorded measurement cited in `CLAIMS.md` | 0/24 caught | 20/24 caught |

The remaining 4/24 are marker-free semantic paraphrases. That is an honest boundary for
this lexical gate: containment comes from the IFC/provenance layer, not from pretending
normalization understands intent.

## Run it

```bash
./run.sh
```

The script runs two commands against the same JSONL corpus:

```bash
FAK_NORMGATE=off go run ./cmd/ctxbench -chain -corpus examples/normgate-evasion/sample-battery.jsonl
go run ./cmd/ctxbench -chain -corpus examples/normgate-evasion/sample-battery.jsonl
```

It is deterministic and Go-only: no model, no network, no API key, no GPU, no Python, and
the sample battery completes in seconds.
Windows users can run the `.sh` launcher from WSL or Git Bash, or run the two `go run`
commands directly from the repository root.

## How to read it

The first run folds the registered chain with `FAK_NORMGATE=off`, so only `ctxmmu` makes
the result-side decision and the obfuscated rows are admitted. The second run keeps
`normgate` enabled: `normgate` runs first, scans normalized/decoded/reversed/de-separated
views, and quarantines every row in the sample.

In the shipped kernel, enabling `normgate` is one blank import:

```go
_ "github.com/anthony-chaudhary/fak/internal/normgate"
```

The production defconfig keeps that line in `internal/registrations`; this example imports
only the result-admission drivers it benchmarks so `ctxbench` stays lightweight.

The legacy `trigger bytes still in context` line only checks raw marker bytes after
admission. Obfuscated payloads do not contain raw markers, so use the `ALLOW` vs
`QUARANTINE` rows as the before/after witness.

## Honest scope

This demo does not claim normgate understands marker-free semantic paraphrases. It shows the
normalization families in the public battery; broader containment still comes from the IFC
and provenance layers.

## Battery rows

The sample covers:

- character-separated marker text
- zero-width and fullwidth marker text
- Cyrillic/Greek homoglyph marker text
- base64 and hex encoded marker text
- reversed marker text
- dot-separated `reveal your system prompt`
- credential formats the raw ctxmmu regex did not enumerate

See the captured output in `EXAMPLE-OUTPUT.md`.
