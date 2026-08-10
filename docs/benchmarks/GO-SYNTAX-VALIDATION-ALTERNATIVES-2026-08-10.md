# Go syntax-validation alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6187](https://github.com/anthony-chaudhary/fak/issues/6187) tracks real toolchain runs and independent resource/cost witnesses.

## Capability boundary and workload

`internal/codelint` backs `fak codelint`. Its native Go pack parses standalone files with `parser.AllErrors`, reports normalized line/column findings, and deliberately does not type-check unresolved identifiers. This packet covers that Go syntax capability only. JSON validation, Python/CUDA subprocess packs, extension routing, summaries, and LSP framing remain separate benchmark debt.

Every arm receives four standalone files: valid source, one syntax error, multiple recoverable syntax errors, and syntax-valid code with an unresolved identifier. Correctness requires exact syntax-valid/invalid classification, no semantic false positive, and valid locations where the arm supports diagnostics.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native Go syntax pack | native | available |
| go/parser first-error-only | tuned syntax baseline | available |
| go test compile | external | unavailable |
| gofmt | external | unavailable |
| go vet | external | unavailable |
| staticcheck | external | unavailable |
| golangci-lint | external | unavailable |
| gopls diagnostics | external | unavailable |

No equivalent first-class fak integration was found. Real external tools remain zero-measurement rows until their binaries run the identical isolated corpus; in-process substitutions are not witnesses. Compile/type tools must be scored against the syntax-only oracle so unresolved-identifier diagnostics do not masquerade as correct syntax findings.

## Completion evidence

Complete arms report correct files, false and missed syntax errors, diagnostic count/location accuracy, startup and steady latency, throughput, CPU/RSS, input bytes, setup/operator time, and total cost. Versions, commands/configuration, raw diagnostics, and independent read-back must be pinned.

`TestCompareGoSyntaxLocalKeepsToolchainAlternativesExplicit` locks inventory, local classification, native all-error detail, and unavailable zeros. `BenchmarkGoSyntaxPackCorpus` runs all four real native file checks per iteration. Local timing is not a cross-tool claim; no tool is ranked until #6187 has complete runs.
