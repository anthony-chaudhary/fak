# Cache-honesty AST lint alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6171](https://github.com/anthony-chaudhary/fak/issues/6171) tracks real static-analyzer runs and independent resource/cost witnesses.

## Capability boundary

`internal/vcacheqa.HonestyLint` parses non-test Go files and reports cache-warmth phrases only when they occur in comments or string literals. It guards against reasoning that required context can be omitted because a provider is believed to retain it. This packet covers that AST lint only. The package's forced-miss reconciliation, witness chain, provenance fence, determinism checker, and aggregate gate are separate benchmark debt.

No equivalent first-class fak integration was found for this source-analysis capability. If one is added or discovered, completion requires a separate `fak + integration` arm rather than silently folding it into another row.

## Same corpus and oracle

Every arm analyzes byte-identical files:

- `live.go`: one planted violating comment at line 3 and one violating string at line 4, plus clean text;
- `clean.go`: valid live code with no violation;
- `excluded_test.go`: a planted phrase that must be excluded;
- `notes.txt`: a textual decoy that must be excluded.

The exact quality oracle is two ordered findings with correct file, line, and phrase, zero false positives, zero false negatives, zero location errors, and no parse failure.

## Arms

| Arm | Class | Local status | Honest boundary |
|---|---|---:|---|
| fak native vcache honesty AST lint | native | available | real `go/parser` + AST comment/string inspection |
| tuned non-test text scan | no-feature baseline | available | one-pass line scan limited to live `.go` files |
| go/analysis analyzer | external | unavailable | real compiled analyzer and diagnostics |
| Semgrep | external | unavailable | pinned Semgrep binary and rule |
| CodeQL | external | unavailable | pinned CodeQL database and query |
| golangci-lint custom analyzer | external | unavailable | pinned runner plus compiled plugin/module |

The text baseline intentionally represents the strongest simple incumbent for the fixed phrase policy, not a strawman whole-tree grep. The native arm's syntax attribution may matter on broader decoy corpora; the fixed local fixture does not establish that advantage.

Unavailable arms keep `Available=false` with zero measurements. Reimplementing their rule engines in Go would not witness those products.

## Completion metrics

- correctness: true/false positives and negatives, file/line/phrase accuracy, excluded-file behavior, parse failures;
- performance/resources: wall latency, bytes/s, CPU seconds, peak RSS, bytes read, process startup amortization;
- cost: rule-authoring/setup time, operator seconds, infrastructure/license charges, and total cost;
- reproducibility: pinned versions, exact rule/query/plugin, raw diagnostics, repetition policy, and independent result read-back.

`TestCompareHonestyLintLocalKeepsStaticAnalyzerAlternativesExplicit` locks the arms, local oracle, and unavailable-arm zeros. `BenchmarkHonestyLint` repeatedly runs the real AST path and validates the exact findings. No local timing is a cross-product claim, and no strongest alternative is ranked until #6171 carries all real runs.
