# Package anatomy: structural orientation beside maturity

`fak maturity` answers **how far a declared capability has progressed through its
lifecycle**. `fak maturity anatomy` answers a different question: **what static
structure must a maintainer understand inside one package, and where does that
package sit in the repository?** Neither command grades the other.

```bash
fak maturity anatomy internal/gateway
fak maturity anatomy internal/gateway --json
fak maturity anatomy --all --limit 10
```

## What is counted

| Group | Evidence | Interpretation |
|---|---|---|
| Shape | production files, functions, statements | Size and surface area, not difficulty by itself. |
| Flow | decision points, aggregate cyclomatic complexity, maximum function complexity, nesting | A McCabe-style syntax approximation. Each function starts at 1; `if`, loops, non-default switch/select arms, and boolean `&&`/`||` add decisions. |
| Outcomes | return sites, lexical success/error/ambiguous exits, conditions mentioning errors | Orientation for outcome handling. These are not feasible paths or runtime frequencies. |
| Contracts | terminating guard clauses, panics, separately counted assumption, expectation, invariant, and requirement comments, TODOs | Places likely to encode preconditions or unfinished expectations. Comment counts are deliberately lexical. |
| Documentation | exported symbols, documented exports, package comment | The public explanation surface; it does not score correctness or usefulness of prose. |
| Position | direct and transitive dependencies/dependents, shortest command distance, cycle membership, CLI reachability | Fan-out, fan-in, layering depth, dependency cycles, and whether the package is transitively reachable from a command. |

The JSON schema is `fak-maturity-anatomy/1`. It preserves ambiguous returns so a
number that merely looks successful is not presented as a happy path.

## Repository portfolio

`--all` analyzes every first-class `internal/<leaf>/**` entry declared by
`dos.toml`. It emits the full package rows in JSON and navigation rankings in
both formats:

- aggregate and per-function complexity;
- maximum single-function complexity;
- undocumented exports;
- direct and transitive dependencies/dependents;
- dependency-cycle membership and command distance;
- separately classified assumption and expectation comments; and
- lexical error exits.

It also reports declared roster entries that have no Go package. Those gaps are
useful architecture evidence rather than silently disappearing from the corpus.
Rankings are **not quality grades or a work queue**: large packages naturally
accumulate more branches, central packages naturally have more dependents, and
lexical counts can contain false positives. Use a ranking to choose where to
read next, then inspect the package-level evidence.

## What this does not claim

Static syntax cannot establish independent executable path count, path
feasibility, branch probability, test coverage, runtime hotness, or whether an
assumption is true. Those need execution traces, coverage/profile data, or a
semantic contract witness. The command says this in text and carries the same
caveats in JSON.


## Open-successor continuity

Every capability below the terminal `default + benchmark` point has one derived next-work row.
That row is keyed as `<lane>:<gap>` and must have **exactly one open** GitHub issue: `fak maturity route`
creates a missing issue, refreshes or keeps the canonical open one, reopens a closed one while that
same gap remains current, and closes duplicate managed issues. Once evidence advances the capability,
the full-portfolio route closes the obsolete key and opens the next rung; terminal capabilities emit
no row, and their obsolete managed successor is closed. Private-boundary rows remain visible but are
never filed publicly.

The command is live by default and covers the whole portfolio; use `--dry-run` to preview or
`--limit N` for a deliberately bounded repair. `.github/workflows/maturity-continuity.yml` runs the
same invariant weekly, default-on, before issue dispatch cycles consume the backlog. Repository
variable `FAK_MATURITY_CONTINUITY=0` is the emergency schedule kill switch; manual runs preview by
default.
