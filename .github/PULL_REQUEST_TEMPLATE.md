<!-- Thanks for contributing! First PR? Welcome — don't sweat the checklist;
     a maintainer will help you through anything unfamiliar.
     New here? CONTRIBUTING.md has a fork-and-PR walkthrough. -->

## What & why

<!-- One or two sentences. Link the issue if there is one: Fixes #NNN -->

## Risk assessment

<!-- Keep this proportionate, but cover every line. "None" requires a concrete
     rationale. Reassess this block whenever scope or implementation changes. -->

- **Risk event:**
- **Exposed subject/system:**
- **Severity:**
- **Likelihood:**
- **Blast radius:**
- **Mitigations/containment:**
- **Rollback:**
- **Negative-path witness:**

## Evidence

<!-- What you ran and what it said. For Go changes, the tests for the package you
     touched (`go test ./internal/<pkg>/...`, or `.\test.ps1 ./internal/<pkg>/`
     on Windows). For docs, that the links you added or moved resolve.
     "Verify, don't trust" applies to our own PRs first: a claim with no check
     run is `not yet`, not done. CI runs the full gate on this PR. -->

## Checklist

- [ ] Commits are signed off (`git commit -s`, DCO — see CONTRIBUTING.md)
- [ ] The check named under **Evidence** was actually run, output quoted
- [ ] Numbers are witnessed; anything simulated is labeled simulated

<!-- ─────────────────────────────────────────────────────────────────────
     MAINTAINERS / IN-REPO AGENTS ONLY — committing directly to the shared
     `main` checkout. Ignore this block if you are contributing from a fork;
     a maintainer applies it when your change lands.

     - [ ] Conventional-Commits subject ending in a `(fak <leaf>)` stamp,
           e.g. `fix(gateway): treat same-tick ready as positive (fak gateway)`
           (leaf names are the `[lanes]` in dos.toml)
     - [ ] Staged by explicit path (`git commit -s -- <paths>`, never `git add -A`)
     - [ ] `make ci` green before the commit; `fak ci-preflight` after it
     The full contract is in AGENTS.md.
     ───────────────────────────────────────────────────────────────────── -->
