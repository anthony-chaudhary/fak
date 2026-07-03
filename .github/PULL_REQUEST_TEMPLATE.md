<!-- Thanks for contributing! First PR? Welcome — don't sweat the checklist;
     a maintainer will help you through anything unfamiliar.
     The full contract lives in CONTRIBUTING.md and AGENTS.md. -->

## What & why

<!-- One or two sentences. Link the issue if there is one: Fixes #NNN -->

## Evidence

<!-- What gate you ran and what it said — e.g. `.\test.ps1 ./internal/<pkg>/`
     for Go changes, or `python tools/docs_scorecard.py --scope reachable`
     for docs. "Verify, don't trust" applies to our own PRs first:
     a claim with no gate run is `not yet`, not done. -->

## Ship-discipline checklist

- [ ] Commits are signed off (`git commit -s`, DCO — see CONTRIBUTING.md)
- [ ] Conventional-Commits subject ending in a `(fak <leaf>)` stamp,
      e.g. `fix(gateway): treat same-tick ready as positive (fak gateway)`
- [ ] Staged by explicit path (`git commit -s -- <paths>`, never `git add -A`)
- [ ] The gate named under **Evidence** was actually run, output quoted
- [ ] Numbers are witnessed; anything simulated is labeled simulated
