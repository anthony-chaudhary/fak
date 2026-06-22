# Contributing to fak / fleet

Thanks for contributing. **Autonomous coding agents are first-class contributors here**
— the same rules below bind a human, a Claude Code session, and a Codex/Cursor/Aider
run alike, because they're enforced *below* the agent layer by git hooks and the DOS
trust kernel, not by trust. This file is the durable contract for everyone — human and
agent alike.

This file is short on purpose; the deep public guides are `fak/ARCHITECTURE.md` (the
extension model), `fak/GROWTH.md` (the layering gates), and **`fak/EXTENDING.md`** — the
golden path if you're a researcher or team who wants to build an *optimization for a
subsystem* (a faster kernel, a new cache backend, a smarter admission rung) on fak
without forking the core.

## Licensing — read this before your first PR

The fak kernel is **Apache-2.0** (`LICENSE`); the project keeps layered-licensing
optionality open while Netra is the steward (see `CLA.md`). Two things gate every
contribution:

1. **DCO sign-off on every commit** — the Developer Certificate of Origin
   (<https://developercertificate.org/>). It certifies you wrote the change (or have the
   right to submit it). Add it with:

   ```
   git commit -s -m "your message"
   ```

   which appends a `Signed-off-by: Your Name <you@example.com>` trailer. The name/email
   must match your commits.

2. **CLA acceptance** — see `CLA.md`. The CLA grants Netra the copyright/patent license
   (including the sublicense right) that keeps the project's layered-licensing optionality
   open while Netra is the steward. Until an automated CLA-assistant is wired up, state in
   your first PR: *"I have read the CLA Document and I hereby sign the CLA."* Corporate
   contributors (employer owns the IP) need a Corporate CLA — contact Netra.

> **Why both, and why now:** the DCO is cheap provenance; the CLA is the relicense-enabling
> grant. Landing them *before the first external PR* is the one irreversible, time-sensitive
> licensing move. **The `CLA.md` text is a draft pending Netra's legal review**; the
> infrastructure is in place, the exact instrument is counsel's call.

Contributions are accepted **inbound = outbound**: your change is licensed to the public
under the same license that governs that part of the tree (today, Apache-2.0 for the
kernel), in addition to the CLA grant to Netra.

## Development workflow

- **Check you're set up first** — `python tools/extend_preflight.py` verifies the git
  guards, the stay-on-trunk state, the ship-stamp convention, and the extension gate
  entry points in one read-only command, then prints the golden path. Building an
  optimization for a subsystem? Start with [`fak/EXTENDING.md`](EXTENDING.md).
- **Touching docs? Keep the scorecard honest.** `python tools/docs_scorecard.py --scope
  reachable` grades every reader-reachable doc on five KPIs (freshness, link integrity,
  structure, readability, evidence) and counts *doc-debt* — the concrete defects a cold
  reader can hit (dead links, stale install pins, unresolved placeholders, missing titles,
  strawman-led headlines, orphans). It is read-only; a non-zero exit is a work-list, not a
  block. Regenerate the scorecard snapshot with `--markdown` after a docs pass. This is the whole-corpus analogue of
  `tools/readme_freshness_audit.py`, which checks the front page.
- **Touching the docs site or the FAQ? Keep discoverability honest.** `python
  tools/seo_aeo_scorecard.py --scope core` grades the published Pages surfaces on five
  SEO/AEO KPIs (title, description, headings, links, answerability) plus site-level
  checks (sitemap, canonical, JSON-LD, `llms-full.txt`) and counts *seo-debt*. If you
  changed the FAQ or `_config.yml`, re-run `python tools/gen_structured_data.py` to
  regenerate the JSON-LD (CI hard-gates that it is in sync). The discoverability
  **scores** are strategic and live in the private repo (`--transfer`); the tool and the
  read-only work-list are public.
- **Tests run through WSL, not native Windows** — `.\fak\test.ps1` (whole suite) or
  `.\fak\test.ps1 ./internal/<pkg>/`. `go build` / `go vet` work natively; only test
  *execution* is blocked on the Windows host. See the Windows note in
  [`fak/GETTING-STARTED.md`](GETTING-STARTED.md) for why. **Never commit a red tree.**
- **Add a feature as a leaf, not a core edit** — `python tools/new_leaf.py <name> --tier
  <tier> [--register]` stamps a conforming skeleton and wires the layering/registration.
  The frozen ABI (`internal/abi`) is additive-only and human-owned; everything else
  attaches through a `Register*` seam. `internal/architest` fails the build on an upward/
  cross-tier import, and `CLAIMS.md` requires every claim to carry exactly one of
  `[SHIPPED]`/`[SIMULATED]`/`[STUB]`.
- **Work directly on `main` — do not open a feature branch.** This is the
  single-source-of-truth operator law (`main-is-single-source`): every contributor,
  human or agent, commits to `main` in the main worktree. Creating a side branch or a
  new worktree to route around a dirty/diverged tree is the `OFF_TRUNK` anti-pattern that
  the trunk guard (`tools/githooks/reference-transaction`) and `dos.toml` actively
  **refuse** — so a doc that tells you to "branch first" would send you straight into a
  blocked commit. `git commit -- <paths>` and `git merge` / `git pull --no-rebase` never
  need a clean tree, so a dirty tree is never a reason to branch: pull/merge in place,
  wait for it to settle, or STOP and surface the blocker. Install the guards once per
  clone with `python tools/install_trunk_guard.py` (arms the trunk guard + the
  public-leak scan).
- **Commit small and by explicit path** (`git commit -- <paths>`, never `git add -A`).
  This is a shared multi-session tree — never stage a peer's uncommitted files. Pull
  before you start and again before each push; push promptly after each green commit.
- **Stamp every commit so it can be verified.** Fleet writes Conventional-Commits
  subjects (`feat(scope): …`, `fix(scope): …`, `docs(scope): …`) with a `(fak <leaf>)`
  trailer naming the lane the work lands in — e.g. `fix(gateway): treat same-tick ready
  as positive timeToReady (fak gateway)`. The DOS verify-referee binds "done" to that
  trailer (`dos verify fak <leaf>`); an un-stamped subject is deliberately *not* treated
  as a ship. Use a `docs(scope): …` subject for doc-only changes (a `fix(`/`feat(` prefix
  on a docs-only diff is read as an unwitnessed code claim). The lane names are the
  `[lanes]` in `dos.toml`. This is **in addition to** the DCO sign-off above, which is the
  separate legal-provenance trailer.

## Reporting issues

Use the GitHub tracker. Security-sensitive reports (a way past the capability floor or the
containment gate) should be raised **privately** — see [`SECURITY.md`](SECURITY.md) — rather
than filed as a public issue.
