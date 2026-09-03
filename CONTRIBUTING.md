# Contributing to fak

> **Primary audience:** people preparing and shipping a code or documentation change to
> this repository. Autonomous coding agents follow the same contributor contract because
> the repository gates enforce it below the agent layer.

This is the maintained contributor route: establish a working checkout, choose the
work route for the change, run the matching checks, and land only a fully gated,
explicitly scoped change on `main`. The sections are sequential: complete the single
next action below first, then continue to route selection and setup. Product evaluation
starts at [`README.md`](README.md);
production operation starts at [`docs/operator/`](docs/operator/).

**Next action:** from the repository root, run `go version` and confirm Go 1.26 or newer
(the `go.mod` toolchain directive can fetch the required toolchain automatically).

## First: which contributor are you?

This page serves two audiences with different mechanics. Pick your row before reading on —
several rules below apply to only one of them.

| You are… | How your change lands | Read |
|---|---|---|
| **An outside contributor** (no write access to this repository) | Fork → branch on your fork → pull request | [Fork and open a pull request](#fork-and-open-a-pull-request), then the route table and the licensing section |
| **A maintainer or an in-repo agent** (working in the shared `main` checkout) | Commit directly to `main` by explicit path | Everything, including [Set up and verify the checkout](#set-up-and-verify-the-checkout) and the trunk rules in [Development workflow](#development-workflow) |

If you are not sure, you are an outside contributor. The direct-to-`main` workflow described
later requires push access to `anthony-chaudhary/fak`; it is not the path for a first PR.

## Fork and open a pull request

The standard GitHub flow. Nothing in this section needs write access to this repository, and
the trunk guard never sees your fork.

```bash
# 1. Fork on GitHub, then clone YOUR fork
git clone https://github.com/<your-username>/fak.git
cd fak
git remote add upstream https://github.com/anthony-chaudhary/fak.git

# 2. Branch on your fork — branches are fine here; the no-branches rule
#    below governs the maintainers' shared checkout, not your fork.
git checkout -b my-change

# 3. Make the change, then check it (see "What your PR needs to pass")
go build ./... && go vet ./...
go test ./internal/<package-you-touched>/...

# 4. Commit with a DCO sign-off — this one IS required (see Licensing)
git commit -s -m "docs(readme): fix the install path"

# 5. Push to your fork and open a PR against anthony-chaudhary/fak main
git push origin my-change
```

Then open the pull request on GitHub and fill in the contributor section of the template.

### What your PR needs to pass

Run the check that matches your change; CI runs the full gate on the PR, so you do not need
to reproduce it locally.

| Your change | Run before pushing |
|---|---|
| Go code | `go build ./...`, `go vet ./...`, and the tests for the package you touched (`fak test ./internal/<pkg>/` or `.\test.ps1 ./internal/<pkg>/` on Windows) |
| Documentation | Confirm every link you added or moved resolves |
| Anything | `git commit -s` (DCO sign-off — enforced) |

You do **not** need `python tools/install_trunk_guard.py`, `make ci`, or `fak-dev ci-preflight`
for a forked PR. Those are the shared-checkout gates described further down; a maintainer
runs the equivalent before your change lands on `main`.

Keeping your fork current:

```bash
git fetch upstream && git merge upstream/main
```

**Windows:** `go build` and `go vet` work natively. Running the *test suite* can hit an OS
Application-Control policy that blocks freshly compiled test binaries — that is an OS quirk,
not a code failure. Run the tests under WSL if you need them; see the Windows note in
[`GETTING-STARTED.md`](GETTING-STARTED.md).

## Connect the change to an operator problem

Before choosing an implementation route, use the canonical
[problems fak exists to solve](docs/problems-we-solve.md) in its two distinct roles. Classify
the work's **problem centrality** (`Core`, `Enabling`, `Stewardship`, or `Peripheral`) to
show how directly it advances fak's connected user problems, then run **all four** P1-P4
checks—managed context, net-true efficiency, bounded adaptation, and integrated operations.
Do not pick one P-number as a priority label. Fill **For / Problem / Today / Better because /
Witness** as well. This keeps a leaf, issue, or plan anchored to removed operator burden
while infusing the product values through design, implementation, proof, and review. Product
direction is not shipped evidence: use [`CLAIMS.md`](CLAIMS.md) and the relevant captured
witness for completion claims.

## Choose the route for your change

| Change | Start here | Proof before landing |
|---|---|---|
| Existing Go package or CLI behavior | The owning package under `internal/` or `cmd/fak/` | A behavior test that fails before the fix and passes after it |
| New subsystem capability | [`EXTENDING.md`](EXTENDING.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md) | A leaf implementation, architecture gate, correctness witness, and measured gain where performance is claimed |
| Documentation | The audience route in [`INDEX.md`](INDEX.md) | Every changed local link resolves, and an independent reader correctly restates the audience, page job, choices, and next action with no ambiguity; visual defects also require a captured render |
| CI/CD contract | [`docs/ci/ci-spec-change-migration.md`](docs/ci/ci-spec-change-migration.md) | All consumers migrated together and `fak-dev ci-preflight` on the committed tip |

New here and want a bounded first change? Choose an open
[`good first issue`](https://github.com/anthony-chaudhary/fak/labels/good%20first%20issue)
or a doc-sized item from the
[good-first popularization board](docs/adoption/good-first-tasks.md). Do not invent work
from an old planning note when a current issue or authority disagrees.

## Set up and verify the checkout

> **Maintainers and in-repo agents only.** This section describes the shared `main` checkout.
> If you are contributing from a fork, you have already done what you need — see
> [What your PR needs to pass](#what-your-pr-needs-to-pass) — and can skip to
> [Licensing](#licensing--read-this-before-your-first-pr).

1. **Maintainers and in-repo agents only — once per clone.** Work from the repository root
   on `main`; this repository does not use contributor feature branches. Install the
   repository hooks with `python tools/install_trunk_guard.py` if
   `git config --get core.hooksPath` does not report `tools/githooks`. The guard polices
   direct commits to *this* shared `main`; it has no role in a fork, so skip it entirely if
   you are opening a pull request.
2. **Everyone — the compile check.** Compile without writing an in-tree binary:
   `fak-dev buildcheck --vet`. It exists because this checkout is shared and a peer's untracked
   Go files would otherwise change the answer; from a fork, plain
   `go build ./... && go vet ./...` is the same check. If no usable `fak` binary exists yet,
   use a unique temporary output (`go build -o <temp-path> ./cmd/fak`), then use that binary
   for subsequent checks.
3. **Everyone runs *a* proof; only direct-to-`main` work runs the full one.** Run the proof
   matched to the change. `make test-fast` is the optional short feedback gate (the ~2-second
   smoke tier: build, vet, and `go test -short ./...`, skipping the weight-backed model
   witnesses); **`make ci` is the required pre-commit green gate** for build, vet, tests, and
   claims lint. **Budget for it:** `make ci` chains 20 targets, including the full
   `go test ./...` over the weight-backed model oracle that CI itself budgets at
   `-timeout=25m` — so plan on **tens of minutes**, not a pause. Keep `make test-affected`
   (a 30-second budget over the changed packages and their importers) as the inner loop and
   save `make ci` for the commit itself. From a fork you do not run it at all: the tests for
   the package you touched are enough, and CI runs the full gate on the PR — see
   [What your PR needs to pass](#what-your-pr-needs-to-pass). On native Windows, run
   tests through `./test.ps1` under WSL because host application control blocks newly
   compiled test executables. Build and vet remain native-safe.
4. **Maintainers and in-repo agents only — direct-to-`main`.** After the explicit-path commit
   and before push, run `fak-dev ci-preflight` as the
   **required committed-tip gate** in a clean temporary checkout, independent of the
   peer-dirty working tree. Thus "green" means both gates in order: `make ci` before the
   commit and `fak-dev ci-preflight` after it. A forked PR has no push-to-`main` step, so this
   gate does not apply to it.

Build-profile details and platform commands live in
[`docs/dev-tooling.md`](docs/dev-tooling.md). Its [proof-depth matrix](docs/dev-tooling.md#match-proof-to-activation-depth) is authoritative for distinguishing a working overlay, prospective native link, committed tip, installed copy, and running activation; a pass at one depth never proves a later depth. `AGENTS.md` is the machine-oriented authority
for shared-tree recovery, proof capture, and guarded commit mechanics.

## Route context

| Dimension | Current contract |
|---|---|
| Mode | Two source-contribution routes: a forked pull request, which needs no write access, and direct commits in the shared `main` checkout. Installed-binary operation belongs to the operator route. |
| Generation | This page is the current `gen/now` contributor front door. Historical release notes and planning records do not override it. |
| Lifecycle | **From a fork:** choose an issue → branch on your fork → implement one coherent change → run the checks for what you touched → `git commit -s` → open a pull request. **In the shared checkout:** choose an issue → implement one coherent change → capture matched proof → run the full gate → commit explicit paths → push `main`. |
| Support | Contributor setup, architecture choice, tests, commit rules, and issue reporting are covered here; product use and production recovery are routed elsewhere. |

## Licensing — read this before your first PR

The fak kernel is **Apache-2.0** (`LICENSE`); the project keeps layered-licensing
optionality open while Netra is the steward (see `CLA.md`). Every contribution is gated by
two things — one binding today, one staged:

1. **DCO sign-off on every commit** *(binding today, enforced by a git hook)* — the
   Developer Certificate of Origin (<https://developercertificate.org/>). It certifies you
   wrote the change (or have the right to submit it). Add it with:

   ```
   git commit -s -m "your message"
   ```

   which appends a `Signed-off-by: Your Name <you@example.com>` trailer. The name/email
   must match your commits.

2. **CLA acceptance** *(staged; the text is a **draft pending Netra's legal review and not
   yet binding**)* — see `CLA.md`. The CLA grants Netra the copyright/patent license
   (including the sublicense right) that keeps the project's layered-licensing optionality
   open while Netra is the steward. The sign-off ritual is staged now so the infrastructure
   is ready before the first external PR, but the exact instrument may change on counsel's
   review before it is declared final. Until an automated CLA-assistant is wired up, state
   in your first PR: *"I have read the CLA Document and I hereby sign the CLA"* —
   acknowledging that the text is a draft subject to revision. Corporate contributors
   (employer owns the IP) need a Corporate CLA — contact Netra.

> **Why both, and why now:** the DCO is cheap provenance; the CLA is the relicense-enabling
> grant. Landing them *before the first external PR* is the one irreversible, time-sensitive
> licensing move. **The `CLA.md` text is a draft pending Netra's legal review**; the
> infrastructure is in place, the exact instrument is counsel's call.

Contributions are accepted **inbound = outbound**: your change is licensed to the public
under the same license that governs that part of the tree (today, Apache-2.0 for the
kernel), in addition to the CLA grant to Netra.

## Dependency-heavy Go tools

New repository tooling is Go, not Python. If a useful tool needs dependencies outside the
root module's reviewed budget, do not add them to the root and do not fall back to Python.
Use a quarantined nested module:

- keep a small, stdlib-only façade in `tools/<name>/` so callers retain
  `go run ./tools/<name> ...`;
- place the dependency-heavy implementation below it, conventionally
  `tools/<name>/<dep-heavy>/go.mod`;
- have the façade enter/invoke that module explicitly, hiding the module boundary;
- run `go test ./internal/dependencyquarantine` before submitting.

The gate pins the root `go.mod` require set and `go.sum`, walks for nested `go.mod` files
rather than maintaining a list, checks every `tools/` façade for non-stdlib imports, and
builds/tests every discovered nested module in CI. Any intentional root dependency-budget
change must update the allowlist in the same reviewed change.
## Development workflow

> **Mixed audience — check the marker on each bullet.** Four of the eight below are
> shared-checkout mechanics a forked pull request never performs: the setup route, the
> work-directly-on-`main` trunk rule, explicit-path commits, and the verification trailer.
> The other four apply to anyone touching this code, fork included — the two documentation
> scorecards, the leaf-extension path, and the Windows note about running tests under WSL.
> If you are contributing from a fork, your required checks end at
> [What your PR needs to pass](#what-your-pr-needs-to-pass); read the rest for context only.

- **Start from the setup route above, and build the working spine before broad proof or optimization.**
  For new work, follow [`docs/spine-first-defaults.md`](docs/spine-first-defaults.md): connect
  the smallest applied end-to-end path, capture its witness, then expand edge/platform/soak
  coverage and optimize against that real baseline. For a subsystem optimization, continue
  through [`EXTENDING.md`](EXTENDING.md); for an existing package, run its focused tests before
  the repository-wide gate.
- **Touching docs? Keep the scorecard honest.** `python tools/docs_scorecard.py --scope
  reachable` grades every reader-reachable doc on five KPIs (freshness, link integrity,
  structure, readability, evidence) and counts *doc-debt* — the concrete defects a cold
  reader can hit (dead links, stale install pins, unresolved placeholders, missing titles,
  strawman-led headlines, orphans). It is read-only; a non-zero exit is a work-list, not a
  block. Regenerate the scorecard snapshot with `--markdown` after a docs pass. This is the whole-corpus analogue of
  `tools/readme_freshness_audit.py`, which checks the front page.
- **Touching the docs site or the FAQ? Keep discoverability honest.** `fak score seo
  --scope core` grades the published Pages surfaces on six
  SEO/AEO KPIs (title, description, headings, links, links_crawlable, answerability)
  plus site-level checks (sitemap, canonical, JSON-LD, `llms-full.txt`, citation_links)
  and counts *seo-debt*. Beyond the presence checks (is the meta/link/JSON-LD there?)
  it runs SUCCESS checks (does it WORK for the consumer): `links_crawlable` flags a link
  that resolves on disk but 404s on the live site, the corpus-wide meta-distinctness pass
  flags a duplicate `<title>`/description search can't tell apart, `citation_links`
  flags a dead llms.txt-map or self-repo GitHub link an answer engine would follow, and
  `llms_full_navigable` flags inlined local links that lost their source-page base path.
  If you
  changed the FAQ or `_config.yml`, re-run `python tools/gen_structured_data.py` to
  regenerate the JSON-LD (CI hard-gates that it is in sync). The discoverability
  **scores** are strategic and live in the private repo; the verb and the
  read-only work-list are public. (This check was a `tools/seo_aeo_scorecard.py` script
  before it was ported to the `fak score seo` verb; the script is gone, so an older
  instruction naming it will fail.)
- **Tests run through WSL, not native Windows** — from the repository root, `.\test.ps1`
  (whole suite) or `.\test.ps1 ./internal/<pkg>/`. The selected WSL distro needs Go 1.26+; the Windows-host Go install is not visible inside WSL, and `GOTOOLCHAIN=auto` cannot bootstrap without a base `go` command. Follow the [`One-time Windows developer setup`](#one-time-windows-developer-setup). `go build` / `go vet` work natively; only test
  *execution* is blocked on the Windows host. See the Windows note in
  [`GETTING-STARTED.md`](GETTING-STARTED.md) for why. **Never commit a red tree.**
- **Add a feature as a leaf, not a core edit** — `fak new-leaf <name> --tier
  <tier> [--register]` stamps a conforming skeleton and wires the layering/registration.
  The frozen ABI (`internal/abi`) is additive-only and human-owned; everything else
  attaches through a `Register*` seam. `internal/architest` fails the build on an upward/
  cross-tier import, and `CLAIMS.md` requires every claim to carry exactly one of
  `[SHIPPED]`/`[SIMULATED]`/`[STUB]`.
- **Work directly on `main` — do not open a feature branch.** *(Shared-checkout rule. It does
  not apply to a fork: branch freely there — see [Fork and open a pull
  request](#fork-and-open-a-pull-request).)* This is the
  single-source-of-truth operator law (`main-is-single-source`): everyone with write access,
  human or agent, commits to `main` in the main worktree. Creating a side branch or a
  new worktree to route around a dirty/diverged tree is the `OFF_TRUNK` anti-pattern that
  the trunk guard (`tools/githooks/reference-transaction`) and `dos.toml` actively
  **refuse** — so a doc that tells you to "branch first" would send you straight into a
  blocked commit. `git commit -- <paths>` and `git merge` / `git pull --no-rebase` never
  need a clean tree, so a dirty tree is never a reason to branch: pull/merge in place,
  wait for it to settle, or STOP and surface the blocker. Install the guards once per
  clone with `python tools/install_trunk_guard.py` (arms the trunk guard + the
  public-leak scan). The *one* sanctioned worktree is the detached per-worker
  build-isolation tree (#1334 / epic #3165): pinned at trunk HEAD, it lands on `main` via
  the serialized `land_worktree_diff` under the worker's lane lease through `fak worktree
  worker prepare|land|reap|list`, so it is not a branch and never trips `OFF_TRUNK`. That
  is the only worktree exception; feature branches and off-trunk commits remain refused.
- **Commit small and by explicit path**. Prefer the repo verbs: `fak sweep [--json]`
  indexes a dirty shared tree by lane, `fak sweep --apply --lane <lane> -m "<subject>"`
  lands a whole lane group, and `fak commit --path <p> -m "<subject>"` lands a narrower
  slice through the same guarded path. Raw `git commit -- <paths>` is the fallback when the
  binary is unavailable; `git add -A` is never allowed. This is a shared multi-session tree
  — never stage a peer's uncommitted files. **Default: once the tree is green, commit AND
  push — you don't wait to be asked.** Green is `make ci` (build + vet + test +
  claims-lint; tests run under WSL via `./test.ps1` on Windows), after which the
  commit-message / file-admission / public-leak / trunk guards run as git hooks. Pull
  before you start and again before each push; push promptly after each green commit, on the
  trunk, with `git commit -s` (DCO) — never force-push. If a guard refuses (`OFF_TRUNK`), a
  peer merge is in flight, or a blocker stands, reconcile in place or STOP first.
- **Stamp every commit so it can be verified.** Fleet writes Conventional-Commits
  subjects (`feat(scope): …`, `fix(scope): …`, `docs(scope): …`) with a `(fak <leaf>)`
  trailer naming the lane the work lands in — e.g. `fix(gateway): treat same-tick ready
  as positive timeToReady (fak gateway)`. The DOS verify-referee binds "done" to that
  trailer (`dos verify fak <leaf>`); an un-stamped subject is deliberately *not* treated
  as a ship. Use a `docs(scope): …` subject for doc-only changes (a `fix(`/`feat(` prefix
  on a docs-only diff is read as an unwitnessed code claim). The lane names are the
  `[lanes]` in `dos.toml`. This is **in addition to** the DCO sign-off above, which is the
  separate legal-provenance trailer.

## Good first issues — where to start

New here? Browse the [`good first issue`](https://github.com/anthony-chaudhary/fak/labels/good%20first%20issue)
label for scoped, well-defined starter work. If nothing's open, these are the standing
on-ramps — each is additive, ships green through `make ci`, and touches no enforced
guard:

- **Add a per-agent integration recipe** under [`docs/integrations/`](docs/integrations/)
  for a harness that doesn't have one yet (the pattern is in the existing
  `claude.md` / `cursor.md` recipes). The lowest-friction first PR.
- **Stamp a new leaf** with `fak new-leaf <name> --tier <tier> [--register]`
  and fill it in — the additive extension path that never edits core. Start from
  [`EXTENDING.md`](EXTENDING.md).
- **Retire one doc-debt item** the docs scorecard names —
  `python tools/docs_scorecard.py --scope reachable` prints a work-list of concrete,
  cold-reader defects (a dead link, a stale install pin, a missing title). Fix one,
  regenerate with `--markdown`.
- **Add a real test** for a package the code-quality scorecard flags untested —
  `python tools/code_quality_scorecard.py` names them; a genuine test (never a stub)
  is exactly the contribution that pays back.

Pick one, read the entry doc it points to, and ship it small and by explicit path.

> **Maintainers — keep the queue stocked.** The `good first issue` label is the front door
> this page advertises twice — here and in
> [Choose the route for your change](#choose-the-route-for-your-change) — and
> `.github/PULL_REQUEST_TEMPLATE.md` welcomes first PRs into the same funnel, so an empty
> queue is a dead link that returns no error. Check it with
> `gh issue list --label "good first issue" --state open` during triage
> and top it up from the *product* backlog, not only from documentation epics — a queue
> stocked entirely out of one meta-epic drains the moment that epic closes. An issue is ready
> for the label when its body names the file to touch, the expected result, and how to verify,
> and a contributor can finish it with a fork and a pull request — no shared checkout, no
> fleet tooling.

## Reporting issues

Use the GitHub tracker. Security-sensitive reports (a way past the capability floor or the
containment gate) should be raised **privately** — see [`SECURITY.md`](SECURITY.md) — rather
than filed as a public issue.

## Code of Conduct

Participation in this project — issues, pull requests, and reviews — is governed by the
[Code of Conduct](.github/CODE_OF_CONDUCT.md). It also names the route for reporting a
problem with someone's conduct.

## One-time Windows developer setup

For WSL-routed tests, install Go 1.26+ inside the selected distro and confirm `go version` there before running `./test.ps1`. Installing Go on the Windows host does not install it in WSL, and `GOTOOLCHAIN=auto` still needs a base `go` command; follow the [official Linux installation steps](https://go.dev/doc/install).

Windows contributors can enable native `go test`, fak's generated test binaries, and the
Fleet-spine discovery path with one elevated setup:

```powershell
go run ./cmd/fak-dev windows-setup              # inspect; makes no changes
go run ./cmd/fak-dev windows-setup --apply      # one UAC prompt; default-deny inbound, allow outbound, suppress listen prompts, install + verify
```

The command idempotently adds Microsoft Defender exclusions for the repository, Go build
cache/temp roots, fak/Fleet state, OpenCode/cache roots, and the Go/fak/agent/compiler tool processes.
It configures the active Windows power plan for High/Ultimate Performance on AC power (disabling Modern
Standby display lock suspension and enabling Away Mode for background tasks) and enables Win32 long paths.
It also installs inbound and outbound Windows Firewall rules for fak's Fleet-spine multicast endpoint.
The endpoint defaults to `239.255.70.65:4765` and follows `FLEET_SPINE_GROUP` /
`FLEET_SPINE_PORT` when those environment variables configure the guard. It reports `READY`
only after reading every setting back. Use
`--json` for machine-readable planning or verification. These are local-machine security
exceptions: review the dry-run first and use them only on a trusted development checkout.
