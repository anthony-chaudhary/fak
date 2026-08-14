---
name: release
description: Perform a full versioned release — bump version, draft release notes, commit, tag, push, and create the GitHub release page. Reads `.claude/project.yaml` for the project's release-context and version-bump helpers; the skill text is universal, the helpers are project-supplied. Use when the user says "cut a release", "ship vX.Y.Z", "release", or after a shippable phase.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Edit, Grep, Glob, Bash, Write
argument-hint: "[summary of changes] [--scope <theme-token>...] [--from-manifest <path>]"
---

# /release — Versioned Release

Semver: `major.minor.patch`. Patch = bug fix, minor = new feature, major = breaking.

**Git authorization.** Invocation of this skill is the user's explicit authorization to run `git add`, `git commit`, `git tag`, and `git push origin main` / `vX.Y.Z` as specified in Steps 5–7. The "never commit/push unless asked" default does NOT apply here — committing and pushing IS the skill's job. Confirmation is still required for anything destructive the steps don't list (force-push, history rewrites, branch deletion, `git reset --hard`).

This repo's trunk is **`main`**. The version source-of-truth is the bare `VERSION` file; release notes live under `docs/releases/vX.Y.Z.md`.

## Project contract

This skill reads `.claude/project.yaml` at the repo root. Keys it uses:

- `python` — interpreter path (default: `python`).
- `helpers.release_context` — script that emits the Step 1 JSON payload.
- `helpers.release_bump` — script that bumps every version-marker file.
- `helpers.release_lock` *(optional)* — the single-writer release lock. Present in
  this repo (`tools/release_lock.py`); when present, `release_bump` refuses to
  bump unless the lock is held.
- `release_notes_dir` — where to write `vX.Y.Z.md` (default: `docs/releases/`).

If `.claude/project.yaml` is missing or these keys are absent, print one line pointing at `.claude/skills/README.md` and stop. Do not improvise file locations.

This repo ships the mechanical helpers (`tools/release_decide.py`, `release_cut.py`, `release_tag.py`, `release_publish.py`, `release_lock.py`, `release_dry_run.py`) that automate Steps 1–7. The skill text below drives them and explains the structural gotchas they enforce by refusing.

## Fast path (hot shared tree): `fak release ship`

On this repo's hot, multi-session trunk, prefer the one-command front door over driving the helpers by hand:

```bash
fak release ship --json            # full dry-run plan — no mutations, no lock
fak release ship --execute --json  # cut, push to trunk, tag, publish
```

It cuts in a throwaway **detached worktree at the origin tip**, so it leaves *this* checkout's dirty peer WIP untouched; takes and auto-renews the single-writer release lock; pushes with `--force-with-lease`; on a same-trunk peer advance during the cut it re-cuts onto the new tip and retries (never force-pushing); and on a benign peer fast-forward past the release commit while it waits for CI, it recognizes the commit is still on trunk and proceeds — only a rewrite that orphans the commit refuses. Reach for this first. Drive the manual **Steps 0–7 below by hand only when `fak release ship` refuses** and you need to see why — each manual step carries the same hot-tree safety corrections the front door automates.

---

## Step 0: Scope (optional — scope by default on a hot shared tree)

If the working tree is routinely hot with peers' edits, a whole-tree release is the risky path — derive a scope from the dirty paths + commit subjects and proceed with it. On a quiet tree the whole dirty set is the release content.

- `--scope <theme-token>` pins the scope explicitly (case-insensitive substring match against paths + commit subjects).
- `--whole-tree` is the explicit opt-out; use it only when the whole dirty tree is known release content.
- `--from-manifest <path>` replaces scope inference with a producer manifest: run `python tools/release_manifest.py consume <path> --json`, proceed only when `staged_paths` is non-empty and every pick is `status: shipped` and reachable from `HEAD`.

## Step 0.5: Acquire the single-writer release lock (if present)

If several `/release`-capable sessions can run at once, take the lock before reading any release state so a second session can't race you on VERSION/tag:

```bash
python tools/release_lock.py acquire --ttl 1800
```

- **`ok: true`** → you hold it; continue. The owner is your session id, so every later `release_lock` / `release_bump` call re-proves ownership automatically.
- **`ok: false, reason: "held"`** → another session is mid-release. **Stop**, report the holder, and let its release finish. A stale lock past its TTL is auto-stolen on the next `acquire`. Don't `--force` unless the holder is known-dead.

**Keep the lock alive across a long cut.** The TTL is a tight crash-recovery window (30 min), not a budget for the whole release — a cut that waits on a slow `-race` CI run can outlive it, and the moment it goes stale the auto-cut tick or another session will steal it and race you on VERSION/tag. If any step (CI wait, tag) might run long, renew on a timer at roughly ttl/3 from the SAME session:

```bash
python tools/release_lock.py renew --ttl 1800   # extend; refuses (exit 3) if the lease was already stolen
```

A `renew` that refuses with `reason: "held by another"` means you already lost the lock — **stop before you push or tag**, don't `--force` over the new holder. (`fak release ship` is the automated hot-tree path; it holds this same lock and auto-renews it before the tag and publish phases, refusing before it tags if the renewal proves the lease was stolen.)

Release a manual lock on **every** exit path including failure (`python tools/release_lock.py release`); a stranded lock self-heals at TTL but releasing promptly is courteous.

## Canonical operator path

Use the first-class front door before helper-level debugging:

```bash
fak release --help
fak release status
fak release plan
fak release decide
fak release cut --dry-run
fak release cut
fak release tag
fak release publish
```

Preserve this order: status/readiness → plan → decide → dry-run cut → execute cut → exact-SHA tag
→ publish. The release lock, green-ancestor gate, and stable-context checks remain binding. Direct
`python tools/release_*.py` commands below are fallback/debug equivalents only, not the preferred
operator interface.

Verify the canonical surface before proceeding: `fak release --help` must list `status`, `plan`,
`decide`, `cut`, `tag`, and `publish`. If one is absent, stop rather than silently falling back;
use a helper only to diagnose the missing first-class wiring.

## Step 1: Decide whether to release

```bash
python tools/release_decide.py --json --limit-commits 300
```

- `decision: "release"` → proceed; use `next_version`, `level`, `themes`.
- `decision: "hold"` → **stop unless the operator overrides the named blocker.**
  - `CI_BASE_RED` — **the latest *decisive* (completed) `main` `ci-fast.yml` run is red.** An in-progress run on a freshly fixed commit does not clear this; `release decide` reads the latest completed fast-gate run. Fix forward, push, and wait for that release-critical fast subset to conclude green before re-deciding.
  - `VERSION_DRIFT`, `VERSION_BEHIND_REACHABLE_TAG`, `WORKFLOW_UNPARSEABLE` — fix, don't cut through.
  - `NOTHING_TO_SHIP` / `BELOW_SIGNIFICANCE` — nothing substantive since the last tag.
- `warnings` are not blockers; surface them in the summary.

## Step 2: Pre-release WIP snapshot (only if the tree is dirty)

Skip on a clean tree — and skip on a hot/shared tree, where the dirty paths are peers' in-flight WIP, not yours: do **not** commit or stash their work (Step 3–5 says the same). There, use `fak release ship` (or the detached-worktree cut in Step 3–5), which reads only the origin tip and leaves the working tree untouched. **Only when you own the whole checkout** (a solo, quiet tree) commit *your own* in-flight WIP first, so each thematic change lands as its own `git log` entry and the release commit carries only the version bump + release note. Group your dirty paths into 2–5 thematic commits, **one `git add` per commit with every path explicit — never `git add -A`/`-u`/`.`**. Match the prefix style of recent commits.

## Step 3–5: Cut (bump VERSION + draft notes + commit)

Preferred mechanical path:

```bash
python tools/release_cut.py --json --limit-commits 300             # no-mutation plan
python tools/release_cut.py --execute --json --limit-commits 300   # runs the dry-run witness
```

The `--execute` cut runs the embedded release-substrate dry-run witness on the just-bumped commit, and it **passes** on a real version bump: the witness (`release_publish_test.py::test_live_cli_dry_run_no_mutation`) reads the *latest existing tag*, not the live VERSION, so it no longer needs the not-yet-minted tag to exist. Let it run — do **not** pass `--skip-dry-run` on a normal cut. It is an emergency-only escape hatch now, not a required step.

⚠ **The one residual manual-path gotcha (dirty tree).** The cut refuses on **`dirty paths outside release cut`** — any path other than VERSION / the release note that is dirty. On a shared tree this is usually a peer's WIP. Do **not** stash peers' work; either wait for a clean window or cut in a **detached worktree** at origin tip (this is exactly what `fak release ship` does for you):

```bash
git worktree add --detach <path> origin/main
# run the cut there with --allow-stale-upstream (HEAD == origin/main, so "stale" is a false alarm)
git worktree remove <path>     # when done
```

Verify the release commit touches ONLY `VERSION`, `docs/releases/vX.Y.Z.md`, any `INSTALL.md` install-pin bumps, and the distribution manifests `server.json` + `CITATION.cff`. (`release_bump` pin-bumps `INSTALL.md` via `targets.install_docs` and `server.json`/`CITATION.cff` via `targets.dist_manifests` — each `files[].changed` flags whether that file actually moved this release; all are clean no-ops when already current. Pass `--date YYYY-MM-DD` so `CITATION.cff`'s `date-released` advances too; without it the version still bumps but the date is left alone. `server.json`'s `oci` identifier tag must match the ghcr image `release-container.yml` pushes, or the MCP Registry lists a back-version — see [docs/fak/mcp-registry.md](../../../docs/fak/mcp-registry.md) "Updating on each release".)

Manual fallback: compute the version, write `<release_notes_dir>/vX.Y.Z.md` mirroring the prior release's front-matter + theme shape, run `python <helpers.release_bump> X.Y.Z --date YYYY-MM-DD`, then `git add -- VERSION INSTALL.md server.json CITATION.cff docs/releases/vX.Y.Z.md` and `git commit -m "vX.Y.Z: <summary>" -- VERSION INSTALL.md server.json CITATION.cff docs/releases/vX.Y.Z.md`. Never `git add -A`. No `Co-Authored-By` line.

## Step 6: Push the release commit, then tag

`release_tag` checks `trunk_reachability` against the **local `refs/heads/main`** ref, not origin, so push the release commit first (it becomes trunk-reachable on origin, and local main catches up), then tag. This is plain sequencing — the helpers enforce it and `fak release ship` does it for you — not a chicken-egg trap:

```bash
git push origin <release-sha>:main          # fast-forward; verify parent == origin tip first
python tools/release_tag.py --version X.Y.Z --ref <release-sha> --json          # preview (runs the dry-run witness)
python tools/release_tag.py --version X.Y.Z --ref <release-sha> --execute --push --json
```

No `--skip-dry-run`: the witness runs on the pushed commit and passes (it reads the latest existing tag; it re-witnesses the same commit the cut already checked, which is harmless). Confirm `ok: true` and every check passes; `trunk_reachability` is the one that lags until the commit is on local main. Verify the tag derefs to the release commit:

```bash
git ls-remote --tags origin 'vX.Y.Z^{}'
```

The `ci` check is advisory NO_SIGNAL until CI runs on the release commit; that's expected. If `git push` rejects (a peer pushed), fast-forward your single release commit on top — never force-push `main`. (`fak release ship` automates exactly this recovery on the same-trunk hot path: it re-cuts the VERSION+note commit onto the advanced trunk tip and retries the leased push up to `--push-retries` times, refusing only if trunk was rewritten rather than fast-forwarded.)

## Step 7: Create the GitHub release page, then artifacts

The tag push fires `release-artifacts.yml`, which cross-compiles the binaries and **decorates** the release page. That workflow now **waits up to ~2 min for the page to appear** before uploading (it polls `gh release view` — issue #369), so the page and the artifacts run are no longer order-sensitive. Create the page from the committed note promptly all the same:

```bash
python tools/release_publish.py --version X.Y.Z --json            # dry-run preview
python tools/release_publish.py --version X.Y.Z --execute --json  # create the GH release
```

Its JSON may report `github_release.status: "missing"` (the pre-check state) even on success — verify with ground truth: `gh release view vX.Y.Z --json tagName,assets`. Only if the artifacts run timed out waiting for the page (a gap wider than its ~2 min poll), re-dispatch it now that the page exists:

```bash
gh workflow run release-artifacts.yml -f tag=vX.Y.Z
```

Confirm the assets land: `gh release view vX.Y.Z --json assets --jq '.assets[].name'` — expect the 4 archives + their `.sha256` sidecars.

## Final summary

Print: version old → new; release tag + commit sha (short); GitHub release URL; any drift warnings or out-of-scope paths left dirty; and (if you took a manual lock) confirm it was released.

## Notes on this repo's release machinery

`release-cadence.yml` runs the same `release_decide → release_cut → release_tag` chain in CI on a schedule (scheduled ticks are dry-run-only readiness checks; a manual dispatch with `dry_run: false` arms the real cut). Three of the four historical chicken-egg gotchas are now absorbed mechanically — the dry-run witness is tag-based and passes on a real bump (no `--skip-dry-run` dance), the push-before-tag ordering is plain sequencing the helpers and `fak release ship` enforce, and `release-artifacts.yml` waits for the release page instead of failing `release not found`. One residual manual-path correction remains: on a hot shared tree, cut in a detached worktree (Step 3–5) so a peer's dirty WIP can't block the cut — and `fak release ship` absorbs even that.
