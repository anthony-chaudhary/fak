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

Release a manual lock on **every** exit path including failure (`python tools/release_lock.py release`); a stranded lock self-heals at TTL but releasing promptly is courteous.

## Step 1: Decide whether to release

```bash
python tools/release_decide.py --json --limit-commits 300
```

- `decision: "release"` → proceed; use `next_version`, `level`, `themes`.
- `decision: "hold"` → **stop unless the operator overrides the named blocker.**
  - `CI_BASE_RED` — **the latest *decisive* (completed) `main` ci.yml run is red.** ⚠ An in-progress run on a freshly-fixed commit does NOT clear this; `release_decide` reads the latest *completed* run. Fix forward, push, and wait for the whole CI run (including any slow `-race` job) to conclude green before re-deciding.
  - `VERSION_DRIFT`, `VERSION_BEHIND_REACHABLE_TAG`, `WORKFLOW_UNPARSEABLE` — fix, don't cut through.
  - `NOTHING_TO_SHIP` / `BELOW_SIGNIFICANCE` — nothing substantive since the last tag.
- `warnings` are not blockers; surface them in the summary.

## Step 2: Pre-release WIP snapshot (only if the tree is dirty)

Skip if the tree is clean. Otherwise commit in-flight WIP *before* drafting the release, so each thematic change lands as its own `git log` entry and the release commit carries only the version bump + release note. Group dirty paths into 2–5 thematic commits, **one `git add` per commit with every path explicit — never `git add -A`/`-u`/`.`**. Match the prefix style of recent commits.

## Step 3–5: Cut (bump VERSION + draft notes + commit)

Preferred mechanical path:

```bash
python tools/release_cut.py --json --limit-commits 300                       # no-mutation plan
python tools/release_cut.py --execute --skip-dry-run --json --limit-commits 300
```

⚠ **`--skip-dry-run` is required.** The embedded dry-run witness runs the release-substrate suite on the just-bumped commit, and one test (`release_publish_test.py::test_live_cli_dry_run_no_mutation`) reads the live VERSION and asserts a matching tag EXISTS — but the tag is minted in Step 6, *after* the cut. So the witness can never pass on a real version bump and the cut auto-unwinds. `--skip-dry-run` bypasses it; the real witness is (a) CI already green on the content and (b) the post-tag suite.

The cut refuses on **`dirty paths outside release cut`** — any path other than VERSION / the release note that is dirty. On a shared tree this is usually a peer's WIP. Do **not** stash peers' work; either wait for a clean window or cut in a **detached worktree** at origin tip:

```bash
git worktree add --detach <path> origin/main
# run the cut there with --allow-stale-upstream (HEAD == origin/main, so "stale" is a false alarm)
git worktree remove <path>     # when done
```

Verify the release commit touches ONLY `VERSION`, `docs/releases/vX.Y.Z.md`, and any `INSTALL.md` install-pin bumps. (`release_bump` pin-bumps `INSTALL.md` too — its `targets.install_docs.files[].changed` flags whether `INSTALL.md` actually moved this release; it is a clean no-op when the pins are already current.)

Manual fallback: compute the version, write `<release_notes_dir>/vX.Y.Z.md` mirroring the prior release's front-matter + theme shape, run `python <helpers.release_bump> X.Y.Z`, then `git add -- VERSION INSTALL.md docs/releases/vX.Y.Z.md` and `git commit -m "vX.Y.Z: <summary>" -- VERSION INSTALL.md docs/releases/vX.Y.Z.md`. Never `git add -A`. No `Co-Authored-By` line.

## Step 6: Push the release commit FIRST, then tag

⚠ `release_tag` checks `trunk_reachability` against the **local `refs/heads/main`** ref, not origin. The release commit must be reachable from local main before it'll tag — so push it first (it becomes trunk-reachable on origin, and local main catches up):

```bash
git push origin <release-sha>:main          # fast-forward; verify parent == origin tip first
python tools/release_tag.py --version X.Y.Z --ref <release-sha> --skip-dry-run --json          # preview
python tools/release_tag.py --version X.Y.Z --ref <release-sha> --skip-dry-run --execute --push --json
```

`--skip-dry-run` for the same chicken-egg reason as Step 5. Confirm `ok: true` and every check passes; `trunk_reachability` is the one that lags until the commit is on local main. Verify the tag derefs to the release commit:

```bash
git ls-remote --tags origin 'vX.Y.Z^{}'
```

The `ci` check is advisory NO_SIGNAL until CI runs on the release commit; that's expected. If `git push` rejects (a peer pushed), fast-forward your single release commit on top — never force-push `main`.

## Step 7: Create the GitHub release page (do NOT skip), then artifacts

⚠ The tag push fires `release-artifacts.yml`, but that workflow only **decorates** an EXISTING release page — it fails with **`release not found`** on every build job if the page doesn't exist yet. Create the page from the committed note FIRST:

```bash
python tools/release_publish.py --version X.Y.Z --json            # dry-run preview
python tools/release_publish.py --version X.Y.Z --execute --json  # create the GH release
```

Its JSON may report `github_release.status: "missing"` (the pre-check state) even on success — verify with ground truth: `gh release view vX.Y.Z --json tagName,assets`. If the tag-push artifacts run already failed on "release not found", re-dispatch it now that the page exists:

```bash
gh workflow run release-artifacts.yml -f tag=vX.Y.Z
```

Confirm the assets land: `gh release view vX.Y.Z --json assets --jq '.assets[].name'` — expect the 4 archives + their `.sha256` sidecars.

## Final summary

Print: version old → new; release tag + commit sha (short); GitHub release URL; any drift warnings or out-of-scope paths left dirty; and (if you took a manual lock) confirm it was released.

## Notes on this repo's release machinery

`release-cadence.yml` runs the same `release_decide → release_cut → release_tag` chain in CI on a schedule (scheduled ticks are dry-run-only readiness checks; a manual dispatch with `dry_run: false` arms the real cut). The four ⚠ gotchas above are the manual-path corrections — they exist because the helpers enforce ordering by refusing, and a hand-driven release hits each refusal in turn.
