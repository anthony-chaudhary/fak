---
title: "Committing shared-clone WIP is an integration pass, not a freeze"
description: "Runbook for an agent told to commit / clean up work in progress on the shared fak clone: land complete, self-contained, building units by explicit path; park the rest; gate the push with an archive-HEAD build. Stop mislabeling collective fleet WIP as untouchable peer work."
---

# Committing shared-clone WIP is an integration pass, not a freeze

The fak fleet shares **one** clone. A session told to *"commit / clean up work
in progress"* will usually find a large pile of uncommitted changes it did not
author this session. The failure mode this note exists to kill: treating that
pile as untouchable **"peer work"** and freezing — refusing to commit anything,
emitting a wall of caution, and leaving the tree exactly as found.

That is the wrong model. The uncommitted changes are the fleet's **collective**
work, and the task is an **integration pass**: land what is finished, park the
rest in a safe place. Commit-by-path snapshots *only* the named paths, so
landing a self-contained unit can never sweep anyone else's edits — the safety
the over-caution reaches for is already guaranteed by the commit mechanic.

## The pass

1. **Build first.** `go build ./...`. If the whole tree is green, the untracked
   def files are already paired with their modified callers — the closures are
   mostly complete, the work is finishable, and the green tree is itself a safe
   place to leave anything you do not land.

2. **Cluster into units.** Group changed + untracked files into coherent units.
   A unit is one feature or fix: its new def file(s) **+** the caller edits that
   use it **+** its `_test.go`. Unrelated changes in the same package are
   separate units (e.g. three docs staged together can be two unrelated
   changes — split them).

3. **Judge each unit on three axes.** Commit only when all three hold:
   - **Complete** — finished and coherent: tests present (or the change is
     trivially correct), no `panic("not implemented")`, no stub, no TODO marking
     unfinished work. *"Builds" only proves it compiles, not that the feature is
     done* — never ship a mid-development unit.
   - **Self-contained** — no non-test file in the unit references a **new**
     top-level symbol (func / type / method / const / var that does not exist at
     `HEAD`) that is defined only in *another still-uncommitted* file. If it
     does, that other file is part of the closure: pull it into the unit or park
     both.
   - **Not sensitive** — guard policy / security admission / signing changes
     (`cmd/fak/guard*`, `guard-default-policy.json`, hooks admission gates) are
     park-by-default. They also make ordinary inspection commands trip
     `SELF_MODIFY` / `ESCALATE`; keep Bash calls simple and single-purpose.

4. **Commit complete + self-contained units by explicit path.** Never
   `git add -A` / bare-commit (the index is shared). Use
   `PYTHONUTF8=1 git commit -s -F <msgfile> -- <paths>`, a Conventional-Commits
   subject whose description leads with a recognized verb (add / fix / implement
   / wire / harden — not derive/extract), and the `(fak <leaf>)` trailer bound to
   the **path** lane. Cite `(#NNNN)` when the diff names an issue.

5. **Park the rest — do not "clean" it.** Parking means *leave it uncommitted*.
   The building working tree is the safe place. **Never** `git stash`,
   `git restore`, or `git checkout` to tidy up — that destroys the shared tree
   and other sessions' work.

6. **Gate the push.** Before pushing, prove origin stays green for *just your
   commits*, not the whole dirty tree:

   ```sh
   d="$(mktemp -d)"            # any dir OUTSIDE the clone
   git archive HEAD | tar -x -C "$d"
   ( cd "$d" && go build ./... && go vet ./... )
   ```

   The archive of `HEAD` contains your path-scoped commits and **excludes** the
   still-uncommitted rest — exactly what origin will receive. Green → `git push`.
   Red → a commit referenced an uncommitted symbol; fix the closure or it should
   not have been called self-contained.

## Why this is safe

- Commit-by-path only snapshots the listed paths from the working tree; every
  other session's uncommitted edits stay untouched.
- The archive-`HEAD` build is the live tree minus the uncommitted remainder, so
  it verifies the pushed state without having to stash or park the real tree.
- Pushing is atomic: peers only ever see the post-push state, which you proved
  builds — transient local commit order never reaches them.

If the trunk push is non-fast-forward, that is normal: stop is self-healing and
your commits ride the next session's merge + push. Stay on `main`, never
force-push.
