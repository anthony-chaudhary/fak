# GOAL — fix reversibility classifier false positives (#2376)

Fix `internal/adjudicator/reversibility.go` so family matchers stop escalating
commands that merely MENTION trigger words. Land it witnessed on trunk, close
#2376. Read it first: `gh issue view 2376`.

## Bug

`commandOutwardFacing`/`commandIrreversible` match with `containsWord`/
`orderedWords` over `commandWords(cmd)` — the WHOLE command, quotes+payloads
included. So `git commit -m "...push"`, `grep -rn "git push" docs/`,
`grep -c mail x`, `echo "run rm -rf"`, `git log --grep "npm publish"` all
misclassify.

## Fix (head-anchor the FAMILIES; keep payload scans)

1. Add a helper: split `cmd` into segments on `;` `&&` `||` `|` `&` and
   newlines; per segment take `commandWords`, then drop leading env-assignment
   tokens (raw `NAME=...`) and wrapper heads (`sudo` `env` `nice` `time`
   `command` `xargs` `doas`).
2. Match multi-word families (`git push`, `docker push`, `npm publish`,
   `cargo publish`, `gem push`, `twine upload`, `gh issue|pr|release …`,
   `git clean`, `git reset --hard`, `terraform destroy`, `kubectl delete`) as a
   PREFIX of a segment's meaningful tokens; single-word families (`slack` `mail`
   `sendmail` `mutt` `rm` `rmdir` `del` `erase` `shred` `truncate` `mkfs`
   `remove-item`) as a segment-HEAD equality. Never subsequence-over-whole-cmd.
3. Anchor httpie/gh-api on the segment head; `curlWrites` already anchored.
4. UNCHANGED (payload inspection): `orderedWords(words,"drop","database"|"table")`
   and the `dd`+`of=/dev/` check stay whole-command scans (SQL/dd arrive as
   args, not heads). Keep `hasDryRunPreview`. Don't touch the `_fak_confirm`
   token machinery.

## Regression fences (add to reversibility_test.go) — MUST hold

MENTION → `reversible`: `grep -rn "git push" docs/`,
`git commit -m "docs: explain when to push"`, `grep -c mail x`,
`echo "never run rm -rf blindly"`, `git log --grep "npm publish"`.
REAL → still escalates: `git push origin main`, `rm -rf build`,
`git commit && git push` (seg 2), `FOO=1 sudo rm -rf x` (after env+sudo strip),
`echo hi | mail bob` (seg 2), `psql -c "drop database x"` (payload),
`git push --dry-run origin main` → reversible. Keep green:
`TestReversibilityConfirmationTokenMustEcho`,
`TestAdjudicateReversibilityGateDoesNotOverrideHardDeny`,
`TestAdjudicateReversibilityGateAllowsConfirmedCallAndStripsToken`.

## Verify then land (hard-self core-lock)

`internal/adjudicator/**` is `hard-self` (CORE_SELF_MODIFY). You MUST:

1. GREEN before commit (Windows → tests via WSL):
   `wsl -e bash -lc "cd /mnt/c/work/fak && go test ./internal/adjudicator/ ./internal/gateway/"`
   then `make ci` (or `scripts/ci.ps1`). Never commit on red.
2. Commit by explicit path with the maintenance witness (operator-authorized via
   #2376):
   ```
   fak commit --path internal/adjudicator/reversibility.go \
     --path internal/adjudicator/reversibility_test.go \
     -m "fix(adjudicator): head-anchor reversibility family matchers so mentions don't escalate #2376 (fak adjudicator)" \
     --core-lock-maintenance-witness "committed:internal/adjudicator/reversibility.go"
   ```
   The witness is a recorded audit fact, not a bypass. If it refuses, STOP and
   report verbatim; never route around it with raw `git commit`.
3. `git push origin main` (fast-forward; never force). Diverged? `git fetch
   origin main` + `git merge origin/main` in place, re-verify, re-push.
4. `gh issue close 2376 --comment "Fixed in <sha>: <one line>"`.

## Honest boundary

A launch is not a ship — #2376 resolves only on a witnessed trunk commit
(`dos commit-audit`). Can't get CI green? Report `not yet` with the failing
witness, leave the tree clean. NEVER weaken a fence to pass, and never land a
change that stops a true-positive (`rm -rf`, `git push`) escalating. Stop after
one witnessed ship + close.
