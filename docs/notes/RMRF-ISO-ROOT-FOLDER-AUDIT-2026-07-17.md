# `.rmrf-iso` root-folder audit — 2026-07-17

## Verdict

`.rmrf-iso` was a disposable, interrupted source-tree isolation copy left in the repository root. It was not a Git worktree, repository, symlink, junction, build output, or durable FAK runtime directory. A companion `.rmrf-overlay-scratch` directory held a Go overlay that replaced `internal/bench/negbench.go` with a package-only stub. Both artifacts were created during the same narrow interval while recursive/forced-delete adjudication was being investigated.

The artifacts have been moved intact, not deleted, to:

`_scratch/quarantine/rmrf-iso-20260717-100007/`

The companion overlay is preserved at `_companion-overlay-scratch/` under that quarantine directory.

## Evidence

- `.rmrf-iso` was created at 10:00:07 local and populated through 10:00:34; `.rmrf-overlay-scratch` was created at 09:58:18 and last changed at 09:59:00.
- The source copy contained 1,800 files, 166 directories, and 16,605,971 bytes. It had no `.git`, no reparse points, and no inner repository metadata.
- Every copied file had a corresponding live-tree path. The copy was therefore a snapshot attempt, not an independent source tree.
- 1,725 of 1,800 files still had the same size as their live-tree counterparts when audited. Seventy-five differed because the shared live tree continued changing after the copy.
- Population stopped alphabetically inside `cmd/`, ending at `cmd/turntaxdemo`; all later top-level trees (`docs`, `internal`, `tools`, and others) were absent. That shape, together with the 27-second creation span, proves the snapshot was interrupted rather than intentionally scoped.
- The companion `overlay.json` mapped `C:/work/fak/internal/bench/negbench.go` to a 14-byte `package bench` stub. Its name, timestamp, and purpose tie it to an isolation experiment rather than normal product operation.
- No tracked file, Git history, active process command line, PowerShell history, or durable repo configuration referenced either root artifact.
- The same work interval introduced the force-only-versus-recursive-delete adjudicator work later shipped as issue #4983 (`internal/adjudicator@r78+gc8e0ac780`). This is strong circumstantial provenance for the experiment, but the exact scratch command was not retained in available shell/session logs, so authorship and exact invocation remain unproven.

## Root causes

1. **Wrong scratch boundary.** A throwaway isolation tree was created directly under the repository root instead of the OS temporary directory or the repo's ignored `_scratch/` boundary.
2. **Non-atomic recursive copy.** The snapshot mechanism copied the live, peer-dirty tree incrementally. It could therefore capture mixed-time peer WIP and leave a valid-looking partial tree when interrupted.
3. **No lifecycle cleanup.** The scratch operation had no guaranteed cleanup/quarantine step, so its partial output survived the experiment.
4. **No ignore protection for ad-hoc dot directories.** The names were not covered by `.gitignore`. Git consequently reported both as untracked.
5. **Buildcheck amplification.** `fak buildcheck` discovers untracked Go files with `git ls-files --others --exclude-standard -- '*.go'`. Because `.rmrf-iso` was unignored, all nested copied Go files entered its masking inventory even though the Go tool itself ignores dot-prefixed directory components. This caused large, misleading `masked_files` output and avoidable scan/overlay work; it did not make those packages part of `go build ./...`.

## Remediation and prevention

- The two root artifacts were moved intact under the already ignored `_scratch/` boundary. Root-level `git status` no longer reports either artifact, and a post-move `fak buildcheck --json ./cmd/fak` contains no `rmrf-iso` or `rmrf-overlay-scratch` paths. The buildcheck still exits red for unrelated peer-dirty live-tree failures; that is not evidence against this cleanup.
- Do not restore this snapshot into the live tree: it is incomplete and contains stale copies of peer-dirty files.
- Future isolation must use the durable `fak worktree worker prepare|land|reap` path when a real source-tree build witness is required, or an OS temporary directory / `_scratch/` for a disposable diagnostic. Do not recursively copy the repository into one of its own unignored children.
- Cleanup should use preview-then-quarantine semantics. Moving an uncertain tree under `_scratch/quarantine/` preserves evidence without exercising recursive deletion against shared-trunk data.
- No broad `/.*/` ignore rule should be added: it would hide accidental root artifacts and reduce the visibility that exposed this incident.

## Completion checks

- Root `.rmrf-iso`: absent.
- Root `.rmrf-overlay-scratch`: absent.
- Quarantined file count and byte count: preserved (1,800 files; 16,605,971 bytes), plus the two companion overlay files.
- Root Git status mentions of `rmrf`: none.
- Buildcheck masking references to either artifact: none.
