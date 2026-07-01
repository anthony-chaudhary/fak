# Proof Artifact Placement

Issue fixes should ship a witness matched to the failure class, but not every witness
belongs in git. Use this placement rule before a worker attaches evidence to a close
comment or stages files.

## Placement Rules

| Artifact type | Allowed location | Commit it? | Rule |
|---|---|---:|---|
| Committed fixture or baseline | `testdata/`, `experiments/`, `docs/baselines/`, or a package-local fixture directory | Yes | Use for deterministic inputs, expected outputs, small JSON/markdown packets, and replayable benchmark or render fixtures that future tests should diff. The artifact must be scrubbed, deterministic, and tied to a test or documented command. |
| Generated temp artifact | OS temp dir, `.cache/`, a gitignored run directory, or a worker scratch directory | No | Use for raw command output, intermediate screenshots, logs, dispatch telemetry, and bulky one-off captures. The close comment may cite a digest, command, and local path, but the file is not staged. Promote only a small scrubbed subset if it becomes a regression fixture. |
| GitHub-only comment or attachment | Issue/PR comment, release note, or GitHub attachment | No | Use for before/after terminal screenshots, human-readable summaries, externally hosted CI links, and evidence that is useful for review but too noisy, private, or non-deterministic for public history. Redact private host, account, token, and session details before posting. |

## Examples

| Scenario | Artifact | Allowed placement |
|---|---|---|
| Parser bug with a stable repro | Minimal input JSON plus expected output | Commit under `testdata/<package>/...` and cover it with a test that fails before the fix. |
| TUI or terminal rendering bug | Before/after screenshot from the reporter's scenario | Attach to the issue or PR comment; commit only a render-witness fixture if the bytes are deterministic and small. |
| Live guard or dispatch run | Raw transcript, account state, session id, API incident notes, or one-off telemetry | Keep in a temp/gitignored run directory or `fak-private`; post a scrubbed summary and command witness in the issue comment. Do not commit operator-private raw evidence. |

If an artifact contains private operational state, use the repo's operator-private
marker in a local-only note or move the raw capture to `fak-private`. If a public
fixture is needed later, create a scrubbed, minimal reproduction rather than
committing the raw capture.
