# Default `fak agent` repository tools in a non-FAK repository — 2026-08-19

**Verdict:** `fak agent` now arms bounded repository-native tools by default when launched from an arbitrary repository, while retaining explicit workspace selection and opt-out controls.

- `TestAgentDefaultCodeToolsArmInNonFakWorkspace` creates a temporary directory with no FAK metadata and proves the default catalog contains exactly `Read`, `Write`, `Edit`, `Bash`, `Grep`, and `Glob`.
- The existing owned-loop witnesses in `internal/agent/codetools_loop_test.go` drive these tools through the real kernel engines: repository reads/search, scratch-repository mutation, focused Bash execution, and rejection of an out-of-tree read before filesystem access.
- The default root is the process working directory. `--code-workspace <directory>` selects another root; `--code-tools=false` independently disables this catalog.
- `TestAgentLaunchPostureReportsBoundedCodeToolsActive` proves `fak doctor launch-posture --entrypoint agent` reports the mechanism as active for a readable workspace and exposes the opt-out.
- Bash remains constrained by the focused code-tool command allowlist. This artifact does not claim unrestricted shell execution.

Structured artifact: [`agent-default-code-tools-non-fak-2026-08-19.json`](agent-default-code-tools-non-fak-2026-08-19.json).

This retires the owned-agent bounded-tool portion of #8089; passthrough ownership, portable cold-tool deferral, and cross-backend vCache signals remain tracked there.
