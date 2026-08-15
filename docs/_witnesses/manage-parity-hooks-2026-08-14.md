# Native managed-hook capability witness — 2026-08-14

## Commands

```powershell
go test ./cmd/fak -run 'Test(ManageParity|InstallManaged|ManageNative)' -count=1
fak manage parity --json
```

## Result

The isolated detached worker worktree passed the focused regression suite. The real
`fak manage parity --json` launch receipt emitted schema `fak-manage-parity/2` and a
`PASS` verdict for Claude Code, Codex CLI, and Gemini CLI.

The receipt types each lifecycle/tool/settings capability as `installed`, `unsupported`,
or `not-requested`, and pins the inventory to Claude Code 2.1.229, OpenAI Codex
0.147.0 (`rust-v0.147.0`), and Gemini CLI 0.45.2 (`v0.45.2`). Codex uses its documented
`--config hooks` events (`Stop`, `PreCompact`, `PreToolUse`, `PostToolUse`). Gemini uses
a launch-scoped `GEMINI_CLI_SYSTEM_SETTINGS_PATH` file for `BeforeTool`, `AfterTool`,
`AfterAgent`, and `SessionEnd`; Gemini 0.45.2 has no PreCompact event, so that one
capability is explicitly `unsupported` rather than represented by a false boolean.

`TestManageNativeHookFailsClosedOnMalformedInput` captures the failure-before/pass-after
boundary: an event-name mismatch returns a typed `deny`, while matching JSON returns
`allow`. The adapters are installed on the existing managed child command immediately
before spawn, so provider/base-URL/policy/argv routing remains shared with legacy guard.
