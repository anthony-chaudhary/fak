# OpenCode Auto-Resume Sessions When Reopened

**Date:** 2026-09-06  
**Scope:** OpenCode harness integration, session continuation lifecycle, TUI routing, and FAK supervisor reattachment.

---

## 1. Context & Architectural Overview

When OpenCode is launched without arguments (`opencode`), it initializes to the **Home** screen (`{ type: "home" }`), prompting for a new task. Unlike Claude Code (which offers interactive conversation picking on startup), OpenCode is designed around explicit CLI resumption flags:

- **`-c, --continue`**: Continues the most recently updated root session.
- **`-s, --session <id>`**: Resumes a specific session by its identifier.

### Upstream Lifecycle & Pre-Render Fast-Path
In OpenCode's implementation (`_scratch/study-opencode/opencode`):
1. **CLI Parsing (`packages/opencode/src/cli/cmd/tui.ts`)**: Accepts `--continue` (`-c`) and `--session` (`-s`).
2. **Initial Route Pre-Arming (`packages/tui/src/app.tsx`)**: When `args.continue` is true, `RouteProvider` pre-arms `initialRoute={ type: "session", sessionID: "dummy" }` instead of `{ type: "home" }`. This prevents any home screen flash.
3. **Blocking Sync Phase (`packages/tui/src/context/sync.tsx`)**: When `args.continue` is true, `listSessions()` is promoted into the initial blocking `Promise.all` alongside provider and agent discovery.
4. **Reactive Navigation (`packages/tui/src/app.tsx`)**: Once the session list arrives, Solid's `createEffect` sorts sessions by `time.updated` descending, selects the newest session where `parentID === undefined` (filtering out subagent child sessions), and navigates directly. If no prior sessions exist, it falls back cleanly to `{ type: "home" }`.
5. **Config Schema Boundary**: Neither `opencode.json` nor `tui.json` carries an `auto_resume` setting because OpenCode validates configuration strictly against `https://opencode.ai/config.json` and rejects unknown fields with `ConfigInvalidError`.

---

## 2. Operational Solutions for Auto-Resume on Reopen

### Approach A: Native Shell Wrapper (Recommended — Zero UI Flicker)

Because OpenCode's `--continue` flag is integrated with the blocking sync phase to prevent UI flicker, wrapping `opencode` in your shell profile so that bare invocations pass `-c` is the cleanest, most performant solution:

#### Windows PowerShell (`$PROFILE`)
```powershell
function opencode {
    if ($args.Count -eq 0) {
        & (Get-Command opencode -CommandType Application) -c
    } else {
        & (Get-Command opencode -CommandType Application) @args
    }
}
```

#### Linux / macOS Bash or Zsh (`~/.bashrc` or `~/.zshrc`)
```bash
opencode() {
    if [ $# -eq 0 ]; then
        command opencode -c
    else
        command opencode "$@"
    fi
}
```

*Behavior:* Running `opencode` with no arguments automatically opens the last active session. If there are no existing sessions, OpenCode cleanly falls back to the home screen prompt. Any explicit flags (e.g. `opencode run "..."`, `opencode --pure`, `opencode -s <id>`) pass through untouched.

---

### Approach B: OpenCode TUI Plugin (`.opencode/plugins/auto-resume.js`)

OpenCode supports TUI plugins (`export default { id, tui: async (api) => ... }`) placed in `.opencode/plugins/` (project-scoped) or `~/.config/opencode/plugins/` (global).

Save this to `.opencode/plugins/auto-resume.js`:

```javascript
export default {
  id: "auto-resume",
  tui: async (api) => {
    // Only redirect if starting on the default home route
    if (api.route.current.name !== "home") return

    try {
      const result = await api.client.session.list({
        start: Date.now() - 30 * 24 * 60 * 60 * 1000,
      })
      const sessions = result?.data ?? []
      const last = sessions
        .filter((s) => !s.parentID)
        .sort((a, b) => (b.time?.updated || 0) - (a.time?.updated || 0))[0]

      if (last?.id) {
        api.route.navigate("session", { sessionID: last.id })
      }
    } catch (err) {
      // Fall back cleanly to home on fetch error
    }
  },
}
```

*Note:* Because TUI plugins initialize asynchronously after the initial view mounts, a brief home-screen flash may occur before the redirect. Approach A avoids this flash.

---

### Approach C: Dedicated FAK Launcher (`fak opencode`)

`fak opencode` provides kernel-adjudicated, prompt-cache-preserving execution of OpenCode. It now directly supports session continuation flags:

```bash
fak opencode --continue                  # continue the last session through fak guard
fak opencode -c                         # short alias
fak opencode --session ses_12345        # resume specific session ID
fak opencode --resume ses_12345         # alias
fak opencode -s ses_12345               # short alias
```

---

## 3. FAK Kernel Updates Completed

The following enhancements have landed in `fak` to make OpenCode session continuation first-class:

1. **`cmd/fak/guard_child.go`**:
   - `guardContinueFlagForAgent`: Added `"opencode"` support (`switch guardAgentBaseName(agentName)` recognizes `"claude"` and `"opencode"`), returning `("--continue", true)`.
   - Enables automatic reattachment across `fak guard` restarts, rate-limit cap parking (`guardCapParkDecision`), transient wire retries (`guardMaybeRetryTransientWireCrash`), and restart audit hop journaling.
   - Hardened `guardAppendContinueFlag` against slice bounds panics on empty command slices (`len(command) == 0`).
2. **`cmd/fak/guard_child_resource_retry.go`**:
   - Generalized `guardResourceReattachUnavailableStatus` to report the agent's real resume command instead of hardcoded `claude`.
   - Updated unsupported transport error text to list `(claude, codex, opencode)`.
3. **`cmd/fak/opencode_launcher.go`**:
   - Added `--continue` (`-c`) and `--resume` (`--session`, `-s`) flags to `runOpencode`.
   - Added validation enforcing mutual exclusivity between `--continue` and `--resume`/`--session`.
   - Fixed non-probe `--auto` and `--pure` flag forwarding to child process.
4. **Test Witnesses**:
   - `cmd/fak/guard_auth_crash_test.go`: Added test cases for bare `opencode`, path prefixes, Windows `.exe`/`.cmd` extensions, backslash path normalization (`C:\tools\opencode.cmd`), case-insensitivity, and empty slice append safety.
   - `cmd/fak/guard_resume_harness_test.go`: Added `TestResumeHarnessOpenCodeGetsContinueFlag` and updated unknown agent fixture to `aider`.
   - `cmd/fak/opencode_launcher_test.go`: Added `TestOpencodeLauncherContinueAndResume` verifying `-c`, `--continue`, `-s`, `--session`, `--resume`, mutual exclusivity erroring, and non-probe flag propagation.
