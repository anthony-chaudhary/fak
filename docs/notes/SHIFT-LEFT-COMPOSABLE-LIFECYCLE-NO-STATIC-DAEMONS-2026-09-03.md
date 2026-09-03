# Shift-Left Invariant: Composable Lifecycles Over Static Background Daemons

**Date:** 2026-09-03  
**Status:** Canonical operational & architectural doctrine  
**Scope:** Developer workstations, local test benches, CI/CD runners across macOS, Linux, and Windows  

---

## 1. The Core Principle: Composability Over Static Daemons

In developer environments, testing harnesses, and local evaluation benches, **there must never be unmanaged, static background processes or auto-restarting daemon definitions** (such as macOS `launchd` LaunchAgents with `KeepAlive=true`, Linux `systemd --user` units with `Restart=always`, or Windows Task Scheduler jobs) running model servers (`llama-server`, `fak serve`, etc.) out-of-band.

Specifically, static background daemons with `KeepAlive=true` (e.g. `com.fak.*` in `~/Library/LaunchAgents/` or `/Library/LaunchDaemons/`) are strictly prohibited on developer benches.

All local execution must be **composable, ephemeral, and bounded**:
- **Lifecycle-bound:** Servers are spawned on-demand by the calling harness/session that requires them.
- **Deterministic teardown:** When the driving command or test finishes (whether successful, failed, or interrupted), all child processes, sockets, and memory reservations must be completely released via disciplined signal traps (`trap cleanup EXIT INT TERM`).
- **Zero background idle residency:** A developer machine or shared test box must remain at zero resident model weight footprint when no test or benchmark is actively executing.

---

## 2. Post-Mortem: How `llama-server` Came Back on macOS

### The Failure Chain
1. **Historical Service Swap:**  
   `com.fak.qwen36-kernel.plist` originally ran a native `fak serve` instance. On 2026-07-04, it was modified to execute `~/llama-b9828/llama-b9828/llama-server` on port `8090` serving a 27B model, retaining `<key>KeepAlive</key><true/>`.
2. **Shadow Gateway Loop:**  
   `com.fak.serve-gateway.plist` was configured to proxy requests to `http://127.0.0.1:8090/v1`. When port 8090 was wedged or unavailable, this service entered an infinite crash-restart loop (logging over 12,300 failed attempts into `launchd_serve.err`).
3. **The False Resolution in Issue #9714:**  
   Earlier cleanup reaped `com.fak.qwen36-model.plist` (renaming it to `com.fak.qwen36-model.plist.disabled-dup`). When issue #9714 audited port 8090, it detected that `com.fak.qwen36-kernel` (not `com.fak.qwen36-model`) was supervising the port. Because #9714 had a strict read-only audit scope, it recorded a `HOLD` verdict without unregistering or disabling the running LaunchAgent.
4. **Memory Contention & Resource Inversion:**  
   `llama-server` holding a 16+ GiB model resident in unified memory meant that on a 36 GiB M3 Pro Mac, macOS VM compression rose to ~13 GiB (33.9% of memory). When native Qwen3.8 benchmarks were attempted, `internal/localadmission` tripped `pressure_critical` (threshold 30%), or the host was forced into heavy swap thrashing.
5. **The KeepAlive Zombie:**  
   Because `KeepAlive` was `true` in `launchd`, any standard `kill` or process termination was immediately countered by launchd restarting `llama-server` within 30 seconds.

---

## 3. Immediate Remediation & Cleanup Commands

When static background daemons are detected or ports are wedged, run the following sequence to clean up:

1. **Boot Out and Disable LaunchAgents / LaunchDaemons:**
   ```bash
   launchctl bootout "gui/$(id -u)/com.fak.qwen36-kernel" 2>/dev/null || true
   launchctl bootout "gui/$(id -u)/com.fak.serve-gateway" 2>/dev/null || true
   launchctl disable "gui/$(id -u)/com.fak.qwen36-kernel" 2>/dev/null || true
   launchctl disable "gui/$(id -u)/com.fak.serve-gateway" 2>/dev/null || true
   ```
2. **Terminate Lingering Listeners via Port:**
   ```bash
   lsof -ti:8090 | xargs kill
   ```
3. **Reap and Remove Plist Files:**
   Permanently remove from `~/Library/LaunchAgents/` and `/Library/LaunchDaemons/`:
   ```bash
   rm -f ~/Library/LaunchAgents/com.fak.qwen36-kernel.plist*
   rm -f ~/Library/LaunchAgents/com.fak.qwen36-model.plist*
   rm -f ~/Library/LaunchAgents/com.fak.serve-gateway.plist*
   sudo rm -f /Library/LaunchDaemons/com.fak.* 2>/dev/null || true
   ```
4. **Confirm Process & Port Clearance:**
   Verify that PID 709 was reaped, port `8090` is completely unallocated, and no `llama-server` or rogue `fak serve` processes exist on the system:
   ```bash
   lsof -i :8090
   pgrep -fl "(llama-server|fak serve)"
   ```

---

## 4. Cross-Platform Equivalents: macOS, Linux / WSL & Windows

Persistent background daemons manifest differently across operating systems. All developer benches must enforce the same composable lifecycle rules regardless of platform:

| Platform | Persistent Mechanism to Prohibit | Equivalence to macOS Launchd `KeepAlive` | Verification & Remediation Command |
|---|---|---|---|
| **macOS** | `~/Library/LaunchAgents/*.plist` or `/Library/LaunchDaemons/*.plist` | `<key>KeepAlive</key><true/>` in launchd plist | `launchctl list \| grep fak`; bootout with `launchctl bootout "gui/$(id -u)/<label>"`; clean port with `lsof -ti:<port> \| xargs kill`; remove plists with `rm -f ~/Library/LaunchAgents/com.fak.*` |
| **Linux / WSL** | `~/.config/systemd/user/*.service` or `/etc/systemd/system/*.service` | `Restart=always` or `Restart=on-failure` in systemd unit | `systemctl --user list-units --type=service \| grep -E "(fak\|llama\|qwen)"`; disable with `systemctl --user disable --now <unit>`; clean port with `lsof -ti:<port> \| xargs kill` or `fuser -k <port>/tcp`; remove unit files |
| **Windows** | Scheduled Tasks in `Task Scheduler` (`schtasks`) | Continuous execution triggers or auto-restart on failure | Query with `Get-ScheduledTask \| Where-Object TaskName -like "*fak*"` or `schtasks /query`; remove task with `Unregister-ScheduledTask -TaskName <name> -Confirm:$false` or `schtasks /delete /tn <name> /f`; terminate port with `Stop-Process -Id (Get-NetTCPConnection -LocalPort <port>).OwningProcess -Force` |

### Linux Audit Findings
- In Linux/WSL environments, `systemd-run --unit=... --collect` was historically used in scripts under `tools/` and `scripts/gcp-*.sh`.
- **Rule for Linux testbeds:** Ephemeral benchmarking scripts must never leave persistent systemd units running without an attached reaper timer (`scripts/gcp-idle-reaper.sh`).
- On developer workstations, no user-level systemd service should automatically launch heavy LLM weights on boot or login.

### Windows Audit Findings
- Windows dev benches must not register auto-starting Scheduled Tasks or Windows Services with `binPath` pointing to model runners without an explicit `--uninstall` or lifecycle reaper script.

---

## 5. Shift-Left Engineering Rules for Contributors & Agents

1. **Never Install Persistent Auto-Restart Daemons on Dev Hosts:**
   - Never write `KeepAlive=true` LaunchAgents or `Restart=always` systemd user units on local development workstations.
   - If an automated service is required for a test, launch it in the foreground or manage it via an ephemeral subprocess with strict lifetime ownership.
2. **Enforce Clean Port & Service Preflight:**
   - Before executing any local serving benchmark, audit whether candidate ports (e.g. `8090`, `8080`, `18085`) are held by unmanaged third-party daemons.
   - If an unexpected listener exists, fail fast with a descriptive refusal rather than attempting to share memory or launch competing 27B weights.
3. **Honor the Native Inference Invariant:**
   - In `AGENTS.md`: Native inference performance tasks must keep model execution fak-native. Running an unmanaged external `llama-server` in the background directly violates this invariant and pollutes system metrics.
4. **Composable Harnesses Only:**
   - All tests and benchmarks must adhere to the pattern:
     ```bash
     server_pid=""
     cleanup() {
       if [[ -n "$server_pid" ]]; then
         kill "$server_pid" 2>/dev/null || true
         wait "$server_pid" 2>/dev/null || true
       fi
     }
     trap cleanup EXIT INT TERM
     ```
   - No process may survive the exit of its invoking script.
