---
title: "Syscall-Level Tool Process Governance, vDSO Fast-Paths, and Orphan Isolation Under High Volume"
description: "Formal specification for process group tracking, automatic orphan tree reaping, port isolation, kernel vDSO fast-paths, and headless execution guarantees under high-volume agent tool execution."
---

# Syscall-Level Tool Process Governance, vDSO Fast-Paths, and Orphan Isolation Under High Volume

> **Contract Authority:** This document specifies the operating system boundaries, process group and job object
> tracking mechanics, recursive orphan reaping invariants, port isolation protocols, kernel vDSO fast-paths, and
> headless execution guarantees for high-volume tool execution in the fak agent kernel.
> The machine-checked Go implementation resides in [`internal/procguard/supervisor.go`](../../internal/procguard/supervisor.go),
> [`internal/procguard/supervisor_unix.go`](../../internal/procguard/supervisor_unix.go), and
> [`internal/procguard/supervisor_windows.go`](../../internal/procguard/supervisor_windows.go), verified by
> [`internal/procguard/supervisor_test.go`](../../internal/procguard/supervisor_test.go).

---

## 1. Overview & High-Volume Problem Statement

Autonomous agent systems operating at scale (e.g., executing hundreds or thousands of tool commands per minute
across multi-agent waves, super-loops, and continuous CI workers) impose extreme stress on host operating system
process and networking abstractions. In un-governed agent harnesses, tool execution routinely delegates to standard
subshell wrappers (`exec.Command("sh", "-c", ...)`, `child_process.exec`, or PowerShell spawns).

Under high volume, this un-governed model exhibits four catastrophic failure modes:

1. **Orphan Proliferation & Subtree Leaks:** Tool scripts (e.g. `npm`, `cargo`, `python`, `git`, compilers, test harnesses)
   frequently spawn child processes, background daemons, or double-forked worker subtrees. When the parent tool process
   exits, crashes, or is terminated via a simple `Process.Kill()` (which signals only the root PID), child processes are
   reparented to `init` (PID 1) on POSIX or remain stranded on Windows. Over 1,000 tool executions, stranded processes
   accumulate, causing file lock contention, memory leaks, and CPU starvation.
2. **Local Port Collisions & Cross-Session Contamination:** Multi-agent workloads frequently spin up local servers
   (e.g., dev servers, mock APIs, database listeners, MCP sidecars) binding to localhost TCP/UDP ports. Without centralized
   session-scoped port tracking, parallel workers collide on identical port allocations (`EADDRINUSE`), or worse, an agent
   inadvertently communicates with a zombie server left behind by an earlier un-reaped session.
3. **Fork/Exec Latency Tax on Read-Only Tool Calls:** Spawning a full shell process for trivial, read-only operations
   (such as querying file timestamps, environment variables, git HEAD references, or tool versions) incurs 10ms to 50ms
   of process creation, dynamic linker binding, and context switching overhead per call. In high-turn agent trajectories
   executing thousands of tool calls, this shell latency dominates total turn duration.
4. **Headless Execution Deadlocks & Terminal/GUI Traps:**
   - **PTY/Buffer Deadlocks:** A tool that emits verbose logs to `stdout` or `stderr` fills the OS pipe buffer (typically
     64 KB). If the runner does not continuously drain the pipe asynchronously, the child process blocks indefinitely on
     `write(2)`, deadlocking the entire agent turn.
   - **Interactive Prompts:** Subcommands attempting interactive prompts (`git credential`, `sudo`, `read`, Python's `input()`,
     or `npm login`) stall waiting on an unclosed `stdin`.
   - **GUI Popups & Console Flashes:** On Windows, untamed child processes trigger console host (`conhost.exe`) window flashes,
     and unhandled exceptions trigger Windows Error Reporting dialogs (`WerFault.exe`), stalling headless execution pipelines.

---

## 2. Syscall-Level Process Governance & Orphan Isolation

### 2.1 Process Groups and Job Objects

To guarantee that zero descendant processes escape termination, the kernel enforces atomic tree encapsulation at the
lowest available operating system boundary before any child process execution begins.

#### POSIX Systems (`!windows`)
1. **Setpgid Discipline:** During process configuration (`ConfigureProcessTreeCancel`), the child process is assigned
   its own process group via `SysProcAttr{Setpgid: true}`. The spawned PID becomes the process group leader:
   $$\text{PGID} = \text{PID}_{\text{root}}$$
2. **Negative PID Signalling:** Termination signals are delivered to the entire process group in a single kernel syscall:
   $$\text{kill}(-\text{PGID}, \text{SIGKILL})$$
   This terminates the process group leader and all descendant children that retained the process group.
3. **Cgroups v2 Fallback:** On modern Linux hosts where cgroup delegation is active, the supervisor places tool trees in a
   dedicated leaf cgroup under `/sys/fs/cgroup/fak/session-<id>/`. Termination writes `1` to `cgroup.kill`, atomically
   freezing and killing all tasks in the cgroup regardless of double-forking or `setpgid` escapes.

#### Windows Systems (`windows`)
1. **Job Object Containment:** The supervisor assigns newly spawned tool processes to a dedicated Windows Job Object
   configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`:
   ```c
   JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli;
   jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
   SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
   AssignProcessToJobObject(hJob, hProcess);
   ```
2. **Kill On Close Invariant:** When the session completes, times out, or closes its job handle, the Windows kernel
   automatically and immediately terminates every process currently assigned to that Job Object.

### 2.2 Fallback Descendant Walk & Recursive Tree Reaping

For environments where process groups or job object assignments are bypassed by external tools (e.g. subcommands that
explicitly call `setsid(2)` or invoke detached daemons), `ProcessSupervisor` maintains a recursive descendant tree
scanner.

Let $P$ be the set of all running processes in the host process table. Let $\text{parent}(p)$ denote the parent PID of process $p$.
The descendant set $\mathcal{D}(r)$ of root process $r$ is defined inductively:
$$\mathcal{D}_0(r) = \{r\}$$
$$\mathcal{D}_{k+1}(r) = \mathcal{D}_k(r) \cup \{p \in P \mid \text{parent}(p) \in \mathcal{D}_k(r)\}$$
$$\mathcal{D}(r) = \bigcup_{k=0}^{\infty} \mathcal{D}_k(r)$$

During `ReapSession(sessionID)`:
1. All root PIDs tracked for `sessionID` are retrieved.
2. For each root PID $r$:
   - If $\text{PGID}(r) > 1$, send `SIGKILL` to $-\text{PGID}(r)$.
   - Enumerate all transitive descendants $\mathcal{D}(r)$ via the relation census (`descendantPIDs`).
   - Iterate over $\mathcal{D}(r)$ in reverse topological order (deepest descendants first):
     $$\forall d \in \mathcal{D}(r) \setminus \{r\}, \quad \text{kill}(d, \text{SIGKILL})$$
   - Signal the root process: $\text{kill}(r, \text{SIGKILL})$.
3. Verify liveness via `processalive.Check(pid)`.
4. Ensure zero active processes remain associated with `sessionID`.

### 2.3 Port Isolation Protocol

To eliminate port collision and rogue socket reuse under high parallel load:

1. **Session Registry:** `ProcessSupervisor` maintains a bidirectional mapping:
   $$\text{Session} \longleftrightarrow \text{Ports} \subset [1024, 65535]$$
2. **Atomic Port Allocation / Registration:**
   - When a tool requires a TCP/UDP port, it must register the port with `TrackPort(sessionID, port)`.
   - If port $k$ is already registered to session $S_A$ ($S_A \ne S_B$), registration for $S_B$ is rejected with `ErrPortConflict`.
3. **Reaping Reclamation:**
   - During `ReapSession(sessionID)`, all ports associated with `sessionID` are unregistered.
   - Any listening processes bound to those ports are terminated as part of the tree reap.

---

## 3. Kernel vDSO Fast-Paths for Read-Only Tool Calls

### 3.1 Architectural Rationale: Fork/Exec Elimination

Invoking a subshell to answer deterministic, read-only queries introduces severe latency:

| Layer | Execution Mechanism | Median Latency | Allocations / FD Churn |
|---|---|---|---|
| **Traditional Tool Call** | `sh -c "cat /path/to/file"` | 12,000 µs – 35,000 µs | High (fork, execve, pipe, pty, sh) |
| **vDSO Tier 1 (Pure)** | In-kernel Go evaluation | < 0.5 µs | Zero OS processes, zero syscalls |
| **vDSO Tier 2 (Content-Cache)** | Version-keyed cache hit | 1.2 µs – 4.0 µs | Zero OS processes, memory-mapped lookup |
| **vDSO Tier 3 (Static)** | Immutable table read | < 0.2 µs | Zero OS processes |

By intercepting read-only tool calls in the kernel before shell dispatch, fak eliminates up to 99.9% of tool invocation
latency.

```
                    ┌──────────────────────────────────────┐
                    │       Incoming Tool Invocation       │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Is Tool Call Read-Only? │
                         └─────────────┬─────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                   YES│                                 │NO
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │     vDSO Fast-Path      │       │   ProcessSupervisor     │
         │ (T1 Pure / T2 Cache)    │       │ (Syscall / Fork / Exec) │
         └────────────┬────────────┘       └────────────┬────────────┘
                      │                                 │
                 Cache Hit                         Spawn With
                      │                            Process Group
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │ Return In-Memory Result │       │ Enforce Deadline & Reap │
         │   (Latency < 10 µs)     │       │   (Zero Orphan Leaks)   │
         └─────────────────────────┘       └─────────────────────────┘
```

### 3.2 3-Tier vDSO Architecture

1. **Tier 1 (Pure Registry):** Pure functions of arguments (formatting, mathematical evaluation, cryptographic digests).
   Gated on strict read-only and idempotent contracts.
2. **Tier 2 (Content Cache):** Read queries whose output depends on file tree state (e.g. file content reads, directory listings).
   - Key: $\mathcal{K} = \text{Hash}(\text{tool}, \text{args}, \text{world\_generation})$.
   - World Generation: A monotonic counter $W \in \mathbb{N}$ incremented whenever any mutating tool completes (file writes, edits, git commits, destructive commands).
   - Invalidation Soundness: $\forall \text{write}: W \leftarrow W + 1$. Any prior cache key $\mathcal{K}_W$ becomes unreachable, guaranteeing zero stale reads.
3. **Tier 3 (Static Table):** Platform constants, static schema manifests, and built-in help text.

---

## 4. Headless Execution Guarantees & Terminal/GUI Isolation

### 4.1 Terminal and PTY Deadlock Prevention

Unattended tool execution must never block on terminal I/O. `ProcessSupervisor` enforces three pipe invariants:

1. **Closed/Null Standard Input:**
   ```go
   cmd.Stdin = nil // or os.Open(os.DevNull)
   ```
   Interactive subcommands attempting to prompt the user immediately receive `EOF` or `EIO`, preventing indefinite hangs.
2. **Asynchronous Bounded Pipe Draining:**
   Standard output and standard error must be consumed continuously via asynchronous goroutines pumping into bounded buffers
   (default 1 MB buffer). If a child process produces unbounded output, the buffer captures the head and tail while
   preventing the pipe from backing up and stalling the child.
3. **PTY Suppression:**
   Tools must not allocate a pseudo-terminal (PTY) unless explicitly demanded by an interactive session. Unattended runs
   must execute with raw pipes and `TERM=dumb` to suppress ANSI animation loops and escape sequences.

### 4.2 Windows GUI Popup & Console Window Suppression

On Windows hosts, child process launches must prevent GUI dialog creation and console flashing:

1. **Process Creation Flags:**
   ```go
   cmd.SysProcAttr = &syscall.SysProcAttr{
       HideWindow:    true,
       CreationFlags: 0x08000000 | 0x00000008, // CREATE_NO_WINDOW | DETACHED_PROCESS
   }
   ```
2. **Error Box Suppression:**
   Before launching tools, child environment and process flags enforce:
   $$\text{SetErrorMode}(\text{SEM\_FAILCRITICALERRORS} \mid \text{SEM\_NOGPFAULTERRORBOX})$$
   This prevents crashes in native subprocesses (e.g., C++ binaries or Python DLL failures) from displaying blocking Windows
   Error Reporting dialog boxes (`WerFault.exe`).

### 4.3 Fail-Closed Deadline Enforcement

Every supervised execution carries a hard wall-clock deadline:

1. **Deadline Specification:** An absolute timestamp $T_{\text{deadline}} = T_{\text{start}} + \Delta t_{\text{max}}$.
2. **Watchdog Cadence:** `ProcessSupervisor` runs a periodic background watchdog ticker ($\tau \le 50\,\text{ms}$).
3. **Escalation Protocol:**
   - At $T_{\text{deadline}}$, the supervisor initiates emergency session reaping.
   - On POSIX: `SIGKILL` is sent immediately to $-\text{PGID}$ and all descendant PIDs.
   - On Windows: The Job Object is terminated or `KillPID` is invoked on the process tree.
   - The session status is marked as `TimedOut = true`.

---

## 5. ProcessSupervisor Architecture & State Machine

### 5.1 Lifecycle State Machine

```
              ┌───────────────┐
              │  UNTRACKED    │
              └───────┬───────┘
                      │ TrackProcess(sessionID, pid)
                      ▼
              ┌───────────────┐
              │    ACTIVE     ├──────────────────────────┐
              └───────┬───────┘                          │
                      │                                  │
         Deadline     │               Parent Exits       │
         Expires      │               Children Live      │
                      ▼                                  ▼
              ┌───────────────┐                  ┌───────────────┐
              │   TIMED_OUT   │                  │ ORPHAN_DETECT │
              └───────┬───────┘                  └───────┬───────┘
                      │                                  │
                      │      ReapSession(sessionID)      │
                      └───────────────┬──────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │    REAPING    │
                              └───────┬───────┘
                                      │ All trees killed, ports freed
                                      ▼
                              ┌───────────────┐
                              │    REAPED     │
                              │ (Zero Leaks)  │
                              └───────────────┘
```

### 5.2 Go Interface Specifications

```go
package procguard

import (
	"context"
	"os/exec"
	"time"
)

// TrackedProcess describes an active or reaped process under supervision.
type TrackedProcess struct {
	PID       int       `json:"pid"`
	SessionID string    `json:"session_id"`
	PGID      int       `json:"pgid,omitempty"`
	StartTime time.Time `json:"start_time"`
	Deadline  time.Time `json:"deadline,omitempty"`
	Cmdline   string    `json:"cmdline,omitempty"`
	Children  []int     `json:"children,omitempty"`
	Ports     []int     `json:"ports,omitempty"`
	TimedOut  bool      `json:"timed_out,omitempty"`
}

// ProcessSupervisor manages process lifecycles, process groups, and port bindings.
type ProcessSupervisor struct {
	// unexported synchronization and mapping state
}

// NewProcessSupervisor creates a new process supervisor with the specified options.
func NewProcessSupervisor(opts ...SupervisorOption) *ProcessSupervisor

// Core Lifecycle Methods:
func (s *ProcessSupervisor) TrackProcess(sessionID string, pid int)
func (s *ProcessSupervisor) TrackProcessWithDeadline(sessionID string, pid int, deadline time.Time)
func (s *ProcessSupervisor) TrackProcessTimeout(sessionID string, pid int, timeout time.Duration)
func (s *ProcessSupervisor) TrackChildPID(sessionID string, childPID int)
func (s *ProcessSupervisor) TrackPort(sessionID string, port int) error
func (s *ProcessSupervisor) ReleasePort(sessionID string, port int)
func (s *ProcessSupervisor) SessionPorts(sessionID string) []int
func (s *ProcessSupervisor) ReapSession(sessionID string) ([]int, error)
func (s *ProcessSupervisor) ActiveProcesses() []TrackedProcess
func (s *ProcessSupervisor) ConfigureCommand(cmd *exec.Cmd, sessionID string)
func (s *ProcessSupervisor) ExecuteCommand(ctx context.Context, sessionID string, cmd *exec.Cmd) error
func (s *ProcessSupervisor) Close() error
```

---

## 6. Mathematical Invariants & Verification Matrix

### 6.1 Formal Invariants

1. **Zero-Leak Invariant:**
   $$\forall S, \quad \text{ReapSession}(S) \implies |\{p \in \text{Processes} \mid \text{session}(p) = S \land \text{alive}(p)\}| = 0$$
   Following session reaping, the count of living processes associated with the session is strictly zero.

2. **Port Exclusivity Invariant:**
   $$\forall S_1 \ne S_2, \quad \text{Ports}(S_1) \cap \text{Ports}(S_2) = \emptyset$$
   No two concurrent sessions may hold the same port allocation.

3. **High-Volume Bounded Execution Invariant:**
   For $N = 1,000$ tool executions across concurrent sessions:
   $$\lim_{t \to \infty} |\text{ActiveProcesses}()| = 0 \quad \text{and} \quad |\text{TrackedPorts}| = 0$$
   After completion and reaping of all $N$ commands, active process tables and port allocations completely drain to zero.

### 6.2 Verification Matrix

| Test Case | Method | Invariant Verified | Target Outcome |
|---|---|---|---|
| `TestSupervisor_SubprocessTrackingAndReaping` | Real subshell + grandchild spawn | Section 2.1, 2.2 | Full tree terminated; `reapedPIDs` contains parent and descendants; `alive=false`. |
| `TestSupervisor_HighVolumeSyntheticCommandsZeroLeaks` | 1,000 concurrent tool executions | Section 6.1 (Inv 1, 3) | 1,000 commands executed; `ActiveProcesses()` length == 0; zero leaked processes. |
| `TestSupervisor_DeadlineTimeoutEnforcement` | Real long-running sleep command | Section 4.3 | Watchdog fires; process automatically reaped; session marked `TimedOut`; zero leaked processes. |
| `TestSupervisor_PortIsolationAndReclaim` | Multiple sessions with shared ports | Section 2.3, 6.1 (Inv 2) | Port conflict rejected; ports freed on reap. |
