# Sandbox Tiering Ladder, Low-Ego OCI/WASI/MCP Contracts, and Gym Lifecycle Specification

> **Status:** Approved Architectural Specification  
> **Issue:** #11534  
> **Packages:** `internal/sandbox` (Tier 2 Foundation-Composite), `internal/gym` (Tier 4 Composer)  
> **Date:** September 2026  
> **Authors:** fak Architecture & Runtime Guild  

---

## 1. Executive Summary & Problem Statement

Autonomous AI agents executing arbitrary bash commands, code interpreters, build pipelines, and external tool calls introduce an unprecedented operational risk surface. Traditional security perimeters assume static application boundaries; in contrast, agentic workloads generate novel code dynamically, invoke compiler toolchains, install external third-party packages, and inspect filesystem trees.

To protect host machines, developer checkouts, and multi-agent workspaces without imposing catastrophic latency or inventing proprietary runtime silos, **fak** defines:

1. **A Tiered Isolation Ladder**: A three-rung progression (**L0 Wasm**, **L1 Host Native**, **L2 Virtual/gVisor**) that dynamically balances startup latency (<1ms to <50ms) against host kernel isolation.
2. **Low-Ego OCI / WASI / MCP Contracts**: Adopting industry-standard runtime abstractions (OCI runtime bundle specifications, WASI system interfaces, and Anthropic Model Context Protocol stdio/SSE bridges) rather than inventing bespoke container formats.
3. **Gym Lifecycle Invariants**: Deterministic, high-throughput reinforcement learning and evaluation execution characterized by:
   - **Sub-10ms Copy-on-Write (CoW) Snapshots** for instant state rollbacks between test episodes.
   - **Deterministic Output Normalization** stripping ANSI color codes, host paths, and volatile line breaks to produce byte-identical SHA256 hashes across disparate hosts.
   - **Fail-Closed Egress** guaranteeing zero unmediated network sockets by default.
   - **DOS Lane Disjointness** ensuring agents cannot mutate or read sibling worker trees.

### Strategic Alignment with fak Core Tenets (P1–P4)

- **P1 (Managed Context):** Sandboxed tool executions strip terminal noise, ANSI escape codes, and host path leaks before stdout/stderr reach model context, eliminating token inflation.
- **P2 (Net-True Efficiency):** Sub-10ms CoW snapshot restores eliminate container startup overhead, enabling thousands of evaluation steps per minute.
- **P3 (Bounded Adaptation):** Explicit capability envelopes (`CapabilityEnvelope`) gate filesystem paths and syscall privileges; ungranted operations fail closed with closed refusal tokens.
- **P4 (Integrated Operations):** All sandbox refusals map directly to the closed vocabulary in `dos.toml` (`SANDBOX_UNAVAILABLE`, `LANE_PATH_ESCAPE`, `EGRESS_BLOCKED`, `SIBLING_LANE_TOUCH`, `SECRET_EXFILTRATION_ATTEMPT`), integrating seamlessly with `fak recover` and supervisor replanning.

---

## 2. Threat Model & Security Boundaries

Agent execution environments operate in an adversarial zero-trust context. Attack vectors originate not only from malicious external prompt injections, but also from hallucinated agent commands, buggy third-party packages, or concurrent worker collisions.

```
+-------------------------------------------------------------------------------+
|                             Model / Agent Loop                                |
+-------------------------------------------------------------------------------+
                                     |  Proposes ToolCall / Bash Execution
                                     v
+-------------------------------------------------------------------------------+
|                       fak Kernel Adjudication Boundary                        |
|   (Policy Engine, Context-MMU, DOS Lane Leases, Capability Floor)             |
+-------------------------------------------------------------------------------+
                                     |  Permitted with Tiered Spec
                                     v
+===============================================================================+
|                           SANDBOX ISOLATION LADDER                            |
|                                                                               |
|  +---------------------+  +---------------------+  +-----------------------+  |
|  |    Tier L0: Wasm    |  | Tier L1: Native OS  |  |  Tier L2: Virtualized |  |
|  |  (Linear Memory)    |  | (Linux NS/Landlock) |  |   (gVisor / MicroVM)  |  |
|  +---------------------+  +---------------------+  +-----------------------+  |
|            |                         |                          |             |
+============|=========================|==========================|=============+
             |                         |                          |
             v                         v                          v
+-------------------------------------------------------------------------------+
|                             Contained Environment                             |
|  - Read-Only System Mounts        - Ephemeral CoW Overlay Workspace           |
|  - Egress Blocked (No Sockets)    - Scrubbed Environment (No Ambient Secrets) |
|  - Strict LaneTree Path Masking   - Deterministic Clock & PRNG                |
+-------------------------------------------------------------------------------+
```

### Threat Vectors & Mitigations

| Threat Vector | Attack Scenario | Mitigation in Sandbox / Gym | Refusal Token |
|---|---|---|---|
| **Host Kernel Compromise** | Agent executes zero-day kernel exploit via untrusted compiler or binary. | **Tier L2 Virtualization** (gVisor user-space kernel or Firecracker microVM) intercepts all syscalls; host kernel remains unexposed. | `SANDBOX_UNAVAILABLE` (if L2 requested but unsupported) |
| **Sibling Lane Tampering** | Concurrent subagent on Lane A attempts to read or modify files in Lane B. | **Lane Tree VFS Isolation** restricts writable mounts strictly to `Spec.LaneTree`. Host paths outside the lease are unmapped or read-only. | `SIBLING_LANE_TOUCH` |
| **Filesystem Directory Escape** | Malicious script uses `cd ../../../etc` or symlink traversal to access host root. | **Path Canonicalization & Landlock/Chroot** traps relative path escapes; access outside `WorkspaceDir` is blocked. | `LANE_PATH_ESCAPE` |
| **Data & Secret Exfiltration** | Prompt injection dumps environment variables (`$AWS_SECRET_KEY`) or uploads repo bytes via `curl`. | **Fail-Closed Network Egress** (unshared netns; default `EgressBlocked`) plus **Environment Sanitization** (stripping all non-whitelisted env vars). | `EGRESS_BLOCKED`, `SECRET_EXFILTRATION_ATTEMPT` |
| **Resource Starvation / DoS** | Fork bomb, unbounded memory allocation, or infinite loop spinning CPU cores. | **Cgroups v2 / Job Object Resource Ceilings** enforcing `MemoryLimitBytes`, `CPULimitPercent`, and wall-clock `TimeoutMS`. | Process killed; exit code recorded with audit log |
| **Digest Non-Determinism** | Volatile timestamps, randomized memory pointers, and terminal color sequences break verification hashes. | **Deterministic Output Normalization** strips ANSI escapes, maps local workspace paths to `/workspace`, and canonicalizes CRLF line endings. | None (normalized output guaranteed) |

---

## 3. Tiered Isolation Ladder

The isolation ladder provides three explicit rungs. A tool execution is assigned to the lowest viable tier that fulfills its required capability envelope.

```
       Isolation Depth
              ^
              |                                 +-----------------------------+
              |                                 |       Tier L2: Virtual      |
              |                                 |  - gVisor / MicroVM (KVM)   |
              |                                 |  - Independent guest kernel |
              |                                 |  - ~20-50ms cold, <10ms CoW |
              |                                 +-----------------------------+
              |                                                ^
              |                                                | Escalate on untrusted code/pkgs
              |                                 +-----------------------------+
              |                                 |    Tier L1: Host Native     |
              |                                 |  - Linux NS + Landlock      |
              |                                 |  - macOS Seatbelt, Win Job  |
              |                                 |  - <5ms startup latency     |
              |                                 +-----------------------------+
              |                                                ^
              |                                                | Escalate on arbitrary native CLI
              |  +-----------------------------+               |
              |  |        Tier L0: Wasm        |---------------+
              |  |  - WASI preview1/preview2   |
              |  |  - In-process memory sandbox|
              |  |  - <1ms startup latency     |
              |  +-----------------------------+
              +-------------------------------------------------------------------> Execution Velocity
```

### Ladder Specification Matrix

| Metric / Dimension | Tier L0: WebAssembly (`l0_wasm`) | Tier L1: Host Native OS (`l1_native_os`) | Tier L2: Virtualized / MicroVM (`l2_virtual`) |
|---|---|---|---|
| **Startup Latency** | **< 1 millisecond** | **< 5 milliseconds** | **~20–50 ms cold; < 10 ms restore** |
| **Memory Footprint** | ~2–8 MB | ~5–15 MB | ~30–64 MB |
| **Isolation Mechanism** | WebAssembly linear memory bounds; WASI capability imports | OS namespaces (PID, MNT, NET, IPC, UTS), cgroups v2, Landlock LSM, seccomp-BPF | User-space kernel (gVisor `runsc`) or hardware virtualization (KVM/Firecracker) |
| **Filesystem View** | Pre-opened virtual directories only | Restricted bind mounts; strict read-only system roots; CoW workspace | Fully isolated virtual rootfs with ephemeral overlay |
| **Network Egress** | Zero socket access by design | Unshared network namespace (loopback only if explicitly permitted) | Virtualized network stack (Netstack / TAP) filtered to default-deny |
| **Host Syscall Surface** | Zero host syscalls | Filtered host syscalls (seccomp allowlist) | Zero direct host syscalls; all syscalls handled by guest/user-space kernel |
| **Ideal Workloads** | Deterministic parsers, regex evaluation, data transforms, safe formatters | Trusted developer tools (`git`, `go build`, `pytest`), local repo analysis | Untrusted package builds (`npm install`, `pip install`), adversarial red-team benchmarks |

### Escalation & Selection Invariants

1. **Default-Deny Escalation:** Every tool request defaults to the lowest viable tier. Escalation from L0 to L1 or L2 requires explicit capability negotiation (e.g., `proc.exec` or `native.binary`).
2. **Untrusted Code Policy:** Any execution payload marked with an untrusted taint label (`abi.TaintTainted`) that attempts to run arbitrary downloaded binaries or shell scripts must be admitted to **Tier L2** or refused.
3. **Graceful Fallback Prohibition:** If a policy mandates Tier L2 (`l2_virtual`) but virtualization is unavailable on the host, the kernel **must not** silently degrade to L1. It must fail closed emitting `ErrSandboxUnavailable` ("SANDBOX_UNAVAILABLE").

---

## 4. Low-Ego OCI / WASI / MCP Contracts

Rather than constructing bespoke container formats or proprietary daemon runtimes, fak adheres to a **low-ego** architecture: it wraps industry-standard protocols with kernel-level governance.

### 4.1 OCI Runtime Specification Contract

For Tier L1 and Tier L2 execution, fak compiles an in-memory or ephemeral OCI `config.json` bundle complying with the Open Container Initiative Runtime Specification:

```json
{
  "ociVersion": "1.0.2",
  "process": {
    "terminal": false,
    "user": { "uid": 1000, "gid": 1000 },
    "args": ["sh", "-c", "go test ./..."],
    "env": ["PATH=/usr/local/bin:/usr/bin:/bin", "FAK_SANDBOX=1"],
    "cwd": "/workspace"
  },
  "root": {
    "path": "/var/lib/fak/rootfs/base",
    "readonly": true
  },
  "mounts": [
    {
      "destination": "/workspace",
      "type": "overlay",
      "source": "overlay",
      "options": ["lowerdir=/repo", "upperdir=/tmp/cow_upper", "workdir=/tmp/cow_work"]
    },
    {
      "destination": "/tmp",
      "type": "tmpfs",
      "source": "tmpfs",
      "options": ["nosuid", "nodev", "mode=1777", "size=67108864"]
    }
  ],
  "linux": {
    "namespaces": [
      { "type": "pid" },
      { "type": "mount" },
      { "type": "ipc" },
      { "type": "uts" },
      { "type": "network" }
    ],
    "resources": {
      "memory": { "limit": 536870912 },
      "cpu": { "shares": 512 }
    }
  }
}
```

- **Read-Only Rootfs:** The base OS filesystem is permanently mounted `MS_RDONLY`.
- **Ephemeral CoW Layer:** The agent workspace is mounted as an OverlayFS upper directory or tmpfs layer. Discarding the upper directory resets the environment instantly.
- **Rootless Execution:** Processes execute as non-root UID `1000:1000` with all Linux capabilities dropped (`CAP_SYS_ADMIN`, `CAP_NET_RAW`, etc. are removed).

### 4.2 WASI System Interface Contract (Tier L0)

For in-process deterministic execution:

- **Pre-opened Directories:** Only directory file descriptors explicitly listed in `Spec.LaneTree` are pre-opened (`wasi_snapshot_preview1.fd_prestat_get`). Attempts to access unmapped descriptors return `WASI_EBADF`.
- **Clock Virtualization:** `clock_time_get` for `CLOCK_REALTIME` returns deterministic timestamps or virtual sequence ticks, preventing timing-based covert channels and non-deterministic test outputs.
- **Deterministic Entropy:** System PRNG (`random_get`) is backed by a deterministic ChaCha20 generator seeded from the session's hash-chained trace ID.

### 4.3 Model Context Protocol (MCP) Bridge Contract

When executing third-party MCP servers (e.g. SQLite, GitHub, filesystem tools):

- **Stdio Transport Containment:** The MCP server process runs inside a designated sandbox tier. Stdin and stdout are framed as JSON-RPC messages mediated by the fak gateway proxy.
- **Subprocess Lockdown:** MCP servers are forbidden from launching arbitrary background subprocesses.
- **Audit Interception:** Every MCP tool call request and response is inspected and logged into the `ExecutionResult.Audits` slice before admission to the model context.

---

## 5. Gym Lifecycle & Architectural Invariants

The **Gym** is fak's execution engine for reinforcement learning, benchmark evaluation (SWE-bench, AgentDojo), and iterative agent trajectory generation. In an evaluation gym, an agent executes hundreds of candidate actions across multiple episodes.

```
+-------------------------------------------------------------------------------+
|                            Gym Episode Lifecycle                              |
+-------------------------------------------------------------------------------+

   1. Init / Warm Base Instance
          |
          v
   2. Snapshot Base State (Filesystem CoW + Memory Checkpoint) [ID: snap-xyz]
          |
          +--------------------------------------------+
          |                                            |
          v                                            |
   3. Execute Action Step (req: ExecutionRequest)       | Episode Action Loop
          |                                            |
          v                                            |
   4. Normalize Output (NormalizeOutput)               |
          |                                            |
          v                                            |
   5. Step Verification / Trajectory Record            |
          |                                            |
          +--------------------------------------------+
          |
          v
   6. Episode Reset / Restore (snap.Restore(ctx) < 10ms)
          |
          v
   7. Ready for Next Episode (Zero Churn / Clean Slate)
```

### Invariant 1: Sub-10ms Copy-on-Write (CoW) Snapshots

- **The Problem:** Traditional container tear-down and rebuild cycles take 1,000ms to 10,000ms. Running a 100-step evaluation trajectory would take minutes of pure idle waiting.
- **The Invariant:** `SnapshotHandle.Restore(ctx)` and `Instance.Reset(ctx)` **must complete within 10 milliseconds** of wall-clock time.
- **Mechanism:**
  - **Filesystem:** Ephemeral OverlayFS upper-directory or Btrfs/ZFS snapshot. A reset unmounts and remounts a fresh tmpfs upper directory in ~2ms.
  - **Memory:** Process fork-server pattern or microVM memory page snapshotting via userfaultfd / KVM dirty ring.
  - **Verification:** Benchmarked continuously via `BenchmarkGymResetLatency`.

### Invariant 2: Deterministic Output Normalization

Raw command output contains host-specific noise that corrupts verification digests:
1. **ANSI Color / Terminal Escape Sequences:** Terminal sequences (e.g., `\x1b[31;1m`, `\x1b[0m`, cursor repositioning `\r`) vary depending on terminal allocation.
2. **Host Workspace Directory Leaks:** Traces like `/home/USER/workspace/fak/cmd/main.go:42` differ across developer laptops and CI machines.
3. **Carriage Return Variations:** Windows tools emit `\r\n`, whereas POSIX tools emit `\n`.

**The Invariant:** `NormalizeOutput(raw, workspaceDir)` guarantees that identical tool commands yield **cryptographically byte-identical output**:
- All ANSI escape codes are stripped.
- All occurrences of `workspaceDir` (and its native slash/backslash variations) are rewritten to canonical `/workspace`.
- All `\r\n` and standalone `\r` carriage returns are normalized to LF (`\n`).
- Trailing whitespace per line is trimmed.

### Invariant 3: Egress Fail-Closed

- **The Invariant:** Network egress is disabled by default (`EgressPolicy = EgressBlocked`).
- **Enforcement:**
  - Network namespace is unshared with no default gateway.
  - Any attempt to open an `AF_INET` or `AF_INET6` socket immediately returns `EPERM` or `ENETUNREACH`.
  - The violation is captured in `ExecutionResult.Audits` and returned with structured token `ErrEgressBlocked` ("EGRESS_BLOCKED").
  - Outbound HTTP access (when explicitly required by a benchmark) must route through fak's proxy with cryptographic TLS SNI validation.

### Invariant 4: DOS Lane Disjointness & Sibling Protection

- **The Invariant:** An agent executing within a sandbox bound to lane $L_A$ can never read or write paths belonging to lane $L_B$.
- **Enforcement:**
  - `Spec.LaneTree` is populated from the lane's declared tree in `dos.toml`.
  - The VFS mount restricts write privileges strictly to globs matching `LaneTree`.
  - Any attempt to access sibling paths triggers `ErrSiblingLaneTouch` ("SIBLING_LANE_TOUCH") or `ErrLanePathEscape` ("LANE_PATH_ESCAPE").

### Invariant 5: Secret Containment & Ambient Environment Sanitization

- **The Invariant:** Ambient host environment variables (`GITHUB_TOKEN`, `AWS_ACCESS_KEY_ID`, `OPENAI_API_KEY`, private tokens) are scrubbed from the sandbox process table.
- **Enforcement:**
  - Only variables explicitly declared in `Spec.Env` or admitted via `FAK_SANDBOX_ENV_ALLOW` are passed to the container process.
  - Attempts to read secret locations (`~/.aws/credentials`, `~/.ssh/id_rsa`) are trapped and refuse with `ErrSecretExfiltrationAttempt` ("SECRET_EXFILTRATION_ATTEMPT").

---

## 6. Error & Refusal Taxonomy

Every rejection originating from the sandbox subsystem produces a structured, machine-verifiable refusal token belonging to the closed vocabulary:

```go
const (
	ErrSandboxUnavailable        = "SANDBOX_UNAVAILABLE"
	ErrLanePathEscape            = "LANE_PATH_ESCAPE"
	ErrEgressBlocked             = "EGRESS_BLOCKED"
	ErrSiblingLaneTouch          = "SIBLING_LANE_TOUCH"
	ErrSecretExfiltrationAttempt = "SECRET_EXFILTRATION_ATTEMPT"
)
```

### Refusal Mapping & Recovery Routing

| Refusal Token | Trigger Condition | Operator / Agent Recovery Action |
|---|---|---|
| `SANDBOX_UNAVAILABLE` | Host platform lacks required virtualization (KVM missing) or runtime daemon is offline. | Escalate to sanctioned fleet compute node (`docs/fleet-compute-nodes.md`) or adjust requested tier. |
| `LANE_PATH_ESCAPE` | Command attempted relative path traversal (`../`) outside the bounded workspace. | Rewrite command to operate strictly within the assigned workspace directory. |
| `EGRESS_BLOCKED` | Command attempted unauthorized outbound network connection. | Use offline cached assets or negotiate explicit allowlist proxy capability. |
| `SIBLING_LANE_TOUCH` | Command attempted to access files owned by another active DOS lease lane. | Acquire the target lane lease via `dos arbitrate` before initiating mutations. |
| `SECRET_EXFILTRATION_ATTEMPT` | Command attempted to inspect host credential stores or dump scrubbed environment. | Quarantine session; flag security violation in supervisor audit log. |

---

## 7. Canonical Interfaces & Go Architecture

The core abstractions reside in `internal/sandbox` (Tier 2) and are orchestrated by `internal/gym` (Tier 4).

### Key Type Architecture

```
                          +-------------------+
                          |     Provider      |
                          +-------------------+
                                    |
                                    | Create(ctx, Spec)
                                    v
                          +-------------------+
                          |     Instance      |
                          +-------------------+
                           /        |        \
                          /         |         \
                         v          v          v
                 Execute(...)   Reset(...)   Close()
                      |
                      v
             +-----------------+
             | ExecutionResult |
             +-----------------+
             | - Stdout        |
             | - Stderr        |
             | - Normalized*   |
             | - Audits        |
             +-----------------+
```

### Type Signatures (`internal/sandbox/types.go`)

```go
// Tier represents the isolation ladder rung.
type Tier string
const (
	TierL0Wasm     Tier = "l0_wasm"
	TierL1NativeOS Tier = "l1_native_os"
	TierL2Virtual  Tier = "l2_virtual"
)

// Spec defines the complete sandbox execution specification.
type Spec struct {
	Tier             Tier             `json:"tier"`
	Rootfs           string           `json:"rootfs,omitempty"`
	WorkspaceDir     string           `json:"workspace_dir"`
	LaneTree         []string         `json:"lane_tree,omitempty"`
	ReadOnlyPaths    []string         `json:"read_only_paths,omitempty"`
	WritablePaths    []string         `json:"writable_paths,omitempty"`
	MemoryLimitBytes int64            `json:"memory_limit_bytes,omitempty"`
	CPULimitPercent  int              `json:"cpu_limit_percent,omitempty"`
	FuelLimit        int64            `json:"fuel_limit,omitempty"`
	TimeoutMS        int64            `json:"timeout_ms,omitempty"`
	Env              []string         `json:"env,omitempty"`
	EgressPolicy     EgressPolicy     `json:"egress_policy"`
	Capabilities     []abi.Capability `json:"capabilities,omitempty"`
}

// Instance represents an active execution environment.
type Instance interface {
	Execute(ctx context.Context, req ExecutionRequest) (ExecutionResult, error)
	Reset(ctx context.Context) error
	Close() error
	Spec() Spec
}

// SnapshotHandle represents an immutable state checkpoint.
type SnapshotHandle interface {
	ID() string
	Restore(ctx context.Context) error
	Release() error
}

// Provider instantiates instances for a designated tier.
type Provider interface {
	Name() string
	Tier() Tier
	Available() bool
	Create(ctx context.Context, spec Spec) (Instance, error)
}
```

---

## 8. Verification Strategy & Architecture Gates

1. **Unit Conformance (`internal/sandbox/spec_test.go`):**
   - Verifies `Tier` properties (`IsolationLevel`, `Valid`, `RequiresVirtualization`).
   - Verifies `Spec` validation, capability envelopes, and JSON roundtrip fidelity.
   - Verifies `NormalizeOutput` across ANSI escape stripping, CRLF normalization, and Windows/Unix workspace path scrubbing.
   - Verifies mock provider, instance execution, and snapshot restore mechanics.
2. **Layering Architecture Gate (`internal/architest/architest_test.go`):**
   - Registers `sandbox` at **Tier 2 (Foundation-Composite)**: depends only on `abi` (Tier 0) and Go standard library.
   - Registers `gym` at **Tier 4 (Composer)**: composes `sandbox`, `policy`, and agent workflows.
   - Verified via `TestEveryPackageDeclaresTier`, `TestTierDeclarationsAreLive`, and `TestNoUpwardImports`.
3. **End-to-End Performance Gate:**
   - Evaluates sub-10ms CoW reset latency under concurrent worker sweeps to ensure zero regression in high-churn agent evaluation loops.
