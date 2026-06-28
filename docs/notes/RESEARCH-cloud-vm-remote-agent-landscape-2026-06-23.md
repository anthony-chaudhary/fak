---
title: "Cloud / VM / remote-control agent landscape + fak first-class-support strategy (2026-06-23)"
description: "A dated scan of recent Cursor + peer announcements (Codex, Jules, Devin, Copilot, Claude Code) and the sandbox-infra layer (E2B, Fly, Modal, Daytona, Cloudflare, Vercel), then a strategy read for where fak should add FIRST-CLASS support for virtual machines, cloud agents, and remote control. Verdict: fak is the adjudication + egress + reuse BOUNDARY for the cloud-agent era — not a VM provider, IDE, or cloud-agent product."
---

# Cloud / VM / remote-control agents — landscape + fak strategy

> A two-part working note. **Part A** scans what Cursor and its peers shipped from
> late 2025 through 2026-06-23 around running agents in the cloud, in VMs/sandboxes, and
> under remote control. **Part B** is the strategy read: where fak earns *first-class*
> support for VMs, cloud, and remote control given what it already is.
>
> **Method & honesty:** a 5-track parallel web scan (99 findings; Cursor; Codex/Jules/Devin;
> Copilot/Claude Code; the sandbox-VM infra layer; remote-control/fleet patterns). The
> adversarial verify pass **did not run** (session rate limit), so dates below are
> *agent-reported-from-primary-sources*, not independently re-checked here. High-confidence
> rows were fetched from vendor primaries (cursor.com/changelog, github.blog, anthropic.com,
> developers.openai.com, blog.google, cognition.com); weaker rows are flagged in
> *Sourcing* at the end. Treat exact days as ±a few.

---

## Part A — The convergence (what the field shipped)

By mid-2026 essentially every coding-agent vendor converged on the **same five-part
shape**. The differences are now execution-substrate and polish, not concept.

1. **Async cloud agent.** Delegate a task; it runs in an *isolated cloud VM/sandbox*
   pre-loaded with the repo; it returns a PR. — OpenAI Codex Cloud, Google Jules,
   Cognition Devin, Cursor Cloud Agents, GitHub Copilot coding agent, Anthropic Claude
   Code on the web, Snowflake CoCo.
2. **Remote/steering surface.** Phone, web, and chat (Slack/Teams/Jira/Linear) to launch,
   approve, redirect, and receive report-backs. — Codex-in-ChatGPT mobile (2026-05),
   Claude Code Remote Control (2026-02), Cursor mobile, "@mention the agent" in Slack for
   all four majors.
3. **Fleets on one objective** with a planner → dispatcher → synthesizer. — Copilot CLI
   `/fleet`, Devin "MultiDevin" (1 manager + up to 10 worker VMs), Cursor multi-agent (up
   to 8) + Multi-Agent Judging.
4. **Mission-control dashboard** to supervise many concurrent sessions by status. —
   Claude Code Agent View (Working / Needs-input / Idle / …), Cursor Agents Window,
   third-party AgentsRoom / pi-agent-dashboard / Hermes.
5. **Standard plumbing underneath.** Remote MCP over Streamable HTTP + OAuth 2.1 for tools;
   A2A (Linux Foundation, 150+ orgs) for agent-to-agent; first-class **local↔cloud
   handoff/resume** tying it together.

### The isolation-primitive consensus

For *untrusted, agent-generated* code the field settled on **hardware-isolated microVMs**,
not containers:

| Primitive | Who | Cold-start | Note |
|---|---|---|---|
| **Firecracker / Kata / Cloud Hypervisor microVM** | E2B, Fly (Machines/Sprites), Vercel Sandbox, Northflank | ~80–200 ms (Daytona warm 27 ms) | dedicated guest kernel per workload — the strong-isolation default |
| **gVisor** (user-space kernel) | Modal | 2–4 s CPU | moderate overhead, Python-native DX, elastic GPU |
| **V8 isolate** | Cloudflare Dynamic Workers | ~few ms | fast/dense but JS-only |
| **OS-level (bubblewrap / Seatbelt)** | Anthropic Claude Code (local) | n/a | the *local* boundary; avoids VM weight |
| **Container / devcontainer** | DIY self-host | ~sub-90 ms | shared-kernel — now seen as the weak baseline for untrusted code |
| **git worktree** | nearly every tool | n/a | **orthogonal**: file/concurrency isolation, *not* a security boundary; composed *under* a sandbox |

The cited framing: *"every hyperscaler reached for their strongest isolation primitive and
pointed it at AI. None of them reached for containers."* (AWS→Firecracker, Google→gVisor,
Azure→Hyper-V.)

### The one security primitive that recurs everywhere

**Deny-by-default network egress + per-destination credential injection so the agent
never sees the token.** Cloudflare's zero-trust egress proxy, Anthropic's git/credential
proxy (tokens stay outside the sandbox; egress relayed via a validating proxy), Daytona /
devcontainer egress firewalls. The agent-needs checklist that fell out of the survey:
**snapshot + fork** (cheap experiment/rollback), **deny-by-default egress** (the security
boundary), **persistence vs. per-second cost** (session caps: Vercel 45 min–5 h, E2B ~24 h;
or persistent VMs like Fly Sprites), **sub-200 ms cold-start** (warm pools + lazy FS).

### Selected dated timeline (load-bearing rows)

| Date | Vendor | What | Cat | Source |
|---|---|---|---|---|
| 2025-09 | GitHub | Copilot coding agent GA — async, runs in GitHub Actions cloud env, opens PRs | cloud-agent | github.blog/changelog/2025-09-25-… |
| 2025-10 | Cursor | 2.0 + Composer: agent-centric IDE, up to 8 parallel agents (worktree **or** remote machine), cloud agents | fleet | cursor.com/blog/2-0 |
| 2025-10 | Anthropic | Claude Code on the web + iOS: tasks in Anthropic-managed VMs, git via credential proxy | cloud-agent | claude.com/blog/claude-code-on-the-web |
| 2025-10 | Anthropic | Sandboxing: bubblewrap/Seatbelt + egress proxy; −84% permission prompts; OSS `sandbox-runtime` | vm-sandbox | anthropic.com/engineering/claude-code-sandboxing |
| 2025-10 | GitHub | Agent HQ — orchestrate *any vendor's* agents from GitHub mission control | fleet | github.blog/…/welcome-home-agents |
| 2026-01 | Fly.io | "Sprites": persistent Firecracker microVMs for agents, Claude pre-installed, checkpoint/restore | vm-sandbox | devclass.com/…/flyio-introduces-sprites |
| 2026-02 | Anthropic | Remote Control — agent stays **local**, phone/web is just a synced window (outbound HTTPS only) | remote | code.claude.com/docs/en/remote-control |
| 2026-03 | Cognition | "Devin manages Devins" — manager + up to 10 worker Devins, each own VM, merged to one PR | fleet | docs.devin.ai/release-notes |
| 2026-03 | Cursor | Self-hosted Cloud Agents — isolated-VM cloud agents behind the customer perimeter | cloud-agent | cursor.com/changelog/03-25-26 |
| 2026-04 | Anthropic | Routines — saved prompt runs on **Schedule / HTTP `/fire` / GitHub-event** triggers, unattended cloud VM | async | code.claude.com/docs/en/routines |
| 2026-04 | GitHub | Copilot CLI `/fleet` — planner→dispatch→poll→synthesize; **shared FS, no file locking** (named hazard) | fleet | github.blog/…/run-multiple-agents-…-fleet |
| 2026-04 | Cloudflare | Sandboxes GA: containers + snapshot/**fork** + zero-trust egress proxy (per-dest credential injection) | vm-sandbox | infoq.com/news/2026/04/cloudflare-sandboxes-ga |
| 2026-04 | LF / A2A | Agent2Agent 1 yr: 150+ orgs; MCP (vertical/tools) + A2A (horizontal/agent-to-agent) = enterprise default | fleet | hpcwire.com/aiwire/2026/04/09/… |
| 2026-05 | OpenAI | Codex in ChatGPT mobile — approve/redirect/watch cloud sandbox tasks from phone | cloud-agent | developers.openai.com/codex/cloud |
| 2026-06 | Cursor | 3.7: `/in-cloud` cloud subagent in its own VM+branch; <10-min env snapshot; reliable local↔cloud handoff | cloud-agent | cursor.com/changelog/cloud-in-agents-window |
| 2026-06 | Anthropic | Claude Code on web: fresh VM (4 vCPU/16 GB/30 GB) per session; `--remote` / `--teleport` cloud↔local | vm-sandbox | code.claude.com/docs/en/claude-code-on-the-web |

---

## Part B — Strategy: fak is the boundary, not the box

### Where fak sits today

fak's productized front door (`fak guard -- claude`) assumes a **local, ephemeral,
single-session, loopback** agent: the gateway binds `127.0.0.1:0` (`cmd/fak/guard.go:163`),
injects *only* `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` into the child (`guard.go:223-227`),
proxies the real provider in passthrough, and is **torn down when the child exits**
(`guard.go:199-208`). The fleet (issue-worker loop) spawns **local `claude -p` processes**
via Python tools on Windows scheduled tasks — no remote spawn, no cross-machine coordination.
Isolation today is **git-worktree only, and only for SWE-bench** (`internal/swebench/fleet.go`);
there is **no network egress policy** (`internal/adjudicator/decide.go` gates only self-modify
writes). Remote control is a **hard-coded Slack↔shell bridge** for the lab (private tooling),
not a general control plane. `fak serve` is **single-model**.

In one line: **the field moved compute into isolated cloud VMs and control onto phones;
fak's front door still assumes the agent is a child process on your laptop.**

### The strategic punchline

Moving compute into a VM and control onto a phone **widens the gap between the agent and
the human.** Every widening of that gap raises the value of a capability floor that is
**structural** — one that denies by shape and doesn't wait for a human to click *approve*.
fak's entire thesis ("deny by structure, not by asking the model or a human to behave") is
worth *more*, not less, exactly where there is **no human in the loop**: autonomous cloud
agents, always-on Automations/Routines, computer-use. fak should not chase the parts the
field already commoditized (the VM, the IDE, the dashboard). It should be **the boundary
that travels into every one of those VMs.**

And fak is uniquely shaped for it: the field is independently re-deriving the two halves of
fak's fused boundary —
- the **egress proxy** (a security gate at the network edge), and
- the **snapshot/fork** (cheap reuse of prior state)

— as *separate* pieces of infra. fak already does capability-floor **and** KV/prefix reuse
at **one in-process hop** (`internal/gateway` vDSO; cross-agent prefix-KV reuse proven in
`cmd/fleetserve`). That fusion is the differentiator to lean on.

### What to NOT build

- **Not a VM/sandbox provider** (E2B/Fly/Modal/Vercel own this; microVM is a hardware play).
- **Not an IDE or mission-control dashboard** (Cursor/Copilot/Agent View own this).
- **Not a cloud-agent product** (Devin/Jules/Codex own this).

fak's lane is the **adjudication + egress + reuse boundary** that any of the above can put
in front of (or inside) their VM.

### First-class capabilities to add, ranked

Each: *the industry pattern it answers · the closest existing fak seam · why it is
fak-shaped (uses the boundary) rather than commodity.*

1. **Egress firewall + credential-injection rung** — deny-by-default outbound, block the
   SSRF/cloud-metadata classes (`169.254.169.254`), and inject the per-destination credential
   so the agent's context never holds the token.
   *Answers:* the one security primitive every sandbox vendor ships.
   *Seam:* a new adjudicator rung beside the self-modify floor (`internal/adjudicator/decide.go`)
   + the gateway request path; reuse the existing `urllint`/`boundarylint` static witnesses.
   *Why fak-shaped:* it's literally another structural deny rung, and it composes with the
   capability floor in the same hop. **Highest leverage:** universal in the field, absent in
   fak, and exactly fak's pattern.
   *Shipped — first increment (2026-06-28):* the **cloud-metadata / link-local SSRF block**,
   the never-legitimate half of the egress rung. `internal/egressfloor` (a pure tier-1
   classifier) + a mandatory, non-elidable `rungEgress` in `internal/adjudicator/decide.go`
   refuse any tool call reaching the instance-metadata family (`169.254.169.254`,
   `metadata.google.internal`, `169.254.170.2`, `100.100.100.100`, `fd00:ec2::254`, and
   `169.254.0.0/16` / `fe80::/10`) with a new `EGRESS_BLOCK` reason — so a guarded agent on a
   VM cannot reach the endpoint that hands out the box's IAM credentials. Witness:
   `fak egress check` + `examples/remote-vm-guard/`. **Still to build:** the policy-configurable
   deny-by-default destination *allow-list* and the per-destination credential injection (the
   rest of this recommendation).

2. **`dos_arbitrate` as universal fleet admission control.** Copilot `/fleet` *names* the
   unsolved hazard, verbatim (verified 2026-06-23): *"Sub-agents share a filesystem with no
   file locking. If two agents write to the same file, the last one to finish wins—silently.
   No error, no merge, just an overwrite."* fak already solves this: lane/tree-disjointness
   admission (`dos_arbitrate`). Expose it as an MCP tool / gateway endpoint any orchestrator
   (Copilot `/fleet`, Cursor multi-agent, Devin MultiDevin) calls *before* dispatching
   parallel agents. *Why fak-shaped:* shipped, differentiated, and the rest of the field is
   hitting the bug fresh. "fak is the missing collision-safety layer for everyone's fleet."

3. **Network-safe gateway + remote-trigger control plane.** Finish the auth story
   (`Config.RequireKey` → mTLS / OAuth 2.1 / short-lived per-session tokens) so the gateway
   is safe off-loopback, then add a `/v1/fak/dispatch` trigger (Schedule / HTTP `/fire` /
   GitHub-event — mirroring Anthropic Routines) and generalize the private lab bridge from
   "Slack↔one shell" to "chat↔adjudicated session."
   *Why fak-shaped:* the differentiator isn't "another way to fire an agent" — it's that
   **every remotely-triggered, unattended run is adjudicated by the floor and witnessed**
   (`dos_commit_audit` / `dos_verify` already exist). Fire-and-forget *with* a verifiable
   capability floor and a witnessed result is precisely what you want when nobody is watching.
   The gap is real and felt: Anthropic's Routines docs (verified 2026-06-23) warn that
   *"a green status… does not mean the task succeeded… open the run to confirm"* — i.e. the
   platform itself ships fire-and-forget **without** a structural success witness, exactly the
   hole `dos_verify` fills. (Routines also already run on a deny-by-default-egress cloud env —
   `403` + `x-deny-reason: host_not_allowed` — corroborating the egress rung in #1.)

4. **Execution-target abstraction so the boundary travels** — `local | worktree | container
   | microVM | remote-SSH` (the four targets Cursor 3.0's Agents Window unifies). `fak guard`
   keeps the gateway wherever it runs; the agent inside any VM just points its base-URL env
   at it. Depends on (3)'s auth work.
   *Seam:* `cmd/fak/guard.go` + `internal/gateway/gateway.go`.

5. **Reuse/snapshot alignment.** When N cloud agents *fork from one snapshot* (Cloudflare
   fork, Fly checkpoint, E2B pause/resume), make fak the layer that also shares the **KV
   prefix cache** across them — the inference-side analog of the FS snapshot.
   *Seam:* `cmd/fleetserve`, `internal/radixkv`, `internal/gateway` vDSO (already proven).
   *Why fak-shaped:* nobody else fuses the security boundary with the reuse boundary.

6. **Verified local↔cloud handoff.** Handoff/resume is now first-class everywhere
   (Cursor `&`, Claude Code `--teleport`). A handoff is a *trust* boundary — you're resuming
   someone else's *claimed* state. Make resume **re-check** what actually shipped vs. what the
   cloud agent claimed (`dos_recall` re-verifies a memory's claims against git; `dos_status`
   folds run liveness/progress/region/resume). Lower priority, distinctive.

### Honest risks

- The **network/auth surface is the riskiest expansion.** Today's safety rests on
  loopback-only (`guard.go:173-175` re-asserts the no-auth-off-host warning). Items 1, 3, and
  4 all depend on finishing real auth (mTLS / OAuth / per-session tokens) *first* — a
  network-reachable kernel with a weak key is worse than no remote support.
- Items 2 and 5 are the **safe, already-differentiated** wins (build on shipped `dos` +
  `fleetserve`); start there while the auth foundation for 1/3/4 is laid.

---

## Sourcing & confidence

- **Directly verified 2026-06-23 (WebFetch against primary)** — the three claims the strategy
  rests on: (a) Copilot `/fleet`'s "shared filesystem, no file locking, silent overwrite"
  hazard (github.blog `/fleet` post, quoted verbatim above → recommendation #2); (b) Claude
  Code's deny-by-default egress via a unix-socket→validating proxy + git credentials kept
  *outside* the sandbox + bubblewrap/Seatbelt + 84% prompt cut (anthropic.com/engineering →
  recommendation #1); (c) Routines' three triggers and the literal
  `POST …/v1/claude_code/routines/{id}/fire` bearer endpoint, plus its deny-by-default cloud
  env and "green ≠ succeeded" caveat (code.claude.com/docs/en/routines → recommendation #3).
- **High confidence** (vendor-primary, fetched): Cursor changelog 2.0→3.8; Copilot GA /
  `/fleet` / Agent HQ (github.blog); Codex Cloud/subagents (developers.openai.com); Jules
  beta + Tools/API (blog.google); Devin 2.0 + MultiDevin + v3 API (cognition.com /
  docs.devin.ai); Claude Code web / sandboxing / Routines / Remote Control (anthropic.com,
  code.claude.com); Cloudflare Sandboxes GA (InfoQ); Fly Sprites (devclass); A2A (HPCwire);
  remote-MCP OAuth 2.1.
- **Medium / weaker** (synthesis or secondary): Windsurf→"Devin Desktop" relaunch; Copilot
  standalone desktop GA (2026-06-17); the Copilot "cloud agent" rename date (~2026-04);
  Jules I/O-2026 specifics (2 M ctx, CI self-heal); Snowflake CoCo endpoint string.
- **Distrust / unverified numbers** (single-snippet): Daytona 850 K daily runs; E2B 375×
  growth / 24 h cap (competitor-sourced); "5 % solved sandboxing / 79 % use agents";
  OpenClaw 369 K★ / 3.2 M users. Not load-bearing for the strategy.
- The **automated verify pass did not run** (session limit); the three strategy-spine claims
  above were instead verified by hand (WebFetch). The remaining *dates* are still
  agent-reported-from-primaries — re-run the verify fan-out before quoting any single date as
  authoritative.

## Verdict

The field commoditized the **VM, the dashboard, and the cloud-agent product**, and
standardized the wiring (MCP + A2A). The two things it is *still re-inventing piecemeal* are
fak's home turf: a **structural egress/credential boundary** and **fleet collision-safety** —
and the thing nobody else has is fak's **fused security + reuse boundary in one hop**. So
fak's first-class "cloud/VM/remote" support is not a VM or a dashboard; it is the **boundary
that rides into every VM and in front of every fleet** — most valuable precisely where the
human has stepped away. Start with `dos_arbitrate`-as-a-service and the egress rung (safe,
differentiated); lay real gateway auth before going network-reachable.
