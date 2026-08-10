# microfleetdemo — the all-in-one native fak proof for tiny agents

`microfleetdemo` runs a complete **24-agent support fleet in one Go process** and
puts fak's micro-agent benefits on one screen. It uses the real native packages,
not a slide or a hand-entered savings table:

- `microagent.Descriptor` references one shared base instead of copying it into every agent;
- `microagent.ManagedContext` keeps every long-running context under a hard cap and preserves artifact pointers while compacting;
- `microagent.TenantQueue` schedules interactive and batch tenants through weighted envelopes;
- `microagent.ResidentCap` plus `HibernationStore` keep only four agents resident and park the other twenty;
- `microagent.ToolExec` routes both tool attempts through a real `kernel.Kernel` + `adjudicator.Adjudicator`, proving a denied refund never reaches the backend;
- `microagent.EgressPolicy` allows the named API host and refuses an off-list destination before network I/O.

No key, model, GPU, browser, subprocess, or network is needed.

## Run the working spine

```bash
go run ./cmd/microfleetdemo
go run ./cmd/microfleetdemo -selfcheck
go run ./cmd/microfleetdemo -json
```

Representative deterministic output (byte counts come from the real encoded
contexts and descriptors; token-turn counts use the demo's documented 4-byte
estimator for both arms):

```text
FAK MICROFLEET — one process, many tiny agents, every native gate
fleet       24 agents × 12 turns; peak resident 4; parked 20 (27820 bytes)
context     541152 -> 42240 resident token-turns (92.2% avoided); 192 compactions; descriptors 96.3% smaller than copied bases
scheduler   batch       12 tasks
scheduler   interactive 12 tasks
tool floor  1 allow · 1 deny · denied dispatches = false
egress      1 allow · 1 deny (off-list destination refused)
VERDICT     fak turns copied context + unbounded residents + best-effort safety into bounded references + fair scheduling + pre-exec denial
PROOF       go run ./cmd/microfleetdemo -selfcheck
```

## What the 92.2% means

The comparison is deliberately narrow and reproducible. The baseline is a
straightforward micro-agent implementation that retains its copied shared base,
task delta, and current turn in each agent context on every turn. The fak arm
retains only the bounded managed context because the immutable base is addressed
by `BaseID`. The metric sums resident context tokens across all 288 agent-turns.
It is a **modeled context-residency reduction for this fixed corpus**, not a
provider bill, latency benchmark, or universal percentage.

The descriptor reduction is the encoded JSON descriptor bytes versus copying the
same base plus task delta once per agent. The residency, compaction, scheduling,
tool-floor, and egress rows are observed effects from the native implementations.
`-selfcheck` refuses success unless both reductions exceed 80%, the resident peak
stays at four, all 20 overflow agents are parked, both tenant queues drain, and
the denied action never dispatches.
