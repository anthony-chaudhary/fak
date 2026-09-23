# Safe single-node agent admission: public runtime work

An independent objective owns one primary `GoalID`; its child tasks, attempts, tool calls, and subprocesses inherit that root while retaining distinct identities. A request containing “do X” and “do Y” creates two goals. The host may retain hundreds of primary goals and thousands of child intents as bounded logical backlog while admitting only a measured number of active process, compiler, and model grants. The acceptance target is safe queuing, reconciliation, and fair service under contention; it is not a throughput or simultaneous-process claim.

| Order | Native ticket | Issue | Production seam |
| --- | --- | --- | --- |
| 1A | [TICKET-01](TICKET-01.md) | [#13489](https://github.com/anthony-chaudhary/fak/issues/13489) | Durable queue lineage separates goal, task, session, and attempt. |
| 1B | [TICKET-02](TICKET-02.md) | [#13490](https://github.com/anthony-chaudhary/fak/issues/13490) | Controller begins a fenced launch and hands stable identity to the wrapper. |
| 1C | [TICKET-08](TICKET-08.md) | [#13491](https://github.com/anthony-chaudhary/fak/issues/13491) | Guarded wrapper registers exact child identity and terminal outcome. |
| 2 | [TICKET-03](TICKET-03.md) | [#13492](https://github.com/anthony-chaudhary/fak/issues/13492) | Production controller uses per-goal fairness and a cross-process host grant. |
| 3 | [TICKET-04](TICKET-04.md) | [#13493](https://github.com/anthony-chaudhary/fak/issues/13493) | Heavy tool subprocesses consume that shared grant before start. |
| 3A | [TICKET-11](TICKET-11.md) | [#13494](https://github.com/anthony-chaudhary/fak/issues/13494) | A public guarded-run executable holds or transfers the grant through contained child exit. |
| 3B | [TICKET-12](TICKET-12.md) | [#13495](https://github.com/anthony-chaudhary/fak/issues/13495) | Long-lived public launcher paths consume admission; a census classifies remaining direct starts. |

Each ticket has a native contract `safe-agent-concurrency-<number>` in the public ticket store and an issue mirror with the same marker. Existing `goalregistry`, `agentqueue` lifecycle methods, `hostgrant`, `agentsched`, `microagent`, `toolprocgate`, and guarded-child paths are reused. Closed #8119 already propagates launch `GoalID`; the remaining identity work is durable queue lineage. Closed #11177 supplied a scheduler primitive, but its observed callers are benchmarks rather than the production controller. These statements describe the source at planning time; each ticket requires its own post-edit call-path witness.

The companion factory owns its own entrypoint adapters, pressure telemetry, and integrated logical-intent safety witness. This public work does not require a private import or a separate host ledger.
