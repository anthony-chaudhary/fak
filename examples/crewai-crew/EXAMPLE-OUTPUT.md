# Captured output

`python examples/crewai-crew/crew_demo.py`, run from the repository root with no
arguments and no environment overrides (the dependency-free local-evaluator path):

```text
== Part 1: every crew tool call passes the capability floor ==
verdict source: local policy.json evaluator

  manager coordination tools:
  Manager     delegate_work      verdict=ALLOW reason=NONE          -> run
  Manager     ask_coworker       verdict=ALLOW reason=NONE          -> run
  worker subtask tools:
  Researcher  web_search         verdict=ALLOW reason=NONE          -> run
  Researcher  fetch_url          verdict=ALLOW reason=NONE          -> run
  Analyst     read_dataset       verdict=ALLOW reason=NONE          -> run
  Analyst     summarize          verdict=ALLOW reason=NONE          -> run
  Writer      send_email         verdict=DENY  reason=POLICY_BLOCK  -> BLOCKED
  Writer      exfiltrate_notes   verdict=DENY  reason=DEFAULT_DENY  -> BLOCKED

== Part 2: manager-role coordination-overhead model (modeled, not wall-clock) ==
  shared crew brief (re-sent per delegation) : 2000 tok
  per-delegation unique instruction          : 300 tok
  manager -> worker delegations              : 12
  naive prefill (brief re-sent each time)    : 27600 tok
  fak  prefill (brief prefilled once, reused): 5600 tok
  modeled prefill reduction                  : 4.93x

  This is deterministic token accounting (the manager-worker specialization of
  fak's fan-out geometry), NOT a measured wall-clock. The authoritative numbers
  live in BENCHMARK-AUTHORITY.md; the host-runnable deterministic ladder is
  `go run ./cmd/fanbench --grid canonical`. The live-model CrewAI-vs-native
  wall-clock comparison is the deferred bench-node run tracked by D-001 (#255).

summary: PASS
```

Exit code: `0`.

The Part 1 verdicts match the kernel directly — for example:

```text
$ go run ./cmd/fak preflight --policy examples/crewai-crew/policy.json --tool send_email --args '{}'
verdict=DENY reason=POLICY_BLOCK by=monitor

$ go run ./cmd/fak preflight --policy examples/crewai-crew/policy.json --tool read_dataset --args '{}'
verdict=ALLOW reason=NONE by=monitor

$ go run ./cmd/fak preflight --policy examples/crewai-crew/policy.json --tool transfer_funds --args '{}'
verdict=DENY reason=SECRET_EXFIL by=monitor
```
