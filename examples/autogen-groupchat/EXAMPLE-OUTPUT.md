# Captured output

`python examples/autogen-groupchat/groupchat_demo.py`, run from the repository root with
no arguments and no environment overrides (the dependency-free local-evaluator path):

```text
== Part 1: every group-chat tool call passes the capability floor ==
verdict source: local policy.json evaluator

  speaker hand-off tools:
  GroupChat   transfer_to_coder  verdict=ALLOW reason=NONE          -> run
  GroupChat   transfer_to_critic verdict=ALLOW reason=NONE          -> run
  agent turn tools:
  Planner     web_search         verdict=ALLOW reason=NONE          -> run
  Planner     list_dir           verdict=ALLOW reason=NONE          -> run
  Coder       read_file          verdict=ALLOW reason=NONE          -> run
  Coder       run_tests          verdict=ALLOW reason=NONE          -> run
  Critic      get_diff           verdict=ALLOW reason=NONE          -> run
  Coder       delete_repo        verdict=DENY  reason=POLICY_BLOCK  -> BLOCKED
  Critic      post_to_pastebin   verdict=DENY  reason=DEFAULT_DENY  -> BLOCKED

== Part 2: conversation-state-preservation model (modeled, not wall-clock) ==
  shared conversation base (system+roster+tools) : 1500 tok
  new tokens appended per turn                   : 250 tok
  group-chat turns                               : 12
  naive prefill (whole transcript re-sent/turn)  : 34500 tok
  fak  prefill (conversation prefix reused)      : 4500 tok
  modeled prefill reduction                      : 7.67x

  This is deterministic token accounting (the multi-turn-conversation
  specialization of fak's shared-prefix reuse), NOT a measured wall-clock. The
  authoritative numbers live in BENCHMARK-AUTHORITY.md; the host-runnable
  deterministic ladder is `go run ./cmd/fanbench --grid canonical`. The
  live-model AutoGen-vs-native wall-clock comparison is the deferred bench-node
  run tracked by D-001 (#255).

summary: PASS
```

Exit code: `0`.

The Part 1 verdicts match the kernel directly — including that an explicit deny outranks
the `transfer_`/`read_` prefixes, and that the fail-closed floor catches an exfil-shaped
tool nobody allow-listed:

```text
$ go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool delete_repo --args '{}'
verdict=DENY reason=POLICY_BLOCK by=monitor

$ go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool post_to_pastebin --args '{}'
verdict=DENY reason=DEFAULT_DENY by=monitor

$ go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool exfiltrate_secrets --args '{}'
verdict=DENY reason=SECRET_EXFIL by=monitor

$ go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool get_diff --args '{}'
verdict=ALLOW reason=NONE by=monitor

$ go run ./cmd/fak preflight --policy examples/autogen-groupchat/policy.json --tool transfer_to_coder --args '{}'
verdict=ALLOW reason=NONE by=monitor
```
