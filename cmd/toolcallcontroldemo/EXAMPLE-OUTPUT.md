# Tool-call control demo captured output

Command, run from the repository root:

```console
go run ./cmd/toolcallcontroldemo -selfcheck -pretty
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{
  "schema": "fak-tool-call-control-demo/1",
  "instruction": "Before any tool call, name the missing evidence and the decision that could change. Reuse fresh results; do not repeat an identical call without a state change. Batch independent reads. If existing evidence is sufficient, answer or stop instead of calling a tool. This is a long-context turn: each continuation replays substantial context, so prefer reuse and one batched read.",
  "decisions": [
    {
      "id": "repeat",
      "action": "reuse",
      "reason": "exact_fresh_result",
      "fingerprint": "ef878c1f6ef9d1e3",
      "reuse_ref": "turn-previous",
      "replay_units_saved": 128000,
      "replay_squared_saved": "16384000000"
    },
    {
      "id": "search-a",
      "action": "allow",
      "reason": "batch_leader",
      "fingerprint": "59fafef7e225d748",
      "batch_key": "repo-inspection",
      "replay_units_saved": 0,
      "replay_squared_saved": ""
    },
    {
      "id": "search-b",
      "action": "batch",
      "reason": "merged_into_batch_leader",
      "fingerprint": "c8f92a1fc63be22c",
      "batch_key": "repo-inspection",
      "replay_units_saved": 128000,
      "replay_squared_saved": "16384000000"
    },
    {
      "id": "browse",
      "action": "defer",
      "reason": "no_actionable_evidence_gap",
      "fingerprint": "716fb4160d553da7",
      "replay_units_saved": 128000,
      "replay_squared_saved": "16384000000"
    },
    {
      "id": "write",
      "action": "allow",
      "reason": "novel_or_required",
      "fingerprint": "b4b44fe6b14fdd2a",
      "replay_units_saved": 0,
      "replay_squared_saved": ""
    }
  ],
  "ablation": [
    {
      "name": "control",
      "calls_executed": 5,
      "unneeded_calls_avoided": 0,
      "needed_calls_suppressed": 0,
      "replay_units_saved": 0,
      "replay_squared_saved": "0"
    },
    {
      "name": "prefilter",
      "calls_executed": 2,
      "unneeded_calls_avoided": 2,
      "needed_calls_suppressed": 0,
      "replay_units_saved": 256000,
      "replay_squared_saved": "32768000000"
    }
  ],
  "self_check": "PASS"
}
```
<!-- END SELFCHECK OUTPUT -->
