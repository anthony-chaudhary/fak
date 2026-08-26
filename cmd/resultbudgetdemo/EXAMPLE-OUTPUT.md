# Tool-result budget demo captured output

Command, run from the repository root:

```console
go run ./cmd/resultbudgetdemo -selfcheck -pretty
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{
  "schema": "fak-result-budget-demo/1",
  "verdict": "PASS",
  "tool_calls": 4,
  "cases": [
    {
      "name": "enforce",
      "requested_items": 500,
      "tool_observed_items": 10,
      "receipt": {
        "decision": "clamp",
        "reason": "requested_items_above_maximum",
        "tool": "github.search_issues",
        "original_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "effective_args_sha256": "cdddf89990481a08f9db069b7f6bdba6ac5f2cce47351c090daff23e74a4c53b",
        "changes": [
          {
            "path": "/per_page",
            "from": 500,
            "to": 10,
            "dimension": "items"
          }
        ],
        "policy": {
          "name": "thimble/default",
          "version": "1.0.0",
          "sha256": "75fdf724ca1a3ee375fc2df1202beebba2fed96def8dedb351e8d6477839495f",
          "mode": "enforce"
        },
        "model_round_trips": 0,
        "actual": {
          "items": 10
        },
        "continuation": {
          "kind": "rerun",
          "available": true
        }
      }
    },
    {
      "name": "observe",
      "requested_items": 500,
      "tool_observed_items": 500,
      "receipt": {
        "decision": "observe",
        "reason": "clamp_proposed_above_maximum",
        "tool": "github.search_issues",
        "original_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "effective_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "proposed_changes": [
          {
            "path": "/per_page",
            "from": 500,
            "to": 10,
            "dimension": "items"
          }
        ],
        "policy": {
          "name": "thimble/default",
          "version": "1.0.0",
          "sha256": "528216b38cbc491e462ab3ed9e7ae0a2cbb5e215a74a15bdf35cde16054f60b4",
          "mode": "observe"
        },
        "model_round_trips": 0,
        "actual": {
          "items": 500
        }
      }
    },
    {
      "name": "exhaustive-intent",
      "requested_items": 500,
      "tool_observed_items": 500,
      "receipt": {
        "decision": "pass",
        "reason": "structured_exhaustive_intent",
        "tool": "github.search_issues",
        "original_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "effective_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "policy": {
          "name": "thimble/default",
          "version": "1.0.0",
          "sha256": "75fdf724ca1a3ee375fc2df1202beebba2fed96def8dedb351e8d6477839495f",
          "mode": "enforce"
        },
        "model_round_trips": 0,
        "actual": {
          "items": 500
        }
      }
    },
    {
      "name": "unknown-tool",
      "requested_items": 500,
      "tool_observed_items": 500,
      "receipt": {
        "decision": "pass",
        "reason": "unknown_tool_contract",
        "tool": "github.unknown_search",
        "original_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "effective_args_sha256": "24933d70a0114a2483bcdc573837c5bbbfac9933fe30a718e9dfe4cb8acc667f",
        "policy": {
          "name": "thimble/default",
          "version": "1.0.0",
          "sha256": "75fdf724ca1a3ee375fc2df1202beebba2fed96def8dedb351e8d6477839495f",
          "mode": "enforce"
        },
        "model_round_trips": 0,
        "actual": {
          "items": 500
        }
      }
    }
  ],
  "self_check": "PASS"
}
```
<!-- END SELFCHECK OUTPUT -->
