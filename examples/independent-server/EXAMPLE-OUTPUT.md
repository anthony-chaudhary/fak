# Independent server receipt handoff captured output

Command, run from the repository root on Linux with Go 1.26.6:

```console
go run ./examples/independent-server -selfcheck
```

The complete receipt includes build- and run-specific digests, process identities,
and elapsed milliseconds. The JSON below is the stable projection copied from an
actual passing run; only those variable evidence fields are omitted.

```json
{
  "schema": "fak.independent-server-selfcheck/v1",
  "verdict": "pass",
  "boundary": {
    "clean_product_directories": 2,
    "server_root": "server-product",
    "harness_root": "harness-product",
    "distinct_roots": true,
    "receipt_only_handoff": true,
    "crossed_paths": [
      "server-receipt.json"
    ],
    "shared_mutable_paths": []
  },
  "receipt": {
    "schema": "fak.server-product/v1",
    "generation": 1,
    "artifact_name": "fixture.gguf",
    "adapter_name": "llama-server",
    "adapter_version": "llama-server fixture version 1",
    "readiness_probe": "fak.server-adapter-probe.v1",
    "capabilities": [
      "chat.completions",
      "health",
      "models.list"
    ],
    "receipt_unchanged": true
  },
  "harness": {
    "schema": "fak.harness-server-resolution/v1",
    "generation": 1,
    "model_alias": "local-code",
    "protocol_family": "openai-http",
    "protocol_revision": "2026-02",
    "chat_calls": 1,
    "chat_response": "HANDOFF_OK",
    "lifecycle_calls": 0
  },
  "teardown": {
    "state": "stopped",
    "owned": true,
    "instance_id_match": true,
    "generation_match": true,
    "process_id_match": true,
    "owned_teardowns": 1,
    "harness_lifecycle_calls": 0
  },
  "readiness_probes": 3,
  "external_network_requests": 0,
  "phase_names": [
    "init",
    "up",
    "harness",
    "down"
  ]
}
```
