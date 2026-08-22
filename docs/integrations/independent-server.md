---
title: "Independent server receipt handoff"
description: "Run the clean-directory server-to-harness selfcheck that keeps lifecycle ownership outside the harness."
---

# Independent server receipt handoff

FAK's `server` product owns initialization, readiness, and teardown. A harness imports the resulting immutable `server-receipt.json`; it does not start, stop, repair, or mutate the server.

Run the complete local witness from the repository root:

```sh
go run ./examples/independent-server -selfcheck
```

The fixture implements the required loopback `llama-server` protocol without a model download or external network. The runner creates clean sibling server and harness directories, crosses only the receipt path, performs a separate harness chat call, stops the receipt-owned process, and rereads the receipt, binding, state, and event log after both products exit. A passing JSON result includes artifact/readiness digests, four elapsed phases, one harness chat, one owned teardown, and zero harness lifecycle events.

For a real local server, use the same ownership sequence with an installed `llama-server` and a digest-pinned GGUF file:

```text
fak server init --dir <server-dir> --name local-code --model <model.gguf> --sha256 <sha256> --executable <llama-server> --json
fak server up --dir <server-dir> --json
fak harness init --dir <harness-dir> --server-receipt <server-dir>/server-receipt.json --server-model local-code --server-protocol openai-http --server-protocol-revision 2026-02 --server-capabilities chat.completions,models.list --server-min-generation 1 --json
fak server down --dir <server-dir> --json
```

The receipt is identity and compatibility evidence, not credential distribution. Keep authentication material in the credential reference named by the receipt rather than in the receipt itself.
