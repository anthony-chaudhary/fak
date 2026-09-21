# TICKET-13445: Native child enforcement attestation

Status: open

Repository: `anthony-chaudhary/fak`

Public issue: [#13445](https://github.com/anthony-chaudhary/fak/issues/13445)

Blocked integration: [#13378](https://github.com/anthony-chaudhary/fak/issues/13378)

Priority: P1

Lane: `cmd`

## Contract

The real `fak chat --receipt` producer must report the enforcement state the child actually applied. The existing `fak ops run --harness native` consumer must qualify child-originated evidence rather than infer effective state from parent arguments.

The optional receipt object uses schema `fak.agent.native.enforcement.v1` and binds:

- effective guard posture;
- policy digest captured when the child loads the floor;
- canonical workspace digest;
- actual system, MCP, skills, and memory toggles.

Requested launch intent remains distinct from effective child evidence. Missing, unsupported, incomplete, or mismatched evidence stays fail-closed. Skills must report false when code tools are not armed, and memory must reflect the resolved memory option. Do not add another receipt transport, redesign policy, enable tools, or weaken the Ops launcher gate.

## Plan-before-build evidence

- Capability query: `go run ./cmd/fak-dev feature query "native child enforcement attestation" --json` returned no matching runtime capability; its results were unrelated documentation cards. This is an `ABSENT` capability result, not permission to invent a parallel mechanism.
- Boundary: Gate 3 places this native CLI/receipt producer in public `fak/cmd/fak`; no private import is required.
- Reuse: extend `nativeAgentReceipt` and the existing `cmdChat` → `runChatHeadless` → `newHeadlessAgentReceipt` path. Preserve existing helper call sites with an internal compatibility wrapper and pass one immutable evidence context from production.
- Exact blast radius: `cmd/fak/agent_native.go`, `cmd/fak/chat.go`, and independent acceptance coverage in `cmd/fak/chat_receipt_test.go`.

## Implementation plan

1. Add the versioned optional enforcement object to `nativeAgentReceipt`.
2. In `cmdChat`, capture effective posture and load-time policy digest after policy installation; canonicalize the workspace actually supplied to code tools; derive capability booleans only from successfully armed catalogs and the resolved memory option.
3. Thread that immutable context through the headless receipt writer, then verify the downstream #13378 consumer without changing its fail-closed rules.

## Existing RED witness

Integrated #13378 witness on 2026-09-20:

```text
go test ./cmd/fak -run '^TestOpsNative' -count=1 -v
```

Result: RED. Seven native-success cases fail with `child did not report a completed, qualified native turn` because current `fak.agent.native.v1` receipts contain no enforcement object. This is producer absence, not evidence that the consumer should trust parent argv.

## Acceptance

- `go test ./cmd/fak -run 'TestNativeAgentEnforcementReceipt' -count=1 -v`
- `go test ./cmd/fak -run '^TestOpsNative' -count=1 -v`
- `go test ./cmd/fak -run 'TestNativeAgentEnforcementReceipt|^TestOpsNative' -count=1 -v`
- `go vet ./cmd/fak`
- `git diff --name-only <base>...HEAD` names only `cmd/fak/agent_native.go`, `cmd/fak/chat.go`, `cmd/fak/chat_receipt_test.go`, and `docs/tickets/fak-ops/TICKET-13445-native-child-enforcement-attestation.md`.

Done means the real production `cmdChat` path emits supported child-originated evidence, the producer witness and downstream #13378 suite execute and pass, legacy receipt decoding remains valid, and an independent judge binds the result.
