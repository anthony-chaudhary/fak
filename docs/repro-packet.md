---
title: "fak Repro Packet — Reproduce the Allow/Deny Boundary Offline"
description: "A no-credential, offline reproduction of fak's tool allow/deny policy: validate a policy manifest, deny a dangerous action, and allow a permitted action."
---

# Reproduce the offline allow/deny boundary

**Primary audience:** an evaluator deciding whether fak's tool-call boundary is worth deeper integration testing.

## Result and scope

The canonical offline proof demonstrates two current behaviors in one deterministic fixture:

| Check | Expected result | Scoped meaning |
|---|---|---|
| Dangerous `refund_payment` proposal | `DENY (POLICY_BLOCK)` | A reviewable manifest blocks the named action before tool execution. |
| Benign `search_kb` proposal | `ALLOW` | The same policy preserves its useful read/search path. |

This is a **current-generation, maintained deterministic witness**. It covers the checked-in customer-support fixture on any supported Go build host. It does not establish live-model behavior, detector recall, production readiness, external endorsement, or fleet-scale performance. Those claims require their own authorities and evidence.

**Default:** use the offline source-checkout route below. It needs no credential, live model, network service, or accelerator.

**Next action:** from a clean checkout with Go 1.26+, validate the manifest, then compare the two preflight results with the table above.

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
```

## Witness 1: Policy Manifest Validates

Command:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
```

Key output:

```text
OK  examples/customer-support-readonly-policy.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 4 tool(s)
allow (prefix)     : read_, get_, search_, list_, lookup_, find_
deny (explicit)    : 6 tool(s)
                     delete_account -> POLICY_BLOCK
                     export_customer_data -> SECRET_EXFIL
                     refund_payment -> POLICY_BLOCK
                     rotate_credentials -> POLICY_BLOCK
                     send_customer_email -> POLICY_BLOCK
                     transfer_funds -> POLICY_BLOCK
```

What this proves: the starter customer-support manifest parses, is fail-closed,
and its dangerous actions cite closed-vocabulary refusal reasons.

## Witness 2: Dangerous Action Denied

Command:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
```

Output:

```text
verdict=DENY reason=POLICY_BLOCK by=monitor
fak: loaded capability floor from examples/customer-support-readonly-policy.json
```

What this proves: a destructive support action is denied before any tool
execution. This is the smallest useful demo for a security lead: edit a manifest,
run one command, see a closed reason code.

## Witness 3: Benign Call Still Allowed

Command:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"
```

Output:

```text
verdict=ALLOW reason=NONE by=monitor
fak: loaded capability floor from examples/customer-support-readonly-policy.json
```

What this proves: the policy is not a blanket block. It preserves the useful
read/search path while denying dangerous writes.

## Fixture requests

The packet is an evaluation fixture rather than a production benchmark. To propose a scrubbed or synthetic boundary failure, use the [agent-tool boundary fixture issue form](https://github.com/anthony-chaudhary/fak/blob/main/.github/ISSUE_TEMPLATE/agent-tool-boundary-fixture.yml). For a framework or host adapter, use the [adapter fixture issue form](https://github.com/anthony-chaudhary/fak/blob/main/.github/ISSUE_TEMPLATE/framework-adapter-fixture.yml).

## Non-Claims

- This is an offline deterministic harness, not a live-model benchmark.
- The detector remains heuristic; this packet demonstrates the boundary behavior
  for this fixture, not broad prompt-injection recall.
- The production-readiness gates in
  [`docs/production-readiness.md`](production-benchmark-methodology.md) still matter.
- No vendor, government, or standards-body endorsement is implied.
