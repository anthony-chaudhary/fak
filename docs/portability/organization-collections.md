# Organization collections

`fak profile continuity org-selfcheck` is the runnable, deterministic two-user journey for issue #6602. It runs without a network service; the organization state is a signed JSON control-plane object suitable for a repository, artifact store, removable media, or an optional self-hosted server.

## Authority and precedence

Collections belong to an explicit organization namespace. Actors receive `owner`, `publisher`, `approver`, `operator`, or `member` roles; owner subsumes administrative roles. Discovery/install never grants activation: installation records content, while activation is a distinct authorization decision.

Policy precedence is exactly **corporate > team > project > personal**. The explanation lists every applicable policy in that order. Two applicable policies at the same scope that disagree fail closed with `POLICY_CONFLICT`; there is no last-writer-wins path. Policy revisions increase monotonically, so stale/replayed overlays are rejected.

Packages are publisher-signed, monotonically versioned, approval-signed, and pinned by collection/version/digest. Rollout starts at ring zero and promotion opens exactly the next declared ring. Ordinary installs reject downgrades; an operator rollback is permitted only to an existing, signed, approved, policy-valid, non-revoked version and leaves activation off.

## Offline reconciliation and revocation

The serialized organization state is authoritative for ownership, policy, versions, rollout, deprecation, retention, and revocation. An offline device retains its local installation records. On reconnect, `ReconcileDevice` applies authoritative state and deterministically changes every active revoked install to:

- `active=false`
- `quarantined=true`
- remediation: remove the revoked package, install an approved non-revoked version, then activate explicitly

A revoked version cannot be newly installed, activated, promoted, or used as a rollback target. Departing users are removed by an owner from the authoritative actor map; reconciliation therefore removes their organization authority without taking ownership of their personal continuity home.

Cross-organization import appends immutable source provenance, re-signs into the receiving namespace, and then evaluates the receiving organization's policy and approvals.

## Audit and live proof

Every decision appends a hash-linked receipt containing identity, package reference, policy ID, decision code, actor, action, previous receipt, and receipt ID. Payloads, keys, revocation prose, and other secrets are excluded. `VerifyAudit` checks every digest and link.

```console
fak profile continuity org-selfcheck
fak profile continuity org-selfcheck --json
```

The journey creates five deterministic actors, publishes and approves a team collection, installs it on canary, rejects early team install, promotes, installs on team, revokes while a device is offline, and reads back quarantine after reconnect. Human and JSON modes expose the same decisions and audit-chain result.
