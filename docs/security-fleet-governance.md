# Security fleet governance with fak

Security teams can deploy one **capability floor** across agent entry points while leaving engineers room to choose a narrower operating profile. The useful boundary is precise: fak can enforce and attest the policy supplied to each local `fak manage`/`fak guard` process; an enterprise must supply the fleet distribution, inventory, update, and admission systems around those processes.

This page separates shipped controls from deployable compositions and proposals. It does not claim a hosted control plane or remote endpoint enforcement that fak does not ship.

## 1. Controls fak ships now

| Need | Shipped primitive | Evidence boundary |
|---|---|---|
| Central capability floor | `fak policy --check FILE`; launch with `fak manage --policy FILE -- <agent>` | The process enforces the manifest it actually loaded. |
| Signed central policy | `fak policy --check ENVELOPE --org-key KEY --org-issuer ISSUER --org-seen-version N` | Verifies a `fak-org-policy/v1` Ed25519 envelope and refuses policy-version rollback below `N`. It does not distribute the envelope. |
| Deterministic read-back | `fak attest --policy FILE --json`; `fak guard policy explain`; `fak guard policy diff --policy FILE` | Proves policy behavior locally and exposes effective amendment posture/drift. |
| Safe local rollout/rollback | `fak policy land-rule --policy FILE --candidate FILE` (dry run), then `--land`; `--rollback` restores its recorded preimage | Mutates one host's policy file and can POST its configured reload URL. Fleet fan-out is external. |
| Narrowing by argument | `arg_rules` (`allow_glob`, `deny_regex`, `max_bytes`, `cli_read_only`) | Engineers can narrow approved tools/resources without adding tool authority. |
| Bounded launch exception | `fak manage --allow-tool TOOL ...` | An exact launch-time grant; hard-danger and self-modification checks still apply. Treat as an operator-controlled exception, not an engineer-owned bypass. |
| Structured escalation | `fak complain ... --from-journal` | Produces a deduplicated, witnessed appeal from a real DENY/QUARANTINE journal row; `--live` performs the GitHub write. |
| Version identity | `fak version` | Reports binary version, build provenance when present, Go version, and platform. It does not force an update. |

The example floor at [`../examples/security-fleet-governance/central-floor.json`](../examples/security-fleet-governance/central-floor.json) permits only three named tools, denies three sensitive actions, marks two internal readers as trusted local sources, and limits their `uri` arguments to canonical `corp://...` namespaces. The URI scheme is an enterprise convention enforced here by existing `allow_glob`; fak does not resolve or own an internal resource catalog.

Run the deterministic demonstration (no API key, model, network, or running gateway):

```text
go run ./examples/security-fleet-governance
```

It validates the manifest, attests the derived probes, checks canonical-resource allow/deny cases through `fak preflight`, previews an engineer narrowing, lands it into a temporary copy, proves the narrower result, rolls back, and proves the original result again.

## 2. Enterprise compositions possible now

### Central floor and named architecture levels

Keep one reviewed central floor, then publish **named, narrowing-only profiles** as data owned by platform/security. For example:

| Profile | Intended freedom | Composition |
|---|---|---|
| `L0-observe` | Read approved internal references only | Central floor with the smallest read set. |
| `L1-team` | Team-scoped reads and bounded ticket creation | Central floor plus a reviewed ArgRules-only narrowing such as `identity-narrowing.json`. |
| `L2-operator` | Explicit operational tools | A separately reviewed central manifest; never derive it by an engineer-authored widen. |

“Architecture level” is an enterprise naming convention, not a shipped fak schema. Store profile name, policy digest, issuer, policy version, and minimum fak version in your deployment inventory. Security owns which profiles exist; engineers may select among profiles approved for their identity/workload and may submit further ArgRules-only narrowing. New authority requires review.

A useful admission rule is:

1. verify the signed envelope and anti-rollback counter with `fak policy --check ... --org-seen-version N`;
2. validate any team narrowing with `fak policy land-rule --policy FLOOR --candidate NARROWING` (dry-run first);
3. reject `fak guard policy diff --policy EFFECTIVE` when it reports a widen;
4. launch only through the managed wrapper with the verified effective policy;
5. retain `fak attest --json`, policy digest, fak version output, and decision-journal location as the workload receipt.

### Fleet rollout, read-back, and rollback

Use the company's existing endpoint manager, device-management system, CI runner image, or workload scheduler as the transport. A safe canary sequence is:

1. **Stage:** distribute a content-addressed policy/envelope and the expected digest without activating it.
2. **Verify:** run `fak policy --check` and `fak attest --policy ... --json` on every target.
3. **Canary:** activate on a bounded cohort. `fak policy land-rule` is suitable for ArgRules-only local changes; whole-manifest replacement remains the deployer's responsibility.
4. **Read back:** collect the file digest, signed-envelope issuer/version, `fak version`, attestation, and `fak guard policy diff` result from the target itself. Distribution success is not read-back.
5. **Expand:** promote only when the target receipts match the intended cohort.
6. **Rollback:** for a landed ArgRules change, call `fak policy land-rule --policy FILE --rollback`; for a whole manifest or binary, restore the enterprise package/image preimage. Re-run read-back after rollback.

`--reload-url` is a local HTTP POST hook for a running gateway. It is not authenticated fleet transport and should not be exposed as one.

### Minimum-version and forced-update posture

fak reports version/build provenance but does not currently ship a fleet-wide forced updater or a minimum-binary-version policy field. Implement that control in the launcher or scheduler:

- pin an approved artifact digest or signed package;
- compare `fak version` with the cohort's declared minimum before launch;
- fail closed or quarantine workloads below the minimum;
- roll forward through canaries, retaining the prior artifact for rollback;
- read back the executable digest and version from each target.

This preserves engineer flexibility above the binary floor: engineers choose an approved profile and local agent workflow, but cannot start an unapproved fak build through the managed entry point.

### Canonical internal resource references

Expose internal systems through stable logical names such as `corp://security/runbooks/incident` rather than environment-specific URLs. Enforce the namespace at the fak boundary with an `arg_rules[].allow_glob`, as the demo does. Resolve the logical name in an enterprise-owned tool adapter that authenticates the caller, maps to the current backend, and logs the canonical name plus resolved target.

The policy proves only that the tool argument fits the approved namespace. The adapter/catalog must prove identity, integrity, freshness, and backend mapping.

### Structured exceptions without bypasses

Engineers should receive fast, evidence-bearing feedback instead of blanket exemptions:

```text
fak complain --summary "identity runbook reference refused" --reason POLICY_BLOCK --tool read_corp_kb --from-journal --args-digest sha256:... --rationale "profile L1-team should admit corp://security/identity/runbooks/..."
```

Run without `--live` to preview the deduplicated issue plan; use `--live` only in an authorized filing workflow. Security can resolve the appeal by correcting an over-broad rule, approving a new named profile, or declining it with the journal witness intact. Engineer-authored narrowing can proceed without escalation; widening remains review-gated.

## 3. Proposed follow-ons (not shipped)

1. **Fleet policy receipt schema:** one standard row joining target identity, binary digest/version, profile, policy digest, signed issuer/version, attestation digest, and rollback preimage.
2. **Minimum-version launch gate:** a first-class manifest/launcher constraint with signed update metadata, expiry, canaries, and break-glass audit.
3. **Canonical resource registry binding:** signed aliases with freshness and resolver identity, rather than relying only on `allow_glob` plus an external adapter.
4. **Named profile compiler:** prove a profile is narrowing-only relative to the central floor and emit the effective policy plus reviewable diff.
5. **Fleet rollout controller:** authenticated distribution, target read-back, cohort health, and one-command rollback built on the local primitives above.
6. **Complaint service-level policy:** typed severity/ownership, bounded temporary approvals, expiry, and automatic re-attestation after resolution.

Until these exist, describe them as enterprise composition or roadmap—not as remote fak enforcement.
