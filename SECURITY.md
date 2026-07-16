# Netra Fused Agent Kernel (`fak`) — Security Policy

**Audience:** security evaluators and operators checking what the current `fak` capability boundary guarantees, how policy configures it, and what evidence demonstrates it.

`fak` puts policy adjudication and result quarantine on the tool-call path. The current security floor is **structural enforcement**, not successful prompt-injection detection: an active policy limits which tools can run, managed-agent quarantine holds classified results out of model context, and gateway/adjudication errors fail closed.

## Current capability floor

| Boundary | Current guarantee | Configuration | Evidence |
|---|---|---|---|
| **Capability floor** | A tool absent from the active policy's allow-list is denied before execution. | Select and review a policy manifest; [`examples/customer-support-readonly-policy.json`](examples/customer-support-readonly-policy.json) is the read-only evaluation example. | The deterministic `fak preflight` check below exercises one denied and one allowed tool without a model. |
| **Containment / quarantine** | In `fak agent`, a result classified as secret-shaped, prompt injection, or poison is held out of model context and replaced by a stub pointer; page-in requires an explicit witness clear. | This shipped gate is automatic on the managed-agent path; its current trigger and page-in contract are itemized in [`CLAIMS.md`](CLAIMS.md). | Run the offline managed-agent proof in the [reproduction packet](docs/repro-packet.md); do not generalize it to unmanaged clients or every backend. |
| **Gateway / adjudication** | Tool calls routed through `fak agent` or the documented `fak` gateway cross the in-process gate; crash, timeout, and malformed-input failures deny rather than execute. | Start from [`GETTING-STARTED.md`](GETTING-STARTED.md), select its production row, and keep the client on the documented path for that backend with the intended policy. An integration that bypasses `fak` is outside this guarantee. | The [reproduction packet](docs/repro-packet.md) and repository tests are the public proof routes; a call that bypasses an advertised managed path is in scope below. |

**Default evaluation route:** check the capability floor first. It is model-independent and separates policy enforcement from detector quality.

**Next action — run this check:**

```bash
fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"  # DENY (POLICY_BLOCK)
fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb --args "{}"       # ALLOW
```

Pass means the first call reports `DENY (POLICY_BLOCK)` and the second reports `ALLOW`. This proves the selected manifest's structural deny/allow path; it does not prove live-model quality, every integration, or every policy.

**Route context:**

- **Mode:** the check above is deterministic preflight mode with no model, key, or GPU. Containment evidence is scoped to `fak agent --offline`; production coverage is the documented `fak agent` or gateway path selected from [`GETTING-STARTED.md`](GETTING-STARTED.md), not an arbitrary integration or unmanaged client.
- **Generation:** this page documents the current `gen/now` floor. Future security work is not a current guarantee unless [`CLAIMS.md`](CLAIMS.md) marks it `[SHIPPED]`.
- **Lifecycle:** `fak` is pre-1.0 with a rolling latest release; the supported line is defined under [Supported versions](#supported-versions).
- **Support boundary:** capability-floor, containment, gateway/adjudication, and policy/signature bypasses are security reports. Detector evasion, permissive policy authoring, and model quality are scoped separately below.

## Reporting a vulnerability

**Please report privately — do not open a public issue for a security-sensitive bug.**

1. **Preferred:** use GitHub's private vulnerability reporting — on this repository, go
   to **Security ▸ Advisories ▸ Report a vulnerability**. This opens a private channel
   visible only to the maintainers.
2. If you cannot use that, contact the maintainers (Netra Systems) privately through
   the contact on the project's GitHub organization.

Please include: what boundary you reached (capability floor, containment/quarantine,
or the gateway), a minimal reproduction, the model/config used, and the impact.

We aim to acknowledge a report within a few business days and to agree on a disclosure
timeline with you. We support coordinated disclosure and will credit reporters who
want credit.

## What is in scope

The floor `fak` actually defends — these *are* security bugs:

- **Capability-floor bypass.** A way to make `fak` execute a tool that the active
  policy does **not** allow-list (the "lever was never wired up" guarantee fails).
- **Containment bypass.** A way to get a quarantined / untrusted tool result admitted
  into the model's context or KV cache when policy said it must be held out — including
  any way to make a removed span fail to be bit-for-bit evicted.
- **Gateway / adjudication bypass.** A way to route a tool call around the in-process
  adjudication boundary, or to make the gate **fail open** (run the call anyway) on
  crash, timeout, or malformed input. The gate is designed to **fail closed**.
- **Policy or signature confusion** that causes a deny to be read as an allow.

## What is explicitly **out** of scope

By design, and stated plainly in the README and `fak/CLAIMS.md`:

- **Evading the injection *detector*.** The heuristic that *flags* suspicious tool
  results is **≈100% evadable by design** — it is a helpful bonus, never the floor. A
  prompt that the detector doesn't flag is **not** a vulnerability, because the detector
  is not what contains the result; the quarantine + capability floor are. (A way to
  defeat the *containment* or the *floor* — see "in scope" above — absolutely is.)
- Findings that require the operator to have already mis-authored a permissive policy
  (e.g. allow-listing a destructive tool) — that's policy authoring, not a gate bypass.
  Reports that improve the *default* floor or the policy linter are still welcome as
  normal issues.
- Capability/quality of the underlying model (hallucination, refusal, etc.).

## Supported versions

`fak` is pre-1.0 and ships a rolling release line; security fixes land on the latest
release (see [`VERSION`](VERSION) and the [releases][rel]). Please verify against the
latest release before reporting.

[rel]: https://github.com/anthony-chaudhary/fak/releases/latest
