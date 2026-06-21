# Security Policy

`fak` is a security tool: it puts a permission gate and a result-quarantine on the
same call path as every tool call, so an agent's effects pass *through* a kernel the
model doesn't control. We take reports against that boundary seriously.

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
