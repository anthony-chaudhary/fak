# Host approval boundary

**Issue:** [#14030](https://github.com/anthony-chaudhary/fak/issues/14030)  
**Observed:** 2026-10-26 (America/Los_Angeles)  
**External source state:** shipped Windows API; cited Microsoft Learn pages updated 2026-06-18  
**Disposition:** source candidate for preparing and verifying a plan; **not an approval mechanism**.

## Decision

The repository has no authenticated, human-origin approval callback from Codex to a trusted
host component. Windows exposes the WebAuthn/Windows Hello primitives needed by a future
broker, but their presence does not make a request authorized. The candidate
`fak-host-approval` therefore has only `prepare`, `verify`, and `status`: it has no
`approve`/`apply`, executes no plan, and writes no Codex or guard configuration.

An agent helper, a Boolean, environment-marker stripping, HTTP headers or bearer tokens, a
caller-signed manifest, or a model-clickable dialog cannot cross this boundary. Each remains
under caller or model control, or proves only possession of caller-provided material.

## What the non-authorizing request freezes

`fak.host-approval.request.v1` binds one proposed two-step plan to:

- SHA-256 identities for the installer, policy, reviewed remote-maintenance context, and every
  native authority adapter, including its fixed arguments;
- the exact Codex server target and the target config path, existence state, and SHA-256 snapshot;
- every reviewed remote profile: target, pinned host key, operation, payload digest, holder,
  exact paths/path-prefixes, units, lease nodes, profile expiry, and authority;
- the task and session identities, preparation time, and an effective expiry no later than
  15 minutes or the earliest profile expiry; and
- a random 256-bit request nonce plus the exact executable/argument plan.

`verify` re-reads these inputs, validates policy and executable-backed authority loading,
checks session and expiry, and compares the exact derived plan. It reports
`authorized:false` and `broker_required:true`. It neither consumes nor journals the nonce.
The nonce is generated on the caller side; caller authorship is freshness input, not authority.
The printed `request_sha256` hashes canonical JSON from the decoded request, not pretty-printed
file bytes. A caller who supplies both the request and its digest establishes no trusted review.

The existing native `fak codex mcp install` path already has rollback. Existing guard
remote-maintenance already loads a strictly pinned authority executable. This helper only
models and rechecks a plan that would invoke those established paths. The scope root retains
ownership of actual remote maintenance and Codex configuration refresh.

## Windows observation and its limit

On 2026-10-26, the passive local probe reported WebAuthn API version **9** and a user-verifying
platform authenticator available. It called only
`WebAuthNGetApiVersionNumber` and
`WebAuthNIsUserVerifyingPlatformAuthenticatorAvailable`; it enrolled no credential, requested
no assertion, and displayed no prompt. This is capability discovery, not hardware evidence and
not proof of an enrolled approval key, a protected broker, or a live human approval.

A separate recovery-coordinator probe reported
`UserConsentVerifier.CheckAvailabilityAsync() == DeviceNotPresent` (enum 1). These are distinct
API results; neither establishes a usable authenticated approval channel. No assertion or
actual human verification has been qualified on this host.

Microsoft documents that:

- the [Windows WebAuthn API](https://learn.microsoft.com/en-us/windows/win32/webauthn/-webauthn-portal)
  connects Win32 applications to Windows Hello or external FIDO2 authenticators;
- [API version discovery](https://learn.microsoft.com/en-us/windows/win32/api/webauthn/nf-webauthn-webauthngetapiversionnumber)
  reports platform capability, while
  [platform-authenticator availability](https://learn.microsoft.com/en-us/windows/win32/api/webauthn/nf-webauthn-webauthnisuserverifyingplatformauthenticatoravailable)
  reports whether such an authenticator is available;
- [`WebAuthNAuthenticatorGetAssertion`](https://learn.microsoft.com/en-us/windows/win32/api/webauthn/nf-webauthn-webauthnauthenticatorgetassertion)
  takes the relying-party ID, client data, and assertion options and returns a signed assertion;
  its [`WEBAUTHN_AUTHENTICATOR_GET_ASSERTION_OPTIONS`](https://learn.microsoft.com/en-us/windows/win32/api/webauthn/ns-webauthn-webauthn_authenticator_get_assertion_options)
  expose the user-verification requirement; and
- the returned [`WEBAUTHN_ASSERTION`](https://learn.microsoft.com/en-us/windows/win32/api/webauthn/ns-webauthn-webauthn_assertion)
  contains authenticator data, credential identity, and signature for server-side verification.

The API, options, and structure pages used here were last updated by Microsoft on 2026-06-18;
the portal states Windows 10 version 1903+ and Windows 11. Refresh this observation when the
Windows WebAuthn API version or Microsoft contract changes.

This is **INSPIRE-ONLY** use of behavioral primary documentation: no Microsoft code, tests,
comments, or assets were copied. Microsoft Learn is product/API documentation, not the
repository-shaped upstream for this work. Source history, tests, issues, pull requests,
roadmaps, and provenance were not reviewed; those are recorded omissions, not evidence that
no other mechanism exists.

## Required broker boundary

Authorization requires a separately provisioned, protected broker/service with all of:

1. an independent service identity, restrictive ACL, and enforced code-integrity policy;
2. a trusted manifest-review UI that renders the entire frozen scope rather than caller prose;
3. a pinned registry of approval credential IDs and public keys established out of band;
4. a broker-generated challenge bound to the request digest, task/session, and expiry;
5. a WebAuthn assertion requiring user presence **and** user verification, with strict checks of
   RP ID hash, expected origin, `webauthn.get` type, challenge, credential ID, authenticator
   flags/counter policy, and signature; and
6. a durable, atomic, one-use nonce journal owned by the broker and separate from execution.

Only that broker may emit an authorization receipt. Execution must remain a later component
that accepts one exact, unexpired, unused receipt and cannot mint, rewrite, or replay it.

## Next follow-on acceptance

Another bounded architecture candidate is typed native-operation admission: a closed broker
handler accepts one validated plan digest and a fixed profile, declares its exact footprint,
and passes existing policy and lease admission before execution. Read-only control diagnostics,
local bridge installation, and remote application are separate operations. A new handler needs
its own admission witness; a blanket command-family allow, relabeled shell call, or caller-made
authority cannot inherit approval from this request or erase an earlier denial.

Implement the broker as a preprovisioned Windows service, then capture one end-to-end witness:
review the frozen manifest in trusted UI; approve with a registry-pinned credential using a
broker-owned challenge with required UP+UV; reject wrong RP/origin/type/signature, mutated scope,
expired request, wrong task/session, and replay; atomically consume the nonce before handing the
receipt to a separate executor. Until that witness exists, status remains **broker absent** and
no live remote-maintenance or config-refresh claim is permitted.

## Product framing and P1-P4

**For:** an operator allowing a narrowly reviewed host change. **Problem:** Codex cannot return
authenticated human-origin consent to repository code. **Today:** the operator uses the existing
independent-shell install/rollback and guarded remote-maintenance paths. **Better because:** the
candidate makes the exact proposed scope inspectable and mutation-detecting without pretending
to authorize it. **Witness:** prepare/verify/status tests plus the future broker acceptance above.

- **P1 — preserved:** a compact request captures review context without adding agent history.
- **P2 — preserved:** no convenience or security-gain claim is made before broker evidence.
- **P3 — advanced:** versioned request and receipt seams fail closed and leave execution separate.
- **P4 — partial:** plan status and verification are observable; live approval and recovery await
  the broker witness.

## Candidate witness

```powershell
.\test.ps1 -count=1 ./internal/hostapproval ./cmd/fak-host-approval
```

This command is pending the parent integrator's independent run.
