---
title: "policy-capability names"
description: "This map positions the current policy-capability coverage backlog. Each entry names the exact repository symbol;"
---
# policy-capability names

This map positions the current `policy-capability` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

- **`issuepolicy`** — the exact `policy-capability` symbol `issuepolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`expertringevictpolicy`** — the exact `policy-capability` symbol `expertringevictpolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`applypolicyruntimelocked`** — the exact `policy-capability` symbol `applypolicyruntimelocked`; use this spelling for that operation rather than the undifferentiated family name.
- **`expertringpolicy`** — the exact `policy-capability` symbol `expertringpolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`guardupstreamposture`** — the exact `policy-capability` symbol `guardupstreamposture`; use this spelling for that operation rather than the undifferentiated family name.
- **`policyloadfailed`** — the exact `policy-capability` symbol `policyloadfailed`; use this spelling for that operation rather than the undifferentiated family name.
- **`policysnapshot`** — the exact `policy-capability` symbol `policysnapshot`; use this spelling for that operation rather than the undifferentiated family name.
- **`posturedrift`** — the exact `policy-capability` symbol `posturedrift`; use this spelling for that operation rather than the undifferentiated family name.
- **`preflightokverdict`** — the exact `policy-capability` symbol `preflightokverdict`; use this spelling for that operation rather than the undifferentiated family name.
- **`reasonunsupportedactivecachecapability`** — the exact `policy-capability` symbol `reasonunsupportedactivecachecapability`; use this spelling for that operation rather than the undifferentiated family name.
- **`selectexpertringevictpolicy`** — the exact `policy-capability` symbol `selectexpertringevictpolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`appendcapabilitygrant`** — the exact `policy-capability` symbol `appendcapabilitygrant`; use this spelling for that operation rather than the undifferentiated family name.
- **`bailpolicychecknofile`** — the exact `policy-capability` symbol `bailpolicychecknofile`; use this spelling for that operation rather than the undifferentiated family name.
- **`bailpolicyloadfailed`** — the exact `policy-capability` symbol `bailpolicyloadfailed`; use this spelling for that operation rather than the undifferentiated family name.
- **`capabilitygrant`** — the exact `policy-capability` symbol `capabilitygrant`; use this spelling for that operation rather than the undifferentiated family name.
- **`capabilitygrantrow`** — the exact `policy-capability` symbol `capabilitygrantrow`; use this spelling for that operation rather than the undifferentiated family name.
- **`cipreflightfailure`** — the exact `policy-capability` symbol `cipreflightfailure`; use this spelling for that operation rather than the undifferentiated family name.
- **`dispatchworkerpreflightrequest`** — the exact `policy-capability` symbol `dispatchworkerpreflightrequest`; use this spelling for that operation rather than the undifferentiated family name.
- **`dispatchworkerpreflightresult`** — the exact `policy-capability` symbol `dispatchworkerpreflightresult`; use this spelling for that operation rather than the undifferentiated family name.
- **`fakpolicyreloadallowwiden`** — the exact `policy-capability` symbol `fakpolicyreloadallowwiden`; use this spelling for that operation rather than the undifferentiated family name.
- **`indexpolicyfinding`** — the exact `policy-capability` symbol `indexpolicyfinding`; use this spelling for that operation rather than the undifferentiated family name.
- **`multimodalpolicy`** — the exact `policy-capability` symbol `multimodalpolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`postureargs`** — the exact `policy-capability` symbol `postureargs`; use this spelling for that operation rather than the undifferentiated family name.
- **`tunefirepolicy`** — the exact `policy-capability` symbol `tunefirepolicy`; use this spelling for that operation rather than the undifferentiated family name.
- **`usagepreflight`** — the exact `policy-capability` symbol `usagepreflight`; use this spelling for that operation rather than the undifferentiated family name.


### runtimecap fallback policy (pin/refuse or local CPU degrade)

runtimecap fallback policy selects whether an unavailable requested backend stays pinned and refuses or may degrade into a declared local CPU envelope; Probe records the selected posture in the runtime-capabilities receipt.

**Distinct from:** It is runtime capability fallback for backend execution, not the deployable policy manifest that governs tool capabilities and not the orchestration fast-profile response to an unavailable speed binding.


### FastIntent.FallbackPolicy (fast-profile fallback action)

FastIntent.FallbackPolicy is the portable orchestration request field that chooses degrade or refuse when the requested fast harness binding is unsupported during fast-plan resolution.

**Distinct from:** It governs orchestration response to an unavailable fast-profile mechanism; it is not runtimecap backend fallback, which chooses pin/refuse versus a declared local CPU execution envelope.


### policyRef (attestation policy digest)

policyRef is the attestation wire object that binds the exact capability-floor manifest path to its SHA-256 digest so a compliance proof can be rechecked against the same policy bytes.

**Distinct from:** It is an attestation reference to one policy artifact, not the policy manifest schema or the compiled in-memory Policy consulted by adjudication.
