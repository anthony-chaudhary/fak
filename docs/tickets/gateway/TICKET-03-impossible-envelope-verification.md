# Gateway impossible-envelope verification — #12655

Fresh behavioral verification passes for public issue
[#12655](https://github.com/anthony-chaudhary/fak/issues/12655).
This evidence-only ticket records execution against tested base
`2d9d99f29d097f75dfdd4e30e3a42d5dc41aa84a`; it changes no source or tests.

## Implementation and provenance

The existing implementation and named regression test were introduced by
`3891cd508c1175c61bb152d1674c3b5262915b18` (reference implementation).
At verification time, GitHub's comparison of that commit with public `main`
reported `ahead`, 225 commits ahead and zero behind, with that commit as the
merge base.

The coordinator's native provenance audit reported historical code/test
authorship as `UNATTESTED` (audit process exit 0). Historical authorship evidence
is unavailable; this ticket does not retroactively attest it. The fresh
behavioral results below are independently scoped to the tested base. They
establish observed behavior, not historical authorship.

## Executed witness

Executed on 2026-09-12 UTC from the tested checkout using the existing Windows
wrapper, which routed execution to WSL with Go 1.26.6 and `GOTOOLCHAIN=auto`.
`FAK_FAST=0` selected the checkout itself rather than the shared source mirror.
Exact PowerShell commands:

```powershell
$env:FAK_FAST='0'
& powershell -NoProfile -File .\test.ps1 -v ./internal/gateway -run TestAdmissionImpossibleRequestReturns400 -count=1
& powershell -NoProfile -File .\test.ps1 -v ./internal/gateway -run 'TestAdmission.*' -count=1
```

Both commands exited 0. The focused run reported:

```text
--- PASS: TestAdmissionImpossibleRequestReturns400 (0.02s)
PASS
ok github.com/anthony-chaudhary/fak/internal/gateway 0.089s
```

The admission acceptance run reported `PASS` for all selected tests and:

```text
PASS
ok github.com/anthony-chaudhary/fak/internal/gateway 0.122s
```

Captured logs are named `fak-12655-7122-focused.log` and
`fak-12655-7122-admission.log` in the task's temporary evidence store.
The coordinator independently read back those logs before requesting this
ticket. Host-specific paths are omitted from this public record.

## Acceptance established

- An idle controller refuses an individually impossible token envelope with
  `VerdictRefused`, rather than transient shedding.
- The existing test makes a real loopback TCP request through `srv.Handler()`
  to `/v1/chat/completions`. The observed HTTP status is 400; assertions verify
  `invalid_request_error` and `context_length_exceeded`, and exclude
  `rate_limit_error` and `scheduler_overloaded` from that response.
- The same test verifies that transient queue overload retains `VerdictShed`
  and maps to HTTP 429 with `scheduler_overloaded`.
- The issue's `TestAdmission.*` acceptance gate passes at the tested base.

The HTTP fixture uses the deterministic mock planner. The refusal is exercised
through the actual admission and HTTP path; this record makes no model
inference, streaming, hardware, or full-gateway-suite claim.

## Closure binding

This ticket supplies fresh behavioral evidence for an evidence-only closure
commit referencing the prior implementation and `Fixes #12655`. Historical
authorship remains unattested. The coordinator owns landing and issue closure;
neither is implied by creation of this ticket.
