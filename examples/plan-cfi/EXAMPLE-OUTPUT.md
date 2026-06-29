# Example output

The witness for plan-CFI is the `internal/plancfi` test suite: it declares the airline-booking
plan and drives a conforming call (→ `Defer`) and a deviating call (→ `RequireApproval`) through
the real `Adjudicator`. Reproduce from the repo root:

```bash
go test -run 'TestConformingCallDefers|TestDeviationEscalates|TestStrictModeDenies|TestNoPlanDefers|TestSequenceMode|TestSessionIsolation' -v ./internal/plancfi
```

Expected output (PASS):

```
=== RUN   TestNoPlanDefers
--- PASS: TestNoPlanDefers (0.00s)
=== RUN   TestConformingCallDefers
--- PASS: TestConformingCallDefers (0.00s)
=== RUN   TestDeviationEscalates
--- PASS: TestDeviationEscalates (0.00s)
=== RUN   TestStrictModeDenies
--- PASS: TestStrictModeDenies (0.00s)
=== RUN   TestSequenceMode
--- PASS: TestSequenceMode (0.00s)
=== RUN   TestSessionIsolation
--- PASS: TestSessionIsolation (0.00s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/plancfi	0.0XXs
```

## What each line proves

```
plan declared for trace "t":  {get_user_details, search_flights, read_refund_policy, book_reservation}  (AllowedSet)

  on-plan  call  get_user_details   →  Defer            (TestConformingCallDefers: CFI has no objection)
  on-plan  call  search_flights     →  Defer
  on-plan  call  read_refund_policy  →  Defer
  on-plan  call  book_reservation   →  Defer

  off-plan call  send_email         →  RequireApproval  (TestDeviationEscalates: the headline — escalate, not deny)
                                        Reason = TRUST_VIOLATION
                                        Meta   = {plancfi: "deviation", tool: "send_email"}
                                        Payload claim = `call "send_email" deviates from the approved plan`

  no plan declared, call "anything"  →  Defer            (TestNoPlanDefers: CFI is opt-in per trace)
  strict mode, off-plan delete_everything → Deny          (TestStrictModeDenies: escalate-vs-deny is one knob)
```

The load-bearing result: the off-plan `send_email` is trapped purely because the tool is not on
the trace's approved plan **shape** — no detector read the call, no argument was scanned. The
verdict is `RequireApproval` (fold rank 50), so the kernel HOLDS the call for human approval
rather than terminating the trace. That hold is proven fail-closed end-to-end in
`internal/kernel/kernel_plancfi_test.go` (`TestRegisteredEscalationVerdictIsHeldFailClosed`):
the escalated call never reaches dispatch.

---

**Provenance of this transcript.** The verdicts, `Reason`, `Meta`, and `Payload` claim above are
read directly from the assertions in `internal/plancfi/plancfi_test.go` (each test fails unless
the real adjudicator returns exactly these values), so they reflect the shipped behavior. The
`go test` PASS block is the expected form of that command's output; it was **not** captured on
this author's box (the local `go test` runner is OS-blocked here and timed out). To capture a
live run, execute the reproduce command above on a host where `go test` runs.
