# Example output

Captured from:

```bash
FAK_BIN=/tmp/fak-playground bash examples/playground/play.sh | cat
```

```text
======================================================================
  fak playground — treat the tool call like a syscall
======================================================================
An agent does not get to touch the world directly; every tool call is a
request the kernel adjudicates first, exactly like a process asking the OS
to run a syscall. Below you will propose four calls against one tiny policy
(examples/playground/policy.json) and read the kernel's verdict for each.

Nothing here needs a key, a model, a GPU, or the network — the verdict is a
pure function of (policy, call), so it is deterministic and yours to poke at.
----------------------------------------------------------------------
STEP 1 — an allowed read — the boring, common case
  proposed call:  read_file {"path":"README.md"}

    fak: loaded capability floor from examples/playground/policy.json
    tool: read_file   args: 20 bytes (sha 7d6441497d2a)
    verdict: ALLOW   by: monitor
    explanation: read_file allowed: an affirmative policy rung permitted it.

    decision chain (9 rung(s), most-restrictive wins):
       [0] grammar.Rung               DEFER     by=grammar
       [1] ratelimit.Limiter          DEFER     by=ratelimit
       [2] preflight.Ladder           DEFER     by=preflight
       [3] engine.residencyGate       DEFER     by=engine-residency
       [4] plancfi.Adjudicator        DEFER     by=plancfi
       [5] ifc.SinkGate               DEFER     by=ifc-sink
       [6] gitgate.GitGate            DEFER     by=gitgate
       [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
    => [8] adjudicator.Adjudicator    ALLOW     by=monitor   <- winner (rank 0)

  what happened:  ALLOW. read_file is on the allow-list, so an affirmative policy rung admits it. Most calls should look like this — the gate is invisible until you reach for something dangerous.
----------------------------------------------------------------------
STEP 2 — an rm -rf — watch it get denied
  proposed call:  run_shell {"cmd":"rm -rf /"}

    fak: loaded capability floor from examples/playground/policy.json
    tool: run_shell   args: 18 bytes (sha c11ec0d6ae74)
    verdict: DENY   reason: DEFAULT_DENY   by: monitor   disposition: TERMINAL
    explanation: run_shell denied by monitor: DEFAULT_DENY (TERMINAL).

    decision chain (9 rung(s), most-restrictive wins):
       [0] grammar.Rung               DEFER     by=grammar
       [1] ratelimit.Limiter          DEFER     by=ratelimit
       [2] preflight.Ladder           DEFER     by=preflight
       [3] engine.residencyGate       DEFER     by=engine-residency
       [4] plancfi.Adjudicator        DEFER     by=plancfi
       [5] ifc.SinkGate               DEFER     by=ifc-sink
       [6] gitgate.GitGate            DEFER     by=gitgate
       [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
    => [8] adjudicator.Adjudicator    DENY      DEFAULT_DENY by=monitor   <- winner (rank 100)

  what happened:  DENY / DEFAULT_DENY. Nobody blocklisted run_shell — it is refused because it was never ADMITTED. A default-deny floor does not have to predict the dangerous call to stop it.
----------------------------------------------------------------------
STEP 3 — an explicit deny — a different kind of no
  proposed call:  send_email {"to":"ceo@corp.example","body":"quarterly numbers attached"}

    fak: loaded capability floor from examples/playground/policy.json
    tool: send_email   args: 61 bytes (sha 9d0167dedf13)
    verdict: DENY   reason: POLICY_BLOCK   by: monitor   disposition: TERMINAL
    explanation: send_email denied by monitor: POLICY_BLOCK (TERMINAL).

    decision chain (9 rung(s), most-restrictive wins):
       [0] grammar.Rung               DEFER     by=grammar
       [1] ratelimit.Limiter          DEFER     by=ratelimit
       [2] preflight.Ladder           DEFER     by=preflight
       [3] engine.residencyGate       DEFER     by=engine-residency
       [4] plancfi.Adjudicator        DEFER     by=plancfi
       [5] ifc.SinkGate               DEFER     by=ifc-sink
       [6] gitgate.GitGate            DEFER     by=gitgate
       [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
    => [8] adjudicator.Adjudicator    DENY      POLICY_BLOCK by=monitor   <- winner (rank 100)

  what happened:  DENY / POLICY_BLOCK. send_email appears in the manifest, but under 'deny', so the reason is POLICY_BLOCK, not DEFAULT_DENY. Same verdict (DENY), a distinct, structured reason a loop can branch on.
----------------------------------------------------------------------
STEP 4 — a redacted secret — allowed, but sanitized first
  proposed call:  create_ticket {"body":"printer on 3rd floor is broken","password":"hunter2"}

    fak: loaded capability floor from examples/playground/policy.json
    tool: create_ticket   args: 62 bytes (sha 41f581de1a5d)
    verdict: TRANSFORM   by: monitor
    redacted: password
    explanation: create_ticket transformed by monitor: rewrote password before dispatch (e.g. secret redaction).

    decision chain (9 rung(s), most-restrictive wins):
       [0] grammar.Rung               DEFER     by=grammar
       [1] ratelimit.Limiter          DEFER     by=ratelimit
       [2] preflight.Ladder           DEFER     by=preflight
       [3] engine.residencyGate       DEFER     by=engine-residency
       [4] plancfi.Adjudicator        DEFER     by=plancfi
       [5] ifc.SinkGate               DEFER     by=ifc-sink
       [6] gitgate.GitGate            DEFER     by=gitgate
       [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
    => [8] adjudicator.Adjudicator    TRANSFORM by=monitor   <- winner (rank 20)

  what happened:  TRANSFORM. create_ticket is allowed, so the call runs — but 'password' is in redact_fields, so the kernel rewrites it to [REDACTED] BEFORE dispatch. A TRANSFORM sanitizes the call; a DENY stops it.
----------------------------------------------------------------------
Three distinct verdicts in four calls: ALLOW (admitted), DENY (refused —
with two different reasons, DEFAULT_DENY vs POLICY_BLOCK), and TRANSFORM
(admitted, then sanitized). None of them was a model "choosing to behave" —
each is the policy file applied to the call. Change policy.json and every
verdict above changes with it.
```
