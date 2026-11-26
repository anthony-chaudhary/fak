# piidemo — ToolGuard PII masking, end to end

One page, two real gateway arms, one synthetic tool output carrying five
protected spans (email, phone, SSN, card number, credential). The left arm is
the default router with ToolGuard off: the bytes reach the model unchanged. The
right arm is the same router with ToolGuard on: the spans are masked before
admission, a `[fak] masked …` note renders in-band, and the admitted bytes are
shown.

Both arms run through the real gateway surface (`POST /v1/messages`,
`POST /v1/fak/admit`). The model behind both arms is fak's deterministic
offline mock planner, so no weights or network are needed.

## Run it

Headless self-check (CI and cross-platform dogfood). It prints one PASS line per
assertion and exits non-zero on any drift:

```sh
go run ./cmd/piidemo -selfcheck
```

Browser demo:

```sh
go run ./cmd/piidemo -addr 127.0.0.1:8158
# open http://127.0.0.1:8158 and click "Run both arms"
```

A captured self-check run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What this does not claim

- It does not claim masking is complete. The span detectors cover the five
  classes in the scenario; other PII shapes are out of scope for this demo.
- It does not claim anything about a live model's behaviour. The planner is a
  deterministic mock, so the result says nothing about how a real model reacts
  to masked input.
- The assertions cover the documented invariants of this one scenario, not a
  general privacy or compliance guarantee.

The claim boundary for the wider ToolGuard posture is in
[`CLAIMS.md`](../../CLAIMS.md).
