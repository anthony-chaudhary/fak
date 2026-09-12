## Current state

Two gateway package tests impose timing outcomes that the runtime does not promise. The pre-first-token heartbeat test correctly checks four seconds of wire silence, then rejects any legal heartbeat after the stream commits. The turn-cost lifecycle test requires a positive ledger timing sample even though TurnCostRecord intentionally omits a phase measured at zero duration; the durable turn_complete event still proves completion.

## GitHub issues

- https://github.com/anthony-chaudhary/fak/issues/12862
- https://github.com/anthony-chaudhary/fak/issues/12863

## Scope

Stabilize only the two test oracles against durable lifecycle and protocol facts.

## Working spine

1. Preserve the full pre-commit heartbeat silence assertion.
2. Deterministically pause after the first token and validate legal typed heartbeats before completion.
3. Assert turn completion from the persisted cost-lifecycle ledger chain.
4. Retain receipt identity, backend phase, turn-count, content reassembly, and DONE checks.

## Gold-plating boundary

Do not modify production, sleeps used as thresholds, heartbeat intervals, duration floors, metrics, or request behavior.

## Acceptance gate

The formerly failing checkout geometry passes each focused test repeatedly and the full gateway package once.

## Done condition

Both timing-sensitive tests validate their actual contracts without requiring scheduler order or a fabricated positive duration.
