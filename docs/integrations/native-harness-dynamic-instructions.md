# Dynamic instructions in a fak-native harness

The public [`pkg/harnesskit`](../../pkg/harnesskit) instruction plane lets a harness adjust application instructions at a declared run, thread, or turn boundary without replacing fak's kernel-owned policy prefix.

## Contract

Implement `harnesskit.InstructionProvider` (or `InstructionProviderFunc`). A provider receives typed runtime facts and returns independently attributable `InstructionFragment` values:

```go
provider := harnesskit.InstructionProviderFunc(func(
    ctx context.Context,
    req harnesskit.InstructionRequest,
) (harnesskit.InstructionSnapshot, error) {
    return harnesskit.InstructionSnapshot{Fragments: []harnesskit.InstructionFragment{{
        ID:         "operator-focus",
        Source:     "my-harness/operator",
        Trust:      harnesskit.TrustApplication,
        Precedence: 10,
        Lifetime:   harnesskit.LifetimeTurn,
        Residency:  harnesskit.ResidencyEphemeralTail,
        Content:    "For this turn, focus on " + req.Facts["focus"] + ".",
    }}}, nil
})
```

Resolve and realize it through the kernel adapter:

```go
realized, err := harnessinstructions.Resolve(ctx, provider, harnesskit.InstructionRequest{
    RunID: "run-1", TurnID: "turn-2", AgentRole: "coder",
    Facts: map[string]string{"focus": "latency"},
})
```

`Realization.PromptValue` is the exact system value. Its audit record includes the complete digest, stable-prefix digest, estimated bytes/tokens, fragment provenance, and inclusion reasons.

## Authority boundary

- Providers may author application, user, and untrusted fragments.
- The host owns policy, final serialization, and the stable prefix.
- A provider claiming host trust or `stable-prefix` residency fails with `CodeDenied`.
- Untrusted fragments cannot claim positive precedence.
- Identical typed inputs and provider output produce deterministic ordering and fingerprints.
- Cancellation reaches provider resolution as `CodeCanceled` with its original cause.

The cache-safe default is a stable fak base followed by application overlays or an ephemeral tail. Full replacement is intentionally not the public default because it could erase the resident policy floor.

## Captured selfcheck

Run:

```bash
go run ./cmd/instructiondemo -selfcheck
```

The demo resolves two turns with different operator focus. It passes only when the complete prompt digest changes while the audited kernel-prefix digest remains byte-identical. Captured output: [`docs/_witnesses/native-harness-dynamic-instructions-2026-08-18.json`](../_witnesses/native-harness-dynamic-instructions-2026-08-18.json).

The witness proves deterministic composition and cache-prefix preservation. It does not claim that one prompt improves model quality or reduces net tokens; those require a workload experiment.
