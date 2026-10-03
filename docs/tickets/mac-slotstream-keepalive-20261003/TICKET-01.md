# Mac Slotstream keepalive direct port
Recon: existing Slotstream C1 study (2026-09-26) approves DIRECT-PORT MIT.
Reuse: extend internal/metalgemm and model generation/Step boundaries; no new runtime.
Source: carloslfu/slotstream@a855c49090330df88b2464814bcc003c8a7c5bfe Sources/Slotstream/GPUKeepAlive.swift:22-169; LICENSE Copyright (c) 2026 Carlos Galarza inspected at exact pin.
Copy: exact finite MSL spin kernel; adapt ObjC lifecycle with completion callbacks rather than unbounded waits. Separate queue; maximum two in flight; balanced nested Go generation leases; unknown power/battery/low power auto disabled; off default until physical A/B establishes benefit.
Boundary: public core GPU runtime. Blast: metalgemm + model only; Step covers manual benchmarks; Generate scopes cover inter-token gaps.
Removal: remove keepalive files, model hook and three defer calls, Darwin callback registration, plus cmd/modelbench stepDecode/runNativeProfileForward scopes and metal_keepalive report field; no persisted state or model format change.
Witness: go test ./internal/metalgemm ./internal/model; go run ./cmd/modelbench same model and decode shape with off/on, collect default-readable counters. Independent tests own policy/lifecycle; compile/GPU slot held until coordinator grants.

## Implementation and qualification policy
The finite Metal kernel is copied verbatim (except source-string indentation) from the pinned MIT source, retaining its notice. The Objective-C wrapper adapts the lifecycle using completion callbacks, never a host waiting thread. A sticky command-buffer failure stops resubmission. The shared flag ends the last lease, and at most two finite buffers remain to drain. Nested leases are idempotent and balanced on panic/cancellation through defer.

`FAK_METAL_KEEPALIVE=off` (also the unset default) creates no GPU keepalive resources. `on` is an explicit experiment; `auto` requires known AC power outside Low Power Mode. Unknown power fails closed. This is an optional module until matched Mac measurements establish a benefit; the upstream M5 result is not a claim about this Mac. Default modelbench decode output includes the bounded `metal_keepalive` process receipt. Manual Step loops get token scopes; the measured modelbench loop and model Generate/GenerateContext additionally hold a lease across token gaps.

Native admission witness: `fak-run fak-sync dos arbitrate --workspace <public-root> --lane metalgemm --kind keyword --mode exclusive --tree internal/metalgemm/*keepalive* internal/model/*generate* cmd/modelbench/* --output json` GREEN acquire, zero live leases (coordinator witness). Existing qualified GDN C8, MLX attention and scratch pooling are already landed; this extends the remaining explicit direct-port Slotstream C1 gap.

## Final reachable route and validation
The Darwin session hook admits both MetalQ4K sessions and the dense Metal decode selector. The first whole-model ON-arm receipt exposed zero submitted buffers: the initial dense-only predicate did not admit the measured Q4_K route. The predicate was corrected to include MetalQ4K before final qualification; the route test and the default receipt bind the keepalive work to the actual Q4_K benchmark path.

Final serialized build and go vet passed for the changed packages. Acceptance commands (physical tests require an available GPU lease):

```text
GOMAXPROCS=2 GOFLAGS=-p=1 GOMEMLIMIT=768MiB go build -o _scratch/mac-slotstream-keepalive/modelbench ./cmd/modelbench
go vet ./internal/metalgemm ./internal/model ./cmd/modelbench
go test ./internal/metalgemm -run TestKeepAlive -count=1
FAK_METAL_KEEPALIVE_TEST=1 go test ./internal/metalgemm -run TestKeepAliveMetalOutputAndDrain -count=1
FAK_METAL_KEEPALIVE_TEST=1 go test ./internal/model -run TestSessionGPUKeepAliveQ4Reachability -count=1
go test ./internal/model -run 'TestGenerateContext|TestGenerateBatchContext' -count=1
```

## Desktop cadence adaptation
Whole-model ON qualification with the upstream 200,000-poll wrapper count produced a sticky `MTLCommandBufferErrorDomain` code 1, “Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)” after 380 submissions on the desktop Mac. The copied MSL body remains unchanged; the wrapper now supplies 2,048 polls per finite pulse. This is an experimental response to the observed driver refusal, retaining at most two buffers, the stop flag, zero host waits and fail-closed behavior. The default receipt exposes actual iterations and maximum completed GPU milliseconds from available command-buffer timestamps, so the revised pulse duration and retry outcome are measurable. The feature remains off by default pending robust qualification.
