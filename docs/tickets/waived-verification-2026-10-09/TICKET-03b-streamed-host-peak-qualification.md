# fix(cmd): qualify the streamed-expert host-peak estimate so the bounded dense route is admitted without an override

## Parent context

Child of TICKET-03 (Vulkan memory accounting and residency) in this directory,
parent #13761 (https://github.com/anthony-chaudhary/fak/issues/13761), serving #13668.
Evidence: operator-held silicon receipt (schema `fak.v41_native_probe.v1`, probed 2026-10-09T23:37Z, binary sha256 `2f3f2298bbae73a0`).

<!-- fak-model-key: waived-vulkan-streamed-host-peak-qualification -->
<!-- fak-public-issue: anthony-chaudhary/fak#13774 -->

```routing
lane: cmd/serve
paths: ["cmd/fak/serve_host_peak_admission.go", "internal/ggufload/host_load_peak_streamed_experts.go", "docs/tickets/waived-verification-2026-10-09/TICKET-03b-streamed-host-peak-qualification.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#13774
```

Process cause: verification-gap

## Scope class

multi-package

## Current state

[HW-WITNESSED] On a Strix Halo appliance (MemTotal 30.97 GiB) at fak 42dc8adb624, V4.1 Flash arm B
behaves as follows:

- With default env, it refuses `HOST_LOAD_PEAK_UNAVAILABLE (header-estimate-unavailable)`.
  The ring options carry no bounded dense working set.
- With `FAK_STREAM_Q4K=1`, it refuses `HOST_LOAD_PEAK_UNAVAILABLE (streamed-staging-peak-unqualified)`
  at `cmd/fak/serve_host_peak_admission.go:113`, even though the estimate fits.
- With `FAK_STREAM_Q4K=1 FAK_SERVE_HOST_PEAK_ADMISSION=off`, the load ran for 21 min. Sampled at
  1 Hz, RssAnon peaked at 12.09 GiB, then stayed flat at 6.72 GiB. Swap never rose.

## Problem

Every streamed route is refused, so the only way to run the route that measured as bounded
is to disable admission entirely. The estimator omits the F32 arena coalescing copy, the
tied-head Q8 storage and some transform lifetimes. Until those terms are covered, a fitting
estimate cannot authorize staging.

## Why this is next

The bounded route is now the only one that loads V4.1 Flash on a 31 GiB host, and it should not
need an override that removes all host protection.

## Working spine

1. Record the estimator's predicted peak for the arm-B option list.
2. Add the omitted lifetime terms, and pin them with header-only tests.
3. Turn `streamed-staging-peak-unqualified` into an admit when the estimate is qualified, and have
   the ring options thread the bounded dense working set by default.

## Core through-line

The estimate is an upper bound on the measured peak (12.09 GiB peak, 6.72 GiB plateau).

## Gold-plating boundary

No loader rewrite and no performance work.

## Verifiable Witness

```bash
go test ./cmd/fak/ -run 'HostLoadPeak|HostPeak' -count=1
go test ./internal/ggufload/ -run 'StreamedExpertHostLoadPeak|StreamedDense' -count=1
```

## Done condition / witness

Witness: the commands above, plus one arm-B silicon log.

- [ ] [SW-VERIFIED] The estimator covers every host anon lifetime the streamed route allocates (named, with tests).
- [ ] [HW-WITNESSED] Arm B is admitted without `FAK_SERVE_HOST_PEAK_ADMISSION=off`, and the measured peak RssAnon is at or below the predicted peak.

## Witness

The HW box needs a 1 Hz RssAnon trace next to the logged predicted peak.

## Definition of done

Both boxes are ticked, and the resolving commit cites this issue.

## Acceptance gate

The touched package tests stay green, and boundary and scrub checks are green.

## Closure binding

The resolving commit cites this public issue and carries the `cmd` or `ggufload` leaf in its subject.

## Likely files

- `cmd/fak/serve_host_peak_admission.go`: :63 bounded-dense requirement, :113 unqualified refusal.
- `internal/ggufload/host_load_peak_streamed_experts.go`: the estimator.
- `cmd/fak/serve_activated_expert_ring.go`: the ring load options (:104).

## Lane

cmd/serve

## Blast radius and affected lanes

`cmd/fak` admission plus `internal/ggufload` estimation. Admission policy for the ordinary resident route does not change.

## Quarantined fallback mechanism

Keep the refusal and the explicit override.

## Problem frame

- Centrality: Core (the native V4.1 load on a Halo)
- P1 Context: preserved - no change to context, KV or prompt handling; only the touched load or test path moves.
- P2 Net value: advanced - the route measured as bounded becomes admissible while host protection stays on.
- P3 Adaptation: preserved - no new external mechanism is adopted; existing fak paths are corrected.
- P4 Operations: advanced - safe admission without an override.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3/9 points toward parent #13761.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13774
