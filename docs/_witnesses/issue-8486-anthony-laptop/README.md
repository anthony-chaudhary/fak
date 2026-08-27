---
title: "Anthony laptop native CUDA benchmark baseline — issue #8486"
description: "Scrubbed three-trial fak-native CUDA benchmark receipt for a Windows laptop, with throughput, VRAM, power, temperature, and stated evidence gaps."
---
# Anthony laptop baseline — issue #8486

This directory is the scrubbed public projection of a real authenticated laptop run. Private bridge coordinates, accounts, addresses, device identifiers, and raw logs remain in the authorized companion evidence store.

The witnessed public result is [`baseline.json`](baseline.json): three successful fak-native CUDA trials with throughput, VRAM, power, temperature, engine identity, and explicit receipt limitations.

## Rerun

From an authorized `fak-private` checkout, update `main`, run the bridge `doctor` and readback self-test, select the registered Anthony laptop session, then execute the canonical native CUDA Qwen2.5-3B Q8_0 P32/D32 benchmark with `-require-non-reference` three times. The exact private bridge invocation and model path stay in the authorized runbook; retaining them here would publish the control route. Retain raw output privately and regenerate `baseline.json` from the three successful receipts.

Preconditions: laptop awake; AC power connected; performance power mode; no competing GPU workload; stable thermal idle before trial 1; preserve any existing reference server. Cleanup removes only benchmark-created processes and temporary artifacts. Do not weaken host policy, enable password automation, or publish bridge coordinates.

## Generation decision

- **Promotion evidence:** promote the node to `registered` for the declared single-user offline, CPU, GPU, dataset, and serving classes because the authenticated private route completed and all three fak-native CUDA trials exited 0.
- **Demotion or retirement evidence:** demote or retire the node from those classes if the authenticated route stops completing, a repeatable three-trial baseline cannot be produced under the stated preconditions, or fleet status can no longer resolve this witness. Keep the explicit public-serving, unattended-overnight, multi-GPU, and datacenter-network exclusions until separately witnessed.
- **Invalidating assumption:** this baseline assumes AC performance mode, thermal idle, no competing GPU workload, the pinned source/model/workload envelope in `baseline.json`, and a native CUDA receipt. A material change to any of those conditions invalidates direct comparison and requires a new baseline rather than silently reusing these numbers.

The public receipt intentionally records the comparable metrics and capability boundary while omitting private topology and uniquely identifying inventory. Known receipt gaps are tracked by #9121.
