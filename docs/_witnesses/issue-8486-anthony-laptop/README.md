# Anthony laptop baseline — issue #8486

This directory is the scrubbed public projection of a real authenticated laptop run. Private bridge coordinates, accounts, addresses, device identifiers, and raw logs remain in the authorized companion evidence store.

## Rerun

From an authorized `fak-private` checkout, update `main`, run the bridge `doctor` and readback self-test, select the registered Anthony laptop session, then execute the canonical native CUDA Qwen2.5-3B Q8_0 P32/D32 benchmark with `-require-non-reference` three times. Retain raw output privately and regenerate `baseline.json` from the three successful receipts.

Preconditions: laptop awake; AC power connected; performance power mode; no competing GPU workload; stable thermal idle before trial 1; preserve any existing reference server. Cleanup removes only benchmark-created processes and temporary artifacts. Do not weaken host policy, enable password automation, or publish bridge coordinates.

The public receipt intentionally records the comparable metrics and capability boundary while omitting private topology and uniquely identifying inventory. Known receipt gaps are tracked by #9121.
