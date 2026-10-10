# TICKET-09: V4.1 sparse-sink non-finite rejection changed a pinned host-control contract

## Current state

At fak `0016a2a6a9f`, `TestV41SharedAttentionDeviceAdapter/unsupported-host-and-selected-sentinel-overflow` (`internal/model/v41_shared_attention_device_test.go:590`) fails. The host-control arm now errors with "sparse sink ... non-finite value=+Inf". Waived commit 1882b2f8849 "reject nonfinite sparse sink arithmetic" changed the host contract that this test pins.

Parent: #13764.

## Working spine

1. Decide which contract is correct: the sentinel-overflow case either expects a typed refusal or the new non-finite error.
2. Make the code and the test agree, using a sentinel or `errors.Is` check rather than matching error text.
3. Confirm that the device adapter and host control refuse the same way.

## Witness

`go test -count=1 -timeout 60m -run 'V41SharedAttentionDeviceAdapter' ./internal/model/` exit 0.

## Done condition

- [ ] One documented contract, tested on both arms, and cited to 1882b2f8849.
