---
title: "Durable forest lifecycle envelopes"
description: "internal/fleetbus schema fak-fleet-lifecycle/1 carries durable forest transitions over the directory bus. Requests support prepare, pause, checkpoint,"
---
# Durable forest lifecycle envelopes

`internal/fleetbus` schema `fak-fleet-lifecycle/1` carries durable forest transitions over the directory bus. Requests support `prepare`, `pause`, `checkpoint`, `restore`, `resume`, `stop`, `cancel`, and `status` with transaction/forest/member identity, generation, deadline, capability, causal predecessor, idempotency key, and root authority. ACKs carry accepted/completed/refused/unsupported plus checkpoint and read-back references.

`LifecycleGate` fails closed for malformed, expired, wrong-generation, unauthorized, and conflicting replay envelopes while accepting byte-equivalent redelivery under one idempotency key. `LifecycleDirBus.Broadcast` writes one durable request per target and reports any failed target instead of claiming a successful broadcast. Member ACKs are separate durable files and can be folded after controller restart.

This directory-bus spine provides local durability. Cross-host delivery guarantees under disconnection/backpressure and authentication/anti-replay cryptography remain dedicated follow-on leaves #6447 and #6448.
