# fabricmapdemo

`fabricmapdemo` is the minimal working spine for modular data movement between
arbitrary storage, memory, compute, and fabric endpoints. Endpoint names such as
`L1` and `L3` are user metadata, not an encoded hierarchy. Every capability is a
directed edge, so reverse movement must be declared and can differ from forward
movement.

```console
go run ./cmd/fabricmapdemo -selfcheck
```

## What you see

The JSON route contains one `L3 -> L1` link whose CPU path is `bypass` and whose
`gpu-direct` label is `yes`. The PASS line means the selfcheck asserted that exact
directed edge rather than merely printing the planner result.

The self-check selects a direct SSD-to-GPU CPU-bypass edge. Use `-manifest` with
JSON containing `graph` and `request` to model other technologies. Link labels
are open-ended so newer transports and properties do not require changing the
planner's type taxonomy.

Go 1.26 or newer is the only prerequisite; the selfcheck needs no model, key,
GPU, network, or external data. With the toolchain and module cache already
available, it completes in a few seconds (2.7 seconds in the captured Windows
run). A cold toolchain download or first compile can take longer. The built-in
graph and request contain no clock or randomness, so the JSON route and PASS
verdict are deterministic. A successful selfcheck exits 0; malformed or
unplannable input exits nonzero. See the [captured output](EXAMPLE-OUTPUT.md).

This demo does not claim to execute a data transfer, benchmark a transport, or
prove that the named hardware exists. It exercises only route selection over a
declared graph; the broader control-plane context is documented in the
[address-space study](../../docs/notes/CONCEPT-STUDY-THERE-IS-NO-ADDRESS-2026-08-20.md).
