# fabricmapdemo

`fabricmapdemo` is the minimal working spine for modular data movement between
arbitrary storage, memory, compute, and fabric endpoints. Endpoint names such as
`L1` and `L3` are user metadata, not an encoded hierarchy. Every capability is a
directed edge, so reverse movement must be declared and can differ from forward
movement.

```console
go run ./cmd/fabricmapdemo -selfcheck
```

The self-check selects a direct SSD-to-GPU CPU-bypass edge. Use `-manifest` with
JSON containing `graph` and `request` to model other technologies. Link labels
are open-ended so newer transports and properties do not require changing the
planner's type taxonomy.
