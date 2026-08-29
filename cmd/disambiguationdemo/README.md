# Disambiguation lookup demo

This CPU-only public demo models the smallest agent lookup path: search an overloaded term, receive every choice, select an explicit scope, and read a canonical contrast.

```bash
go run ./cmd/disambiguationdemo -selfcheck
# QUERY runtime: ambiguous choices=[agent-application gateway-serving guard-enforcement model-serving worker-execution]
# SCOPE runtime=gateway-serving -> runtime
# CONTRAST ...
# SELFCHECK PASS: public local index only; no model, key, GPU, network, or private data
```

## What you see

`QUERY` exposes all five exact matches instead of guessing a meaning. `SCOPE`
records the explicit choice, `CONTRAST` supplies the index's canonical distinction,
and `SELFCHECK PASS` means those lookup assertions completed successfully.

Use `-json` for the deterministic `fak-disambiguation-demo/1` receipt. The demo calls the same public `Search` and `QueryScoped` seams as an agent integration and owns no terminology data.

The only prerequisite is Go 1.26 or newer. With the toolchain and module cache
already available, the selfcheck completes in a few seconds (1.9 seconds in the
captured Windows run); a cold toolchain download or first compile can take longer.
The command exits 0 on success. Its fixed local index and query keep the receipt deterministic.
See the [captured selfcheck](EXAMPLE-OUTPUT.md) and the
[full terminology index](../../docs/generated/disambiguation/INDEX.md).

This demo does not claim the terminology catalog is complete or that it resolves
free-form intent. It proves only the public ambiguous-search and explicitly scoped
lookup path shown above.
