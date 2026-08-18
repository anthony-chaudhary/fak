# Disambiguation lookup demo

This CPU-only public demo models the smallest agent lookup path: search an overloaded term, receive every choice, select an explicit scope, and read a canonical contrast.

```bash
go run ./cmd/disambiguationdemo -selfcheck
# QUERY runtime: ambiguous choices=[agent-application gateway-serving guard-enforcement model-serving worker-execution]
# SCOPE runtime=gateway-serving -> runtime
# CONTRAST ...
# SELFCHECK PASS: public local index only; no model, key, GPU, network, or private data
```

Use `-json` for the deterministic `fak-disambiguation-demo/1` receipt. The demo calls the same public `Search` and `QueryScoped` seams as an agent integration and owns no terminology data.
