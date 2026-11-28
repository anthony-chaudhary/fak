# Server metrics endpoint discovery ladder

Finding a server's Prometheus endpoint is an ordered ladder, not a single guess. Each
rung is tried in order and the first rung that yields an address wins, so an explicit
operator choice outranks inferred ones. If no rung finds an endpoint the result is a
typed error (`ErrNoDiscoveryEndpoint`), never an empty address.

Inspired by ai-dynamo/aiperf `src/aiperf/server_metrics/discovery/kubernetes.py:39`
(`ab610add4e8876fca223197596c728e17365f381`, Apache-2.0). This is policy only: it makes no
cluster calls and is not wired into a serve path. The ladder lives in
`internal/gateway/discovery_policy.go`; witness:
`go test ./internal/gateway -run TestDiscoveryPolicy -count=1`.
