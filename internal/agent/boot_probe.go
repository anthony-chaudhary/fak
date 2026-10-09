package agent

import "context"

type bootProbeKey struct{}

// WithBootProbe marks a planner call as a gateway-internal boot probe (the
// readiness warmup turn and the startup coherence probe), not a served turn. The
// in-kernel planner still runs it end to end, but keeps it out of the
// process-global serving taps (cacheobs.Default) so fak_gateway_kv_prefix_* counts
// exactly the turns the gateway's latency histograms and perf ledger count.
func WithBootProbe(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, bootProbeKey{}, true)
}

// IsBootProbe reports whether ctx was marked by WithBootProbe.
func IsBootProbe(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	probe, _ := ctx.Value(bootProbeKey{}).(bool)
	return probe
}
