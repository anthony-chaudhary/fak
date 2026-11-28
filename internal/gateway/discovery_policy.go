package gateway

import "errors"

// INSPIRE of ai-dynamo/aiperf (Apache-2.0),
// src/aiperf/server_metrics/discovery/kubernetes.py:39 @ab610add4e8876fca223197596c728e17365f381:
// an ordered ladder for finding a server's Prometheus endpoint. Each rung is tried in
// order; the first that yields an address wins, and an explicit operator choice always
// outranks inferred ones. Policy and host oracle only: no cluster access, not wired in.

// DiscoveryRung is one way of locating a metrics endpoint.
type DiscoveryRung struct {
	Name string
	// Find returns an address and whether this rung found one.
	Find func() (string, bool)
}

// ErrNoDiscoveryEndpoint means no rung located an endpoint.
var ErrNoDiscoveryEndpoint = errors.New("gateway: no metrics endpoint discovered")

// DiscoverMetricsEndpoint walks the ladder in order and reports the winning rung.
func DiscoverMetricsEndpoint(ladder []DiscoveryRung) (addr, rung string, err error) {
	for _, r := range ladder {
		if r.Find == nil {
			continue
		}
		if a, ok := r.Find(); ok && a != "" {
			return a, r.Name, nil
		}
	}
	return "", "", ErrNoDiscoveryEndpoint
}
