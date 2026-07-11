package dispatchtick

import "sort"

type DiscoveryReconcile struct {
	Routable   []string `json:"routable"`
	Readmitted []string `json:"readmitted,omitempty"`
}

// ReconcileDiscovery resets the soft-evicted routable set to authoritative
// discovery when that source has remained stable for the reconcile interval.
func ReconcileDiscovery(authoritative []string, softEvicted map[string]bool, discoveryStable bool) DiscoveryReconcile {
	out := DiscoveryReconcile{}
	for _, id := range authoritative {
		if id == "" {
			continue
		}
		if !softEvicted[id] || discoveryStable {
			out.Routable = append(out.Routable, id)
			if softEvicted[id] && discoveryStable {
				out.Readmitted = append(out.Readmitted, id)
			}
		}
	}
	sort.Strings(out.Routable)
	sort.Strings(out.Readmitted)
	return out
}
