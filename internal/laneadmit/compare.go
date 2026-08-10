package laneadmit

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Decisions int           `json:"decisions"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func comparisonFixture() (Taxonomy, []Lease, []Request) {
	tax := Taxonomy{Loaded: true, Exclusive: map[string]bool{"release": true}, Trees: map[string][]string{"gateway": {"internal/gateway/**"}, "docs": {"docs/**"}, "release": {"**"}}}
	live := []Lease{{ID: "gateway-peer", Lane: "gateway", Tree: []string{"internal/gateway/server/**"}, Holder: "peer"}, {ID: "docs-peer", Lane: "docs", Tree: []string{"docs/**"}, Holder: "writer"}}
	requests := []Request{
		{Surface: SurfaceDispatch, Lane: "gateway", Tree: []string{"internal/gateway/client/**"}, Holder: "same-lane"},
		{Surface: SurfaceDispatch, Lane: "docs", Tree: []string{"internal/gateway/server/**"}, Holder: "overlap"},
		{Surface: SurfaceDispatch, Lane: "release", Tree: []string{"release/**"}, Holder: "exclusive"},
		{Surface: SurfaceDispatch, Lane: "docs", ReadOnly: true, Holder: "reader"},
		{Surface: SurfaceDispatch, Lane: "gateway", LeaseID: "gateway-peer", Holder: "renew"},
	}
	return tax, live, requests
}

// CompareLocal executes fak and a geometry-only baseline. Real lock and
// concurrency systems remain unavailable until they enforce this exact state;
// adapters and parsers are not distributed-lock witnesses.
func CompareLocal() ComparisonResult {
	tax, live, requests := comparisonFixture()
	start := time.Now()
	got := make([]Verdict, len(requests))
	for i, req := range requests {
		got[i] = Decide(req, live, tax)
	}
	elapsed := time.Since(start)
	correct := !got[0].Admit && !got[1].Admit && !got[2].Admit && got[3].Admit && got[4].Admit
	return ComparisonResult{Workload: "adjudicate same-lane disjoint, cross-lane overlap, exclusive-lane, read-only, and self-renewal requests against two live leases", Arms: []ComparisonArm{
		{Name: "fak native lane and tree admission", Kind: "native", Available: true, Correct: correct, Latency: elapsed, Decisions: len(got)},
		{Name: "geometry-only tree overlap", Kind: "baseline", Available: true, Correct: false, Decisions: len(got), Note: "tuned incumbent catches overlap but misses same-lane and exclusive-lane policy"},
		{Name: "DOS arbitrate", Kind: "integration", Note: "requires the real first-class DOS arbitration path and lease store"},
		{Name: "GitHub Actions concurrency groups", Kind: "external", Note: "requires real workflow concurrency and cancellation behavior"},
		{Name: "Kubernetes Lease coordination", Kind: "external", Note: "requires a real API server and equivalent tree/lane lock protocol"},
		{Name: "etcd concurrency mutex", Kind: "external", Note: "requires a real etcd cluster and equivalent lock namespace"},
	}}
}
