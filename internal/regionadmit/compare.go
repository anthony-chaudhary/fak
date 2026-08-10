package regionadmit

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// ComparisonArm is one implementation of the same shared-region admission
// workload. Unavailable arms deliberately retain zero measurements.
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

// ComparisonResult records the common workload and its explicit alternatives.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func comparisonFixture() (Taxonomy, []Lease, []Request) {
	tax := Taxonomy{
		Exclusive: map[string]bool{"release": true},
		Trees: map[string][]string{
			"gateway":      {"internal/gateway/**"},
			"gateway/http": {"internal/gateway/http/**"},
			"docs":         {"docs/**"},
			"release":      {"**"},
		},
	}
	live := []Lease{
		{ID: "gateway-peer", Lane: "gateway", Tree: []string{"internal/gateway/server/**"}, Holder: "peer"},
		{ID: "docs-peer", Lane: "docs", Tree: []string{"docs/**"}, Holder: "writer"},
	}
	requests := []Request{
		{Actor: "narrow", Lane: "gateway", Tree: []string{"internal/gateway/client/**"}},
		{Actor: "overlap", Lane: "docs", Tree: []string{"internal/gateway/server/**"}},
		{Actor: "exclusive", Lane: "release", Tree: []string{"release/**"}},
		{Actor: "reader", Lane: "docs", ReadOnly: true},
		{Actor: "renew", Lane: "gateway", SelfID: "gateway-peer", Tree: []string{"internal/gateway/server/**"}},
		{Actor: "sub-lane", Lane: "gateway/http", Tree: []string{"internal/gateway/http/client/**"}},
	}
	return tax, live, requests
}

func geometryOnlyDecide(req Request, live []Lease, tax Taxonomy) bool {
	if req.ReadOnly {
		return true
	}
	tree := ResolveTree(req, tax)
	for _, lease := range live {
		if lease.ReadOnly || (req.SelfID != "" && req.SelfID == lease.ID) {
			continue
		}
		if dispatchorder.TreesOverlap(tree, lease.Tree) {
			return false
		}
	}
	return true
}

// CompareLocal executes fak and the tuned geometry-only incumbent. Distributed
// lock adapters and parsers are not witnesses for the real acquisition paths.
func CompareLocal() ComparisonResult {
	tax, live, requests := comparisonFixture()
	want := []bool{true, false, false, true, true, false}

	start := time.Now()
	nativeCorrect := true
	for i, req := range requests {
		if Decide(req, live, tax).Admit != want[i] {
			nativeCorrect = false
		}
	}
	nativeLatency := time.Since(start)

	start = time.Now()
	baselineCorrect := true
	for i, req := range requests {
		if geometryOnlyDecide(req, live, tax) != want[i] {
			baselineCorrect = false
		}
	}
	baselineLatency := time.Since(start)

	return ComparisonResult{
		Workload: "adjudicate narrowed-disjoint, cross-lane overlap, exclusive-lane, read-only, self-renewal, and hierarchical sub-lane requests against two live leases",
		Arms: []ComparisonArm{
			{Name: "fak native shared region admission", Kind: "native", Available: true, Correct: nativeCorrect, Latency: nativeLatency, Decisions: len(requests)},
			{Name: "geometry-only region overlap", Kind: "baseline", Available: true, Correct: baselineCorrect, Latency: baselineLatency, Decisions: len(requests), Note: "tuned incumbent honors read-only and self-renewal but misses exclusivity and hierarchical lane serialization"},
			{Name: "fak + DOS arbitrate", Kind: "integration", Note: "requires the real first-class DOS arbiter and lease store"},
			{Name: "fak + Git-ref leases", Kind: "integration", Note: "requires real leaseref acquisition and cross-process visibility"},
			{Name: "Kubernetes Lease coordination", Kind: "external", Note: "requires a real API server and equivalent region namespace"},
			{Name: "etcd concurrency mutex", Kind: "external", Note: "requires a real etcd cluster and equivalent region namespace"},
			{Name: "GitHub Actions concurrency groups", Kind: "external", Note: "requires real workflow concurrency and cancellation behavior"},
		},
	}
}
