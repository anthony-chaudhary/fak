package gateway

import (
	"fmt"
	"testing"
	"time"
)

// affinityLen reports the number of recorded affinities under the policy lock so
// the capacity witness can read the live map without racing concurrent picks.
func affinityLen(p *CacheAwarePolicy) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.affinities)
}

// TestPickWithAffinity_CapacityBounds is the #10883 witness: transient session
// affinity keys that are never queried again must not grow the map without bound.
// The pre-fix policy retains every unique key until it is re-queried after its
// TTL, so 50k one-shot session ids leave 50k live entries. This test fails on the
// parent (map exceeds 10000) and passes once the map is capacity-bounded.
func TestPickWithAffinity_CapacityBounds(t *testing.T) {
	const max = 10000
	simTime := time.Unix(2000, 0)
	p := NewCacheAwarePolicy(nil, DefaultSkewThreshold()).
		WithClock(func() time.Time { return simTime })

	candidates := []PlannerReplica{{Name: "w0"}, {Name: "w1"}}

	for i := 0; i < max*5; i++ {
		key := fmt.Sprintf("transient-sess-%06d", i)
		if _, ok := p.PickWithAffinity(candidates, []string{"prompt"}, key, nil); !ok {
			t.Fatalf("pick %d failed", i)
		}
		if got := affinityLen(p); got > max {
			t.Fatalf("affinities map grew to %d after %d one-shot keys, want <= %d", got, i+1, max)
		}
	}

	if got := affinityLen(p); got > max {
		t.Fatalf("final affinities size = %d, want <= %d", got, max)
	}
}
