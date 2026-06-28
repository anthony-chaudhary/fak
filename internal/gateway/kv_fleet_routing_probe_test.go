package gateway

import (
	"encoding/json"
	"math"
	"testing"
)

// TestKVAwareFleetRoutingHitRate is the committed witness for the kv-cache-aware
// fleet-routing rows in the industry scorecard (kv-cache-aware-routing,
// kv-aware-routing-locality). It pins the deterministic hit-rates the harness
// produces on the standing workload, so the committed artifact
// experiments/kv-fleet-routing/kv-fleet-routing-hitrate-20260627.json and the
// scorecard `fak_value` cannot silently drift from the code that produced them.
func TestKVAwareFleetRoutingHitRate(t *testing.T) {
	res := MeasureKVAwareFleetRouting(DefaultKVFleetWorkload)

	const (
		wantStream   = 464
		wantBlindHit = 0.8512931034482759 // cache-blind round-robin (what ReplicaRouter does today)
		wantAwareHit = 0.9396551724137931 // KV-aware locality routing
		wantLift     = 1.1037974683544303 // aware / blind
		eps          = 1e-9
	)

	if got := res.CacheBlind.Requests; got != wantStream {
		t.Fatalf("cache-blind requests = %d, want %d", got, wantStream)
	}
	if got := res.KVAware.Requests; got != wantStream {
		t.Fatalf("kv-aware requests = %d, want %d", got, wantStream)
	}
	if math.Abs(res.CacheBlind.HitRate-wantBlindHit) > eps {
		t.Errorf("cache-blind hit-rate = %.16f, want %.16f", res.CacheBlind.HitRate, wantBlindHit)
	}
	if math.Abs(res.KVAware.HitRate-wantAwareHit) > eps {
		t.Errorf("kv-aware hit-rate = %.16f, want %.16f", res.KVAware.HitRate, wantAwareHit)
	}
	if math.Abs(res.HitRateLift-wantLift) > eps {
		t.Errorf("hit-rate lift = %.16f, want %.16f", res.HitRateLift, wantLift)
	}

	// The whole point of locality routing: it must beat the cache-blind baseline
	// on the same request stream — the fleet analogue of the on-instance
	// FCFS -> cache-aware recovery.
	if res.KVAware.HitRate <= res.CacheBlind.HitRate {
		t.Errorf("kv-aware hit-rate %.4f must exceed cache-blind %.4f",
			res.KVAware.HitRate, res.CacheBlind.HitRate)
	}
	// And it must clear the published cross-replica cache-hit bar it is read
	// against (Baseten-on-Dynamo 0.89), the apples-to-apples competitor point.
	if res.KVAware.HitRate < res.Competitor.CrossReplicaHit {
		t.Errorf("kv-aware hit-rate %.4f below competitor cross-replica hit %.4f",
			res.KVAware.HitRate, res.Competitor.CrossReplicaHit)
	}

	b, _ := json.MarshalIndent(res, "", "  ")
	t.Logf("\n%s", b)
}
