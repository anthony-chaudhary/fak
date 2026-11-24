package routing

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
// fak-test:justify why=regression when=changed:platform/routing/balance_select.go
func TestBalanceSelectMixedReceipt(t *testing.T) {
	group := BalanceGroup{
		Name: "mixed", Members: []string{"drained", "slow", "unknown", "saturated", "fast", "tie"},
		Weights: map[string]int{"drained": 0, "slow": 3, "fast": 2},
	}
	snap := BalanceSnapshot{Group: group.Name, Observed: true, Members: map[string]MemberScore{
		"drained":   {Load: -.2, Latency: 2, Measured: true},
		"slow":      {Load: .4, Latency: .8, Measured: true},
		"saturated": {Load: 2, Latency: -.2, Measured: true},
		"fast":      {Load: .1, Latency: .1, Measured: true},
		"tie":       {Load: .1, Latency: .1, Measured: true},
	}}
	// Freeze the complete pre-optimization contract: order, every score and
	// receipt field, clamping, exclusion order, tie-breaking and audit text.
	want := map[BalanceStrategy]string{
		BalanceStrategyPriority:     "9b766e23b4d1807af972826101468fa7d9c247195a71e9026b9d1f94b07a30f7",
		BalanceStrategyWeightedHash: "32352b53e74f535f4d905cad60f28f87e4a176e78ad37cb8e793cb645be7c1b4",
		BalanceStrategyLeastLoad:    "57bb6263d94205560da3ec4076876324e67119a26390861aaf9d9b0ea951d1c9",
		BalanceStrategyLeastLatency: "407ffc7784852517aab571d456e9314d6f9b71a77dabc4a9d9d441a73112638a",
	}
	for _, strategy := range []BalanceStrategy{BalanceStrategyPriority, BalanceStrategyWeightedHash, BalanceStrategyLeastLoad, BalanceStrategyLeastLatency} {
		t.Run(string(strategy), func(t *testing.T) {
			group.Strategy = strategy
			got := (BalancePolicy{}).Select(group, snap, "request-mixed")
			blob, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if hash := fmt.Sprintf("%x", sha256.Sum256(blob)); hash != want[strategy] {
				t.Fatalf("selection receipt changed: hash=%s, want=%s; receipt=%s", hash, want[strategy], blob)
			}
		})
	}
}

// fak-test:runtime fast est=5s lane=default
// fak-test:justify why=regression when=changed:platform/routing/balance_select.go
func BenchmarkBalanceSelectScaling(b *testing.B) {
	for _, n := range []int{64, 512, 4096} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			group := BalanceGroup{Name: "scale", Strategy: BalanceStrategyLeastLoad, Members: make([]string, n)}
			snap := BalanceSnapshot{Group: group.Name, Members: make(map[string]MemberScore, n)}
			for i := range group.Members {
				m := fmt.Sprintf("member-%06d", i)
				group.Members[i] = m
				snap.Members[m] = MemberScore{Load: float64(i%17) / 20, Latency: .2, Measured: true}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				got := (BalancePolicy{}).Select(group, snap, "request-scale")
				if len(got.Members) != n || got.Plan.Chosen != got.Members[0] {
					b.Fatal("incomplete selection")
				}
			}
		})
	}
}
