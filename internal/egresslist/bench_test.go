package egresslist

import "testing"

// BenchmarkEgressList exercises host matching and decisions across blocked, allowed, and fallthrough targets.
func BenchmarkEgressList(b *testing.B) {
	list := NewBuilder().
		AddRules("community-block", []string{"badware.example", "tracker.test", "ads.network"}, Block).
		AddRules("operator-allow", []string{"api.badware.example", "docs.tracker.test"}, Allow).
		Build()

	hosts := []string{
		"badware.example",
		"sub.badware.example",
		"api.badware.example",
		"docs.tracker.test",
		"allowed.external.domain",
		"plain.example.org",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h := hosts[i%len(hosts)]
		_ = list.Decide(h)
	}
}
