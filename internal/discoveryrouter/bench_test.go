package discoveryrouter

import (
	"testing"
)

func BenchmarkDiscoveryRouter(b *testing.B) {
	p := Plan{
		Adapters: []Adapter{
			fakeAdapter{
				name:     "docs",
				relevant: true,
				hits: []Evidence{
					{Owner: "docs/index.md", Score: 7, Reason: "title match"},
					{Owner: "docs/arch.md", Score: 5, Reason: "section match"},
				},
				watermark: "gabc",
			},
			fakeAdapter{
				name:     "code",
				relevant: true,
				hits: []Evidence{
					{Owner: "internal/router.go", Score: 9, Reason: "symbol match"},
				},
				watermark: "rev1",
			},
			fakeAdapter{
				name:     "sessions",
				relevant: false,
			},
		},
	}
	skip := map[string]bool{"sessions": true}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := p.Run("discovery routing query", 5, skip)
		if len(r.Results) == 0 {
			b.Fatal("unexpected empty results")
		}
	}
}
