package geminicache

import (
	"testing"
	"time"
)

// BenchmarkGeminiCache benchmarks identity derivation, admission checking, and reference validation.
func BenchmarkGeminiCache(b *testing.B) {
	prefix := []byte("stable agent instructions and tool context for prompt caching")
	model := "models/gemini-2.5-flash"
	account := "acct-bench"
	project := "project-bench"
	location := "us-central1"
	id := NewIdentity(account, project, location, model, prefix)
	now := time.Date(2026, 8, 21, 16, 0, 0, 0, time.UTC)
	receipt := Receipt{
		Schema:   ProvenanceSchema,
		Identity: id,
		State:    StateActive,
		Object: CachedContent{
			Name:       "cachedContents/cache-bench",
			Model:      model,
			ExpireTime: now.Add(2 * time.Hour),
			UsageMetadata: &UsageMetadata{
				TotalTokenCount: 1250,
			},
		},
		Observed: now,
	}
	admission := Admission{
		PredictedReuseValueUSD: 3.50,
		CreationStorageCostUSD: 1.00,
		TTL:                    time.Hour,
		MaxTTL:                 4 * time.Hour,
		PrivacyAllowed:         true,
	}

	b.Run("NewIdentity", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			got := NewIdentity(account, project, location, model, prefix)
			if got.PrefixDigest == "" {
				b.Fatal("missing digest")
			}
		}
	})

	b.Run("AdmissionCheck", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := admission.Check(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Reference", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			name, err := Reference(RouteGenerateContent, id, receipt, prefix, now)
			if err != nil || name == "" {
				b.Fatalf("reference failed: %v", err)
			}
		}
	})

	b.Run("ParallelReference", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				name, err := Reference(RouteGenerateContent, id, receipt, prefix, now)
				if err != nil || name == "" {
					b.Fatalf("parallel reference failed: %v", err)
				}
			}
		})
	})
}
