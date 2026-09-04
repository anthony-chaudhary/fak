package geminicache

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func BenchmarkNewIdentity(b *testing.B) {
	prefix := []byte("system instructions and large context prompt prefix for gemini cache")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := NewIdentity("acct-1", "proj-1", "us-central1", "models/gemini-2.5-flash", prefix)
		if id.Account == "" {
			b.Fatal("empty account")
		}
	}
}

func BenchmarkAdmissionCheck(b *testing.B) {
	adm := Admission{
		PredictedReuseValueUSD: 2.5,
		CreationStorageCostUSD: 1.0,
		TTL:                    time.Hour,
		MaxTTL:                 4 * time.Hour,
		PrivacyAllowed:         true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := adm.Check(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReferenceResolution(b *testing.B) {
	now := time.Date(2026, 8, 21, 16, 0, 0, 0, time.UTC)
	model := "models/gemini-2.5-flash"
	prefix := []byte("stable agent instructions and tool context")
	id := NewIdentity("acct-a", "project-a", "us-central1", model, prefix)
	receipt := Receipt{
		Schema:   ProvenanceSchema,
		Identity: id,
		Object: CachedContent{
			Name:       "cachedContents/cache-12345",
			Model:      model,
			ExpireTime: now.Add(time.Hour),
		},
		State:    StateActive,
		Observed: now,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := Reference(RouteGenerateContent, id, receipt, prefix, now)
		if err != nil || ref == "" {
			b.Fatalf("Reference failed: %v", err)
		}
	}
}

func BenchmarkMockClientRoundTrip(b *testing.B) {
	now := time.Date(2026, 8, 21, 16, 0, 0, 0, time.UTC)
	model := "models/gemini-2.5-flash"
	prefix := []byte("agent instructions")
	id := NewIdentity("acct-a", "project-a", "us-central1", model, prefix)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(CachedContent{
			Name:          "cachedContents/cache-bench",
			Model:         model,
			ExpireTime:    now.Add(time.Hour),
			UsageMetadata: &UsageMetadata{TotalTokenCount: 1500},
		})
	}))
	defer srv.Close()

	client := Client{
		BaseURL:    srv.URL + "/v1beta",
		HTTPClient: srv.Client(),
		Now:        func() time.Time { return now },
		Capabilities: Capabilities{
			GenerateContent: true,
			Models:          map[string]bool{model: true},
			Locations:       map[string]bool{"us-central1": true},
		},
	}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec, err := client.Get(ctx, id, "cachedContents/cache-bench")
		if err != nil || rec.State != StateActive {
			b.Fatalf("client.Get failed: %v", err)
		}
	}
}
