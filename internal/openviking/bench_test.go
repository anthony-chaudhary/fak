package openviking

import (
	"encoding/json"
	"testing"
)

func BenchmarkOpenViking(b *testing.B) {
	b.Run("ValidateBaseURL", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := validateBaseURL("https://api.openviking.example"); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ValidateSessionID", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := validateSessionID("session-token-valid-123"); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("MarshalSearchContextRequest", func(b *testing.B) {
		req := SearchContextRequest{
			Query:     "benchmark query execution",
			SessionID: "session-42",
			Limit:     10,
			MaxTokens: 2048,
			Purpose:   "benchmarking",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			data, err := json.Marshal(req)
			if err != nil || len(data) == 0 {
				b.Fatal(err)
			}
		}
	})

	b.Run("ParseResponseEnvelope", func(b *testing.B) {
		raw := []byte(`{"status":"ok","result":{"session_id":"s-1","message_count":10,"added":1,"pending_tokens":42},"telemetry":{"latency_ms":12},"profile":["cache_hit=true"]}`)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var env responseEnvelope
			if err := json.Unmarshal(raw, &env); err != nil {
				b.Fatal(err)
			}
		}
	})
}
