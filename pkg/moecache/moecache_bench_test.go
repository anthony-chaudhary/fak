package moecache_test

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/moecache"
)

// sampleTelemetry builds the record a live serve emits for one report window: a
// populated ring tier and a checkpoint tier, with several axes deliberately left
// UNKNOWN so the benchmark measures the honest known/unknown wire path too.
func sampleTelemetry() moecache.ExpertCacheTelemetry {
	tel := moecache.New()
	tel.Shape = moecache.ModelShape{Experts: 384, TopK: 6, Layers: 40, ActivatedFraction: 6.0 / 384.0}
	tel.Coverage.Lookups = moecache.Known(4096)
	tel.Coverage.Refusals = moecache.Known(7)

	dram := tel.Tier(moecache.TierDRAM)
	dram.CapacityBytes = moecache.Known[int64](8 << 30)
	dram.ResidentBytes = moecache.Known[int64](6 << 30)
	dram.ResidentCount = moecache.Known(256)
	dram.EvictionCount = moecache.Known[int64](31)
	dram.HitRate = moecache.Known(0.87)
	dram.MissBytes = moecache.Known[int64](512 << 20)
	dram.HitBytes = moecache.Unknown[int64](moecache.ReasonNotDerivable)
	dram.BeladyRegret = moecache.Unknown[float64](moecache.ReasonNoReplayTrace)

	ck := tel.Tier(moecache.TierCheckpoint)
	ck.CapacityBytes = moecache.Known[int64](64 << 30)
	ck.ResidentBytes = moecache.Known[int64](40 << 30)
	ck.ResidentCount = moecache.Known(1024)
	ck.HitRate = moecache.Known(0.62)
	ck.NVMeBytes = moecache.Known[int64](2 << 30)

	return tel
}

// BenchmarkMarshalTelemetry measures the production serialize path: the live
// serve marshals one ExpertCacheTelemetry per report window, and every unknown
// axis must stay value-free on the wire. This is the operation the gateway does
// on the hot path, so it is the one worth a baseline.
func BenchmarkMarshalTelemetry(b *testing.B) {
	tel := sampleTelemetry()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blob, err := json.Marshal(tel)
		if err != nil {
			b.Fatalf("marshal telemetry: %v", err)
		}
		if len(blob) == 0 {
			b.Fatal("marshal produced no bytes")
		}
	}
}

// BenchmarkUnmarshalTelemetry measures the consumer side: the private gateway
// decodes the record and reads tier metrics back, preserving the known/unknown
// boundary. Round-tripping is what proves the boundary survives the wire.
func BenchmarkUnmarshalTelemetry(b *testing.B) {
	tel := sampleTelemetry()
	blob, err := json.Marshal(tel)
	if err != nil {
		b.Fatalf("seed marshal: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var back moecache.ExpertCacheTelemetry
		if err := json.Unmarshal(blob, &back); err != nil {
			b.Fatalf("unmarshal telemetry: %v", err)
		}
		dram, ok := back.Find(moecache.TierDRAM)
		if !ok {
			b.Fatal("DRAM tier missing after round-trip")
		}
		if _, known := dram.HitBytes.Get(); known {
			b.Fatal("unknown axis became known after round-trip")
		}
	}
}

// BenchmarkTierLookup measures Find over the populated tier slice - the accessor
// the report renderer calls once per tier per report.
func BenchmarkTierLookup(b *testing.B) {
	tel := sampleTelemetry()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, ok := tel.Find(moecache.TierCheckpoint); !ok {
			b.Fatal("checkpoint tier missing")
		}
	}
}
