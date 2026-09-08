package armtracking

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

// BenchmarkArmTracking_Record exercises reg.Record in a b.N loop across multiple arms.
func BenchmarkArmTracking_Record(b *testing.B) {
	reg := NewRegistry()
	const armCount = 200
	armIDs := make([]string, armCount)
	for j := 0; j < armCount; j++ {
		armIDs[j] = fmt.Sprintf("arm_%04d", j)
	}

	res := ArmResult{
		Workload:      "inference_qwen",
		ArmKind:       ArmKindCandidate,
		PrimaryMetric: "throughput_tok_s",
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
		Metadata: ArmMetadata{
			Dimension: "quantization",
			Feature:   "q4k_gemv",
			Hardware:  "strix_halo",
			CommitSHA: "0123456789abcdef",
		},
		Timestamp: time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res.ArmID = armIDs[i%armCount]
		res.PrimaryValue = float64(50 + (i % 100))
		if _, err := reg.Record(res, "bench record"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_RecordExistingArm exercises updating an existing arm in a b.N loop.
func BenchmarkArmTracking_RecordExistingArm(b *testing.B) {
	reg := NewRegistry()
	res := ArmResult{
		ArmID:         "cpu_reference",
		Workload:      "inference_qwen",
		ArmKind:       ArmKindBaseline,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  50.0,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
		Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
	}
	if _, err := reg.Record(res, "initial baseline"); err != nil {
		b.Fatalf("initial Record failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res.PrimaryValue = float64(50 + (i % 50))
		if _, err := reg.Record(res, "update baseline"); err != nil {
			b.Fatalf("Record update failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_Leaderboard exercises reg.Leaderboard with multiple arms in a b.N loop.
func BenchmarkArmTracking_Leaderboard(b *testing.B) {
	reg := NewRegistry()
	const armCount = 25
	for j := 0; j < armCount; j++ {
		res := ArmResult{
			ArmID:         fmt.Sprintf("arm_%02d", j),
			Workload:      "inference_qwen",
			ArmKind:       ArmKindSweep,
			PrimaryMetric: "throughput_tok_s",
			PrimaryValue:  float64(40 + j*3),
			PrimaryUnit:   "tok/s",
			Direction:     HigherIsBetter,
			Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := reg.Record(res, "init arm"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := reg.Leaderboard("inference_qwen")
		if err != nil || len(rows) != armCount {
			b.Fatalf("Leaderboard failed: err=%v, count=%d", err, len(rows))
		}
	}
}

// BenchmarkArmTracking_CompareToNextBest exercises reg.CompareToNextBest in a b.N loop.
func BenchmarkArmTracking_CompareToNextBest(b *testing.B) {
	reg := NewRegistry()
	const armCount = 15
	armIDs := make([]string, armCount)
	for j := 0; j < armCount; j++ {
		armIDs[j] = fmt.Sprintf("arm_%02d", j)
		res := ArmResult{
			ArmID:         armIDs[j],
			Workload:      "inference_qwen",
			ArmKind:       ArmKindAblation,
			PrimaryMetric: "latency_ms",
			PrimaryValue:  float64(10 + j*2),
			PrimaryUnit:   "ms",
			Direction:     LowerIsBetter,
			Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := reg.Record(res, "init arm"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		targetArm := armIDs[i%armCount]
		cmp, err := reg.CompareToNextBest("inference_qwen", targetArm)
		if err != nil || cmp == nil {
			b.Fatalf("CompareToNextBest failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_AuditHistory exercises reg.AuditHistory in a b.N loop.
func BenchmarkArmTracking_AuditHistory(b *testing.B) {
	reg := NewRegistry()
	for j := 0; j < 5; j++ {
		armID := fmt.Sprintf("arm_%02d", j)
		for k := 0; k < 10; k++ {
			res := ArmResult{
				ArmID:         armID,
				Workload:      "inference_qwen",
				ArmKind:       ArmKindCandidate,
				PrimaryMetric: "throughput_tok_s",
				PrimaryValue:  float64(50 + k*3),
				PrimaryUnit:   "tok/s",
				Direction:     HigherIsBetter,
				Timestamp:     time.Date(2026, 9, 8, 12, k, 0, 0, time.UTC),
			}
			if _, err := reg.Record(res, fmt.Sprintf("step %d", k)); err != nil {
				b.Fatalf("Record failed: %v", err)
			}
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		events, err := reg.AuditHistory("inference_qwen", "")
		if err != nil || len(events) == 0 {
			b.Fatalf("AuditHistory failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_Validation exercises ArmResult.Validate in a b.N loop.
func BenchmarkArmTracking_Validation(b *testing.B) {
	res := ArmResult{
		ArmID:         "candidate_v1",
		Workload:      "inference_qwen",
		ArmKind:       ArmKindCandidate,
		PrimaryMetric: "throughput_tok_s",
		PrimaryValue:  142.5,
		PrimaryUnit:   "tok/s",
		Direction:     HigherIsBetter,
		Metadata: ArmMetadata{
			Dimension: "quantization",
			Feature:   "q4k_gemv",
			Hardware:  "strix_halo",
			CommitSHA: "0123456789abcdef",
			Witness:   "experiments/qwen/strix_q4k.json",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := res.Validate(); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_MarshalJSON exercises json.Marshal of a Registry in a b.N loop.
func BenchmarkArmTracking_MarshalJSON(b *testing.B) {
	reg := NewRegistry()
	for j := 0; j < 10; j++ {
		res := ArmResult{
			ArmID:         fmt.Sprintf("arm_%02d", j),
			Workload:      "inference_qwen",
			ArmKind:       ArmKindSweep,
			PrimaryMetric: "throughput_tok_s",
			PrimaryValue:  float64(50 + j*10),
			PrimaryUnit:   "tok/s",
			Direction:     HigherIsBetter,
			Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := reg.Record(res, "bench setup"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reg.mu.RLock()
		data, err := json.Marshal(reg)
		reg.mu.RUnlock()
		if err != nil || len(data) == 0 {
			b.Fatalf("json.Marshal failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_UnmarshalJSON exercises json.Unmarshal of a Registry in a b.N loop.
func BenchmarkArmTracking_UnmarshalJSON(b *testing.B) {
	reg := NewRegistry()
	for j := 0; j < 10; j++ {
		res := ArmResult{
			ArmID:         fmt.Sprintf("arm_%02d", j),
			Workload:      "inference_qwen",
			ArmKind:       ArmKindSweep,
			PrimaryMetric: "throughput_tok_s",
			PrimaryValue:  float64(50 + j*10),
			PrimaryUnit:   "tok/s",
			Direction:     HigherIsBetter,
			Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := reg.Record(res, "bench setup"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}
	data, err := json.Marshal(reg)
	if err != nil {
		b.Fatalf("json.Marshal setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var decoded Registry
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatalf("json.Unmarshal failed: %v", err)
		}
	}
}

// BenchmarkArmTracking_SaveAndLoad exercises SaveRegistry and LoadRegistry in a b.N loop.
func BenchmarkArmTracking_SaveAndLoad(b *testing.B) {
	reg := NewRegistry()
	for j := 0; j < 5; j++ {
		res := ArmResult{
			ArmID:         fmt.Sprintf("arm_%02d", j),
			Workload:      "inference_qwen",
			ArmKind:       ArmKindCandidate,
			PrimaryMetric: "throughput_tok_s",
			PrimaryValue:  float64(50 + j*10),
			PrimaryUnit:   "tok/s",
			Direction:     HigherIsBetter,
			Timestamp:     time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		}
		if _, err := reg.Record(res, "init"); err != nil {
			b.Fatalf("Record failed: %v", err)
		}
	}
	dir := b.TempDir()
	path := filepath.Join(dir, "bench-registry.json")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveRegistry(reg, path); err != nil {
			b.Fatalf("SaveRegistry failed: %v", err)
		}
		loaded, err := LoadRegistry(path)
		if err != nil || loaded == nil {
			b.Fatalf("LoadRegistry failed: %v", err)
		}
	}
}

// BenchmarkRegistry_Record exercises reg.Record in a b.N loop.
func BenchmarkRegistry_Record(b *testing.B) {
	BenchmarkArmTracking_Record(b)
}

// BenchmarkRegistry_CompareToNextBest exercises reg.CompareToNextBest in a b.N loop.
func BenchmarkRegistry_CompareToNextBest(b *testing.B) {
	BenchmarkArmTracking_CompareToNextBest(b)
}

// BenchmarkRegistry_Leaderboard exercises reg.Leaderboard in a b.N loop.
func BenchmarkRegistry_Leaderboard(b *testing.B) {
	BenchmarkArmTracking_Leaderboard(b)
}
