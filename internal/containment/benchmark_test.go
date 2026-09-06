package containment

import (
	"strconv"
	"testing"
)

var (
	benchInt64Sink  int64
	benchRecordSink APUAllocationRecord
	benchBoolSink   bool
)

func BenchmarkModelMetadataEstimatedMemory(b *testing.B) {
	meta := ModelMetadata{
		ModelName:       "qwen3-30b-q4k",
		ParameterCount:  30_000_000_000,
		WeightBytes:     20 * GiB,
		ContextTokens:   32768,
		KVBytesPerToken: 65536,
		ScratchpadBytes: 2 * GiB,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchInt64Sink = meta.EstimatedMemoryBytes()
	}
}

func BenchmarkEstimateAPUResidentMemory(b *testing.B) {
	meta := ModelMetadata{
		ModelName:       "qwen3-30b-q4k",
		ParameterCount:  30_000_000_000,
		WeightBytes:     20 * GiB,
		ContextTokens:   32768,
		KVBytesPerToken: 65536,
		ScratchpadBytes: 2 * GiB,
	}

	b.Run("GTTBypass", func(b *testing.B) {
		cgroup := int64(2 * GiB)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInt64Sink = EstimateAPUResidentMemory(cgroup, meta, true)
		}
	})

	b.Run("AccurateCgroup", func(b *testing.B) {
		cgroup := int64(30 * GiB)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInt64Sink = EstimateAPUResidentMemory(cgroup, meta, true)
		}
	})

	b.Run("NonAPU", func(b *testing.B) {
		cgroup := int64(2 * GiB)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchInt64Sink = EstimateAPUResidentMemory(cgroup, meta, false)
		}
	})
}

func BenchmarkEvaluateAPUAllocation(b *testing.B) {
	meta := ModelMetadata{
		ModelName:       "deepseek-lite",
		ParameterCount:  16_000_000_000,
		WeightBytes:     10 * GiB,
		ContextTokens:   16384,
		KVBytesPerToken: 32768,
		ScratchpadBytes: 1 * GiB,
	}

	b.Run("Bypass", func(b *testing.B) {
		cgroup := int64(3 * GiB)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRecordSink = EvaluateAPUAllocation("instance-worker-1", cgroup, meta, true)
		}
	})

	b.Run("Direct", func(b *testing.B) {
		cgroup := int64(14 * GiB)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRecordSink = EvaluateAPUAllocation("instance-worker-1", cgroup, meta, true)
		}
	})
}

func BenchmarkArbitrateAPUHeadroom(b *testing.B) {
	aperture := int64(120 * GiB)

	for _, count := range []int{2, 8, 32} {
		records := make([]APUAllocationRecord, count)
		for j := 0; j < count; j++ {
			meta := ModelMetadata{WeightBytes: int64((j + 1) * GiB)}
			records[j] = EvaluateAPUAllocation("inst-"+strconv.Itoa(j), int64(j*GiB), meta, true)
		}

		b.Run(strconv.Itoa(count)+"Instances", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				consumed, free, hasCap := ArbitrateAPUHeadroom(aperture, records)
				benchInt64Sink = consumed + free
				benchBoolSink = hasCap
			}
		})
	}
}
