// Package issuecost benchmarks measure throughput, allocation profiles, and
// scaling behavior for issue cost aggregations, JSONL parsing, and C9 tier
// calibration folds across production operating envelopes (10 to 1,000 rows).
//
// The benchmark topology models high-throughput dispatch telemetry from worker
// runs. Benchmarks require linear scaling for ledger parsing and join folds,
// sub-millisecond nearest-rank percentile resolution, and bounded allocation
// overhead to prevent GC pressure during batch processing.
package issuecost

import (
	"fmt"
	"testing"
)

// Package-level sinks prevent compiler dead-code elimination and ensure
// benchmarked operations are fully evaluated during b.N measurement loops.
var (
	benchSinkReport            Report
	benchSinkFloat             float64
	benchSinkString            string
	benchSinkRows              []IssueCost
	benchSinkBytes             []byte
	benchSinkCalibrationReport CalibrationReport
	benchSinkRecommendations   []Recommendation
)

// makeBenchLedger synthesizes an issue cost ledger with n records across
// realistic elapsed times, cyclic retry attempts, and mixed terminal outcomes
// to benchmark sort order and percentile calculation topologies.
// Operating envelope: n records ranging from 10 to 1,000 issues.
func makeBenchLedger(n int) []IssueCost {
	outcomes := []Outcome{Shipped, Blocked, Abandoned}
	rows := make([]IssueCost, n)
	for i := 0; i < n; i++ {
		rows[i] = IssueCost{
			Issue:      n - i,
			ElapsedSec: float64(10 + (i%20)*5),
			Attempts:   (i % 3) + 1,
			Outcome:    outcomes[i%len(outcomes)],
		}
	}
	return rows
}

// makeBenchCalibrationData constructs paired tier decisions and witnessed
// outcomes across 5 outcome classes to evaluate C9 join topology and
// recommendation generation under balanced multi-tier distributions.
// Operating envelope: n records spanning decisions, escalations, and reverts.
func makeBenchCalibrationData(n int) ([]TierDecision, []WitnessedOutcome) {
	tiers := []Tier{TierT0, TierT1, TierT2}
	decisions := make([]TierDecision, n)
	outcomes := make([]WitnessedOutcome, n)
	for i := 0; i < n; i++ {
		issue := i + 1
		chosen := tiers[i%len(tiers)]
		optimal := tiers[(i+1)%len(tiers)]
		decisions[i] = TierDecision{
			Issue:    issue,
			Chosen:   chosen,
			Required: optimal,
			Optimal:  optimal,
		}
		switch i % 5 {
		case 0:
			outcomes[i] = WitnessedOutcome{
				Issue:           issue,
				CommitWitnessed: true,
				TestsGreen:      true,
				Closed:          true,
				Turns:           3,
			}
		case 1:
			outcomes[i] = WitnessedOutcome{
				Issue:     issue,
				Escalated: true,
				Turns:     5,
			}
		case 2:
			outcomes[i] = WitnessedOutcome{
				Issue:           issue,
				CommitWitnessed: true,
				TestsGreen:      true,
				Closed:          true,
				Reverted:        true,
				Turns:           4,
			}
		case 3:
			outcomes[i] = WitnessedOutcome{
				Issue:   issue,
				Refused: true,
				Turns:   1,
			}
		case 4:
			outcomes[i] = WitnessedOutcome{
				Issue:           issue,
				Closed:          true,
				CommitWitnessed: false,
				TestsGreen:      true,
				Turns:           2,
			}
		}
	}
	return decisions, outcomes
}

// BenchmarkSummary benchmarks the primary cost summary fold (BenchmarkCostSummary)
// across 10, 100, and 1,000 issue rows. Operating envelope: 10-1,000 ledger rows.
// Allocation target: <= 17 KB/op and <= 6 allocs/op at 1,000 rows. Scaling
// ceiling: O(N log N) bounded by internal nearest-rank percentile sort passes.
func BenchmarkSummary(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := makeBenchLedger(count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkReport = Summary(rows)
			}
		})
	}
}

// BenchmarkMedian measures nearest-rank median extraction over a 100-row
// ledger. Operating envelope: 100 rows. Allocation target: <= 2 KB/op and
// <= 4 allocs/op. Scaling ceiling: O(N log N) dominated by elapsed time sorting.
// Latency budget: sub-2 microsecond resolution under standard CPU execution.
func BenchmarkMedian(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFloat = Median(rows)
	}
}

// BenchmarkP95 measures nearest-rank 95th percentile latency calculation.
// Operating envelope: 100 rows. Allocation target: <= 2 KB/op and
// <= 4 allocs/op. Scaling ceiling: O(N log N) dominated by rank selection.
// Latency budget: sub-2 microsecond resolution matching fleetmetrics nearest-rank.
func BenchmarkP95(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFloat = P95(rows)
	}
}

// BenchmarkReportRender measures string rendering of a computed cost Report.
// Operating envelope: 100-row summary report. Allocation target: <= 128 B/op and
// <= 3 allocs/op. Scaling ceiling: O(1) constant-format buffer construction.
// Latency budget: sub-microsecond formatting throughput.
func BenchmarkReportRender(b *testing.B) {
	rep := Summary(makeBenchLedger(100))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = rep.Render()
	}
}

// BenchmarkSortedByIssue measures ascending sorting of issue records by issue ID.
// Operating envelope: 100 rows. Allocation target: <= 5 KB/op and <= 4 allocs/op
// for slice duplication. Scaling ceiling: O(N log N) via sort.SliceStable.
// Memory topology: preserves input slice immutability via defensive copy.
func BenchmarkSortedByIssue(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRows = SortedByIssue(rows)
	}
}

// BenchmarkAppendRow measures single-row JSONL byte buffer serialization.
// Operating envelope: 1 record. Allocation target: <= 208 B/op and
// <= 3 allocs/op. Scaling ceiling: O(1) per row without persistent allocations.
// Throughput target: zero allocations when appending into an adequately sized preallocated buffer.
func BenchmarkAppendRow(b *testing.B) {
	row := IssueCost{
		Issue:      42,
		ElapsedSec: 123.45,
		Attempts:   2,
		Outcome:    Shipped,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := AppendRow(nil, row)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

// BenchmarkParseLedger evaluates JSONL parsing across 10, 100, and 1,000 rows.
// Operating envelope: 10-1,000 rows (up to ~430 KB). Allocation target: linear
// with row count (~6 allocs/row). Scaling ceiling: O(N) throughput >= 20 MB/s.
// Error behavior: validates syntax and outcome enum validation on each line.
func BenchmarkParseLedger(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_rows", count), func(b *testing.B) {
			rows := makeBenchLedger(count)
			var buf []byte
			for _, r := range rows {
				var err error
				buf, err = AppendRow(buf, r)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.SetBytes(int64(len(buf)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				parsed, err := ParseLedger(buf)
				if err != nil {
					b.Fatal(err)
				}
				benchSinkRows = parsed
			}
		})
	}
}

// BenchmarkCalibrate measures C9 tier-outcome join and recommendation synthesis.
// Operating envelope: 10-1,000 decision-outcome pairs. Allocation target:
// <= 85 KB/op and <= 26 allocs/op at 1,000 pairs. Scaling ceiling: O(N) linear
// hash join and single-pass bucket aggregation.
// Data topology: joins disparate decision and outcome streams by issue ID.
func BenchmarkCalibrate(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_pairs", count), func(b *testing.B) {
			decisions, outcomes := makeBenchCalibrationData(count)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkCalibrationReport = Calibrate(decisions, outcomes)
			}
		})
	}
}

// BenchmarkCalibrationRender evaluates textual rendering of CalibrationReport.
// Operating envelope: 100 joined pairs. Allocation target: <= 5 KB/op and
// <= 42 allocs/op. Scaling ceiling: O(T + B) for T tiers and B outcome buckets.
// Formatting profile: builds multi-section diagnostic text with zero reflection.
func BenchmarkCalibrationRender(b *testing.B) {
	decisions, outcomes := makeBenchCalibrationData(100)
	rep := Calibrate(decisions, outcomes)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = rep.Render()
	}
}

// BenchmarkSortedRecommendations benchmarks recommendation sorting (BenchmarkRecommend)
// by tier precedence. Operating envelope: 3-tier advisory slice. Allocation target:
// <= 376 B/op and <= 4 allocs/op. Scaling ceiling: O(K log K) for K <= 5 tiers.
// Stability: deterministic tie-breaking by action ordinal.
func BenchmarkSortedRecommendations(b *testing.B) {
	recs := []Recommendation{
		{Tier: TierT2, Action: ActionHold},
		{Tier: TierT0, Action: ActionExpandCheaper},
		{Tier: TierT1, Action: ActionRaiseFloor},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRecommendations = SortedRecommendations(recs)
	}
}
