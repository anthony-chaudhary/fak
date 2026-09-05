package issuecost

import (
	"fmt"
	"testing"
)

var (
	benchSinkReport            Report
	benchSinkFloat             float64
	benchSinkString            string
	benchSinkRows              []IssueCost
	benchSinkBytes             []byte
	benchSinkCalibrationReport CalibrationReport
	benchSinkRecommendations   []Recommendation
)

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

func BenchmarkMedian(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFloat = Median(rows)
	}
}

func BenchmarkP95(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFloat = P95(rows)
	}
}

func BenchmarkReportRender(b *testing.B) {
	rep := Summary(makeBenchLedger(100))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = rep.Render()
	}
}

func BenchmarkSortedByIssue(b *testing.B) {
	rows := makeBenchLedger(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRows = SortedByIssue(rows)
	}
}

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

func BenchmarkCalibrationRender(b *testing.B) {
	decisions, outcomes := makeBenchCalibrationData(100)
	rep := Calibrate(decisions, outcomes)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = rep.Render()
	}
}

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
