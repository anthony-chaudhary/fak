package trendreport

import (
	"testing"
)

type benchRow struct {
	Date        string `json:"date"`
	GeneratedAt string `json:"generated_at"`
	Debt        int    `json:"debt"`
}

func BenchmarkAppendLedgerLine(b *testing.B) {
	r := benchRow{
		Date:        "2026-09-04",
		GeneratedAt: "2026-09-04T12:00:00Z",
		Debt:        7,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := AppendLedgerLine(r)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkStamp(b *testing.B) {
	opts := Opts{
		Workspace:   "/work/fak",
		Commit:      "abc123def456",
		GeneratedAt: "2026-09-04T12:00:00Z",
		Date:        "2026-09-04",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Stamp("fak-cadence-report/1", opts)
	}
}

func BenchmarkAdvisoryGate(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = AdvisoryGate("CADENCE", "cadence_recorded", "cadence recorded; all green", "cadence_unmeasured")
	}
}

func BenchmarkWithGate(b *testing.B) {
	e := Stamp("fak-cadence-report/1", Opts{
		Workspace:   "/work/fak",
		Commit:      "abc123def456",
		GeneratedAt: "2026-09-04T12:00:00Z",
		Date:        "2026-09-04",
	})
	v := GateVerdict{Exit: 0, Message: "CADENCE OK: all green"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.WithGate(v)
	}
}

func BenchmarkDirectionWord(b *testing.B) {
	deltas := []int{5, -3, 0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DirectionWord(deltas[i%len(deltas)])
	}
}

func BenchmarkTriageEnvelope(b *testing.B) {
	e := Envelope{
		Finding:    "cadence_unmeasured",
		Reason:     "scores unmeasured",
		NextAction: "repair scores, then rerun `fak cadence`",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = TriageEnvelope("CADENCE", e)
	}
}

func BenchmarkAdvisoryGateTriaged(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = AdvisoryGateTriaged("CADENCE", "cadence_unmeasured", "scores unmeasured", "repair scores, then rerun `fak cadence`", "cadence_unmeasured", true)
	}
}
