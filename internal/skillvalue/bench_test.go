package skillvalue

import (
	"fmt"
	"strings"
	"testing"
)

var (
	benchSinkRollup     Rollup
	benchSinkRows       []SessionRow
	benchSinkGateReport GateReport
	benchSinkIDs        []string
	benchSinkBool       bool
)

func makeBenchSessions(count int) ([]SessionRow, map[string]string) {
	rows := make([]SessionRow, count)
	basis := make(map[string]string)
	for i := 0; i < count; i++ {
		taskClass := fmt.Sprintf("task_class_%d", i%10)
		skill1 := fmt.Sprintf("skill_%d", i%15)
		skill2 := fmt.Sprintf("skill_%d", (i+3)%15)
		basis[skill1] = "ablation:matched-pass-delta"
		if i%5 != 0 {
			basis[skill2] = "ablation:matched-pass-delta"
		}
		skills := []string{skill1}
		if i%2 == 0 {
			skills = append(skills, skill2)
		}
		rows[i] = SessionRow{
			Schema:    LedgerSchema,
			SessionID: fmt.Sprintf("sess_%d", i),
			TaskClass: taskClass,
			Skills:    skills,
			Pass:      i%3 != 0,
			CostUSD:   float64(i%10) * 0.25,
			LatencyMS: float64(i%100) * 15.0,
		}
	}
	return rows, basis
}

func BenchmarkCompute(b *testing.B) {
	sessions, basis := makeBenchSessions(100)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkRollup = Compute(sessions, basis)
	}
}

func BenchmarkComputeLarge(b *testing.B) {
	sessions, basis := makeBenchSessions(1000)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkRollup = Compute(sessions, basis)
	}
}

func BenchmarkParseLedger(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 100; i++ {
		switch i % 4 {
		case 0:
			sb.WriteString(fmt.Sprintf(`{"schema":"%s","session_id":"s%d","task_class":"cls%d","skills":["k%d"],"pass":true,"cost_usd":1.2,"latency_ms":150}`+"\n", LedgerSchema, i, i%5, i%10))
		case 1:
			sb.WriteString(fmt.Sprintf(`{"schema":"other-schema","session_id":"s%d","task_class":"cls%d"}`+"\n", i, i%5))
		case 2:
			sb.WriteString("not-json-line\n")
		case 3:
			sb.WriteString(fmt.Sprintf(`{"schema":"%s","session_id":"s%d","task_class":"cls%d","skills":[],"pass":false,"cost_usd":0.5,"latency_ms":80}`+"\n", LedgerSchema, i, i%5))
		}
	}
	content := sb.String()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkRows = ParseLedger(content)
	}
}

func BenchmarkRollupGate(b *testing.B) {
	sessions, basis := makeBenchSessions(200)
	rollup := Compute(sessions, basis)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkGateReport = rollup.Gate()
	}
}

func BenchmarkRollupAutoRevert(b *testing.B) {
	sessions, basis := makeBenchSessions(200)
	rollup := Compute(sessions, basis)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkIDs = rollup.AutoRevert()
	}
}

func BenchmarkSkillValueHasFlag(b *testing.B) {
	sv := SkillValue{
		SkillID: "skill_test",
		Flags:   []string{FlagInsufficientEvidence, FlagNetNegative, FlagNoValuationBasis},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchSinkBool = sv.HasFlag(FlagNoValuationBasis)
	}
}
