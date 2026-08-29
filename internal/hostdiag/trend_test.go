package hostdiag

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func trendLine(id string, at time.Time, event, app string) string {
	data, _ := json.Marshal(map[string]any{"schema": CorrelationSchema, "correlation_id": id, "time_ms": at.UnixMilli(), "event_name": event, "app": app})
	return string(data)
}

func TestSummarizeTrendMixedDedupeHalfOpenMetrics(t *testing.T) {
	now := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	recentStart := now.Add(-48 * time.Hour)
	baseStart := recentStart.Add(-48 * time.Hour)
	lines := []string{
		`{"schema":"foreign/1","secret":"must be ignored"}`,
		trendLine("before", baseStart.Add(-time.Millisecond), "APP_CRASH", "old.exe"),
		trendLine("b0", baseStart, "APP_CRASH", `C:\\Users\\private\\Fak.EXE`),
		trendLine("boundary", recentStart, "APP_HANG", "fak.exe"),
		trendLine("r2", recentStart.Add(25*time.Hour), "RADAR_PRE_LEAK_64", "z.exe"),
		trendLine("r2", recentStart.Add(26*time.Hour), "APP_CRASH", "duplicate.exe"),
		trendLine("end", now, "APP_CRASH", "future.exe"),
	}
	got, err := SummarizeTrend(strings.NewReader(strings.Join(lines, "\n")), now, 48*time.Hour, 48*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if got.Recent.Total != 2 || got.Recent.Crash != 0 || got.Recent.Hang != 2 || got.Recent.ActiveUTCDays != 2 {
		t.Fatalf("recent=%+v", got.Recent)
	}
	if got.Baseline.Total != 1 || got.Baseline.Crash != 1 || got.Baseline.Hang != 0 {
		t.Fatalf("baseline=%+v", got.Baseline)
	}
	if len(got.Recent.PerDay) != 2 || got.Recent.PerDay[0].Date >= got.Recent.PerDay[1].Date {
		t.Fatalf("per_day=%+v", got.Recent.PerDay)
	}
	if got.Baseline.TopApps[0].Identity != "fak.exe" {
		t.Fatalf("apps=%+v", got.Baseline.TopApps)
	}
	if !got.Comparison.Observational || got.Comparison.Causal || got.Comparison.TotalDelta != 1 {
		t.Fatalf("comparison=%+v", got.Comparison)
	}
}

func TestSummarizeTrendRejectsMalformedCorrelationButIgnoresValidForeign(t *testing.T) {
	now := time.Now().UTC()
	for _, input := range []string{
		`{"schema":"` + CorrelationSchema + `","time_ms":1,"event_name":"APP_CRASH"}`,
		`{"schema":"` + CorrelationSchema + `","correlation_id":"x","time_ms":"bad","event_name":"APP_CRASH"}`,
		`{"schema":"` + CorrelationSchema + `","correlation_id":"x","time_ms":1,"event_name":"private text"}`,
		`{"schema":`,
	} {
		if _, err := SummarizeTrend(strings.NewReader(input), now, time.Hour, time.Hour); err == nil {
			t.Fatalf("accepted %q", input)
		}
	}
	if _, err := SummarizeTrend(strings.NewReader(`{"schema":"foreign/1","arbitrary":[1]}`), now, time.Hour, time.Hour); err != nil {
		t.Fatal(err)
	}
}

func TestSummarizeTrendBoundedDeterministicAndFiniteZeroBaseline(t *testing.T) {
	now := time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC)
	var lines []string
	for i := 11; i >= 0; i-- {
		lines = append(lines, trendLine(fmt.Sprintf("id-%d", i), now.Add(-time.Hour), "EVENT_"+string(rune('A'+i)), fmt.Sprintf("app-%02d.exe", i)))
	}
	got, err := SummarizeTrend(strings.NewReader(strings.Join(lines, "\n")), now, 24*time.Hour, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Recent.TopApps) != 10 || len(got.Recent.TopEvents) != 10 || got.Recent.TopApps[0].Identity != "app-00.exe" { //boundarylint:ignore CHANGE_DETECTOR_TEST the trend report contract caps both ranked lists at exactly ten entries
		t.Fatalf("tops apps=%+v events=%+v", got.Recent.TopApps, got.Recent.TopEvents)
	}
	if !got.Comparison.BaselineZero || got.Comparison.TotalRatio != 0 || got.Comparison.CrashRatio != 0 || got.Comparison.HangRatio != 0 {
		t.Fatalf("comparison=%+v", got.Comparison)
	}
	data, err := json.Marshal(got)
	if err != nil || strings.Contains(string(data), "Inf") || strings.Contains(string(data), "NaN") {
		t.Fatalf("json=%s err=%v", data, err)
	}
}
