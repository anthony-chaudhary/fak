package sessionjournal

import (
	"strings"
	"testing"
)

func TestParseEventsReportCountsContentFreeDegradation(t *testing.T) {
	valid := `{"schema":"fak.sessionjournal.v1","kind":"open","id":"ok","ts":"2026-08-18T00:00:00Z"}`
	input := valid + "\n" + `{not-json` + "\n" + `{"schema":"future/2","id":"wrong"}` + "\n" + `{"schema":"fak.sessionjournal.v1","kind":"open"}` + "\n\n"
	events, health := ParseEventsReport(input)
	if len(events) != 1 || health.AcceptedRows != 1 || health.MalformedRows != 1 || health.WrongSchemaRows != 1 || health.MissingIDRows != 1 || health.BlankRows != 1 || !health.Degraded() {
		t.Fatalf("events=%+v health=%+v", events, health)
	}
	b := strings.ToLower(health.ScanError + health.ReadError)
	if strings.Contains(b, "ok") || strings.Contains(b, "not-json") {
		t.Fatalf("health retained content: %+v", health)
	}
}

func TestLoadFileReportClassifiesMissingWithoutPath(t *testing.T) {
	_, health := LoadFileReport(t.TempDir() + "/private-name.jsonl")
	if health.ReadError != "not_found" || !health.Degraded() {
		t.Fatalf("health=%+v", health)
	}
}
