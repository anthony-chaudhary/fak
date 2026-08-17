package customizationindex

import (
	"strings"
	"testing"
	"time"
)

const fixture = `{
  "schema":"fak-agent-customization-index/1",
  "updated_at":"2026-08-17",
  "maintenance":{"review_interval_days":30,"status_values":["present","partial","absent"],"disposition_values":["default","watch"]},
  "layers":[{"id":"authoring"},{"id":"presentation"}],
  "sources":[{"id":"fresh","observed_at":"2026-08-10","checked_revision":"abc"},{"id":"stale","observed_at":"2026-06-01","checked_revision":"def"}],
  "axes":[{"axis_id":"instructions","layer":"authoring","user_need":"configure","evidence":["fresh"],"fak_status":"present","disposition":"default"},{"axis_id":"views","layer":"presentation","user_need":"show it","evidence":["stale"],"fak_status":"absent","disposition":"watch"}]
}`

func TestCheckReportsFreshnessAndGroups(t *testing.T) {
	index, err := Read(strings.NewReader(fixture))
	if err != nil {
		t.Fatal(err)
	}
	report := Check(index, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
	if !report.Valid || report.DueSources != 1 || report.Axes != 2 || len(report.Groups) != 2 {
		t.Fatalf("report=%+v", report)
	}
	if report.Sources[0].ID != "fresh" || report.Sources[0].Due || report.Sources[1].ID != "stale" || !report.Sources[1].Due {
		t.Fatalf("sources=%+v", report.Sources)
	}
}

func TestCheckFindsContractFailures(t *testing.T) {
	broken := strings.Replace(fixture, `"axis_id":"views"`, `"axis_id":"instructions"`, 1)
	broken = strings.Replace(broken, `"evidence":["stale"]`, `"evidence":["missing"]`, 1)
	broken = strings.Replace(broken, `"fak_status":"absent"`, `"fak_status":"maybe"`, 1)
	index, err := Read(strings.NewReader(broken))
	if err != nil {
		t.Fatal(err)
	}
	report := Check(index, time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC))
	joined := strings.Join(report.Errors, "\n")
	for _, want := range []string{"duplicate axis", "unknown source", "invalid status"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in %s", want, joined)
		}
	}
	if report.Valid {
		t.Fatal("broken index reported valid")
	}
}

func TestReadAcceptsAuthoritativeIndexMetadata(t *testing.T) {
	input := `{"schema":"fak-agent-customization-index/1","updated_at":"2026-08-17","scope":"full field map","maintenance":{"review_interval_days":30,"dedupe_key":"axis_id","source_identity":"repository@revision","required_refresh_fields":["updated_at"],"status_values":["present"],"disposition_values":["default"]},"layers":[{"id":"authoring","question":"what is assembled?"}],"sources":[{"id":"source","kind":"repository","url":"https://example.test/repo","observed_at":"2026-08-17","checked_revision":"abc","license":"MIT","evidence":["settings"]}],"axes":[{"axis_id":"instructions","layer":"authoring","user_need":"configure","examples":["rules"],"evidence":["source"],"fak_status":"present","disposition":"default"}]}`
	index, err := Read(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	if index.Scope != "full field map" || index.Layers[0].Question == "" || index.Sources[0].URL == "" || len(index.Axes[0].Examples) != 1 {
		t.Fatalf("metadata lost: %+v", index)
	}
}

func TestReadRejectsUnknownShape(t *testing.T) {
	_, err := Read(strings.NewReader(strings.Replace(fixture, `"updated_at"`, `"surprise"`, 1)))
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("err=%v", err)
	}
}
