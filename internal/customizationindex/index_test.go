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

func TestCheckFailClosedInvariants(t *testing.T) {
	asOf := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)

	cases := []struct {
		name      string
		mutate    func(idx *Index)
		wantError string
	}{
		{
			name: "schema_mismatch",
			mutate: func(idx *Index) {
				idx.Schema = "fak-agent-customization-index/99"
			},
			wantError: `schema "fak-agent-customization-index/99", want "fak-agent-customization-index/1"`,
		},
		{
			name: "non_positive_review_interval",
			mutate: func(idx *Index) {
				idx.Maintenance.ReviewIntervalDays = 0
			},
			wantError: "maintenance.review_interval_days must be positive",
		},
		{
			name: "empty_axis_id",
			mutate: func(idx *Index) {
				idx.Axes[0].ID = "   "
			},
			wantError: "axis requires axis_id and user_need",
		},
		{
			name: "empty_user_need",
			mutate: func(idx *Index) {
				idx.Axes[0].UserNeed = ""
			},
			wantError: "axis requires axis_id and user_need",
		},
		{
			name: "unknown_layer",
			mutate: func(idx *Index) {
				idx.Axes[0].Layer = "nonexistent_layer"
			},
			wantError: "references unknown layer",
		},
		{
			name: "duplicate_layer",
			mutate: func(idx *Index) {
				idx.Layers = append(idx.Layers, Layer{ID: idx.Layers[0].ID})
			},
			wantError: "duplicate layer",
		},
		{
			name: "duplicate_source",
			mutate: func(idx *Index) {
				idx.Sources = append(idx.Sources, Source{ID: idx.Sources[0].ID, ObservedAt: "2026-08-10"})
			},
			wantError: "duplicate source",
		},
		{
			name: "invalid_disposition",
			mutate: func(idx *Index) {
				idx.Axes[0].Disposition = "unsupported_disposition"
			},
			wantError: "has invalid disposition",
		},
		{
			name: "missing_evidence",
			mutate: func(idx *Index) {
				idx.Axes[0].Evidence = nil
			},
			wantError: "requires evidence",
		},
		{
			name: "invalid_observed_at",
			mutate: func(idx *Index) {
				idx.Sources[0].ObservedAt = "not-a-date"
			},
			wantError: "has invalid observed_at",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			idxCopy, err := Read(strings.NewReader(fixture))
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(&idxCopy)
			report := Check(idxCopy, asOf)
			if report.Valid {
				t.Fatalf("expected report to be invalid for %s", tc.name)
			}
			joined := strings.Join(report.Errors, "\n")
			if !strings.Contains(joined, tc.wantError) {
				t.Fatalf("missing expected error %q in:\n%s", tc.wantError, joined)
			}
		})
	}
}

func TestCheckDeterminism(t *testing.T) {
	index, err := Read(strings.NewReader(fixture))
	if err != nil {
		t.Fatal(err)
	}
	asOf := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	first := Check(index, asOf)

	for i := 0; i < 5; i++ {
		repeat := Check(index, asOf)
		if repeat.Valid != first.Valid || repeat.DueSources != first.DueSources || repeat.Axes != first.Axes {
			t.Fatalf("nondeterministic summary: got %+v, want %+v", repeat, first)
		}
		if len(repeat.Sources) != len(first.Sources) {
			t.Fatalf("nondeterministic sources length")
		}
		for j := range repeat.Sources {
			if repeat.Sources[j] != first.Sources[j] {
				t.Fatalf("nondeterministic source at index %d: %+v vs %+v", j, repeat.Sources[j], first.Sources[j])
			}
		}
		if len(repeat.Groups) != len(first.Groups) {
			t.Fatalf("nondeterministic groups length")
		}
		for j := range repeat.Groups {
			if repeat.Groups[j] != first.Groups[j] {
				t.Fatalf("nondeterministic group at index %d: %+v vs %+v", j, repeat.Groups[j], first.Groups[j])
			}
		}
	}
}

func TestParseAsOf(t *testing.T) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.FixedZone("TEST", -4*3600))
	fallback, err := ParseAsOf("", now)
	if err != nil {
		t.Fatalf("unexpected error on empty asOf: %v", err)
	}
	if fallback != now.UTC() {
		t.Fatalf("got %v, want UTC fallback %v", fallback, now.UTC())
	}

	parsed, err := ParseAsOf("2026-08-17", now)
	if err != nil {
		t.Fatalf("unexpected error on valid date: %v", err)
	}
	want := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	if !parsed.Equal(want) {
		t.Fatalf("got %v, want %v", parsed, want)
	}

	_, err = ParseAsOf("invalid-date", now)
	if err == nil || !strings.Contains(err.Error(), "as-of must be YYYY-MM-DD") {
		t.Fatalf("expected error on malformed date, got %v", err)
	}
}
