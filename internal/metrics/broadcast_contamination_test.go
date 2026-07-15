package metrics

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBroadcastContaminationRanksFanoutAndEmitsSeries(t *testing.T) {
	recorder := NewBroadcastContaminationRecorder(2)
	singleton := recorder.Record(BroadcastDirective{
		ID:              "local-correction",
		NegframeFlagged: true,
		Consumers:       []string{"planner"},
	})
	wide := recorder.Record(BroadcastDirective{
		ID:              "france-to-china-swap",
		NegframeFlagged: true, // routing remains intact; upstream flagged inverted content
		Consumers:       []string{"planner", "executor", "reviewer", "executor"},
	})

	if singleton.Radius != 1 || singleton.High {
		t.Fatalf("singleton=%+v", singleton)
	}
	if wide.Radius != 3 || !wide.High || wide.Radius <= singleton.Radius {
		t.Fatalf("wide=%+v singleton=%+v", wide, singleton)
	}
	if strings.Join(wide.Consumers, ",") != "executor,planner,reviewer" {
		t.Fatalf("distinct consumers=%v", wide.Consumers)
	}

	rows := recorder.Rows()
	if len(rows) != 2 || rows[1].DirectiveID != "france-to-china-swap" || !rows[1].High {
		t.Fatalf("series rows=%+v", rows)
	}
	encoded, err := json.Marshal(rows[1])
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"directive_id":"france-to-china-swap"`, `"radius":3`, `"high_blast_radius":true`} {
		if !strings.Contains(string(encoded), want) {
			t.Fatalf("series row %s missing %s", encoded, want)
		}
	}

	report := recorder.Report()
	if report.Observed != 2 || report.Low != 1 || report.High != 1 || report.MaxRadius != 3 || report.Threshold != 2 {
		t.Fatalf("report=%+v", report)
	}
	prom := report.Prometheus()
	for _, want := range []string{
		`fak_broadcast_contamination_total{severity="low"} 1`,
		`fak_broadcast_contamination_total{severity="high"} 1`,
		`fak_broadcast_contamination_max_radius 3`,
		`fak_broadcast_contamination_threshold 2`,
	} {
		if !strings.Contains(prom, want) {
			t.Fatalf("metrics missing %q: %s", want, prom)
		}
	}
}

func TestBroadcastContaminationIgnoresUnflaggedDirective(t *testing.T) {
	recorder := NewBroadcastContaminationRecorder(2)
	got := recorder.Record(BroadcastDirective{
		ID:              "ordinary-broadcast",
		NegframeFlagged: false,
		Consumers:       []string{"a", "b", "c"},
	})
	if got.Radius != 0 || got.High || got.Flagged {
		t.Fatalf("unflagged=%+v", got)
	}
	if len(recorder.Rows()) != 0 || recorder.Report().Observed != 0 {
		t.Fatalf("unflagged directive entered series: rows=%+v report=%+v", recorder.Rows(), recorder.Report())
	}
}

func TestBroadcastContaminationThresholdIsConfigurable(t *testing.T) {
	consumers := []string{"one", "two", "three"}
	if got := BlastRadius(BroadcastDirective{NegframeFlagged: true, Consumers: consumers}, 4); got.High {
		t.Fatalf("radius below configured threshold marked high: %+v", got)
	}
	if got := BlastRadius(BroadcastDirective{NegframeFlagged: true, Consumers: consumers}, 3); !got.High {
		t.Fatalf("radius at configured threshold not marked high: %+v", got)
	}
}
