package scorecardpane

import (
	"reflect"
	"testing"
	"time"
)

func TestCanCarry(t *testing.T) {
	base := &Baseline{
		Commit:       "base123",
		TotalDebt:    8,
		Metrics:      map[string]int{"a": 3, "b": 5},
		GradeWeights: map[string]int{"a": 1, "b": 2},
	}
	cardA := Card{Key: "a", Debt: "a_debt", Corpus: []string{"docs/"}}
	cardNoCorpus := Card{Key: "a", Debt: "a_debt"} // never carried

	cases := []struct {
		name    string
		card    Card
		changed []string
		base    *Baseline
		want    bool
	}{
		{"untouched corpus carries", cardA, []string{"src/x.go"}, base, true},
		{"empty changed carries", cardA, nil, base, true},
		{"touched corpus measures", cardA, []string{"docs/readme.md"}, base, false},
		{"no corpus never carries", cardNoCorpus, nil, base, false},
		{"nil baseline never carries", cardA, nil, nil, false},
		{"missing baseline debt measures", Card{Key: "z", Corpus: []string{"docs/"}}, nil, base, false},
		{"missing baseline weight measures",
			Card{Key: "w", Corpus: []string{"docs/"}},
			nil,
			&Baseline{Metrics: map[string]int{"w": 1}, GradeWeights: map[string]int{}},
			false},
		{"unmappable weight measures",
			Card{Key: "w", Corpus: []string{"docs/"}},
			nil,
			&Baseline{Metrics: map[string]int{"w": 1}, GradeWeights: map[string]int{"w": 3}}, // 3 is not a letter weight
			false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CanCarry(tc.card, tc.changed, tc.base); got != tc.want {
				t.Fatalf("CanCarry=%v want %v", got, tc.want)
			}
		})
	}
}

// A carried metric must reproduce its baseline (debt, grade_weight) contribution
// exactly through the UNCHANGED fold — this is the load-bearing correctness property.
func TestCarriedMetricReproducesBaselineContribution(t *testing.T) {
	base := &Baseline{
		Metrics:      map[string]int{"a": 3},
		GradeWeights: map[string]int{"a": 2}, // weight 2 -> grade C
	}
	card := Card{Key: "a", Debt: "a_debt", Label: "alpha", Corpus: []string{"docs/"}}
	m := carriedMetric(card, base)
	if m.Debt == nil || *m.Debt != 3 {
		t.Fatalf("carried debt=%v want 3", m.Debt)
	}
	if m.Grade == nil || *m.Grade != "C" {
		t.Fatalf("carried grade=%v want C (weight 2)", m.Grade)
	}
	if !m.Carried || m.Verdict != "CARRIED" {
		t.Fatalf("carried metric not flagged: carried=%v verdict=%q", m.Carried, m.Verdict)
	}
	p := Fold([]Metric{m}, base, "/repo", "head")
	if p.TotalDebt != 3 {
		t.Fatalf("fold total_debt=%d want 3 (carried debt)", p.TotalDebt)
	}
	if p.GradeDebt != 2 {
		t.Fatalf("fold grade_debt=%d want 2 (weight of C)", p.GradeDebt)
	}
	if p.Metrics[0].EffGrade != "C" {
		t.Fatalf("carried eff_grade=%q want C", p.Metrics[0].EffGrade)
	}
}

// CollectSince must carry corpus-disjoint cards (no subprocess) and measure the rest,
// preserving canonical order. Uses missing-script cards so the "measure" path returns a
// fast error metric instead of spawning a subprocess.
func TestCollectSinceCarriesUntouched(t *testing.T) {
	orig := Cards
	t.Cleanup(func() { Cards = orig })
	Cards = []Card{
		{Key: "a", Debt: "a_debt", Label: "alpha", Script: "missing_a.py", Corpus: []string{"docs/"}},
		{Key: "b", Debt: "b_debt", Label: "beta", Script: "missing_b.py", Corpus: []string{"src/"}},
	}
	base := &Baseline{
		Commit:       "base123",
		TotalDebt:    8,
		Metrics:      map[string]int{"a": 3, "b": 5},
		GradeWeights: map[string]int{"a": 1, "b": 2},
	}
	root := t.TempDir()

	t.Run("nothing changed carries all -> fold equals baseline", func(t *testing.T) {
		metrics, info := CollectSince(root, "python3", time.Second, "HEAD", nil, base)
		if info.Measured != 0 || info.Carried != 2 {
			t.Fatalf("measured=%d carried=%d want 0/2", info.Measured, info.Carried)
		}
		if !reflect.DeepEqual(info.CarriedKeys, []string{"a", "b"}) {
			t.Fatalf("carried keys=%v want [a b]", info.CarriedKeys)
		}
		for _, m := range metrics {
			if !m.Carried || m.Debt == nil {
				t.Fatalf("metric %s not carried: %+v", m.Key, m)
			}
		}
		p := Fold(metrics, base, root, "head")
		if p.TotalDebt != base.TotalDebt {
			t.Fatalf("carried-all fold total=%d want baseline %d", p.TotalDebt, base.TotalDebt)
		}
	})

	t.Run("touched corpus measures only that card", func(t *testing.T) {
		metrics, info := CollectSince(root, "python3", time.Second, "HEAD", []string{"src/x.go"}, base)
		if info.Measured != 1 || info.Carried != 1 {
			t.Fatalf("measured=%d carried=%d want 1/1", info.Measured, info.Carried)
		}
		if !reflect.DeepEqual(info.CarriedKeys, []string{"a"}) {
			t.Fatalf("carried keys=%v want [a]", info.CarriedKeys)
		}
		// canonical order preserved: a (carried) then b (measured -> error, debt nil)
		if metrics[0].Key != "a" || !metrics[0].Carried {
			t.Fatalf("metrics[0]=%+v want carried a", metrics[0])
		}
		if metrics[1].Key != "b" || metrics[1].Carried || metrics[1].Debt != nil {
			t.Fatalf("metrics[1]=%+v want measured b (error, debt nil)", metrics[1])
		}
	})
}
