package metrics

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNegationTaxCountsPositiveAndSeededTurns(t *testing.T) {
	var r NegationTaxRecorder
	if got := r.Record("Keep working and commit the result."); got.Total != 0 {
		t.Fatalf("positive tax=%+v", got)
	}
	got := r.Record("Do not forget to commit. Never delete the audit trail.")
	if got.Mechanical != 1 || got.Judgement != 1 || got.Total != 2 {
		t.Fatalf("seeded tax=%+v", got)
	}
	rep := r.Report()
	if rep.Turns != 2 || rep.Mechanical != 1 || rep.Judgement != 1 || rep.Total != 2 {
		t.Fatalf("report=%+v", rep)
	}
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"mechanical":1`, `"judgement":1`, `"total":2`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("json %s missing %s", b, want)
		}
	}
	prom := rep.Prometheus()
	for _, want := range []string{`fak_negation_tax_total{tier="mechanical"} 1`, `fak_negation_tax_total{tier="judgement"} 1`, `fak_negation_tax_turns_total 2`} {
		if !strings.Contains(prom, want) {
			t.Fatalf("metrics missing %q: %s", want, prom)
		}
	}
	artifact := Report{KPIs: KPIs{NegationTax: rep}}
	if !strings.Contains(string(artifact.JSON()), `"negation_tax"`) {
		t.Fatalf("report JSON missing tax: %s", artifact.JSON())
	}
}
