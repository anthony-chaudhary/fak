package scorecardportfolio

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAuditRanksDiscoveryAndDebtContractGaps(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("tools/hidden_scorecard.py", `SCHEMA = "hidden/1"
def main(): pass
if __name__ == "__main__": main()
`)
	write("tools/capped_scorecard.py", `SCHEMA = "capped/1"
debt = min(raw_debt, 100)
if __name__ == "__main__": pass
`)
	write("tools/scorecard_control_pane.py", `SCORECARDS: list[dict[str, str]] = [
 {"key": "bound"}, {"key": "bound"}, {"key": "fractional"},
]
`)
	write("internal/scorecardpane/controlpane.go", `package scorecardpane
var Cards = []Card{{Key: "bound"}, {Key: "fractional"}}
type Card struct{ Key string }
`)
	write("cmd/fak/score.go", `package main
var scoreRoutes = map[string]func(){"native": nil}
`)
	write("tools/scorecard_baseline.json", `{"metrics":{"bound":0,"fractional":1.5},"detector_versions":{}}`)

	report, err := Audit(DefaultConfig(root))
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != Schema || report.DomainDebtIncluded {
		t.Fatalf("bad contract: %#v", report)
	}
	if report.CoverageDebt != len(report.Gaps) || report.CoverageDebtUnit != "portfolio_gap" {
		t.Fatalf("coverage debt is not a raw gap count: %#v", report)
	}
	want := map[string]bool{
		"python_producer_unbound:hidden":       false,
		"producer_debt_capped:capped":          false,
		"duplicate_pane_binding:bound":         false,
		"go_score_route_unbound:native":        false,
		"baseline_debt_non_integer:fractional": false,
		"detector_version_unpinned:bound":      false,
	}
	for _, gap := range report.Gaps {
		key := gap.Kind + ":" + gap.Producer
		if _, ok := want[key]; ok {
			want[key] = true
		}
		if gap.Rank < 1 || gap.Reach < 1 || gap.Consequence == "" || len(gap.Provenance) == 0 {
			t.Fatalf("gap lacks ranking evidence: %#v", gap)
		}
	}
	for key, seen := range want {
		if !seen {
			t.Errorf("missing gap %s; got %#v", key, report.Gaps)
		}
	}
}

func TestAuditIsDeterministic(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		path := filepath.Join(root, filepath.FromSlash(rel))
		_ = os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("tools/a_scorecard.py", `if __name__ == "__main__": pass`)
	write("tools/scorecard_control_pane.py", "SCORECARDS: list[dict[str, str]] = [\n]\n")
	write("internal/scorecardpane/controlpane.go", "package p\ntype Card struct{Key string}\nvar Cards=[]Card{}\n")
	write("cmd/fak/score.go", "package main\nvar scoreRoutes=map[string]func(){}\n")
	write("tools/scorecard_baseline.json", `{"metrics":{},"detector_versions":{}}`)
	a, err := Audit(DefaultConfig(root))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Audit(DefaultConfig(root))
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Gaps) != len(b.Gaps) {
		t.Fatal("nondeterministic gap count")
	}
	for i := range a.Gaps {
		if a.Gaps[i].Kind != b.Gaps[i].Kind || a.Gaps[i].Producer != b.Gaps[i].Producer {
			t.Fatalf("nondeterministic order: %#v %#v", a.Gaps, b.Gaps)
		}
	}
}

func TestLiveRepositoryWitness(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	report, err := Audit(DefaultConfig(root))
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != Schema || report.CoverageDebt != len(report.Gaps) {
		t.Fatalf("invalid live report contract: %#v", report)
	}
	limit := len(report.Gaps)
	if limit > 10 {
		limit = 10
	}
	for _, gap := range report.Gaps[:limit] {
		t.Logf("rank=%d kind=%s producer=%s reach=%d consequence=%s provenance=%v", gap.Rank, gap.Kind, gap.Producer, gap.Reach, gap.Consequence, gap.Provenance)
	}
}
