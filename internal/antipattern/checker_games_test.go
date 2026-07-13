package antipattern

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectCheckerGames(t *testing.T) {
	root := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("candidate.py", `def solve():
    print("PASS")
`)
	write("honest.py", `def solve():
    return calculate_answer()
`)
	write("checker_test.go", `package fixture
import "testing"
func TestGamed(t *testing.T) {
    t.Skip("pretend green")
    t.Fatal("real assertion")
}
func TestConditionalSkipIsLegitimate(t *testing.T) {
    if testing.Short() { t.Skip("slow") }
    t.Error("real assertion")
}
`)

	got := detectCheckerGames(root)
	if len(got) != 2 {
		t.Fatalf("findings = %#v, want hardcoded-output and short-circuit findings", got)
	}
	for _, finding := range got {
		if finding.Class != CheckerGames {
			t.Fatalf("class = %q, want %q", finding.Class, CheckerGames)
		}
	}
}

func TestCheckerGamesCountsTowardDebt(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "submission.sh"), []byte("echo 'SUCCESS'\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	findings, _ := Collect(root, nil)
	if len(findings) != 1 || findings[0].Class != CheckerGames {
		t.Fatalf("findings = %#v, want one checker-games finding", findings)
	}
	payload := Build(root, nil)
	if payload.KPIs[0].Value <= 0 {
		t.Fatalf("antipattern_debt = %v, want positive", payload.KPIs[0].Value)
	}
}
