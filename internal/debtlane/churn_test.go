package debtlane

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseChurnLogDropsBulkCommits(t *testing.T) {
	var b strings.Builder
	b.WriteString("@@aaa\n\ninternal/foo/a.go\ninternal/foo/b.go\n")
	b.WriteString("@@bulk\n\n")
	for i := 0; i <= maxChurnFilesPerCommit; i++ {
		fmt.Fprintf(&b, "internal/bar/f%d.go\n", i)
	}
	b.WriteString("@@ccc\n\ncmd/fak/main.go\n")
	commits := ParseChurnLog([]byte(b.String()))
	if len(commits) != 2 {
		t.Fatalf("commits = %d, want 2 (bulk commit dropped)", len(commits))
	}
	if len(commits[0]) != 2 || commits[1][0] != "cmd/fak/main.go" {
		t.Fatalf("unexpected commits: %v", commits)
	}
}

func TestAttachChurnCreditsDeepestUnitOncePerCommit(t *testing.T) {
	lanes := []DebtLane{
		{Lane: "foo", UnitOfWork: "internal/foo"},
		{Lane: "foo_sub", UnitOfWork: `internal\foo\sub`},
		{Lane: "idle", UnitOfWork: "internal/idle"},
	}
	AttachChurn(lanes, [][]string{
		{"internal/foo/a.go", "internal/foo/b.go"},     // foo once, not twice
		{"internal/foo/sub/x.go"},                      // deepest owner only
		{"internal/foo/a.go", "internal/foo/sub/y.go"}, // both
	})
	got := map[string]int{}
	for _, l := range lanes {
		got[l.Lane] = l.RecentCommits
	}
	if got["foo"] != 2 || got["foo_sub"] != 2 || got["idle"] != 0 {
		t.Fatalf("churn = %v, want foo=2 foo_sub=2 idle=0", got)
	}
}

func TestChurnFactorMonotoneAndCapped(t *testing.T) {
	if ChurnFactor(0) != 1 {
		t.Fatalf("ChurnFactor(0) = %v, want 1", ChurnFactor(0))
	}
	prev := 1.0
	for _, n := range []int{1, 3, 7, 15} {
		f := ChurnFactor(n)
		if f <= prev {
			t.Fatalf("ChurnFactor(%d) = %v not > %v", n, f, prev)
		}
		prev = f
	}
	if ChurnFactor(10000) != maxChurnFactor {
		t.Fatalf("ChurnFactor not capped: %v", ChurnFactor(10000))
	}
}

// TestPlanWavesPrefersChurnedDebt pins that among lanes of comparable debt the
// one under active change is planned first, and that without churn data the
// order is the plain total-debt order.
func TestPlanWavesPrefersChurnedDebt(t *testing.T) {
	cold := DebtLane{Lane: "cold", UnitOfWork: "internal/cold", Criticality: CriticalityEnabling, Weight: 2, Maturity: 4, TargetMaturity: 8, MaturityGap: 4, TotalDebt: 10}
	hot := DebtLane{Lane: "hot", UnitOfWork: "internal/hot", Criticality: CriticalityEnabling, Weight: 2, Maturity: 5, TargetMaturity: 8, MaturityGap: 3, TotalDebt: 7}
	opts := WavePlanOptions{WaveSize: 1, Graph: map[string]map[string]struct{}{}}

	plain := PlanWaves(Report{Lanes: []DebtLane{cold, hot}}, opts)
	if plain.Waves[0].Lanes[0].Lane != "cold" {
		t.Fatalf("without churn wave-1 = %s, want cold (higher debt)", plain.Waves[0].Lanes[0].Lane)
	}
	hot.RecentCommits = 7
	churned := PlanWaves(Report{Lanes: []DebtLane{cold, hot}}, opts)
	if churned.Waves[0].Lanes[0].Lane != "hot" {
		t.Fatalf("with churn wave-1 = %s, want hot (7 recent commits)", churned.Waves[0].Lanes[0].Lane)
	}
}

// TestScanAttachesChurnByDefaultWindow drives Scan end to end with a stubbed
// git log and checks RecentCommits lands on the touched lane only.
func TestScanAttachesChurnByDefaultWindow(t *testing.T) {
	tmp := t.TempDir()
	for _, p := range []string{"internal/hotpkg", "internal/coldpkg"} {
		dir := filepath.Join(tmp, p)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(p)
		if err := os.WriteFile(filepath.Join(dir, name+".go"), []byte("package "+name+"\n\n// X is a fixture.\nfunc X() int { return 1 }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	orig := gitChurnLog
	defer func() { gitChurnLog = orig }()
	var gotDays int
	gitChurnLog = func(root string, days int) ([]byte, error) {
		gotDays = days
		return []byte("@@1\ninternal/hotpkg/hotpkg.go\n@@2\ninternal/hotpkg/hotpkg.go\n"), nil
	}
	report, err := Scan(Options{WorkspaceRoot: tmp, TargetRepo: "fak", ChurnDays: DefaultChurnDays, TopN: 50})
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if gotDays != DefaultChurnDays {
		t.Fatalf("git log window = %d, want %d", gotDays, DefaultChurnDays)
	}
	got := map[string]int{}
	for _, l := range report.Lanes {
		got[l.Lane] = l.RecentCommits
	}
	if got["hotpkg"] != 2 || got["coldpkg"] != 0 {
		t.Fatalf("recent commits = %v, want hotpkg=2 coldpkg=0", got)
	}
}
