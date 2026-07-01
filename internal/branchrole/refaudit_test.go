package branchrole

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestClassifyHardcodedRef(t *testing.T) {
	cases := []struct {
		path string
		line string
		want string
	}{
		{".github/workflows/ci.yml", "branches: [main, master]", RefClassWorkflowCovered},
		{"tools/extend_preflight.py", `branch == "master"`, RefClassDevelopmentSource},
		{"tools/fleet_control_pane.py", `DEFAULT_WORKTREE_MASTER_REF = "origin/master"`, RefClassDevelopmentSource},
		{"docs/stable-releases/2026-06-stable.md", "committed to `master`", RefClassHistorical},
		{"tools/bench_migrate.py", `"branch": "master"`, RefClassFixture},
		{"tools/demo_robustness_scorecard.py", `@(latest|main|master)`, RefClassPublicGuard},
		{"cmd/fak/new.go", `git fetch origin main`, RefClassUnclassified},
	}
	for _, tc := range cases {
		if got := ClassifyHardcodedRef(tc.path, tc.line); got != tc.want {
			t.Fatalf("ClassifyHardcodedRef(%q, %q) = %q, want %q", tc.path, tc.line, got, tc.want)
		}
	}
}

func TestHardcodedRefAuditCurrentTreeClassified(t *testing.T) {
	root := repoRootForRefAudit(t)
	findings, err := AuditHardcodedRefs(root)
	if err != nil {
		t.Fatalf("AuditHardcodedRefs: %v", err)
	}
	var unclassified []string
	classes := map[string]int{}
	for _, finding := range findings {
		classes[finding.Class]++
		if finding.Class == RefClassUnclassified {
			unclassified = append(unclassified, finding.Path+":"+itoa(finding.Line)+" "+finding.Text)
		}
	}
	if len(unclassified) > 0 {
		t.Fatalf("unclassified hard-coded branch refs:\n%s", strings.Join(unclassified, "\n"))
	}
	for _, want := range []string{RefClassDevelopmentSource, RefClassWorkflowCovered, RefClassAuditDoc} {
		if classes[want] == 0 {
			t.Fatalf("audit saw no %s rows; classes=%v", want, classes)
		}
	}
}

func repoRootForRefAudit(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("dos.toml not found from %s", dir)
		}
		dir = parent
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
