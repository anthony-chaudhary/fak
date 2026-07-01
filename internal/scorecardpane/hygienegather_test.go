package scorecardpane

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// writeFixture writes rel under root, creating parent dirs.
func writeFixture(t *testing.T, root, rel, body string) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// initFixtureRepo builds a tiny tracked tree with a couple of seeded hygiene defects
// and commits it, so CollectHygiene reads a real git-tracked tree.
func initFixtureRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	writeFixture(t, root, "README.md", "# Fixture\n\nSee [the guide](docs/guide.md).\n")
	writeFixture(t, root, "INDEX.md", "# Index\n\n- [guide](docs/guide.md)\n- [dead](docs/missing.md)\n")
	writeFixture(t, root, "docs/guide.md", "# Guide\n\nPlain prose that reads fine.\n")
	// an orphan: reader-facing, linked from no index/hub.
	writeFixture(t, root, "docs/orphan.md", "# Orphan\n\nNobody links to me.\n")
	// an AI-tell doc.
	writeFixture(t, root, "docs/marketing.md", "# Marketing\n\nWe delve into a seamless world-class design.\n")

	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	run("add", "-A")
	run("commit", "-q", "-m", "fixture")
	return root
}

func TestCollectHygieneOverFixtureTree(t *testing.T) {
	root := initFixtureRepo(t)
	p := CollectHygiene(root)
	if p.Schema != HygieneSchema {
		t.Fatalf("schema: want %q, got %q", HygieneSchema, p.Schema)
	}
	if p.Verdict == "AUDIT_ERROR" {
		t.Fatalf("a real git tree must not error: %s", p.Reason)
	}
	debtByKPI := p.Corpus.DebtByKPI
	// the dead INDEX.md -> docs/missing.md link is index_integrity debt.
	if debtByKPI["index_integrity"] < 1 {
		t.Errorf("dead index entry must produce index_integrity debt, got %d", debtByKPI["index_integrity"])
	}
	// docs/orphan.md is reachable from no index/hub.
	if debtByKPI["orphans"] < 1 {
		t.Errorf("orphan doc must produce orphans debt, got %d", debtByKPI["orphans"])
	}
	// the marketing doc carries AI-tell phrases.
	if debtByKPI["ai_tells"] < 1 {
		t.Errorf("AI-tell phrases must produce ai_tells debt, got %d", debtByKPI["ai_tells"])
	}
	if p.Corpus.HygieneDebt == 0 {
		t.Fatalf("seeded defects must produce nonzero hygiene_debt")
	}
}

func TestCollectHygieneNonGitDir(t *testing.T) {
	root := t.TempDir()
	p := CollectHygiene(root)
	if p.Verdict != "AUDIT_ERROR" {
		t.Fatalf("a non-git dir must AUDIT_ERROR, got %q", p.Verdict)
	}
}

func TestHygieneMarkdownRoundTrip(t *testing.T) {
	root := initFixtureRepo(t)
	p := CollectHygiene(root)
	md := RenderHygieneMarkdown(p, "2026-06-29")
	if !strings.Contains(md, "# Repo-hygiene scorecard") {
		t.Fatalf("markdown must carry the title header")
	}
	if !strings.Contains(md, "Hygiene-debt (total HARD defects)") {
		t.Fatalf("markdown must carry the headline table")
	}
	if !strings.Contains(md, "Composite value") || strings.Contains(md, "Composite score") {
		t.Fatalf("markdown must lead with continuous value, got:\n%s", md)
	}
	if !strings.Contains(md, "2026-06-29") {
		t.Fatalf("markdown must embed the stamp")
	}
	// every KPI with defects must appear as a work-list section.
	if p.Corpus.DebtByKPI["orphans"] > 0 && !strings.Contains(md, "`orphans`") {
		t.Fatalf("orphans defects must render a work-list section")
	}
}

func TestHygieneCompareNxVerdict(t *testing.T) {
	root := initFixtureRepo(t)
	current := CollectHygiene(root)
	// synthesize a worse baseline so the compare reports a reduction.
	baseline := current
	baseCorpus := baseline.Corpus
	baseCorpus.HygieneDebt = current.Corpus.HygieneDebt*4 + 4
	baseline.Corpus = baseCorpus
	out := RenderHygieneCompare(baseline, current)
	if !strings.Contains(out, "hygiene-debt:") || !strings.Contains(out, "VERDICT:") {
		t.Fatalf("compare must render the debt delta + verdict: %q", out)
	}
	if !strings.Contains(out, "value:") || strings.Contains(out, "/100") {
		t.Fatalf("compare must render continuous value instead of /100 score: %q", out)
	}
}
