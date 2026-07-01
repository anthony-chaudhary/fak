package workflowaudit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

// noCutoverRoles is the current regime: all four roles == "main".
func noCutoverRoles() branchrole.Roles { return branchrole.Defaults() }

// writeWorkflows writes a set of name->content workflow files into a temp dir.
func writeWorkflows(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return dir
}

// classOf returns the class of the first reference matching (file, raw), or "".
func classOf(rep Report, file, raw string) RefClass {
	for _, r := range rep.Refs {
		if r.File == file && r.Raw == raw {
			return r.Class
		}
	}
	return ""
}

// TestAuditClassifiesEveryIdiom asserts each of the three reference idioms (plus the
// hidden shell ref and the inline/block list forms) lands in the right class.
func TestAuditClassifiesEveryIdiom(t *testing.T) {
	files := map[string]string{
		"ci.yml":      "on:\n  push:\n    branches: [main, master, fak-v0.1]\n  pull_request:\n",
		"release.yml": "on:\n  push:\n    tags: [\"v*\"]\n",
		"feed.yml":    "jobs:\n  post:\n    if: github.ref_name == 'main' || github.ref_name == 'master'\n",
		"bench.yml": "    branches:\n      - main\n      - master\n" +
			"      run: gh run list --branch main --workflow bench.yml\n" +
			"    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/fak-v0.1'\n",
	}
	allow := ParseAllowlist(strings.Join([]string{
		"ci.yml:master", "ci.yml:fak-v0.1",
		"feed.yml:master",
		"bench.yml:master", "bench.yml:fak-v0.1", "bench.yml:main",
	}, "\n"))

	rep, err := Audit(writeWorkflows(t, files), noCutoverRoles(), allow)
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}

	want := []struct {
		file, raw string
		class     RefClass
	}{
		{"ci.yml", "main", ClassDevelopment},        // branches filter on the dev role
		{"ci.yml", "master", ClassLegacy},           // legacy arm, allowlisted
		{"ci.yml", "fak-v0.1", ClassLegacy},         // legacy compat branch
		{"release.yml", "v*", ClassTag},             // tags filter
		{"feed.yml", "main", ClassReleaseFrontDoor}, // ref_name gate on the role branch
		{"feed.yml", "master", ClassLegacy},         // legacy front-door arm
		{"bench.yml", "main", ClassDevelopment},     // block-list branches form (first main)
		{"bench.yml", "v*", ""},                     // absent -- bench has no tags filter
	}
	for _, w := range want {
		if w.class == "" {
			continue
		}
		if got := classOf(rep, w.file, w.raw); got != w.class {
			t.Errorf("%s ref %q: class=%q want %q", w.file, w.raw, got, w.class)
		}
	}

	// The hidden `--branch main` shell ref must be detected as a hidden-shell-ref kind.
	var sawShell bool
	for _, r := range rep.Refs {
		if r.File == "bench.yml" && r.Kind == KindHiddenShellRef && r.Raw == "main" {
			sawShell = true
			if r.Class != ClassLegacy {
				t.Errorf("hidden shell --branch main: class=%q want %q (allowlisted)", r.Class, ClassLegacy)
			}
		}
	}
	if !sawShell {
		t.Error("hidden `--branch main` shell ref was not detected")
	}

	if !rep.Clean() {
		t.Errorf("fully-allowlisted fixture should be clean; unclassified=%v", rep.Unclassified)
	}
}

// TestDetectsNewUnclassifiedGate proves the regression gate fires: a development-path
// branch filter naming neither a role nor an allowlisted token is ClassUnclassified and
// makes the report dirty. This is the half of the witness that proves the gate bites.
func TestDetectsNewUnclassifiedGate(t *testing.T) {
	files := map[string]string{
		"rogue.yml": "on:\n  push:\n    branches: [feature-x]\n",
	}
	rep, err := Audit(writeWorkflows(t, files), noCutoverRoles(), DefaultAllowlist())
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}
	if rep.Clean() {
		t.Fatal("a raw `branches: [feature-x]` filter must red the gate, but report was clean")
	}
	if classOf(rep, "rogue.yml", "feature-x") != ClassUnclassified {
		t.Errorf("feature-x class=%q want %q", classOf(rep, "rogue.yml", "feature-x"), ClassUnclassified)
	}
}

// TestLiveTreeFullyClassified is the regression fence on the REAL workflows: every branch/
// tag reference in .github/workflows today is a configured role, a tag, or an intentional
// allowlisted legacy arm. A future PR that adds an unclassified development-path reference
// (a raw feature-branch filter, a new hard-coded `ref_name == 'staging'` gate) without
// classifying it reds this test, and with it `go test ./...` and `make ci`.
func TestLiveTreeFullyClassified(t *testing.T) {
	root := repoRoot(t)
	dir := filepath.Join(root, ".github", "workflows")
	roles, err := branchrole.Load(root)
	if err != nil {
		t.Fatalf("load branch roles: %v", err)
	}
	rep, err := Audit(dir, roles, DefaultAllowlist())
	if err != nil {
		t.Fatalf("Audit live tree: %v", err)
	}
	if rep.Files == 0 {
		t.Fatal("no workflow files scanned -- wrong directory?")
	}
	if !rep.Clean() {
		var lines []string
		for _, r := range rep.Unclassified {
			lines = append(lines, r.File+":"+itoa(r.Line)+" "+r.Kind+" "+r.Raw)
		}
		t.Fatalf("live workflows have %d unclassified development-path reference(s):\n  %s\n"+
			"Classify each: add the branch to dos.toml [branch_roles] if it is a real role, or add the "+
			"(file, token) to internal/workflowaudit/allow.txt if it is an intentional legacy/compat arm.",
			len(rep.Unclassified), strings.Join(lines, "\n  "))
	}
}

// TestAllowlistIsBoundedAndParsed asserts the embedded allowlist parses to a non-empty,
// bounded set -- it must exist (else the live tree would not be clean) and stay small.
func TestAllowlistIsBoundedAndParsed(t *testing.T) {
	a := DefaultAllowlist()
	if a.Len() == 0 {
		t.Fatal("embedded allowlist is empty -- allow.txt failed to embed or parse")
	}
	if a.Len() > 80 {
		t.Errorf("allowlist has %d entries -- it should only ever SHRINK toward the cutover; "+
			"a growing allowlist means new legacy refs are being added, not retired", a.Len())
	}
	if !a.Has("bench.yml", "main") {
		t.Error("the documented hidden `bench.yml --branch main` baseline ref must be allowlisted")
	}
}

// TestBlockDeterministic guards the generator: two renders of one report are byte-identical
// so the committed snapshot regenerates with no diff.
func TestBlockDeterministic(t *testing.T) {
	rep, err := Audit(repoWorkflows(t), branchrole.Defaults(), DefaultAllowlist())
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}
	if Block(rep) != Block(rep) {
		t.Fatal("Block(rep) is not deterministic across two calls")
	}
	got, ok := Extract(Block(rep))
	if !ok || got != Block(rep) {
		t.Fatal("Extract should round-trip the rendered block")
	}
}

// TestFreshDetectsDrift asserts Fresh reds when the embedded block drifts from the report.
func TestFreshDetectsDrift(t *testing.T) {
	rep, err := Audit(repoWorkflows(t), branchrole.Defaults(), DefaultAllowlist())
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}
	doc := Scaffold()
	spliced, err := Splice(doc, rep)
	if err != nil {
		t.Fatalf("Splice: %v", err)
	}
	if !Fresh(spliced, rep) {
		t.Fatal("a freshly spliced doc must be Fresh")
	}
	// Mutate a token that only appears inside the generated block (the verdict line),
	// never in the Scaffold header prose, so the drift is genuinely inside the markers.
	stale := strings.Replace(spliced, "ALL CLASSIFIED", "ALL_MUTATED", 1)
	if stale == spliced {
		t.Fatal("expected the verdict token inside the spliced block")
	}
	if Fresh(stale, rep) {
		t.Fatal("a mutated block must not be Fresh")
	}
}

// TestLiveDocFresh is the golden-file fence: the committed docs/ci/workflow-branch-audit.md
// block must equal the block regenerated from the live workflows. It reds the trunk the
// moment a workflow's branch filters change without regenerating the report. Regenerate
// with `fak workflow-audit --write-doc`. This lives in the tier-1 package (not cmd/fak) so
// the fence stays green even while a peer's WIP breaks the cmd/fak build.
func TestLiveDocFresh(t *testing.T) {
	root := repoRoot(t)
	roles, err := branchrole.Load(root)
	if err != nil {
		t.Fatalf("load branch roles: %v", err)
	}
	rep, err := Audit(filepath.Join(root, ".github", "workflows"), roles, DefaultAllowlist())
	if err != nil {
		t.Fatalf("Audit: %v", err)
	}
	docPath := filepath.Join(root, filepath.FromSlash(DocRel))
	raw, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("read %s: %v (run `fak workflow-audit --write-doc`)", DocRel, err)
	}
	if !Fresh(string(raw), rep) {
		t.Fatalf("%s block drifted from the live workflows; run `fak workflow-audit --write-doc`", DocRel)
	}
}

// --- small test helpers (no internal imports beyond branchrole) ---

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found walking up from test dir")
		}
		dir = parent
	}
}

func repoWorkflows(t *testing.T) string {
	t.Helper()
	return filepath.Join(repoRoot(t), ".github", "workflows")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
