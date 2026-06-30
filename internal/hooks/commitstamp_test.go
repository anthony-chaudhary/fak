package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeLintRepo lays down a minimal repo root: a dos.toml declaring a handful of lanes + trees,
// plus a couple of real package dirs (one declared, one NOT) so the recognition / undeclared-real
// / typo branches are all reachable.
func writeLintRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `# test taxonomy
[lanes]
concurrent = [
  "gateway", "policy", "cmd", "docs",
]
exclusive = ["abi", "release", "global"]
autopick = ["gateway", "policy"]

[lanes.trees]
gateway = ["internal/gateway/**"]
policy  = ["internal/policy/**"]
cmd     = ["cmd/**"]
docs    = ["docs/**", "README.md", "INDEX.md", "llms.txt", "llms-full.txt", "llms-updates.txt"]
release = ["VERSION", "docs/releases/**"]
dos     = ["dos.toml", ".gitignore"]
tools   = ["tools/**", "scripts/**"]
global  = ["**/*"]

[stamp]
trailer_stamp = true
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	// internal/gateway is declared; internal/undeclaredleaf is a real package with NO lane.
	for _, d := range []string{"internal/gateway", "internal/policy", "internal/undeclaredleaf", "cmd/fak", "cmd/somedemo"} {
		if err := os.MkdirAll(filepath.Join(root, filepath.FromSlash(d)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func hasIssueContaining(r CommitLintReport, sub string) bool { return anyContains(r.Issues, sub) }
func hasNoteContaining(r CommitLintReport, sub string) bool  { return anyContains(r.Notes, sub) }

func anyContains(xs []string, sub string) bool {
	for _, x := range xs {
		if strings.Contains(x, sub) {
			return true
		}
	}
	return false
}

func TestLintCommitMessage_cleanLeafAndShim(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"feat(gateway): add the slot reclaim path (fak gateway)",
		[]string{"internal/gateway/server.go", "cmd/fak/serve.go"},
		root,
	)
	if !r.OK {
		t.Fatalf("expected OK, got issues=%v", r.Issues)
	}
	if !r.Gradeable || r.StampKind != "trailer" || r.Leaf != "gateway" || !r.LeafMatches {
		t.Fatalf("unexpected report: %+v", r)
	}
	// gateway is one of the touched lanes (the cmd/fak shim must NOT force a `cmd` mismatch).
	if !r.LeafRecognized {
		t.Fatalf("gateway should be a recognized declared lane")
	}
}

func TestLintCommitMessage_nounLedNoTrailer(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("gateway slot reclaim improvements", []string{"internal/gateway/server.go"}, root)
	if r.OK {
		t.Fatalf("expected NOT ok")
	}
	if !hasIssueContaining(r, "witness-gradeable") {
		t.Errorf("want gradeability issue, got %v", r.Issues)
	}
	if !hasIssueContaining(r, "no ship-stamp") {
		t.Errorf("want no-stamp issue, got %v", r.Issues)
	}
	if r.SuggestTrailer != "(fak gateway)" {
		t.Errorf("want suggested (fak gateway), got %q", r.SuggestTrailer)
	}
}

func TestLintCommitMessage_offLaneTypo(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("fix(gateway): correct the reclaim (fak gatway)", []string{"internal/gateway/server.go"}, root)
	if r.OK {
		t.Fatalf("expected NOT ok for a typo'd leaf")
	}
	if r.LeafRecognized {
		t.Errorf("gatway should NOT be recognized")
	}
	if !hasIssueContaining(r, "off-lane stamp") || !hasIssueContaining(r, "did you mean `(fak gateway)`") {
		t.Errorf("want off-lane + did-you-mean hint, got %v", r.Issues)
	}
}

func TestLintCommitMessage_laneMismatch(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("feat(policy): add a rule (fak gateway)", []string{"internal/policy/rules.go"}, root)
	if r.OK {
		t.Fatalf("expected NOT ok for a stamp/path lane mismatch")
	}
	if r.LeafMatches {
		t.Errorf("gateway must not match a policy-only path set")
	}
	if !hasIssueContaining(r, "stamp/path lane mismatch") {
		t.Errorf("want mismatch issue, got %v", r.Issues)
	}
}

func TestLintCommitMessage_undeclaredRealLeafIsNoteNotIssue(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"feat(undeclaredleaf): add a thing (fak undeclaredleaf)",
		[]string{"internal/undeclaredleaf/x.go"},
		root,
	)
	if !r.OK {
		t.Fatalf("a real-but-undeclared leaf should be OK (advisory), got issues=%v", r.Issues)
	}
	if !r.LeafRecognized {
		t.Errorf("a real package dir should be recognized")
	}
	if !hasNoteContaining(r, "no dos.toml lane declares it") {
		t.Errorf("want the undeclared-lane advisory note, got %v", r.Notes)
	}
}

func TestLintCommitMessage_cmdDemoLeaf(t *testing.T) {
	root := writeLintRepo(t)
	// A cmd/<dir> demo stamped with its dir name binds to the cmd lane's tree (#518): accepted.
	r := LintCommitMessage("feat(somedemo): add the demo (fak somedemo)", []string{"cmd/somedemo/main.go"}, root)
	if !r.LeafMatches {
		t.Fatalf("a cmd/<dir> demo stamped (fak <dir>) must match, got issues=%v", r.Issues)
	}
}

func TestLintCommitMessage_releaseExempt(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("v0.29.0: cut the release", []string{"VERSION"}, root)
	if !r.OK {
		t.Fatalf("a release subject should be exempt+OK, got %v", r.Issues)
	}
	if r.StampKind != "release" {
		t.Errorf("want stamp kind release, got %q", r.StampKind)
	}
	// VERSION is a bare (non-glob) tree entry -> resolves to the release lane.
	if len(r.PathLanes) != 1 || r.PathLanes[0] != "release" {
		t.Errorf("want VERSION -> release lane, got %v", r.PathLanes)
	}
}

func TestLintCommitMessage_rootDocsResolveToDocsLane(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("docs(readme): update the entrypoint (fak docs)", []string{"README.md", "INDEX.md", "llms.txt"}, root)
	if !r.OK {
		t.Fatalf("expected root docs to lint cleanly, got issues=%v notes=%v", r.Issues, r.Notes)
	}
	if len(r.PathLanes) != 1 || r.PathLanes[0] != "docs" {
		t.Fatalf("root docs PathLanes = %v, want [docs]", r.PathLanes)
	}
	if !r.LeafMatches {
		t.Fatalf("(fak docs) should match root docs paths: %+v", r)
	}
	if r.SuggestTrailer != "(fak docs)" {
		t.Fatalf("SuggestTrailer = %q, want (fak docs)", r.SuggestTrailer)
	}
}

func TestLintCommitMessage_dosTomlResolvesToDosLane(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("fix(dos): route taxonomy edits (fak dos)", []string{"dos.toml", ".gitignore"}, root)
	if !r.OK {
		t.Fatalf("expected dos.toml to lint cleanly, got issues=%v notes=%v", r.Issues, r.Notes)
	}
	if len(r.PathLanes) != 1 || r.PathLanes[0] != "dos" {
		t.Fatalf("dos.toml PathLanes = %v, want [dos]", r.PathLanes)
	}
	if !r.LeafMatches {
		t.Fatalf("(fak dos) should match dos.toml: %+v", r)
	}
	if r.SuggestTrailer != "(fak dos)" {
		t.Fatalf("SuggestTrailer = %q, want (fak dos)", r.SuggestTrailer)
	}
}

func TestLintCommitMessage_scriptsResolveToToolsLane(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("fix(tools): refresh helper script (fak tools)", []string{"scripts/gcp-fleet-janitor.sh"}, root)
	if !r.OK {
		t.Fatalf("expected scripts/ helper to lint cleanly, got issues=%v notes=%v", r.Issues, r.Notes)
	}
	if len(r.PathLanes) != 1 || r.PathLanes[0] != "tools" {
		t.Fatalf("scripts helper PathLanes = %v, want [tools]", r.PathLanes)
	}
	if !r.LeafMatches {
		t.Fatalf("(fak tools) should match scripts/ helper: %+v", r)
	}
}

func TestLintCommitMessage_mergeExempt(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage("Merge origin/main into main", []string{"internal/gateway/x.go"}, root)
	if !r.OK || r.StampKind != "exempt" {
		t.Fatalf("a merge subject should be exempt+OK, got ok=%v kind=%q issues=%v", r.OK, r.StampKind, r.Issues)
	}
}

func TestLintCommitMessage_gateOnAbstainAdvisory(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"feat(gateway): gate L3 promotion on durability class (fak gateway)",
		[]string{"internal/gateway/x.go"},
		root,
	)
	if !r.OK {
		t.Fatalf("verb-led `gate ... on ...` is gradeable; should be OK with an advisory, got %v", r.Issues)
	}
	if !hasNoteContaining(r, "ABSTAIN") {
		t.Errorf("want the gate-X-on-Y abstain advisory note, got %v", r.Notes)
	}
}

func TestLintCommitMessage_noTaxonomySkipsRecognition(t *testing.T) {
	// root="" => dos.toml unreadable => recognition is SKIPPED, never failed. A novel leaf on a
	// matching-by-convention path is then OK.
	r := LintCommitMessage("feat(gateway): add a thing (fak gateway)", []string{"internal/gateway/x.go"}, "")
	if !r.OK {
		t.Fatalf("with no taxonomy, a convention-matching stamp should be OK, got %v", r.Issues)
	}
}

func TestStampOf(t *testing.T) {
	cases := []struct {
		subject  string
		wantKind string
		wantLeaf string
	}{
		{"feat(x): do it (fak gateway)", "trailer", "gateway"},
		{"feat(x): do it (refs fak gateway)", "trailer", "gateway"},
		{"fak/gateway: do it", "direct", "gateway"},
		{"v1.2.3: release", "release", ""},
		{"feat(x): do it", "none", ""},
	}
	for _, c := range cases {
		k, l := stampOf(c.subject)
		if k != c.wantKind || l != c.wantLeaf {
			t.Errorf("stampOf(%q) = (%q,%q), want (%q,%q)", c.subject, k, l, c.wantKind, c.wantLeaf)
		}
	}
}

// TestLintCommitMessage_fixNoTestWantsSymptomWitness — a fix touching Go source but no test
// earns the symptom-witness advisory (#1326), and it is advisory only (OK stays true).
func TestLintCommitMessage_fixNoTestWantsSymptomWitness(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"fix(gateway): treat same-tick ready as positive (fak gateway)",
		[]string{"internal/gateway/server.go"},
		root,
	)
	if !r.OK {
		t.Fatalf("the advisory must NOT block; want OK, got issues=%v", r.Issues)
	}
	if !hasNoteContaining(r, "symptom witness") {
		t.Errorf("want the symptom-witness advisory note, got %v", r.Notes)
	}
}

// TestLintCommitMessage_fixWithTestNoSymptomNote — the same fix that ALSO touches a test carries
// its witness, so no advisory fires.
func TestLintCommitMessage_fixWithTestNoSymptomNote(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"fix(gateway): treat same-tick ready as positive (fak gateway)",
		[]string{"internal/gateway/server.go", "internal/gateway/server_test.go"},
		root,
	)
	if hasNoteContaining(r, "symptom witness") {
		t.Errorf("a fix carrying a test should earn no symptom-witness note, got %v", r.Notes)
	}
}

// TestLintCommitMessage_featNoSymptomNote — the heuristic is scoped to `fix(...)`; a feat touching
// only source is not nudged.
func TestLintCommitMessage_featNoSymptomNote(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"feat(gateway): add the slot reclaim path (fak gateway)",
		[]string{"internal/gateway/server.go"},
		root,
	)
	if hasNoteContaining(r, "symptom witness") {
		t.Errorf("feat must not earn the symptom-witness note, got %v", r.Notes)
	}
}

// TestLintCommitMessage_fixDocsOnlyNoSymptomNote — a fix that touches no Go source (docs/config
// only) has no testable symptom surface, so it is not nudged.
func TestLintCommitMessage_fixDocsOnlyNoSymptomNote(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessage(
		"fix(docs): correct the gateway flag name (fak docs)",
		[]string{"docs/gateway.md"},
		root,
	)
	if hasNoteContaining(r, "symptom witness") {
		t.Errorf("a docs-only fix should earn no symptom-witness note, got %v", r.Notes)
	}
}

// TestFixWantsSymptomWitness_unit drives the helper directly across the path-classification edges
// (testdata excluded; a bare .go vs a _test.go; non-fix types).
func TestFixWantsSymptomWitness_unit(t *testing.T) {
	cases := []struct {
		name    string
		subject string
		paths   []string
		want    bool // want a non-empty advisory
	}{
		{"fix-source-only", "fix(x): repair the off-by-one (fak x)", []string{"internal/x/x.go"}, true},
		{"fix-with-test", "fix(x): repair (fak x)", []string{"internal/x/x.go", "internal/x/x_test.go"}, false},
		{"fix-testdata-is-not-source", "fix(x): refresh fixture (fak x)", []string{"internal/x/testdata/in.go"}, false},
		{"fix-no-go", "fix(x): tweak config (fak x)", []string{"internal/x/config.yaml"}, false},
		{"feat-source", "feat(x): add (fak x)", []string{"internal/x/x.go"}, false},
		{"perf-source", "perf(x): speed up (fak x)", []string{"internal/x/x.go"}, false},
		{"no-paths", "fix(x): repair (fak x)", nil, false},
	}
	for _, tc := range cases {
		got := fixWantsSymptomWitness(tc.subject, tc.paths) != ""
		if got != tc.want {
			t.Errorf("%s: fixWantsSymptomWitness=%v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestCommitMsgVerdict_advisoryVerbsRecognized — a commit that ADDS an advising lint/gate (the
// commit-gardening surface, #1326) leads with a real action verb; the gate must grade it, not
// ABSTAIN as it did on the original "advise a symptom witness" subject.
func TestCommitMsgVerdict_advisoryVerbsRecognized(t *testing.T) {
	subjects := []string{
		"feat(hooks): advise a symptom witness for fix(...) commits (fak hooks)",
		"feat(hooks): nudge fix commits toward a red-then-green test (fak hooks)",
		"feat(gate): recommend a bindable issue link on close (fak gate)",
		"feat(gate): warn on a noun-led subject (fak gate)",
	}
	for _, s := range subjects {
		if ok, why := CommitMsgVerdict(s); !ok {
			t.Errorf("CommitMsgVerdict(%q) = not-ok (%s), want gradeable", s, why)
		}
	}
}

func TestLaneForPath_conventionFallback(t *testing.T) {
	var empty laneTaxonomy // not loaded: pure convention
	cases := map[string]string{
		"internal/gateway/server.go": "gateway",
		"cmd/fak/serve.go":           "cmd",
		"docs/x.md":                  "docs",
		"README.md":                  "", // root file: no lane
	}
	for path, want := range cases {
		if got := laneForPath(path, empty); got != want {
			t.Errorf("laneForPath(%q) = %q, want %q", path, got, want)
		}
	}
}
