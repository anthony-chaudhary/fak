package adjudicator

// A Go test file that this adjudicator admitted as new-file authorship earlier
// in the same trace, and that still holds the admitted content, is in-progress
// authorship: a typed edit to it is admitted under any lane. Repository state is
// never the anchor, so removing a committed test from the index does not unlock
// it, and neither does a case-variant path, a directory alias, a swapped
// directory, another session, or an ambiguous edit payload.

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const authoredProbeBody = "package probe\n\nimport \"testing\"\n\nfunc TestProbe(t *testing.T) {\n\trep0 := 1\n\tif rep0 != 1 {\n\t\tt.Fatal(\"probe\")\n\t}\n}\n"

// Each fixture role holds distinct bytes, so a refusal is attributable to the
// rule under test and not to another fixture's object sharing the content.
var (
	authoredGateBody     = strings.Replace(authoredProbeBody, "TestProbe", "TestGate", 1)
	authoredReplacedBody = strings.Replace(authoredProbeBody, "TestProbe", "TestReplaced", 1)
	authoredStagedBody   = strings.Replace(authoredProbeBody, "TestProbe", "TestStaged", 1)
)

const authoredProbeTrace = "trace-authored-probe"

func authoredProbeGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=probe", "GIT_AUTHOR_EMAIL=probe@example.invalid",
		"GIT_COMMITTER_NAME=probe", "GIT_COMMITTER_EMAIL=probe@example.invalid")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func authoredProbeWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// authoredProbeRepo builds a repository holding one committed gating test in
// pkg/ and returns the repository root and that test's absolute path.
func authoredProbeRepo(t *testing.T) (root, committed string) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	authoredProbeGit(t, root, "init", "-q", ".")
	authoredProbeGit(t, root, "config", "index.version", "2")
	authoredProbeGit(t, root, "config", "commit.gpgsign", "false")
	committed = filepath.Join(root, "pkg", "gate_test.go")
	authoredProbeWrite(t, committed, authoredGateBody)
	authoredProbeGit(t, root, "add", "pkg/gate_test.go")
	authoredProbeGit(t, root, "commit", "-q", "--no-verify", "-m", "gate")
	return root, committed
}

func authoredProbeCall(t *testing.T, trace, tool string, args map[string]any) *abi.ToolCall {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	c := inlineCall(tool, string(raw))
	c.Meta = map[string]string{"lane": "implementation"}
	c.TraceID = trace
	return c
}

func authoredProbeEdit(t *testing.T, trace, path, old, repl string) *abi.ToolCall {
	t.Helper()
	return authoredProbeCall(t, trace, "Edit", map[string]any{
		"file_path": filepath.ToSlash(path), "old_string": old, "new_string": repl,
	})
}

func authoredProbeRefused(v abi.Verdict) bool {
	return v.Kind == abi.VerdictDeny && abi.ReasonName(v.Reason) == "TEST_TAMPER_REFUSED"
}

// authoredProbeAuthor drives the new-file Write through the adjudicator and,
// once admitted, performs the write the tool would have performed.
func authoredProbeAuthor(t *testing.T, a *Adjudicator, trace, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	v := a.Adjudicate(context.Background(), authoredProbeCall(t, trace, "Write", map[string]any{
		"file_path": filepath.ToSlash(path), "content": body,
	}))
	if v.Kind == abi.VerdictDeny {
		t.Fatalf("new test file authorship must be admitted, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	authoredProbeWrite(t, path, body)
}

func authoredProbeDirAlias(t *testing.T, alias, target string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		if out, err := exec.Command("cmd", "/c", "mklink", "/J", alias, target).CombinedOutput(); err != nil {
			t.Fatalf("mklink /J: %v: %s", err, out)
		}
		return
	}
	if err := os.Symlink(target, alias); err != nil {
		t.Fatalf("symlink: %v", err)
	}
}

func TestImmunityAdmitsEditOfSessionAuthoredTestFile(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	wip := filepath.Join(root, "pkg", "wip_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, wip, authoredProbeBody)

	admit := func(name string, c *abi.ToolCall) {
		t.Helper()
		if v := a.Adjudicate(context.Background(), c); v.Kind == abi.VerdictDeny {
			t.Fatalf("%s: an edit of a test file authored in this trace must be admitted, got %v/%s", name, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	admit("strengthening edit", authoredProbeEdit(t, authoredProbeTrace, wip, "if rep0 != 1 {", "if rep0 != 1 || t.Name() == \"\" {"))
	// That edit was never applied to disk; the file still holds the authored
	// content, which stays acceptable.
	admit("edit after an unapplied edit", authoredProbeEdit(t, authoredProbeTrace, wip, "\trep0 := 1\n\tif rep0 != 1 {", "\tif t.Name() == \"\" {"))

	// Apply an admitted edit, then keep editing the result.
	next := strings.Replace(authoredProbeBody, "t.Fatal(\"probe\")", "t.Fatalf(\"rep0=%d\", rep0)", 1)
	admit("edit that is then applied", authoredProbeEdit(t, authoredProbeTrace, wip, "t.Fatal(\"probe\")", "t.Fatalf(\"rep0=%d\", rep0)"))
	authoredProbeWrite(t, wip, next)
	admit("edit of the applied result", authoredProbeCall(t, authoredProbeTrace, "Edit", map[string]any{
		"file_path": filepath.ToSlash(wip),
		"edits": []any{
			map[string]any{"old_string": "rep0", "new_string": "got", "replace_all": true},
		},
	}))

	// Line-ending style alone does not break the chain.
	authoredProbeWrite(t, wip, strings.ReplaceAll(next, "\n", "\r\n"))
	admit("edit after a line-ending change", authoredProbeEdit(t, authoredProbeTrace, wip, "rep0 := 1", "rep0 := 2 - 1"))
	admit("whole-file rewrite", authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{
		"file_path": filepath.ToSlash(wip), "content": "package probe\n\nimport \"testing\"\n\nfunc TestProbe(t *testing.T) { t.Log(\"ok\") }\n",
	}))
}

// A slice, not a map: subtest order must be stable so go test -v output
// reproduces byte-for-byte across hosts.
func TestImmunityRefusesCommittedTestRemovedFromIndex(t *testing.T) {
	for _, c := range []struct {
		name     string
		plumbing []string
	}{
		{"update-index --force-remove", []string{"update-index", "--force-remove", "pkg/gate_test.go"}},
		{"read-tree --empty", []string{"read-tree", "--empty"}},
		{"rm --cached -r", []string{"rm", "-q", "--cached", "-r", "pkg"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			root, committed := authoredProbeRepo(t)
			a := New(testImmunityBasePolicy())
			authoredProbeAuthor(t, a, authoredProbeTrace, filepath.Join(root, "pkg", "wip_test.go"), authoredProbeBody)
			authoredProbeGit(t, root, c.plumbing...)
			if _, err := os.Stat(committed); err != nil {
				t.Fatalf("committed test must remain on disk: %v", err)
			}
			weaken := authoredProbeEdit(t, authoredProbeTrace, committed, "\t\tt.Fatal(\"probe\")\n", "")
			if v := a.Adjudicate(context.Background(), weaken); !authoredProbeRefused(v) {
				t.Fatalf("a committed gating test removed from the index must stay refused, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
			}
			overwrite := authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{
				"file_path": filepath.ToSlash(committed), "content": "package probe\n",
			})
			if v := a.Adjudicate(context.Background(), overwrite); !authoredProbeRefused(v) {
				t.Fatalf("overwriting a committed gating test removed from the index must stay refused, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}

func TestImmunityRefusesCaseVariantOfTrackedTest(t *testing.T) {
	root, committed := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())

	// A case-variant spelling of a committed test never earns the exemption.
	variant := filepath.Join(filepath.Dir(committed), "GATE_test.go")
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, variant, "rep0 != 1", "false")); !authoredProbeRefused(v) {
		t.Fatalf("case-variant path of a committed test: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// An authored file whose index entry differs only by letter case is tracked.
	wip := filepath.Join(root, "pkg", "wip_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, wip, authoredProbeBody)
	blob := authoredProbeGit(t, root, "hash-object", "-w", "pkg/wip_test.go")
	authoredProbeGit(t, root, "update-index", "--add", "--cacheinfo", "100644,"+blob+",pkg/WIP_test.go")
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "rep0 != 1", "false")); !authoredProbeRefused(v) {
		t.Fatalf("authored file listed in the index under another case: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// On a case-sensitive filesystem Gate_test.go and gate_test.go are distinct
// files, so authoring a byte-identical twin under the other spelling must not
// earn an edit of the committed one, even once the index no longer lists it.
// Where the filesystem folds case the twin is not a new file at all.
func TestImmunityRefusesCommittedTestBehindCaseTwin(t *testing.T) {
	root, committed := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	twin := filepath.Join(root, "pkg", "Gate_test.go")

	v := a.Adjudicate(context.Background(), authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{
		"file_path": filepath.ToSlash(twin), "content": authoredGateBody,
	}))
	committedInfo, err := os.Stat(committed)
	if err != nil {
		t.Fatal(err)
	}
	if twinInfo, err := os.Stat(twin); err == nil && os.SameFile(committedInfo, twinInfo) {
		if !authoredProbeRefused(v) {
			t.Fatalf("case twin that is the committed file itself: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
		}
	} else {
		if v.Kind == abi.VerdictDeny {
			t.Fatalf("a distinct new test file must be admitted, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
		}
		authoredProbeWrite(t, twin, authoredGateBody)
	}

	authoredProbeGit(t, root, "read-tree", "--empty")
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, committed, "\t\tt.Fatal(\"probe\")\n", "")); !authoredProbeRefused(v) {
		t.Fatalf("committed test behind a case twin: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// A test this trace authored and then committed is a gating test from then on:
// emptying the index afterwards does not bring the exemption back.
func TestImmunityRefusesAuthoredTestOnceCommitted(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	wip := filepath.Join(root, "pkg", "wip_test.go")
	body := strings.Replace(authoredProbeBody, "TestProbe", "TestProbeCommitted", 1)
	authoredProbeAuthor(t, a, authoredProbeTrace, wip, body)

	// One compound step: nothing observes the index while it lists the file.
	authoredProbeGit(t, root, "add", "pkg/wip_test.go")
	authoredProbeGit(t, root, "commit", "-q", "--no-verify", "-m", "wip")
	authoredProbeGit(t, root, "read-tree", "--empty")

	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "\t\tt.Fatal(\"probe\")\n", "")); !authoredProbeRefused(v) {
		t.Fatalf("authored test after commit and index wipe: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// Seeing an authored file listed in an index retires its entry for good, even
// if the file is later unstaged and its staged object pruned away.
func TestImmunityRetiresAuthoredTestOnceIndexed(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	wip := filepath.Join(root, "pkg", "wip_test.go")
	body := strings.Replace(authoredProbeBody, "TestProbe", "TestProbeIndexed", 1)
	authoredProbeAuthor(t, a, authoredProbeTrace, wip, body)
	edit := func() abi.Verdict {
		return a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "if rep0 != 1 {", "if rep0 != 1 || t.Name() == \"\" {"))
	}

	if v := edit(); v.Kind == abi.VerdictDeny {
		t.Fatalf("authored test before staging must be editable, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	authoredProbeGit(t, root, "add", "pkg/wip_test.go")
	if v := edit(); !authoredProbeRefused(v) {
		t.Fatalf("authored test while staged: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	authoredProbeGit(t, root, "update-index", "--force-remove", "pkg/wip_test.go")
	authoredProbeGit(t, root, "prune")
	if v := edit(); !authoredProbeRefused(v) {
		t.Fatalf("authored test after unstaging and pruning: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// The ledger cap belongs to one adjudicator: filling one leaves another able
// to record authorship.
func TestImmunityAuthorshipCapIsPerAdjudicator(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	full := New(testImmunityBasePolicy())
	for i := 0; i < 300; i++ {
		name := filepath.Join(root, "pkg", "fill"+strconv.Itoa(i)+"_test.go")
		v := full.Adjudicate(context.Background(), authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{
			"file_path": filepath.ToSlash(name), "content": authoredProbeBody,
		}))
		if v.Kind == abi.VerdictDeny {
			t.Fatalf("new test file authorship must be admitted, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
		}
	}

	other := New(testImmunityBasePolicy())
	wip := filepath.Join(root, "pkg", "wip_test.go")
	authoredProbeAuthor(t, other, authoredProbeTrace, wip, authoredProbeBody)
	if v := other.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "if rep0 != 1 {", "if rep0 != 1 || t.Name() == \"\" {")); v.Kind == abi.VerdictDeny {
		t.Fatalf("a second adjudicator must still record authorship, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestImmunityRefusesDirectoryAliasOfTrackedTest(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	alias := filepath.Join(root, "alias")
	authoredProbeDirAlias(t, alias, filepath.Join(root, "pkg"))

	through := filepath.Join(alias, "gate_test.go")
	if _, err := os.Stat(through); err != nil {
		t.Fatalf("alias must reach the committed test: %v", err)
	}
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, through, "rep0 != 1", "false")); !authoredProbeRefused(v) {
		t.Fatalf("committed test reached through a directory alias: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// Authorship through an alias is never recorded, so it earns no later edit.
	viaAlias := filepath.Join(alias, "wip_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, viaAlias, authoredProbeBody)
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, viaAlias, "rep0 != 1", "false")); !authoredProbeRefused(v) {
		t.Fatalf("file authored through a directory alias: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestImmunityRefusesSwappedDirectoryUnderAuthoredPath(t *testing.T) {
	root, _ := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())

	// Author a byte-identical twin of the committed test in a scratch directory,
	// then move the committed package into the scratch directory's place.
	twin := filepath.Join(root, "scratch", "gate_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, twin, authoredGateBody)
	if err := os.Rename(filepath.Join(root, "scratch"), filepath.Join(root, "scratch-old")); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(root, "pkg"), filepath.Join(root, "scratch")); err != nil {
		t.Fatal(err)
	}
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, twin, "rep0 != 1", "false")); !authoredProbeRefused(v) {
		t.Fatalf("committed test moved under an authored path: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestImmunityAuthoredEditFailsClosed(t *testing.T) {
	root, committed := authoredProbeRepo(t)
	a := New(testImmunityBasePolicy())
	wip := filepath.Join(root, "pkg", "wip_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, wip, authoredProbeBody)
	path := filepath.ToSlash(wip)

	// On disk but never authored through this adjudicator.
	foreign := filepath.Join(root, "pkg", "foreign_test.go")
	authoredProbeWrite(t, foreign, authoredProbeBody)

	// Authored here, then replaced out of band.
	replaced := filepath.Join(root, "pkg", "replaced_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, replaced, authoredReplacedBody)
	authoredProbeWrite(t, replaced, strings.Replace(authoredReplacedBody, "rep0 != 1", "rep0 != 7", 1))

	// Authored here, then staged.
	staged := filepath.Join(root, "pkg", "staged_test.go")
	authoredProbeAuthor(t, a, authoredProbeTrace, staged, authoredStagedBody)
	authoredProbeGit(t, root, "add", "pkg/staged_test.go")

	// Control: the authored file is editable, so each refusal below is earned.
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "if rep0 != 1 {", "if rep0 != 1 || t.Name() == \"\" {")); v.Kind == abi.VerdictDeny {
		t.Fatalf("control: authored test must be editable, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	cases := map[string]*abi.ToolCall{
		"committed test never authored here":          authoredProbeEdit(t, authoredProbeTrace, committed, "\t\tt.Fatal(\"probe\")\n", ""),
		"untracked test not authored in this trace":   authoredProbeEdit(t, authoredProbeTrace, foreign, "rep0 != 1", "false"),
		"authored test edited from another trace":     authoredProbeEdit(t, "trace-other", wip, "rep0 != 1", "false"),
		"authored test edited with no trace":          authoredProbeEdit(t, "", wip, "rep0 != 1", "false"),
		"authored test replaced out of band":          authoredProbeEdit(t, authoredProbeTrace, replaced, "rep0 != 7", "false"),
		"authored test that is now staged":            authoredProbeEdit(t, authoredProbeTrace, staged, "rep0 != 1", "false"),
		"edit that renames a test into TestMain":      authoredProbeEdit(t, authoredProbeTrace, wip, "TestProbe(t *testing.T)", "TestMain(m *testing.M)"),
		"edit that splices TestMain from a fragment":  authoredProbeEdit(t, authoredProbeTrace, wip, "Probe(", "Main("),
		"edit that adds os.Exit":                      authoredProbeEdit(t, authoredProbeTrace, wip, "rep0 := 1", "rep0 := 1\n\tos.Exit(0)"),
		"edit whose anchor is absent":                 authoredProbeEdit(t, authoredProbeTrace, wip, "no such anchor", "x"),
		"edit with no modelled content":               authoredProbeCall(t, authoredProbeTrace, "Edit", map[string]any{"file_path": path}),
		"write that empties the file":                 authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{"file_path": path, "content": ""}),
		"delete tool":                                 authoredProbeCall(t, authoredProbeTrace, "delete_file", map[string]any{"path": path}),
		"shell write":                                 authoredProbeCall(t, authoredProbeTrace, "Bash", map[string]any{"command": "echo x > " + path}),
		"old text supplied under two spellings":       authoredProbeCall(t, authoredProbeTrace, "Edit", map[string]any{"file_path": path, "old_string": "rep0 != 1", "oldString": "t.Fatal(\"probe\")", "new_string": "rep0 != 1"}),
		"new text supplied under two spellings":       authoredProbeCall(t, authoredProbeTrace, "Edit", map[string]any{"file_path": path, "old_string": "rep0 != 1", "new_string": "rep0 != 1", "newString": "false"}),
		"whole body alongside a replacement":          authoredProbeCall(t, authoredProbeTrace, "Edit", map[string]any{"file_path": path, "content": authoredProbeBody, "old_string": "rep0 != 1", "new_string": "false"}),
		"whole body under an unmodelled second field": authoredProbeCall(t, authoredProbeTrace, "Write", map[string]any{"file_path": path, "content": authoredProbeBody, "file_text": "package probe\n"}),
	}
	for name, c := range cases {
		if v := a.Adjudicate(context.Background(), c); !authoredProbeRefused(v) {
			t.Errorf("%s: want TEST_TAMPER_REFUSED, got %v/%s", name, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// The ledger does not outlive the run.
	a.ResetRun()
	if v := a.Adjudicate(context.Background(), authoredProbeEdit(t, authoredProbeTrace, wip, "if rep0 != 1 {", "if rep0 != 1 || t.Name() == \"\" {")); !authoredProbeRefused(v) {
		t.Errorf("authored test after the run was reset: want TEST_TAMPER_REFUSED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}
