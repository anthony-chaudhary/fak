package hooks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// budgetFixture describes one staged commit. index holds the blob sizes the commit ships;
// worktree holds the files on disk, which may differ from the index or be absent. The fail*
// flags make one git read exit non-zero so each fail-closed branch is reached.
type budgetFixture struct {
	index         map[string]int64
	worktree      map[string]int
	cachedAttr    string
	treeish       string
	indexPaths    []string          // the tracked listing (ls-files)
	unlisted      []string          // staged names the tree listing lacks
	renames       map[string]string // new path -> old path, staged as R
	binary        map[string]bool   // staged blobs with a NUL in their first 8000 bytes
	merging       bool              // MERGE_HEAD exists in the git dir
	failNameList  bool
	failRevParse  bool
	emptyRevParse bool
	failTracked   bool
	failCheckAttr bool
	failWriteTree bool
	failLsTree    bool
	failGrep      bool
}

func (f budgetFixture) diff(t *testing.T) *StagedDiff {
	t.Helper()
	root := t.TempDir()
	for rel, n := range f.worktree {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(strings.Repeat("a", n)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if f.merging {
		if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, ".git", "MERGE_HEAD"), []byte("0123\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	run := func(_ context.Context, _ string, args ...string) (string, int, error) {
		switch args[0] {
		case "write-tree":
			if f.failWriteTree {
				return "", 128, nil
			}
			return "0123456789abcdef0123456789abcdef01234567\n", 0, nil
		case "ls-tree":
			if f.failLsTree {
				return "", 128, nil
			}
			var b strings.Builder
			for p, n := range f.index {
				if n < 0 {
					// A gitlink: git lists it as a commit object with no size.
					fmt.Fprintf(&b, "160000 commit 0000000000000000000000000000000000000000 -\t%s\x00", p)
					continue
				}
				fmt.Fprintf(&b, "100644 blob 0000000000000000000000000000000000000000 %10d\t%s\x00", n, p)
			}
			return b.String(), 0, nil
		case "rev-parse":
			if f.failRevParse {
				return "", 128, nil
			}
			if f.emptyRevParse {
				return "\n", 0, nil
			}
			return ".git/MERGE_HEAD\n", 0, nil
		case "ls-files":
			if f.failTracked {
				return "", 128, nil
			}
			return strings.Join(f.indexPaths, "\x00") + "\x00", 0, nil
		case "check-attr":
			if f.failCheckAttr {
				return "", 128, nil
			}
			if !containsArg(args, "--cached") {
				t.Errorf("check-attr must read the index with --cached; argv %v", args)
			}
			// git answers one triple per queried path, so an unanswered path is distinguishable.
			var b strings.Builder
			for _, p := range afterDoubleDash(args) {
				value := "unspecified"
				if strings.Contains(f.cachedAttr, p+"\x00export-ignore\x00set\x00") {
					value = "set"
				}
				fmt.Fprintf(&b, "%s\x00export-ignore\x00%s\x00", p, value)
			}
			return b.String(), 0, nil
		case "--literal-pathspecs":
			if !containsArg(args, "-I") || !containsArg(args, "-z") {
				t.Errorf("text check must be git grep -I -z over the index; argv %v", args)
			}
			if f.failGrep {
				return "", 128, nil
			}
			var b strings.Builder
			for _, p := range afterDoubleDash(args) {
				if !f.binary[p] {
					b.WriteString(p + "\x00")
				}
			}
			return b.String(), 0, nil
		case "diff":
			if !containsArg(args, "-z") || !containsArg(args, "--name-status") || !containsArg(args, "-M") {
				t.Errorf("staged scan must be NUL-delimited --name-status with -M; argv %v", args)
				return "", 1, nil
			}
			if f.failNameList {
				return "", 128, nil
			}
			names := append(f.stagedNames(), f.unlisted...)
			sort.Strings(names)
			var b strings.Builder
			for _, p := range names {
				if old, ok := f.renames[p]; ok {
					fmt.Fprintf(&b, "R100\x00%s\x00%s\x00", old, p)
					continue
				}
				fmt.Fprintf(&b, "A\x00%s\x00", p)
			}
			return b.String(), 0, nil
		}
		return "", 1, nil
	}
	return &StagedDiff{
		Root:              root,
		run:               run,
		Treeish:           f.treeish,
		AddedRenamedPaths: f.stagedNames(),
		IndexPaths:        f.indexPaths,
		fileCache:         map[string]fileEntry{},
	}
}

// stagedNames is the sorted set of index paths the commit adds or renames.
func (f budgetFixture) stagedNames() []string {
	var names []string
	for p := range f.index {
		names = append(names, p)
	}
	sort.Strings(names)
	return names
}

// afterDoubleDash returns the pathspecs that follow the -- separator in argv.
func afterDoubleDash(args []string) []string {
	for i, a := range args {
		if a == "--" {
			return args[i+1:]
		}
	}
	return nil
}

func containsArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

// budgetDiff is the common case: worktree and index agree on every file's size.
// attrOut is the index's export-ignore answer.
func budgetDiff(t *testing.T, sizes map[string]int, attrOut string, treeish string, indexPaths ...string) *StagedDiff {
	t.Helper()
	index := map[string]int64{}
	for rel, n := range sizes {
		index[rel] = int64(n)
	}
	return budgetFixture{index: index, worktree: sizes, cachedAttr: attrOut, treeish: treeish, indexPaths: indexPaths}.diff(t)
}

// budgetFindings returns only the commit-level findings (no file), which is where the
// per-commit budget and the fail-closed refusals report.
func budgetFindings(t *testing.T, d *StagedDiff) []Finding {
	t.Helper()
	got, err := gateFileAdmission(d)
	if err != nil {
		t.Fatal(err)
	}
	var out []Finding
	for _, f := range got {
		if f.File == "" {
			out = append(out, f)
		}
	}
	return out
}

// withBudget sets the per-commit budget for one test and restores it afterwards.
func withBudget(t *testing.T, bytes int64) {
	t.Helper()
	orig := commitShippedBudget
	t.Cleanup(func() { commitShippedBudget = orig })
	commitShippedBudget = bytes
}

func TestFileAdmission_CommitShippedBudget_RefusesBulkAddAndNamesLargest(t *testing.T) {
	d := budgetDiff(t, map[string]int{"assets/big.bin": 300 * 1024, "assets/small.bin": 1024}, "", ":")
	got := budgetFindings(t, d)
	if len(got) != 1 {
		t.Fatalf("want one commit-level budget finding, got %+v", got)
	}
	for _, want := range []string{"assets/big.bin", "export-ignore", "sha256", "per commit"} {
		if !strings.Contains(got[0].Detail, want) {
			t.Errorf("budget finding must contain %q; got %q", want, got[0].Detail)
		}
	}
}

func TestFileAdmission_CommitShippedBudget_AdmitsOrdinaryCommit(t *testing.T) {
	d := budgetDiff(t, map[string]int{"assets/a.bin": 100 * 1024, "assets/b.bin": 100 * 1024}, "", ":")
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("200 KiB across two files is under the 256 KiB budget; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_BoundaryIsInclusive(t *testing.T) {
	withBudget(t, 8)

	atCap := budgetDiff(t, map[string]int{"assets/blob.bin": 8}, "", ":")
	if got := budgetFindings(t, atCap); len(got) != 0 {
		t.Errorf("a commit at exactly the budget must be admitted; got %+v", got)
	}
	overCap := budgetDiff(t, map[string]int{"assets/blob.bin": 9}, "", ":")
	if got := budgetFindings(t, overCap); len(got) != 1 {
		t.Errorf("a commit one byte over the budget must be refused; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_ExportIgnoredDoesNotCount(t *testing.T) {
	d := budgetDiff(t, map[string]int{"docs/receipts/big.bin": 300 * 1024}, "docs/receipts/big.bin\x00export-ignore\x00set\x00", ":")
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("export-ignored bytes never reach the module zip and must not count; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_NestedModuleDoesNotCount(t *testing.T) {
	d := budgetDiff(t, map[string]int{"docs/nightrun/data/big.bin": 300 * 1024}, "", ":",
		"docs/nightrun/go.mod", "docs/nightrun/data/big.bin")
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("bytes under a nested go.mod are a separate module and must not count; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_GoSourceDoesNotCount(t *testing.T) {
	d := budgetDiff(t, map[string]int{"internal/tables/generated.go": 300 * 1024}, "", ":")
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("Go sources are not the non-Go budget; got %+v", got)
	}
}

// A .go file is exempt only while its staged blob is text. A binary blob with a .go name ships
// the same bytes as any other file, so it is billed.
func TestFileAdmission_CommitShippedBudget_BinaryGoIsBilled(t *testing.T) {
	d := budgetFixture{
		index:   map[string]int64{"internal/blob/data.go": 300 * 1024},
		binary:  map[string]bool{"internal/blob/data.go": true},
		treeish: ":",
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "internal/blob/data.go") {
		t.Fatalf("a binary blob named *.go must be billed like any file; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_LargeTextGoIsAdmitted(t *testing.T) {
	d := budgetFixture{
		index:   map[string]int64{"internal/tables/generated.go": 300 * 1024},
		treeish: ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("a large text .go is source, not shipped bytes; got %+v", got)
	}
}

// A rename bills its full new blob, like an addition: moving shipped bytes is not free. The
// refusal names the env override that admits a deliberate pure move for one commit.
func TestFileAdmission_CommitShippedBudget_PureRenameBillsFullSize(t *testing.T) {
	cases := []struct {
		name    string
		size    int64
		refused bool
	}{
		{"pure move of a 300 KiB shipped asset is refused", 300 * 1024, true},
		{"pure move of a 100 KiB shipped asset is admitted", 100 * 1024, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := budgetFixture{
				index:   map[string]int64{"assets/new.bin": tc.size},
				renames: map[string]string{"assets/new.bin": "assets/old.bin"},
				treeish: ":",
			}.diff(t)
			got := budgetFindings(t, d)
			if !tc.refused {
				if len(got) != 0 {
					t.Fatalf("a move under the budget must be admitted; got %+v", got)
				}
				return
			}
			if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
				t.Fatalf("a pure move must bill its full size and be refused; got %+v", got)
			}
			if !strings.Contains(got[0].Detail, "FAK_COMMIT_SHIPPED_BYTES=307200") || !strings.Contains(got[0].Detail, "pure move") {
				t.Errorf("refusal must name the one-commit override for a pure move; got %q", got[0].Detail)
			}
		})
	}
}

func TestFileAdmission_CommitShippedBudget_PureRenameAdmittedWhenBudgetRaised(t *testing.T) {
	withBudget(t, 400*1024)
	d := budgetFixture{
		index:   map[string]int64{"assets/new.bin": 300 * 1024},
		renames: map[string]string{"assets/new.bin": "assets/old.bin"},
		treeish: ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("a deliberate pure move admitted by raising the budget must pass; got %+v", got)
	}
}

// The budget is skipped only while MERGE_HEAD exists; the same bulk commit is refused without it.
func TestFileAdmission_CommitShippedBudget_MergeInProgressIsSkipped(t *testing.T) {
	bulk := map[string]int64{"assets/big.bin": 300 * 1024}
	if got := budgetFindings(t, budgetFixture{index: bulk, treeish: ":"}.diff(t)); len(got) != 1 {
		t.Fatalf("control: the bulk commit must be refused outside a merge; got %+v", got)
	}
	if got := budgetFindings(t, budgetFixture{index: bulk, treeish: ":", merging: true}.diff(t)); len(got) != 0 {
		t.Fatalf("a merge commit must not be budgeted again; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_OversizedWorktreeIndexSmallAdmitted(t *testing.T) {
	// The inverse divergence: the index blob is small, so the commit ships little.
	d := budgetFixture{
		index:    map[string]int64{"assets/a.bin": 1024},
		worktree: map[string]int{"assets/a.bin": 300 * 1024},
		treeish:  ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("the budget sizes what the commit ships, not a larger worktree copy; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_SizesIndexBlobNotWorktree(t *testing.T) {
	// The commit ships the 300 KiB index blob even though the worktree copy was truncated.
	d := budgetFixture{
		index:    map[string]int64{"assets/big.bin": 300 * 1024},
		worktree: map[string]int{"assets/big.bin": 0},
		treeish:  ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 1 {
		t.Fatalf("a truncated worktree copy must not hide the indexed bytes; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_DeletedWorktreeCopyStillCounts(t *testing.T) {
	d := budgetFixture{
		index:    map[string]int64{"assets/big.bin": 300 * 1024},
		worktree: map[string]int{},
		treeish:  ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 1 {
		t.Fatalf("a staged file deleted from the worktree still ships its index blob; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_WorktreeOnlyAttrDoesNotExempt(t *testing.T) {
	// An uncommitted worktree edit adds export-ignore; the commit does not carry it, so the
	// bytes still ship to the zip and must count.
	d := budgetFixture{
		index:      map[string]int64{"docs/receipts/big.bin": 300 * 1024},
		worktree:   map[string]int{"docs/receipts/big.bin": 300 * 1024},
		cachedAttr: "",
		treeish:    ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 1 {
		t.Fatalf("an uncommitted export-ignore must not exempt a staged commit; got %+v", got)
	}
}

// Each git read on the budget path must refuse the commit, not skip the check. The small file
// keeps the budget itself from refusing, so the refusal can only come from the failed read.
func TestFileAdmission_CommitShippedBudget_FailedReadsFailClosed(t *testing.T) {
	cases := []struct {
		name    string
		fixture budgetFixture
		wantIn  string
	}{
		{"name listing", budgetFixture{failNameList: true}, "list the staged names"},
		{"merge probe", budgetFixture{failRevParse: true}, "read whether a merge is in progress"},
		{"merge probe empty", budgetFixture{emptyRevParse: true}, "read whether a merge is in progress"},
		{"tracked listing", budgetFixture{failTracked: true}, "list the tracked paths"},
		{"export-ignore query", budgetFixture{failCheckAttr: true}, "read export-ignore"},
		{"text check", budgetFixture{failGrep: true}, "check which staged .go files are text"},
		{"write-tree", budgetFixture{failWriteTree: true}, "size the staged blobs"},
		{"tree listing", budgetFixture{failLsTree: true}, "size the staged blobs"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.fixture
			// A staged .go is needed so the text check runs; both files are small, so only a
			// failed read can refuse.
			f.index = map[string]int64{"assets/a.bin": 1024, "internal/small.go": 1024}
			f.worktree = map[string]int{"assets/a.bin": 1024}
			f.treeish = ":"
			got := budgetFindings(t, f.diff(t))
			if len(got) != 1 {
				t.Fatalf("a failed %s read must refuse the commit; got %+v", tc.name, got)
			}
			if !strings.Contains(got[0].Detail, tc.wantIn) || !strings.Contains(got[0].Detail, "fail closed") {
				t.Errorf("refusal must name the failed read and say fail closed; got %q", got[0].Detail)
			}
		})
	}
}

func TestFileAdmission_CommitShippedBudget_FailedSizeLookupFailsClosed(t *testing.T) {
	d := budgetFixture{
		index:         map[string]int64{"assets/a.bin": 1024},
		worktree:      map[string]int{"assets/a.bin": 1024},
		treeish:       ":",
		failWriteTree: true,
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 {
		t.Fatalf("an unreadable size must refuse the commit, not skip the file; got %+v", got)
	}
	if !strings.Contains(got[0].Detail, "size the staged blobs") || !strings.Contains(got[0].Detail, "fail closed") {
		t.Errorf("refusal must say the size read failed closed; got %q", got[0].Detail)
	}
}

func TestFileAdmission_CommitShippedBudget_PathMissingFromListingFailsClosed(t *testing.T) {
	d := budgetFixture{
		index:    map[string]int64{"assets/a.bin": 1024},
		worktree: map[string]int{"assets/a.bin": 1024},
		treeish:  ":",
		unlisted: []string{"assets/unlisted.bin"},
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "assets/unlisted.bin") {
		t.Fatalf("a staged path the tree listing lacks must be refused as unsized; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_GitlinkShipsNoBytes(t *testing.T) {
	d := budgetFixture{
		index:   map[string]int64{"vendor/sub": -1},
		treeish: ":",
	}.diff(t)
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("a gitlink carries no blob bytes and must not be refused as unsized; got %+v", got)
	}
}

// Under core.quotepath=true git C-quotes non-ASCII names in the name-status listing, so the
// budget must look sizes up by the raw NUL-delimited names. Real repos, real staged index.
func TestFileAdmission_CommitShippedBudget_NonASCIINameUnderQuotepath(t *testing.T) {
	cases := []struct {
		name    string
		size    int
		refused bool
	}{
		{"naïve big.bin", 300 * 1024, true},
		{"naïve small.bin", 100 * 1024, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			gitFixture(t, root, "init", "-q")
			gitFixture(t, root, "config", "core.quotepath", "true")
			if err := os.WriteFile(filepath.Join(root, tc.name), []byte(strings.Repeat("a", tc.size)), 0o644); err != nil {
				t.Fatal(err)
			}
			gitFixture(t, root, "add", "--", tc.name)
			d, err := readStagedDiffWith(context.Background(), realRunner, root)
			if err != nil {
				t.Fatal(err)
			}
			got := budgetFindings(t, d)
			if tc.refused {
				if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds") || !strings.Contains(got[0].Detail, tc.name) {
					t.Fatalf("a bulk non-ASCII name must be refused by the budget, not as unsized; got %+v", got)
				}
				return
			}
			if len(got) != 0 {
				t.Fatalf("an ordinary non-ASCII name must be admitted; got %+v", got)
			}
		})
	}
}

// An export-ignored non-ASCII bulk file never reaches the zip, so it must not count. The
// attribute lookup has to match the raw name under core.quotepath=true, not the C-quoted one.
func TestFileAdmission_CommitShippedBudget_ExportIgnoredNonASCIIUnderQuotepath(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init", "-q")
	gitFixture(t, root, "config", "core.quotepath", "true")
	const name = "bulké.bin"
	if err := os.WriteFile(filepath.Join(root, ".gitattributes"), []byte(name+" export-ignore\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, name), []byte(strings.Repeat("a", 300*1024)), 0o644); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "--", ".gitattributes", name)
	d, err := readStagedDiffWith(context.Background(), realRunner, root)
	if err != nil {
		t.Fatal(err)
	}
	if got := budgetFindings(t, d); len(got) != 0 {
		t.Fatalf("an export-ignored non-ASCII bulk file must be admitted; got %+v", got)
	}
}

// Renames out of a path that never shipped are billed in full like any other addition.
func TestFileAdmission_CommitShippedBudget_RenameFromExportIgnoredSourceBillsFull(t *testing.T) {
	d := budgetFixture{
		index:      map[string]int64{"assets/new.bin": 300 * 1024},
		renames:    map[string]string{"assets/new.bin": "docs/receipts/old.bin"},
		cachedAttr: "docs/receipts/old.bin\x00export-ignore\x00set\x00",
		treeish:    ":",
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
		t.Fatalf("moving an export-ignored file into a shipped path must bill it in full; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_RenameFromNestedModuleSourceBillsFull(t *testing.T) {
	d := budgetFixture{
		index:      map[string]int64{"assets/new.bin": 300 * 1024},
		renames:    map[string]string{"assets/new.bin": "docs/nightrun/data/old.bin"},
		indexPaths: []string{"docs/nightrun/go.mod", "docs/nightrun/data/old.bin"},
		treeish:    ":",
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
		t.Fatalf("moving a nested-module file into a shipped path must bill it in full; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_RenameFromTextGoSourceBillsFull(t *testing.T) {
	d := budgetFixture{
		index:   map[string]int64{"assets/generated.bin": 300 * 1024},
		renames: map[string]string{"assets/generated.bin": "internal/tables/generated.go"},
		treeish: ":",
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
		t.Fatalf("renaming a text .go into a shipped path must bill it in full; got %+v", got)
	}
}

// Real repo: a binary .go moved to a shipped path is billed at its full size, not credited.
func TestFileAdmission_CommitShippedBudget_RenameFromBinaryGoBilledInRealRepo(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init", "-q")
	old := filepath.Join(root, "internal", "blob", "old.go")
	if err := os.MkdirAll(filepath.Dir(old), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(old, make([]byte, 300*1024), 0o644); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "--", "internal/blob/old.go")
	gitFixture(t, root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "base")
	if err := os.MkdirAll(filepath.Join(root, "assets"), 0o755); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "mv", "internal/blob/old.go", "assets/new.bin")
	d, err := readStagedDiffWith(context.Background(), realRunner, root)
	if err != nil {
		t.Fatal(err)
	}
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
		t.Fatalf("a move of a billable binary .go must bill its full size; got %+v", got)
	}
}

func TestFileAdmission_CommitShippedBudget_RenameFromBinaryGoBillsFullSize(t *testing.T) {
	d := budgetFixture{
		index:   map[string]int64{"assets/new.bin": 300 * 1024},
		renames: map[string]string{"assets/new.bin": "internal/blob/old.go"},
		binary:  map[string]bool{"internal/blob/old.go": true},
		treeish: ":",
	}.diff(t)
	got := budgetFindings(t, d)
	if len(got) != 1 || !strings.Contains(got[0].Detail, "commit adds 300 KiB") {
		t.Fatalf("a move of a billable binary .go must bill its full size; got %+v", got)
	}
}
