package architest

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The module zip is what every `go install` user downloads and compiles, and
// cmd/go refuses to build one whose files sum to more than 500 MiB uncompressed.
// That refusal is silent for users: @latest keeps resolving to the last tag
// that fit.
//
// The module is the whole repository minus nested modules and export-ignore
// paths, so it grows with every receipt, witness dump and vendored header
// anyone commits. These budgets are a ratchet set just above today's size, so
// the commit that adds bulk is the one that goes red. Lower them whenever a
// carve lands; raise them only for product Go, in the commit that needs it.
const (
	moduleZipProxyCapBytes  = 500 << 20
	moduleZipBudgetBytes    = 375 << 20
	moduleZipNonGoBudget    = 190 << 20
	moduleZipNonGoFileLimit = 5 << 20
)

const moduleZipRemedy = `Everything tracked in the root module ships to every go install user. Evidence,
research, media and fixtures belong under a tree that is out of the module zip (its own
go.mod, as docs/research and visuals have), and one artifact should be a receipt or
summary, not a raw dump: over the per-file limit, commit a pointer (path, sha256, size).
Raise a budget only for product Go that has genuinely grown, in the same commit, with
the reason.`

func TestRootModuleZipBudget(t *testing.T) {
	root, err := gitOutput(".", "rev-parse", "--show-toplevel")
	if err != nil {
		if os.Getenv("CI") == "" {
			t.Skipf("not a git checkout: %v", err)
		}
		t.Fatalf("CI must measure the module zip, but this is not a git checkout: %v", err)
	}
	repo := strings.TrimSpace(string(root))

	tree, err := gitOutput(repo, "ls-tree", "-r", "-l", "-z", "HEAD")
	if err != nil {
		t.Fatalf("git ls-tree: %v", err)
	}
	sizes := map[string]int64{}
	var paths []string
	for _, rec := range bytes.Split(tree, []byte{0}) {
		meta, path, ok := strings.Cut(string(rec), "\t")
		if !ok {
			continue
		}
		fields := strings.Fields(meta)
		if len(fields) != 4 || fields[1] != "blob" {
			continue
		}
		n, err := strconv.ParseInt(fields[3], 10, 64)
		if err != nil {
			t.Fatalf("blob size for %s: %v", path, err)
		}
		sizes[path] = n
		paths = append(paths, path)
	}

	nested := map[string]bool{}
	for _, p := range paths {
		if dir, ok := strings.CutSuffix(p, "/go.mod"); ok {
			nested[dir] = true
		}
	}

	ignored, err := exportIgnored(repo, paths)
	if err != nil {
		t.Fatalf("git check-attr: %v", err)
	}

	var total, nonGo int64
	byDir := map[string]int64{}
	nonGoByDir := map[string]int64{}
	var oversized []string
	sawRootGoMod := false
	for _, p := range paths {
		if ignored[p] || inNestedModule(p, nested) {
			continue
		}
		if p == "go.mod" {
			sawRootGoMod = true
		}
		total += sizes[p]
		byDir[topDirs(p)] += sizes[p]
		if !strings.HasSuffix(p, ".go") {
			nonGo += sizes[p]
			nonGoByDir[topDirs(p)] += sizes[p]
			if sizes[p] > moduleZipNonGoFileLimit {
				oversized = append(oversized, fmt.Sprintf("  %10s  %s", mib(sizes[p]), p))
			}
		}
	}

	if !sawRootGoMod || total < 1<<20 {
		t.Fatalf("measured nothing: root go.mod seen=%v, total=%d bytes over %d paths", sawRootGoMod, total, len(paths))
	}
	t.Logf("root module source tree: %s of %s budget (proxy cap %s); non-Go %s of %s",
		mib(total), mib(moduleZipBudgetBytes), mib(moduleZipProxyCapBytes), mib(nonGo), mib(moduleZipNonGoBudget))

	if len(oversized) > 0 {
		t.Errorf("non-Go file(s) over %s inside the installable module:\n%s\n%s",
			mib(moduleZipNonGoFileLimit), strings.Join(oversized, "\n"), moduleZipRemedy)
	}
	if nonGo > moduleZipNonGoBudget {
		t.Errorf("non-Go content in the root module is %s, over the %s budget.\n%s\nLargest non-Go directories:\n%s",
			mib(nonGo), mib(moduleZipNonGoBudget), moduleZipRemedy, largest(nonGoByDir, 12))
	}
	if total > moduleZipBudgetBytes {
		t.Errorf("root module source tree is %s, over the %s budget (proxy cap %s).\n%s\nLargest directories:\n%s",
			mib(total), mib(moduleZipBudgetBytes), mib(moduleZipProxyCapBytes), moduleZipRemedy, largest(byDir, 12))
	}
}

func gitOutput(dir string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	return cmd.Output()
}

// exportIgnored asks git itself, so the test and `git archive` (which cmd/go
// uses to build the zip) agree on every attribute pattern.
func exportIgnored(repo string, paths []string) (map[string]bool, error) {
	cmd := exec.Command("git", "check-attr", "-z", "--stdin", "export-ignore")
	cmd.Dir = repo
	cmd.Stdin = strings.NewReader(strings.Join(paths, "\x00") + "\x00")
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	fields := bytes.Split(out, []byte{0})
	ignored := map[string]bool{}
	for i := 0; i+2 < len(fields); i += 3 {
		if string(fields[i+2]) == "set" {
			ignored[string(fields[i])] = true
		}
	}
	return ignored, nil
}

func inNestedModule(path string, nested map[string]bool) bool {
	for dir := path; ; {
		i := strings.LastIndexByte(dir, '/')
		if i < 0 {
			return false
		}
		dir = dir[:i]
		if nested[dir] {
			return true
		}
	}
}

func topDirs(path string) string {
	parts := strings.SplitN(path, "/", 3)
	if len(parts) < 3 {
		return parts[0]
	}
	return parts[0] + "/" + parts[1]
}

func mib(n int64) string {
	return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
}

func largest(byDir map[string]int64, n int) string {
	dirs := make([]string, 0, len(byDir))
	for d := range byDir {
		dirs = append(dirs, d)
	}
	sort.Slice(dirs, func(i, j int) bool { return byDir[dirs[i]] > byDir[dirs[j]] })
	if len(dirs) > n {
		dirs = dirs[:n]
	}
	var b strings.Builder
	for _, d := range dirs {
		fmt.Fprintf(&b, "  %10s  %s\n", mib(byDir[d]), d)
	}
	return b.String()
}
