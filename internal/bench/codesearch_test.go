package bench

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type codeSearchFixtureFile struct {
	name string
	body string
}

var codeSearchFixture = []codeSearchFixtureFile{
	{"a.go", "package p\nfunc Alpha(){ sharedNeedle() }\n"},
	{"b.go", "package p\n// sharedNeedle appears here\nfunc Beta(){}\n"},
	{"c.go", "package p\nfunc Gamma(){ short() }\n"},
	{"d.go", "package p\nconst Unicode = \"καφές\"\n"},
	{"e.go", "package p\nfunc Empty(){}\n"},
	{"f.go", "package p\nfunc Again(){ sharedNeedle() }\n"},
}

var codeSearchQueries = []string{"sharedNeedle", "short", "καφές", "absent"}

var codeSearchExpected = map[string][]string{
	"sharedNeedle": {"a.go:2", "b.go:2", "f.go:2"},
	"short":        {"c.go:2"},
	"καφές":        {"d.go:2"},
	"absent":       nil,
}

func materializeCodeSearchFixture(tb testing.TB) string {
	tb.Helper()
	dir := tb.TempDir()
	for _, f := range codeSearchFixture {
		if err := os.WriteFile(filepath.Join(dir, f.name), []byte(f.body), 0o600); err != nil {
			tb.Fatal(err)
		}
	}
	return dir
}

func parseCodeSearchLocations(root, output string) ([]string, error) {
	var locations []string
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(strings.TrimSuffix(line, "\r"), ":", 3)
		if len(parts) < 2 {
			return nil, strconv.ErrSyntax
		}
		path := strings.TrimPrefix(filepath.ToSlash(parts[0]), "./")
		if filepath.IsAbs(path) {
			var err error
			path, err = filepath.Rel(root, path)
			if err != nil {
				return nil, err
			}
		}
		locations = append(locations, filepath.ToSlash(path)+":"+parts[1])
	}
	sort.Strings(locations)
	return locations, nil
}

func runCodeSearchCommand(tb testing.TB, dir, name string, args ...string) string {
	tb.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if exit, ok := err.(*exec.ExitError); ok && exit.ExitCode() == 1 {
		return ""
	}
	if err != nil {
		tb.Fatalf("%s: %v: %s", name, err, out)
	}
	return string(out)
}

func assertCodeSearchArm(tb testing.TB, name string, search func(string) string) {
	tb.Helper()
	for _, query := range codeSearchQueries {
		got := search(query)
		want := codeSearchExpected[query]
		if strings.Join(strings.Fields(got), " ") != strings.Join(want, " ") {
			tb.Fatalf("%s query %q: got %q want %q", name, query, got, want)
		}
	}
}

func TestCodeSearchExternalCommandArms(t *testing.T) {
	dir := materializeCodeSearchFixture(t)
	if _, err := exec.LookPath("rg"); err != nil {
		t.Fatal("ripgrep unavailable")
	}
	assertCodeSearchArm(t, "ripgrep", func(query string) string {
		out := runCodeSearchCommand(t, dir, "rg", "--fixed-strings", "--line-number", "--with-filename", "--sort", "path", query, ".")
		locations, err := parseCodeSearchLocations(dir, out)
		if err != nil {
			t.Fatal(err)
		}
		return strings.Join(locations, " ")
	})

	if _, err := exec.LookPath("git"); err != nil {
		t.Fatal("git unavailable")
	}
	runCodeSearchCommand(t, dir, "git", "init", "-q")
	runCodeSearchCommand(t, dir, "git", "add", "--", "a.go", "b.go", "c.go", "d.go", "e.go", "f.go")
	assertCodeSearchArm(t, "git grep", func(query string) string {
		out := runCodeSearchCommand(t, dir, "git", "grep", "-n", "-F", query, "--", "*.go")
		locations, err := parseCodeSearchLocations(dir, out)
		if err != nil {
			t.Fatal(err)
		}
		return strings.Join(locations, " ")
	})
}

func BenchmarkCodeSearchCommandArms(b *testing.B) {
	dir := materializeCodeSearchFixture(b)
	runCodeSearchCommand(b, dir, "git", "init", "-q")
	runCodeSearchCommand(b, dir, "git", "add", "--", "a.go", "b.go", "c.go", "d.go", "e.go", "f.go")
	arms := []struct {
		name string
		find func(string) string
	}{
		{"ripgrep", func(q string) string {
			return runCodeSearchCommand(b, dir, "rg", "--fixed-strings", "--line-number", "--with-filename", "--sort", "path", q, ".")
		}},
		{"git-grep", func(q string) string { return runCodeSearchCommand(b, dir, "git", "grep", "-n", "-F", q, "--", "*.go") }},
	}
	for _, arm := range arms {
		b.Run(arm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for _, q := range codeSearchQueries {
					arm.find(q)
				}
			}
		})
	}
}
