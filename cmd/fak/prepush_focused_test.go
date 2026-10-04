package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const focusedHelperEnv = "FAK_FOCUSED_PREPUSH_HELPER"

// The hook finds this Go test executable under the name fak. Dispatch before
// testing parses the CLI arguments; no shell launcher or second Go build is needed.
func init() {
	name := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	if os.Getenv(focusedHelperEnv) != "1" || (name != "fak" && name != "go") {
		return
	}
	trace, err := os.OpenFile(os.Getenv("FAK_FOCUSED_PREPUSH_TRACE"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(90)
	}
	if err := json.NewEncoder(trace).Encode(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(90)
	}
	if err := trace.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(90)
	}
	forbidden := func(step string) {
		fmt.Fprintln(os.Stderr, "FOCUSED_FORBIDDEN_STEP", step)
		os.Exit(91)
	}
	if name == "go" {
		forbidden("compile a fallback verifier")
	}
	prepushExtractTip = func(string, string) (string, error) { forbidden("archive"); return "", nil }
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		forbidden("list graph")
		return nil, nil, 0, nil
	}
	prepushListTestOnly = func(string, []string) map[string]bool { forbidden("list test-only"); return nil }
	prepushBuild = func(string, []string) (string, bool) { forbidden("build"); return "", false }
	prepushAcquireBuildSlot = func(bool) (bool, func()) { forbidden("build slot"); return false, func() {} }
	prepushSuccessCommonDir = func(string) string { forbidden("full-build receipt/claim"); return "" }
	prepushTestQuality = func(io.Writer, io.Writer, []string) int { forbidden("live test-quality scan"); return 2 }
	args := os.Args[1:]
	if len(args) >= 2 && args[0] == "hooks" {
		switch args[1] {
		case "import-witness":
			os.Exit(runHooksImportWitness(os.Stdout, os.Stderr, args[2:]))
		case "pre-push":
			os.Exit(runHooksPrePush(os.Stdout, os.Stderr, args[2:]))
		}
	}
	fmt.Fprintln(os.Stderr, "FOCUSED_UNEXPECTED_COMMAND", args)
	os.Exit(92)
}

// fak-test:runtime medium est=8s lane=default
func TestPrepushFocusedBuildOffGuards(t *testing.T) {
	f := newFocusedFixture(t)
	shell, err := exec.LookPath("sh")
	if err != nil && runtime.GOOS == "windows" {
		for _, candidate := range []string{`C:\Program Files\Git\usr\bin\sh.exe`, `C:\Program Files\Git\bin\bash.exe`} {
			if _, statErr := os.Stat(candidate); statErr == nil {
				shell, err = candidate, nil
				break
			}
		}
	}
	if err != nil {
		t.Fatalf("real pre-push regression requires a POSIX shell: %v", err)
	}
	// Resolve the tested checkout, not runtime.Caller's possibly trimmed source
	// name or the temporary Git fixture. Native landing builds use -trimpath.
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	rootOut, err := windowgate.CommandContext(ctx, "git", "-C", cwd, "rev-parse", "--show-toplevel").CombinedOutput()
	if err != nil {
		t.Fatalf("locate tested checkout: %v\n%s", err, rootOut)
	}
	hook := filepath.Join(strings.TrimSpace(string(rootOut)), "tools", "githooks", "pre-push")
	if info, err := os.Stat(hook); err != nil || !info.Mode().IsRegular() {
		t.Fatalf("real hook unavailable at %s: %v", hook, err)
	}
	zero := strings.Repeat("0", len(f.base))
	row := func(tip, base string) string { return "refs/heads/main " + tip + " refs/heads/main " + base + "\n" }
	cases := []struct {
		name, input, tip, base, diagnostic string
		code                               int
		wantConcept, noCalls               bool
	}{
		{name: "clean exact range ignores dirty HEAD and index", input: row(f.good, f.base), tip: f.good, base: f.base, wantConcept: true},
		{name: "committed missing import ignores untracked repair", input: row(f.badImport, f.good), tip: f.badImport, base: f.good, code: 1, diagnostic: "IMPORT_OF_UNCOMMITTED_PACKAGE"},
		{name: "earlier concept admission ignores later and staged repair", input: row(f.badConcept, f.base), tip: f.badConcept, base: f.base, code: 1, diagnostic: "CONCEPT_ADMISSION", wantConcept: true},
		{name: "same tip clean against its actual remote old", input: row(f.badConcept, f.introduced), tip: f.badConcept, base: f.introduced, wantConcept: true},
		{name: "same tip different bases are not deduplicated", input: row(f.badConcept, f.introduced) + row(f.badConcept, f.base), tip: f.badConcept, base: f.base, code: 1, diagnostic: "CONCEPT_ADMISSION", wantConcept: true},
		{name: "clean new ref admits complete positioned tip", input: row(f.good, zero), tip: f.good, base: zero, wantConcept: true},
		{name: "new ref checks ancestor concept over whole tip", input: row(f.badConcept, zero), tip: f.badConcept, base: zero, code: 1, diagnostic: "CONCEPT_ADMISSION", wantConcept: true},
		{name: "new ref checks ancestor import over whole tip", input: row(f.badImport, zero), tip: f.badImport, base: zero, code: 1, diagnostic: "IMPORT_OF_UNCOMMITTED_PACKAGE"},
		{name: "deleted ref does not substitute dirty HEAD", input: row(zero, f.badImport), noCalls: true},
		{name: "malformed row fails before checking any ref", input: row(f.good, f.base) + "refs/heads/main bad refs/heads/main\n", code: 1, diagnostic: "PREPUSH_UPDATE_INVALID", noCalls: true},
		{name: "unreadable pushed object fails closed", input: row(strings.Repeat("f", len(f.good)), f.base), tip: strings.Repeat("f", len(f.good)), base: f.base, code: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, code, calls := f.run(t, shell, []string{hook}, tc.input)
			if code != tc.code || (tc.diagnostic != "" && !strings.Contains(out, tc.diagnostic)) {
				t.Fatalf("code=%d want=%d diagnostic=%q\n%s", code, tc.code, tc.diagnostic, out)
			}
			if tc.noCalls {
				if len(calls) != 0 {
					t.Fatalf("no structural child expected: %v", calls)
				}
				return
			}
			if focusedCallCount(calls, "import-witness", "--rev", tc.tip) == 0 {
				t.Fatalf("exact pushed-tip import witness did not run: %v", calls)
			}
			if tc.wantConcept && focusedCallCount(calls, "pre-push", "--skip-build", "", "--base", tc.base, "--tip", tc.tip) == 0 {
				t.Fatalf("exact focused range admission did not run: %v", calls)
			}
			if tc.code == 0 {
				if len(calls) != 2 || focusedCallCount(calls, "import-witness") != 1 || focusedCallCount(calls, "pre-push") != 1 {
					t.Fatalf("one import witness and one range admission required per clean pair: %v", calls)
				}
				if !strings.Contains(out, "trunk-build-gate: SKIPPED_EXPLICIT") ||
					!strings.Contains(out, "concept-admission: PASSED") ||
					!strings.Contains(out, "test-quality: SKIPPED_EXPLICIT (live-working-tree-advisory)") {
					t.Fatalf("ordinary hook must report skipped compile and advisory quality work:\n%s", out)
				}
			}
		})
	}
	t.Run("missing native fak cannot compile a fallback", func(t *testing.T) {
		isolated := f
		isolated.bin = t.TempDir()
		isolated.path = isolated.bin
		for _, name := range []string{"git", "cat", "awk", "sort"} {
			path, err := exec.LookPath(name)
			if err != nil {
				t.Fatalf("real shell dependency %s: %v", name, err)
			}
			focusedLinkExecutable(t, path, filepath.Join(isolated.bin, filepath.Base(path)))
		}
		goName := "go"
		if runtime.GOOS == "windows" {
			goName += ".exe"
		}
		focusedLinkExecutable(t, f.executable, filepath.Join(isolated.bin, goName))
		out, code, calls := isolated.run(t, shell, []string{hook}, row(f.good, f.base))
		if code == 0 || len(calls) != 0 || !strings.Contains(out, "fak") {
			t.Fatalf("missing verifier must refuse without compiling: code=%d calls=%v\n%s", code, calls, out)
		}
	})
}

// fak-test:runtime medium est=2s lane=default
func TestPrepushFocusedReportCannotBecomeBuildEvidence(t *testing.T) {
	f := newFocusedFixture(t)
	out, code, _ := f.run(t, f.executable, []string{"hooks", "pre-push", "--skip-build", "--root", f.root, "--base", f.base, "--tip", f.good, "--json"}, "")
	if code != 0 {
		t.Fatalf("clean focused admission: code=%d\n%s", code, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("JSON result: %v\n%s", err, out)
	}
	for key, want := range map[string]any{
		"ok": true, "ref": f.good, "base_sha": f.base,
		"verdict": "SKIPPED_EXPLICIT", "build_verdict": "SKIPPED_EXPLICIT",
		"concept_admission": "PASSED", "test_quality": "SKIPPED_EXPLICIT",
		"test_quality_scope": "live-working-tree-advisory",
	} {
		if got[key] != want {
			t.Errorf("%s=%v want=%v; result=%s", key, got[key], want, out)
		}
	}
}

type focusedFixture struct {
	root, bin, path, executable, base, introduced, badConcept, good, badImport string
}

func newFocusedFixture(t *testing.T) focusedFixture {
	t.Helper()
	f := focusedFixture{root: t.TempDir(), bin: t.TempDir()}
	f.git(t, "init", "-q")
	f.git(t, "symbolic-ref", "HEAD", "refs/heads/main")
	f.git(t, "config", "user.name", "Focused Push Test")
	f.git(t, "config", "user.email", "focused-push@example.invalid")
	f.git(t, "config", "core.hooksPath", t.TempDir())
	write := func(path, body string) {
		t.Helper()
		name := filepath.Join(f.root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(name), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(message string) string {
		t.Helper()
		f.git(t, "add", ".")
		f.git(t, "commit", "-q", "-m", message)
		return f.git(t, "rev-parse", "HEAD")
	}
	const source = "internal/demo/demo.go"
	const meta = "tools/concept_disambiguation_scorecard.data/_meta.json"
	const rows = "tools/concept_disambiguation_scorecard.data/rows-cache.json"
	write("go.mod", "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n")
	write(source, "package demo\nconst Value = 1\n")
	write(meta, `{"families":[{"id":"cache","roots":["cache"]}]}`)
	write(rows, `{"rows":[]}`)
	f.base = commit("base")
	write(source, "package demo\nconst Value = 1\nconst CacheAdded = 2\n")
	f.introduced = commit("introduce unpositioned concept")
	write("README.md", "unrelated descendant; concept exists only in an ancestor delta\n")
	f.badConcept = commit("advance without touching the concept")
	write(rows, `{"rows":[{"id":"cache-added","family":"cache","grounding":"CacheAdded"}]}`)
	f.good = commit("position the concept in the committed corpus")
	write("internal/caller/caller.go", "package caller\nimport (\n\t_ \"github.com/anthony-chaudhary/fak/internal/missing\"\n)\n")
	commit("import package absent from the committed tip")
	write("README.md", "another unrelated descendant; missing import is inherited\n")
	f.badImport = commit("advance without touching the missing import")
	// The current checkout contains a tempting repair, while its source is dirty
	// in the opposite direction. Neither index nor working tree is admission input.
	write("internal/missing/missing.go", "package missing\n")
	write(meta, `{"families":[{"id":"cache","roots":["cache"],"ignore":["cacheadded"]}]}`)
	f.git(t, "add", meta)
	write(source, "package demo\nconst CacheDirtyOnly = 3\n")
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f.executable = filepath.Join(f.bin, "fak")
	if runtime.GOOS == "windows" {
		f.executable += ".exe"
	}
	focusedLinkExecutable(t, binary, f.executable)
	return f
}

func focusedLinkExecutable(t *testing.T, source, target string) {
	t.Helper()
	if err := os.Link(source, target); err != nil {
		in, err := os.Open(source)
		if err != nil {
			t.Fatal(err)
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0755)
		if err != nil {
			t.Fatal(err)
		}
		_, copyErr := io.Copy(out, in)
		closeErr := out.Close()
		if copyErr != nil || closeErr != nil {
			t.Fatalf("copy Go helper executable: %v / %v", copyErr, closeErr)
		}
	}
}

func (f focusedFixture) git(t *testing.T, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "git", args...)
	cmd.Dir = f.root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fixture git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (f focusedFixture) run(t *testing.T, executable string, args []string, input string) (string, int, [][]string) {
	t.Helper()
	trace := filepath.Join(t.TempDir(), "calls.jsonl")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, executable, args...)
	cmd.Dir, cmd.Stdin = f.root, strings.NewReader(input)
	for _, item := range os.Environ() {
		key := strings.SplitN(item, "=", 2)[0]
		if strings.HasPrefix(key, "FLEET_") || strings.HasPrefix(key, "FAK_FOCUSED_PREPUSH_") || strings.EqualFold(key, "PATH") {
			continue
		}
		cmd.Env = append(cmd.Env, item)
	}
	path := f.path
	if path == "" {
		path = f.bin + string(os.PathListSeparator) + os.Getenv("PATH")
	}
	cmd.Env = append(cmd.Env, focusedHelperEnv+"=1", "FAK_FOCUSED_PREPUSH_TRACE="+trace,
		"PATH="+path,
		"FLEET_BUILD_GUARD=off", "FLEET_WORKFLOW_GUARD=off", "FLEET_TIER_GUARD=off",
		"FLEET_POPUP_GUARD=off", "FLEET_REVIEW_GUARD=off", "FLEET_REFPRUNE_GUARD=off")
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			code = exit.ExitCode()
		} else {
			t.Fatalf("start helper/hook: %v", err)
		}
	}
	if ctx.Err() != nil || strings.Contains(string(out), "FOCUSED_FORBIDDEN_STEP") || strings.Contains(string(out), "FOCUSED_UNEXPECTED_COMMAND") {
		t.Fatalf("focused path performed forbidden work or did not terminate: %v\n%s", ctx.Err(), out)
	}
	for _, name := range []string{"fak-prepush-success", "fak-committed-build-success"} {
		entries, err := os.ReadDir(filepath.Join(f.root, ".git", name))
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("focused path emitted full-build evidence in %s: %v", name, entries)
		}
	}
	data, err := os.ReadFile(trace)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	var calls [][]string
	dec := json.NewDecoder(bytes.NewReader(data))
	for {
		var call []string
		if err := dec.Decode(&call); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		calls = append(calls, call)
	}
	return string(out), code, calls
}

func focusedCallCount(calls [][]string, command string, flagValues ...string) int {
	count := 0
	for _, args := range calls {
		if len(args) < 2 || args[0] != "hooks" || args[1] != command {
			continue
		}
		match := true
		for i := 0; i < len(flagValues); i += 2 {
			found := false
			for j := 2; j < len(args); j++ {
				if args[j] == flagValues[i] && (flagValues[i+1] == "" || (j+1 < len(args) && args[j+1] == flagValues[i+1])) {
					found = true
				}
			}
			match = match && found
		}
		if match {
			count++
		}
	}
	return count
}
