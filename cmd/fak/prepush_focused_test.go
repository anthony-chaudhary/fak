package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
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
	if name == "git-remote-fak-stall" && os.Getenv("FAK_FOCUSED_PREPUSH_STALL_ADDR") != "" {
		// A real Git remote-helper child holds this connection until the
		// production deadline terminates the fetch process group.
		conn, err := net.DialTimeout("tcp", os.Getenv("FAK_FOCUSED_PREPUSH_STALL_ADDR"), 5*time.Second)
		if err != nil {
			os.Exit(94)
		}
		_, _ = io.WriteString(conn, "started")
		_, _ = io.Copy(io.Discard, conn)
		_ = conn.Close()
		os.Exit(0)
	}
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
	normalBuild := os.Getenv("FAK_FOCUSED_PREPUSH_NORMAL_BUILD") == "1"
	if os.Getenv("FAK_FOCUSED_PREPUSH_REAL_BUILD") == "1" {
		prepushTestQuality = func(io.Writer, io.Writer, []string) int { return 0 }
	} else if normalBuild {
		// Exercise the real hook and range admission with only compilation
		// stubbed. No compiler or archive is needed for the protocol witness.
		prepushExtractTip = func(string, string) (string, error) { return os.MkdirTemp("", "prepush-protocol-") }
		prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
			return map[string]string{"internal/demo/demo.go": "./internal/demo"}, nil, 1, nil
		}
		prepushListTestOnly = func(string, []string) map[string]bool { return nil }
		prepushBuild = func(string, []string) (string, bool) { return "", true }
		prepushAcquireBuildSlot = func(bool) (bool, func()) { return true, func() {} }
		prepushTestQuality = func(io.Writer, io.Writer, []string) int { return 0 }
	} else {
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
	}
	args := os.Args[1:]
	if len(args) >= 2 && args[0] == "workflow" {
		os.Exit(runWorkflow(os.Stdout, os.Stderr, os.Stdin, args[1:]))
	}
	if len(args) >= 2 && args[0] == "hooks" {
		switch args[1] {
		case "import-witness":
			os.Exit(runHooksImportWitness(os.Stdout, os.Stderr, args[2:]))
		case "pre-push":
			for _, arg := range args[2:] {
				if arg != "--hydrate-bases" || os.Getenv("FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE") == "" {
					continue
				}
				// Advance the peer after the hook observed the missing old object,
				// before the native acquisition starts. The protocol row stays fixed.
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				for _, argv := range [][]string{{"commit", "--allow-empty", "-m", "advance during admission"}, {"push", "origin", "main"}} {
					cmd := windowgate.CommandContext(ctx, "git", argv...)
					cmd.Dir = os.Getenv("FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE")
					if out, err := cmd.CombinedOutput(); err != nil {
						fmt.Fprintln(os.Stderr, err, string(out))
						os.Exit(93)
					}
				}
			}
			os.Exit(runHooksPrePush(os.Stdout, os.Stderr, args[2:]))
		}
	}
	fmt.Fprintln(os.Stderr, "FOCUSED_UNEXPECTED_COMMAND", args)
	os.Exit(92)
}

func prepushHookUnderTest(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// Go tests run from their package directory, including committed archives.
	hook := filepath.Clean(filepath.Join(cwd, "..", "..", "tools", "githooks", "pre-push"))
	if info, err := os.Stat(hook); err != nil || !info.Mode().IsRegular() {
		t.Fatalf("real hook unavailable at %s: %v", hook, err)
	}
	return hook
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
	hook := prepushHookUnderTest(t)
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
	buildMode := "off"
	if os.Getenv("FAK_FOCUSED_PREPUSH_NORMAL_BUILD") == "1" {
		buildMode = "block"
		cmd.Env = append(cmd.Env, "FAK_FOCUSED_PREPUSH_NORMAL_BUILD=1")
	}
	if os.Getenv("FAK_FOCUSED_PREPUSH_REAL_BUILD") == "1" {
		buildMode = "block"
		cmd.Env = append(cmd.Env, "FAK_FOCUSED_PREPUSH_REAL_BUILD=1")
	}
	workflowMode, tierMode, popupMode, reviewMode := "off", "off", "off", "off"
	if os.Getenv("FAK_FOCUSED_PREPUSH_WORKFLOW") == "1" {
		workflowMode = "block"
	}
	if peer := os.Getenv("FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE"); peer != "" {
		cmd.Env = append(cmd.Env, "FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE="+peer)
	}
	if os.Getenv("FAK_FOCUSED_PREPUSH_ALL_CONTENT_GATES") == "1" {
		workflowMode, tierMode, popupMode, reviewMode = "block", "block", "block", "block"
	}
	refpruneMode := "off"
	if limit := os.Getenv("FAK_FOCUSED_PREPUSH_REFPRUNE_MAX"); limit != "" {
		refpruneMode = "block"
		cmd.Env = append(cmd.Env, "FLEET_REFPRUNE_MAX="+limit)
	}
	cmd.Env = append(cmd.Env, focusedHelperEnv+"=1", "FAK_FOCUSED_PREPUSH_TRACE="+trace,
		"PATH="+path,
		"FLEET_BUILD_GUARD="+buildMode, "FLEET_WORKFLOW_GUARD="+workflowMode, "FLEET_TIER_GUARD="+tierMode,
		"FLEET_POPUP_GUARD="+popupMode, "FLEET_REVIEW_GUARD="+reviewMode, "FLEET_REFPRUNE_GUARD="+refpruneMode)
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

// fak-test:runtime medium est=8s lane=default
func TestPrepushNormalBuildUsesExactProtocolBases(t *testing.T) {
	t.Setenv("FAK_FOCUSED_PREPUSH_NORMAL_BUILD", "1")
	f := newFocusedFixture(t)
	shell := gitHookShell(t)
	hook := prepushHookUnderTest(t)
	// A stale tracking ref must never replace either protocol base. Two
	// destinations may send the same tip against different remote old objects.
	f.git(t, "update-ref", "refs/remotes/origin/main", f.good)
	row := func(base string) string { return "refs/heads/main " + f.good + " refs/heads/main " + base + "\n" }
	zero := strings.Repeat("0", len(f.base))
	out, code, calls := f.run(t, shell, []string{hook}, row(f.base)+row(f.introduced)+row(zero))
	if code != 0 {
		t.Fatalf("normal build code=%d\n%s", code, out)
	}
	for _, base := range []string{f.base, f.introduced, zero} {
		if focusedCallCount(calls, "pre-push", "--base", base, "--tip", f.good) != 1 {
			t.Fatalf("exact old:new pair missing or duplicated: base=%s calls=%v", base, calls)
		}
	}
	out, code, calls = f.run(t, shell, []string{hook}, "refs/heads/main "+zero+" refs/heads/main "+f.good+"\n")
	if code != 0 || focusedCallCount(calls, "pre-push") != 0 {
		t.Fatalf("deletion compiled dirty HEAD: code=%d calls=%v\n%s", code, calls, out)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestPrepushDeletionOnlySkipsAllContentGatesButRetainsRefPolicy(t *testing.T) {
	t.Setenv("FAK_FOCUSED_PREPUSH_NORMAL_BUILD", "1")
	t.Setenv("FAK_FOCUSED_PREPUSH_ALL_CONTENT_GATES", "1")
	f := newFocusedFixture(t)
	hook := prepushHookUnderTest(t)
	row := "refs/heads/main " + strings.Repeat("0", len(f.badImport)) + " refs/heads/main " + f.badImport + "\n"
	out, code, calls := f.run(t, gitHookShell(t), []string{hook}, row)
	if code != 0 || len(calls) != 0 {
		t.Fatalf("deletion ran content checks: code=%d calls=%v\n%s", code, calls, out)
	}
	t.Setenv("FAK_FOCUSED_PREPUSH_REFPRUNE_MAX", "0")
	out, code, calls = f.run(t, gitHookShell(t), []string{hook}, row)
	if code != 1 || len(calls) != 0 || !strings.Contains(out, "REF_PRUNE (blocked)") {
		t.Fatalf("deletion lost ref policy: code=%d calls=%v\n%s", code, calls, out)
	}
}

// fak-test:runtime medium est=8s lane=default
func TestPrepushAdvertisedBasePeerRace(t *testing.T) {
	t.Setenv("FAK_FOCUSED_PREPUSH_NORMAL_BUILD", "1")
	t.Setenv("FAK_FOCUSED_PREPUSH_WORKFLOW", "1")
	f := newFocusedFixture(t)
	remote := filepath.Join(t.TempDir(), "remote.git")
	f.git(t, "init", "--bare", remote)
	f.git(t, "push", remote, f.good+":refs/heads/main")
	f.git(t, "remote", "add", "origin", remote)
	peer := focusedFixture{root: filepath.Join(t.TempDir(), "peer")}
	f.git(t, "clone", "--no-local", "--branch", "main", remote, peer.root)
	peer.git(t, "config", "user.name", "Peer Push Test")
	peer.git(t, "config", "user.email", "peer-push@example.invalid")
	peer.git(t, "config", "core.hooksPath", t.TempDir())
	if err := os.WriteFile(filepath.Join(peer.root, "internal/demo/demo.go"), []byte("package demo\nconst Value = 1\nconst CacheAdded = 2\nconst PeerValue = 3\n"), 0600); err != nil {
		t.Fatal(err)
	}
	peer.git(t, "add", "internal/demo/demo.go")
	peer.git(t, "commit", "-m", "peer advances after the pusher last fetched")
	old := peer.git(t, "rev-parse", "HEAD")
	peer.git(t, "push", "origin", "main")
	if advertisedBaseObjectPresent(t, f.root, old) {
		t.Fatal("race fixture already has the peer's advertised old object")
	}
	zero := strings.Repeat("0", len(old))
	row := func(tip, base, target string) string {
		return "refs/heads/main " + tip + " refs/heads/" + target + " " + base + "\n"
	}
	input := row(f.good, old, "main")
	hook := prepushHookUnderTest(t)
	shell := gitHookShell(t)
	out, code, calls := f.run(t, shell, []string{hook, "origin", remote}, input+"malformed\n")
	if code == 0 || len(calls) != 0 || !strings.Contains(out, "PREPUSH_UPDATE_INVALID") || advertisedBaseObjectPresent(t, f.root, old) {
		t.Fatalf("malformed tail admitted a prefix: code=%d calls=%v\n%s", code, calls, out)
	}
	out, code, _ = f.run(t, shell, []string{hook, "origin", filepath.Join(t.TempDir(), "absent.git")}, input)
	if code == 0 || advertisedBaseObjectPresent(t, f.root, old) {
		t.Fatalf("failed push endpoint substituted configured origin: code=%d\n%s", code, out)
	}
	refs := f.git(t, "for-each-ref", "--format=%(refname) %(objectname)")
	fetchHead := filepath.Join(f.root, ".git", "FETCH_HEAD")
	if err := os.WriteFile(fetchHead, []byte("unchanged fetch head\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE", peer.root)
	input += row(f.good, f.base, "master") + row(f.good, zero, "new") + row(zero, old, "deleted")
	out, code, calls = f.run(t, shell, []string{hook, "origin", remote}, input)
	if code != 0 {
		t.Fatalf("advertised-base availability must precede exact-range admission: code=%d\n%s", code, out)
	}
	if !advertisedBaseObjectPresent(t, f.root, old) || peer.git(t, "rev-parse", "HEAD") == old {
		t.Fatal("missing exact object was not acquired across the controlled second peer advance")
	}
	if got := f.git(t, "for-each-ref", "--format=%(refname) %(objectname)"); got != refs {
		t.Fatalf("object acquisition changed refs:\nbefore=%s\nafter=%s", refs, got)
	}
	if got, err := os.ReadFile(fetchHead); err != nil || string(got) != "unchanged fetch head\n" {
		t.Fatalf("object acquisition changed shared FETCH_HEAD: %q, %v", got, err)
	}
	if focusedCallCount(calls, "pre-push", "--hydrate-bases", old) != 1 {
		t.Fatalf("missing bases were not acquired once: %v", calls)
	}
	for _, base := range []string{old, f.base, zero} {
		if focusedCallCount(calls, "pre-push", "--base", base, "--tip", f.good) != 1 {
			t.Fatalf("original old:new pair was replaced or dropped: base=%s calls=%v", base, calls)
		}
		workflow := 0
		for _, call := range calls {
			if len(call) > 1 && call[0] == "workflow" && strings.Contains(strings.Join(call, " "), "--base "+base+" --tip "+f.good) {
				workflow++
			}
		}
		if workflow != 1 {
			t.Fatalf("workflow consumer did not receive exact pair: base=%s calls=%v", base, calls)
		}
	}
	t.Setenv("FAK_FOCUSED_PREPUSH_ADVANCE_REMOTE", "")
	out, code, calls = f.run(t, shell, []string{hook, "origin", filepath.Join(t.TempDir(), "absent.git")}, row(f.good, old, "main"))
	if code != 0 || focusedCallCount(calls, "pre-push", "--hydrate-bases", "") != 0 {
		t.Fatalf("known-local old commit attempted transport: code=%d calls=%v\n%s", code, calls, out)
	}
	for _, unavailable := range []string{strings.Repeat("f", len(old)), f.git(t, "rev-parse", f.good+":internal/demo/demo.go")} {
		out, code, calls = f.run(t, shell, []string{hook, "origin", remote}, row(f.good, unavailable, "main"))
		if code == 0 || !strings.Contains(out, "PREPUSH_BASE_UNAVAILABLE") || strings.Contains(out, "TRUNK_WOULD_NOT_COMPILE") || focusedCallCount(calls, "pre-push", "--tip", f.good) != 0 {
			t.Fatalf("unavailable/non-commit base entered compilation: code=%d calls=%v\n%s", code, calls, out)
		}
	}
	// Availability is not integration authority: the original local tip remains
	// behind the peer, so ordinary Git publication must still reject it.
	remoteBefore := peer.git(t, "rev-parse", "HEAD")
	out, code, _ = f.run(t, "git", []string{"push", "--porcelain", remote, f.good + ":refs/heads/main"}, "")
	if code == 0 || !strings.Contains(out, "[rejected]") || peer.git(t, "ls-remote", "origin", "refs/heads/main") != remoteBefore+"\trefs/heads/main" {
		t.Fatalf("availability changed non-fast-forward publication: code=%d\n%s", code, out)
	}
}

func advertisedBaseObjectPresent(t *testing.T, root, oid string) bool {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "git", "--no-lazy-fetch", "--no-replace-objects", "-C", root, "cat-file", "-e", oid)
	return cmd.Run() == nil
}

// fak-test:runtime medium est=5s lane=default
func TestPrepushAdvertisedBasePreservesRealCompileRefusal(t *testing.T) {
	t.Setenv("FAK_FOCUSED_PREPUSH_REAL_BUILD", "1")
	f := newFocusedFixture(t)
	f.git(t, "reset", "--hard", f.good)
	if err := os.WriteFile(filepath.Join(f.root, "internal/demo/demo.go"), []byte("package demo\nconst Value = MissingDefinition\nconst CacheAdded = 2\n"), 0600); err != nil {
		t.Fatal(err)
	}
	f.git(t, "add", "internal/demo/demo.go")
	f.git(t, "commit", "-m", "introduce an actual compile regression")
	bad := f.git(t, "rev-parse", "HEAD")
	row := "refs/heads/main " + bad + " refs/heads/main " + f.good + "\n"
	out, code, calls := f.run(t, gitHookShell(t), []string{prepushHookUnderTest(t), "origin", filepath.Join(t.TempDir(), "absent.git")}, row)
	if code != 1 || !strings.Contains(out, "TRUNK_WOULD_NOT_COMPILE") || !strings.Contains(out, "MissingDefinition") || strings.Contains(out, "PREPUSH_BASE_UNAVAILABLE") {
		t.Fatalf("real compiler refusal lost its classification: code=%d\n%s", code, out)
	}
	if focusedCallCount(calls, "pre-push", "--hydrate-bases", "") != 0 {
		t.Fatalf("known-local compile-red base attempted transport: %v", calls)
	}
}

// fak-test:runtime medium est=32s lane=default
func TestPrepushAdvertisedBaseAcquisitionDeadline(t *testing.T) {
	f := newFocusedFixture(t)
	f.git(t, "config", "protocol.fak-stall.allow", "always")
	name := "git-remote-fak-stall"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	focusedLinkExecutable(t, f.executable, filepath.Join(f.bin, name))
	t.Setenv("PATH", f.bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	listener, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	_ = listener.SetDeadline(time.Now().Add(35 * time.Second))
	t.Setenv("FAK_FOCUSED_PREPUSH_STALL_ADDR", listener.Addr().String())
	type drainResult struct {
		output string
		err    error
	}
	drained := make(chan drainResult, 1)
	go func() {
		conn, err := listener.AcceptTCP()
		if err != nil {
			drained <- drainResult{err: err}
			return
		}
		defer conn.Close()
		_ = conn.SetReadDeadline(time.Now().Add(35 * time.Second))
		out, err := io.ReadAll(conn)
		drained <- drainResult{output: string(out), err: err}
	}()
	var out, diagnostic bytes.Buffer
	start := time.Now()
	// Use only the existing CLI entry point so the complete changed test file
	// also compiles in the native parent overlay for the PeerRace symptom proof.
	code := runHooksPrePush(&out, &diagnostic, []string{"--root", f.root, "--hydrate-bases", strings.Repeat("f", 40), "--remote", "fak-stall::https://credential-must-not-appear@example.invalid/repo"})
	if elapsed := time.Since(start); elapsed > 33*time.Second {
		t.Fatalf("acquisition exceeded its single deadline and drain allowance: %s", elapsed)
	}
	_ = listener.SetDeadline(time.Now().Add(time.Second))
	select {
	case result := <-drained:
		if result.err != nil || result.output != "started" {
			t.Fatalf("live fetch descendant was not started and drained: %+v; diagnostic=%s", result, &diagnostic)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("fetch descendant retained its live connection after cancellation")
	}
	if code != 2 || !strings.Contains(diagnostic.String(), "PREPUSH_BASE_UNAVAILABLE") || strings.Contains(diagnostic.String(), "credential-must-not-appear") || out.Len() != 0 {
		t.Fatalf("failed acquisition leaked an endpoint or emitted success: stdout=%s stderr=%s", &out, &diagnostic)
	}
}
