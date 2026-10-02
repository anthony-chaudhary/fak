package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
)

// fak-test:runtime integration est=1s lane=default
func TestIssueCreateDedupeRepository(t *testing.T) {
	// Keep the production fetcher and exec boundary: an injected backlog cannot
	// detect a repository flag lost between issue creation and gh issue list.
	withIssueCreateDedupeFetcher(t, fetchIssueCreateDedupeBacklog)
	binDir := buildIssueCreateDedupeGH(t)
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("FAK_TEST_DEDUPE_BACKLOG", nearDupBacklog)
	t.Chdir(t.TempDir()) // An archive cwd must not determine an explicit destination.

	for _, tc := range []struct {
		name     string
		repo     string
		cap      int
		warnOnly bool
		failGH   bool
	}{
		{name: "explicit repository", repo: "owner/destination", cap: 7},
		{name: "verbatim repository", repo: "  owner/destination  ", cap: 7},
		{name: "inferred repository and default cap"},
		{name: "blank repository and negative cap", repo: " \t ", cap: -1},
		{name: "fetch fails open", repo: "owner/destination", cap: 7, failGH: true},
		{name: "warn only", repo: "owner/destination", cap: 7, warnOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			capture := filepath.Join(t.TempDir(), "argv.json")
			t.Setenv("FAK_TEST_DEDUPE_ARGV", capture)
			t.Setenv("FAK_TEST_DEDUPE_FAIL", strconv.FormatBool(tc.failGH))
			body := "## Parent context\n#1\n\n## Core through-line\nChange -> seam -> outcome -> witness.\n\n## Gold-plating boundary\nNo extras." + validIssueCreateProblemFrame("Core")
			args := []string{
				"--title", "fix(dispatch): the preflight lane-lock gate ignores contract-lease TTL",
				"--body", body, "--estimate-points", "1", "--parent-baseline-points", "1",
				"--target-envelope", "- paths: >= 1 command", "--witnessed-envelope", "- paths: 1 command",
				"--dedupe-checked", "--dedupe-cap", strconv.Itoa(tc.cap), "--dry-run", "--json",
			}
			if tc.repo != "" {
				args = append(args, "--repo", tc.repo)
			}
			if tc.warnOnly {
				args = append(args, "--dedupe-warn-only")
			}
			var out, errb bytes.Buffer
			code := runIssueCreateWithCleanScrub(&out, &errb, args, func([]string) (string, string, bool) {
				t.Fatal("a dry run must not execute gh issue create")
				return "", "", false
			})
			var result issueCreateResult
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatalf("decode result: %v; code=%d stdout=%s stderr=%s", err, code, &out, &errb)
			}
			data, err := os.ReadFile(capture)
			if err != nil {
				t.Fatalf("gh subprocess did not capture argv: %v; stderr=%s", err, &errb)
			}
			var receipt struct {
				Runtime string
				Args    []string
			}
			if err := json.Unmarshal(data, &receipt); err != nil || receipt.Runtime != "native-go-gh" {
				t.Fatalf("invalid subprocess receipt: %s; err=%v", data, err)
			}
			cap := tc.cap
			if cap <= 0 {
				cap = issuefanout.DefaultDedupeCap
			}
			wantList := []string{"issue", "list", "--state", "open", "--limit", strconv.Itoa(cap), "--json", "number,title,body"}
			if strings.TrimSpace(tc.repo) != "" {
				wantList = append(wantList, "--repo", tc.repo)
			}
			if !reflect.DeepEqual(receipt.Args, wantList) {
				t.Errorf("gh issue list argv=%q, want %q", receipt.Args, wantList)
			}
			if len(result.Args) < 2 || result.Args[0] != "issue" || result.Args[1] != "create" || result.Repo != tc.repo {
				t.Fatalf("wrong create destination: repo=%q args=%q", result.Repo, result.Args)
			}
			var createRepo []string
			for i, arg := range result.Args {
				if arg == "--repo" && i+1 < len(result.Args) {
					createRepo = append(createRepo, result.Args[i+1])
				}
			}
			var wantRepo []string
			if strings.TrimSpace(tc.repo) != "" {
				wantRepo = []string{tc.repo}
			}
			if !reflect.DeepEqual(createRepo, wantRepo) {
				t.Errorf("gh issue create repo=%q, want %q", createRepo, wantRepo)
			}
			wantCode := 3
			if tc.failGH || tc.warnOnly {
				wantCode = 0
			}
			if code != wantCode || result.Dedupe == nil || !result.Dedupe.Checked || result.Dedupe.Cap != cap || result.Dedupe.Refused != (wantCode == 3) {
				t.Fatalf("code=%d want %d, dedupe=%+v; stderr=%s", code, wantCode, result.Dedupe, &errb)
			}
			if tc.failGH {
				if result.Dedupe.Scanned != 0 || !strings.Contains(result.Dedupe.Error, "exit status 23") || !strings.Contains(errb.String(), "failing open") {
					t.Fatalf("fetch error did not fail open: %+v; stderr=%s", result.Dedupe, &errb)
				}
			} else if result.Dedupe.Scanned != 1 || len(result.Dedupe.Matches) != 1 {
				t.Fatalf("expected the subprocess backlog to be checked: %+v", result.Dedupe)
			}
			if tc.warnOnly && !strings.Contains(errb.String(), "continuing: --dedupe-warn-only") {
				t.Fatalf("missing warn-only notice: %s", &errb)
			}
		})
	}
}

func buildIssueCreateDedupeGH(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "main.go")
	// This native child substitutes only the external service. The command under
	// test still builds and executes its actual gh argv, without shell quoting.
	const program = `package main
import (
	"encoding/json"
	"fmt"
	"os"
)
func main() {
	data, err := json.Marshal(struct { Runtime string; Args []string }{"native-go-gh", os.Args[1:]})
	if err != nil { panic(err) }
	if err := os.WriteFile(os.Getenv("FAK_TEST_DEDUPE_ARGV"), data, 0600); err != nil { panic(err) }
	if len(os.Args) < 3 || os.Args[1] != "issue" || os.Args[2] != "list" {
		fmt.Fprintln(os.Stderr, "fixture accepts only gh issue list")
		os.Exit(24)
	}
	if os.Getenv("FAK_TEST_DEDUPE_FAIL") == "true" {
		fmt.Fprintln(os.Stderr, "fixture backlog unavailable")
		os.Exit(23)
	}
	fmt.Print(os.Getenv("FAK_TEST_DEDUPE_BACKLOG"))
}
`
	if err := os.WriteFile(source, []byte(program), 0o600); err != nil {
		t.Fatal(err)
	}
	name := "gh"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "build", "-o", filepath.Join(dir, name), source)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build native gh fixture: %v\n%s", err, output)
	}
	return dir
}
