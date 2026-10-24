package issuesolved

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestMain runs a native fake gh without shell helpers, network, or credentials.
func TestMain(m *testing.M) {
	if os.Getenv("FAK_ISSUESOLVED_GH_HELPER") == "1" && strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe") == "gh" {
		if os.Getenv("GH_PROMPT_DISABLED") != "1" || os.Getenv("GH_NO_UPDATE_NOTIFIER") != "1" {
			fmt.Fprintln(os.Stderr, "fake gh: interactive configuration")
			os.Exit(98)
		}
		switch os.Getenv("FAK_ISSUESOLVED_GH_MODE") {
		case "failure":
			fmt.Fprintln(os.Stdout, "[]")
			fmt.Fprintln(os.Stderr, "synthetic gh: rate limit exceeded")
			os.Exit(7)
		case "hang":
			fmt.Fprintln(os.Stdout, "fake gh started")
			time.Sleep(3 * time.Second)
			os.Exit(0)
		default:
			fmt.Fprintln(os.Stderr, "synthetic gh: warning, not JSON")
			fmt.Fprintln(os.Stdout, `[{"number":12161,"title":"fixture","closedAt":"2026-09-08T10:00:00Z","stateReason":"COMPLETED","url":"https://example.invalid/issues/12161"}]`)
			os.Exit(0)
		}
	}
	os.Exit(m.Run())
}

func installFakeGh(t *testing.T) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	bin := t.TempDir()
	name := "gh"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	src, err := os.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	dst, err := os.OpenFile(filepath.Join(bin, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dst, src)
	closeErr := dst.Close()
	if err := errors.Join(copyErr, closeErr); err != nil {
		t.Fatal(err)
	}
	// No real gh is reachable, even if command resolution regresses.
	t.Setenv("PATH", bin)
	t.Setenv("FAK_ISSUESOLVED_GH_HELPER", "1")
}

// fak-test:runtime fast est=100ms lane=default
func TestGhExecutor_Stderr_Timeout(t *testing.T) {
	installFakeGh(t)
	t.Run("stderr", func(t *testing.T) {
		t.Setenv("FAK_ISSUESOLVED_GH_MODE", "failure")
		_, err := collectIssuesFromGh(nil, "fixture/repo", time.Time{}, time.Now(), defaultGhExecutor)
		if err == nil || !strings.Contains(err.Error(), "gh issue list:") || !strings.Contains(err.Error(), "synthetic gh: rate limit exceeded") {
			t.Fatalf("missing gh stderr diagnostic: %v", err)
		}
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 {
			t.Fatalf("lost underlying gh exit error: %v", err)
		}
	})
	t.Run("json stdout excludes stderr", func(t *testing.T) {
		t.Setenv("FAK_ISSUESOLVED_GH_MODE", "success")
		since := time.Date(2026, 9, 8, 0, 0, 0, 0, time.UTC)
		items, err := collectIssuesFromGh(nil, "fixture/repo", since, since.Add(24*time.Hour), defaultGhExecutor)
		want := []IssueItem{{
			Number: 12161, Title: "fixture", ClosedAt: since.Add(10 * time.Hour),
			StateReason: "COMPLETED", URL: "https://example.invalid/issues/12161",
		}}
		if err != nil || !reflect.DeepEqual(items, want) {
			t.Fatalf("gh JSON changed or stderr leaked into stdout: items=%v, want=%v, err=%v", items, want, err)
		}
	})
	t.Run("parent cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := defaultGhExecutor(ctx, "issue", "list"); !errors.Is(err, context.Canceled) {
			t.Fatalf("parent cancellation = %v, want context.Canceled", err)
		}
	})
	t.Run("earlier parent deadline", func(t *testing.T) {
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancel()
		if _, err := defaultGhExecutor(ctx, "issue", "list"); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("parent deadline = %v, want context.DeadlineExceeded", err)
		}
	})
}

// fak-test:runtime fast est=500ms lane=default
func TestGhExecutor_OwnTimeout(t *testing.T) {
	installFakeGh(t)
	t.Setenv("FAK_ISSUESOLVED_GH_MODE", "hang")
	// The default entry point supplies ghexec.DefaultTimeout to this same helper.
	// A short budget proves process termination without adding a minute to the suite.
	out, err := runGhWithTimeout(context.Background(), 250*time.Millisecond, "issue", "list")
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || string(out) != "fake gh started\n" {
		t.Fatalf("gh was not started and terminated by its deadline: stdout=%q, err=%v", out, err)
	}
}

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()
	if opts.Hours != 24 {
		t.Errorf("expected Hours=24, got %f", opts.Hours)
	}
	if opts.PublicRepo != "anthony-chaudhary/fak" {
		t.Errorf("expected PublicRepo='anthony-chaudhary/fak', got %q", opts.PublicRepo)
	}
	if opts.PrivateRepo != "anthony-chaudhary/fak-private" {
		t.Errorf("expected PrivateRepo='anthony-chaudhary/fak-private', got %q", opts.PrivateRepo)
	}
	if !opts.IncludePrivate {
		t.Errorf("expected IncludePrivate=true, got false")
	}
	if opts.Source != "auto" {
		t.Errorf("expected Source='auto', got %q", opts.Source)
	}
}

func TestOptions_TimeCalculation(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)

	// Case 1: Hours provided, Since is zero -> Since = Now.Add(-Hours * time.Hour)
	opts1 := Options{
		Now:   fixedNow,
		Hours: 9.0,
	}
	norm1, err := normalizeOptions(opts1)
	if err != nil {
		t.Fatal(err)
	}
	expectedSince1 := fixedNow.Add(-9 * time.Hour)
	if !norm1.Since.Equal(expectedSince1) {
		t.Errorf("expected Since=%v, got %v", expectedSince1, norm1.Since)
	}
	if norm1.Hours != 9.0 {
		t.Errorf("expected Hours=9.0, got %f", norm1.Hours)
	}

	// Case 2: Since provided, Hours is zero -> Hours calculated from difference
	explicitSince := time.Date(2026, 9, 8, 3, 25, 0, 0, time.UTC)
	opts2 := Options{
		Now:   fixedNow,
		Since: explicitSince,
	}
	norm2, err := normalizeOptions(opts2)
	if err != nil {
		t.Fatal(err)
	}
	if norm2.Hours != 9.0 {
		t.Errorf("expected Hours=9.0, got %f", norm2.Hours)
	}

	// Case 3: Both zero -> Hours defaults to 24, Since = Now - 24h
	opts3 := Options{
		Now: fixedNow,
	}
	norm3, err := normalizeOptions(opts3)
	if err != nil {
		t.Fatal(err)
	}
	if norm3.Hours != 24 {
		t.Errorf("expected Hours=24, got %f", norm3.Hours)
	}
	expectedSince3 := fixedNow.Add(-24 * time.Hour)
	if !norm3.Since.Equal(expectedSince3) {
		t.Errorf("expected Since=%v, got %v", expectedSince3, norm3.Since)
	}
}

func TestCollect_GhParsing_TimeFiltering_StateReason(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)
	// Window: 2026-09-08 03:25 to 12:25 UTC (9.0h)

	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		repo := ""
		for i, a := range args {
			if a == "--repo" && i+1 < len(args) {
				repo = args[i+1]
			}
		}

		if repo == "anthony-chaudhary/fak" {
			issues := []map[string]any{
				// In window, COMPLETED
				{
					"number":      101,
					"title":       "fix(agent): align buffer-pool contention witness",
					"closedAt":    "2026-09-08T10:15:00Z",
					"stateReason": "COMPLETED",
					"url":         "https://github.com/anthony-chaudhary/fak/issues/101",
				},
				// In window, NOT_PLANNED
				{
					"number":      102,
					"title":       "chore: obsolete proposal",
					"closedAt":    "2026-09-08T09:30:00Z",
					"stateReason": "NOT_PLANNED",
					"url":         "https://github.com/anthony-chaudhary/fak/issues/102",
				},
				// Outside window: before Since (closed 2 days ago)
				{
					"number":      100,
					"title":       "old bug fixed earlier",
					"closedAt":    "2026-09-06T10:00:00Z",
					"stateReason": "COMPLETED",
					"url":         "https://github.com/anthony-chaudhary/fak/issues/100",
				},
				// Outside window: after Until
				{
					"number":      103,
					"title":       "future issue",
					"closedAt":    "2026-09-08T15:00:00Z",
					"stateReason": "COMPLETED",
					"url":         "https://github.com/anthony-chaudhary/fak/issues/103",
				},
			}
			return json.Marshal(issues)
		}

		if repo == "anthony-chaudhary/fak-private" {
			issues := []map[string]any{
				// In window, completed (lowercase)
				{
					"number":      201,
					"title":       "private fix: credential rotation",
					"closedAt":    "2026-09-08T08:00:00Z",
					"stateReason": "completed",
					"url":         "https://github.com/anthony-chaudhary/fak-private/issues/201",
				},
				// In window, not_planned (lowercase)
				{
					"number":      202,
					"title":       "private test wontfix",
					"closedAt":    "2026-09-08T06:00:00Z",
					"stateReason": "not_planned",
					"url":         "https://github.com/anthony-chaudhary/fak-private/issues/202",
				},
			}
			return json.Marshal(issues)
		}

		return nil, fmt.Errorf("unexpected repo: %s", repo)
	}

	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "rev-list" {
			if strings.Contains(dir, "fak-private") {
				return []byte("38\n"), nil
			}
			return []byte("180\n"), nil
		}
		return nil, errors.New("unsupported git command")
	}

	opts := Options{
		Hours:          9.0,
		Now:            fixedNow,
		IncludePrivate: true,
		GhExecutor:     mockGh,
		GitExecutor:    mockGit,
	}

	rep, err := Collect(context.Background(), opts)
	if err != nil {
		t.Fatalf("Collect failed: %v", err)
	}

	if rep.Schema != Schema {
		t.Errorf("expected Schema=%q, got %q", Schema, rep.Schema)
	}
	if rep.Hours != 9.0 {
		t.Errorf("expected Hours=9.0, got %f", rep.Hours)
	}
	if len(rep.Repos) != 2 {
		t.Fatalf("expected 2 repos, got %d", len(rep.Repos))
	}

	// Verify Public repo
	pub := rep.Repos[0]
	if pub.Repo != "anthony-chaudhary/fak" {
		t.Errorf("expected public repo name 'anthony-chaudhary/fak', got %q", pub.Repo)
	}
	if pub.Source != "github" {
		t.Errorf("expected public Source='github', got %q", pub.Source)
	}
	if pub.Solved != 1 {
		t.Errorf("expected public Solved=1, got %d", pub.Solved)
	}
	if pub.NotPlanned != 1 {
		t.Errorf("expected public NotPlanned=1, got %d", pub.NotPlanned)
	}
	if pub.Closed != 2 {
		t.Errorf("expected public Closed=2, got %d", pub.Closed)
	}
	if pub.CommitCount != 180 {
		t.Errorf("expected public CommitCount=180, got %d", pub.CommitCount)
	}
	if len(pub.Issues) != 2 {
		t.Fatalf("expected 2 filtered issues in public repo, got %d", len(pub.Issues))
	}

	// Verify Private repo
	priv := rep.Repos[1]
	if priv.Repo != "anthony-chaudhary/fak-private" {
		t.Errorf("expected private repo name 'anthony-chaudhary/fak-private', got %q", priv.Repo)
	}
	if priv.Source != "github" {
		t.Errorf("expected private Source='github', got %q", priv.Source)
	}
	if priv.Solved != 1 {
		t.Errorf("expected private Solved=1, got %d", priv.Solved)
	}
	if priv.NotPlanned != 1 {
		t.Errorf("expected private NotPlanned=1, got %d", priv.NotPlanned)
	}
	if priv.Closed != 2 {
		t.Errorf("expected private Closed=2, got %d", priv.Closed)
	}
	if priv.CommitCount != 38 {
		t.Errorf("expected private CommitCount=38, got %d", priv.CommitCount)
	}

	// Verify Totals
	if rep.TotalSolved != 2 {
		t.Errorf("expected TotalSolved=2, got %d", rep.TotalSolved)
	}
	if rep.TotalNotPlanned != 2 {
		t.Errorf("expected TotalNotPlanned=2, got %d", rep.TotalNotPlanned)
	}
	if rep.TotalClosed != 4 {
		t.Errorf("expected TotalClosed=4, got %d", rep.TotalClosed)
	}
	if rep.TotalCommits != 218 {
		t.Errorf("expected TotalCommits=218, got %d", rep.TotalCommits)
	}
}

func TestCollect_FallbackToGit(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)

	// GhExecutor fails (e.g. network down, rate limit, no gh auth)
	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		return nil, errors.New("gh: command failed (connection refused)")
	}

	// GitExecutor returns git log output with referenced issues
	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "rev-list" {
			return []byte("42\n"), nil
		}
		if len(args) > 0 && args[0] == "log" {
			// Two records separated by \x1e, fields separated by \x1f
			// record 1: fixes issue 501
			rec1 := "sha1\x1f2026-09-08T10:00:00Z\x1ffix(core): resolve buffer leak (#501)\x1fCloses #501"
			// record 2: fixes issue 502
			rec2 := "sha2\x1f2026-09-08T09:00:00Z\x1ffeat(engine): optimize dispatch (fak #502)\x1fSigned-off-by: test"
			// record 3: duplicate reference to #501 (should be deduped)
			rec3 := "sha3\x1f2026-09-08T08:00:00Z\x1ftest(core): add regression test for #501\x1f"
			return []byte(rec1 + "\x1e" + rec2 + "\x1e" + rec3 + "\x1e"), nil
		}
		return nil, errors.New("unknown git command")
	}

	opts := Options{
		Hours:          9.0,
		Now:            fixedNow,
		IncludePrivate: false,
		Source:         "auto",
		GhExecutor:     mockGh,
		GitExecutor:    mockGit,
	}

	rep, err := Collect(context.Background(), opts)
	if err != nil {
		t.Fatalf("Collect fallback failed: %v", err)
	}

	if len(rep.Repos) != 1 {
		t.Fatalf("expected 1 repo, got %d", len(rep.Repos))
	}
	r := rep.Repos[0]
	if r.Source != "git" {
		t.Errorf("expected Source='git', got %q", r.Source)
	}
	if r.CommitCount != 42 {
		t.Errorf("expected CommitCount=42, got %d", r.CommitCount)
	}
	if r.Solved != 2 {
		t.Errorf("expected Solved=2, got %d", r.Solved)
	}
	if r.Closed != 2 {
		t.Errorf("expected Closed=2, got %d", r.Closed)
	}
	if r.NotPlanned != 0 {
		t.Errorf("expected NotPlanned=0, got %d", r.NotPlanned)
	}
	if len(r.Issues) != 2 {
		t.Fatalf("expected 2 unique issues extracted from git, got %d", len(r.Issues))
	}
	if r.Issues[0].Number != 501 && r.Issues[1].Number != 501 {
		t.Errorf("expected issue 501 in issues list, got %+v", r.Issues)
	}
	if r.Issues[0].Number != 502 && r.Issues[1].Number != 502 {
		t.Errorf("expected issue 502 in issues list, got %+v", r.Issues)
	}
}

func TestCollect_SourceExplicitGit(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	ghCalled := false
	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		ghCalled = true
		return nil, errors.New("gh should not be called when Source=git")
	}

	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "rev-list" {
			return []byte("15\n"), nil
		}
		if len(args) > 0 && args[0] == "log" {
			rec := "sha1\x1f2026-09-08T11:00:00Z\x1ffix: resolve (#99)\x1f"
			return []byte(rec + "\x1e"), nil
		}
		return nil, nil
	}

	opts := Options{
		Source:         "git",
		Now:            fixedNow,
		IncludePrivate: false,
		GhExecutor:     mockGh,
		GitExecutor:    mockGit,
	}

	rep, err := Collect(context.Background(), opts)
	if err != nil {
		t.Fatalf("Collect failed: %v", err)
	}
	if ghCalled {
		t.Errorf("GhExecutor was called unexpectedly when Source='git'")
	}
	if rep.Repos[0].Source != "git" {
		t.Errorf("expected Source='git', got %q", rep.Repos[0].Source)
	}
	if rep.TotalSolved != 1 {
		t.Errorf("expected TotalSolved=1, got %d", rep.TotalSolved)
	}
}

func TestCollect_SourceExplicitGithubError(t *testing.T) {
	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		return nil, errors.New("gh authentication error")
	}

	opts := Options{
		Source:         "github",
		IncludePrivate: false,
		GhExecutor:     mockGh,
	}

	_, err := Collect(context.Background(), opts)
	if err == nil {
		t.Fatalf("expected error when Source='github' and gh fails, got nil")
	}
	if !strings.Contains(err.Error(), "gh authentication error") {
		t.Errorf("expected error to contain gh message, got %v", err)
	}
}

func TestRenderSummary(t *testing.T) {
	fixedSince := time.Date(2026, 9, 8, 3, 25, 0, 0, time.UTC)
	fixedUntil := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)

	rep := &Report{
		Schema:          Schema,
		GeneratedAt:     fixedUntil,
		Since:           fixedSince,
		Until:           fixedUntil,
		Hours:           9.0,
		TotalSolved:     60,
		TotalClosed:     70,
		TotalNotPlanned: 10,
		TotalCommits:    218,
		Repos: []RepoReport{
			{
				Repo:        "anthony-chaudhary/fak",
				Solved:      46,
				Closed:      48,
				NotPlanned:  2,
				CommitCount: 180,
			},
			{
				Repo:        "anthony-chaudhary/fak-private",
				Solved:      14,
				Closed:      22,
				NotPlanned:  8,
				CommitCount: 38,
			},
		},
	}

	var buf bytes.Buffer
	if err := RenderSummary(&buf, rep); err != nil {
		t.Fatalf("RenderSummary failed: %v", err)
	}

	output := buf.String()
	lines := strings.Split(strings.TrimRight(output, "\n"), "\n")
	if len(lines) > 4 {
		t.Errorf("expected <= 4 lines summary, got %d lines:\n%s", len(lines), output)
	}

	expectedPrefix := "Issues solved in last 9.0h (2026-09-08 03:25 to 12:25 UTC):"
	if !strings.HasPrefix(lines[0], expectedPrefix) {
		t.Errorf("expected header %q, got %q", expectedPrefix, lines[0])
	}

	expectedPub := "  anthony-chaudhary/fak:         46 solved (48 closed, 2 not planned) | 180 commits"
	if lines[1] != expectedPub {
		t.Errorf("line 1 mismatch:\nwant: %q\ngot:  %q", expectedPub, lines[1])
	}

	expectedPriv := "  anthony-chaudhary/fak-private: 14 solved (22 closed, 8 not planned) | 38 commits"
	if lines[2] != expectedPriv {
		t.Errorf("line 2 mismatch:\nwant: %q\ngot:  %q", expectedPriv, lines[2])
	}

	expectedTotal := "  TOTAL:                         60 solved (70 closed, 10 not planned) | 218 trunk commits"
	if lines[3] != expectedTotal {
		t.Errorf("line 3 mismatch:\nwant: %q\ngot:  %q", expectedTotal, lines[3])
	}
}

func TestRenderDetailed(t *testing.T) {
	fixedSince := time.Date(2026, 9, 8, 3, 25, 0, 0, time.UTC)
	fixedUntil := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)
	issueTime := time.Date(2026, 9, 8, 8, 0, 0, 0, time.UTC)

	rep := &Report{
		Schema:          Schema,
		GeneratedAt:     fixedUntil,
		Since:           fixedSince,
		Until:           fixedUntil,
		Hours:           9.0,
		TotalSolved:     1,
		TotalClosed:     1,
		TotalNotPlanned: 0,
		TotalCommits:    10,
		Repos: []RepoReport{
			{
				Repo:        "anthony-chaudhary/fak",
				Solved:      1,
				Closed:      1,
				NotPlanned:  0,
				CommitCount: 10,
				Issues: []IssueItem{
					{
						Number:      1234,
						Title:       "fix(agent): handle race condition",
						ClosedAt:    issueTime,
						StateReason: "COMPLETED",
						URL:         "https://github.com/anthony-chaudhary/fak/issues/1234",
					},
				},
			},
		},
	}

	var buf bytes.Buffer
	if err := RenderDetailed(&buf, rep); err != nil {
		t.Fatalf("RenderDetailed failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "Issues solved in last 9.0h") {
		t.Errorf("expected summary in detailed view")
	}
	if !strings.Contains(out, "#1234") {
		t.Errorf("expected issue #1234 in detailed view")
	}
	if !strings.Contains(out, "fix(agent): handle race condition") {
		t.Errorf("expected issue title in detailed view")
	}
	if !strings.Contains(out, "2026-09-08 08:00 UTC") {
		t.Errorf("expected issue timestamp in detailed view")
	}
}

func TestRenderJSON(t *testing.T) {
	fixedSince := time.Date(2026, 9, 8, 3, 25, 0, 0, time.UTC)
	fixedUntil := time.Date(2026, 9, 8, 12, 25, 0, 0, time.UTC)

	rep := &Report{
		Schema:          Schema,
		GeneratedAt:     fixedUntil,
		Since:           fixedSince,
		Until:           fixedUntil,
		Hours:           9.0,
		TotalSolved:     5,
		TotalClosed:     6,
		TotalNotPlanned: 1,
		TotalCommits:    20,
		Repos: []RepoReport{
			{
				Repo:        "anthony-chaudhary/fak",
				Solved:      5,
				Closed:      6,
				NotPlanned:  1,
				CommitCount: 20,
			},
		},
	}

	var buf bytes.Buffer
	if err := RenderJSON(&buf, rep); err != nil {
		t.Fatalf("RenderJSON failed: %v", err)
	}

	var decoded Report
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("failed to decode JSON output: %v", err)
	}

	if decoded.Schema != Schema {
		t.Errorf("expected Schema=%q, got %q", Schema, decoded.Schema)
	}
	if decoded.TotalSolved != 5 {
		t.Errorf("expected TotalSolved=5, got %d", decoded.TotalSolved)
	}
	if decoded.TotalClosed != 6 {
		t.Errorf("expected TotalClosed=6, got %d", decoded.TotalClosed)
	}
	if decoded.TotalNotPlanned != 1 {
		t.Errorf("expected TotalNotPlanned=1, got %d", decoded.TotalNotPlanned)
	}
	if decoded.TotalCommits != 20 {
		t.Errorf("expected TotalCommits=20, got %d", decoded.TotalCommits)
	}
}

func TestResolvePrivateDir_Env(t *testing.T) {
	tempDir := t.TempDir()
	t.Setenv("FAK_PRIVATE_ROOT", tempDir)
	got := resolvePrivateDir(".", "")
	if got != tempDir {
		t.Errorf("expected %q, got %q", tempDir, got)
	}
}

func TestReport_ProvenanceAndFilter(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	fixedSince := time.Date(2026, 9, 8, 3, 0, 0, 0, time.UTC)

	var capturedGhArgs []string
	mockGh := func(ctx context.Context, args ...string) ([]byte, error) {
		capturedGhArgs = args
		issues := []map[string]any{
			{
				"number":      12162,
				"title":       "feat(issuesolved): record provenance in report schema",
				"closedAt":    "2026-09-08T05:00:00Z",
				"stateReason": "COMPLETED",
				"url":         "https://github.com/anthony-chaudhary/fak/issues/12162",
			},
		}
		return json.Marshal(issues)
	}

	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		if len(args) > 0 && args[0] == "rev-list" {
			return []byte("5\n"), nil
		}
		return nil, nil
	}

	// 1. Verify GitHub source and search filter generation
	optsGh := Options{
		Since:          fixedSince,
		Now:            fixedNow,
		IncludePrivate: false,
		Source:         "github",
		GhExecutor:     mockGh,
		GitExecutor:    mockGit,
	}

	repGh, err := Collect(context.Background(), optsGh)
	if err != nil {
		t.Fatalf("Collect failed for github source: %v", err)
	}
	if len(repGh.Repos) != 1 {
		t.Fatalf("expected 1 repo, got %d", len(repGh.Repos))
	}
	if repGh.Repos[0].Source != "github" {
		t.Errorf("expected Repos[0].Source='github', got %q", repGh.Repos[0].Source)
	}

	// Verify gh search args contain --search closed:>=<since>
	hasSearch := false
	expectedFilter := fmt.Sprintf("closed:>=%s", fixedSince.UTC().Format(time.RFC3339))
	for i, arg := range capturedGhArgs {
		if arg == "--search" && i+1 < len(capturedGhArgs) && capturedGhArgs[i+1] == expectedFilter {
			hasSearch = true
			break
		}
	}
	if !hasSearch {
		t.Errorf("expected captured gh args to contain '--search %s', got: %v", expectedFilter, capturedGhArgs)
	}

	// Verify JSON serialization includes "source": "github"
	var bufGh bytes.Buffer
	if err := RenderJSON(&bufGh, repGh); err != nil {
		t.Fatalf("RenderJSON failed: %v", err)
	}
	if !strings.Contains(bufGh.String(), `"source": "github"`) {
		t.Errorf("expected JSON to contain '\"source\": \"github\"', got: %s", bufGh.String())
	}

	// 2. Verify Git source provenance
	optsGit := Options{
		Since:          fixedSince,
		Now:            fixedNow,
		IncludePrivate: false,
		Source:         "git",
		GitExecutor: func(ctx context.Context, dir string, args ...string) ([]byte, error) {
			if len(args) > 0 && args[0] == "rev-list" {
				return []byte("2\n"), nil
			}
			if len(args) > 0 && args[0] == "log" {
				rec := "sha1\x1f2026-09-08T04:00:00Z\x1ffix: git issue (#12162)\x1f"
				return []byte(rec + "\x1e"), nil
			}
			return nil, nil
		},
	}

	repGit, err := Collect(context.Background(), optsGit)
	if err != nil {
		t.Fatalf("Collect failed for git source: %v", err)
	}
	if repGit.Repos[0].Source != "git" {
		t.Errorf("expected Repos[0].Source='git', got %q", repGit.Repos[0].Source)
	}

	var bufGit bytes.Buffer
	if err := RenderJSON(&bufGit, repGit); err != nil {
		t.Fatalf("RenderJSON failed: %v", err)
	}
	if !strings.Contains(bufGit.String(), `"source": "git"`) {
		t.Errorf("expected JSON to contain '\"source\": \"git\"', got: %s", bufGit.String())
	}

	// 3. Verify Auto fallback to Git provenance
	optsAutoFallback := Options{
		Since:          fixedSince,
		Now:            fixedNow,
		IncludePrivate: false,
		Source:         "auto",
		GhExecutor: func(ctx context.Context, args ...string) ([]byte, error) {
			return nil, errors.New("gh offline")
		},
		GitExecutor: func(ctx context.Context, dir string, args ...string) ([]byte, error) {
			if len(args) > 0 && args[0] == "rev-list" {
				return []byte("1\n"), nil
			}
			return nil, nil
		},
	}

	repAuto, err := Collect(context.Background(), optsAutoFallback)
	if err != nil {
		t.Fatalf("Collect failed for auto fallback: %v", err)
	}
	if repAuto.Repos[0].Source != "git" {
		t.Errorf("expected Repos[0].Source='git' on gh failure, got %q", repAuto.Repos[0].Source)
	}

	// 4. Verify that zero Since does not generate --search flag in collectIssuesFromGh
	var capturedZeroArgs []string
	_, _ = collectIssuesFromGh(context.Background(), "anthony-chaudhary/fak", time.Time{}, fixedNow, func(ctx context.Context, args ...string) ([]byte, error) {
		capturedZeroArgs = args
		return []byte("[]"), nil
	})
	for _, arg := range capturedZeroArgs {
		if arg == "--search" {
			t.Errorf("expected no '--search' arg when since is zero, got: %v", capturedZeroArgs)
		}
	}
}

func TestCollect_MissingPrivateDir(t *testing.T) {
	fixedNow := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	missingDir := filepath.Join(t.TempDir(), "nonexistent-fak-private")

	// Case 1: normalizeOptions disables IncludePrivate when Source="git" and PrivateDir does not exist
	optsNorm := Options{
		Source:         "git",
		Now:            fixedNow,
		IncludePrivate: true,
		PublicDir:      ".",
		PrivateDir:     missingDir,
	}
	norm, err := normalizeOptions(optsNorm)
	if err != nil {
		t.Fatal(err)
	}
	if norm.IncludePrivate {
		t.Errorf("expected normalizeOptions to set IncludePrivate=false for missing private dir in git mode")
	}

	// Case 2: Collect with Source="git" and missing private dir succeeds gracefully (zero exit, nil error)
	rep, err := Collect(context.Background(), optsNorm)
	if err != nil {
		t.Fatalf("expected Collect to succeed gracefully when private dir is missing, got error: %v", err)
	}
	if rep == nil {
		t.Fatal("expected non-nil Report")
	}
	if len(rep.Repos) != 1 {
		t.Errorf("expected 1 repo (public only), got %d repos", len(rep.Repos))
	}
	if rep.Repos[0].Repo != "anthony-chaudhary/fak" {
		t.Errorf("expected public repo in report, got %q", rep.Repos[0].Repo)
	}

	// Case 3: Direct collectRepo on missing directory in git mode returns (rr, nil)
	rrDirect, errDirect := collectRepo(context.Background(), "anthony-chaudhary/fak-private", missingDir, optsNorm, nil, nil)
	if errDirect != nil {
		t.Fatalf("expected collectRepo on missing dir to return nil error, got: %v", errDirect)
	}
	if rrDirect.CommitCount != 0 {
		t.Errorf("expected 0 commits for missing dir, got %d", rrDirect.CommitCount)
	}
	if len(rrDirect.Issues) != 0 {
		t.Errorf("expected 0 issues for missing dir, got %d", len(rrDirect.Issues))
	}

	// Case 4: Source="git" with custom GitExecutor returning chdir failure on missing private dir
	mockGit := func(ctx context.Context, dir string, args ...string) ([]byte, error) {
		if dir == missingDir {
			return nil, fmt.Errorf("chdir %s: no such file or directory", missingDir)
		}
		if len(args) > 0 && args[0] == "rev-list" {
			return []byte("1\n"), nil
		}
		if len(args) > 0 && args[0] == "log" {
			rec := "sha1\x1f2026-09-08T11:00:00Z\x1ffix: public issue (#10)\x1f"
			return []byte(rec + "\x1e"), nil
		}
		return nil, nil
	}

	optsMock := Options{
		Source:         "git",
		Now:            fixedNow,
		IncludePrivate: true,
		PublicDir:      ".",
		PrivateDir:     missingDir,
		GitExecutor:    mockGit,
	}

	repMock, errMock := Collect(context.Background(), optsMock)
	if errMock != nil {
		t.Fatalf("expected Collect with mock git to succeed gracefully when private dir is missing, got: %v", errMock)
	}
	if repMock == nil {
		t.Fatal("expected non-nil Report for mock git")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCollect_TimeValidation(t *testing.T) {
	now := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name string
		opts Options
	}{
		{"negative hours", Options{Hours: -2}},
		{"negative hours with since", Options{Hours: -2, Since: now.Add(-time.Hour)}},
		{"nan hours", Options{Hours: math.NaN()}},
		{"infinite hours", Options{Hours: math.Inf(1)}},
		{"overflow hours", Options{Hours: float64(math.MaxInt64) / float64(time.Hour)}},
		{"subnanosecond hours", Options{Hours: math.SmallestNonzeroFloat64}},
		{"future since", Options{Since: now.Add(time.Hour)}},
		{"zero window", Options{Since: now}},
		{"overflow since", Options{Since: now.AddDate(-300, 0, 0)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := tc.opts
			opts.Now = now
			opts.GhExecutor = func(context.Context, ...string) ([]byte, error) {
				t.Fatal("invalid window invoked gh")
				return nil, nil
			}
			opts.GitExecutor = func(context.Context, string, ...string) ([]byte, error) {
				t.Fatal("invalid window invoked git")
				return nil, nil
			}
			if rep, err := Collect(context.Background(), opts); err == nil || rep != nil {
				t.Fatalf("Collect = (%v, %v), want nil report and error", rep, err)
			}
		})
	}
	largestHours := math.Nextafter(float64(math.MaxInt64)/float64(time.Hour), 0)
	for _, tc := range []struct {
		name string
		opts Options
		want time.Duration
	}{
		{"zero value default", Options{}, 24 * time.Hour},
		{"fractional hours", Options{Hours: 0.5}, 30 * time.Minute},
		{"one nanosecond", Options{Hours: 1 / float64(time.Hour)}, time.Nanosecond},
		{"largest safe hours", Options{Hours: largestHours}, time.Duration(largestHours * float64(time.Hour))},
		{"since authoritative", Options{Hours: 24, Since: now.Add(-2 * time.Hour)}, 2 * time.Hour},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := tc.opts
			opts.Now = now
			opts.Source = "github"
			opts.GhExecutor = func(context.Context, ...string) ([]byte, error) { return []byte("[]"), nil }
			opts.GitExecutor = func(context.Context, string, ...string) ([]byte, error) { return []byte("0"), nil }
			rep, err := Collect(context.Background(), opts)
			if err != nil {
				t.Fatal(err)
			}
			if !rep.Since.Equal(now.Add(-tc.want)) || rep.Hours != tc.want.Hours() || !rep.Until.Equal(now) {
				t.Fatalf("inconsistent report: since=%v until=%v hours=%v", rep.Since, rep.Until, rep.Hours)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestOptions_TimeCalculationUsesWallTime(t *testing.T) {
	now := time.Now()
	since := now.Add(-time.Hour)
	opts, err := normalizeOptions(Options{Now: now, Since: since})
	if err != nil {
		t.Fatal(err)
	}
	// Equality, rather than Time.Equal, also detects retained monotonic readings.
	if opts.Now != now.Round(0) || opts.Since != since.Round(0) || opts.Hours != 1 {
		t.Fatalf("window retained monotonic readings or inconsistent hours: %+v", opts)
	}
}
