package issuesolved

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

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
	norm1 := normalizeOptions(opts1)
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
	norm2 := normalizeOptions(opts2)
	if norm2.Hours != 9.0 {
		t.Errorf("expected Hours=9.0, got %f", norm2.Hours)
	}

	// Case 3: Both zero -> Hours defaults to 24, Since = Now - 24h
	opts3 := Options{
		Now: fixedNow,
	}
	norm3 := normalizeOptions(opts3)
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
