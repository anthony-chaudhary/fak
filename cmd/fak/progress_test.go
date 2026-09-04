package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRunProgressReportsDeliveredWIPAndIssues(t *testing.T) {
	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current, WIPFilesDelta: -1}, nil
	}
	oldCmd, oldNow := progressCommand, progressNow
	t.Cleanup(func() { progressCommand = oldCmd; progressNow = oldNow })
	progressNow = func() time.Time { return time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC) }
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(key, "git rev-list") && strings.Contains(key, "--until="):
			return []byte("4\n"), nil
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("2\n"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M tracked\x00?? new\x00A  staged\x00"), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return []byte("4\t2\ttracked\x001\t0\tstaged\x00"), nil
		case strings.Contains(key, "--state closed") && strings.Contains(key, ".."):
			return []byte(`[{"number":8},{"number":9}]`), nil
		case strings.Contains(key, "--state closed"):
			return []byte(`[{"number":1}]`), nil
		case strings.Contains(key, "--state open"):
			return []byte(`[{"number":2},{"number":3}]`), nil
		}
		t.Fatalf("unexpected %s", key)
		return nil, nil
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var got progressReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "CONVERGING" || got.Commits != 2 || got.CommitsPer10M != 2 || got.WIP.Files != 3 || got.WIP.Untracked != 1 || got.WIP.Staged != 1 || got.WIP.Unstaged != 1 || !got.GitHub.Available || got.GitHub.RecentlyClosed != 1 || got.GitHub.OpenTotal != 2 || got.Baseline.Commits != 4 || got.Baseline.IssuesClosed != 2 || got.Baseline.CommitsPer10M != 2 || got.Baseline.IssueClosesPer10M != 1 || got.IssueClosesPer10M != 1 || got.CLQ != 1.0 || got.DrainVelocityPerHour != 12.0 || got.WIPHalfLifeMinutes != 0.0 {
		t.Fatalf("unexpected report: %+v", got)
	}
	for _, key := range []string{`"clq"`, `"wip_halflife_minutes"`, `"drain_velocity_per_hour"`} {
		if !strings.Contains(out.String(), key) {
			t.Fatalf("json missing %s: %s", key, out.String())
		}
	}
}

func TestRunProgressDoesNotCallWIPProgressAndFailsSoftWithoutGitHub(t *testing.T) {
	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Closing: current, Reason: "GitHub inventory unavailable at one or both boundaries"}, nil
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		if strings.HasPrefix(key, "git rev-list") {
			return []byte("0"), nil
		}
		if strings.HasPrefix(key, "git status") {
			return []byte(" M tracked\x00"), nil
		}
		if strings.HasPrefix(key, "git diff --numstat") {
			return nil, nil
		}
		return nil, errors.New("offline")
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "10m"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	s := out.String()
	for _, want := range []string{"UNKNOWN", "commits=0", "files=1", "GitHub: unavailable"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in %s", want, s)
		}
	}
}

func TestParseProgressWIPCountsRenameOnce(t *testing.T) {
	var r progressReport
	paths := parseProgressWIP([]byte("R  new\x00old\x00UU conflict\x00"), &r)
	if r.WIP.Files != 2 || r.WIP.Staged != 2 || r.WIP.Conflicts != 1 || len(paths) != 2 {
		t.Fatalf("unexpected WIP: %+v", r.WIP)
	}
}

func TestRunProgressOneQuietWindowDoesNotOverAlert(t *testing.T) {
	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current}, nil
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(key, "git rev-list") && strings.Contains(key, "--until="):
			return []byte("2"), nil
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("0"), nil
		case strings.HasPrefix(key, "git status"):
			return nil, nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return nil, nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, errors.New("unexpected " + key)
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "20m"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "FLOW_STALLED") {
		t.Fatalf("one quiet window over-alerted: %s", out.String())
	}
}

func TestRunProgressStalledAfterDeclaredConsecutiveWindows(t *testing.T) {
	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current}, nil
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("0"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M unfinished\x00"), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return nil, nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, errors.New("unexpected " + key)
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "20m", "--stall-after", "3", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	var got progressReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "FLOW_STALLED" || got.StallAfterWindows != 3 || got.WIP.Files != 1 {
		t.Fatalf("unexpected report: %+v", got)
	}
}

func TestCollectProgressWIPDetailsMeasuresMagnitudeAgeAndUntrackedBytes(t *testing.T) {
	dir := t.TempDir()
	tracked := filepath.Join(dir, "tracked.txt")
	untracked := filepath.Join(dir, "new.bin")
	if err := os.WriteFile(tracked, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(untracked, []byte{1, 2, 3, 4, 5}, 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Date(2026, 8, 17, 10, 0, 0, 0, time.UTC)
	if err := os.Chtimes(tracked, old, old); err != nil {
		t.Fatal(err)
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		if name == "git" && len(args) > 1 && args[0] == "diff" {
			return []byte("7\t3\ttracked.txt\x00-\t-\timage.png\x00"), nil
		}
		return nil, errors.New("unexpected command")
	}
	var r progressReport
	err := collectProgressWIPDetails(dir, []progressWIPPath{{path: "tracked.txt"}, {path: "new.bin", untracked: true}, {path: "missing.txt", untracked: true}}, old.Add(95*time.Minute), &r)
	if err != nil {
		t.Fatal(err)
	}
	if r.WIP.Additions != 7 || r.WIP.Deletions != 3 || r.WIP.BinaryFiles != 1 || r.WIP.UntrackedBytes != 5 || r.WIP.OldestExistingMinutes != 95 || r.WIP.AgeFilesObserved != 2 || r.WIP.AgeFilesUnavailable != 1 {
		t.Fatalf("unexpected details: %+v", r.WIP)
	}
}

func TestCollectProgressWIPDetailsDoesNotReadOutsideRepository(t *testing.T) {
	dir := t.TempDir()
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, _ string, _ ...string) ([]byte, error) { return nil, nil }
	var r progressReport
	if err := collectProgressWIPDetails(dir, []progressWIPPath{{path: "../outside"}}, time.Now(), &r); err != nil {
		t.Fatal(err)
	}
	if r.WIP.AgeFilesUnavailable != 1 || r.WIP.AgeFilesObserved != 0 {
		t.Fatalf("outside path was observed: %+v", r.WIP)
	}
}

func TestRunProgressRecentCommitsCannotMaskGrowingInventory(t *testing.T) {
	oldCmd, oldInventory := progressCommand, progressInventory
	t.Cleanup(func() { progressCommand, progressInventory = oldCmd, oldInventory })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		joined := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(joined, "rev-list --count"):
			return []byte("2\n"), nil
		case strings.Contains(joined, "status --porcelain"):
			return []byte(" M tracked\x00"), nil
		case strings.Contains(joined, "diff --numstat"):
			return []byte("4\t2\ttracked\x00"), nil
		case strings.Contains(joined, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(joined, "--state open"):
			return []byte("[{\"number\":1},{\"number\":2}]"), nil
		}
		return nil, fmt.Errorf("unexpected command %s", joined)
	}
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current, WIPFilesDelta: 1, OpenIssuesDelta: 2}, nil
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got progressReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "DIVERGING" || !strings.Contains(got.NextAction, "before launching") {
		t.Fatalf("verdict=%q next=%q", got.Verdict, got.NextAction)
	}
}

func TestObserveProgressInventoryComparesPersistedOpeningBoundary(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	opening := progressInventorySnapshot{ObservedAt: now.Add(-10 * time.Minute), WIPFiles: 8, WIPLines: 50, UntrackedBytes: 20, OldestWIPMinutes: 40, OpenIssues: 100, GitHubAvailable: true}
	path, err := progressInventoryPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeProgressInventoryHistory(path, progressInventoryHistory{Schema: "fak-progress-inventory/1", Snapshots: []progressInventorySnapshot{opening}}); err != nil {
		t.Fatal(err)
	}
	closing := progressInventorySnapshot{ObservedAt: now, WIPFiles: 6, WIPLines: 40, UntrackedBytes: 10, OldestWIPMinutes: 50, OpenIssues: 98, GitHubAvailable: true}
	flow, err := observeProgressInventory(dir, 10*time.Minute, closing)
	if err != nil {
		t.Fatal(err)
	}
	if !flow.Available || flow.WIPFilesDelta != -2 || flow.WIPLinesDelta != -10 || flow.UntrackedBytesDelta != -10 || flow.OldestWIPMinutesDelta != 10 || flow.OpenIssuesDelta != -2 {
		t.Fatalf("flow=%+v", flow)
	}
}

func TestObserveProgressInventoryRequiresBothGitHubBoundaries(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	path, _ := progressInventoryPath(dir)
	opening := progressInventorySnapshot{ObservedAt: now.Add(-10 * time.Minute), GitHubAvailable: false}
	if err := writeProgressInventoryHistory(path, progressInventoryHistory{Schema: "fak-progress-inventory/1", Snapshots: []progressInventorySnapshot{opening}}); err != nil {
		t.Fatal(err)
	}
	flow, err := observeProgressInventory(dir, 10*time.Minute, progressInventorySnapshot{ObservedAt: now, GitHubAvailable: true})
	if err != nil {
		t.Fatal(err)
	}
	if flow.Available || !strings.Contains(flow.Reason, "GitHub") {
		t.Fatalf("flow=%+v", flow)
	}
}

func TestProgressCLQ(t *testing.T) {
	cases := []struct {
		name      string
		conflicts int
		untracked int
		diffLines int64
		want      float64
	}{
		{name: "clean", conflicts: 0, untracked: 0, diffLines: 0, want: 1.00},
		{name: "single_conflict", conflicts: 1, untracked: 0, diffLines: 0, want: 0.00},
		{name: "conflicts_override_all", conflicts: 2, untracked: 50, diffLines: 2000, want: 0.00},
		{name: "untracked_10_penalty", conflicts: 0, untracked: 10, diffLines: 0, want: 0.95},
		{name: "untracked_20_penalty", conflicts: 0, untracked: 20, diffLines: 0, want: 0.90},
		{name: "untracked_60_max_penalty", conflicts: 0, untracked: 60, diffLines: 0, want: 0.70},
		{name: "untracked_100_capped_penalty", conflicts: 0, untracked: 100, diffLines: 0, want: 0.70},
		{name: "diff_exact_500_no_penalty", conflicts: 0, untracked: 0, diffLines: 500, want: 1.00},
		{name: "diff_501_penalty", conflicts: 0, untracked: 0, diffLines: 501, want: 0.80},
		{name: "untracked_and_diff_combined", conflicts: 0, untracked: 20, diffLines: 550, want: 0.70},
		{name: "max_penalties_combined", conflicts: 0, untracked: 60, diffLines: 1000, want: 0.50},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := computeCLQ(tc.conflicts, tc.untracked, tc.diffLines)
			if got != tc.want {
				t.Fatalf("computeCLQ(%d, %d, %d) = %v; want %v", tc.conflicts, tc.untracked, tc.diffLines, got, tc.want)
			}
		})
	}
}

func TestProgressWIPHalfLife(t *testing.T) {
	cases := []struct {
		name          string
		ages          []float64
		oldestMinutes int64
		want          float64
	}{
		{name: "odd_count_median", ages: []float64{10.0, 25.0, 40.0}, oldestMinutes: 40, want: 25.00},
		{name: "even_count_median", ages: []float64{10.0, 20.0, 30.0, 40.0}, oldestMinutes: 40, want: 25.00},
		{name: "even_count_fractional", ages: []float64{10.0, 25.0}, oldestMinutes: 25, want: 17.50},
		{name: "single_observed", ages: []float64{15.5}, oldestMinutes: 15, want: 15.50},
		{name: "unsorted_observed", ages: []float64{50.0, 10.0, 30.0}, oldestMinutes: 50, want: 30.00},
		{name: "fallback_to_oldest_half", ages: nil, oldestMinutes: 50, want: 25.00},
		{name: "empty_and_zero_oldest", ages: nil, oldestMinutes: 0, want: 0.00},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := computeWIPHalfLife(tc.ages, tc.oldestMinutes)
			if got != tc.want {
				t.Fatalf("computeWIPHalfLife(%v, %d) = %v; want %v", tc.ages, tc.oldestMinutes, got, tc.want)
			}
		})
	}
}

func TestProgressDrainVelocity(t *testing.T) {
	cases := []struct {
		name          string
		commitsPer10M float64
		want          float64
	}{
		{name: "zero_commits", commitsPer10M: 0.0, want: 0.00},
		{name: "two_commits", commitsPer10M: 2.0, want: 12.00},
		{name: "fractional_rate", commitsPer10M: 1.5, want: 9.00},
		{name: "small_rate", commitsPer10M: 0.33, want: 1.98},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := computeDrainVelocity(tc.commitsPer10M)
			if got != tc.want {
				t.Fatalf("computeDrainVelocity(%v) = %v; want %v", tc.commitsPer10M, got, tc.want)
			}
		})
	}
}

func TestProgressJSONIncludes10xWIPMetrics(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	fileA := filepath.Join(dir, "file_a.go")
	fileB := filepath.Join(dir, "file_b.go")
	if err := os.WriteFile(fileA, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileB, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	simNow := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	modA := simNow.Add(-10 * time.Minute)
	modB := simNow.Add(-30 * time.Minute)
	if err := os.Chtimes(fileA, modA, modA); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(fileB, modB, modB); err != nil {
		t.Fatal(err)
	}

	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current}, nil
	}

	oldCmd, oldNow := progressCommand, progressNow
	t.Cleanup(func() { progressCommand = oldCmd; progressNow = oldNow })
	progressNow = func() time.Time { return simNow }

	// Build 10 untracked files in git status to trigger -0.05 untracked penalty
	var statusBuilder strings.Builder
	statusBuilder.WriteString(" M file_a.go\x00 M file_b.go\x00")
	for i := 0; i < 10; i++ {
		statusBuilder.WriteString(fmt.Sprintf("?? untracked_%d.go\x00", i))
	}

	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(key, "git rev-list") && strings.Contains(key, "--until="):
			return []byte("6\n"), nil
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("3\n"), nil // 3 commits in 10m -> CommitsPer10M = 3.0 -> DrainVelocity = 18.0
		case strings.HasPrefix(key, "git status"):
			return []byte(statusBuilder.String()), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			// 300 additions + 250 deletions = 550 diff lines (> 500 lines -> -0.20 penalty)
			return []byte("300\t250\tfile_a.go\x00"), nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, fmt.Errorf("unexpected command %s", key)
	}

	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--json", "--dir", dir}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}

	raw := out.String()
	for _, wantKey := range []string{`"clq"`, `"wip_halflife_minutes"`, `"drain_velocity_per_hour"`} {
		if !strings.Contains(raw, wantKey) {
			t.Fatalf("missing %s in json output: %s", wantKey, raw)
		}
	}

	var report progressReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}

	// CLQ: 1.0 base - 0.05 (10 untracked files) - 0.20 (>500 lines diff) = 0.75
	if report.CLQ != 0.75 {
		t.Fatalf("CLQ = %v, want 0.75", report.CLQ)
	}

	// WIPHalfLifeMinutes: median of [10.0, 30.0] = 20.0
	if report.WIPHalfLifeMinutes != 20.0 {
		t.Fatalf("WIPHalfLifeMinutes = %v, want 20.0", report.WIPHalfLifeMinutes)
	}

	// DrainVelocityPerHour: CommitsPer10M(3.0) * 6.0 = 18.0
	if report.DrainVelocityPerHour != 18.0 {
		t.Fatalf("DrainVelocityPerHour = %v, want 18.0", report.DrainVelocityPerHour)
	}
}

func TestProgressHumanOutputIncludes10xWIPMetrics(t *testing.T) {
	oldInventory := progressInventory
	t.Cleanup(func() { progressInventory = oldInventory })
	progressInventory = func(_ string, _ time.Duration, current progressInventorySnapshot) (progressFlow, error) {
		return progressFlow{Available: true, Closing: current}, nil
	}
	oldCmd := progressCommand
	t.Cleanup(func() { progressCommand = oldCmd })
	progressCommand = func(_, name string, args ...string) ([]byte, error) {
		key := name + " " + strings.Join(args, " ")
		switch {
		case strings.HasPrefix(key, "git rev-list"):
			return []byte("0"), nil
		case strings.HasPrefix(key, "git status"):
			return []byte(" M tracked\x00"), nil
		case strings.HasPrefix(key, "git diff --numstat"):
			return nil, nil
		case strings.Contains(key, "--state closed"):
			return []byte("[]"), nil
		case strings.Contains(key, "--state open"):
			return []byte("[]"), nil
		}
		return nil, errors.New("offline")
	}
	var out, errOut bytes.Buffer
	if code := runProgress(&out, &errOut, []string{"--baseline", "10m"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	s := out.String()
	for _, want := range []string{"metrics: clq=", "wip_halflife=", "drain_velocity="} {
		if !strings.Contains(s, want) {
			t.Fatalf("human output missing %q: %s", want, s)
		}
	}
}
