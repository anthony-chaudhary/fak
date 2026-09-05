package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orgdebt"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func cmdOrgDebtScore(argv []string) {
	os.Exit(runOrgDebtScore(os.Stdout, os.Stderr, argv))
}

func runOrgDebtScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score org-debt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON report")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	limit := fs.Int("limit", 50, "max recent commits to audit for trunk hygiene")
	fixtureIssues := fs.String("fixture-issues", "", "path to JSON fixture file containing issues")
	fixtureCommits := fs.String("fixture-commits", "", "path to JSON fixture file containing commits")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score org-debt: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	input, err := orgdebt.ScanWorkspace(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak score org-debt: scan workspace: %v\n", err)
		return 1
	}

	// Load issues from fixture or empty
	if *fixtureIssues != "" {
		raw, err := os.ReadFile(*fixtureIssues)
		if err != nil {
			fmt.Fprintf(stderr, "fak score org-debt: read fixture-issues: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(raw, &input.Issues); err != nil {
			fmt.Fprintf(stderr, "fak score org-debt: parse fixture-issues: %v\n", err)
			return 2
		}
	}

	// Load commits from fixture or git log
	if *fixtureCommits != "" {
		raw, err := os.ReadFile(*fixtureCommits)
		if err != nil {
			fmt.Fprintf(stderr, "fak score org-debt: read fixture-commits: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(raw, &input.Commits); err != nil {
			fmt.Fprintf(stderr, "fak score org-debt: parse fixture-commits: %v\n", err)
			return 2
		}
	} else {
		commits, err := fetchRecentCommits(root, *limit)
		if err == nil {
			input.Commits = commits
		}
	}

	payload := orgdebt.Evaluate(input)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak score org-debt: %v\n", err)
			return 1
		}
		return 0
	}

	debt, _ := payload.Corpus[orgdebt.DebtKey].(int)
	score, _ := payload.Corpus["score"].(float64)
	grade, _ := payload.Corpus["grade"].(string)
	value, _ := payload.Corpus["value"].(float64)

	fmt.Fprintf(stdout, "Organization Debt Scorecard: Grade %s (score %.1f/100, debt %d defects, value %.3f)\n",
		grade, score, debt, value)
	fmt.Fprintf(stdout, "Finding: %s\n", payload.Finding)
	fmt.Fprintf(stdout, "Next action: %s\n\n", payload.NextAction)

	fmt.Fprintln(stdout, "KPI breakdown:")
	for _, kpi := range payload.KPIs {
		fmt.Fprintf(stdout, "  [%-16s] %-20s = %5.1f (defects: %d) - %s\n",
			kpi.Group, kpi.Key, kpi.Score, len(kpi.Defects), kpi.Detail)
	}

	if debt > 0 {
		fmt.Fprintln(stdout, "\nActive organization defects (worst-first):")
		count := 0
		for _, kpi := range payload.KPIs {
			for _, d := range kpi.Defects {
				count++
				fmt.Fprintf(stdout, "  %2d. [%s] %s\n", count, kpi.Key, d)
				if count >= 20 {
					fmt.Fprintln(stdout, "  ... (truncated)")
					break
				}
			}
			if count >= 20 {
				break
			}
		}
	}

	return 0
}

const defaultGitTimeout = 60 * time.Second

func fetchRecentCommits(root string, limit int) ([]orgdebt.Commit, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultGitTimeout)
	defer cancel()
	return fetchRecentCommitsContext(ctx, root, limit)
}

func fetchRecentCommitsContext(ctx context.Context, root string, limit int) ([]orgdebt.Commit, error) {
	args := []string{
		"-C", root,
		"log",
		fmt.Sprintf("-n%d", limit),
		"--pretty=format:__COMMIT__%H%x00%P%x00%s",
		"--numstat",
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	configureDispatchHelperCommand(cmd)
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	var commits []orgdebt.Commit
	scanner := bufio.NewScanner(bytes.NewReader(out))
	var cur *orgdebt.Commit

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "__COMMIT__") {
			if cur != nil {
				commits = append(commits, *cur)
			}
			parts := strings.Split(strings.TrimPrefix(line, "__COMMIT__"), "\x00")
			sha := ""
			parents := []string{}
			subject := ""
			if len(parts) > 0 {
				sha = parts[0]
			}
			if len(parts) > 1 && strings.TrimSpace(parts[1]) != "" {
				parents = strings.Fields(parts[1])
			}
			if len(parts) > 2 {
				subject = parts[2]
			}
			cur = &orgdebt.Commit{
				SHA:     sha,
				Parents: parents,
				Subject: subject,
			}
			continue
		}

		if cur != nil && strings.TrimSpace(line) != "" {
			fields := strings.Fields(line)
			if len(fields) >= 3 {
				add, _ := strconv.Atoi(fields[0])
				del, _ := strconv.Atoi(fields[1])
				path := fields[2]
				cur.LinesAdded += add
				cur.LinesDeleted += del
				cur.FilesTouched = append(cur.FilesTouched, path)
				if strings.Contains(path, "_test.go") {
					cur.TestLines += add
				}
			}
		}
	}
	if cur != nil {
		commits = append(commits, *cur)
	}

	return commits, nil
}
