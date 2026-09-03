package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workspin"
)

func cmdWorkspin(argv []string) { os.Exit(runWorkspin(os.Stdout, os.Stderr, argv)) }

func runWorkspin(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("workspin", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", ".", "repository to inspect")
	since := fs.Duration("since", 28*24*time.Hour, "bounded commit history window")
	buckets := fs.Int("buckets", 4, "number of recent buckets")
	bucket := fs.Duration("bucket", 7*24*time.Hour, "bucket width")
	high := fs.Int("high-activity", 3, "observations that make a bucket high activity")
	sustained := fs.Int("sustained", 2, "consecutive high-activity/no-delivery buckets required")
	issues := fs.String("issues", "", "optional JSON array or JSONL issue observations")
	nowText := fs.String("now", "", "RFC3339 end time (default now)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	now := time.Now().UTC()
	if *nowText != "" {
		var err error
		now, err = time.Parse(time.RFC3339, *nowText)
		if err != nil {
			fmt.Fprintf(stderr, "workspin: parse -now: %v\n", err)
			return 2
		}
	}
	observations, err := gitWorkspinObservations(*repo, now.Add(-*since))
	if err != nil {
		fmt.Fprintf(stderr, "workspin: %v\n", err)
		return 2
	}
	if *issues != "" {
		extra, err := readWorkspinIssues(*issues)
		if err != nil {
			fmt.Fprintf(stderr, "workspin: %v\n", err)
			return 2
		}
		observations = append(observations, extra...)
	}
	cfg := workspin.DefaultConfig(now)
	cfg.BucketCount, cfg.BucketWidth, cfg.HighActivity, cfg.Sustained = *buckets, *bucket, *high, *sustained
	report := workspin.Audit(observations, cfg)
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, report); err != nil {
			fmt.Fprintf(stderr, "workspin: encode json: %v\n", err)
			return 2
		}
		return 0
	}
	fmt.Fprintf(stdout, "workspin: %s — %s\n", report.Verdict, strings.Join(report.Reasons, "; "))
	for _, b := range report.Buckets {
		fmt.Fprintf(stdout, "%s activity=%d maintenance=%d delivery=%d\n", b.Start.Format("2006-01-02"), b.Activity, b.Maintenance, b.Delivery)
	}
	return 0
}

func gitWorkspinObservations(repo string, since time.Time) ([]workspin.Observation, error) {
	const sep = "\x1f"
	cmd := exec.Command("git", "-C", repo, "log", "--since="+since.Format(time.RFC3339), "--no-merges", "--numstat", "--format=COMMIT"+sep+"%aI"+sep+"%s"+sep+"%b")
	configureDispatchHelperCommand(cmd)
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git log: %w", err)
	}
	var out []workspin.Observation
	var current *workspin.Observation
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "COMMIT"+sep) {
			if current != nil {
				out = append(out, *current)
			}
			fields := strings.SplitN(strings.TrimPrefix(line, "COMMIT"+sep), sep, 3)
			if len(fields) != 3 {
				current = nil
				continue
			}
			at, parseErr := time.Parse(time.RFC3339, fields[0])
			if parseErr != nil {
				current = nil
				continue
			}
			subject, body := fields[1], fields[2]
			current = &workspin.Observation{At: at, Kind: workspin.Commit, Summary: subject, Witnessed: strings.Contains(body, "(fak ") || strings.Contains(subject, "(fak ")}
			continue
		}
		if current == nil {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 3 {
			add, e1 := strconv.Atoi(fields[0])
			del, e2 := strconv.Atoi(fields[1])
			if e1 == nil && e2 == nil {
				current.Changed += add + del
			}
		}
	}
	if current != nil {
		out = append(out, *current)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func readWorkspinIssues(path string) ([]workspin.Observation, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read issues: %w", err)
	}
	var out []workspin.Observation
	if err := json.Unmarshal(raw, &out); err == nil {
		for i := range out {
			if out[i].Kind == "" {
				out[i].Kind = workspin.Issue
			}
		}
		return out, nil
	}
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	for scanner.Scan() {
		var o workspin.Observation
		if err := json.Unmarshal(scanner.Bytes(), &o); err != nil {
			return nil, fmt.Errorf("parse issues: %w", err)
		}
		if o.Kind == "" {
			o.Kind = workspin.Issue
		}
		out = append(out, o)
	}
	return out, scanner.Err()
}
