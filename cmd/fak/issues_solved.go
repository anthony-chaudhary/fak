package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuesolved"
)

var collectIssuesSolvedFunc = issuesolved.Collect

func cmdIssuesSolved(argv []string) {
	os.Exit(runIssuesSolved(os.Stdout, os.Stderr, argv))
}

func runIssuesSolved(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak issues-solved", flag.ContinueOnError)
	fs.SetOutput(stderr)

	hours := fs.Float64("hours", 24, "hours to look back (default: 24)")
	since := fs.String("since", "", "optional duration e.g. 9h, or RFC3339 timestamp")
	repo := fs.String("repo", "anthony-chaudhary/fak", "public repository name (default: anthony-chaudhary/fak)")
	privateRepo := fs.String("private-repo", "anthony-chaudhary/fak-private", "private repository name (default: anthony-chaudhary/fak-private)")
	noPrivate := fs.Bool("no-private", false, "exclude private repository (default: false)")
	publicDir := fs.String("public-dir", ".", "public repo checkout path (default: .)")
	privateDir := fs.String("private-dir", "", "private repo checkout path")
	source := fs.String("source", "auto", "data collection source (choices: auto, github, git)")
	asJSON := fs.Bool("json", false, "emit report as JSON")
	var detailed bool
	fs.BoolVar(&detailed, "detailed", false, "show detailed issue listings")
	fs.BoolVar(&detailed, "list", false, "alias for --detailed")

	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issues-solved: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	if *hours < 0 {
		fmt.Fprintln(stderr, "fak issues-solved: --hours must be non-negative")
		return 2
	}
	if *hours == 0 && *since == "" {
		fmt.Fprintln(stderr, "fak issues-solved: --hours must be positive")
		return 2
	}

	src := strings.ToLower(strings.TrimSpace(*source))
	if src != "auto" && src != "github" && src != "git" {
		fmt.Fprintf(stderr, "fak issues-solved: --source must be auto, github, or git, got %q\n", *source)
		return 2
	}

	var sinceTime time.Time
	now := time.Now()
	var hoursVal float64 = *hours

	if *since != "" {
		hoursExplicit := false
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "hours" {
				hoursExplicit = true
			}
		})
		if hoursExplicit {
			fmt.Fprintln(stderr, "fak issues-solved: cannot specify both --hours and --since")
			return 2
		}

		st, err := parseSince(*since, now)
		if err != nil {
			fmt.Fprintf(stderr, "fak issues-solved: %v\n", err)
			return 2
		}
		sinceTime = st
		hoursVal = 0
	}

	opts := issuesolved.Options{
		Hours:          hoursVal,
		Since:          sinceTime,
		PublicRepo:     strings.TrimSpace(*repo),
		PrivateRepo:    strings.TrimSpace(*privateRepo),
		PublicDir:      strings.TrimSpace(*publicDir),
		PrivateDir:     strings.TrimSpace(*privateDir),
		IncludePrivate: !*noPrivate,
		ExcludePrivate: *noPrivate,
		Source:         src,
		Now:            now,
	}

	rep, err := collectIssuesSolvedFunc(context.Background(), opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak issues-solved: %v\n", err)
		return 1
	}

	var renderErr error
	if *asJSON {
		renderErr = issuesolved.RenderJSON(stdout, rep)
	} else if detailed {
		renderErr = issuesolved.RenderDetailed(stdout, rep)
	} else {
		renderErr = issuesolved.RenderSummary(stdout, rep)
	}
	if renderErr != nil {
		fmt.Fprintf(stderr, "fak issues-solved: %v\n", renderErr)
		return 1
	}

	return 0
}

func parseSince(s string, now time.Time) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, nil
	}
	if d, err := time.ParseDuration(s); err == nil {
		if d < 0 {
			d = -d
		}
		return now.Add(-d), nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		if t.After(now) {
			return time.Time{}, fmt.Errorf("invalid --since %q: timestamp cannot be in the future", s)
		}
		return t, nil
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		if t.After(now) {
			return time.Time{}, fmt.Errorf("invalid --since %q: timestamp cannot be in the future", s)
		}
		return t, nil
	}
	return time.Time{}, fmt.Errorf("invalid --since %q: must be duration (e.g. 9h) or RFC3339 timestamp", s)
}
