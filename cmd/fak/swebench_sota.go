package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/swebenchsota"
)

// runSwebenchSotaSnapshot is the CLI shell for `fak swebench sota-snapshot`. It
// is the Go port of tools/swebench_sota_snapshot.py: extract the official
// leaderboard JSON (from --html locally or a live --url fetch), fold the SOTA
// snapshot, and write JSON (+ optional markdown). Exit codes: 0 ok, 1 write
// failure, 2 usage / fetch / validation failure.
func runSwebenchSotaSnapshot(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("swebench sota-snapshot", flag.ContinueOnError)
	fs.SetOutput(stderr)
	url := fs.String("url", swebenchsota.DefaultURL, "official leaderboard URL")
	htmlPath := fs.String("html", "", "read a saved official leaderboard HTML page instead of fetching")
	timeout := fs.Float64("timeout-seconds", 20, "HTTP fetch timeout in seconds")
	overallGroup := fs.String("overall-group", "Verified", "leaderboard group for the overall SOTA")
	sameScaffoldGroup := fs.String("same-scaffold-group", "bash-only", "leaderboard group for the same-scaffold SOTA")
	focalPattern := fs.String("focal-pattern", `\bGLM-5\b`, "case-insensitive regexp selecting focal rows")
	limit := fs.Int("limit", 10, "top-N window size per group")
	out := fs.String("out", "", "write the snapshot JSON here")
	md := fs.String("md", "", "write the house-style markdown report here")
	asJSON := fs.Bool("json", false, "print the snapshot JSON to stdout (default when --out is empty)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak swebench sota-snapshot: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	source, sourceRef, err := readSwebenchSource(*htmlPath, *url, time.Duration(*timeout*float64(time.Second)))
	if err != nil {
		fmt.Fprintf(stderr, "fak swebench sota-snapshot: %v\n", err)
		return 2
	}

	doc, err := swebenchsota.BuildSnapshot(source, sourceRef, utcNow(), swebenchsota.Options{
		URL:               *url,
		OverallGroup:      *overallGroup,
		SameScaffoldGroup: *sameScaffoldGroup,
		FocalPattern:      *focalPattern,
		Limit:             *limit,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak swebench sota-snapshot: %v\n", err)
		return 2
	}
	if problems := swebenchsota.Check(doc); len(problems) != 0 {
		fmt.Fprintf(stderr, "fak swebench sota-snapshot: %s\n", strings.Join(problems, "; "))
		return 2
	}

	if *out != "" {
		if err := writeSwebenchText(*out, string(jsonIndent(doc))+"\n"); err != nil {
			fmt.Fprintf(stderr, "fak swebench sota-snapshot: write %s: %v\n", *out, err)
			return 1
		}
	}
	if *md != "" {
		if err := writeSwebenchText(*md, swebenchsota.RenderMarkdown(doc)); err != nil {
			fmt.Fprintf(stderr, "fak swebench sota-snapshot: write %s: %v\n", *md, err)
			return 1
		}
	}
	if *asJSON || *out == "" {
		if err := writeIndentedJSON(stdout, doc); err != nil {
			fmt.Fprintf(stderr, "fak swebench sota-snapshot: encode json: %v\n", err)
			return 1
		}
	}

	fmt.Fprintf(stderr, "\n== fak swebench sota-snapshot ==\nsource: %s\n", sourceRef)
	if doc.OverallSOTA.Top != nil {
		fmt.Fprintf(stderr, "overall SOTA      : %v (%v%%)\n", doc.OverallSOTA.Top.Name, doc.OverallSOTA.Top.ResolvedPct)
	}
	if doc.SameScaffoldSOTA.Top != nil {
		fmt.Fprintf(stderr, "same-scaffold SOTA: %v (%v%%)\n", doc.SameScaffoldSOTA.Top.Name, doc.SameScaffoldSOTA.Top.ResolvedPct)
	}
	return 0
}

// readSwebenchSource returns the leaderboard HTML and a source-ref label, from a
// local file when htmlPath is set, otherwise via a live fetch.
func readSwebenchSource(htmlPath, url string, timeout time.Duration) (string, string, error) {
	if htmlPath != "" {
		b, err := os.ReadFile(htmlPath)
		if err != nil {
			return "", "", err
		}
		return string(b), strings.ReplaceAll(htmlPath, "\\", "/"), nil
	}
	src, err := swebenchsota.Fetch(url, timeout)
	if err != nil {
		return "", "", err
	}
	return src, url, nil
}

// writeSwebenchText writes text to path, creating parent directories.
func writeSwebenchText(path, text string) error {
	if dir := dirOf(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, []byte(text), 0o644)
}

// dirOf returns the directory portion of a slash-or-backslash path.
func dirOf(path string) string {
	i := strings.LastIndexAny(path, "/\\")
	if i < 0 {
		return ""
	}
	return path[:i]
}

// utcNow returns the current UTC time as an ISO-8601 Z-suffixed string, matching
// the Python tool's utc_now().
func utcNow() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05.000000Z")
}
