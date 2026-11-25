// Command docreap ranks stale, orphaned docs and archives them without breaking links.
//
//	docreap rank [--top N] [--recursive] [--json] [--appeal]
//	docreap archive <doc> --reason R --successor <path|none> [--scope S] [--dry-run]
//	docreap check
//
// Exit: 0 ok, 1 check/archive refusal, 2 could not run.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/docreap"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const usage = "usage: docreap rank|archive|check [flags]  (see `docreap <command> -h`)"

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	cmd, rest := args[0], args[1:]
	if cmd != "rank" && cmd != "archive" && cmd != "check" {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	fs := flag.NewFlagSet("docreap "+cmd, flag.ContinueOnError)
	fs.SetOutput(stderr)
	top := 30
	var recursive, asJSON, appeal, dryRun bool
	var reason, successor, scope, todayFlag string
	switch cmd {
	case "rank":
		fs.IntVar(&top, "top", 30, "rows to list")
		fs.BoolVar(&recursive, "recursive", false, "include docs/ subdirectories")
		fs.BoolVar(&asJSON, "json", false, "emit JSON")
		fs.BoolVar(&appeal, "appeal", false, "add the tools/doc_appeal_scorecard.py prose score (one python run per doc)")
		fs.StringVar(&todayFlag, "today", "", "override today's date (YYYY-MM-DD)")
	case "archive":
		fs.StringVar(&reason, "reason", "", "one of: "+strings.Join(docreap.Reasons, ", "))
		fs.StringVar(&successor, "successor", "", "repo-relative replacement path, or 'none'")
		fs.StringVar(&scope, "scope", "", "what the historical record describes")
		fs.BoolVar(&dryRun, "dry-run", false, "print the plan without changing the tree")
		fs.StringVar(&todayFlag, "today", "", "override today's date (YYYY-MM-DD)")
	}
	doc := ""
	if cmd == "archive" && len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
		doc, rest = rest[0], rest[1:]
	}
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	if cmd == "archive" && doc == "" && fs.NArg() == 1 {
		doc = fs.Arg(0)
	} else if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "docreap: unexpected arguments:", strings.Join(fs.Args(), " "))
		return 2
	}
	today := time.Now()
	if todayFlag != "" {
		t, err := time.Parse("2006-01-02", todayFlag)
		if err != nil {
			fmt.Fprintln(stderr, "docreap: --today must be YYYY-MM-DD")
			return 2
		}
		today = t
	}
	root, err := docreap.RepoRoot(".")
	if err != nil {
		fmt.Fprintln(stderr, "could not run:", err)
		return 2
	}

	switch cmd {
	case "check":
		bad, err := docreap.Check(root)
		if err != nil {
			fmt.Fprintln(stderr, "could not run:", err)
			return 2
		}
		for _, rel := range bad {
			fmt.Fprintf(stdout, "ARCHIVE_BANNER_MISSING %s: needs '> **Archived DATE** · reason: R · replacement: X · scope: S'\n", rel)
		}
		if len(bad) > 0 {
			return 1
		}
		return 0
	case "archive":
		if doc == "" || reason == "" || successor == "" {
			fmt.Fprintln(stderr, "docreap archive: need <doc>, --reason and --successor")
			return 2
		}
		if scope == "" {
			last := "its last edit"
			if dates, err := docreap.LastCommitDates(root); err == nil && dates[doc] != "" {
				last = dates[doc]
			}
			scope = "state of the repository as of " + last
		}
		err := docreap.Archive(root, docreap.ArchiveOptions{
			Doc: doc, Reason: reason, Successor: successor, Scope: scope, Today: today, DryRun: dryRun,
		}, stdout)
		if errors.Is(err, docreap.ErrRefused) {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if err != nil {
			fmt.Fprintln(stderr, "could not run:", err)
			return 2
		}
		return 0
	}

	var score docreap.AppealFunc
	if appeal {
		score = pythonAppeal(root)
	}
	rows, err := docreap.Rank(root, recursive, today, score)
	if err != nil {
		fmt.Fprintln(stderr, "could not run:", err)
		return 2
	}
	if top >= 0 && len(rows) > top {
		rows = rows[:top]
	}
	if asJSON {
		if rows == nil {
			rows = []docreap.Row{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rows)
		return 0
	}
	fmt.Fprintf(stdout, "%5s  %-10s  %4s  %6s  %-15s  path — why\n", "score", "last", "refs", "appeal", "action")
	for _, r := range rows {
		a, why := "-", "-"
		if r.Appeal != nil {
			a = fmt.Sprintf("%.0f", *r.Appeal)
		}
		if len(r.Reasons) > 0 {
			why = strings.Join(r.Reasons, ", ")
		}
		fmt.Fprintf(stdout, "%5.1f  %-10s  %4d  %6s  %-15s  %s — %s\n",
			r.Score, r.LastCommit, r.DocRefs+r.CodeRefs, a, r.Action, r.Path, why)
	}
	return 0
}

// pythonAppeal scores a doc with the grandfathered prose scorer. It is an optional
// signal: a doc the scorer cannot read simply ranks without it.
func pythonAppeal(root string) docreap.AppealFunc {
	script := filepath.Join(root, "tools", "doc_appeal_scorecard.py")
	return func(rel, _ string) (float64, bool) {
		for _, py := range []string{"python3", "python"} {
			// The scorer exits 1 on appeal debt and still prints its payload.
			out, _ := windowgate.Command(py, script, "--workspace", root, "--target", rel, "--json").Output()
			var payload struct {
				Doc *struct {
					AppealScore *float64 `json:"appeal_score"`
				} `json:"doc"`
			}
			if json.Unmarshal(out, &payload) == nil && payload.Doc != nil && payload.Doc.AppealScore != nil {
				return *payload.Doc.AppealScore, true
			}
		}
		return 0, false
	}
}
