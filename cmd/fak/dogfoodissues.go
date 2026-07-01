package main

// fak dogfood-issues — the backlog bridge from the recent-feature dogfood
// scorecard to stable, deduplicated GitHub issues. It reads a dogfood report.json,
// folds out the scorecard ACTION items, and creates or updates exactly one stable
// issue per item (dedup by an HTML-comment marker key). Safe by default: a dry-run
// that prints what it WOULD do and never touches the network; --live is the
// explicit opt-in that fetches existing issues and shells out to `gh`.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

func cmdDogfoodIssues(argv []string) { os.Exit(runDogfoodIssues(os.Stdout, os.Stderr, argv)) }

// runDogfoodIssues is the testable core: it returns the process exit code instead
// of calling os.Exit, and takes its streams explicitly. Exit codes: 0 ok, 1 a live
// gh create/edit failed, 2 usage/parse/IO error.
func runDogfoodIssues(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dogfood-issues", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	limit := fs.Int("limit", 300, "existing issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture/list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	maxReportAge := fs.Duration("max-report-age", dogfoodissues.DefaultMaxReportAge, "stale report threshold before --live is refused (default 24h)")
	allowStaleReport := fs.Bool("allow-stale-report", false, "allow --live even when the selected report is older than --max-report-age")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *maxReportAge < 0 {
		fmt.Fprintln(stderr, "dogfood-issues: --max-report-age must be non-negative")
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak dogfood-issues: give exactly one path to a recent-feature dogfood report.json")
		return 2
	}

	reportPath, err := filepath.Abs(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "dogfood-issues: %v\n", err)
		return 2
	}

	report, err := dogfoodissues.LoadReport(reportPath)
	if err != nil {
		fmt.Fprintf(stderr, "dogfood-issues: %v\n", err)
		return 2
	}
	freshness, err := dogfoodissues.ReportFreshnessForFile(reportPath, time.Now(), *maxReportAge, *allowStaleReport)
	if err != nil {
		fmt.Fprintf(stderr, "dogfood-issues: %v\n", err)
		return 2
	}
	items := dogfoodissues.ExtractActionItems(report, reportPath)

	mode := "dry-run"
	if *live {
		mode = "live"
	}
	if *live && freshness.Stale && !*allowStaleReport {
		result := dogfoodissues.Result{
			Schema:          dogfoodissues.Schema,
			Mode:            mode,
			Report:          reportPath,
			ReportFreshness: &freshness,
			Planned:         []dogfoodissues.PlanRow{},
			Synced:          []dogfoodissues.SyncRow{},
			Refused:         true,
			Error:           "stale_report",
		}
		if *asJSON {
			if err := writeIndentedJSON(stdout, result); err != nil {
				fmt.Fprintf(stderr, "dogfood-issues: encode json: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprintln(stdout, dogfoodissues.Render(result))
		}
		fmt.Fprintf(stderr, "dogfood-issues: %s\n", dogfoodissues.StaleReportMessage(freshness))
		return 2
	}

	var existing []dogfoodissues.Issue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "dogfood-issues: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "dogfood-issues: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = dogfoodissues.FetchExistingIssues(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "dogfood-issues: %v\n", err)
			return 2
		}
	}

	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, dogfoodissues.BuildOptions{
		Live:          *live,
		DedupeChecked: *live || *fetchExisting || *existingJSON != "",
		DedupeCap:     issueSyncScanLimit(*limit),
	})
	result := dogfoodissues.Result{
		Schema:          dogfoodissues.Schema,
		Mode:            mode,
		Report:          reportPath,
		ReportFreshness: &freshness,
		Planned:         plan,
		Synced:          []dogfoodissues.SyncRow{},
		Skipped:         skipped,
	}
	if *live && len(plan) > 0 {
		result.Synced = dogfoodissues.Sync(plan, *repo, []string(labels), nil)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "dogfood-issues: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, dogfoodissues.Render(result))
	}

	if *live {
		for _, row := range result.Synced {
			if !row.OK {
				return 1
			}
		}
	}
	return 0
}

// stringList is a repeatable string flag (e.g. --label a --label b).
type stringList []string

func (l *stringList) String() string {
	if l == nil {
		return ""
	}
	return fmt.Sprintf("%v", []string(*l))
}

func (l *stringList) Set(v string) error {
	*l = append(*l, v)
	return nil
}
