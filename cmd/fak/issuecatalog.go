package main

// fak issue-catalog — the durable bridge from a reviewable performance-enablement
// gap catalog to stable, deduplicated GitHub issues. It reads a catalog JSON (an
// embedded default, or --catalog FILE), reviews each row against the shared issue
// contract, and creates OR updates exactly one stable issue per row (dedup by an
// HTML-comment marker key). Safe by default: a dry-run that prints what it WOULD do
// and never touches the network; --live is the explicit opt-in that fetches
// existing issues and shells out to `gh`.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecatalog"
)

func cmdIssueCatalog(argv []string) { os.Exit(runIssueCatalog(os.Stdout, os.Stderr, argv)) }

// runIssueCatalog is the testable core: it returns the process exit code instead of
// calling os.Exit, and takes its streams explicitly. Exit codes: 0 ok, 1 a live gh
// create/edit failed, 2 usage/parse/IO error.
func runIssueCatalog(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue-catalog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	catalogPath := fs.String("catalog", "", "catalog JSON file; default is the embedded perf-enablement catalog")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	limit := fs.Int("limit", 1500, "existing-issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	milestoneFilter := fs.String("milestone", "", "only rows whose milestone contains this substring (case-insensitive)")
	lensFilter := fs.String("lens", "", "only rows whose lens equals this (default-off|small-blocker|not-advertised|dogfood|benchmark)")
	keyPrefix := fs.String("key-prefix", "", "only rows whose key starts with this prefix (e.g. perf/kv-cache/)")
	maxCreate := fs.Int("max-create", 0, "cap the number of CREATE actions this run (0 = no cap); updates are never capped — for waved rollout")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	var rows []issuecatalog.Row
	var srcName string
	var err error
	if *catalogPath != "" {
		rows, err = issuecatalog.LoadCatalog(*catalogPath)
		srcName = *catalogPath
	} else {
		rows, err = issuecatalog.DefaultRows()
		srcName = issuecatalog.DefaultCatalogPath + " (embedded)"
	}
	if err != nil {
		fmt.Fprintf(stderr, "issue-catalog: %v\n", err)
		return 2
	}

	rows = filterRows(rows, *milestoneFilter, *lensFilter, *keyPrefix)
	scanLimit := issueSyncScanLimit(*limit)

	var existing []issuecatalog.Issue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "issue-catalog: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "issue-catalog: --existing-json must contain a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = issuecatalog.FetchExistingIssues(*repo, scanLimit)
		if err != nil {
			fmt.Fprintf(stderr, "issue-catalog: %v\n", err)
			return 2
		}
	}

	dedupeChecked := *existingJSON != "" || *fetchExisting || *live
	plan, skipped := issuecatalog.BuildPlan(rows, existing, issuecatalog.Options{
		Live:          *live,
		DedupeChecked: dedupeChecked,
		DedupeCap:     scanLimit,
	})

	plan, capped := capCreates(plan, *maxCreate)

	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := issuecatalog.Result{
		Schema:  issuecatalog.Schema,
		Mode:    mode,
		Catalog: srcName,
		Total:   len(rows),
		Planned: plan,
		Synced:  []issuecatalog.SyncRow{},
		Skipped: skipped,
	}
	if *live && len(plan) > 0 {
		result.Synced = issuecatalog.Sync(plan, *repo, nil)
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "issue-catalog: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, issuecatalog.Render(result))
		if capped > 0 {
			fmt.Fprintf(stdout, "  (--max-create %d: %d create(s) held back this wave)\n", *maxCreate, capped)
		}
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

func issueSyncScanLimit(limit int) int {
	if limit > 0 {
		return limit
	}
	return 100
}

func filterRows(rows []issuecatalog.Row, milestone, lens, keyPrefix string) []issuecatalog.Row {
	milestone = strings.ToLower(strings.TrimSpace(milestone))
	lens = strings.ToLower(strings.TrimSpace(lens))
	keyPrefix = strings.TrimSpace(keyPrefix)
	if milestone == "" && lens == "" && keyPrefix == "" {
		return rows
	}
	out := make([]issuecatalog.Row, 0, len(rows))
	for _, r := range rows {
		if milestone != "" && !strings.Contains(strings.ToLower(r.Milestone), milestone) {
			continue
		}
		if lens != "" && strings.ToLower(strings.TrimSpace(r.Lens)) != lens {
			continue
		}
		if keyPrefix != "" && !strings.HasPrefix(r.Key, keyPrefix) {
			continue
		}
		out = append(out, r)
	}
	return out
}

// capCreates keeps every update but holds back create actions beyond max (0 = no
// cap), so a --live run can roll out in verifiable waves. Returns the trimmed plan
// and how many creates were held back.
func capCreates(plan []issuecatalog.PlanRow, max int) ([]issuecatalog.PlanRow, int) {
	if max <= 0 {
		return plan, 0
	}
	out := make([]issuecatalog.PlanRow, 0, len(plan))
	creates, held := 0, 0
	for _, row := range plan {
		if row.Action == "create" {
			if creates >= max {
				held++
				continue
			}
			creates++
		}
		out = append(out, row)
	}
	return out, held
}
