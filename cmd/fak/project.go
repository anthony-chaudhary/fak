package main

// fak project -- the ProjectsV2 board control-pane fold. It reshapes the board that
// .github/workflows/project-board-sync.yml writes into (and cmd/fak/dispatch_project_fields.go
// reads for dispatch ranking) into the SAME schema/ok/verdict/finding/next_action
// envelope `fak milestone report` uses, so the board becomes an operator-visible
// dimension instead of a write-only sync target. It is READ/REPORT only: it never
// writes to the board and never changes dispatch ranking.
//
//	fak project report                       # fold + render the board snapshot (live)
//	fak project report --json                # the machine-readable envelope
//	fak project report --check               # advisory gate (exit 1 only if unmeasured)
//	fak project report --from-items items.json   # fold a fixture hermetically (no gh)
//
// Live reads are gated on FAK_DISPATCH_PROJECT_NUMBER (the same knob the dispatch
// reader uses) and need a gh token with read:project scope; absent either, the fold is a
// visible UNMEASURED verdict, never a silent green.

import (
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

	"github.com/anthony-chaudhary/fak/internal/projectreport"
)

func cmdProject(argv []string) {
	dispatchSubcommands("project", "report", argv,
		subcommand{"report", runProjectReport},
	)
}

func runProjectReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak project report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromItems := fs.String("from-items", "", "fold the board from this JSON items file instead of reading live (- for stdin). A JSON array of {issue,status,generation,priority}.")
	asJSON := fs.Bool("json", false, "emit the machine-readable JSON envelope")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the board failed to MEASURE (a drifted-but-measured board still exits 0)")
	projectNumber := fs.Int("project-number", 0, "ProjectsV2 number to read live (default: $"+dispatchProjectNumberEnv+")")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak project report: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := repoRoot()
	now := time.Now().UTC()
	opts := projectreport.FoldOpts{
		Commit:      projectHeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        now.Format("2006-01-02"),
	}

	var report projectreport.Report
	if *fromItems != "" {
		items, err := loadProjectItems(*fromItems)
		if err != nil {
			fmt.Fprintf(stderr, "fak project report: %v\n", err)
			return 2
		}
		report = projectreport.Fold(items, opts)
	} else {
		number := *projectNumber
		if number <= 0 {
			number, _ = strconv.Atoi(strings.TrimSpace(os.Getenv(dispatchProjectNumberEnv)))
		}
		if number <= 0 {
			report = projectreport.Unmeasured(dispatchProjectNumberEnv+" is unset — no board to read", opts)
		} else if items, ok := fetchProjectReportItems(root, number); ok {
			report = projectreport.Fold(items, opts)
		} else {
			report = projectreport.Unmeasured("could not read ProjectsV2 board (gh read:project scope or project number)", opts)
		}
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, report)
	} else {
		fmt.Fprintln(stdout, projectreport.Render(report))
	}

	if *check {
		// Advisory MEASURE gate: only a board that failed to measure is a non-zero exit.
		// A measured board with drift (ACTION) is a real signal, so it exits 0 here.
		if !report.Measured {
			return 1
		}
		return 0
	}
	if report.OK {
		return 0
	}
	return 1
}

// loadProjectItems reads a JSON array of board items from a file (or stdin for "-").
func loadProjectItems(path string) ([]projectreport.Item, error) {
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = readFromFile(path)
	}
	if err != nil {
		return nil, err
	}
	var items []projectreport.Item
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("parse --from-items payload: %w", err)
	}
	return items, nil
}

// fetchProjectReportItems reads the board's items with their Status / Generation /
// Priority single-selects, reusing the same `gh api graphql` path as
// dispatchFetchProjectFieldsGH but also capturing the Generation field the report folds.
// Returns (items, true) on a clean read, (nil, false) on any failure — the caller folds
// a false into a visible UNMEASURED verdict.
func fetchProjectReportItems(root string, number int) ([]projectreport.Item, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	query := `query($owner:String!,$number:Int!){repositoryOwner(login:$owner){projectV2(number:$number){items(first:100){nodes{content{... on Issue{number}} fieldValues(first:20){nodes{... on ProjectV2ItemFieldSingleSelectValue{name field{... on ProjectV2SingleSelectField{name}}}}}}}}}}`
	owner := "anthony-chaudhary"
	cmd := exec.CommandContext(ctx, "gh", "api", "graphql", "-f", "query="+query, "-F", "owner="+owner, "-F", "number="+strconv.Itoa(number))
	cmd.Dir = root
	raw, err := cmd.Output()
	if err != nil {
		return nil, false
	}
	items, ok := parseProjectReportItems(raw)
	return items, ok
}

// parseProjectReportItems folds the GraphQL payload into report items. Split from the
// fetch so it is unit-testable without gh.
func parseProjectReportItems(raw []byte) ([]projectreport.Item, bool) {
	var doc struct {
		Data struct {
			RepositoryOwner struct {
				Project struct {
					Items struct {
						Nodes []struct {
							Content struct {
								Number int `json:"number"`
							} `json:"content"`
							FieldValues struct {
								Nodes []struct {
									Name  string `json:"name"`
									Field struct {
										Name string `json:"name"`
									} `json:"field"`
								} `json:"nodes"`
							} `json:"fieldValues"`
						} `json:"nodes"`
					} `json:"items"`
				} `json:"projectV2"`
			} `json:"repositoryOwner"`
		} `json:"data"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return nil, false
	}
	var items []projectreport.Item
	for _, node := range doc.Data.RepositoryOwner.Project.Items.Nodes {
		if node.Content.Number <= 0 {
			continue
		}
		it := projectreport.Item{Issue: node.Content.Number}
		for _, v := range node.FieldValues.Nodes {
			switch strings.ToLower(strings.TrimSpace(v.Field.Name)) {
			case "status":
				it.Status = v.Name
			case "generation":
				it.Generation = v.Name
			case "priority":
				it.Priority = v.Name
			}
		}
		items = append(items, it)
	}
	return items, true
}

// projectHeadCommit returns the short HEAD sha for the provenance stamp, "" on failure.
func projectHeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
