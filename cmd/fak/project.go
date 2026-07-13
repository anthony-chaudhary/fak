package main

// fak project -- the ProjectsV2 board control-pane surface. It reshapes the board that
// .github/workflows/project-board-sync.yml writes into (and cmd/fak/dispatch_project_fields.go
// reads for dispatch ranking) into the SAME schema/ok/verdict/finding/next_action
// envelope `fak milestone report` uses, and posts it to Slack through the SAME durable
// scoreboard outbox `fak milestone post` / `fak steering` use — so the board becomes an
// operator-visible dimension instead of a write-only sync target. It is READ/REPORT
// only: it never writes to the board and never changes dispatch ranking.
//
//	fak project report                       # fold + render the board snapshot (live)
//	fak project report --json                # the machine-readable envelope
//	fak project report --check               # advisory gate (exit 1 only if unmeasured)
//	fak project report --from-items items.json   # fold a fixture hermetically (no gh)
//	fak project report --append-history      # trend a dated row into docs/project/history.jsonl
//	fak project post --dry-run               # render the exact Slack card; do not post
//	fak project post                         # post the card to #project (durable outbox)
//	fak project selfcheck                    # deterministic fold + ledger/trend proof (no gh, no key)
//
// The durable ledger + per-tick trend mirror `fak milestone report`: a scheduled tick
// (cadence.yml) appends one row per week so the board distribution is trended, not just
// snapshotted.
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
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectreport"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

func cmdProject(argv []string) {
	dispatchSubcommands("project", "report | post | selfcheck", argv,
		subcommand{"report", runProjectReport},
		subcommand{"post", runProjectPost},
		subcommand{"selfcheck", runProjectSelfcheck},
	)
}

// runProjectSelfcheck runs the deterministic source-level proof for the project fold —
// no gh, no key, no fixtures — the project twin of `fak milestone selfcheck`.
func runProjectSelfcheck(stdout, stderr io.Writer, argv []string) int {
	return runReportSelfcheck(stdout, stderr, argv, "project", projectreport.Selfcheck,
		"SELFCHECK OK -- the project fold is a mirror: a fully-classified board is OK, an "+
			"unclassified item is ACTION drift, an unreachable board folds to a visible UNMEASURED, "+
			"and the durable ledger row + per-tick trend round-trip.")
}

func runProjectReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak project report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromItems := fs.String("from-items", "", "fold the board from this JSON items file instead of reading live (- for stdin). A JSON array of {issue,status,generation,priority}.")
	asJSON := fs.Bool("json", false, "emit the machine-readable JSON envelope")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the board failed to MEASURE (a drifted-but-measured board still exits 0)")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger ("+projectreport.DefaultLedgerRel+") and trend it")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+projectreport.DefaultLedgerRel+")")
	projectNumber := fs.Int("project-number", 0, "ProjectsV2 number to read live (default: $"+dispatchProjectNumberEnv+")")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak project report: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	report, err := loadProjectReport(*fromItems, "", *projectNumber)
	if err != nil {
		fmt.Fprintf(stderr, "fak project report: %v\n", err)
		return 2
	}

	// Attach the per-tick trend vs the durable ledger (read-only), and -- only under
	// --append-history -- durably append this tick so the trend accrues across weeks.
	ledgerPath := projectLedgerPath(*ledger)
	report = attachProjectTrend(report, ledgerPath)
	if *appendHistory {
		if err := appendLedgerFile(ledgerPath, projectreport.RowFromReport(report), trendreport.AppendLedgerLine); err != nil {
			fmt.Fprintf(stderr, "fak project report: append ledger: %v\n", err)
			return 1
		}
		if !*asJSON && !*check {
			rel, relErr := filepath.Rel(repoRoot(), ledgerPath)
			if relErr != nil || rel == "" {
				rel = ledgerPath
			}
			fmt.Fprintf(stdout, "appended project row -> %s\n", filepath.ToSlash(rel))
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

// runProjectPost folds the board (live, from a --from-items fixture, or from a pre-rolled
// --report-json), renders the card, and posts it to the #project channel through the same
// durable scoreboard outbox tail every other feeder uses -- the project twin of
// `fak milestone post`.
func runProjectPost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak project post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fromItems := fs.String("from-items", "", "fold the board from this JSON items file instead of reading live (- for stdin)")
	reportJSON := fs.String("report-json", "", "post a pre-rolled projectreport.Report JSON from this file (- for stdin) instead of folding live")
	projectNumber := fs.Int("project-number", 0, "ProjectsV2 number to read live (default: $"+dispatchProjectNumberEnv+")")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_PROJECT_CHANNEL)")
	token := fs.String("token", "", "override bot token (default: $FAK_PROJECT_TOKEN, then $FAK_SCOREBOARD_TOKEN)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}

	report, err := loadProjectReport(*fromItems, *reportJSON, *projectNumber)
	if err != nil {
		fmt.Fprintf(stderr, "fak project post: %v\n", err)
		return 2
	}
	// Trend the card against the durable ledger too (a pre-rolled --report-json that
	// already carries a trend is left untouched by attachProjectTrend).
	report = attachProjectTrend(report, projectLedgerPath(""))

	src := *source
	if src == "" {
		src = defaultSource()
	}
	card := projectCard(report, src)
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           card,
		channel:        *channel,
		token:          *token,
		dryRun:         *dryRun,
		label:          "fak project post",
		chanEnv:        "FAK_PROJECT_CHANNEL",
		resolveChannel: resolveProjectChannel,
		resolveToken:   resolveProjectToken,
	})
}

// projectLedgerPath resolves the durable project ledger path: an explicit override,
// else <repo root>/docs/project/history.jsonl (the committed trend ledger).
func projectLedgerPath(override string) string {
	if override != "" {
		return override
	}
	return filepath.Join(repoRoot(), filepath.FromSlash(projectreport.DefaultLedgerRel))
}

// attachProjectTrend folds the per-tick trend vs the durable ledger onto the report
// (read-only). A report that already carries a trend — e.g. a pre-rolled --report-json
// appended by a prior tick — is returned untouched, so the trend is never recomputed
// against a stale ledger.
func attachProjectTrend(report projectreport.Report, ledgerPath string) projectreport.Report {
	if report.Trend != nil {
		return report
	}
	prior := readLedgerFile(ledgerPath, projectreport.ParseLedger)
	return report.WithTrend(projectreport.TrendVsLast(projectreport.RowFromReport(report), prior))
}

// loadProjectReport builds the report from exactly one source, in precedence order:
// a pre-rolled --report-json, a --from-items fixture, else a live board read gated on the
// project number. A missing project number or an unreadable board folds to UNMEASURED.
func loadProjectReport(fromItems, reportJSON string, projectNumber int) (projectreport.Report, error) {
	now := time.Now().UTC()
	opts := projectreport.FoldOpts{
		Commit:      projectHeadCommit(repoRoot()),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        now.Format("2006-01-02"),
	}
	if reportJSON != "" {
		return loadPreRolledProjectReport(reportJSON)
	}
	if fromItems != "" {
		items, err := loadProjectItems(fromItems)
		if err != nil {
			return projectreport.Report{}, err
		}
		return projectreport.Fold(items, opts), nil
	}
	number := projectNumber
	if number <= 0 {
		number, _ = strconv.Atoi(strings.TrimSpace(os.Getenv(dispatchProjectNumberEnv)))
	}
	if number <= 0 {
		return projectreport.Unmeasured(dispatchProjectNumberEnv+" is unset — no board to read", opts), nil
	}
	items, ok := fetchProjectReportItems(repoRoot(), number)
	if !ok {
		return projectreport.Unmeasured("could not read ProjectsV2 board (gh read:project scope or project number)", opts), nil
	}
	return projectreport.Fold(items, opts), nil
}

// loadProjectItems reads a JSON array of board items from a file (or stdin for "-").
func loadProjectItems(path string) ([]projectreport.Item, error) {
	raw, err := readProjectPath(path)
	if err != nil {
		return nil, err
	}
	var items []projectreport.Item
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("parse --from-items payload: %w", err)
	}
	return items, nil
}

// loadPreRolledProjectReport reads a folded projectreport.Report JSON (or stdin for "-").
func loadPreRolledProjectReport(path string) (projectreport.Report, error) {
	raw, err := readProjectPath(path)
	if err != nil {
		return projectreport.Report{}, err
	}
	var report projectreport.Report
	if err := json.Unmarshal(raw, &report); err != nil {
		return projectreport.Report{}, fmt.Errorf("parse --report-json payload: %w", err)
	}
	return report, nil
}

func readProjectPath(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return readFromFile(path)
}

// projectCard folds the report into a scoreboard.Update — the same card type
// `fak steering` posts — so `fak project post` renders through the proven Slack
// block/text machinery with zero new rendering surface. The unclassified count is the
// headline debt; the field distributions ride the Lines.
func projectCard(r projectreport.Report, source string) scoreboard.Update {
	up := scoreboard.Update{
		Title:    "project board",
		Verdict:  r.Verdict,
		Detail:   r.Finding,
		NextStep: r.NextAction,
		Source:   source,
	}
	if !r.Measured {
		return up
	}
	up.DebtKey = "unclassified"
	up.Debt = strconv.Itoa(len(r.Unclassified))
	up.Lines = []string{fmt.Sprintf("total %d item(s)", r.Total)}
	if line := projectDistLine(r.ByStatus, nil); line != "" {
		up.Lines = append(up.Lines, "status "+line)
	}
	if line := projectDistLine(r.ByGeneration, []string{"now", "next", "second-next", "future"}); line != "" {
		up.Lines = append(up.Lines, "horizon "+line)
	}
	if line := projectDistLine(r.ByPriority, nil); line != "" {
		up.Lines = append(up.Lines, "priority "+line)
	}
	if r.Trend != nil {
		up.Lines = append(up.Lines, "trend "+r.Trend.Summary)
	}
	return up
}

// projectDistLine renders a "k n · k n" distribution for the card Lines. When order is
// non-nil those keys lead; the remainder sort by descending count then key, with the
// "(unset)" bucket last.
func projectDistLine(m map[string]int, order []string) string {
	if len(m) == 0 {
		return ""
	}
	const unset = "(unset)"
	seen := map[string]bool{}
	var keys []string
	for _, k := range order {
		if _, ok := m[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}
	var rest []string
	for k := range m {
		if !seen[k] && k != unset {
			rest = append(rest, k)
		}
	}
	sort.Slice(rest, func(i, j int) bool {
		if m[rest[i]] != m[rest[j]] {
			return m[rest[i]] > m[rest[j]]
		}
		return rest[i] < rest[j]
	})
	keys = append(keys, rest...)
	if _, ok := m[unset]; ok {
		keys = append(keys, unset)
	}
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s %d", k, m[k]))
	}
	return strings.Join(parts, " · ")
}

// resolveProjectChannel: --channel wins (handled by slackPostTail), else FAK_PROJECT_CHANNEL.
// It deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL (the scoreboard CLI's own
// default target), so the project surface never misroutes to #scoreboard.
func resolveProjectChannel() string {
	return strings.TrimSpace(os.Getenv("FAK_PROJECT_CHANNEL"))
}

// resolveProjectToken: --token wins, else FAK_PROJECT_TOKEN, else the shared
// FAK_SCOREBOARD_TOKEN (the same transport every feeder shares).
func resolveProjectToken() string {
	if v := strings.TrimSpace(os.Getenv("FAK_PROJECT_TOKEN")); v != "" {
		return v
	}
	return strings.TrimSpace(os.Getenv("FAK_SCOREBOARD_TOKEN"))
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
	configureDispatchSpawn(cmd)
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

// projectHeadCommit returns the HEAD sha for the provenance stamp, "" on failure.
func projectHeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = root
	configureDispatchSpawn(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
