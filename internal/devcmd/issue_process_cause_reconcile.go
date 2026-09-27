package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const processCauseRepairLabel = "needs-process-cause"

type issueProcessCauseEvent struct {
	Action string `json:"action"`
	Issue  struct {
		Number int    `json:"number"`
		Body   string `json:"body"`
		State  string `json:"state"`
		Labels []struct {
			Name string `json:"name"`
		} `json:"labels"`
	} `json:"issue"`
	Repository struct {
		FullName string `json:"full_name"`
	} `json:"repository"`
}

type issueProcessCauseResult struct {
	OK       bool       `json:"ok"`
	Valid    bool       `json:"valid"`
	DryRun   bool       `json:"dry_run"`
	Issue    int        `json:"issue"`
	Repo     string     `json:"repo"`
	Primary  string     `json:"primary,omitempty"`
	Detail   string     `json:"detail,omitempty"`
	Errors   []string   `json:"errors,omitempty"`
	Commands [][]string `json:"commands"`
}

type issueProcessCauseRunner func(args []string) (stdout, stderr string, ok bool)

type issueProcessCauseLiveIssue struct {
	Body   string
	State  string
	Labels []struct {
		Name string
	}
}

func runIssueProcessCauseReconcile(stdout, stderr io.Writer, argv []string) int {
	return runIssueProcessCauseReconcileWith(stdout, stderr, argv, nil)
}

func runIssueProcessCauseReconcileWith(stdout, stderr io.Writer, argv []string, runner issueProcessCauseRunner) int {
	fs := flag.NewFlagSet("fak-dev issue reconcile-process-cause", flag.ContinueOnError)
	fs.SetOutput(stderr)
	eventFile := fs.String("event-file", "", "GitHub issues event JSON")
	dryRun := fs.Bool("dry-run", false, "print the planned GitHub mutations")
	jsonOut := fs.Bool("json", false, "emit a machine-readable result")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if strings.TrimSpace(*eventFile) == "" {
		fmt.Fprintln(stderr, "fak-dev issue reconcile-process-cause: --event-file is required")
		return 2
	}
	raw, err := os.ReadFile(*eventFile)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue reconcile-process-cause: read event: %v\n", err)
		return 2
	}
	var event issueProcessCauseEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue reconcile-process-cause: parse event: %v\n", err)
		return 2
	}
	if event.Issue.Number <= 0 || strings.TrimSpace(event.Repository.FullName) == "" {
		fmt.Fprintln(stderr, "fak-dev issue reconcile-process-cause: event must include issue.number and repository.full_name")
		return 2
	}

	body, state := event.Issue.Body, event.Issue.State
	labels := make([]string, 0, len(event.Issue.Labels))
	for _, label := range event.Issue.Labels {
		labels = append(labels, strings.TrimSpace(label.Name))
	}
	if !*dryRun {
		if runner == nil {
			runner = runTaskHandoffGH
		}
		viewArgs := []string{"issue", "view", strconv.Itoa(event.Issue.Number), "--repo", event.Repository.FullName, "--json", "body,state,labels"}
		liveJSON, errText, ok := runner(viewArgs)
		if !ok {
			fmt.Fprintf(stderr, "fak-dev issue reconcile-process-cause: gh %s failed: %s\n", strings.Join(viewArgs, " "), strings.TrimSpace(errText))
			return 1
		}
		var live issueProcessCauseLiveIssue
		if err := json.Unmarshal([]byte(liveJSON), &live); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue reconcile-process-cause: parse current issue: %v\n", err)
			return 1
		}
		body, state = live.Body, live.State
		labels = labels[:0]
		for _, label := range live.Labels {
			labels = append(labels, strings.TrimSpace(label.Name))
		}
	}

	readout := issuepolicy.AssessProcessCause(body)
	commands := planIssueProcessCauseCommands(event.Issue.Number, event.Repository.FullName, state, labels, readout)
	result := issueProcessCauseResult{
		OK: readout.Valid, Valid: readout.Valid, DryRun: *dryRun,
		Issue: event.Issue.Number, Repo: event.Repository.FullName,
		Primary: readout.Primary, Detail: readout.Detail,
		Errors: readout.Errors, Commands: commands,
	}

	if !*dryRun {
		for _, args := range commands {
			_, errText, ok := runner(args)
			if !ok {
				fmt.Fprintf(stderr, "fak-dev issue reconcile-process-cause: gh %s failed: %s\n", strings.Join(args, " "), strings.TrimSpace(errText))
				return 1
			}
		}
	}
	if *jsonOut {
		if code := writeJSON(stdout, result); code != 0 {
			return code
		}
	} else if readout.Valid {
		fmt.Fprintf(stdout, "issue #%d process cause reconciled: %s\n", event.Issue.Number, issuepolicy.ProcessCauseLabel(readout.Primary))
	} else {
		fmt.Fprintf(stderr, "issue #%d process cause rejected: %s\n", event.Issue.Number, strings.Join(readout.Errors, "; "))
	}
	if !readout.Valid {
		return 3
	}
	return 0
}

func planIssueProcessCauseCommands(number int, repo, state string, labels []string, readout issuepolicy.ProcessCauseReadout) [][]string {
	issueNumber := strconv.Itoa(number)
	labelSet := make(map[string]bool, len(labels))
	for _, label := range labels {
		labelSet[label] = true
	}
	var processLabels []string
	for _, label := range labels {
		if strings.HasPrefix(strings.ToLower(label), issuepolicy.ProcessCauseLabelPrefix) {
			processLabels = append(processLabels, label)
		}
	}
	sort.Strings(processLabels)

	if readout.Valid {
		want := issuepolicy.ProcessCauseLabel(readout.Primary)
		commands := [][]string{{"label", "create", want, "--repo", repo, "--force", "--color", "0E8A16", "--description", "Declared development-process cause"}}
		edit := []string{"issue", "edit", issueNumber, "--repo", repo}
		if !labelSet[want] {
			edit = append(edit, "--add-label", want)
		}
		for _, label := range processLabels {
			if label != want {
				edit = append(edit, "--remove-label", label)
			}
		}
		if labelSet[processCauseRepairLabel] {
			edit = append(edit, "--remove-label", processCauseRepairLabel)
		}
		if len(edit) > 5 {
			commands = append(commands, edit)
		}
		if labelSet[processCauseRepairLabel] && strings.EqualFold(strings.TrimSpace(state), "closed") {
			commands = append(commands, []string{"issue", "reopen", issueNumber, "--repo", repo, "--comment", "Process cause repaired; reopening the issue for normal triage."})
		}
		return commands
	}

	commands := [][]string{{"label", "create", processCauseRepairLabel, "--repo", repo, "--force", "--color", "D93F0B", "--description", "Issue is closed until its process-cause declaration is repaired"}}
	edit := []string{"issue", "edit", issueNumber, "--repo", repo, "--add-label", processCauseRepairLabel}
	for _, label := range processLabels {
		edit = append(edit, "--remove-label", label)
	}
	commands = append(commands, edit)
	if !labelSet[processCauseRepairLabel] {
		message := "Process-cause validation failed: " + strings.Join(readout.Errors, "; ") + ". Edit the issue to include exactly one `Process cause: <value>` declaration. Allowed values: concurrency, infrastructure-lag, model-failure, harness-failure, scoping-failure, verification-gap, handoff-failure, other, unknown, none. Concurrency also requires exactly one `Process cause detail: <value>` using shared-state, lease-contention, integration-order, resource-contention, or ownership-overlap. A valid edit will remove this hold and reopen the issue."
		commands = append(commands, []string{"issue", "comment", issueNumber, "--repo", repo, "--body", message})
	}
	if !strings.EqualFold(strings.TrimSpace(state), "closed") {
		commands = append(commands, []string{"issue", "close", issueNumber, "--repo", repo, "--reason", "not planned"})
	}
	return commands
}
