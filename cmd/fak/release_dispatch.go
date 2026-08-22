package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

const releaseDispatchWorkflow = "release-cadence.yml"

type releaseDispatchRunner func(name string, args ...string) (string, error)

var releaseDispatchRun releaseDispatchRunner = runReleaseDispatchCommand

type releaseDispatchResult struct {
	OK             bool     `json:"ok"`
	DryRun         bool     `json:"dry_run"`
	Dispatched     bool     `json:"dispatched"`
	Workflow       string   `json:"workflow"`
	Ref            string   `json:"ref"`
	DecisionOnly   bool     `json:"decision_only"`
	RequireCIGreen bool     `json:"require_ci_green"`
	Force          bool     `json:"force"`
	Command        []string `json:"command"`
	ActionsURL     string   `json:"actions_url,omitempty"`
	RunURL         string   `json:"run_url,omitempty"`
	NextAction     string   `json:"next_action"`
}

func runReleaseDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("release dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	execute := fs.Bool("execute", false, "dispatch the release-cadence workflow (default previews)")
	asJSON := fs.Bool("json", false, "emit a machine-readable result")
	ref := fs.String("ref", "main", "branch or tag containing the workflow to dispatch")
	decisionOnly := fs.Bool("plan-only", false, "run release decision and planning without cutting a release")
	requireCI := fs.Bool("require-ci-green", true, "require the current trunk CI state to be green")
	force := fs.Bool("force", false, "bypass only the substantive-commit floor")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak release dispatch: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	*ref = strings.TrimSpace(*ref)
	if *ref == "" {
		fmt.Fprintln(stderr, "fak release dispatch: --ref must not be empty")
		return 2
	}

	args := []string{"workflow", "run", releaseDispatchWorkflow, "--ref", *ref,
		"-f", fmt.Sprintf("dry_run=%t", *decisionOnly),
		"-f", fmt.Sprintf("require_ci_green=%t", *requireCI),
		"-f", fmt.Sprintf("force=%t", *force),
	}
	result := releaseDispatchResult{
		OK:             true,
		DryRun:         !*execute,
		Dispatched:     false,
		Workflow:       releaseDispatchWorkflow,
		Ref:            *ref,
		DecisionOnly:   *decisionOnly,
		RequireCIGreen: *requireCI,
		Force:          *force,
		Command:        append([]string{"gh"}, args...),
		NextAction:     "rerun with --execute to dispatch the guarded CI release",
	}

	if *execute {
		out, err := releaseDispatchRun("gh", args...)
		if err != nil {
			result.OK = false
			result.DryRun = false
			result.NextAction = "fix GitHub authentication or workflow access, then retry the same command"
			emitReleaseDispatchResult(stdout, result, *asJSON)
			fmt.Fprintf(stderr, "fak release dispatch: %v", err)
			if strings.TrimSpace(out) != "" {
				fmt.Fprintf(stderr, ": %s", strings.TrimSpace(out))
			}
			fmt.Fprintln(stderr)
			return 1
		}
		result.DryRun = false
		result.Dispatched = true
		result.ActionsURL = releaseActionsURL()
		result.RunURL = releaseDispatchedRunURL(*ref)
		if *decisionOnly {
			result.NextAction = "open the run URL and verify the release decision and plan complete"
		} else {
			result.NextAction = "open the run URL and verify the release reaches publication verification"
		}
	}

	emitReleaseDispatchResult(stdout, result, *asJSON)
	return 0
}

func runReleaseDispatchCommand(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Env = os.Environ()
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func releaseDispatchedRunURL(ref string) string {
	out, err := releaseDispatchRun("gh", "run", "list", "--workflow", releaseDispatchWorkflow,
		"--event", "workflow_dispatch", "--branch", ref, "--limit", "1", "--json", "url", "--jq", ".[0].url")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

func releaseActionsURL() string {
	out, err := releaseDispatchRun("gh", "repo", "view", "--json", "url", "--jq", ".url")
	if err != nil || strings.TrimSpace(out) == "" {
		return ""
	}
	return strings.TrimRight(strings.TrimSpace(out), "/") + "/actions/workflows/" + releaseDispatchWorkflow
}

func emitReleaseDispatchResult(w io.Writer, result releaseDispatchResult, asJSON bool) {
	if asJSON {
		_ = json.NewEncoder(w).Encode(result)
		return
	}
	verdict := "PREVIEW"
	if result.Dispatched {
		verdict = "DISPATCHED"
	} else if !result.OK {
		verdict = "ERROR"
	}
	fmt.Fprintf(w, "%s: release cadence on %s (decision_only=%t require_ci_green=%t force=%t)\n", verdict, result.Ref, result.DecisionOnly, result.RequireCIGreen, result.Force)
	if result.RunURL != "" {
		fmt.Fprintf(w, "Run: %s\n", result.RunURL)
	} else if result.ActionsURL != "" {
		fmt.Fprintf(w, "Actions: %s\n", result.ActionsURL)
	}
	fmt.Fprintf(w, "Next: %s\n", result.NextAction)
}
