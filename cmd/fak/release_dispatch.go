package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const releaseDispatchWorkflow = "release-cadence.yml"

const releaseDispatchEndpoint = "repos/{owner}/{repo}/actions/workflows/" + releaseDispatchWorkflow + "/dispatches"

type releaseDispatchRunner func(name string, args ...string) (string, error)

var (
	releaseDispatchRun   releaseDispatchRunner = runReleaseDispatchCommand
	releaseDispatchNow                         = time.Now
	releaseDispatchSleep                       = time.Sleep
	releaseDispatchPoll                        = 3 * time.Second
)

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
	RunID          int64    `json:"run_id,omitempty"`
	RunURL         string   `json:"run_url,omitempty"`
	Status         string   `json:"status,omitempty"`
	Conclusion     string   `json:"conclusion,omitempty"`
	Verdict        string   `json:"verdict"`
	NextAction     string   `json:"next_action"`
}

type releaseDispatchResponse struct {
	RunID  int64  `json:"workflow_run_id"`
	RunURL string `json:"html_url"`
	APIURL string `json:"run_url"`
}

type releaseDispatchRunState struct {
	RunID      int64  `json:"databaseId"`
	RunURL     string `json:"url"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
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
	wait := fs.Bool("wait", false, "wait for the exact dispatched run's terminal verdict")
	timeout := fs.Duration("timeout", 30*time.Minute, "maximum wait for --wait")
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
	if *wait && !*execute {
		fmt.Fprintln(stderr, "fak release dispatch: --wait requires --execute")
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak release dispatch: --timeout must be positive")
		return 2
	}

	workflowArgs := []string{"workflow", "run", releaseDispatchWorkflow, "--ref", *ref,
		"-f", fmt.Sprintf("dry_run=%t", *decisionOnly),
		"-f", fmt.Sprintf("require_ci_green=%t", *requireCI),
		"-f", fmt.Sprintf("force=%t", *force),
	}
	result := releaseDispatchResult{
		OK:             true,
		DryRun:         !*execute,
		Workflow:       releaseDispatchWorkflow,
		Ref:            *ref,
		DecisionOnly:   *decisionOnly,
		RequireCIGreen: *requireCI,
		Force:          *force,
		Command:        append([]string{"gh"}, workflowArgs...),
		Verdict:        "preview",
		NextAction:     "rerun with --execute to dispatch the guarded CI release",
	}

	if *execute {
		if *wait {
			apiArgs := releaseDispatchAPIArgs(*ref, *decisionOnly, *requireCI, *force)
			result.Command = append([]string{"gh"}, apiArgs...)
			out, err := releaseDispatchRun("gh", apiArgs...)
			if err != nil {
				return failReleaseDispatch(stdout, stderr, result, *asJSON, "github_refused", out, err)
			}
			var dispatched releaseDispatchResponse
			if err := json.Unmarshal([]byte(out), &dispatched); err != nil || dispatched.RunID == 0 || strings.TrimSpace(dispatched.RunURL) == "" {
				if err == nil {
					err = fmt.Errorf("GitHub returned no exact run details")
				}
				return failReleaseDispatch(stdout, stderr, result, *asJSON, "github_refused", out, err)
			}
			result.DryRun = false
			result.Dispatched = true
			result.RunID = dispatched.RunID
			result.RunURL = dispatched.RunURL
			result.ActionsURL = releaseActionsURL()
			if err := waitReleaseDispatch(&result, *timeout); err != nil {
				result.OK = false
				result.NextAction = "inspect the exact run URL, then retry only after resolving the terminal refusal"
				emitReleaseDispatchResult(stdout, result, *asJSON)
				fmt.Fprintf(stderr, "fak release dispatch: %v\n", err)
				return 1
			}
			result.NextAction = "release workflow completed successfully"
		} else {
			out, err := releaseDispatchRun("gh", workflowArgs...)
			if err != nil {
				return failReleaseDispatch(stdout, stderr, result, *asJSON, "github_refused", out, err)
			}
			result.DryRun = false
			result.Dispatched = true
			result.Verdict = "dispatched"
			result.ActionsURL = releaseActionsURL()
			result.RunURL = releaseDispatchedRunURL(*ref)
			if *decisionOnly {
				result.NextAction = "open the run URL and verify the release decision and plan complete"
			} else {
				result.NextAction = "open the run URL and verify the release reaches publication verification"
			}
		}
	}

	emitReleaseDispatchResult(stdout, result, *asJSON)
	return 0
}

func releaseDispatchAPIArgs(ref string, decisionOnly, requireCI, force bool) []string {
	return []string{"api", "--method", "POST", releaseDispatchEndpoint,
		"-f", "ref=" + ref,
		"-F", fmt.Sprintf("inputs[dry_run]=%t", decisionOnly),
		"-F", fmt.Sprintf("inputs[require_ci_green]=%t", requireCI),
		"-F", fmt.Sprintf("inputs[force]=%t", force),
		"-F", "return_run_details=true",
	}
}

func waitReleaseDispatch(result *releaseDispatchResult, timeout time.Duration) error {
	deadline := releaseDispatchNow().Add(timeout)
	for {
		out, err := releaseDispatchRun("gh", "run", "view", strconv.FormatInt(result.RunID, 10), "--json", "databaseId,url,status,conclusion")
		if err != nil {
			result.Verdict = "github_refused"
			return fmt.Errorf("read exact run %d: %w: %s", result.RunID, err, strings.TrimSpace(out))
		}
		var state releaseDispatchRunState
		if err := json.Unmarshal([]byte(out), &state); err != nil || state.RunID != result.RunID {
			result.Verdict = "github_refused"
			return fmt.Errorf("GitHub returned mismatched details for exact run %d", result.RunID)
		}
		result.RunURL = state.RunURL
		result.Status = state.Status
		result.Conclusion = state.Conclusion
		if state.Status == "completed" {
			result.Verdict = releaseConclusionVerdict(state.Conclusion)
			if state.Conclusion == "success" {
				return nil
			}
			return fmt.Errorf("exact run %d completed with conclusion %q", result.RunID, state.Conclusion)
		}
		remaining := deadline.Sub(releaseDispatchNow())
		if remaining <= 0 {
			result.Verdict = "timed_out"
			return fmt.Errorf("timed out waiting %s for exact run %d", timeout, result.RunID)
		}
		if remaining > releaseDispatchPoll {
			remaining = releaseDispatchPoll
		}
		releaseDispatchSleep(remaining)
	}
}

func releaseConclusionVerdict(conclusion string) string {
	switch conclusion {
	case "success":
		return "passed"
	case "cancelled":
		return "cancelled"
	default:
		return "failed"
	}
}

func failReleaseDispatch(stdout, stderr io.Writer, result releaseDispatchResult, asJSON bool, verdict, out string, err error) int {
	result.OK = false
	result.DryRun = false
	result.Verdict = verdict
	result.NextAction = "fix GitHub authentication or workflow access, then retry the same command"
	emitReleaseDispatchResult(stdout, result, asJSON)
	fmt.Fprintf(stderr, "fak release dispatch: %v", err)
	if strings.TrimSpace(out) != "" {
		fmt.Fprintf(stderr, ": %s", strings.TrimSpace(out))
	}
	fmt.Fprintln(stderr)
	return 1
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
	verdict := strings.ToUpper(result.Verdict)
	fmt.Fprintf(w, "%s: release cadence on %s (decision_only=%t require_ci_green=%t force=%t)\n", verdict, result.Ref, result.DecisionOnly, result.RequireCIGreen, result.Force)
	if result.RunID != 0 {
		fmt.Fprintf(w, "Run: %d %s\n", result.RunID, result.RunURL)
	} else if result.RunURL != "" {
		fmt.Fprintf(w, "Run: %s\n", result.RunURL)
	} else if result.ActionsURL != "" {
		fmt.Fprintf(w, "Actions: %s\n", result.ActionsURL)
	}
	if result.Status != "" {
		fmt.Fprintf(w, "Status: %s conclusion=%s\n", result.Status, result.Conclusion)
	}
	fmt.Fprintf(w, "Next: %s\n", result.NextAction)
}
