package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskdecision"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdTask(argv []string) { os.Exit(runTask(os.Stdout, os.Stderr, argv)) }

func runTask(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		taskUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "handoff":
		return runTaskHandoff(stdout, stderr, argv[1:])
	case "decision":
		return runTaskDecision(stdout, stderr, argv[1:])
	case "sample":
		return runTaskSample(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		taskUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak task: unknown subcommand %q\n", argv[0])
		taskUsage(stderr)
		return 2
	}
}

type taskDecisionResult struct {
	Schema  string               `json:"schema"`
	Mode    string               `json:"mode"`
	File    string               `json:"file"`
	Entry   taskdecision.Entry   `json:"entry,omitempty"`
	Entries []taskdecision.Entry `json:"entries,omitempty"`
}

func runTaskDecision(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak task decision: subcommand required (append | list)")
		return 2
	}
	switch argv[0] {
	case "append":
		return runTaskDecisionAppend(stdout, stderr, argv[1:])
	case "list":
		return runTaskDecisionList(stdout, stderr, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak task decision: unknown subcommand %q (append | list)\n", argv[0])
		return 2
	}
}

func runTaskDecisionAppend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("task decision append", flag.ContinueOnError)
	fs.SetOutput(stderr)
	taskID := fs.String("task", "", "task id for the scoped decision log")
	logPath := fs.String("log", "", "decision log JSONL path (default: .fak/task-decisions/<task>.jsonl)")
	decision := fs.String("decision", "", "decision made")
	rationale := fs.String("rationale", "", "why the decision was made")
	evidenceRef := fs.String("evidence-ref", "", "witness/reference for the decision")
	unixNano := fs.Int64("time-unix-nano", 0, "optional event timestamp for deterministic fixtures")
	asJSON := fs.Bool("json", false, "emit JSON")
	var openThreads stringList
	fs.Var(&openThreads, "open-thread", "open thread/risk to reload after compaction (repeatable)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak task decision append: unexpected positional arguments")
		return 2
	}
	entry := taskdecision.Normalize(taskdecision.Entry{
		TaskID:      *taskID,
		Decision:    *decision,
		Rationale:   *rationale,
		EvidenceRef: *evidenceRef,
		OpenThreads: []string(openThreads),
		UnixNano:    *unixNano,
	})
	if err := taskdecision.Validate(entry); err != nil {
		fmt.Fprintf(stderr, "fak task decision append: %v\n", err)
		return 2
	}
	path := taskDecisionLogPath(*logPath, entry.TaskID)
	if err := taskdecision.Append(path, entry); err != nil {
		fmt.Fprintf(stderr, "fak task decision append: %v\n", err)
		return 1
	}
	result := taskDecisionResult{Schema: "fak.task-decision-result.v1", Mode: "append", File: path, Entry: entry}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak task decision append")
	}
	fmt.Fprintf(stdout, "task decision appended: task=%s file=%s decision=%q\n", entry.TaskID, path, entry.Decision)
	return 0
}

func runTaskDecisionList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("task decision list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	taskID := fs.String("task", "", "task id to load")
	logPath := fs.String("log", "", "decision log JSONL path (default: .fak/task-decisions/<task>.jsonl)")
	limit := fs.Int("limit", taskdecision.DefaultReloadLimit, "maximum newest entries to load into reset context")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak task decision list: unexpected positional arguments")
		return 2
	}
	if strings.TrimSpace(*taskID) == "" {
		fmt.Fprintln(stderr, "fak task decision list: --task is required")
		return 2
	}
	path := taskDecisionLogPath(*logPath, *taskID)
	entries, err := taskdecision.Load(path, *taskID, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak task decision list: %v\n", err)
		return 1
	}
	result := taskDecisionResult{Schema: "fak.task-decision-result.v1", Mode: "list", File: path, Entries: entries}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak task decision list")
	}
	fmt.Fprintf(stdout, "task decisions: task=%s entries=%d file=%s\n", strings.TrimSpace(*taskID), len(entries), path)
	if text := taskdecision.Render(entries); text != "" {
		fmt.Fprintln(stdout, text)
	}
	return 0
}

func taskDecisionLogPath(explicit, taskID string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	root := resolveRoot("")
	if root == "" {
		root = "."
	}
	return taskdecision.DefaultPath(root, taskID)
}

type taskHandoffResult struct {
	Schema  string                        `json:"schema"`
	Mode    string                        `json:"mode"`
	File    string                        `json:"file"`
	Review  taskmgr.HandoffReview         `json:"review"`
	Planned []taskmgr.HandoffIssuePlanRow `json:"planned"`
	Synced  []taskHandoffSyncRow          `json:"synced,omitempty"`
}

type taskHandoffSyncRow struct {
	Key    string `json:"key"`
	Action string `json:"action"`
	OK     bool   `json:"ok"`
	Stdout string `json:"stdout,omitempty"`
	Stderr string `json:"stderr,omitempty"`
}

type taskHandoffRunner func(args []string) (stdout, stderr string, ok bool)

func runTaskHandoff(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("task handoff", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "task handoff JSON file")
	repo := fs.String("repo", "", "owner/repo for gh; default is current repo")
	limit := fs.Int("limit", 300, "existing issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture/list of existing gh issues for dry-run tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to classify create vs update")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable review/plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *file == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak task handoff: pass exactly --file HANDOFF.json")
		return 2
	}

	path, err := filepath.Abs(*file)
	if err != nil {
		fmt.Fprintf(stderr, "fak task handoff: %v\n", err)
		return 2
	}
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak task handoff: %v\n", err)
		return 2
	}
	var handoff taskmgr.Handoff
	if err := json.Unmarshal(b, &handoff); err != nil {
		fmt.Fprintf(stderr, "fak task handoff: parse %s: %v\n", path, err)
		return 2
	}

	review := taskmgr.ReviewHandoffWithOptions(handoff, taskmgr.HandoffReviewOptions{
		StrictScope:   true,
		Live:          *live,
		DedupeChecked: *live || *fetchExisting || *existingJSON != "",
		DedupeCap:     taskHandoffIssueScanLimit(*limit),
	})
	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := taskHandoffResult{
		Schema: "fak.task-handoff-result.v1",
		Mode:   mode,
		File:   path,
		Review: review,
	}

	if review.OK && len(handoff.NextSteps) > 0 {
		existing, err := loadTaskHandoffIssues(*existingJSON, *fetchExisting || *live, *repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak task handoff: %v\n", err)
			return 2
		}
		result.Planned = taskmgr.BuildHandoffIssuePlan(handoff, existing)
		if *live && len(result.Planned) > 0 {
			result.Synced = syncTaskHandoffPlan(result.Planned, *repo, []string(labels), nil)
		}
	}

	if *asJSON {
		if err := writeIndentedJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak task handoff: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderTaskHandoff(result))
	}

	if !review.OK {
		return 3
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

func taskHandoffIssueScanLimit(limit int) int {
	return issueSyncScanLimit(limit)
}

func loadTaskHandoffIssues(existingJSON string, fetch bool, repo string, limit int) ([]taskmgr.HandoffIssue, error) {
	if existingJSON != "" {
		b, err := os.ReadFile(existingJSON)
		if err != nil {
			return nil, err
		}
		var existing []taskmgr.HandoffIssue
		if err := json.Unmarshal(b, &existing); err != nil {
			return nil, fmt.Errorf("--existing-json must contain a JSON list: %w", err)
		}
		return existing, nil
	}
	if !fetch {
		return nil, nil
	}
	if limit <= 0 {
		limit = 300
	}
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit), "--json", "number,title,body,state,url"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := runTaskHandoffGH(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var existing []taskmgr.HandoffIssue
	if err := json.Unmarshal([]byte(stdout), &existing); err != nil {
		return nil, err
	}
	return existing, nil
}

func syncTaskHandoffPlan(plan []taskmgr.HandoffIssuePlanRow, repo string, labels []string, runner taskHandoffRunner) []taskHandoffSyncRow {
	run := runner
	if run == nil {
		run = runTaskHandoffGH
	}
	results := make([]taskHandoffSyncRow, 0, len(plan))
	for _, row := range plan {
		args := taskHandoffGHArgs(row, repo, labels)
		stdout, stderr, ok := run(args)
		results = append(results, taskHandoffSyncRow{
			Key:    row.Key,
			Action: row.Action,
			OK:     ok,
			Stdout: strings.TrimSpace(stdout),
			Stderr: strings.TrimSpace(stderr),
		})
	}
	return results
}

func taskHandoffGHArgs(row taskmgr.HandoffIssuePlanRow, repo string, labels []string) []string {
	var args []string
	if row.Action == "update" {
		num := ""
		if row.Number != nil {
			num = strconv.Itoa(*row.Number)
		}
		args = []string{"issue", "edit", num, "--title", row.Title, "--body", row.Body}
	} else {
		args = []string{"issue", "create", "--title", row.Title, "--body", row.Body}
		for _, label := range mergeTaskHandoffLabels(row.Labels, labels) {
			args = append(args, "--label", label)
		}
	}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	return args
}

func runTaskHandoffGH(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

func mergeTaskHandoffLabels(a, b []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, group := range [][]string{a, b} {
		for _, label := range group {
			label = strings.TrimSpace(label)
			if label == "" || seen[label] {
				continue
			}
			seen[label] = true
			out = append(out, label)
		}
	}
	return out
}

func renderTaskHandoff(r taskHandoffResult) string {
	lines := []string{
		fmt.Sprintf("task-handoff: %s  verdict=%s  ok=%t  issue_count=%d", r.Mode, r.Review.Verdict, r.Review.OK, r.Review.IssueCount),
		fmt.Sprintf("  file: %s", r.File),
	}
	if r.Review.TaskID != "" {
		lines = append(lines, fmt.Sprintf("  task: %s", r.Review.TaskID))
	}
	for _, reason := range r.Review.Reasons {
		lines = append(lines, "  refuses: "+reason)
	}
	for _, row := range r.Planned {
		target := "new issue"
		if row.Number != nil {
			target = "#" + strconv.Itoa(*row.Number)
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s: %s (key=%s)", row.Action, target, row.Title, row.Key))
	}
	if r.Mode == "dry-run" && len(r.Planned) > 0 {
		lines = append(lines, "  dry-run: pass --live to create/update issues with gh")
	}
	return strings.Join(lines, "\n")
}

func runTaskSample(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("task sample", flag.ContinueOnError)
	fs.SetOutput(stderr)
	taskID := fs.String("task", "process", "task id to stamp in the sample")
	title := fs.String("title", "process sample", "task title")
	stepID := fs.String("step", "snapshot", "step id to stamp in the sample")
	concept := fs.String("concept", "observe", "concept bucket for the step")
	done := fs.Float64("done", 0, "completed work units")
	total := fs.Float64("total", 0, "total work units, if known")
	unit := fs.String("unit", "", "work unit label")
	asJSON := fs.Bool("json", false, "emit the full JSON snapshot")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *done < 0 || *total < 0 {
		fmt.Fprintln(stderr, "fak task sample: --done and --total must be non-negative")
		return 2
	}

	m := taskmgr.NewManager()
	task, err := m.StartTask(taskmgr.TaskSpec{TaskID: *taskID, Title: *title, Total: *total, Unit: *unit})
	if err != nil {
		fmt.Fprintf(stderr, "fak task sample: %v\n", err)
		return 2
	}
	step, err := task.StartStep(taskmgr.StepSpec{StepID: *stepID, Title: "sample process resources", Concept: *concept, Total: *total, Unit: *unit})
	if err != nil {
		fmt.Fprintf(stderr, "fak task sample: %v\n", err)
		return 2
	}
	if *done > 0 || *total > 0 || *unit != "" {
		if err := task.SetProgress(*done, *total, *unit); err != nil {
			fmt.Fprintf(stderr, "fak task sample: %v\n", err)
			return 2
		}
		if err := step.SetProgress(*done, *total, *unit); err != nil {
			fmt.Fprintf(stderr, "fak task sample: %v\n", err)
			return 2
		}
	}

	snap := m.Snapshot()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, snap, "fak task sample")
	}
	renderTaskSample(stdout, snap)
	return 0
}

func renderTaskSample(w io.Writer, snap taskmgr.Snapshot) {
	fmt.Fprintf(w, "process pid=%d uptime=%s cpu=%s heap=%s sys=%s goroutines=%d\n",
		snap.ProcessID,
		secondsText(snap.UptimeSeconds),
		secondsText(snap.Resource.CPUSeconds),
		bytesText(snap.Resource.HeapAllocBytes),
		bytesText(snap.Resource.SysBytes),
		snap.Resource.Goroutines,
	)
	for _, task := range snap.Tasks {
		fmt.Fprintf(w, "task %-18s %-8s liveness=%-8s runtime=%s progress=%s",
			task.TaskID, task.State, livenessText(task.LivenessClass), secondsText(task.RuntimeSeconds), progressText(task.Progress))
		if task.ETASeconds != nil {
			fmt.Fprintf(w, " eta=%s", secondsText(*task.ETASeconds))
		}
		fmt.Fprintln(w)
		for _, step := range task.Steps {
			concept := step.Concept
			if concept == "" {
				concept = "-"
			}
			fmt.Fprintf(w, "  step %-17s concept=%-10s %-8s liveness=%-8s runtime=%s cpu_delta=%s progress=%s",
				step.StepID, concept, step.State, livenessText(step.LivenessClass),
				secondsText(step.RuntimeSeconds), secondsText(step.Resource.Delta.CPUSeconds), progressText(step.Progress))
			if step.ETASeconds != nil {
				fmt.Fprintf(w, " eta=%s", secondsText(*step.ETASeconds))
			}
			fmt.Fprintln(w)
		}
	}
	if len(snap.Concepts) > 0 {
		fmt.Fprintln(w, "concepts:")
		for _, c := range snap.Concepts {
			fmt.Fprintf(w, "  %-12s steps=%d running=%d runtime=%s cpu=%s\n",
				c.Concept, c.Steps, c.RunningSteps, secondsText(c.RuntimeSeconds), secondsText(c.CPUSeconds))
		}
	}
}

func livenessText(class taskmgr.LivenessClass) string {
	if class == "" {
		return "-"
	}
	return string(class)
}

func progressText(p taskmgr.Progress) string {
	if p.Total <= 0 {
		if p.Done > 0 {
			if strings.TrimSpace(p.Unit) == "" {
				return trimFloat(p.Done)
			}
			return trimFloat(p.Done) + " " + strings.TrimSpace(p.Unit)
		}
		return "unknown"
	}
	base := trimFloat(p.Done) + "/" + trimFloat(p.Total)
	if p.Unit != "" {
		base += " " + p.Unit
	}
	if p.Percent != nil {
		base += " (" + trimFloat(*p.Percent) + "%)"
	}
	return base
}

func secondsText(v float64) string {
	return trimFloat(v) + "s"
}

func bytesText(v uint64) string {
	const unit = 1024
	if v < unit {
		return strconv.FormatUint(v, 10) + "B"
	}
	value := float64(v)
	for _, suffix := range []string{"KiB", "MiB", "GiB", "TiB"} {
		value /= unit
		if value < unit {
			return trimFloat(value) + suffix
		}
	}
	return trimFloat(value) + "PiB"
}

func trimFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', 3, 64)
}

func taskUsage(w io.Writer) {
	fmt.Fprint(w, `fak task - process-local task-manager snapshot

  fak task sample [--json] [--task ID] [--title T] [--step ID]
                  [--concept NAME] [--done N --total N --unit UNIT]
  fak task handoff --file HANDOFF.json [--json] [--existing-json FILE]
                   [--fetch-existing] [--live] [--repo owner/repo]
                   [--label LABEL ...]
  fak task decision append --task ID --decision STR --rationale STR --evidence-ref REF
                           [--open-thread STR ...] [--log FILE] [--json]
  fak task decision list   --task ID [--limit N] [--log FILE] [--json]

The sample command emits the same snapshot shape a long-running fak process can embed:
process resources, task/step wall time, concept runtime, progress, and ETA when known.

The handoff command gates a completed task's next-step push: the JSON must carry a
StateDone task with a VerifiedDone witness, a current-state summary, and either one
or two concrete next steps or a no-next-step reason. Dry-run prints stable GitHub
issue create/update decisions; --live is required to call gh.

The decision command writes the task-scoped, append-only reasoning log that
session resets reload as bounded carryover: decision, rationale, evidence_ref, and
open_threads.
`)
}
