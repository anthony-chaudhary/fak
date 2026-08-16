package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"text/tabwriter"
	"time"
)

// schedscan is the 10×-observability window onto the fleet's Windows Scheduled
// Tasks — the FleetResumeWatchdog / FakFleetJanitor / User* jobs that run the
// background garden unattended. watchdog_autoheal.go only ever asks "is the task
// Ready?"; it never surfaces WHY a task is dead. The single most common death is a
// LastTaskResult of 0x800710E0 ("The operator or administrator has refused the
// request") — Task Scheduler's way of saying an Interactive-logon task had no
// logged-on session to run in. That code is invisible to Get-ScheduledTask.State
// (the task still reports Ready); you only see it in Get-ScheduledTaskInfo.
//
// schedscan enumerates every fleet task, joins the live State with the last-run
// result + last/next run times + missed-run count, DECODES each result hex into
// its human meaning + a remediation hint, and rolls the fleet up into
// failing/idle/running/disabled buckets — as a human table or a JSON doc a cron
// or watchdog can gate on. The decode/classify/parse core is pure and
// cross-platform; only the live enumeration is Windows-only (and `--from` feeds a
// captured snapshot in on any OS).
//
// The second lie schedscan refuses to believe (#5095): LastTaskResult=0 through a
// `conhost.exe --headless <program> ...` launch shim. conhost exits 0 regardless
// of the wrapped program's exit code (verified live: `conhost --headless cmd /c
// exit 7` returns 0), so for the many fleet tasks that launch through that shim a
// zero result is NOT evidence the last run succeeded — the resume watchdog once
// sat dead for hours on a param-binding failure while reporting result=0. Each
// task's Action is captured, and a conhost-shimmed exit-0 is demoted from "idle"
// to "unverified" with a judge-by-heartbeat-freshness remediation.

const schedScanSchema = "fak.schedscan/1"

// schedScanDefaultFilter matches the fleet's own scheduled tasks by name. The
// fleet installers register under Fak*/Fleet*/User* prefixes (see
// watchdogAutohealServicesForGOOS and tools/register_*.ps1); --all drops the
// filter to inspect every task on the box.
const schedScanDefaultFilter = `(?i)^(fak|fleet|user)`

// schedResultMeaning is the decoded meaning of a Windows Task Scheduler "Last Run
// Result" code. Severity buckets it for the health rollup; Hint carries the
// concrete remediation when the code names a known misconfiguration.
type schedResultMeaning struct {
	Code     uint32 `json:"code"`
	Hex      string `json:"hex"`
	Message  string `json:"message"`
	Severity string `json:"severity"` // ok | running | info | warn | fail
	Hint     string `json:"hint,omitempty"`
}

// schedResultEntry is a table row keyed by the normalized unsigned code.
type schedResultEntry struct {
	Message  string
	Severity string
	Hint     string
}

// schedResultTable holds the well-known Task Scheduler result codes. The messages
// are the exact strings Windows' own FormatMessage returns for each code (verified
// via System.ComponentModel.Win32Exception), so the decode never drifts from what
// an operator sees in taskschd.msc. Anything not listed falls through to the
// generic HRESULT classifier in decodeSchedTaskResult.
var schedResultTable = map[uint32]schedResultEntry{
	0x0:     {"The operation completed successfully.", "ok", ""},
	0x1:     {"Incorrect function — the task's action exited 1 (generic failure).", "fail", "inspect the task's program/script exit path"},
	0x2:     {"The system cannot find the file specified.", "fail", "the task Action's program path is wrong or missing"},
	0xA:     {"The environment is incorrect.", "fail", ""},
	0x41300: {"The task is ready to run at its next scheduled time.", "ok", ""},
	0x41301: {"The task is currently running.", "running", ""},
	0x41302: {"The task is disabled.", "warn", "re-enable with Enable-ScheduledTask"},
	0x41303: {"The task has not yet run.", "info", ""},
	0x41304: {"There are no more runs scheduled for this task.", "warn", "the trigger has expired; re-arm the schedule"},
	0x41305: {"One or more of the properties needed to run this task on a schedule have not been set.", "warn", "the task has no active schedule; re-arm its trigger"},
	0x41306: {"The last run of the task was terminated by the user.", "warn", ""},
	0x41307: {"Either the task has no triggers or the existing triggers are disabled or not set.", "warn", "add or enable a trigger; the task will not fire on its own"},
	0x4131B: {"The task is registered, but not all specified triggers will start the task.", "warn", "one or more triggers are invalid; review the schedule"},
	0x4131C: {"The task is registered, but may fail to start. Batch logon privilege needs to be enabled for the task principal.", "warn",
		"grant the run-as principal 'Log on as a batch job', or migrate the task to S4U"},
	0x41325:    {"The Task Scheduler service has asked the task to run.", "info", ""},
	0x8004131F: {"An instance of this task is already running.", "info", "usually benign — a prior run overlapped this trigger"},
	0x80070002: {"The system cannot find the file specified.", "fail", "the task Action's program path is wrong or missing"},
	0x80070005: {"Access is denied.", "fail", "the run-as principal lacks rights (e.g. 'Log on as a batch job')"},
	0x800710E0: {"The operator or administrator has refused the request.", "fail",
		"the task cannot run in its current logon context — usually an Interactive-logon task with no logged-on session; migrate it to S4U (\"Run whether user is logged on or not\") or grant the principal 'Log on as a batch job'"},
	0xFFFFFFFF: {"Unknown error (0xFFFFFFFF) — the task action returned -1 or the last run did not complete cleanly.", "fail", ""},
}

// normalizeSchedResult folds a raw LastTaskResult into an unsigned 32-bit code.
// Get-ScheduledTaskInfo hands the field back as a signed Int32 in some PowerShell
// builds and an unsigned UInt32 in others, so 0x800710E0 arrives as either
// 2147946720 or -2147020576 (and 0xFFFFFFFF as -1). Masking to 32 bits collapses
// both spellings onto the same key the decode table uses.
func normalizeSchedResult(raw int64) uint32 {
	return uint32(raw & 0xFFFFFFFF)
}

// decodeSchedTaskResult resolves a raw result code to its meaning. Known codes
// come from schedResultTable; unknown codes are classified structurally:
//   - 0 is success.
//   - The SCHED_S_* success facility (0x00041300–0x0004132F) is informational.
//   - Any code with the HRESULT failure bit set (>= 0x80000000) is a failure.
//   - Any OTHER non-zero code is a failure too: Task Scheduler stores the task
//     action's process exit code in this field, and a non-zero exit is a failed
//     run (the 0=success convention every fleet Fak*/Fleet*/User* action follows).
//     Treating it as fail is what keeps the health rollup and --strict from reading
//     a broken task as healthy-idle — the whole reason schedscan exists. (A few
//     tools, e.g. robocopy, use small non-zero codes to mean success; that only
//     matters under --all, never the default fleet filter.)
func decodeSchedTaskResult(raw int64) schedResultMeaning {
	code := normalizeSchedResult(raw)
	hex := fmt.Sprintf("0x%X", code)
	if e, ok := schedResultTable[code]; ok {
		return schedResultMeaning{Code: code, Hex: hex, Message: e.Message, Severity: e.Severity, Hint: e.Hint}
	}
	switch {
	case code == 0:
		return schedResultMeaning{Code: code, Hex: hex, Message: "The operation completed successfully.", Severity: "ok"}
	case code >= 0x41300 && code <= 0x4132F:
		return schedResultMeaning{Code: code, Hex: hex, Message: "Task Scheduler status " + hex + " (informational).", Severity: "info"}
	case code >= 0x80000000:
		return schedResultMeaning{Code: code, Hex: hex, Message: "Failure HRESULT " + hex + " (unmapped).", Severity: "fail",
			Hint: "look up " + hex + " in the Windows error reference"}
	default:
		return schedResultMeaning{Code: code, Hex: hex, Message: "The task action exited with a non-zero result (" + hex + ") — its last run likely failed.", Severity: "fail",
			Hint: "inspect the task action's exit path; a non-zero exit usually means the last run failed"}
	}
}

// schedScanTaskInfo is one row of the Get-ScheduledTask ⋈ Get-ScheduledTaskInfo
// join as emitted by the PowerShell probe (or a --from snapshot). ActionExecute /
// ActionArguments carry the task's first Action so the classifier can tell which
// tasks launch through an exit-code-masking shim; older --from snapshots without
// the fields simply decode them empty (no shim ⇒ no demotion).
type schedScanTaskInfo struct {
	TaskName           string `json:"TaskName"`
	TaskPath           string `json:"TaskPath"`
	State              string `json:"State"`
	LogonType          string `json:"LogonType"`
	RunLevel           string `json:"RunLevel"`
	UserId             string `json:"UserId"`
	LastRunTime        string `json:"LastRunTime"`
	LastTaskResult     int64  `json:"LastTaskResult"`
	NextRunTime        string `json:"NextRunTime"`
	NumberOfMissedRuns int64  `json:"NumberOfMissedRuns"`
	ActionExecute      string `json:"ActionExecute"`
	ActionArguments    string `json:"ActionArguments"`
	// ActionWorkingDirectory is the first Action's working directory. The launcher
	// posture audit (#2174) reports it because "which tree does this task actually
	// run in" is half of whether an out-of-tree script is recoverable at all.
	ActionWorkingDirectory string `json:"ActionWorkingDirectory"`
}

// schedScanTaskReport is the decoded, classified per-task record in the JSON doc.
// ResultMasked marks a row whose action launches through an exit-code-masking
// shim (conhost.exe): its zero LastTaskResult carries no health signal (#5095).
type schedScanTaskReport struct {
	Name         string             `json:"name"`
	Path         string             `json:"path,omitempty"`
	State        string             `json:"state"`
	LogonType    string             `json:"logon_type,omitempty"`
	RunLevel     string             `json:"run_level,omitempty"`
	UserID       string             `json:"user_id,omitempty"`
	LastRun      string             `json:"last_run,omitempty"`
	NextRun      string             `json:"next_run,omitempty"`
	MissedRuns   int64              `json:"missed_runs,omitempty"`
	Action       string             `json:"action,omitempty"`
	Result       schedResultMeaning `json:"result"`
	ResultMasked bool               `json:"result_masked,omitempty"`
	Status       string             `json:"status"`
	Failing      bool               `json:"failing"`
}

// schedActionMasksExit reports whether a task Action's executable is an
// exit-code-masking launch shim — today, conhost.exe: `conhost --headless <prog>`
// exits 0 no matter what the wrapped program returned (verified live on this
// fleet, #5095), so LastTaskResult records conhost's 0, never the script's real
// exit. Tolerates quoting and full paths ("C:\Windows\System32\conhost.exe").
// A NON-zero result through the shim is still meaningful (it is Task Scheduler
// itself failing to launch, e.g. 0x800710E0), so this only ever weakens a zero.
func schedActionMasksExit(execute string) bool {
	return schedExeBase(execute) == "conhost"
}

// applySchedExitMask demotes a healthy-looking row whose exit 0 arrived through a
// masking shim: "idle" (the only bucket exit 0 can produce) becomes "unverified",
// never failing — the fleet launches most tasks through conhost, so latching
// --strict on the shim itself would red the whole fleet forever. The rewritten
// meaning carries the #5095 remediation: judge the task by heartbeat freshness,
// not LastTaskResult. Every other status (failing/running/disabled/degraded)
// passes through untouched — those verdicts do not rest on a trusted zero.
func applySchedExitMask(status string, r schedResultMeaning, masked bool) (string, schedResultMeaning) {
	if !masked || status != "idle" || r.Code != 0 {
		return status, r
	}
	r.Severity = "warn"
	r.Message = "Exit 0 through a conhost.exe launch shim — conhost swallows the wrapped script's exit code, so this result does not prove the last run succeeded (#5095)."
	r.Hint = "judge this task by heartbeat freshness (fleet_status.ps1 HEALTH), not LastTaskResult; conhost --headless masks inner failures as exit 0"
	return "unverified", r
}

// classifySchedTask rolls a task's live State and decoded result into one health
// bucket. An operator-disabled task drops out of the health/strict signal FIRST,
// even if its last recorded run failed: LastTaskResult is stale history the task
// will not re-run, so a deliberately-disabled task must not latch --strict. After
// that a failure result dominates the live State (that is the whole point — a task
// can report State=Ready while its last run was refused with 0x800710E0), then the
// explicit running state, then idle/healthy.
func classifySchedTask(state string, r schedResultMeaning) (status string, failing bool) {
	st := strings.ToLower(strings.TrimSpace(state))
	if st == "disabled" || (r.Severity == "warn" && r.Code == 0x41302) {
		return "disabled", false
	}
	switch r.Severity {
	case "fail":
		return "failing", true
	}
	switch {
	case st == "running" || r.Severity == "running":
		return "running", false
	case r.Severity == "warn":
		return "degraded", false
	case st == "ready" || r.Severity == "ok" || r.Severity == "info":
		return "idle", false
	default:
		return "unknown", false
	}
}

// schedScanDoc is the full fleet rollup.
type schedScanDoc struct {
	Schema       string                `json:"schema"`
	GeneratedUTC string                `json:"generated_utc"`
	OS           string                `json:"os"`
	Source       string                `json:"source"`
	Filter       string                `json:"filter"`
	Count        int                   `json:"count"`
	FailingCount int                   `json:"failing_count"`
	Counts       map[string]int        `json:"counts"`
	Tasks        []schedScanTaskReport `json:"tasks"`
}

// parseSchedTaskJSON parses ConvertTo-Json output into rows. PowerShell unwraps a
// single-element pipeline to a bare object (not a one-element array), and emits
// nothing for an empty pipeline, so tolerate array / object / empty forms.
func parseSchedTaskJSON(raw string) ([]schedScanTaskInfo, error) {
	s := strings.TrimSpace(raw)
	if s == "" || strings.EqualFold(s, "null") {
		return nil, nil
	}
	switch s[0] {
	case '[':
		var rows []schedScanTaskInfo
		if err := json.Unmarshal([]byte(s), &rows); err != nil {
			return nil, err
		}
		return rows, nil
	case '{':
		var one schedScanTaskInfo
		if err := json.Unmarshal([]byte(s), &one); err != nil {
			return nil, err
		}
		return []schedScanTaskInfo{one}, nil
	default:
		return nil, fmt.Errorf("unexpected JSON payload: begins with %q", s[0])
	}
}

// buildSchedScanDoc decodes, classifies, filters and orders the rows. Failing
// tasks sort first (that is what a human scanning the table wants to see), then by
// name, so the output is stable and action-first.
func buildSchedScanDoc(rows []schedScanTaskInfo, filter *regexp.Regexp, source, generatedUTC string) schedScanDoc {
	doc := schedScanDoc{
		Schema:       schedScanSchema,
		GeneratedUTC: generatedUTC,
		OS:           runtime.GOOS,
		Source:       source,
		Counts:       map[string]int{},
		Tasks:        []schedScanTaskReport{},
	}
	if filter != nil {
		doc.Filter = filter.String()
	}
	for _, row := range rows {
		if filter != nil && !filter.MatchString(row.TaskName) {
			continue
		}
		res := decodeSchedTaskResult(row.LastTaskResult)
		status, failing := classifySchedTask(row.State, res)
		masked := schedActionMasksExit(row.ActionExecute)
		status, res = applySchedExitMask(status, res, masked)
		doc.Tasks = append(doc.Tasks, schedScanTaskReport{
			Name:         row.TaskName,
			Path:         strings.TrimSpace(row.TaskPath),
			State:        row.State,
			LogonType:    row.LogonType,
			RunLevel:     row.RunLevel,
			UserID:       row.UserId,
			LastRun:      row.LastRunTime,
			NextRun:      row.NextRunTime,
			MissedRuns:   row.NumberOfMissedRuns,
			Action:       strings.TrimSpace(strings.TrimSpace(row.ActionExecute) + " " + strings.TrimSpace(row.ActionArguments)),
			Result:       res,
			ResultMasked: masked,
			Status:       status,
			Failing:      failing,
		})
		doc.Counts[status]++
		if failing {
			doc.FailingCount++
		}
	}
	doc.Count = len(doc.Tasks)
	sort.SliceStable(doc.Tasks, func(i, j int) bool {
		if doc.Tasks[i].Failing != doc.Tasks[j].Failing {
			return doc.Tasks[i].Failing
		}
		return doc.Tasks[i].Name < doc.Tasks[j].Name
	})
	return doc
}

// schedScanLiveScript is the constant PowerShell probe. It enumerates every task,
// joins each with its Get-ScheduledTaskInfo (State alone hides LastTaskResult), and
// stringifies the two DateTimes to ISO-8601 so ConvertTo-Json can't render them as
// the ambiguous \/Date(...)\/ form. Read-only: no /Create, so it never trips the
// windowgate interactive-task gate. Name filtering happens in Go, keeping this
// script parameter-free (no interpolation, no injection surface).
const schedScanLiveScript = `$ErrorActionPreference='SilentlyContinue'
Get-ScheduledTask | ForEach-Object {
  $i = $_ | Get-ScheduledTaskInfo
  $a = $_.Actions | Select-Object -First 1
  [pscustomobject]@{
    TaskName = $_.TaskName
    TaskPath = $_.TaskPath
    State = [string]$_.State
    LogonType = [string]$_.Principal.LogonType
    RunLevel = [string]$_.Principal.RunLevel
    UserId = [string]$_.Principal.UserId
    LastRunTime = if ($i.LastRunTime) { $i.LastRunTime.ToString('o') } else { '' }
    LastTaskResult = [int64]$i.LastTaskResult
    NextRunTime = if ($i.NextRunTime) { $i.NextRunTime.ToString('o') } else { '' }
    NumberOfMissedRuns = [int64]$i.NumberOfMissedRuns
    ActionExecute = [string]$a.Execute
    ActionArguments = [string]$a.Arguments
    ActionWorkingDirectory = [string]$a.WorkingDirectory
  }
} | ConvertTo-Json -Depth 3`

// schedScanQueryLive runs the PowerShell probe. It reuses watchdogRunCommand so it
// inherits windowgate.ConfigureBackgroundCommand (no popped console window).
func schedScanQueryLive(ctx context.Context, run watchdogCommandRunner) (string, error) {
	return run(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", schedScanLiveScript)
}

func cmdSchedScan(argv []string) {
	os.Exit(runSchedScan(os.Stdout, os.Stderr, argv))
}

// runSchedScan is the testable core. Exit codes mirror the sibling stallscan
// convention so a cron/watchdog gate can separate "a task is failing" from "the
// scan itself broke": 0 clean, 1 runtime error (live query / parse / encode), 2
// usage, 3 a failing task was found under --strict.
func runSchedScan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("schedscan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "schedscan")
	jsonOut := fs.Bool("json", false, "emit the JSON rollup instead of the human table")
	filterExpr := fs.String("filter", schedScanDefaultFilter, "case-insensitive regexp matched against the task name")
	all := fs.Bool("all", false, "scan every scheduled task (ignore --filter)")
	from := fs.String("from", "", "read Get-ScheduledTaskInfo JSON from a file instead of a live query (works off-Windows)")
	failingOnly := fs.Bool("failing-only", false, "show only failing tasks (table rows and the JSON task list)")
	strict := fs.Bool("strict", false, "exit 3 if any task is failing (1=runtime error, 2=usage)")
	launchers := fs.Bool("launchers", false, "report LAUNCHER POSTURE (headless/process-isolated contract, #2174) instead of task health")
	xmlDir := fs.String("xml-dir", "", "audit the versioned task definitions in this directory (implies --launchers; works off-Windows)")
	repoRootDir := fs.String("repo-root", "", "repo root used to decide whether a launched script is in-tree (default: the enclosing checkout)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	filterSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "filter" {
			filterSet = true
		}
	})

	var filter *regexp.Regexp
	if !*all {
		re, err := regexp.Compile(*filterExpr)
		if err != nil {
			fmt.Fprintf(stderr, "fak schedscan: bad --filter regexp: %v\n", err)
			return 2
		}
		filter = re
	}
	root := strings.TrimSpace(*repoRootDir)
	if root == "" {
		root = repoRoot()
	}

	// --xml-dir audits the repo-owned task DEFINITIONS (tools/scheduled-tasks/)
	// rather than the live Task Scheduler, which is what lets the launcher-posture
	// gate run in CI on any OS. A static definition carries no run history, so it
	// feeds the launcher report only — reporting health from a LastTaskResult nobody
	// ever recorded would be exactly the trusted zero schedscan exists to refuse.
	// The Fak*/Fleet*/User* name filter is dropped here unless the caller asked for
	// one explicitly: everything in the capture directory is repo-owned by
	// construction, so prefix-filtering it would silently drop rows from a report
	// whose whole job is to be exhaustive.
	if *xmlDir != "" {
		if !filterSet {
			filter = nil
		}
		rows, err := loadSchedTaskXMLDir(*xmlDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak schedscan: --xml-dir: %v\n", err)
			return 2
		}
		doc := buildSchedLauncherDoc(rows, filter, root, *xmlDir, time.Now().UTC().Format(time.RFC3339), false)
		return emitSchedLauncherDoc(stdout, stderr, doc, *jsonOut, *strict)
	}

	var raw, source string
	var liveErr error
	if *from != "" {
		b, err := os.ReadFile(*from)
		if err != nil {
			fmt.Fprintf(stderr, "fak schedscan: --from: %v\n", err)
			return 2
		}
		raw, source = string(b), *from
	} else {
		if runtime.GOOS != "windows" {
			fmt.Fprintln(stderr, "fak schedscan: a live scan queries the Windows Task Scheduler; run on Windows or pass --from <json>")
			return 2
		}
		ctx, cancel := context.WithTimeout(ctx(), 60*time.Second)
		defer cancel()
		out, err := schedScanQueryLive(ctx, watchdogRunCommand)
		if err != nil && strings.TrimSpace(out) == "" {
			fmt.Fprintf(stderr, "fak schedscan: live query failed: %v\n", err)
			return 1
		}
		// A non-empty stream with a non-nil err is a noisy-but-usable run; keep the
		// err so that if the payload fails to parse we surface the real PowerShell
		// error (which embeds the combined output) instead of the generic parse error.
		raw, source, liveErr = out, "live", err
	}

	rows, err := parseSchedTaskJSON(raw)
	if err != nil {
		if liveErr != nil {
			fmt.Fprintf(stderr, "fak schedscan: live query failed: %v\n", liveErr)
		} else {
			fmt.Fprintf(stderr, "fak schedscan: parse task JSON: %v\n", err)
		}
		return 1
	}

	if *launchers {
		doc := buildSchedLauncherDoc(rows, filter, root, source, time.Now().UTC().Format(time.RFC3339), true)
		return emitSchedLauncherDoc(stdout, stderr, doc, *jsonOut, *strict)
	}

	doc := buildSchedScanDoc(rows, filter, source, time.Now().UTC().Format(time.RFC3339))

	if *jsonOut {
		// Honor --failing-only in JSON too (the table honors it via renderSchedScanTable):
		// prune the task list to failures. FailingCount and Counts stay the full-fleet
		// rollup so the fleet context is not lost.
		if *failingOnly {
			kept := make([]schedScanTaskReport, 0, doc.FailingCount)
			for _, t := range doc.Tasks {
				if t.Failing {
					kept = append(kept, t)
				}
			}
			doc.Tasks = kept
			doc.Count = len(kept)
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(doc); err != nil {
			fmt.Fprintf(stderr, "fak schedscan: encode: %v\n", err)
			return 1
		}
	} else {
		renderSchedScanTable(stdout, doc, *failingOnly)
	}

	if *strict && doc.FailingCount > 0 {
		return 3
	}
	return 0
}

// renderSchedScanTable prints the action-first human view: a header summary, a
// tabwriter grid (failing tasks first), and a remediation hint under each failing
// task so the fix travels with the finding.
func renderSchedScanTable(w io.Writer, doc schedScanDoc, failingOnly bool) {
	fmt.Fprintf(w, "fleet scheduled tasks: %d total, %d failing  (%s)\n", doc.Count, doc.FailingCount, schedScanCountsLine(doc.Counts))
	if doc.Count == 0 {
		fmt.Fprintln(w, "  (no tasks matched — try --all or a wider --filter)")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "STATUS\tNAME\tLOGON\tLAST RESULT\tLAST RUN\tNEXT RUN\tMISS")
	shown := 0
	for _, t := range doc.Tasks {
		if failingOnly && !t.Failing {
			continue
		}
		shown++
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s %s\t%s\t%s\t%d\n",
			strings.ToUpper(t.Status), t.Name, schedScanDash(t.LogonType),
			t.Result.Hex, truncateTableField(t.Result.Message, 40),
			schedScanShortTime(t.LastRun), schedScanShortTime(t.NextRun), t.MissedRuns)
	}
	tw.Flush()
	if shown == 0 {
		fmt.Fprintln(w, "  (no failing tasks)")
		return
	}
	// Remediation hints for the failing (and mask-unverified, #5095) tasks, deduped
	// by hint so a fleet-wide misconfiguration (all 19 tasks refused for the same
	// reason, or all launched through the same conhost shim) prints once.
	seen := map[string]bool{}
	for _, t := range doc.Tasks {
		if (!t.Failing && t.Status != "unverified") || t.Result.Hint == "" || seen[t.Result.Hint] {
			continue
		}
		seen[t.Result.Hint] = true
		fmt.Fprintf(w, "\n  %s (%s): %s\n", t.Result.Hex, t.Result.Message, t.Result.Hint)
	}
}

func schedScanCountsLine(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
	}
	if len(parts) == 0 {
		return "empty"
	}
	return strings.Join(parts, " ")
}

func schedScanDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

// schedScanShortTime trims an ISO-8601 timestamp to minute precision for the table
// (the seconds/zone add width without adding signal in a fleet scan).
func schedScanShortTime(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.Format("2006-01-02 15:04")
	}
	// Non-RFC3339 fallback (only reachable via a hand-crafted --from snapshot): keep
	// the leading date+time, but trim by rune so a multibyte rune straddling the cut
	// is never split into invalid UTF-8. For the ASCII ISO strings the live probe
	// emits this is byte-identical to the old s[:16].
	if len(s) >= 16 {
		r := []rune(s)
		if len(r) > 16 {
			r = r[:16]
		}
		return strings.Replace(string(r), "T", " ", 1)
	}
	return s
}
