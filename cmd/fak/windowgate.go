package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const windowgateSchema = "fak-windowgate/1"

type windowgatePayload struct {
	Schema      string             `json:"schema"`
	OK          bool               `json:"ok"`
	Verdict     string             `json:"verdict"`
	Finding     string             `json:"finding"`
	Reason      string             `json:"reason"`
	NextAction  string             `json:"next_action"`
	Workspace   string             `json:"workspace"`
	Counts      map[string]int     `json:"counts"`
	Suppression map[string]int     `json:"suppression,omitempty"`
	Violations  []string           `json:"violations"`
	Watchlist   []string           `json:"watchlist,omitempty"`
	Tools       map[string]int     `json:"watchlist_tools,omitempty"`
	Files       map[string]int     `json:"watchlist_files,omitempty"`
	Dirs        map[string]int     `json:"watchlist_dirs,omitempty"`
	LiveTasks   *liveTaskPayload   `json:"live_tasks,omitempty"`
	Windows     *liveWindowPayload `json:"visible_windows,omitempty"`
}

type liveTaskPayload struct {
	OK         bool     `json:"ok"`
	Scanned    int      `json:"scanned"`
	Violations []string `json:"violations,omitempty"`
	Watchlist  []string `json:"watchlist,omitempty"`
}

type liveWindowPayload struct {
	OK         bool                              `json:"ok"`
	Scanned    int                               `json:"scanned"`
	Violations []string                          `json:"violations,omitempty"`
	Watchlist  []string                          `json:"watchlist,omitempty"`
	Findings   []windowgate.VisibleWindowFinding `json:"findings,omitempty"`
	Categories map[string]int                    `json:"categories,omitempty"`
}

func cmdWindowgate(argv []string) { os.Exit(runWindowgate(os.Stdout, os.Stderr, argv)) }

func runWindowgate(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && (argv[0] == "scan" || argv[0] == "report") {
		argv = argv[1:]
	}
	fs := flag.NewFlagSet("windowgate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	failCandidates := fs.Bool("fail-on-candidates", false, "also exit non-zero when advisory console-tool candidates remain")
	liveTasks := fs.Bool("live-tasks", false, "also audit already-installed Windows Scheduled Tasks")
	visibleWindows := fs.Bool("visible-windows", false, "also audit currently visible top-level windows")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak windowgate: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := strings.TrimSpace(*workspace)
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	rep, err := windowgate.ScanTree(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak windowgate: %v\n", err)
		return 1
	}
	payload := buildWindowgatePayload(root, rep, *failCandidates)
	if *liveTasks {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		live, err := scanLiveScheduledTasks(ctx)
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "fak windowgate: live tasks: %v\n", err)
			return 1
		}
		attachLiveTaskPayload(&payload, live, *failCandidates)
	}
	if *visibleWindows {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		visible, err := scanVisibleWindows(ctx)
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "fak windowgate: visible windows: %v\n", err)
			return 1
		}
		attachVisibleWindowPayload(&payload, visible, *failCandidates)
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak windowgate: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderWindowgate(payload))
	}
	if !payload.OK {
		return 1
	}
	return 0
}

func scanLiveScheduledTasks(ctx context.Context) (windowgate.LiveTaskReport, error) {
	if runtime.GOOS != "windows" {
		return windowgate.LiveTaskReport{}, nil
	}
	const ps = `$ErrorActionPreference='SilentlyContinue'
Get-ScheduledTask | ForEach-Object {
  $task = $_
  foreach ($action in $task.Actions) {
    [pscustomobject]@{
      task_path = "$($task.TaskPath)"
      task_name = "$($task.TaskName)"
      state = "$($task.State)"
      logon_type = "$($task.Principal.LogonType)"
      user = "$($task.Principal.UserId)"
      execute = "$($action.Execute)"
      arguments = "$($action.Arguments)"
    }
  }
} | ConvertTo-Json -Depth 4`
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", ps)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return windowgate.LiveTaskReport{}, fmt.Errorf("read scheduled tasks: %w", err)
	}
	tasks, err := windowgate.ParseLiveScheduledTasks(out)
	if err != nil {
		return windowgate.LiveTaskReport{}, err
	}
	return windowgate.ClassifyLiveScheduledTasks(tasks), nil
}

func scanVisibleWindows(ctx context.Context) (windowgate.VisibleWindowReport, error) {
	if runtime.GOOS != "windows" {
		return windowgate.VisibleWindowReport{}, nil
	}
	const ps = `$ErrorActionPreference='SilentlyContinue'
$procs = @{}
Get-CimInstance Win32_Process | ForEach-Object { $procs[[int]$_.ProcessId] = $_ }
Get-Process | Where-Object { $_.MainWindowHandle -ne 0 -and -not [string]::IsNullOrWhiteSpace($_.MainWindowTitle) } | ForEach-Object {
  $cim = $procs[[int]$_.Id]
  $cmd = ""
  $ppid = 0
  $pname = ""
  $pcmd = ""
  $gpid = 0
  $gname = ""
  $gcmd = ""
  if ($null -ne $cim) {
    $cmd = "$($cim.CommandLine)"
    $ppid = [int]$cim.ParentProcessId
    if ($procs.ContainsKey($ppid)) {
      $parent = $procs[$ppid]
      $pname = "$($parent.Name)"
      $pcmd = "$($parent.CommandLine)"
      $gpid = [int]$parent.ParentProcessId
      if ($procs.ContainsKey($gpid)) {
        $grand = $procs[$gpid]
        $gname = "$($grand.Name)"
        $gcmd = "$($grand.CommandLine)"
      }
    }
  }
  [pscustomobject]@{
    pid = [int]$_.Id
    name = "$($_.ProcessName)"
    title = "$($_.MainWindowTitle)"
    path = "$($_.Path)"
    command_line = "$cmd"
    parent_pid = $ppid
    parent_name = "$pname"
    parent_command_line = "$pcmd"
    grandparent_pid = $gpid
    grandparent_name = "$gname"
    grandparent_command_line = "$gcmd"
  }
} | ConvertTo-Json -Depth 4`
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", ps)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return windowgate.VisibleWindowReport{}, fmt.Errorf("read visible windows: %w", err)
	}
	windows, err := windowgate.ParseVisibleWindows(out)
	if err != nil {
		return windowgate.VisibleWindowReport{}, err
	}
	return windowgate.ClassifyVisibleWindows(windows), nil
}

func buildWindowgatePayload(root string, rep windowgate.Report, failCandidates bool) windowgatePayload {
	violations := append([]string{}, rep.PSInstallers...)
	violations = append(violations, rep.PSStartProcesses...)
	violations = append(violations, rep.PySpawns...)
	violations = append(violations, rep.GoExecs...)
	watchlist := append([]string{}, rep.PyCandidates...)
	watchlist = append(watchlist, rep.GoCandidates...)
	counts := map[string]int{
		"ps_installers":       len(rep.PSInstallers),
		"ps_start_processes":  len(rep.PSStartProcesses),
		"py_spawns":           len(rep.PySpawns),
		"py_watchlist":        len(rep.PyCandidates),
		"go_execs":            len(rep.GoExecs),
		"go_watchlist":        len(rep.GoCandidates),
		"py_explicit_modules": len(rep.PyExplicitModules),
		"py_default_modules":  len(rep.PyDefaultModules),
	}
	p := windowgatePayload{
		Schema:      windowgateSchema,
		OK:          len(violations) == 0,
		Verdict:     "OK",
		Finding:     "no_desktop_popup_clear",
		Reason:      "hard-ratcheted scheduled-task, Python, and Go background helper paths are window-suppressed",
		NextAction:  "keep new background subprocesses on windowgate.ConfigureBackgroundCommand or the Python no_window_creationflags helper",
		Workspace:   root,
		Counts:      counts,
		Suppression: suppressionCounts(rep),
		Violations:  violations,
		Watchlist:   watchlist,
		Tools:       watchlistToolCounts(watchlist),
		Files:       watchlistFileCounts(watchlist),
		Dirs:        watchlistDirCounts(watchlist),
	}
	if len(violations) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_regression"
		p.Reason = fmt.Sprintf("%d hard popup regression(s): background helpers can still flash visible console windows", len(violations))
		p.NextAction = "make scheduled tasks off-desktop/headless, add Python creationflags, or call windowgate.ConfigureBackgroundCommand before running Go helper commands"
		return p
	}
	if len(watchlist) > 0 {
		p.Finding = "no_desktop_popup_watchlist"
		p.Reason = fmt.Sprintf("hard popup gate is clean; %d console-tool launch(es) remain on the advisory watchlist", len(watchlist))
		p.NextAction = "review the watchlist and either classify each launch as foreground/interactive or route helper calls through windowgate.ConfigureBackgroundCommand"
		if failCandidates {
			p.OK = false
			p.Verdict = "ACTION"
		}
	}
	return p
}

func attachLiveTaskPayload(p *windowgatePayload, live windowgate.LiveTaskReport, failWatchlist bool) {
	p.LiveTasks = &liveTaskPayload{
		OK:         live.OK(),
		Scanned:    live.Scanned,
		Violations: append([]string{}, live.Violations...),
		Watchlist:  append([]string{}, live.Watchlist...),
	}
	if len(live.Violations) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_live_task_regression"
		p.Reason = fmt.Sprintf("%d live Scheduled Task action(s) can still flash visible console windows", len(live.Violations))
		p.NextAction = "re-register or disable the named task with an off-desktop principal, conhost --headless, pythonw.exe, or -WindowStyle Hidden"
		return
	}
	if failWatchlist && len(live.Watchlist) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_live_task_watchlist"
		p.Reason = fmt.Sprintf("live Scheduled Task hard gate is clean; %d interactive hidden/headless task(s) remain on the review watchlist", len(live.Watchlist))
		p.NextAction = "review the live task watchlist and keep child subprocesses suppressed or move the task off-desktop"
	}
}

func attachVisibleWindowPayload(p *windowgatePayload, visible windowgate.VisibleWindowReport, failWatchlist bool) {
	p.Windows = &liveWindowPayload{
		OK:         visible.OK(),
		Scanned:    visible.Scanned,
		Violations: append([]string{}, visible.Violations...),
		Watchlist:  append([]string{}, visible.Watchlist...),
		Findings:   append([]windowgate.VisibleWindowFinding{}, visible.Findings...),
		Categories: visibleWindowCategoryCounts(visible.Findings),
	}
	if len(visible.Violations) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_visible_window_regression"
		p.Reason = fmt.Sprintf("%d visible automation window(s) are currently on the desktop", len(visible.Violations))
		p.NextAction = "stop or relaunch the named process through a hidden/headless path, then trace its launcher back into windowgate"
		return
	}
	if failWatchlist && len(visible.Watchlist) > 0 {
		p.OK = false
		p.Verdict = "ACTION"
		p.Finding = "no_desktop_popup_visible_window_watchlist"
		p.Reason = fmt.Sprintf("visible-window hard gate is clean; %d visible console/browser automation window(s) remain on the review watchlist", len(visible.Watchlist))
		p.NextAction = "review the visible-window watchlist and close, move off-screen, or classify attended windows"
	}
}

func renderWindowgate(p windowgatePayload) string {
	var b strings.Builder
	status := "OK"
	if !p.OK {
		status = "ACTION"
	}
	fmt.Fprintf(&b, "windowgate: %s (%s)\n", status, p.Finding)
	fmt.Fprintf(&b, "workspace: %s\n", p.Workspace)
	fmt.Fprintf(&b, "hard: ps=%d py=%d go=%d  watchlist: py=%d go=%d\n",
		p.Counts["ps_installers"]+p.Counts["ps_start_processes"], p.Counts["py_spawns"], p.Counts["go_execs"], p.Counts["py_watchlist"], p.Counts["go_watchlist"])
	if len(p.Suppression) > 0 {
		fmt.Fprintf(&b, "suppression: %s\n", renderToolCounts(p.Suppression))
	}
	if len(p.Violations) > 0 {
		b.WriteString("\nviolations:\n")
		for _, row := range p.Violations {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
	}
	if len(p.Watchlist) > 0 {
		if len(p.Tools) > 0 {
			fmt.Fprintf(&b, "watchlist tools: %s\n", renderToolCounts(p.Tools))
		}
		if len(p.Dirs) > 0 {
			fmt.Fprintf(&b, "watchlist dirs: %s\n", renderToolCounts(p.Dirs))
		}
		b.WriteString("\nwatchlist:\n")
		for _, row := range p.Watchlist {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
	}
	if p.LiveTasks != nil {
		fmt.Fprintf(&b, "\nlive tasks: scanned=%d violations=%d watchlist=%d\n",
			p.LiveTasks.Scanned, len(p.LiveTasks.Violations), len(p.LiveTasks.Watchlist))
		for _, row := range p.LiveTasks.Violations {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
		if len(p.LiveTasks.Watchlist) > 0 {
			b.WriteString("live task watchlist:\n")
			for _, row := range p.LiveTasks.Watchlist {
				fmt.Fprintf(&b, "  - %s\n", row)
			}
		}
	}
	if p.Windows != nil {
		fmt.Fprintf(&b, "\nvisible windows: scanned=%d violations=%d watchlist=%d\n",
			p.Windows.Scanned, len(p.Windows.Violations), len(p.Windows.Watchlist))
		if len(p.Windows.Categories) > 0 {
			fmt.Fprintf(&b, "visible-window categories: %s\n", renderToolCounts(p.Windows.Categories))
		}
		for _, row := range p.Windows.Violations {
			fmt.Fprintf(&b, "  - %s\n", row)
		}
		if len(p.Windows.Watchlist) > 0 {
			b.WriteString("visible-window watchlist:\n")
			for _, row := range p.Windows.Watchlist {
				fmt.Fprintf(&b, "  - %s\n", row)
			}
		}
	}
	fmt.Fprintf(&b, "\nreason: %s\nnext: %s", p.Reason, p.NextAction)
	return b.String()
}

func visibleWindowCategoryCounts(findings []windowgate.VisibleWindowFinding) map[string]int {
	out := map[string]int{}
	for _, finding := range findings {
		key := strings.TrimSpace(finding.Category)
		if key != "" {
			out[key]++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func suppressionCounts(rep windowgate.Report) map[string]int {
	out := map[string]int{}
	if len(rep.PyExplicitModules) > 0 {
		out["py_explicit_modules"] = len(rep.PyExplicitModules)
	}
	if len(rep.PyDefaultModules) > 0 {
		out["py_default_modules"] = len(rep.PyDefaultModules)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

var (
	windowgatePyToolRE = regexp.MustCompile(`subprocess ([^ ]+) launch`)
	windowgateGoToolRE = regexp.MustCompile(`exec\.Command(?:Context)?\([^"]*"([^"]+)"`)
)

func watchlistToolCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		tool := ""
		if m := windowgatePyToolRE.FindStringSubmatch(row); m != nil {
			tool = m[1]
		} else if m := windowgateGoToolRE.FindStringSubmatch(row); m != nil {
			tool = filepath.Base(strings.ReplaceAll(m[1], "\\", "/"))
		}
		tool = strings.ToLower(strings.TrimSpace(tool))
		if tool != "" {
			out[tool]++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistFileCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		file := watchlistRowFile(row)
		if file != "" {
			out[file]++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistDirCounts(rows []string) map[string]int {
	out := map[string]int{}
	for _, row := range rows {
		file := watchlistRowFile(row)
		if file == "" {
			continue
		}
		dir := "."
		if i := strings.LastIndex(file, "/"); i >= 0 {
			dir = file[:i]
		}
		out[dir]++
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func watchlistRowFile(row string) string {
	i := strings.Index(row, ":")
	if i <= 0 {
		return ""
	}
	return filepath.ToSlash(row[:i])
}

func renderToolCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
	}
	return strings.Join(parts, " ")
}
