//go:build windows

package main

import (
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
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/terminalrelief"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type terminalReliefProcess struct {
	PID, ParentPID    int
	Name, CommandLine string
}
type terminalReliefSnapshot struct {
	PID, Handles, Threads int
	Processes             []terminalReliefProcess
}

func newTerminalReliefBackgroundCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

func newTerminalReliefDetachedCommand(argv []string) *exec.Cmd {
	cmd := exec.Command(argv[0], argv[1:]...)
	windowgate.ConfigureDetachedCommand(cmd)
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP
	return cmd
}

var gatherTerminalReliefSnapshotFn = gatherTerminalReliefSnapshot
var launchTerminalReliefCommandFn = func(argv []string) error {
	return newTerminalReliefDetachedCommand(argv).Start()
}
var stopTerminalReliefHostFn = func(pid int) error {
	return newTerminalReliefBackgroundCommand("taskkill.exe", "/PID", strconv.Itoa(pid), "/T", "/F").Run()
}

func cmdTerminalRelief(args []string) { os.Exit(runTerminalRelief(os.Stdout, os.Stderr, args)) }
func runTerminalRelief(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("terminal-relief", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "launch restorable dashboards detached, then replace the leaking Windows Terminal host")
	asJSON := fs.Bool("json", false, "emit JSON")
	handles := fs.Int("handle-threshold", 10000, "terminal handle threshold")
	threads := fs.Int("thread-threshold", 500, "terminal thread threshold")
	consecutive := fs.Int("consecutive", 3, "consecutive pressured observations required")
	cooldown := fs.Duration("cooldown", time.Hour, "minimum time between applied relief")
	statePath := fs.String("state", defaultTerminalReliefStatePath(), "durable relief state")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *handles <= 0 || *threads <= 0 || *consecutive <= 0 || *cooldown < 0 {
		return 2
	}
	snap, err := gatherTerminalReliefSnapshotFn()
	if err != nil {
		fmt.Fprintf(stderr, "fak terminal-relief: %v\n", err)
		return 1
	}
	facts := terminalReliefFacts(snap)
	decision := terminalrelief.Decide(facts, terminalrelief.Load(*statePath), terminalrelief.Config{HandleThreshold: *handles, ThreadThreshold: *threads, Consecutive: *consecutive, Cooldown: *cooldown}, time.Now(), *apply)
	if decision.Apply {
		for _, dashboard := range facts.Dashboards {
			argv := append([]string(nil), dashboard.Argv...)
			if err := launchTerminalReliefCommandFn(argv); err != nil {
				fmt.Fprintf(stderr, "fak terminal-relief: relaunch dashboard: %v\n", err)
				return 1
			}
		}
		if err := stopTerminalReliefHostFn(facts.PID); err != nil {
			fmt.Fprintf(stderr, "fak terminal-relief: stop terminal host: %v\n", err)
			return 1
		}
	}
	if err := terminalrelief.Save(*statePath, decision.State); err != nil {
		fmt.Fprintf(stderr, "fak terminal-relief: save state: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(decision); err != nil {
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "terminal relief: %s — %s (run %d; handles=%d threads=%d dashboards=%d)\n", decision.Verdict, decision.Reason, decision.Consecutive, facts.Handles, facts.Threads, len(facts.Dashboards))
	}
	if decision.Verdict == "ABSTAIN" {
		return 3
	}
	return 0
}
func defaultTerminalReliefStatePath() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "Fleet", "terminal-relief.json")
}
func terminalReliefFacts(s terminalReliefSnapshot) terminalrelief.Facts {
	byParent := map[int][]terminalReliefProcess{}
	for _, p := range s.Processes {
		byParent[p.ParentPID] = append(byParent[p.ParentPID], p)
	}
	queue := []int{s.PID}
	seen := map[int]bool{s.PID: true}
	var unsafe []string
	var dashboards []terminalrelief.Command
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		for _, p := range byParent[id] {
			if seen[p.PID] {
				continue
			}
			seen[p.PID] = true
			queue = append(queue, p.PID)
			name := strings.ToLower(p.Name)
			if terminalReliefUnsafeProcess(name) {
				unsafe = append(unsafe, fmt.Sprintf("%s(pid=%d)", p.Name, p.PID))
			}
			if name == "fak.exe" {
				if argv, ok := parseFakInfoCommand(p.CommandLine); ok {
					dashboards = append(dashboards, terminalrelief.Command{Argv: argv})
				}
			}
		}
	}
	sort.Strings(unsafe)
	sort.Slice(dashboards, func(i, j int) bool {
		return strings.Join(dashboards[i].Argv, "\x00") < strings.Join(dashboards[j].Argv, "\x00")
	})
	return terminalrelief.Facts{PID: s.PID, Handles: s.Handles, Threads: s.Threads, UnsafeDescendants: unsafe, Dashboards: dashboards}
}
func terminalReliefUnsafeProcess(name string) bool {
	switch name {
	case "claude.exe", "codex.exe", "devenv.exe", "emacs.exe", "excel.exe", "notepad.exe", "notepad++.exe", "vim.exe", "winword.exe":
		return true
	default:
		return false
	}
}

func parseFakInfoCommand(line string) ([]string, bool) {
	fields := strings.Fields(line)
	if len(fields) < 2 || strings.ToLower(filepath.Base(strings.Trim(fields[0], "\""))) != "fak.exe" || fields[1] != "info" {
		return nil, false
	}
	out := make([]string, len(fields))
	for i, f := range fields {
		out[i] = strings.Trim(f, "\"")
	}
	return out, true
}
func gatherTerminalReliefSnapshot() (terminalReliefSnapshot, error) {
	script := `$t=Get-Process WindowsTerminal -ErrorAction SilentlyContinue|Sort-Object StartTime|Select-Object -First 1;if(!$t){[pscustomobject]@{pid=0;handles=0;threads=0;processes=@()}|ConvertTo-Json -Depth 4;exit};$p=@(Get-CimInstance Win32_Process|%{[pscustomobject]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;name=$_.Name;command_line=[string]$_.CommandLine}});[pscustomobject]@{pid=$t.Id;handles=$t.Handles;threads=$t.Threads.Count;processes=$p}|ConvertTo-Json -Depth 4 -Compress`
	out, err := newTerminalReliefBackgroundCommand("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script).Output()
	if err != nil {
		return terminalReliefSnapshot{}, err
	}
	var raw struct {
		PID, Handles, Threads int
		Processes             []struct {
			PID         int    `json:"pid"`
			ParentPID   int    `json:"parent_pid"`
			Name        string `json:"name"`
			CommandLine string `json:"command_line"`
		}
	}
	if err := json.Unmarshal(out, &raw); err != nil {
		return terminalReliefSnapshot{}, err
	}
	s := terminalReliefSnapshot{PID: raw.PID, Handles: raw.Handles, Threads: raw.Threads}
	for _, p := range raw.Processes {
		s.Processes = append(s.Processes, terminalReliefProcess{PID: p.PID, ParentPID: p.ParentPID, Name: p.Name, CommandLine: p.CommandLine})
	}
	return s, nil
}
