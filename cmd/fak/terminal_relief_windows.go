//go:build windows

package main

import (
	"context"
	"encoding/json"
	"errors"
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

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/terminalbarrier"
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
	barrierDeadline := fs.Duration("barrier-deadline", 30*time.Second, "deadline for every managed pause/checkpoint acknowledgement")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *handles <= 0 || *threads <= 0 || *consecutive <= 0 || *cooldown < 0 || *barrierDeadline <= 0 {
		return 2
	}
	snap, err := gatherTerminalReliefSnapshotFn()
	if err != nil {
		fmt.Fprintf(stderr, "fak terminal-relief: %v\n", err)
		return 1
	}
	facts, managed := terminalReliefFacts(snap)
	prior := terminalrelief.Load(*statePath)
	now := time.Now()
	decision := terminalrelief.Decide(facts, prior, terminalrelief.Config{HandleThreshold: *handles, ThreadThreshold: *threads, Consecutive: *consecutive, Cooldown: *cooldown}, now, *apply)
	// #6436: no host is ever replaced outside a lifecycle transaction. The barrier
	// discovers the managed forest, publishes prepare/pause requests, and only calls
	// the actuator once every active member has acknowledged and checkpointed.
	transaction := fmt.Sprintf("terminal-relief-%d-%d", facts.PID, now.UTC().UnixNano())
	report := terminalbarrier.Coordinator{
		Bus:      fleetbus.LifecycleDirBus{Root: filepath.Dir(*statePath)},
		Actuator: &terminalReliefActuator{pid: facts.PID, dashboards: facts.Dashboards},
	}.Replace(context.Background(), decision.Apply, terminalbarrier.ForestUnderHost(facts.PID, 1, managed), transaction, now.Add(*barrierDeadline))
	if decision.Apply && report.Verdict != "READY" {
		// The barrier never reached the actuator, so nothing was stopped: keep the
		// cooldown unspent and report the abstention instead of a phantom apply.
		decision.Verdict, decision.Reason, decision.Apply = "ABSTAIN", "managed pause barrier did not quiesce the forest: "+report.Reason, false
		decision.State.LastApplied, decision.State.Consecutive = prior.LastApplied, decision.Consecutive
	}
	if err := terminalrelief.Save(*statePath, decision.State); err != nil {
		fmt.Fprintf(stderr, "fak terminal-relief: save state: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := json.NewEncoder(stdout).Encode(terminalReliefRecord{Decision: decision, Barrier: report}); err != nil {
			return 1
		}
		fmt.Fprintln(stderr, report.MonitorLine())
	} else {
		fmt.Fprintf(stdout, "terminal relief: %s — %s (run %d; handles=%d threads=%d dashboards=%d)\n", decision.Verdict, decision.Reason, decision.Consecutive, facts.Handles, facts.Threads, len(facts.Dashboards))
		fmt.Fprintln(stdout, report.MonitorLine())
	}
	if decision.Verdict == "ABSTAIN" {
		return 3
	}
	return 0
}

// terminalReliefRecord is the durable record the stall monitor logs: the relief
// decision inline, plus the lifecycle transaction that gated it.
type terminalReliefRecord struct {
	terminalrelief.Decision
	Barrier terminalbarrier.Report `json:"barrier"`
}

// terminalReliefActuator performs the only destructive step in the transaction, and
// only after the barrier proved every managed member quiesced and restorable.
type terminalReliefActuator struct {
	pid        int
	dashboards []terminalrelief.Command
}

// StopHost replaces the leaking host. The restorable dashboards are relaunched
// detached first so the operator surface never disappears, then the host is replaced.
func (a *terminalReliefActuator) StopHost(context.Context) error {
	for _, dashboard := range a.dashboards {
		if err := launchTerminalReliefCommandFn(append([]string(nil), dashboard.Argv...)); err != nil {
			return fmt.Errorf("relaunch dashboard: %w", err)
		}
	}
	return stopTerminalReliefHostFn(a.pid)
}

// RestoreHost is the read-back half of the replacement: a host is only restored when
// a detached dashboard survived it.
func (a *terminalReliefActuator) RestoreHost(context.Context) error {
	if len(a.dashboards) == 0 {
		return errors.New("no restorable dashboard survived replacement")
	}
	return nil
}
func defaultTerminalReliefStatePath() string {
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "Fleet", "terminal-relief.json")
}

// terminalReliefFacts returns the pressure facts and, alongside them, the managed
// forest membership the lifecycle barrier addresses under this host.
func terminalReliefFacts(s terminalReliefSnapshot) (terminalrelief.Facts, []terminalbarrier.ManagedProcess) {
	byParent := map[int][]terminalReliefProcess{}
	for _, p := range s.Processes {
		byParent[p.ParentPID] = append(byParent[p.ParentPID], p)
	}
	queue := []int{s.PID}
	seen := map[int]bool{s.PID: true}
	var unsafe []string
	var dashboards []terminalrelief.Command
	var managed []terminalbarrier.ManagedProcess
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
					managed = append(managed, terminalbarrier.ManagedProcess{MemberID: fmt.Sprintf("fak-info-%d", p.PID), Image: p.Name})
				}
			}
		}
	}
	sort.Strings(unsafe)
	sort.Slice(dashboards, func(i, j int) bool {
		return strings.Join(dashboards[i].Argv, "\x00") < strings.Join(dashboards[j].Argv, "\x00")
	})
	sort.Slice(managed, func(i, j int) bool { return managed[i].MemberID < managed[j].MemberID })
	return terminalrelief.Facts{PID: s.PID, Handles: s.Handles, Threads: s.Threads, UnsafeDescendants: unsafe, Dashboards: dashboards}, managed
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
