package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	desktopConsoleSoakSchema  = "fak-desktop-console-soak/1"
	maxDesktopConsoleSoakRuns = 1000
)

var desktopConsoleSoakLabels = []string{"codex-root", "pwsh", "node", "fak-mcp"}

type desktopConsoleSoakProcessCounts struct {
	TrackedTotal int            `json:"tracked_total"`
	Families     map[string]int `json:"families"`
}

type desktopConsoleSoakRun struct {
	Index                int                           `json:"index"`
	OK                   bool                          `json:"ok"`
	ProcessFamilies      int                           `json:"process_families"`
	SharedHiddenConsoles int                           `json:"shared_hidden_consoles"`
	VisibleWindows       int                           `json:"visible_windows"`
	SurvivingChildren    []int                         `json:"surviving_children,omitempty"`
	Report               desktopConsoleSelfcheckReport `json:"report"`
	Reason               string                        `json:"reason"`
}

type desktopConsoleSoakReport struct {
	Schema               string                          `json:"schema"`
	OK                   bool                            `json:"ok"`
	Applicable           bool                            `json:"applicable"`
	Platform             string                          `json:"platform"`
	RunsRequested        int                             `json:"runs_requested"`
	RunsCompleted        int                             `json:"runs_completed"`
	ProcessReports       int                             `json:"process_reports"`
	ExpectedPerRun       int                             `json:"expected_processes_per_run"`
	SharedHiddenConsoles int                             `json:"shared_hidden_consoles"`
	VisibleWindows       int                             `json:"visible_windows"`
	SurvivingChildren    int                             `json:"surviving_children"`
	Before               desktopConsoleSoakProcessCounts `json:"before_process_counts"`
	After                desktopConsoleSoakProcessCounts `json:"after_process_counts"`
	ConsoleHostIncreases map[string]int                  `json:"console_host_increases,omitempty"`
	Runs                 []desktopConsoleSoakRun         `json:"runs,omitempty"`
	Reason               string                          `json:"reason"`
}

type desktopConsoleSoakRunner func(io.Writer, io.Writer, bool) int
type desktopConsoleSoakSnapshot func() (desktopConsoleSoakProcessCounts, error)
type desktopConsoleSoakWaitBaseline func(desktopConsoleSoakProcessCounts, time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error)
type desktopConsoleSoakSurvivors func([]int, time.Duration) []int

func runDesktopConsoleSoak(stdout, stderr io.Writer, asJSON bool, runs int) int {
	if runtime.GOOS != "windows" {
		rep := desktopConsoleSoakReport{
			Schema: desktopConsoleSoakSchema, OK: true, Applicable: false, Platform: runtime.GOOS,
			RunsRequested: runs, ExpectedPerRun: len(desktopConsoleSoakLabels),
			Reason: "Windows console-window semantics are not applicable on this host",
		}
		return writeDesktopConsoleSoakResult(stdout, stderr, rep, asJSON)
	}
	rep, err := executeDesktopConsoleSoak(
		runs,
		runDesktopConsoleSelfcheck,
		snapshotDesktopConsoleSoakProcesses,
		waitDesktopConsoleSoakProcessBaseline,
		waitDesktopConsoleSoakSurvivors,
	)
	if err != nil {
		return failDesktopConsoleSoak(stderr, err)
	}
	return writeDesktopConsoleSoakResult(stdout, stderr, rep, asJSON)
}

func executeDesktopConsoleSoak(
	runs int,
	runner desktopConsoleSoakRunner,
	snapshot desktopConsoleSoakSnapshot,
	waitBaseline desktopConsoleSoakWaitBaseline,
	survivors desktopConsoleSoakSurvivors,
) (desktopConsoleSoakReport, error) {
	if runs < 1 || runs > maxDesktopConsoleSoakRuns {
		return desktopConsoleSoakReport{}, fmt.Errorf("soak runs must be between 1 and %d", maxDesktopConsoleSoakRuns)
	}
	before, err := snapshot()
	if err != nil {
		return desktopConsoleSoakReport{}, fmt.Errorf("capture before process counts: %w", err)
	}
	rep := desktopConsoleSoakReport{
		Schema: desktopConsoleSoakSchema, OK: true, Applicable: true, Platform: "windows",
		RunsRequested: runs, ExpectedPerRun: len(desktopConsoleSoakLabels), Before: before,
		Runs: make([]desktopConsoleSoakRun, 0, runs),
	}
	for i := 1; i <= runs; i++ {
		var out, runErr bytes.Buffer
		code := runner(&out, &runErr, true)
		var one desktopConsoleSelfcheckReport
		decodeErr := json.Unmarshal(out.Bytes(), &one)
		row := desktopConsoleSoakRun{Index: i, Report: one}
		if decodeErr != nil {
			row.Reason = fmt.Sprintf("decode selfcheck JSON: %v", decodeErr)
		} else {
			row.ProcessFamilies = countDesktopConsoleSoakLabels(one.Processes)
			row.SharedHiddenConsoles = one.SharedHiddenConsoles
			row.VisibleWindows = one.VisibleWindows
			pids := make([]int, 0, len(one.Processes))
			for _, process := range one.Processes {
				pids = append(pids, process.PID)
			}
			row.SurvivingChildren = survivors(pids, 10*time.Second)
			row.Reason = validateDesktopConsoleSoakRun(code, runErr.String(), one, row)
		}
		row.OK = row.Reason == ""
		if row.OK {
			row.Reason = "four process families shared one hidden console, exposed zero windows, and exited"
		}
		rep.Runs = append(rep.Runs, row)
		rep.RunsCompleted++
		rep.ProcessReports += len(one.Processes)
		rep.SharedHiddenConsoles += one.SharedHiddenConsoles
		rep.VisibleWindows += one.VisibleWindows
		rep.SurvivingChildren += len(row.SurvivingChildren)
		if !row.OK {
			rep.OK = false
			break
		}
	}
	after, increases, err := waitBaseline(before, 10*time.Second)
	if err != nil {
		return desktopConsoleSoakReport{}, fmt.Errorf("capture after process counts: %w", err)
	}
	rep.After = after
	rep.ConsoleHostIncreases = increases
	if len(increases) > 0 {
		rep.OK = false
	}
	if rep.OK && rep.RunsCompleted == runs {
		rep.Reason = fmt.Sprintf("%d normal-executable runs reported four process families, one shared hidden console, zero visible windows, and no surviving witness children", runs)
	} else if len(increases) > 0 {
		rep.Reason = fmt.Sprintf("tracked console host counts did not return to baseline: %s", desktopConsoleSoakIncreases(increases))
	} else {
		rep.Reason = fmt.Sprintf("soak failed on run %d: %s", rep.RunsCompleted, rep.Runs[len(rep.Runs)-1].Reason)
	}
	return rep, nil
}

func validateDesktopConsoleSoakRun(code int, stderr string, rep desktopConsoleSelfcheckReport, row desktopConsoleSoakRun) string {
	if code != 0 {
		return fmt.Sprintf("selfcheck exit %d: %s", code, strings.TrimSpace(stderr))
	}
	if !rep.Applicable {
		return fmt.Sprintf("selfcheck did not report applicable Windows evidence: %s", rep.Reason)
	}
	if row.ProcessFamilies != len(desktopConsoleSoakLabels) || len(rep.Processes) != len(desktopConsoleSoakLabels) {
		return fmt.Sprintf("reported %d/%d required process families across %d rows", row.ProcessFamilies, len(desktopConsoleSoakLabels), len(rep.Processes))
	}
	if row.SharedHiddenConsoles != 1 {
		return fmt.Sprintf("shared hidden consoles = %d, want 1", row.SharedHiddenConsoles)
	}
	if row.VisibleWindows != 0 {
		return fmt.Sprintf("visible windows = %d, want 0", row.VisibleWindows)
	}
	if len(row.SurvivingChildren) != 0 {
		return fmt.Sprintf("witness children still alive: %v", row.SurvivingChildren)
	}
	if !rep.OK {
		return fmt.Sprintf("selfcheck did not pass: %s", rep.Reason)
	}
	return ""
}

func countDesktopConsoleSoakLabels(processes []desktopConsoleSelfcheckProcess) int {
	seen := make(map[string]bool, len(processes))
	for _, process := range processes {
		seen[process.Label] = true
	}
	count := 0
	for _, label := range desktopConsoleSoakLabels {
		if seen[label] {
			count++
		}
	}
	return count
}

func writeDesktopConsoleSoakResult(stdout, stderr io.Writer, rep desktopConsoleSoakReport, asJSON bool) int {
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			return failDesktopConsoleSoak(stderr, err)
		}
	} else {
		verdict := "PASS"
		if !rep.Applicable {
			verdict = "SKIP"
		} else if !rep.OK {
			verdict = "FAIL"
		}
		fmt.Fprintf(stdout, "%s windowgate soak: %s\n", verdict, rep.Reason)
		fmt.Fprintf(stdout, "  runs=%d/%d process_reports=%d hidden_consoles=%d visible_windows=%d surviving_children=%d\n",
			rep.RunsCompleted, rep.RunsRequested, rep.ProcessReports, rep.SharedHiddenConsoles, rep.VisibleWindows, rep.SurvivingChildren)
		fmt.Fprintf(stdout, "  before=%s after=%s\n", desktopConsoleSoakCounts(rep.Before), desktopConsoleSoakCounts(rep.After))
	}
	if !rep.OK {
		return 1
	}
	return 0
}

func desktopConsoleSoakCounts(counts desktopConsoleSoakProcessCounts) string {
	keys := make([]string, 0, len(counts.Families))
	for key := range counts.Families {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts.Families[key]))
	}
	return fmt.Sprintf("tracked=%d {%s}", counts.TrackedTotal, strings.Join(parts, " "))
}

func desktopConsoleSoakIncreases(increases map[string]int) string {
	return desktopConsoleSoakCounts(desktopConsoleSoakProcessCounts{Families: increases})
}

func failDesktopConsoleSoak(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "fak windowgate --soak: %v\n", err)
	return 1
}
