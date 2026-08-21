package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"
)

func TestRunWindowgateSupportsBoundedSoak(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runWindowgate(&stdout, &stderr, []string{"--soak", "2", "--json"}); code != 0 {
		t.Fatalf("windowgate soak code=%d stderr=%s", code, stderr.String())
	}
	var got desktopConsoleSoakReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode soak JSON: %v\n%s", err, stdout.String())
	}
	if got.RunsRequested != 2 {
		t.Fatalf("runs requested = %d, want 2", got.RunsRequested)
	}
}

func TestExecuteDesktopConsoleSoakCapturesEveryRunAndCleanup(t *testing.T) {
	nextPID := 100
	runner := func(stdout, stderr io.Writer, asJSON bool) int {
		processes := make([]desktopConsoleSelfcheckProcess, 0, len(desktopConsoleSoakLabels))
		for _, label := range desktopConsoleSoakLabels {
			nextPID++
			processes = append(processes, desktopConsoleSelfcheckProcess{Label: label, PID: nextPID})
		}
		rep := desktopConsoleSelfcheckReport{
			Schema: desktopConsoleSelfcheckSchema, OK: true, Applicable: true, Platform: "windows",
			SharedHiddenConsoles: 1, Processes: processes, Reason: "hidden",
		}
		if err := writeDesktopConsoleSelfcheck(stdout, rep, asJSON); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
	before := desktopConsoleSoakProcessCounts{TrackedTotal: 7, Families: map[string]int{"conhost": 1}}
	rep, err := executeDesktopConsoleSoak(
		3,
		runner,
		func() (desktopConsoleSoakProcessCounts, error) { return before, nil },
		func(got desktopConsoleSoakProcessCounts, _ time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error) {
			return got, nil, nil
		},
		func([]int, time.Duration) []int { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if !rep.OK || rep.RunsCompleted != 3 || len(rep.Runs) != 3 || rep.ProcessReports != 12 || rep.SharedHiddenConsoles != 3 || rep.VisibleWindows != 0 || rep.SurvivingChildren != 0 {
		t.Fatalf("soak did not preserve per-run evidence: %+v", rep)
	}
	for _, run := range rep.Runs {
		if !run.OK || run.ProcessFamilies != 4 || len(run.Report.Processes) != 4 {
			t.Fatalf("run evidence = %+v, want four clean process families", run)
		}
	}
}

func TestExecuteDesktopConsoleSoakFailsOnVisibleWindow(t *testing.T) {
	runner := desktopConsoleSoakTestRunner(1)
	rep, err := executeDesktopConsoleSoak(2, runner, emptyDesktopConsoleSoakSnapshot, stableDesktopConsoleSoakBaseline, func([]int, time.Duration) []int { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if rep.OK || rep.RunsCompleted != 1 || rep.VisibleWindows != 1 || !strings.Contains(rep.Reason, "visible windows") {
		t.Fatalf("visible-window soak = %+v, want first-run failure", rep)
	}
}

func TestExecuteDesktopConsoleSoakFailsOnSurvivingChild(t *testing.T) {
	rep, err := executeDesktopConsoleSoak(1, desktopConsoleSoakTestRunner(0), emptyDesktopConsoleSoakSnapshot, stableDesktopConsoleSoakBaseline, func([]int, time.Duration) []int { return []int{44} })
	if err != nil {
		t.Fatal(err)
	}
	if rep.OK || rep.SurvivingChildren != 1 || !strings.Contains(rep.Reason, "still alive") {
		t.Fatalf("surviving-child soak = %+v, want cleanup failure", rep)
	}
}

func TestExecuteDesktopConsoleSoakFailsWhenConsoleHostsAccumulate(t *testing.T) {
	rep, err := executeDesktopConsoleSoak(
		1,
		desktopConsoleSoakTestRunner(0),
		emptyDesktopConsoleSoakSnapshot,
		func(desktopConsoleSoakProcessCounts, time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error) {
			return desktopConsoleSoakProcessCounts{Families: map[string]int{"conhost": 1}}, map[string]int{"conhost": 1}, nil
		},
		func([]int, time.Duration) []int { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if rep.OK || rep.ConsoleHostIncreases["conhost"] != 1 || !strings.Contains(rep.Reason, "baseline") {
		t.Fatalf("console-host accumulation = %+v, want failure", rep)
	}
}

func desktopConsoleSoakTestRunner(visible int) desktopConsoleSoakRunner {
	return func(stdout, stderr io.Writer, asJSON bool) int {
		processes := make([]desktopConsoleSelfcheckProcess, 0, len(desktopConsoleSoakLabels))
		for i, label := range desktopConsoleSoakLabels {
			processes = append(processes, desktopConsoleSelfcheckProcess{Label: label, PID: 40 + i})
		}
		rep := desktopConsoleSelfcheckReport{
			Schema: desktopConsoleSelfcheckSchema, OK: visible == 0, Applicable: true, Platform: "windows",
			SharedHiddenConsoles: 1, VisibleWindows: visible, Processes: processes, Reason: "test",
		}
		if err := writeDesktopConsoleSelfcheck(stdout, rep, asJSON); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		return 0
	}
}

func emptyDesktopConsoleSoakSnapshot() (desktopConsoleSoakProcessCounts, error) {
	return desktopConsoleSoakProcessCounts{Families: map[string]int{}}, nil
}

func stableDesktopConsoleSoakBaseline(before desktopConsoleSoakProcessCounts, _ time.Duration) (desktopConsoleSoakProcessCounts, map[string]int, error) {
	return before, nil, nil
}
