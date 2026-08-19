package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
)

func rwMergeSessiondiagCandidates(plan []resume.WatchdogPlanRow, windowHours float64) ([]resume.WatchdogPlanRow, sessiondiag.WatchdogCandidateReport) {
	report := rwSessiondiagCandidates(windowHours)
	seen := map[string]bool{}
	for _, row := range plan {
		seen[strings.TrimSpace(row.Session)] = true
	}
	for _, candidate := range report.Candidates {
		if seen[candidate.Session] {
			continue
		}
		plan = append(plan, resume.WatchdogPlanRow{Session: candidate.Session, Harness: candidate.Harness, CWD: candidate.CWD, Disp: candidate.Reason})
		seen[candidate.Session] = true
	}
	return plan, report
}

func rwSessiondiagCandidates(windowHours float64) sessiondiag.WatchdogCandidateReport {
	exe := strings.TrimSpace(os.Getenv("FAK_DEV_EXE"))
	if exe == "" {
		exe = "fak-dev"
	}
	since := strconv.FormatFloat(windowHours, 'f', -1, 64) + "h"
	cmd := exec.Command(exe, "sessiondiag", "--inventory", "--watchdog-candidates", "--json", "--since", since)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return sessiondiag.WatchdogCandidateReport{}
	}
	var report sessiondiag.WatchdogCandidateReport
	if json.Unmarshal(stdout.Bytes(), &report) != nil || report.Schema != sessiondiag.WatchdogCandidateSchema {
		return sessiondiag.WatchdogCandidateReport{}
	}
	return report
}
