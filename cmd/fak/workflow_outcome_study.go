package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/workflowoutcomestudy"
)

func runWorkflowOutcomeStudy(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 && (args[0] == "pr-cohort" || args[0] == "prcohort") {
		return runWorkflowOutcomeStudyPRCohort(stdout, stderr, args[1:])
	}
	fs := flag.NewFlagSet("sessions workflow-outcome-study", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak-workflow-outcome-study/1 JSON")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if code, done := parseFlagsRejectArgs(fs, args, stderr); done {
		return code
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak sessions workflow-outcome-study --input STUDY.json [--json]")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions workflow-outcome-study: %v\n", err)
		return 1
	}
	var study workflowoutcomestudy.Study
	if err := json.Unmarshal(raw, &study); err != nil {
		fmt.Fprintf(stderr, "fak sessions workflow-outcome-study: decode: %v\n", err)
		return 1
	}
	report, err := workflowoutcomestudy.Analyze(study)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions workflow-outcome-study: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak sessions workflow-outcome-study")
	}
	fmt.Fprintf(stdout, "workflow outcome study %s: complete-pairs=%d/%d blind-grades=%d gain-claim-ready=%t\n", report.StudyID, report.CompletePairs, report.TaskCount, report.BlindGrades, report.GainClaimReady)
	fmt.Fprintln(stdout, report.EvidenceNote)
	return 0
}

func runWorkflowOutcomeStudyPRCohort(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("workflow-outcome-study pr-cohort", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak-wip-pr-cohort/1 input dataset JSON (optional; defaults to matched benchmark cohort)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if code, done := parseFlagsRejectArgs(fs, args, stderr); done {
		return code
	}

	var data workflowoutcomestudy.ArmData
	if *input != "" {
		raw, err := os.ReadFile(*input)
		if err != nil {
			fmt.Fprintf(stderr, "fak pr-cohort: %v\n", err)
			return 1
		}
		if err := json.Unmarshal(raw, &data); err != nil {
			fmt.Fprintf(stderr, "fak pr-cohort: decode: %v\n", err)
			return 1
		}
	} else {
		data = workflowoutcomestudy.DefaultCohortDataset()
	}

	report := workflowoutcomestudy.EvaluatePRIsolationCohort(data)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak pr-cohort")
	}

	fmt.Fprintf(stdout, "pr-isolation-cohort: decision=%s cohort_size=%d classes=%v\n", report.Decision, report.CohortSize, report.IssueClasses)
	for _, m := range report.Metrics {
		status := "passed"
		if !m.Passed {
			status = "failed"
		}
		fmt.Fprintf(stdout, "  %s: detached=%.1f %s pr_lane=%.1f %s (threshold=%.1f %s, %s)\n",
			m.Name, m.DetachedWorkerValue, m.Unit, m.PRLaneValue, m.Unit, m.Threshold, m.Unit, status)
	}
	fmt.Fprintf(stdout, "rationale: %s\n", report.Rationale)
	return 0
}
