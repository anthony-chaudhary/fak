package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	releaseStatusCIModulePath        = "github.com/anthony-chaudhary/fak"
	releaseStatusCIMaxLogBytes       = 8 << 20
	releaseStatusCIMaxWorkUnits      = 200
	releaseStatusCIMaxTestsPerUnit   = 200
	releaseStatusCIMaxCauses         = 64
	releaseStatusCIHumanWorkUnitCap  = 10
	releaseStatusCIHumanCauseCap     = 3
	releaseStatusCIUnknownStep       = "UNKNOWN STEP"
	releaseStatusCIStatusNotRequired = "not_required"
)

var (
	releaseStatusRunGHJSON = releaseStatusRunExternalJSON
	releaseStatusRunGHText = releaseStatusRunExternalText

	releaseStatusCITestFailRE = regexp.MustCompile(`^--- FAIL: ((?:Test|Fuzz|Example)[A-Za-z0-9_]*) \(`)
	releaseStatusCIPkgFailRE  = regexp.MustCompile(`^FAIL\s+(` + regexp.QuoteMeta(releaseStatusCIModulePath) + `(?:/[A-Za-z0-9._~/-]+)?)(?:\s|$)`)
	releaseStatusCIANSI       = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)
	releaseStatusCISafeArg    = regexp.MustCompile(`^[A-Za-z0-9_./:=+@-]+$`)
)

type releaseStatusCITarget struct {
	Source   string
	Workflow string
	Run      map[string]any
}

type releaseStatusCILogLine struct {
	Job     string
	Step    string
	Message string
}

type releaseStatusCIWorkUnit struct {
	Package   string
	Target    string
	Tests     []string
	Job       string
	Step      string
	Reproduce string
	Argv      []string
	Race      bool
}

type releaseStatusCIStep struct {
	Name       string
	Conclusion string
}

type releaseStatusCIJob struct {
	Name       string
	Conclusion string
	Steps      []releaseStatusCIStep
}

var releaseStatusCICauseOrder = []string{
	"workflow_admission_billing",
	"checkout",
	"toolchain_setup",
	"artifact_transfer",
	"compile_build",
	"formatting",
	"vet",
	"architecture_boundary",
	"go_package_tests",
	"race_test_timeout",
	"claims",
	"repository_hygiene",
	"leakage_security",
	"generated_artifact_drift",
	"commit_provenance",
	"unknown",
}

var releaseStatusCICauseDetails = map[string]string{
	"workflow_admission_billing": "the workflow or job was not admitted, commonly because of an Actions billing/account gate",
	"checkout":                   "repository checkout or source retrieval failed",
	"toolchain_setup":            "the Go/toolchain setup step failed",
	"artifact_transfer":          "a cache or artifact upload/download step failed",
	"compile_build":              "compilation or the build gate failed",
	"formatting":                 "the formatting gate failed",
	"vet":                        "the Go vet gate failed",
	"architecture_boundary":      "an architecture/dependency-boundary gate failed",
	"go_package_tests":           "one or more Go packages have named failing tests",
	"race_test_timeout":          "a race-enabled test or timeout gate failed",
	"claims":                     "the claims/proof contract failed",
	"repository_hygiene":         "repository hygiene or worktree cleanliness failed",
	"leakage_security":           "a leakage, secret, credential, or security gate failed",
	"generated_artifact_drift":   "a generated artifact or checked-in derived document drifted",
	"commit_provenance":          "a DOS, DCO, commit, or ship-witness contract failed",
	"unknown":                    "the failed job/step did not match a known release CI family",
}

// releaseStatusCIDiagnosis reads logs only when CI_BASE_RED is the compatibility
// blocker. The decisive workflow comes from release_decide's ci_source; the exact
// run comes from the matching release_context signal, never from an unrelated
// hard-coded workflow query.
func releaseStatusCIDiagnosis(root string, decision, contextPayload map[string]any) map[string]any {
	if !releaseStatusContains(releaseStatusStringSlice(decision["blockers"]), "CI_BASE_RED") {
		return map[string]any{
			"status": releaseStatusCIStatusNotRequired,
			"scope":  "decisive",
			"kind":   "",
			"reason": "CI_BASE_RED is not present; no failed-log read was performed",
			"detail": "CI_BASE_RED is not present; no failed-log read was performed",
		}
	}

	target := releaseStatusSelectCITarget(decision, contextPayload)
	if target.Workflow == "" {
		return releaseStatusCIUnavailable(target, "release decision did not identify an effective CI workflow")
	}
	if conclusion := releaseStatusString(target.Run["conclusion"]); conclusion != "" && !releaseStatusCIConclusionFailed(conclusion) {
		return releaseStatusCIUnavailable(target, fmt.Sprintf(
			"release decision says CI_BASE_RED, but the matching %s context run now concludes %s; refusing to diagnose a different run",
			target.Workflow, conclusion,
		))
	}
	runID := releaseStatusCIRunID(target.Run)
	if runID == 0 {
		return releaseStatusCIUnavailable(target, "effective decision/context evidence did not identify the exact run that produced CI_BASE_RED; refusing to substitute a newer run")
	}

	var errs []string
	viewAny, err := releaseStatusRunGHJSON(
		root, 60*time.Second, "gh", "run", "view", strconv.FormatInt(runID, 10),
		"--json", "databaseId,name,workflowName,displayTitle,headBranch,headSha,status,conclusion,url,event,createdAt,updatedAt,jobs",
	)
	view := releaseStatusMap(viewAny)
	if err != nil {
		errs = append(errs, "run metadata: "+err.Error())
	}
	if conclusion := releaseStatusString(view["conclusion"]); conclusion != "" && !releaseStatusCIConclusionFailed(conclusion) {
		unavailable := releaseStatusCIUnavailable(target, fmt.Sprintf(
			"selected %s run %d now concludes %s, not red; refusing to diagnose a different failure",
			target.Workflow, runID, conclusion,
		))
		unavailable["run"] = releaseStatusCINormalizeRun(target, view)
		return unavailable
	}

	logText, logTruncated, logErr := releaseStatusRunGHText(
		root, 90*time.Second, "gh", "run", "view", strconv.FormatInt(runID, 10), "--log-failed",
	)
	if logErr != nil {
		errs = append(errs, "failed log: "+logErr.Error())
	}

	fallbackText := ""
	if strings.TrimSpace(logText) == "" {
		var fallbackTruncated bool
		fallbackText, fallbackTruncated, err = releaseStatusRunGHText(
			root, 30*time.Second, "gh", "run", "view", strconv.FormatInt(runID, 10),
		)
		logTruncated = logTruncated || fallbackTruncated
		if err != nil {
			errs = append(errs, "run summary: "+err.Error())
		}
	}

	diagnosis := releaseStatusBuildCIDiagnosis(target, view, logText, fallbackText)
	if logTruncated {
		diagnosis["truncated"] = true
	}
	if len(errs) > 0 {
		diagnosis["errors"] = errs
	}
	return diagnosis
}

func releaseStatusSelectCITarget(decision, contextPayload map[string]any) releaseStatusCITarget {
	source := strings.TrimSpace(releaseStatusString(decision["ci_source"]))
	baseSource := strings.TrimSuffix(source, "+ancestor")
	var key, fallbackWorkflow string
	switch baseSource {
	case "fast":
		key = "ci_fast"
		fallbackWorkflow = strings.TrimSpace(os.Getenv("FAK_RELEASE_FAST_CI_WORKFLOW"))
		if fallbackWorkflow == "" {
			fallbackWorkflow = "ci-fast.yml"
		}
	case "whole":
		key = "ci_on_head"
		fallbackWorkflow = "ci.yml"
	default:
		return releaseStatusCITarget{Source: baseSource}
	}
	signal := releaseStatusMap(contextPayload[key])
	workflow := releaseStatusFirstString(releaseStatusString(signal["workflow"]), fallbackWorkflow)
	run := releaseStatusMap(signal["latest_trunk_ci"])
	return releaseStatusCITarget{Source: baseSource, Workflow: workflow, Run: run}
}

func releaseStatusCIUnavailable(target releaseStatusCITarget, reason string) map[string]any {
	inspectArgv := []string{"gh", "run", "list", "--branch", "main", "--status", "completed", "--limit", "5"}
	inspect := releaseStatusCICommandText(inspectArgv)
	return map[string]any{
		"status":            "unavailable",
		"scope":             "decisive",
		"source":            target.Source,
		"workflow":          target.Workflow,
		"kind":              "unknown",
		"reason":            reason,
		"detail":            reason,
		"action":            "inspect_ci",
		"inspect_command":   inspect,
		"inspect_argv":      inspectArgv,
		"failed_job_count":  0,
		"failed_step_count": 0,
		"work_unit_count":   0,
		"failed_test_count": 0,
		"causes": []any{
			map[string]any{
				"kind":            "unknown",
				"detail":          releaseStatusCICauseDetails["unknown"],
				"inspect_command": inspect,
				"inspect_argv":    inspectArgv,
			},
		},
		"work_units": []any{},
	}
}

func releaseStatusBuildCIDiagnosis(target releaseStatusCITarget, view map[string]any, logText, fallbackText string) map[string]any {
	run := releaseStatusCINormalizeRun(target, view)
	runID := releaseStatusCIRunID(run)
	inspect, inspectArgv := releaseStatusCIInspect(runID)
	jobs := releaseStatusCIJobs(view)
	lines := releaseStatusCIParseLog(logText)
	lines = releaseStatusCIAttachFailedSteps(lines, jobs)
	units, unitsTruncated := releaseStatusCIWorkUnits(lines)
	causes, causesTruncated := releaseStatusCICauses(
		jobs, lines, fallbackText,
		releaseStatusFirstString(releaseStatusString(view["conclusion"]), releaseStatusString(target.Run["conclusion"])),
		units, inspect, inspectArgv,
	)

	failedTests := 0
	workUnitMaps := make([]any, 0, len(units))
	for _, unit := range units {
		failedTests += len(unit.Tests)
		workUnitMaps = append(workUnitMaps, map[string]any{
			"kind":           "go_package_tests",
			"package":        unit.Package,
			"target":         unit.Target,
			"tests":          unit.Tests,
			"test_count":     len(unit.Tests),
			"test_scope":     "top_level",
			"job":            unit.Job,
			"step":           unit.Step,
			"reproduce":      unit.Reproduce,
			"reproduce_argv": unit.Argv,
			"race":           unit.Race,
		})
	}

	failedJobs, failedSteps := releaseStatusCIFailureCounts(jobs)
	action := "fix_ci"
	status := "actionable"
	if releaseStatusCIHasCause(causes, "workflow_admission_billing") {
		action = "fix_ci_billing"
		status = "blocked"
	} else if len(causes) == 0 {
		action = "inspect_ci"
		status = "undifferentiated"
		causes = []any{
			map[string]any{
				"kind":            "unknown",
				"detail":          releaseStatusCICauseDetails["unknown"],
				"inspect_command": inspect,
				"inspect_argv":    inspectArgv,
			},
		}
	}

	summary := releaseStatusCISummary(target.Workflow, runID, len(units), failedTests, len(causes))
	kind := "unknown"
	if len(units) > 0 {
		kind = "go_package_tests"
	} else if first := releaseStatusCIFirstCause(map[string]any{"causes": causes}); len(first) > 0 {
		kind = releaseStatusFirstString(releaseStatusString(first["kind"]), kind)
	}
	out := map[string]any{
		"status":            status,
		"scope":             "decisive",
		"source":            target.Source,
		"workflow":          target.Workflow,
		"run":               run,
		"kind":              kind,
		"summary":           summary,
		"detail":            summary,
		"reason":            summary,
		"action":            action,
		"inspect_command":   inspect,
		"inspect_argv":      inspectArgv,
		"failed_job_count":  failedJobs,
		"failed_step_count": failedSteps,
		"work_unit_count":   len(units),
		"failed_test_count": failedTests,
		"causes":            causes,
		"work_units":        workUnitMaps,
	}
	if unitsTruncated || causesTruncated {
		out["truncated"] = true
	}
	return out
}

func releaseStatusCINormalizeRun(target releaseStatusCITarget, view map[string]any) map[string]any {
	runID := releaseStatusFirstInt64(
		releaseStatusCIRunID(view),
		releaseStatusCIRunID(target.Run),
	)
	return map[string]any{
		"database_id": runID,
		"databaseId":  runID,
		"workflow": releaseStatusFirstString(
			releaseStatusString(view["workflowName"]),
			releaseStatusString(view["name"]),
			target.Workflow,
		),
		"head_sha": releaseStatusFirstString(
			releaseStatusString(view["headSha"]),
			releaseStatusString(target.Run["head_sha"]),
			releaseStatusString(target.Run["headSha"]),
		),
		"headSha": releaseStatusFirstString(
			releaseStatusString(view["headSha"]),
			releaseStatusString(target.Run["head_sha"]),
			releaseStatusString(target.Run["headSha"]),
		),
		"status": releaseStatusFirstString(
			releaseStatusString(view["status"]),
			releaseStatusString(target.Run["status"]),
		),
		"conclusion": releaseStatusFirstString(
			releaseStatusString(view["conclusion"]),
			releaseStatusString(target.Run["conclusion"]),
		),
		"display_title": releaseStatusFirstString(
			releaseStatusString(view["displayTitle"]),
			releaseStatusString(target.Run["display_title"]),
			releaseStatusString(target.Run["displayTitle"]),
		),
		"displayTitle": releaseStatusFirstString(
			releaseStatusString(view["displayTitle"]),
			releaseStatusString(target.Run["display_title"]),
			releaseStatusString(target.Run["displayTitle"]),
		),
		"url": releaseStatusFirstString(
			releaseStatusString(view["url"]),
			releaseStatusString(target.Run["url"]),
		),
	}
}

func releaseStatusCIRunID(run map[string]any) int64 {
	return releaseStatusFirstInt64(releaseStatusInt64(run["database_id"]), releaseStatusInt64(run["databaseId"]))
}

func releaseStatusFirstInt64(values ...int64) int64 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func releaseStatusInt64(v any) int64 {
	switch value := v.(type) {
	case int:
		return int64(value)
	case int64:
		return value
	case float64:
		return int64(value)
	case string:
		n, _ := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
		return n
	default:
		return 0
	}
}

func releaseStatusCIJobs(view map[string]any) []releaseStatusCIJob {
	var jobs []releaseStatusCIJob
	for _, rawJob := range releaseStatusAnySlice(view["jobs"]) {
		jobMap := releaseStatusMap(rawJob)
		job := releaseStatusCIJob{
			Name:       releaseStatusString(jobMap["name"]),
			Conclusion: releaseStatusString(jobMap["conclusion"]),
		}
		for _, rawStep := range releaseStatusAnySlice(jobMap["steps"]) {
			stepMap := releaseStatusMap(rawStep)
			job.Steps = append(job.Steps, releaseStatusCIStep{
				Name:       releaseStatusString(stepMap["name"]),
				Conclusion: releaseStatusString(stepMap["conclusion"]),
			})
		}
		jobs = append(jobs, job)
	}
	return jobs
}

func releaseStatusAnySlice(v any) []any {
	switch values := v.(type) {
	case []any:
		return values
	case []map[string]any:
		out := make([]any, len(values))
		for i := range values {
			out[i] = values[i]
		}
		return out
	default:
		return nil
	}
}

func releaseStatusCIParseLog(text string) []releaseStatusCILogLine {
	text = releaseStatusCIANSI.ReplaceAllString(strings.ReplaceAll(text, "\r\n", "\n"), "")
	rawLines := strings.Split(text, "\n")
	lines := make([]releaseStatusCILogLine, 0, len(rawLines))
	for _, raw := range rawLines {
		if raw == "" || strings.HasPrefix(raw, "# Captured ") {
			continue
		}
		job, step, message := releaseStatusCIParseLogLine(raw)
		lines = append(lines, releaseStatusCILogLine{Job: job, Step: step, Message: message})
	}
	return lines
}

func releaseStatusCIAttachFailedSteps(lines []releaseStatusCILogLine, jobs []releaseStatusCIJob) []releaseStatusCILogLine {
	onlyFailedStep := map[string]string{}
	for _, job := range jobs {
		var failed []string
		for _, step := range job.Steps {
			if releaseStatusCIConclusionFailed(step.Conclusion) {
				failed = append(failed, step.Name)
			}
		}
		if len(failed) == 1 {
			onlyFailedStep[job.Name] = failed[0]
		}
	}
	out := append([]releaseStatusCILogLine(nil), lines...)
	for i := range out {
		if out[i].Step == "" || strings.EqualFold(out[i].Step, releaseStatusCIUnknownStep) {
			if step := onlyFailedStep[out[i].Job]; step != "" {
				out[i].Step = step
			}
		}
	}
	return out
}

func releaseStatusCIParseLogLine(line string) (job, step, message string) {
	parts := strings.SplitN(line, "\t", 3)
	if len(parts) != 3 {
		return "", "", line
	}
	rest := parts[2]
	space := strings.IndexByte(rest, ' ')
	if space <= 0 || !strings.Contains(rest[:space], "T") {
		return "", "", line
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), rest[space+1:]
}

func releaseStatusCIWorkUnits(lines []releaseStatusCILogLine) ([]releaseStatusCIWorkUnit, bool) {
	type outputKey struct {
		job  string
		step string
	}
	type unitKey struct {
		job  string
		step string
		pkg  string
	}
	pending := map[outputKey][]string{}
	unitsByKey := map[unitKey]*releaseStatusCIWorkUnit{}
	truncated := false

	for _, line := range lines {
		k := outputKey{job: line.Job, step: line.Step}
		if match := releaseStatusCITestFailRE.FindStringSubmatch(line.Message); len(match) == 2 {
			if len(pending[k]) < releaseStatusCIMaxTestsPerUnit {
				pending[k] = append(pending[k], match[1])
			} else {
				truncated = true
			}
			continue
		}
		match := releaseStatusCIPkgFailRE.FindStringSubmatch(line.Message)
		if len(match) != 2 {
			continue
		}
		pkg := match[1]
		if !releaseStatusCIValidPackage(pkg) {
			truncated = true
			delete(pending, k)
			continue
		}
		uk := unitKey{job: line.Job, step: line.Step, pkg: pkg}
		unit := unitsByKey[uk]
		if unit == nil {
			if len(unitsByKey) >= releaseStatusCIMaxWorkUnits {
				truncated = true
				delete(pending, k)
				continue
			}
			unit = &releaseStatusCIWorkUnit{
				Package: pkg,
				Target:  releaseStatusCIPackageTarget(pkg),
				Job:     line.Job,
				Step:    line.Step,
				Race:    releaseStatusCIStepIsRace(line.Step),
			}
			unitsByKey[uk] = unit
		}
		unit.Tests = append(unit.Tests, pending[k]...)
		delete(pending, k)
	}

	units := make([]releaseStatusCIWorkUnit, 0, len(unitsByKey))
	for _, unit := range unitsByKey {
		unit.Tests = releaseStatusCIDedupeSorted(unit.Tests)
		unit.Reproduce, unit.Argv = releaseStatusCIReproduceCommand(unit.Target, unit.Tests, unit.Race)
		units = append(units, *unit)
	}
	sort.Slice(units, func(i, j int) bool {
		if units[i].Package != units[j].Package {
			return units[i].Package < units[j].Package
		}
		if units[i].Job != units[j].Job {
			return units[i].Job < units[j].Job
		}
		return units[i].Step < units[j].Step
	})
	return units, truncated
}

func releaseStatusCIValidPackage(pkg string) bool {
	if pkg == releaseStatusCIModulePath {
		return true
	}
	if !strings.HasPrefix(pkg, releaseStatusCIModulePath+"/") {
		return false
	}
	for _, segment := range strings.Split(strings.TrimPrefix(pkg, releaseStatusCIModulePath+"/"), "/") {
		if segment == "" || segment == ".." || segment == "." {
			return false
		}
	}
	return true
}

func releaseStatusCIPackageTarget(pkg string) string {
	if pkg == releaseStatusCIModulePath {
		return "."
	}
	if strings.HasPrefix(pkg, releaseStatusCIModulePath+"/") {
		return "./" + strings.TrimPrefix(pkg, releaseStatusCIModulePath+"/")
	}
	return pkg
}

func releaseStatusCIReproduceCommand(target string, tests []string, race bool) (string, []string) {
	argv := []string{"fak", "test", target, "--"}
	if race {
		argv = append(argv, "-race")
	}
	topLevel := make([]string, 0, len(tests))
	for _, test := range tests {
		if slash := strings.IndexByte(test, '/'); slash >= 0 {
			test = test[:slash]
		}
		if test != "" {
			topLevel = append(topLevel, test)
		}
	}
	topLevel = releaseStatusCIDedupeSorted(topLevel)
	if len(topLevel) == 0 {
		argv = append(argv, "-count=1")
		return releaseStatusCICommandText(argv), argv
	}
	escaped := make([]string, len(topLevel))
	for i, test := range topLevel {
		escaped[i] = regexp.QuoteMeta(test)
	}
	argv = append(argv, "-run", `^(?:`+strings.Join(escaped, "|")+`)$`, "-count=1")
	return releaseStatusCICommandText(argv), argv
}

func releaseStatusCIDedupeSorted(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func releaseStatusCICauses(jobs []releaseStatusCIJob, lines []releaseStatusCILogLine, fallbackText, runConclusion string, units []releaseStatusCIWorkUnit, inspect string, inspectArgv []string) ([]any, bool) {
	type logKey struct {
		job  string
		step string
	}
	logByKey := map[logKey]*strings.Builder{}
	logByJob := map[string]*strings.Builder{}
	for _, line := range lines {
		keyBuilder := logByKey[logKey{job: line.Job, step: line.Step}]
		if keyBuilder == nil {
			keyBuilder = &strings.Builder{}
			logByKey[logKey{job: line.Job, step: line.Step}] = keyBuilder
		}
		keyBuilder.WriteString(line.Message)
		keyBuilder.WriteByte('\n')
		builder := logByJob[line.Job]
		if builder == nil {
			builder = &strings.Builder{}
			logByJob[line.Job] = builder
		}
		builder.WriteString(line.Message)
		builder.WriteByte('\n')
	}

	type cause struct {
		Kind string
		Job  string
		Step string
	}
	seen := map[cause]bool{}
	var causes []cause
	appendKinds := func(job, step, text string) {
		for _, kind := range releaseStatusCICauseKinds(job, step, text) {
			c := cause{Kind: kind, Job: job, Step: step}
			if !seen[c] {
				seen[c] = true
				causes = append(causes, c)
			}
		}
	}

	for _, job := range jobs {
		var failed []releaseStatusCIStep
		for _, step := range job.Steps {
			if releaseStatusCIConclusionFailed(step.Conclusion) {
				failed = append(failed, step)
			}
		}
		for _, step := range failed {
			text := step.Conclusion
			if builder := logByKey[logKey{job: job.Name, step: step.Name}]; builder != nil {
				text = builder.String()
			} else if len(failed) == 1 {
				if builder := logByJob[job.Name]; builder != nil {
					text += "\n" + builder.String()
				}
			}
			appendKinds(job.Name, step.Name, text)
		}
		if len(failed) == 0 && releaseStatusCIConclusionFailed(job.Conclusion) {
			text := strings.Join([]string{job.Conclusion, runConclusion, fallbackText}, "\n")
			if builder := logByJob[job.Name]; builder != nil {
				text += "\n" + builder.String()
			}
			appendKinds(job.Name, "", text)
		}
	}
	if len(jobs) == 0 {
		var all strings.Builder
		all.WriteString(runConclusion)
		all.WriteByte('\n')
		all.WriteString(fallbackText)
		for _, line := range lines {
			all.WriteByte('\n')
			all.WriteString(line.Message)
		}
		appendKinds("", "", all.String())
	}
	if len(units) > 0 {
		first := units[0]
		c := cause{Kind: "go_package_tests", Job: first.Job, Step: first.Step}
		if !seen[c] {
			seen[c] = true
			causes = append(causes, c)
		}
	}

	order := map[string]int{}
	for i, kind := range releaseStatusCICauseOrder {
		order[kind] = i
	}
	sort.SliceStable(causes, func(i, j int) bool {
		if order[causes[i].Kind] != order[causes[j].Kind] {
			return order[causes[i].Kind] < order[causes[j].Kind]
		}
		if causes[i].Job != causes[j].Job {
			return causes[i].Job < causes[j].Job
		}
		return causes[i].Step < causes[j].Step
	})

	truncated := false
	if len(causes) > releaseStatusCIMaxCauses {
		causes = causes[:releaseStatusCIMaxCauses]
		truncated = true
	}
	out := make([]any, 0, len(causes))
	for _, c := range causes {
		out = append(out, map[string]any{
			"kind":            c.Kind,
			"job":             c.Job,
			"step":            c.Step,
			"detail":          releaseStatusCICauseDetails[c.Kind],
			"inspect_command": inspect,
			"inspect_argv":    inspectArgv,
		})
	}
	return out, truncated
}

func releaseStatusCICauseKinds(job, step, logText string) []string {
	text := strings.ToLower(strings.Join([]string{job, step, logText}, "\n"))
	raceText := strings.ReplaceAll(text, "no -race", "")
	var kinds []string
	add := func(kind string, patterns ...string) {
		for _, pattern := range patterns {
			if strings.Contains(text, pattern) {
				kinds = append(kinds, kind)
				return
			}
		}
	}

	add("workflow_admission_billing", "billing", "payments have failed", "not started because", "startup_failure", "action_required")
	add("checkout", "actions/checkout", "checkout", "git fetch", "source retrieval")
	add("toolchain_setup", "actions/setup-go", "setup go", "setup-go", "toolchain setup", "go toolchain")
	add("artifact_transfer", "actions/cache", "cache go", "upload-artifact", "download-artifact", "artifact upload", "artifact download")
	add("compile_build", "\nbuild\n", " compile", "compilation", "[build failed]", "undefined:", "cannot use ")
	add("formatting", "gofmt", "formatting gate", "format check")
	add("vet", "\nvet\n", "go vet", " vet gate")
	add("architecture_boundary", "architest", "architecture gate", "dependency boundary")
	add("go_package_tests", "go test", "--- fail:", "\nfail\t"+strings.ToLower(releaseStatusCIModulePath))
	if strings.Contains(raceText, "data race") ||
		strings.Contains(raceText, "test timed out") ||
		strings.Contains(raceText, "timed_out") ||
		strings.Contains(raceText, "timeout") ||
		strings.Contains(strings.ToLower(step), "race") && !strings.Contains(strings.ToLower(step), "no -race") {
		kinds = append(kinds, "race_test_timeout")
	}
	add("claims", "claims lint", "claim gate", "claims/", "claims.md")
	add("repository_hygiene", "repository hygiene", "repo hygiene", "dirty worktree", "untracked path", "tree-doctor")
	add("leakage_security", "leakage", "secret", "credential", "security gate")
	add("generated_artifact_drift", "generated artifact", "drifted", "stale generated", "--write-doc", "regenerate")
	add("commit_provenance", "commit witness", "dco", "signed-off-by", "ship witness", "ship-stamp", "unwitnessed", "dos verify")
	if len(kinds) == 0 {
		kinds = append(kinds, "unknown")
	}
	return releaseStatusCIDedupeKinds(kinds)
}

func releaseStatusCIDedupeKinds(values []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, expected := range releaseStatusCICauseOrder {
		for _, value := range values {
			if value == expected && !seen[value] {
				seen[value] = true
				out = append(out, value)
			}
		}
	}
	return out
}

func releaseStatusCIConclusionFailed(conclusion string) bool {
	switch strings.ToLower(strings.TrimSpace(conclusion)) {
	case "failure", "failed", "cancelled", "timed_out", "action_required", "startup_failure":
		return true
	default:
		return false
	}
}

func releaseStatusCIFailureCounts(jobs []releaseStatusCIJob) (failedJobs, failedSteps int) {
	for _, job := range jobs {
		if releaseStatusCIConclusionFailed(job.Conclusion) {
			failedJobs++
		}
		for _, step := range job.Steps {
			if releaseStatusCIConclusionFailed(step.Conclusion) {
				failedSteps++
			}
		}
	}
	return failedJobs, failedSteps
}

func releaseStatusCIHasCause(causes []any, want string) bool {
	for _, raw := range causes {
		if releaseStatusString(releaseStatusMap(raw)["kind"]) == want {
			return true
		}
	}
	return false
}

func releaseStatusCISummary(workflow string, runID int64, workUnits, failedTests, causes int) string {
	runLabel := workflow
	if runID != 0 {
		runLabel = fmt.Sprintf("%s run %d", workflow, runID)
	}
	if workUnits > 0 {
		return fmt.Sprintf("%s decomposes into %d failed Go package work unit(s) and %d top-level named failing test(s)", runLabel, workUnits, failedTests)
	}
	return fmt.Sprintf("%s has %d typed failure cause(s) and no package-level Go test unit", runLabel, causes)
}

func releaseStatusCIInspect(runID int64) (string, []string) {
	if runID != 0 {
		argv := []string{"gh", "run", "view", strconv.FormatInt(runID, 10), "--log-failed"}
		return releaseStatusCICommandText(argv), argv
	}
	argv := []string{"gh", "run", "list", "--branch", "main", "--status", "completed", "--limit", "5"}
	return releaseStatusCICommandText(argv), argv
}

func releaseStatusCICommandText(argv []string) string {
	rendered := make([]string, len(argv))
	for i, arg := range argv {
		if releaseStatusCISafeArg.MatchString(arg) {
			rendered[i] = arg
			continue
		}
		if !strings.Contains(arg, "'") {
			rendered[i] = "'" + arg + "'"
			continue
		}
		rendered[i] = strconv.Quote(arg)
	}
	return strings.Join(rendered, " ")
}

func releaseStatusCIStepIsRace(step string) bool {
	lower := strings.ToLower(step)
	return strings.Contains(lower, "race") && !strings.Contains(lower, "no -race")
}

func releaseStatusCIFirstWorkUnit(ciDiag map[string]any) map[string]any {
	units := releaseStatusAnySlice(ciDiag["work_units"])
	if len(units) == 0 {
		return nil
	}
	return releaseStatusMap(units[0])
}

func releaseStatusCIFirstCause(ciDiag map[string]any) map[string]any {
	causes := releaseStatusAnySlice(ciDiag["causes"])
	if len(causes) == 0 {
		return nil
	}
	return releaseStatusMap(causes[0])
}

func releaseStatusCIDiagnosisLines(ciDiag map[string]any) []string {
	status := releaseStatusString(ciDiag["status"])
	if status == "" || status == releaseStatusCIStatusNotRequired {
		return nil
	}
	summary := releaseStatusFirstString(releaseStatusString(ciDiag["summary"]), releaseStatusString(ciDiag["reason"]))
	lines := []string{"  ci diagnosis: " + summary}
	units := releaseStatusAnySlice(ciDiag["work_units"])
	for i, raw := range units {
		if i >= releaseStatusCIHumanWorkUnitCap {
			lines = append(lines, fmt.Sprintf("    ... %d more work unit(s); use --json for the complete diagnosis", len(units)-i))
			break
		}
		unit := releaseStatusMap(raw)
		lines = append(lines, fmt.Sprintf(
			"    work unit: %s (%d test(s)) - %s",
			releaseStatusFirstString(releaseStatusString(unit["target"]), releaseStatusString(unit["package"])),
			releaseStatusInt(unit["test_count"]),
			releaseStatusString(unit["reproduce"]),
		))
	}
	causes := releaseStatusAnySlice(ciDiag["causes"])
	shownCauses := 0
	for _, raw := range causes {
		cause := releaseStatusMap(raw)
		if len(units) > 0 && releaseStatusString(cause["kind"]) == "go_package_tests" {
			continue
		}
		if shownCauses >= releaseStatusCIHumanCauseCap {
			lines = append(lines, "    ... more cause(s); use --json for the complete diagnosis")
			break
		}
		lines = append(lines, fmt.Sprintf(
			"    cause: %s - %s; inspect with `%s`",
			releaseStatusString(cause["kind"]),
			releaseStatusString(cause["detail"]),
			releaseStatusFirstString(releaseStatusString(cause["inspect_command"]), releaseStatusString(ciDiag["inspect_command"])),
		))
		shownCauses++
	}
	return lines
}

type releaseStatusCILimitedBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (b *releaseStatusCILimitedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.limit - b.buf.Len()
	if remaining > 0 {
		if remaining > len(p) {
			remaining = len(p)
		}
		_, _ = b.buf.Write(p[:remaining])
	}
	if len(p) > remaining {
		b.truncated = true
	}
	return len(p), nil
}

func (b *releaseStatusCILimitedBuffer) Result() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String(), b.truncated
}

func releaseStatusRunExternalText(root string, timeout time.Duration, name string, args ...string) (string, bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = root
	var out releaseStatusCILimitedBuffer
	out.limit = releaseStatusCIMaxLogBytes
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	text, truncated := out.Result()
	if ctx.Err() == context.DeadlineExceeded {
		return text, truncated, fmt.Errorf("%s timed out after %s", name, timeout)
	}
	if err != nil {
		return text, truncated, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, releaseStatusTail(text, 1000))
	}
	return text, truncated, nil
}
