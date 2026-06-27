package swebench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
)

// This file is the resolve-rate EVAL seam: turning a predictions file into the
// standard SWE-bench Verified number (% issues resolved) via the OFFICIAL harness
// — the identical path the external bench tool grades with. bench shells to
// `python -m swebench.harness.run_evaluation --predictions_path <dir>/preds.json
// --run_id <id> --max_workers N`, parses `Resolved X/Y` from stdout, and reads
// logs/run_evaluation/<run_id>/report.json (Benchmark repo commands/_swebench_grade.py).
// We do the same locally when the harness + Docker are present, and otherwise
// return an HONESTLY GATED result — on this Mac there is no Docker, so resolve-rate
// is a DGX-only metric and we say so rather than fabricate a number.

// EvalCapability reports whether the official resolve-rate harness can run on this
// box. Both Docker and the swebench python module are required; the eval builds a
// per-repo Docker image, applies the predicted + test patch, and runs the
// FAIL_TO_PASS / PASS_TO_PASS suites inside it.
type EvalCapability struct {
	DockerPresent  bool   `json:"docker_present"`
	HarnessPresent bool   `json:"harness_present"` // `python -m swebench` importable
	Python         string `json:"python"`          // resolved python interpreter, if any
	Runnable       bool   `json:"runnable"`        // DockerPresent && HarnessPresent
	Reason         string `json:"reason"`          // why not runnable, when applicable
}

// DetectEvalCapability probes for Docker + the swebench harness. python is the
// interpreter to probe with (e.g. "python3" or a venv python); empty defaults to
// "python3" then "python".
func DetectEvalCapability(python string) EvalCapability {
	cap := EvalCapability{}
	if _, err := exec.LookPath("docker"); err == nil {
		cap.DockerPresent = true
	}
	for _, cand := range pythonCandidates(python) {
		if _, err := exec.LookPath(cand); err != nil {
			continue
		}
		cap.Python = cand
		// `python -c "import swebench"` — cheap importability probe.
		if err := exec.Command(cand, "-c", "import swebench").Run(); err == nil {
			cap.HarnessPresent = true
			break
		}
	}
	cap.Runnable = cap.DockerPresent && cap.HarnessPresent
	switch {
	case !cap.DockerPresent && !cap.HarnessPresent:
		cap.Reason = "no Docker and no swebench harness — resolve-rate is a DGX/Docker-only metric on this box"
	case !cap.DockerPresent:
		cap.Reason = "Docker not found — the harness builds a per-repo image to run the tests"
	case !cap.HarnessPresent:
		cap.Reason = "swebench python module not importable — `pip install swebench[harness]` (or run scripts/install_swebench.py)"
	}
	return cap
}

func pythonCandidates(python string) []string {
	if python != "" {
		return []string{python}
	}
	return []string{"python3", "python"}
}

// EvalConfig drives a local resolve-rate run.
type EvalConfig struct {
	PredictionsPath string // path to preds.json
	DatasetName     string // e.g. "princeton-nlp/SWE-bench_Verified"
	RunID           string // names the harness output dir (logs/run_evaluation/<run_id>)
	MaxWorkers      int    // harness parallelism (default 4)
	Python          string // interpreter (default: detected)
}

// EvalResult is the resolve-rate outcome — Available=false with a Reason when the
// box cannot run the harness (the honest local case), Available=true with the
// resolved/total counts when it ran.
type EvalResult struct {
	Available      bool     `json:"available"`
	Reason         string   `json:"reason,omitempty"`
	Resolved       int      `json:"resolved"`
	Total          int      `json:"total"`
	ResolveRatePct float64  `json:"resolve_rate_pct"`
	ResolvedIDs    []string `json:"resolved_ids,omitempty"`
	ReportPath     string   `json:"report_path,omitempty"`
	Command        string   `json:"command"` // the exact harness invocation (for reproduction / DGX runs)
}

var resolvedRe = regexp.MustCompile(`(?i)Resolved\s+(\d+)\s*/\s*(\d+)`)

// RunEval runs the official harness locally if it can, else returns a gated
// EvalResult. It always populates Command with the exact invocation so a DGX run
// is one copy-paste away even when this box can't grade.
func RunEval(cfg EvalConfig) (EvalResult, error) {
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = 4
	}
	if cfg.DatasetName == "" {
		cfg.DatasetName = "princeton-nlp/SWE-bench_Verified"
	}
	if cfg.RunID == "" {
		cfg.RunID = "fak-swebench"
	}
	cap := DetectEvalCapability(cfg.Python)
	python := cap.Python
	if python == "" {
		python = "python3"
	}
	// Normalize the predictions path to absolute ONCE. We cd the child into the
	// predictions dir (mirrors bench's `cd preds_dir`), so a relative
	// --predictions_path would be resolved against that dir a second time
	// (out/preds.json -> cd out -> open out/out/preds.json, file-not-found). An
	// absolute arg is unambiguous regardless of the child's cwd.
	absPreds, err := filepath.Abs(cfg.PredictionsPath)
	if err != nil {
		absPreds = cfg.PredictionsPath
	}
	predsDir := filepath.Dir(absPreds)
	cmdStr := fmt.Sprintf(
		"%s -m swebench.harness.run_evaluation --dataset_name %s --predictions_path %s --run_id %s --max_workers %d",
		python, cfg.DatasetName, absPreds, cfg.RunID, cfg.MaxWorkers)

	res := EvalResult{Command: cmdStr}
	if !cap.Runnable {
		res.Available = false
		res.Reason = cap.Reason
		return res, nil
	}

	cmd := exec.Command(python, "-m", "swebench.harness.run_evaluation",
		"--dataset_name", cfg.DatasetName,
		"--predictions_path", absPreds,
		"--run_id", cfg.RunID,
		"--max_workers", strconv.Itoa(cfg.MaxWorkers))
	// Run in the predictions dir so the harness's per-model summary lands beside
	// preds.json (mirrors bench's `cd preds_dir` in commands/_swebench_grade.py).
	cmd.Dir = predsDir

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return res, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return res, err
	}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	lastResolved, lastTotal := -1, -1
	for sc.Scan() {
		line := sc.Text()
		fmt.Fprintln(os.Stderr, line) // surface progress live, like bench's stream
		if m := resolvedRe.FindStringSubmatch(line); m != nil {
			lastResolved, _ = strconv.Atoi(m[1])
			lastTotal, _ = strconv.Atoi(m[2])
		}
	}
	waitErr := cmd.Wait()

	// Prefer the structured report.json; fall back to the parsed `Resolved X/Y`.
	reportPath := filepath.Join(predsDir, "logs", "run_evaluation", cfg.RunID, "report.json")
	if r, total, ids, ok := parseEvalReport(reportPath); ok {
		res.Available = true
		res.Resolved, res.Total, res.ResolvedIDs = r, total, ids
		res.ReportPath = reportPath
	} else if lastResolved >= 0 {
		res.Available = true
		res.Resolved, res.Total = lastResolved, lastTotal
	} else if waitErr != nil {
		return res, fmt.Errorf("harness run failed and produced no report: %w", waitErr)
	} else {
		// Ran cleanly but emitted neither a report.json nor a parseable
		// "Resolved X/Y" line — surface that instead of an empty gated Reason.
		res.Reason = "harness completed but produced no report.json and no parseable 'Resolved X/Y' line"
	}
	if res.Total > 0 {
		res.ResolveRatePct = 100 * float64(res.Resolved) / float64(res.Total)
	}
	return res, nil
}

// parseEvalReport reads the harness report.json defensively across its field-name
// variants (resolved_ids / resolved_instances; total_instances / total). Returns
// ok=false if the file is absent or unparseable.
func parseEvalReport(path string) (resolved, total int, ids []string, ok bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, nil, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return 0, 0, nil, false
	}
	if arr, n, found := firstRawCount(m, "resolved_ids", "resolved_instances", "resolved"); found {
		ids = arr
		resolved = n
		ok = true
	}
	// Denominator, mirroring bench's _count_report exactly:
	//   graded_total = report["total_instances"] or (len(resolved) + len(unresolved))
	// We do NOT use submitted_instances (it can exceed resolved+unresolved) and we
	// do NOT fabricate total=resolved — an absent total stays 0 so the caller's
	// `if Total > 0` guard keeps the rate at an honest 0%, never a false 100%.
	if n, found := firstRawInt(m, "total_instances", "total"); found {
		total = n
	}
	if total == 0 {
		unresolved := 0
		if _, n, found := firstRawCount(m, "unresolved_ids", "unresolved_instances"); found {
			unresolved = n
		}
		if resolved+unresolved > 0 {
			total = resolved + unresolved
		}
	}
	return resolved, total, ids, ok
}

// firstRawCount scans keys in order and returns the first present value as a
// (slice, count) pair. A value that decodes as a JSON string-array yields the
// array and its length; one that decodes as a bare int yields a nil slice and
// that int. found is false when no listed key is present and decodable.
func firstRawCount(m map[string]json.RawMessage, keys ...string) (ids []string, count int, found bool) {
	for _, k := range keys {
		raw, present := m[k]
		if !present {
			continue
		}
		var arr []string
		if json.Unmarshal(raw, &arr) == nil {
			return arr, len(arr), true
		}
		var n int
		if json.Unmarshal(raw, &n) == nil {
			return nil, n, true
		}
	}
	return nil, 0, false
}

// firstRawInt scans keys in order and returns the first present value that
// decodes as a bare int. found is false when no listed key is present and
// decodable.
func firstRawInt(m map[string]json.RawMessage, keys ...string) (int, bool) {
	for _, k := range keys {
		raw, present := m[k]
		if !present {
			continue
		}
		var n int
		if json.Unmarshal(raw, &n) == nil {
			return n, true
		}
	}
	return 0, false
}

// EvalCommandHint returns the copy-pasteable harness command for a predictions
// file, for running the resolve grade on a Docker-capable box (the DGX) when this
// box cannot. Mirrors the invocation bench uses.
func EvalCommandHint(predictionsPath, runID string, maxWorkers int) string {
	if runID == "" {
		runID = "fak-swebench"
	}
	if maxWorkers <= 0 {
		maxWorkers = 4
	}
	return fmt.Sprintf(
		"python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path %s --run_id %s --max_workers %d",
		predictionsPath, runID, maxWorkers)
}
