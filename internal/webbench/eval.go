package webbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

// evalHarnessTimeout bounds a single RunEval harness invocation. The harness
// drives real browsers over the network, so it gets a generous ceiling; a
// wedged run degrades to a timeout verdict instead of hanging forever. It is a
// var (not a const) so tests can drive the timeout path with a tiny deadline.
var evalHarnessTimeout = 15 * time.Minute

// runBoundedHarness runs the harness command under a deadline. CommandContext
// kills the child when the context expires and WaitDelay bounds the lingering
// wait on the child's I/O pipes, so a wedged browser run cannot block the
// caller indefinitely. On expiry it returns an error wrapping
// context.DeadlineExceeded (recognisable via errors.Is).
func runBoundedHarness(timeout time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.WaitDelay = 10 * time.Second
	out, err := cmd.CombinedOutput()
	if err != nil && ctx.Err() == context.DeadlineExceeded {
		return out, fmt.Errorf("harness timed out after %s: %w", timeout, ctx.Err())
	}
	return out, err
}

// EvalCapability reports whether the official web benchmark harness can run on this box.
// For web benchmarks, this typically means a browser runtime (Playwright, Selenium) and
// the benchmark's evaluation scripts.
type EvalCapability struct {
	BrowserPresent bool   `json:"browser_present"` // playwright, selenium, etc.
	HarnessPresent bool   `json:"harness_present"` // benchmark's eval scripts
	Python         string `json:"python"`          // resolved python interpreter
	Runnable       bool   `json:"runnable"`
	Reason         string `json:"reason"`
}

// DetectEvalCapability probes for browser + harness availability.
func DetectEvalCapability(python string) EvalCapability {
	cap := EvalCapability{}
	for _, cand := range pythonCandidates(python) {
		if _, err := exec.LookPath(cand); err != nil {
			continue
		}
		cap.Python = cand
		// Check for browser automation libraries.
		if err := exec.Command(cand, "-c", "import playwright").Run(); err == nil {
			cap.BrowserPresent = true
		} else if err := exec.Command(cand, "-c", "import selenium").Run(); err == nil {
			cap.BrowserPresent = true
		}
		// Check for common harness modules (browser-use, webvoyager, etc).
		if err := exec.Command(cand, "-c", "import browser_use").Run(); err == nil {
			cap.HarnessPresent = true
			break
		}
	}
	cap.Runnable = cap.BrowserPresent && cap.HarnessPresent
	switch {
	case !cap.BrowserPresent && !cap.HarnessPresent:
		cap.Reason = "no browser runtime and no harness - task success is a harness-only metric on this box"
	case !cap.BrowserPresent:
		cap.Reason = "browser runtime not found - install playwright or selenium"
	case !cap.HarnessPresent:
		cap.Reason = "benchmark harness not found - install the benchmark's eval package"
	}
	return cap
}

func pythonCandidates(python string) []string {
	if python != "" {
		return []string{python}
	}
	return []string{"python3", "python"}
}

// EvalConfig drives a local success-rate run.
type EvalConfig struct {
	PredictionsPath string // path to predictions.json (agent trajectories)
	Benchmark       string // "browser-agent", "webvoyager", etc.
	RunID           string // names the harness output dir
	MaxWorkers      int    // harness parallelism
	Python          string // interpreter
}

// EvalResult is the success-rate outcome.
type EvalResult struct {
	Available      bool     `json:"available"`
	Reason         string   `json:"reason,omitempty"`
	Passed         int      `json:"passed"`
	Total          int      `json:"total"`
	SuccessRatePct float64  `json:"success_rate_pct"`
	PassedIDs      []string `json:"passed_ids,omitempty"`
	ReportPath     string   `json:"report_path,omitempty"`
	Command        string   `json:"command"` // exact harness invocation for reproduction
}

var successRe = regexp.MustCompile(`(?i)passed\s+(\d+)\s*/\s*(\d+)`)

// RunEval runs the official harness locally if it can, else returns a gated result.
func RunEval(cfg EvalConfig) (EvalResult, error) {
	res := EvalResult{
		Command: buildHarnessCommand(cfg),
	}

	cap := DetectEvalCapability(cfg.Python)
	if !cap.Runnable {
		res.Available = false
		res.Reason = cap.Reason
		return res, nil
	}

	// Run the harness - actual invocation depends on benchmark.
	// For browser-agent benchmark, invoke the evaluation script.
	harnessCmd := buildHarnessCommand(cfg)
	cmdParts := strings.Fields(harnessCmd)
	if len(cmdParts) < 2 {
		return res, fmt.Errorf("invalid harness command: %s", harnessCmd)
	}
	// Bounded: the harness drives real browsers over the network, which can wedge
	// (a hung page, a stalled fetch) with no natural deadline.
	output, err := runBoundedHarness(evalHarnessTimeout, cmdParts[0], cmdParts[1:]...)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return res, err
		}
		return res, fmt.Errorf("harness failed: %w: %s", err, string(output))
	}

	// Parse output.
	matches := successRe.FindStringSubmatch(string(output))
	if len(matches) < 3 {
		return res, fmt.Errorf("harness output did not match expected pattern")
	}
	res.Passed = parseInt(matches[1])
	res.Total = parseInt(matches[2])
	if res.Total > 0 {
		res.SuccessRatePct = 100.0 * float64(res.Passed) / float64(res.Total)
	}
	res.Available = true
	return res, nil
}

func buildHarnessCommand(cfg EvalConfig) string {
	py := cfg.Python
	if py == "" {
		py = "python3"
	}
	// Build actual harness command based on benchmark type.
	switch cfg.Benchmark {
	case "browser-agent", "webvoyager":
		// Use browser-use evaluation harness
		return fmt.Sprintf("%s -m browser_use.eval --predictions %s --run_id %s --max_workers %d",
			py, cfg.PredictionsPath, cfg.RunID, cfg.MaxWorkers)
	default:
		// Generic harness invocation
		return fmt.Sprintf("%s -m %s.eval --predictions %s --run_id %s",
			py, cfg.Benchmark, cfg.PredictionsPath, cfg.RunID)
	}
}

func parseInt(s string) int {
	var n int
	for _, r := range s {
		if r >= '0' && r <= '9' {
			n = n*10 + int(r-'0')
		}
	}
	return n
}

// DescribeToFile writes a summary to JSON for later consumption.
func DescribeToFile(s Summary, path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadSummary reads a previously-written summary.
func LoadSummary(path string) (Summary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Summary{}, err
	}
	var s Summary
	if err := json.Unmarshal(data, &s); err != nil {
		return Summary{}, err
	}
	return s, nil
}
