package scorecardpane

// collect.go — the impure shell around the control-pane fold: run each scorecard
// as a subprocess, parse its --json, and fold. Ported from the Python collect() /
// run_scorecard / load_baseline / head_commit. The pure surface (Fold, ComputeTrend,
// CheckGate) is tested directly over fixture payloads; this shell is the one process
// startup the issue asks for (Go shells the python/go cards instead of the python
// tool re-launching python for each card).

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// RunScorecard runs one card and returns its parsed payload or an error string,
// mirroring the Python run_scorecard: a card with a Cmd is shlex-split and run; a
// script card runs `python <tools/script> --json`; a missing script reports the
// "missing scorecard" error; non-JSON stdout reports the exit + last error line.
func RunScorecard(root string, card Card, python string, timeout time.Duration) (map[string]any, string) {
	var argv []string
	if card.Cmd != "" {
		argv = strings.Fields(card.Cmd)
	} else {
		scriptPath := filepath.Join(root, "tools", card.Script)
		if _, err := os.Stat(scriptPath); err != nil {
			return nil, "missing scorecard: tools/" + card.Script
		}
		argv = []string{python, scriptPath, "--json"}
	}
	if len(argv) == 0 {
		return nil, "empty command"
	}
	argv = rewriteGoRunFak(argv, os.Args[0])

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, "timed out"
	}

	var payload map[string]any
	if jErr := json.Unmarshal([]byte(stdout.String()), &payload); jErr != nil {
		tail := lastLine(stderr.String())
		if tail == "" {
			tail = lastLine(stdout.String())
		}
		code := 0
		if exitErr, ok := err.(*exec.ExitError); ok {
			code = exitErr.ExitCode()
		}
		if len(tail) > 160 {
			tail = tail[:160]
		}
		return nil, "non-JSON output (exit " + itoa(code) + "): " + tail
	}
	return payload, ""
}

// Collect runs every card and folds nothing — it returns the per-card Metric rows,
// matching the Python collect(). The default python is the PATH "python".
func Collect(root, python string, timeout time.Duration) []Metric {
	if python == "" {
		python = defaultPython()
	}
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	metrics := make([]Metric, 0, len(Cards))
	for _, card := range Cards {
		payload, errMsg := RunScorecard(root, card, python, timeout)
		metrics = append(metrics, MetricFromPayload(card, payload, errMsg))
	}
	return metrics
}

func CollectBudgeted(root, python string, timeout, remaining time.Duration) []Metric {
	if remaining <= 0 {
		return budgetExhaustedMetrics()
	}
	if timeout > remaining {
		timeout = remaining
	}
	return Collect(root, python, timeout)
}

func CollectBudgetedParallel(root, python string, timeout, remaining time.Duration, workers int) []Metric {
	if remaining <= 0 {
		return budgetExhaustedMetrics()
	}
	return CollectBudgeted(root, python, timeout, remaining)
}

func budgetExhaustedMetrics() []Metric {
	out := make([]Metric, 0, len(Cards))
	for _, card := range Cards {
		out = append(out, MetricFromPayload(card, nil, "budget exhausted before running card"))
	}
	return out
}

// LoadBaseline reads the pinned baseline file, or returns nil when absent/unreadable
// (matching the Python load_baseline returning None — an unpinned trend).
func LoadBaseline(path string) *Baseline {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var doc Baseline
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil
	}
	return &doc
}

// HeadCommitShort returns the short HEAD sha for the workspace, or "unknown".
// Ported from the Python head_commit.
func HeadCommitShort(root string) string {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "unknown"
	}
	return s
}

func defaultPython() string {
	for _, c := range []string{"python3", "python"} {
		if _, err := exec.LookPath(c); err == nil {
			return c
		}
	}
	return "python"
}

func rewriteGoRunFak(argv []string, fakBin string) []string {
	if len(argv) < 3 || argv[0] != "go" || argv[1] != "run" || !isCmdFakTarget(argv[2]) {
		return argv
	}
	if strings.TrimSpace(fakBin) == "" {
		return argv
	}
	out := make([]string, 0, len(argv)-2)
	out = append(out, fakBin)
	out = append(out, argv[3:]...)
	return out
}

func isCmdFakTarget(target string) bool {
	norm := filepath.ToSlash(strings.ReplaceAll(strings.TrimSpace(target), `\`, "/"))
	for strings.Contains(norm, "//") {
		norm = strings.ReplaceAll(norm, "//", "/")
	}
	norm = strings.TrimPrefix(norm, "./")
	return norm == "cmd/fak"
}

func lastLine(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
