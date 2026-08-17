package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

const codexToolRouterTarget = "codex_core::tools::router"

var (
	codexToolNameRE  = regexp.MustCompile(`tool_name=(?:"([^"]+)"|([^ ]+))`)
	codexToolErrorRE = regexp.MustCompile(`dispatch_tool_call_with_terminal_outcome: error=([^\r\n]+)`)
)

type codexToolErrorSummary struct {
	Total          int
	OutcomeErrors  int
	ContractErrors int
	HookErrors     int
	OtherErrors    int
	Categories     map[string]int
}

func RunCodexToolErrors(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-tool-errors", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	codexHome := fs.String("codex-home", "", "Codex home containing logs_2.sqlite")
	since := fs.Duration("since", 24*time.Hour, "lookback window")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-tool-errors [--codex-home DIR] [--since 24h]")
		return 2
	}
	return runCodexToolErrorDiagnosis(stdout, stderr, *codexHome, *since, time.Now())
}

func runCodexToolErrorDiagnosis(stdout, stderr io.Writer, codexHome string, since time.Duration, now time.Time) int {
	if strings.TrimSpace(codexHome) == "" {
		codexHome = strings.TrimSpace(os.Getenv("CODEX_HOME"))
	}
	if codexHome == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintf(stderr, "fak sessiondiag: resolve Codex home: %v\n", err)
			return 1
		}
		codexHome = filepath.Join(home, ".codex")
	}
	dbPath := filepath.Join(codexHome, "logs_2.sqlite")
	if _, err := os.Stat(dbPath); err != nil {
		fmt.Fprintf(stderr, "fak sessiondiag: read Codex log: %v\n", err)
		return 1
	}
	summary, err := queryCodexToolErrors(dbPath, now.Add(-since))
	if err != nil {
		fmt.Fprintf(stderr, "fak sessiondiag: query Codex log: %v\n", err)
		return 1
	}
	writeCodexToolErrorDiagnosis(stdout, codexHome, since, summary)
	return 0
}

func queryCodexToolErrors(path string, after time.Time) (codexToolErrorSummary, error) {
	summary := codexToolErrorSummary{Categories: make(map[string]int)}
	python, err := exec.LookPath("python")
	if err != nil && runtime.GOOS != "windows" {
		python, err = exec.LookPath("python3")
	}
	if err != nil {
		return summary, errors.New("Python sqlite reader not found")
	}
	script := `import json,sqlite3,sys
p=sys.argv[1]; uri='file:'+p.replace('\\','/')+'?mode=ro&immutable=0'
c=sqlite3.connect(uri,uri=True,timeout=5); c.execute('pragma query_only=on')
rows=c.execute("select target,coalesce(feedback_log_body,'') from logs where ts>=? and level='ERROR' order by id",(int(sys.argv[2]),)).fetchall()
print(json.dumps(rows))`
	cmd := exec.Command(python, "-c", script, path, fmt.Sprint(after.Unix()))
	var out, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &stderr
	if err := cmd.Run(); err != nil {
		return summary, fmt.Errorf("read-only Codex error query failed: %s", redactDiagError(stderr.String()))
	}
	var rows [][2]string
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		return summary, fmt.Errorf("decode Codex error query: %w", err)
	}
	for _, row := range rows {
		summary.Total++
		classifyCodexToolError(&summary, row[0], row[1])
	}
	return summary, nil
}

func classifyCodexToolError(summary *codexToolErrorSummary, target, body string) {
	lower := strings.ToLower(body)
	if strings.Contains(strings.ToLower(target), "hook") && (strings.Contains(lower, "posttool") || strings.Contains(lower, "post_tool_use") || strings.Contains(lower, "post-tool")) {
		summary.HookErrors++
		summary.Categories["post-tool hook failure"]++
		return
	}
	if target != codexToolRouterTarget && !strings.Contains(body, "dispatch_tool_call_with_terminal_outcome") {
		summary.OtherErrors++
		summary.Categories["other Codex error"]++
		return
	}
	tool := "tool"
	if matches := codexToolNameRE.FindAllStringSubmatch(body, -1); len(matches) > 0 {
		last := matches[len(matches)-1]
		tool = last[1]
		if tool == "" {
			tool = last[2]
		}
	}
	outcome := "non-success outcome"
	if matches := codexToolErrorRE.FindAllStringSubmatch(body, -1); len(matches) > 0 {
		outcome = strings.TrimSpace(matches[len(matches)-1][1])
	}
	if strings.Contains(strings.ToLower(outcome), "invalid arguments") {
		summary.ContractErrors++
		summary.Categories[tool+": invalid arguments"]++
		return
	}
	summary.OutcomeErrors++
	switch {
	case strings.HasPrefix(outcome, "Exit code:"):
		summary.Categories[tool+": "+outcome]++
	case strings.Contains(strings.ToLower(outcome), "timed out"):
		summary.Categories[tool+": timeout"]++
	default:
		summary.Categories[tool+": "+outcome]++
	}
}

func writeCodexToolErrorDiagnosis(w io.Writer, codexHome string, since time.Duration, s codexToolErrorSummary) {
	fmt.Fprintf(w, "Codex tool-error diagnosis (%s, %s)\n", durationLabel(since), codexHome)
	fmt.Fprintf(w, "  logged ERROR rows : %d\n", s.Total)
	fmt.Fprintf(w, "  tool outcomes     : %d (commands/tools returned non-success; Codex currently logs these at ERROR)\n", s.OutcomeErrors)
	fmt.Fprintf(w, "  contract defects  : %d (invalid tool arguments; prevent before dispatch)\n", s.ContractErrors)
	fmt.Fprintf(w, "  post-tool hooks   : %d (actual hook failures)\n", s.HookErrors)
	fmt.Fprintf(w, "  other errors      : %d\n", s.OtherErrors)
	keys := make([]string, 0, len(s.Categories))
	for key := range s.Categories {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(w, "    %4d  %s\n", s.Categories[key], key)
	}
	switch {
	case s.HookErrors > 0:
		fmt.Fprintln(w, "verdict: HOOK_FAILURES_PRESENT — inspect the named hook before retrying")
	case s.ContractErrors > 0:
		fmt.Fprintln(w, "verdict: SHIFT_LEFT — reject malformed tool calls before execution")
	case s.OutcomeErrors > 0:
		fmt.Fprintln(w, "verdict: OUTCOMES_NOT_HOOKS — recurring red rows are post-call command outcomes, not PostTool hook failures")
	default:
		fmt.Fprintln(w, "verdict: CLEAN — no Codex ERROR rows in this window")
	}
}

func durationLabel(d time.Duration) string {
	if d <= 0 {
		return "all"
	}
	return d.Round(time.Second).String()
}
