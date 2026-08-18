package devcmd

import (
	"bytes"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestClassifyCodexToolErrorsSeparatesOutcomesContractsAndHooks(t *testing.T) {
	got := codexToolErrorSummary{Categories: make(map[string]int)}
	for _, row := range [][2]string{
		{codexToolRouterTarget, `tool_name="shell_command":dispatch_tool_call_with_terminal_outcome: error=Exit code: 1`},
		{codexToolRouterTarget, `tool_name="shell_command":dispatch_tool_call_with_terminal_outcome: error=Invalid arguments: missing field command`},
		{"codex_hooks", `PostToolUse hook failed: exit 1`},
		{"transport", `transport disconnected`},
	} {
		got.Total++
		classifyCodexToolError(&got, row[0], row[1])
	}
	if got.Total != 4 || got.OutcomeErrors != 1 || got.ContractErrors != 1 || got.HookErrors != 1 || got.OtherErrors != 1 {
		t.Fatalf("summary = %+v", got)
	}
	var out bytes.Buffer
	writeCodexToolErrorDiagnosis(&out, `C:\Codex`, 24*time.Hour, got)
	text := out.String()
	for _, want := range []string{"tool outcomes     : 1", "contract defects  : 1", "post-tool hooks   : 1", "HOOK_FAILURES_PRESENT"} {
		if !strings.Contains(text, want) {
			t.Fatalf("output missing %q:\n%s", want, text)
		}
	}
}

func TestWriteCodexToolErrorDiagnosisNamesOutcomeOnlyRecurrence(t *testing.T) {
	var out bytes.Buffer
	writeCodexToolErrorDiagnosis(&out, "home", time.Hour, codexToolErrorSummary{
		Total: 9, OutcomeErrors: 9, Categories: map[string]int{"shell_command: Exit code: 1": 9},
	})
	if !strings.Contains(out.String(), "OUTCOMES_NOT_HOOKS") {
		t.Fatalf("output = %s", out.String())
	}
}

func TestQueryCodexToolErrorsForThreadFiltersTheGateNumerator(t *testing.T) {
	python, err := exec.LookPath("python")
	if err != nil {
		t.Skip("python sqlite fixture unavailable")
	}
	path := filepath.Join(t.TempDir(), "logs.sqlite")
	script := `import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); c.execute("create table logs(id integer primary key,ts integer,level text,target text,feedback_log_body text,thread_id text)")
rows=[(1,200,"ERROR","codex_core::tools::router","tool_name=shell_command dispatch_tool_call_with_terminal_outcome: error=Exit code: 1","thread-a"),(2,200,"ERROR","codex_core::tools::router","tool_name=shell_command dispatch_tool_call_with_terminal_outcome: error=Exit code: 1","thread-b")]
c.executemany("insert into logs values(?,?,?,?,?,?)",rows); c.commit()`
	if out, err := exec.Command(python, "-c", script, path).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v: %s", err, out)
	}
	got, err := queryCodexToolErrorsForThread(path, time.Unix(100, 0), "thread-a")
	if err != nil {
		t.Fatal(err)
	}
	if got.Total != 1 || got.OutcomeErrors != 1 {
		t.Fatalf("summary=%+v", got)
	}
}
