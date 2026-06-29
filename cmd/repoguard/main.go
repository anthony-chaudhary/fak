// Command repoguard refuses a DESTRUCTIVE or out-of-tree write before it escapes
// the repo — the Go port of tools/repo_guard.py, run as a single compiled binary
// so the Claude Code PreToolUse hook fires WITHOUT spawning a Python interpreter
// on every tool call (DIRECTION.md: the request path stays interpreter-free).
//
// Two surfaces, one pure core (guard.go):
//
//	repoguard --hook                 # PreToolUse hook: read the tool call as JSON on
//	                                 # stdin, emit a deny decision on a violation.
//	repoguard --check "<cmd>" --json # classify one Bash command (control-pane / CI).
//	repoguard --selftest             # run the built-in case table and exit.
//
// Fail-OPEN on any internal error (a guard bug must never wedge a live fleet).
// Soften with FAK_REPO_GUARD=warn (log, allow) or disable with FAK_REPO_GUARD=off.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

func main() {
	hook := flag.Bool("hook", false, "Claude Code PreToolUse hook mode (reads JSON on stdin)")
	selftest := flag.Bool("selftest", false, "run the built-in case table and exit")
	check := flag.String("check", "", "classify a single Bash command and report")
	workspace := flag.String("workspace", "", "workspace root (default: nearest .git above cwd)")
	asJSON := flag.Bool("json", false, "machine-readable output for --check")
	flag.Parse()

	switch {
	case *selftest:
		os.Exit(runSelftest(os.Stdout))
	case *hook:
		os.Exit(runHook(os.Stdin, os.Stdout, os.Stderr))
	case *check != "":
		os.Exit(runCheck(*check, *workspace, *asJSON, os.Stdout))
	default:
		flag.Usage()
		os.Exit(0)
	}
}

// hookSpecificOutput is the PreToolUse decision protocol payload.
type hookSpecificOutput struct {
	HookEventName            string `json:"hookEventName"`
	PermissionDecision       string `json:"permissionDecision"`
	PermissionDecisionReason string `json:"permissionDecisionReason"`
}

type hookDecision struct {
	HookSpecificOutput hookSpecificOutput `json:"hookSpecificOutput"`
}

type hookPayload struct {
	ToolName  string         `json:"tool_name"`
	ToolInput map[string]any `json:"tool_input"`
	Cwd       string         `json:"cwd"`
}

// runHook parses a PreToolUse payload and emits a deny decision on a violation.
// Fail-open on any error (defense-in-depth must never wedge the fleet). Always
// returns 0 — a deny is signalled through the JSON decision on stdout, not the
// exit code. Mirrors repo_guard.run_hook.
func runHook(stdin io.Reader, stdout, stderr io.Writer) int {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_REPO_GUARD")))
	if mode == "" {
		mode = "enforce"
	}
	if mode == "off" {
		return 0
	}
	violations, err := func() ([]repoguard.Violation, error) {
		raw, err := io.ReadAll(stdin)
		if err != nil {
			return nil, err
		}
		var payload hookPayload
		if len(strings.TrimSpace(string(raw))) == 0 {
			return nil, nil
		}
		if err := json.Unmarshal(raw, &payload); err != nil {
			return nil, err
		}
		cwd := payload.Cwd
		if cwd == "" {
			cwd, _ = os.Getwd()
		}
		workspaceRoot := repoguard.FindRepoRoot(cwd)
		safeRoots := repoguard.SafeRootsForWorkspace(workspaceRoot)
		return repoguard.Evaluate(payload.ToolName, payload.ToolInput, workspaceRoot, safeRoots), nil
	}()
	if err != nil {
		fmt.Fprintf(stderr, "repo_guard: internal error, allowing (%v)\n", err)
		return 0
	}
	if len(violations) == 0 {
		return 0
	}
	reason := repoguard.RenderReason(violations)
	if mode == "warn" {
		fmt.Fprintf(stderr, "repo_guard (advisory): %s\n", reason)
		return 0
	}
	// enforce: deny via the PreToolUse decision protocol. SetEscapeHTML(false) so the
	// reason reads `->` like the Python original, not the HTML-escaped `>`.
	enc := json.NewEncoder(stdout)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(hookDecision{HookSpecificOutput: hookSpecificOutput{
		HookEventName:            "PreToolUse",
		PermissionDecision:       "deny",
		PermissionDecisionReason: reason,
	}})
	fmt.Fprintf(stderr, "repo_guard: DENY %s\n", reason)
	return 0
}

// runCheck classifies a single Bash command and reports. Mirrors the --check arm
// of repo_guard.main: exit 1 iff there is at least one out-of-tree violation.
func runCheck(command, workspace string, asJSON bool, stdout io.Writer) int {
	ws := repoguard.FindRepoRoot(orCwd(workspace))
	safeRoots := repoguard.SafeRootsForWorkspace(ws)
	violations := repoguard.ClassifyCommand(command, ws, safeRoots)
	if violations == nil {
		violations = []repoguard.Violation{} // marshal as [] (matches the Python --json shape), never null
	}
	if asJSON {
		payload := struct {
			Schema     string                `json:"schema"`
			OK         bool                  `json:"ok"`
			Workspace  string                `json:"workspace"`
			Violations []repoguard.Violation `json:"violations"`
		}{repoguard.Schema, len(violations) == 0, ws, violations}
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		_ = enc.Encode(payload)
	} else if len(violations) > 0 {
		fmt.Fprintf(stdout, "DENY  %s\n", repoguard.RenderReason(violations))
	} else {
		fmt.Fprintf(stdout, "ALLOW  no out-of-tree write in: %s\n", command)
	}
	if len(violations) > 0 {
		return 1
	}
	return 0
}

func orCwd(workspace string) string {
	if workspace != "" {
		return workspace
	}
	cwd, _ := os.Getwd()
	return cwd
}

// runSelftest runs the built-in deny/allow case table. Mirrors repo_guard._selftest:
// exit 1 iff any case mis-classifies. The fixtures (WS/HOME/SAFE) are the exact
// roots the production hook composes, so this proves the classifier end to end
// with no filesystem.
func runSelftest(stdout io.Writer) int {
	const ws = "C:/Users/u/work/fak"
	const home = "C:/Users/u"
	safe := []string{"/tmp", "/var/tmp", "C:/Users/u/.cache", "C:/Users/u/Downloads"}
	safe = append(safe, repoguard.AgentStateRoots(home, []string{".claude", ".claude-gem8-netra", ".claudex", "Documents"})...)
	safe = append(safe, repoguard.PrivateCompanionRoots(ws)...)

	type tc struct {
		tool  string
		input map[string]any
	}
	deny := []tc{
		{"Bash", cmd("go build -o ../tools/.bin/fak.exe ./cmd/fak")},
		{"Bash", cmd("rm -rf ../tools")},
		{"Bash", cmd("rm -rf /c/Users/u/work/tools")},
		{"Bash", cmd("echo x > ../tools/y")},
		{"Bash", cmd("cp a.txt ../tools/b.txt")},
		{"Bash", cmd("mv internal/x ../sibling/x")},
		{"Bash", cmd("rm -rf /")},
		{"Bash", cmd("cd src && rm -rf ../../other")},
		{"Write", fp("../tools/poison.txt")},
		{"Write", fp("C:/Users/u/work/tools/poison.txt")},
		{"Write", fp("C:/Users/u/work/fak-private-evil/x.md")},
		{"Write", fp("C:/Users/u/work/fak-ci/x.md")},
		{"Write", fp("C:/Users/u/.claudex/leak.md")},
	}
	allow := []tc{
		{"Bash", cmd("go build -o fak.exe ./cmd/fak")},
		{"Bash", cmd("go build -o tools/.bin/fak.exe ./cmd/fak")},
		{"Bash", cmd("rm -rf ./build")},
		{"Bash", cmd("rm -rf internal/model/.cache")},
		{"Bash", cmd("echo x > /tmp/log.txt")},
		{"Bash", cmd("cp a.txt /var/tmp/b.txt")},
		{"Bash", cmd("cp a.txt ~/.cache/b.txt")},
		{"Bash", cmd("grep -o ../foo internal/policy/x.go")},
		{"Bash", cmd("cat ../README.md")},
		{"Bash", cmd("mv internal/a internal/b")},
		{"Write", fp("internal/policy/x.go")},
		{"Write", fp("examples/repo-guard-policy.json")},
		{"Write", fp("C:/Users/u/.claude-gem8-netra/projects/C--Users-u-work-fak/memory/note.md")},
		{"Write", fp("C:/Users/u/work/fak-private/MEMORY-glm52-2026-06-21.md")},
		// the null / std-stream device sinks: harmless, never a sibling repo.
		{"Bash", cmd("make ci > /dev/null 2>&1")},
		{"Bash", cmd("go test ./... > /dev/null")},
		{"Bash", cmd("echo done >> /dev/stderr")},
	}

	fails := 0
	for _, c := range deny {
		if len(repoguard.Evaluate(c.tool, c.input, ws, safe)) == 0 {
			fmt.Fprintf(stdout, "  FAIL (expected DENY, got allow): %s %v\n", c.tool, c.input)
			fails++
		}
	}
	for _, c := range allow {
		if v := repoguard.Evaluate(c.tool, c.input, ws, safe); len(v) > 0 {
			fmt.Fprintf(stdout, "  FAIL (expected ALLOW, got %v): %s %v\n", v, c.tool, c.input)
			fails++
		}
	}
	total := len(deny) + len(allow)
	fmt.Fprintf(stdout, "repo_guard selftest: %d/%d passed (%d deny, %d allow)\n", total-fails, total, len(deny), len(allow))
	if fails > 0 {
		return 1
	}
	return 0
}

func cmd(c string) map[string]any { return map[string]any{"command": c} }
func fp(p string) map[string]any  { return map[string]any{"file_path": p} }
