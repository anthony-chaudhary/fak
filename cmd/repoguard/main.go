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
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/repoguard"
)

func main() {
	hook := flag.Bool("hook", false, "Claude Code PreToolUse hook mode (reads JSON on stdin)")
	selftest := flag.Bool("selftest", false, "run the built-in case table and exit")
	check := flag.String("check", "", "classify a single Bash command and report")
	workspace := flag.String("workspace", "", "workspace root (default: nearest .git above cwd)")
	asJSON := flag.Bool("json", false, "machine-readable output for --check / --summary")
	summary := flag.Bool("summary", false, "read the decision journal and show the accumulated guard value")
	recentN := flag.Int("recent", 10, "how many recent findings to list under --summary")
	flag.Parse()

	switch {
	case *selftest:
		os.Exit(runSelftest(os.Stdout))
	case *hook:
		os.Exit(runHook(os.Stdin, os.Stdout, os.Stderr))
	case *summary:
		os.Exit(runSummary(*workspace, *recentN, *asJSON, os.Stdout))
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
	SessionID string         `json:"session_id"`
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
	var payload hookPayload
	var workspaceRoot string
	raw, err := io.ReadAll(stdin)
	if err != nil {
		fmt.Fprintf(stderr, "repo_guard: internal error, allowing (%v)\n", err)
		return 0
	}
	if len(strings.TrimSpace(string(raw))) == 0 {
		return 0
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		cwd, _ := os.Getwd()
		workspaceRoot = repoguard.FindRepoRoot(cwd)
		recordMalformedPayload(workspaceRoot, err, stderr)
		fmt.Fprintf(stderr, "repo_guard: malformed hook payload, allowing (%v)\n", err)
		return 0
	}
	cwd := payload.Cwd
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	workspaceRoot = repoguard.FindRepoRoot(cwd)
	safeRoots := repoguard.SafeRootsForWorkspace(workspaceRoot)
	hints := repoguard.Hints{
		LiveMonitorIDs:   liveMonitorIDsForRead(payload, workspaceRoot, stderr),
		LeafDeclarations: leafDeclarationsForWrite(payload, workspaceRoot),
	}
	violations := repoguard.EvaluateWithHints(payload.ToolName, payload.ToolInput, workspaceRoot, safeRoots, hints)
	if len(violations) == 0 {
		return 0
	}
	// Every classified violation resolves to a per-reason severity (default
	// permissive; the hard blocks are an opt-in dial). The master switch FAK_REPO_GUARD
	// still wins: "off" skips all, "warn" CAPS every rung at advisory. Bucket the
	// findings by resolved severity — the decision is per reason, not per rung.
	overrides := severityOverridesFromEnv()
	severityOf := func(reason string) repoguard.Severity {
		return repoguard.ResolveSeverity(reason, overrides, mode)
	}
	var denying, advisory, recorded, acted []repoguard.Violation
	for _, v := range violations {
		switch severityOf(v.Reason) {
		case repoguard.SeverityOff:
			// dropped entirely: no record, no stderr, no deny
		case repoguard.SeverityRecord:
			recorded = append(recorded, v)
			acted = append(acted, v)
		case repoguard.SeverityWarn:
			advisory = append(advisory, v)
			acted = append(acted, v)
		case repoguard.SeverityDeny:
			denying = append(denying, v)
			acted = append(acted, v)
		}
	}
	// Advisory findings surface on stderr (the fix-hint is the value); record-level
	// findings are SILENT — journal only, nothing enters the model's context.
	if len(advisory) > 0 {
		fmt.Fprintf(stderr, "repo_guard (advisory): %s\n", repoguard.RenderReason(advisory))
	}
	// Durably record every finding we act on — deny, advisory, AND silent record —
	// so the value the guard delivered in a long session is a fact on disk, not an
	// ephemeral stderr line the harness scrolls away (and, for record-level, the
	// only trace at all). Fail-open: a journal error is logged and swallowed, never
	// allowed to change the decision below.
	recordDecisions(payload, workspaceRoot, mode, acted, severityOf, stderr)
	if len(denying) == 0 {
		return 0
	}
	reason := repoguard.RenderReason(denying)
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

func recordMalformedPayload(workspaceRoot string, parseErr error, stderr io.Writer) {
	row := repoguard.DecisionRecord{
		Schema:   repoguard.DecisionRecordSchema,
		Ts:       time.Now().UTC().Format(time.RFC3339),
		Tool:     "PreToolUse",
		Decision: "record",
		Mode:     "enforce",
		Reason:   "MALFORMED_HOOK_PAYLOAD",
		Why:      parseErr.Error(),
	}
	if err := repoguard.AppendDecisions(repoguard.DecisionJournalPath(workspaceRoot), []repoguard.DecisionRecord{row}); err != nil {
		fmt.Fprintf(stderr, "repo_guard: malformed-payload journal write failed, continuing (%v)\n", err)
	}
}

// recordDecisions appends the guard's findings to the durable decision journal.
// Best-effort and fail-open: any error is logged to stderr and swallowed so the
// hook's decision is never affected by a journal problem. The timestamp is taken
// here (the command layer owns the clock; the pure core stays deterministic).
func recordDecisions(payload hookPayload, workspaceRoot, mode string, violations []repoguard.Violation, severityOf func(reason string) repoguard.Severity, stderr io.Writer) {
	if len(violations) == 0 {
		return
	}
	if workspaceRoot == "" {
		return
	}
	ts := time.Now().UTC().Format(time.RFC3339)
	rows := repoguard.DecisionsFromViolations(violations, payload.ToolName, payload.SessionID, mode, ts, severityOf)
	path := repoguard.DecisionJournalPath(workspaceRoot)
	if err := repoguard.AppendDecisions(path, rows); err != nil {
		// Fail-open on a journal error — but stay quiet if EVERY finding in this
		// batch is silent-record: a record-level finding contributes nothing to the
		// model's context by design, and a leaked "write failed" notice on stderr
		// would break exactly that contract. When some finding already surfaced
		// (advisory/deny), the operator is seeing stderr anyway, so the notice helps.
		nonSilent := false
		for _, v := range violations {
			if severityOf(v.Reason) > repoguard.SeverityRecord {
				nonSilent = true
				break
			}
		}
		if nonSilent {
			fmt.Fprintf(stderr, "repo_guard: decision journal write failed, continuing (%v)\n", err)
		}
	}
}

// severityOverridesFromEnv reads the per-reason severity dial
// (FAK_REPO_GUARD_SEVERITY=REASON=level,REASON=level). Env reads live in the
// command layer; the pure core parses the spec. A blank/absent var means no
// overrides — the default posture applies.
func severityOverridesFromEnv() map[string]repoguard.Severity {
	return repoguard.ParseSeverityOverrides(os.Getenv("FAK_REPO_GUARD_SEVERITY"))
}

func liveMonitorIDsForRead(payload hookPayload, workspaceRoot string, stderr io.Writer) map[string]bool {
	if payload.ToolName != "Read" {
		return nil
	}
	if _, ok := repoguard.LiveMonitorOutputTaskID(hookReadPath(payload.ToolInput)); !ok {
		return nil
	}
	ids, err := repoguard.LiveMonitorTaskIDsFromJournalFile(repoGuardToolprocJournalPath(workspaceRoot), payload.SessionID)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Fprintf(stderr, "repo_guard: live Monitor hint unavailable (%v)\n", err)
		}
		return nil
	}
	return ids
}

// leafDeclarationsForWrite loads the lane/tier taxonomy for the UNDECLARED_LEAF
// rung. Gated twice so the hook stays cheap on the tool calls that can never
// trip it: only Write-class tools, and only when the path actually lands in an
// internal/<leaf> tree — every other call reads no files at all.
func leafDeclarationsForWrite(payload hookPayload, workspaceRoot string) repoguard.LeafDeclarations {
	switch payload.ToolName {
	case "Write", "Edit", "MultiEdit", "NotebookEdit":
	default:
		return repoguard.LeafDeclarations{}
	}
	fp := hookWritePath(payload.ToolInput)
	if _, ok := repoguard.LeafForWritePath(fp, workspaceRoot); !ok {
		return repoguard.LeafDeclarations{}
	}
	return repoguard.LeafDeclarationsForWorkspace(workspaceRoot)
}

func hookWritePath(input map[string]any) string {
	if v, ok := input["file_path"].(string); ok && v != "" {
		return v
	}
	if v, ok := input["notebook_path"].(string); ok {
		return v
	}
	return ""
}

func repoGuardToolprocJournalPath(workspaceRoot string) string {
	if p := strings.TrimSpace(os.Getenv("FAK_REPO_GUARD_TOOLPROC_JOURNAL")); p != "" {
		return p
	}
	if p := strings.TrimSpace(os.Getenv("FAK_TOOLPROC_JOURNAL")); p != "" {
		return p
	}
	return filepath.Join(workspaceRoot, ".fak", "toolproc", "journal.jsonl")
}

func hookReadPath(input map[string]any) string {
	if v, ok := input["file_path"].(string); ok && v != "" {
		return v
	}
	if v, ok := input["path"].(string); ok {
		return v
	}
	return ""
}

// runCheck classifies a single Bash command and reports. Mirrors the --check arm
// of repo_guard.main: exit 1 iff there is at least one out-of-tree violation.
func runCheck(command, workspace string, asJSON bool, stdout io.Writer) int {
	ws := repoguard.FindRepoRoot(orCwd(workspace))
	safeRoots := repoguard.SafeRootsForWorkspace(ws)
	violations := repoguard.ClassifyCommand(command, ws, safeRoots)
	violations = append(violations, repoguard.ClassifyInteractive(command)...)
	violations = append(violations, repoguard.ClassifySleepWait(command)...)
	violations = append(violations, repoguard.ClassifyForegroundNetworkLoop(command)...)
	violations = append(violations, repoguard.ClassifyForegroundPowerShellInventory(command)...)
	if violations == nil {
		violations = []repoguard.Violation{} // marshal as [] (matches the Python --json shape), never null
	}
	// Resolve each finding's severity through the SAME dial the live hook uses, so
	// --check mirrors production: ok and the exit code track deny-level findings
	// only; record/warn ride along in the JSON but never fail the check.
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_REPO_GUARD")))
	if mode == "" {
		mode = "enforce"
	}
	overrides := severityOverridesFromEnv()
	denying := 0
	for _, v := range violations {
		if repoguard.ResolveSeverity(v.Reason, overrides, mode) == repoguard.SeverityDeny {
			denying++
		}
	}
	if asJSON {
		payload := struct {
			Schema     string                `json:"schema"`
			OK         bool                  `json:"ok"`
			Workspace  string                `json:"workspace"`
			Violations []repoguard.Violation `json:"violations"`
		}{repoguard.Schema, denying == 0, ws, violations}
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		_ = enc.Encode(payload)
	} else if denying > 0 {
		fmt.Fprintf(stdout, "DENY  %s\n", repoguard.RenderReason(violations))
	} else if len(violations) > 0 {
		// Non-deny findings (record or warn under the current dial) still classified —
		// report them, but the check passes.
		fmt.Fprintf(stdout, "WARN  %s\n", repoguard.RenderReason(violations))
	} else {
		fmt.Fprintf(stdout, "ALLOW  no out-of-tree write or would-hang interactive form in: %s\n", command)
	}
	if denying > 0 {
		return 1
	}
	return 0
}

// runSummary reads the decision journal and prints the accumulated guard value:
// what the guard denied, by reason, over the life of the journal. This is the
// "show that value" half — a live session's saves become a countable fact. A
// missing journal is a clean empty summary, not an error.
func runSummary(workspace string, recentN int, asJSON bool, stdout io.Writer) int {
	ws := repoguard.FindRepoRoot(orCwd(workspace))
	path := repoguard.DecisionJournalPath(ws)
	sum, err := repoguard.SummarizeDecisionsFile(path, recentN)
	if err != nil {
		fmt.Fprintf(stdout, "repo_guard: could not read decision journal %s (%v)\n", path, err)
		return 1
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		enc.SetIndent("", "  ")
		_ = enc.Encode(sum)
		return 0
	}
	fmt.Fprintln(stdout, repoguard.RenderSummary(sum))
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
		// would-hang interactive forms (#2080): refused pre-execution.
		{"Bash", cmd("git rebase -i HEAD~3")},
		{"Bash", cmd("git commit")},
		{"Bash", cmd("git add -p")},
		{"Bash", cmd("vim cmd/fak/main.go")},
		{"Bash", cmd("gh auth login")},
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
		// benign non-interactive forms of the #2080 curated set pass unchanged.
		{"Bash", cmd(`git commit -s -m "fix(x): y" -- cmd/fak/main.go`)},
		{"Bash", cmd("git add -- cmd/fak/main.go")},
		{"Bash", cmd("git commit --amend --no-edit")},
		{"Bash", cmd("gh auth login --with-token < token.txt")},
		{"Bash", cmd("git log --oneline | less")},
		{"Bash", cmd("GIT_SEQUENCE_EDITOR=: git rebase -i HEAD~3")},
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
