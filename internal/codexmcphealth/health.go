package codexmcphealth

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	TransportDeadServerOK = "TRANSPORT_DEAD_SERVER_OK"
	ServerDead            = "SERVER_DEAD"
	ReconnectOK           = "RECONNECT_OK"
	StaleChildren         = "STALE_CHILDREN"

	DefaultPolicy = "examples/dev-agent-policy.json"
)

var DiagnosticStates = []string{
	TransportDeadServerOK,
	ServerDead,
	ReconnectOK,
	StaleChildren,
}

var NextStep = map[string]string{
	TransportDeadServerOK: "Configured fak stdio server is healthy but the in-session Codex transport is dead. Reconnect / respawn the MCP client; do not retry the dead transport.",
	ServerDead:            "The configured fak stdio server failed its own smoke. Fix the server before reconnecting; retrying the transport is futile.",
	ReconnectOK:           "Server answers and the in-session transport is alive. No action; MCP calls should succeed.",
	StaleChildren:         "Stray 'fak serve --stdio' children are present. Review the inventory and reap explicitly with --reap <pid ...>; do not blind-kill.",
}

type SmokeResult struct {
	OK      bool   `json:"ok"`
	Verdict string `json:"verdict"`
	Status  string `json:"status"`
	Engine  string `json:"engine"`
	Content string `json:"content"`
	Reason  string `json:"reason"`
}

type ChildProc struct {
	PID     int    `json:"pid"`
	Command string `json:"command"`
}

type Diagnostic struct {
	State          string      `json:"state"`
	NextStep       string      `json:"next_step"`
	ServerOK       bool        `json:"server_ok"`
	TransportAlive bool        `json:"transport_alive"`
	Smoke          SmokeResult `json:"smoke"`
	StaleChildren  []ChildProc `json:"stale_children"`
	InventoryNote  string      `json:"inventory_note"`
}

type Options struct {
	Root           string
	Policy         string
	TransportAlive bool
	Timeout        time.Duration
}

type ReapResult struct {
	PID    int    `json:"pid"`
	Reaped bool   `json:"reaped"`
	Detail string `json:"detail"`
}

func ExpectedVersion(root string) string {
	b, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func FakBinary(root string) string {
	name := "fak"
	if runtime.GOOS == "windows" {
		name = "fak.exe"
	}
	candidate := filepath.Join(root, name)
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	if exe, err := os.Executable(); err == nil {
		return exe
	}
	return candidate
}

func SmokeFrames() []byte {
	init := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "initialize",
		"params":  map[string]any{"protocolVersion": "2025-06-18"},
	}
	inited := map[string]any{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	}
	call := map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "fak_read",
			"arguments": map[string]any{"file_path": "VERSION"},
		},
	}
	var buf bytes.Buffer
	for _, frame := range []map[string]any{init, inited, call} {
		b, _ := json.Marshal(frame)
		buf.Write(b)
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

func ParseSmokeOutput(stdout, expectedVersion string) SmokeResult {
	var callResp map[string]any
	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			continue
		}
		if numberInt(obj["id"]) == 2 {
			callResp = obj
		}
	}
	if callResp == nil {
		return SmokeResult{OK: false, Reason: "no tools/call response (id=2) in server output"}
	}
	if errObj, ok := callResp["error"]; ok {
		return SmokeResult{OK: false, Reason: fmt.Sprintf("server returned JSON-RPC error: %v", errObj)}
	}
	result, _ := callResp["result"].(map[string]any)
	if b, _ := result["isError"].(bool); b {
		return SmokeResult{OK: false, Reason: "tool result isError=true"}
	}
	contentList, _ := result["content"].([]any)
	if len(contentList) == 0 {
		return SmokeResult{OK: false, Reason: "tool result has empty content"}
	}
	first, _ := contentList[0].(map[string]any)
	text, _ := first["text"].(string)
	var inner map[string]any
	if err := json.Unmarshal([]byte(text), &inner); err != nil {
		return SmokeResult{OK: false, Reason: "tool result text is not JSON"}
	}
	verdictMap, _ := inner["verdict"].(map[string]any)
	verdict, _ := verdictMap["kind"].(string)
	res, _ := inner["result"].(map[string]any)
	status, _ := res["status"].(string)
	meta, _ := res["meta"].(map[string]any)
	engine, _ := meta["engine"].(string)
	rawContent, _ := res["content"].(string)
	fileBody := rawContent
	var nested map[string]any
	if err := json.Unmarshal([]byte(rawContent), &nested); err == nil {
		if s, _ := nested["content"].(string); s != "" {
			fileBody = s
		}
	}
	ok := verdict == "ALLOW" && status == "OK" && engine == "fakread"
	if expectedVersion != "" {
		ok = ok && strings.Contains(fileBody, expectedVersion)
	}
	reason := ""
	if !ok {
		reason = fmt.Sprintf("smoke mismatch: verdict=%q status=%q engine=%q content=%q", verdict, status, engine, fileBody)
		if expectedVersion != "" {
			reason += fmt.Sprintf(" (want ALLOW/OK/fakread/%s)", expectedVersion)
		}
	}
	return SmokeResult{
		OK:      ok,
		Verdict: verdict,
		Status:  status,
		Engine:  engine,
		Content: strings.TrimSpace(fileBody),
		Reason:  reason,
	}
}

func RunServerSmoke(root, policy string, timeout time.Duration) SmokeResult {
	if policy == "" {
		policy = DefaultPolicy
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	binary := FakBinary(root)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "serve", "--stdio", "--policy", policy)
	cmd.Dir = root
	cmd.Stdin = bytes.NewReader(SmokeFrames())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return SmokeResult{OK: false, Reason: fmt.Sprintf("smoke timed out after %s", timeout)}
	}
	res := ParseSmokeOutput(stdout.String(), ExpectedVersion(root))
	if !res.OK && err != nil && res.Reason == "" {
		res.Reason = fmt.Sprintf("server exited with %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	if !res.OK && err != nil && res.Reason != "" && strings.TrimSpace(stderr.String()) != "" {
		res.Reason += ": " + strings.TrimSpace(stderr.String())
	}
	return res
}

func ParsePowerShellInventory(stdout string) []ChildProc {
	stdout = strings.TrimSpace(stdout)
	if stdout == "" {
		return nil
	}
	dec := json.NewDecoder(strings.NewReader(stdout))
	dec.UseNumber()
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return nil
	}
	var rows []any
	switch v := raw.(type) {
	case []any:
		rows = v
	case map[string]any:
		rows = []any{v}
	default:
		return nil
	}
	var out []ChildProc
	for _, row := range rows {
		obj, ok := row.(map[string]any)
		if !ok {
			continue
		}
		cmdline, _ := obj["CommandLine"].(string)
		if cmdline == "" {
			cmdline, _ = obj["commandline"].(string)
		}
		if !strings.Contains(cmdline, "serve") || !strings.Contains(cmdline, "--stdio") {
			continue
		}
		pid, ok := parseJSONInt(obj["ProcessId"])
		if !ok {
			pid, ok = parseJSONInt(obj["processid"])
		}
		if !ok {
			continue
		}
		out = append(out, ChildProc{PID: pid, Command: cmdline})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

func InventoryStaleChildren(root string) ([]ChildProc, string) {
	if runtime.GOOS != "windows" {
		return nil, "process inventory gated to windows (platform=" + runtime.GOOS + ")"
	}
	fakPath := strings.ReplaceAll(FakBinary(root), "'", "''")
	ps := "Get-CimInstance Win32_Process | " +
		"Where-Object { $_.ExecutablePath -eq '" + fakPath + "' } | " +
		"Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", ps)
	out, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, "inventory probe timed out"
	}
	if err != nil {
		return nil, "inventory probe failed: " + err.Error()
	}
	children := ParsePowerShellInventory(string(out))
	if len(children) == 0 {
		return nil, "no stray fak serve --stdio children found"
	}
	return children, ""
}

func ReapChildren(pids []int) []ReapResult {
	results := make([]ReapResult, 0, len(pids))
	for _, pid := range pids {
		if runtime.GOOS == "windows" {
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			cmd := exec.CommandContext(ctx, "taskkill", "/PID", strconv.Itoa(pid), "/F")
			out, err := cmd.CombinedOutput()
			cancel()
			detail := strings.TrimSpace(string(out))
			if err != nil && detail == "" {
				detail = err.Error()
			}
			results = append(results, ReapResult{PID: pid, Reaped: err == nil, Detail: detail})
			continue
		}
		proc, err := os.FindProcess(pid)
		if err == nil {
			err = proc.Kill()
		}
		detail := "terminated"
		if err != nil {
			detail = err.Error()
		}
		results = append(results, ReapResult{PID: pid, Reaped: err == nil, Detail: detail})
	}
	return results
}

func Classify(smoke SmokeResult, transportAlive bool, staleChildren []ChildProc, inventoryNote string) Diagnostic {
	state := ReconnectOK
	switch {
	case !smoke.OK:
		state = ServerDead
	case !transportAlive:
		state = TransportDeadServerOK
	case len(staleChildren) > 0:
		state = StaleChildren
	}
	return Diagnostic{
		State:          state,
		NextStep:       NextStep[state],
		ServerOK:       smoke.OK,
		TransportAlive: transportAlive,
		Smoke:          smoke,
		StaleChildren:  staleChildren,
		InventoryNote:  inventoryNote,
	}
}

func Diagnose(opt Options) Diagnostic {
	root := opt.Root
	if root == "" {
		root = "."
	}
	smoke := RunServerSmoke(root, opt.Policy, opt.Timeout)
	children, note := InventoryStaleChildren(root)
	return Classify(smoke, opt.TransportAlive, children, note)
}

func RenderTable(diag Diagnostic) string {
	lines := []string{
		"diagnostic: " + diag.State,
		fmt.Sprintf("  server_ok        : %v", diag.ServerOK),
		fmt.Sprintf("  transport_alive  : %v", diag.TransportAlive),
		fmt.Sprintf("  smoke            : verdict=%s status=%s engine=%s content=%s",
			dash(diag.Smoke.Verdict), dash(diag.Smoke.Status), dash(diag.Smoke.Engine), dash(diag.Smoke.Content)),
	}
	if diag.Smoke.Reason != "" {
		lines = append(lines, "  smoke_reason     : "+diag.Smoke.Reason)
	}
	if len(diag.StaleChildren) > 0 {
		lines = append(lines, fmt.Sprintf("  stale_children   : %d", len(diag.StaleChildren)))
		for _, child := range diag.StaleChildren {
			lines = append(lines, fmt.Sprintf("    - pid=%d %s", child.PID, child.Command))
		}
	} else if diag.InventoryNote != "" {
		lines = append(lines, "  stale_children   : 0 ("+diag.InventoryNote+")")
	}
	lines = append(lines, "  next_step        : "+diag.NextStep)
	return strings.Join(lines, "\n")
}

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func numberInt(v any) int {
	n, _ := parseJSONInt(v)
	return n
}

func parseJSONInt(v any) (int, bool) {
	switch n := v.(type) {
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	case float64:
		return int(n), true
	case int:
		return n, true
	case string:
		i, err := strconv.Atoi(n)
		return i, err == nil
	default:
		return 0, false
	}
}
