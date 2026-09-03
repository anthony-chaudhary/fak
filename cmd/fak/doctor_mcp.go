package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

const doctorMCPSchema = "fak-doctor-mcp/1"

type doctorMCPStage struct {
	Name        string `json:"name"`
	Status      string `json:"status"`
	Cause       string `json:"cause,omitempty"`
	Detail      string `json:"detail,omitempty"`
	Remediation string `json:"remediation,omitempty"`
}

type doctorMCPReport struct {
	Schema     string           `json:"schema"`
	Server     string           `json:"server"`
	Command    string           `json:"command"`
	Args       []string         `json:"args,omitempty"`
	ConfigPath string           `json:"config_path,omitempty"`
	Stages     []doctorMCPStage `json:"stages"`
	OK         bool             `json:"ok"`
}

func runDoctorMCP(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor mcp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	server := fs.String("server", "fak", "Codex mcp_servers entry to diagnose")
	config := fs.String("config", defaultCodexConfigPath(), "Codex config.toml path")
	command := fs.String("command", "", "explicit MCP executable (bypasses config lookup)")
	timeout := fs.Duration("timeout", 5*time.Second, "initialize response timeout")
	asJSON := fs.Bool("json", false, "emit stable JSON report")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	rep := diagnoseMCP(*server, *config, *command, fs.Args(), *timeout)
	if *asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		writeDoctorMCPHuman(stdout, rep)
	}
	if !rep.OK {
		return 1
	}
	return 0
}

func defaultCodexConfigPath() string {
	if home := strings.TrimSpace(os.Getenv("CODEX_HOME")); home != "" {
		return filepath.Join(home, "config.toml")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".codex", "config.toml")
}

func diagnoseMCP(server, configPath, explicit string, explicitArgs []string, timeout time.Duration) doctorMCPReport {
	rep := doctorMCPReport{Schema: doctorMCPSchema, Server: server, ConfigPath: configPath, Stages: []doctorMCPStage{}}
	add := func(s doctorMCPStage) { rep.Stages = append(rep.Stages, s) }
	command, args := strings.TrimSpace(explicit), explicitArgs
	if command == "" {
		var err error
		var source string
		command, args, source, err = resolveCodexMCPEntry(configPath, server)
		if err != nil {
			add(doctorMCPStage{Name: "config_parse", Status: "fail", Cause: "CONFIG_INVALID", Detail: err.Error(), Remediation: "fix " + configPath + " or pass --command explicitly"})
			return rep
		}
		add(doctorMCPStage{Name: "config_parse", Status: "pass", Detail: "resolved " + source + " without printing env/secrets"})
	} else {
		rep.ConfigPath = ""
		add(doctorMCPStage{Name: "config_parse", Status: "skip", Detail: "explicit command"})
	}
	rep.Command, rep.Args = command, append([]string(nil), args...)
	resolved, err := exec.LookPath(command)
	if err != nil && errors.Is(err, exec.ErrDot) {
		target := command
		if resolved != "" {
			target = resolved
		}
		if abs, absErr := filepath.Abs(target); absErr == nil {
			resolved = abs
			err = nil
		}
	}
	if err != nil {
		cause := "EXECUTABLE_MISSING"
		remediation := "install/rebuild the configured executable, then update mcp_servers." + server + ".command"
		if errors.Is(err, os.ErrPermission) {
			cause = "EXECUTABLE_PERMISSION_DENIED"
			remediation = "grant execute permission to the configured executable or select an executable command"
		}
		add(doctorMCPStage{Name: "executable_resolution", Status: "fail", Cause: cause, Detail: err.Error(), Remediation: remediation})
		return rep
	}
	if abs, e := filepath.Abs(resolved); e == nil {
		resolved = abs
	}
	rep.Command = resolved
	versionDetail := resolved
	versionCmd := exec.Command(resolved, "--version")
	if out, err := versionCmd.CombinedOutput(); err == nil {
		if line := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0]); line != "" {
			versionDetail += " version=" + line
		}
	}
	add(doctorMCPStage{Name: "executable_resolution", Status: "pass", Detail: versionDetail})
	if policy := flagValue(args, "--policy"); policy != "" {
		data, readErr := os.ReadFile(policy)
		if readErr != nil {
			add(doctorMCPStage{Name: "policy_readability", Status: "fail", Cause: "POLICY_UNREADABLE", Detail: readErr.Error(), Remediation: "fix --policy path/permissions: " + policy})
			return rep
		}
		if !json.Valid(data) {
			add(doctorMCPStage{Name: "policy_readability", Status: "fail", Cause: "POLICY_MALFORMED", Detail: "policy is not valid JSON", Remediation: "repair the JSON policy: " + policy})
			return rep
		}
		add(doctorMCPStage{Name: "policy_readability", Status: "pass", Detail: policy})
	} else {
		add(doctorMCPStage{Name: "policy_readability", Status: "skip", Detail: "no --policy argument"})
	}
	add(sessionRegistryRecoveryStage())
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, resolved, args...)
	// A diagnosis must not touch the operator's production descriptor store. The served
	// child gets an isolated scratch registry; a corrupt production file therefore remains
	// byte-identical and cannot be quarantined merely by probing initialize.
	isolatedRegistry := filepath.Join(os.TempDir(), fmt.Sprintf("fak-doctor-mcp-%d-session-registry.json", os.Getpid()))
	cmd.Env = append(os.Environ(), sessionRegistryEnv+"="+isolatedRegistry)
	in, err := cmd.StdinPipe()
	if err != nil {
		return mcpStageFail(rep, "child_spawn", "SPAWN_FAILED", err, "check executable permissions")
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return mcpStageFail(rep, "child_spawn", "SPAWN_FAILED", err, "check executable permissions")
	}
	errPipe, err := cmd.StderrPipe()
	if err != nil {
		return mcpStageFail(rep, "child_spawn", "SPAWN_FAILED", err, "check executable permissions")
	}
	if err := cmd.Start(); err != nil {
		return mcpStageFail(rep, "child_spawn", "SPAWN_FAILED", err, "check executable permissions and architecture")
	}
	add(doctorMCPStage{Name: "child_spawn", Status: "pass", Detail: fmt.Sprintf("pid=%d", cmd.Process.Pid)})
	request := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"fak-doctor","version":"1"}}}` + "\n"
	if _, err := io.WriteString(in, request); err != nil {
		_ = cmd.Process.Kill()
		return mcpStageFail(rep, "initialize_write", "INITIALIZE_WRITE_FAILED", err, "run the command directly and inspect stderr")
	}
	_ = in.Close()
	add(doctorMCPStage{Name: "initialize_write", Status: "pass"})
	type lineResult struct {
		line string
		err  error
	}
	ch := make(chan lineResult, 1)
	go func() { line, e := bufio.NewReader(out).ReadString('\n'); ch <- lineResult{line, e} }()
	var lr lineResult
	select {
	case lr = <-ch:
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		add(doctorMCPStage{Name: "initialize_response", Status: "fail", Cause: "INITIALIZE_TIMEOUT", Detail: timeout.String(), Remediation: "run the configured command directly; inspect blocking startup work and stderr"})
		add(doctorMCPStage{Name: "exit_status", Status: "fail", Cause: "CHILD_TIMEOUT", Detail: ctx.Err().Error(), Remediation: "remove blocking startup work or raise --timeout after measuring it"})
		return rep
	}
	stderrBytes, _ := io.ReadAll(io.LimitReader(errPipe, 4096))
	_ = cmd.Process.Kill()
	waitErr := cmd.Wait()
	stderrSummary := strings.TrimSpace(string(stderrBytes))
	if len(stderrSummary) > 500 {
		stderrSummary = stderrSummary[:500] + "…"
	}
	if stderrSummary == "" {
		add(doctorMCPStage{Name: "stderr_summary", Status: "pass", Detail: "empty"})
	} else {
		add(doctorMCPStage{Name: "stderr_summary", Status: "warn", Detail: stderrSummary})
	}
	line := strings.TrimSpace(lr.line)
	var rpc struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  json.RawMessage `json:"result"`
		Error   json.RawMessage `json:"error"`
	}
	if json.Unmarshal([]byte(line), &rpc) != nil || rpc.JSONRPC != "2.0" || len(rpc.ID) == 0 {
		cause := "STDOUT_CONTAMINATION"
		detail := line
		if line == "" {
			cause = "EARLY_EXIT"
			detail = fmt.Sprint(lr.err, "; ", waitErr)
		}
		add(doctorMCPStage{Name: "stdout_protocol_purity", Status: "fail", Cause: cause, Detail: detail, Remediation: "write logs only to stderr; stdout must contain newline-delimited JSON-RPC"})
		add(doctorMCPStage{Name: "exit_status", Status: "fail", Cause: "CHILD_EXIT", Detail: fmt.Sprint(waitErr), Remediation: "run the configured command directly and inspect stderr_summary"})
		return rep
	}
	add(doctorMCPStage{Name: "stdout_protocol_purity", Status: "pass"})
	if len(rpc.Error) > 0 && string(rpc.Error) != "null" {
		add(doctorMCPStage{Name: "initialize_response", Status: "fail", Cause: "INITIALIZE_ERROR", Detail: string(rpc.Error), Remediation: "inspect policy/config and the stderr_summary stage"})
		return rep
	}
	add(doctorMCPStage{Name: "initialize_response", Status: "pass", Detail: "valid JSON-RPC initialize result"})
	if waitErr != nil && !errors.Is(waitErr, os.ErrProcessDone) {
		add(doctorMCPStage{Name: "exit_status", Status: "pass", Detail: "terminated after successful probe"})
	} else {
		add(doctorMCPStage{Name: "exit_status", Status: "pass", Detail: "probe complete"})
	}
	rep.OK = true
	return rep
}

// sessionRegistryRecoveryStage surfaces corrupt-registry recovery counts and
// the last recovery time for the production session registry (#4658). It is
// strictly read-only (no locks, no file creation) and privacy-safe: the
// ledger it reads records outcome/cause/size/time, never descriptor contents.
func sessionRegistryRecoveryStage() doctorMCPStage {
	registry := defaultSessionRegistryPath()
	stats, ok, err := session.ReadRecoveryStats(registry)
	if err != nil {
		return doctorMCPStage{Name: "session_registry_recovery", Status: "skip", Detail: "recovery ledger unreadable: " + err.Error()}
	}
	if !ok {
		return doctorMCPStage{Name: "session_registry_recovery", Status: "pass", Detail: "no corrupt-registry recoveries recorded"}
	}
	evidence, evidenceErr := session.QuarantineEvidenceCount(registry)
	detail := fmt.Sprintf("recoveries_total=%d evidence_current=%d last=%s cause=%s",
		stats.Total, evidence, stats.LastAt.UTC().Format(time.RFC3339), stats.LastCause)
	if evidenceErr != nil {
		detail = fmt.Sprintf("recoveries_total=%d last=%s cause=%s", stats.Total, stats.LastAt.UTC().Format(time.RFC3339), stats.LastCause)
	}
	return doctorMCPStage{Name: "session_registry_recovery", Status: "warn", Detail: detail}
}

func mcpStageFail(rep doctorMCPReport, name, cause string, err error, remediation string) doctorMCPReport {
	rep.Stages = append(rep.Stages, doctorMCPStage{Name: name, Status: "fail", Cause: cause, Detail: err.Error(), Remediation: remediation})
	return rep
}

func flagValue(args []string, name string) string {
	for i, a := range args {
		if a == name && i+1 < len(args) {
			return args[i+1]
		}
		if strings.HasPrefix(a, name+"=") {
			return strings.TrimPrefix(a, name+"=")
		}
	}
	return ""
}

var errCodexMCPEntryNotFound = errors.New("Codex MCP entry not found")

type codexMCPGetResult struct {
	Name           string `json:"name"`
	Enabled        bool   `json:"enabled"`
	DisabledReason any    `json:"disabled_reason"`
	Transport      struct {
		Type    string   `json:"type"`
		Command string   `json:"command"`
		Args    []string `json:"args"`
	} `json:"transport"`
}

func resolveCodexMCPEntry(path, name string) (string, []string, string, error) {
	command, args, err := readCodexMCPEntry(path, name)
	if err == nil {
		return command, args, "mcp_servers." + name, nil
	}
	if !errors.Is(err, errCodexMCPEntryNotFound) {
		return "", nil, "", err
	}
	command, args, err = readCodexEffectiveMCPEntry(path, name)
	if err != nil {
		return "", nil, "", fmt.Errorf("%w; effective Codex lookup failed: %v", errCodexMCPEntryNotFound, err)
	}
	return command, args, "effective Codex MCP registration " + name, nil
}

var runCodexMCPGet = func(ctx context.Context, configPath, name string) ([]byte, []byte, error) {
	cmd := exec.CommandContext(ctx, "codex", "mcp", "get", name, "--json")
	// Plugin registrations are folded by CODEX_HOME rather than serialized into
	// config.toml. Point Codex at the same home the operator asked us to inspect.
	cmd.Env = append(os.Environ(), "CODEX_HOME="+filepath.Dir(configPath))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.Bytes(), stderr.Bytes(), err
}

func readCodexEffectiveMCPEntry(configPath, name string) (string, []string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	stdout, stderr, err := runCodexMCPGet(ctx, configPath, name)
	if err != nil {
		if ctx.Err() != nil {
			return "", nil, fmt.Errorf("codex mcp get timed out")
		}
		detail := strings.TrimSpace(string(stderr))
		if detail == "" {
			detail = err.Error()
		}
		return "", nil, fmt.Errorf("codex mcp get %s: %s", name, detail)
	}
	var got codexMCPGetResult
	if err := json.Unmarshal(stdout, &got); err != nil {
		return "", nil, fmt.Errorf("decode codex mcp get %s: %w", name, err)
	}
	if !got.Enabled {
		return "", nil, fmt.Errorf("Codex MCP server %s is disabled", name)
	}
	if got.Transport.Type != "stdio" {
		return "", nil, fmt.Errorf("Codex MCP server %s uses unsupported %q transport", name, got.Transport.Type)
	}
	if got.Transport.Command == "" {
		return "", nil, fmt.Errorf("Codex MCP server %s has no stdio command", name)
	}
	return got.Transport.Command, got.Transport.Args, nil
}

func readCodexMCPEntry(path, name string) (string, []string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", nil, err
	}
	section := "mcp_servers." + name
	active := false
	var command string
	var args []string
	for _, raw := range strings.Split(string(b), "\n") {
		line := strings.TrimSpace(strings.SplitN(raw, "#", 2)[0])
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			active = strings.Trim(line, "[]") == section
			continue
		}
		if !active {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "command":
			command, err = parseTOMLString(strings.TrimSpace(val))
		case "args":
			args, err = parseTOMLStringArray(strings.TrimSpace(val))
		}
		if err != nil {
			return "", nil, fmt.Errorf("%s: %w", section, err)
		}
	}
	if command == "" {
		return "", nil, fmt.Errorf("%w: %s command not found", errCodexMCPEntryNotFound, section)
	}
	return command, args, nil
}

func parseTOMLString(v string) (string, error) {
	if len(v) < 2 {
		return "", errors.New("invalid TOML string")
	}
	if v[0] == '\'' && v[len(v)-1] == '\'' {
		return v[1 : len(v)-1], nil
	}
	if v[0] != '"' || v[len(v)-1] != '"' {
		return "", errors.New("invalid TOML string")
	}
	var s string
	if err := json.Unmarshal([]byte(v), &s); err != nil {
		return "", err
	}
	return s, nil
}

func parseTOMLStringArray(v string) ([]string, error) {
	if !strings.HasPrefix(v, "[") || !strings.HasSuffix(v, "]") {
		return nil, errors.New("invalid TOML string array")
	}
	body := strings.TrimSpace(v[1 : len(v)-1])
	if body == "" {
		return nil, nil
	}
	var out []string
	var cur strings.Builder
	quote := byte(0)
	esc := false
	flush := func() error {
		s, err := parseTOMLString(strings.TrimSpace(cur.String()))
		if err != nil {
			return err
		}
		out = append(out, s)
		cur.Reset()
		return nil
	}
	for i := 0; i < len(body); i++ {
		c := body[i]
		if quote == 0 {
			if c == ',' {
				if err := flush(); err != nil {
					return nil, err
				}
				continue
			}
			if c == '\'' || c == '"' {
				quote = c
			}
			cur.WriteByte(c)
			continue
		}
		cur.WriteByte(c)
		if quote == '"' && c == '\\' && !esc {
			esc = true
			continue
		}
		if c == quote && !esc {
			quote = 0
		}
		esc = false
	}
	if quote != 0 {
		return nil, errors.New("unterminated TOML string")
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return out, nil
}

func writeDoctorMCPHuman(w io.Writer, rep doctorMCPReport) {
	fmt.Fprintf(w, "== fak doctor mcp: %s ==\n", rep.Server)
	for _, s := range rep.Stages {
		fmt.Fprintf(w, "[%s] %s", strings.ToUpper(s.Status), s.Name)
		if s.Cause != "" {
			fmt.Fprintf(w, " (%s)", s.Cause)
		}
		if s.Detail != "" {
			fmt.Fprintf(w, ": %s", s.Detail)
		}
		fmt.Fprintln(w)
		if s.Remediation != "" {
			fmt.Fprintf(w, "       fix: %s\n", s.Remediation)
		}
	}
	if rep.OK {
		fmt.Fprintln(w, "doctor mcp: healthy")
	} else {
		fmt.Fprintln(w, "doctor mcp: startup failed")
	}
}
