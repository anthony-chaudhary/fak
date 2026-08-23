package devcmd

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

const codexHookProfileSchema = "fak/codex-hook-profile/v1"

type executableIdentity struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256,omitempty"`
	Exists bool   `json:"exists"`
	Role   string `json:"role"`
}
type effectiveHook struct {
	EventName    string               `json:"event_name"`
	Key          string               `json:"key"`
	Source       string               `json:"source"`
	SourcePath   string               `json:"source_path"`
	PluginID     string               `json:"plugin_id,omitempty"`
	DisplayOrder int                  `json:"display_order"`
	Enabled      bool                 `json:"enabled"`
	Managed      bool                 `json:"managed"`
	CurrentHash  string               `json:"current_hash"`
	TrustStatus  string               `json:"trust_status"`
	Command      string               `json:"command"`
	Matcher      string               `json:"matcher,omitempty"`
	TimeoutSec   int                  `json:"timeout_sec"`
	State        string               `json:"state"`
	Remediation  string               `json:"remediation,omitempty"`
	Identities   []executableIdentity `json:"identities,omitempty"`
}
type hookProfileReport struct {
	Schema          string             `json:"schema"`
	CodexHome       string             `json:"codex_home"`
	Workspace       string             `json:"workspace"`
	ConfigPath      string             `json:"config_path"`
	CodexExecutable executableIdentity `json:"codex_executable"`
	Hooks           []effectiveHook    `json:"hooks"`
	Warnings        []string           `json:"warnings,omitempty"`
	Errors          []string           `json:"errors,omitempty"`
	Verdict         string             `json:"verdict"`
}
type appServerReply struct {
	ID     int `json:"id"`
	Result struct {
		Data []struct {
			CWD   string `json:"cwd"`
			Hooks []struct {
				Key          string `json:"key"`
				EventName    string `json:"eventName"`
				Command      string `json:"command"`
				Matcher      string `json:"matcher"`
				TimeoutSec   int    `json:"timeoutSec"`
				SourcePath   string `json:"sourcePath"`
				Source       string `json:"source"`
				PluginID     string `json:"pluginId"`
				DisplayOrder int    `json:"displayOrder"`
				Enabled      bool   `json:"enabled"`
				IsManaged    bool   `json:"isManaged"`
				CurrentHash  string `json:"currentHash"`
				TrustStatus  string `json:"trustStatus"`
			} `json:"hooks"`
			Warnings []string `json:"warnings"`
			Errors   []string `json:"errors"`
		} `json:"data"`
	} `json:"result"`
	Error json.RawMessage `json:"error"`
}

func RunCodexHookProfile(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-hook-profile", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	home := fs.String("codex-home", "", "active Codex home")
	workspace := fs.String("workspace", ".", "workspace whose effective hooks are resolved")
	binary := fs.String("codex-binary", "", "Codex executable")
	asJSON := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-hook-profile [--codex-home DIR] [--workspace DIR] [--codex-binary FILE] [--json]")
		return 2
	}
	if *home == "" {
		*home = os.Getenv("CODEX_HOME")
	}
	if *home == "" {
		h, e := os.UserHomeDir()
		if e != nil {
			return 1
		}
		*home = filepath.Join(h, ".codex")
	}
	r, e := inspectCodexHookProfile(*home, *workspace, *binary)
	if e != nil {
		fmt.Fprintf(stderr, "codex-hook-profile: %v\n", e)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
	} else {
		writeHookProfile(stdout, r)
	}
	if r.Verdict != "HEALTHY" {
		return 1
	}
	return 0
}
func inspectCodexHookProfile(home, workspace, binary string) (hookProfileReport, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return inspectCodexHookProfileContext(ctx, home, workspace, binary)
}

func inspectCodexHookProfileContext(ctx context.Context, home, workspace, binary string) (hookProfileReport, error) {
	home, _ = filepath.Abs(home)
	workspace, _ = filepath.Abs(workspace)
	if binary == "" {
		binary = discoverCodexBinary()
	}
	if binary == "" {
		return hookProfileReport{}, errors.New("Codex executable not found")
	}
	identity := identifyFile(binary, "codex-app-server")
	reply, err := queryCodexHooks(home, workspace, binary)
	if err != nil {
		return hookProfileReport{}, err
	}
	r := hookProfileReport{Schema: codexHookProfileSchema, CodexHome: home, Workspace: workspace, ConfigPath: filepath.Join(home, "config.toml"), CodexExecutable: identity, Verdict: "HEALTHY"}
	if len(reply.Result.Data) == 0 {
		return r, errors.New("hooks/list returned no workspace entry")
	}
	entry := reply.Result.Data[0]
	r.Warnings = entry.Warnings
	r.Errors = entry.Errors
	for _, h := range entry.Hooks {
		event := normalizeHookEvent(h.EventName)
		if !isObservedHookEvent(event) {
			continue
		}
		x := effectiveHook{EventName: event, Key: h.Key, Source: h.Source, SourcePath: h.SourcePath, PluginID: h.PluginID, DisplayOrder: h.DisplayOrder, Enabled: h.Enabled, Managed: h.IsManaged, CurrentHash: h.CurrentHash, TrustStatus: h.TrustStatus, Command: h.Command, Matcher: h.Matcher, TimeoutSec: h.TimeoutSec}
		classifyEffectiveHook(&x, home)
		r.Hooks = append(r.Hooks, x)
		if x.State != "effective" {
			r.Verdict = "UNHEALTHY"
		}
	}
	sort.Slice(r.Hooks, func(i, j int) bool {
		if r.Hooks[i].EventName == r.Hooks[j].EventName {
			return r.Hooks[i].DisplayOrder < r.Hooks[j].DisplayOrder
		}
		return r.Hooks[i].EventName < r.Hooks[j].EventName
	})
	for _, event := range []string{"pre_tool_use", "post_tool_use", "stop", "subagent_stop"} {
		found := false
		for _, h := range r.Hooks {
			found = found || h.EventName == event
		}
		if !found {
			r.Errors = append(r.Errors, "missing effective "+event+" hook")
			r.Verdict = "UNHEALTHY"
		}
	}
	if len(r.Errors) > 0 {
		r.Verdict = "UNHEALTHY"
	}
	return r, nil
}
func isObservedHookEvent(event string) bool {
	switch event {
	case "pre_tool_use", "post_tool_use", "stop", "subagent_stop", "stop_failure":
		return true
	default:
		return false
	}
}

func normalizeHookEvent(s string) string {
	s = strings.ReplaceAll(s, "-", "_")
	var b strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteByte('_')
		}
		b.WriteRune(r)
	}
	return strings.ToLower(b.String())
}
func classifyEffectiveHook(h *effectiveHook, home string) {
	switch {
	case !h.Enabled:
		h.State = "disabled"
		h.Remediation = "enable this hook in hooks.state"
	case h.TrustStatus == "modified":
		h.State = "stale_hash"
		h.Remediation = "review the changed command and trust its currentHash"
	case h.TrustStatus == "untrusted":
		h.State = "untrusted"
		h.Remediation = "review the command and trust its currentHash"
	case h.TrustStatus != "trusted" && h.TrustStatus != "managed":
		h.State = "unknown"
		h.Remediation = "inspect hooks/list trustStatus"
	default:
		h.State = "effective"
	}
	if h.State == "effective" && hookCommandPlatformIncompatible(h.Command, runtime.GOOS) {
		h.State = "platform_incompatible"
		h.Remediation = "declare a commandWindows entry that invokes the native Codex hook adapter"
	}
	h.Identities = resolveHookIdentities(*h, home)
	for _, id := range h.Identities {
		if !id.Exists && h.State == "effective" {
			h.State = "missing_executable"
			h.Remediation = "restore or reinstall the executable named by the effective hook"
		}
	}
}
func hookCommandPlatformIncompatible(command, goos string) bool {
	if goos != "windows" {
		return false
	}
	// Codex executes the selected Windows command in PowerShell. POSIX shell
	// expansion and redirection therefore fail before the hook backend runs.
	for _, marker := range []string{"${", "command -p sh", "2>/dev/null", "|| python3", "[ -n ", "[ -z "} {
		if strings.Contains(command, marker) {
			return true
		}
	}
	return false
}

func resolveHookIdentities(h effectiveHook, home string) []executableIdentity {
	var out []executableIdentity
	if h.PluginID != "" && h.SourcePath != "" {
		root := filepath.Dir(filepath.Dir(h.SourcePath))
		launcher := filepath.Join(root, "bin", "dos-hook")
		if strings.Contains(h.Command, "bin/dos-hook") {
			out = append(out, identifyFile(launcher, "hook-launcher"))
			name := "dos-hook-" + runtime.GOOS + "-" + runtime.GOARCH
			if runtime.GOOS == "windows" {
				name += ".exe"
			}
			out = append(out, identifyFile(filepath.Join(root, "bin", name), "native-hook-binary"))
		}
	}
	return out
}
func identifyFile(path, role string) executableIdentity {
	id := executableIdentity{Path: path, Role: role}
	f, e := os.Open(path)
	if e != nil {
		return id
	}
	defer f.Close()
	sum := sha256.New()
	if _, e = io.Copy(sum, f); e == nil {
		id.Exists = true
		id.SHA256 = "sha256:" + hex.EncodeToString(sum.Sum(nil))
	}
	return id
}
func discoverCodexBinary() string {
	if app := os.Getenv("APPDATA"); app != "" {
		p := filepath.Join(app, "npm", "node_modules", "@openai", "codex", "node_modules", "@openai", "codex-win32-x64", "vendor", "x86_64-pc-windows-msvc", "bin", "codex.exe")
		if _, e := os.Stat(p); e == nil {
			return p
		}
	}
	p, _ := exec.LookPath("codex")
	return p
}
func queryCodexHooks(home, workspace, binary string) (appServerReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return queryCodexHooksContext(ctx, home, workspace, binary)
}

func queryCodexHooksContext(ctx context.Context, home, workspace, binary string) (appServerReply, error) {
	cmd := exec.CommandContext(ctx, binary, "app-server", "--stdio")
	cmd.Env = append(os.Environ(), "CODEX_HOME="+home)
	stdin, e := cmd.StdinPipe()
	if e != nil {
		return appServerReply{}, e
	}
	stdout, e := cmd.StdoutPipe()
	if e != nil {
		return appServerReply{}, e
	}
	var errbuf bytes.Buffer
	cmd.Stderr = &errbuf
	if e = cmd.Start(); e != nil {
		return appServerReply{}, e
	}
	enc := json.NewEncoder(stdin)
	for _, v := range []any{map[string]any{"method": "initialize", "id": 1, "params": map[string]any{"clientInfo": map[string]string{"name": "fak_diag", "title": "fak diagnostic", "version": "1"}, "capabilities": map[string]bool{"experimentalApi": true}}}, map[string]any{"method": "initialized", "params": map[string]any{}}, map[string]any{"method": "hooks/list", "id": 2, "params": map[string]any{"cwds": []string{workspace}}}} {
		if e = enc.Encode(v); e != nil {
			return appServerReply{}, e
		}
	}
	s := bufio.NewScanner(stdout)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for s.Scan() {
		var r appServerReply
		if json.Unmarshal(s.Bytes(), &r) == nil && r.ID == 2 {
			_ = cmd.Process.Kill()
			if len(r.Error) > 0 {
				return r, fmt.Errorf("hooks/list: %s", r.Error)
			}
			return r, nil
		}
	}
	_ = cmd.Process.Kill()
	if ctx.Err() != nil {
		return appServerReply{}, ctx.Err()
	}
	return appServerReply{}, fmt.Errorf("hooks/list produced no response: %s", strings.TrimSpace(errbuf.String()))
}
func writeHookProfile(w io.Writer, r hookProfileReport) {
	fmt.Fprintf(w, "Codex effective hook profile\nCODEX_HOME: %s\nconfig: %s\nworkspace: %s\nCodex binary: %s %s\n", r.CodexHome, r.ConfigPath, r.Workspace, r.CodexExecutable.Path, r.CodexExecutable.SHA256)
	for _, h := range r.Hooks {
		fmt.Fprintf(w, "%s order=%d state=%s enabled=%t trust=%s source=%s sourcePath=%s\n  key: %s\n  command: %s\n", h.EventName, h.DisplayOrder, h.State, h.Enabled, h.TrustStatus, h.Source, h.SourcePath, h.Key, h.Command)
		for _, id := range h.Identities {
			fmt.Fprintf(w, "  %s: %s exists=%t %s\n", id.Role, id.Path, id.Exists, id.SHA256)
		}
	}
	fmt.Fprintf(w, "verdict: %s\n", r.Verdict)
}
