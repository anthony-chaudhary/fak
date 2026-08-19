package devcmd

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	hookProfileSchemaVersion = "fak.codex_hook_profile.v1"

	hookProfileVerdictOK               = "OK"
	hookProfileVerdictAction           = "ACTION"
	hookProfileStatusReady             = "ready"
	hookProfileStatusShadowed          = "shadowed"
	hookProfileStatusDisabled          = "disabled"
	hookProfileStatusStaleHash         = "stale-hash"
	hookProfileStatusMissingExecutable = "missing-executable"
	hookProfileStatusStaleBinary       = "stale-binary"
	hookProfileStatusUnknown           = "unknown"
)

type codexHookProfileQueryInput struct {
	CodexHome        string
	CodexBin         string
	WorkingDirectory string
	LogDBPath        string
	RepoRoot         string
	RecentLogRows    int
	ObservedAt       time.Time
}

type codexHookProfileBuildInput struct {
	ObservedAt         time.Time
	WorkingDirectory   string
	DeclaredCodexHome  string
	ActiveCodexHome    string
	DefaultCodexHome   string
	AppServerUserAgent string
	CodexExecutable    codexExecutableIdentity
	LogDBPath          string
	ActiveLogDBPath    string
	TrunkHead          string
	Homes              []codexHomeObservation
	Hooks              []codexEffectiveHook
	Warnings           []string
	Errors             []codexHookSourceError
	RecentToolFailures codexRecentToolFailures
}

type codexHookProfileReport struct {
	Schema             string                  `json:"schema"`
	ObservedAt         time.Time               `json:"observed_at"`
	Verdict            string                  `json:"verdict"`
	ActiveCodexHome    string                  `json:"active_codex_home"`
	Profile            codexProfileIdentity    `json:"profile"`
	LogStore           codexLogStoreIdentity   `json:"log_store"`
	ConfigPrecedence   []string                `json:"config_precedence_high_to_low"`
	HookMergeSemantics string                  `json:"hook_merge_semantics"`
	Homes              []codexHomeObservation  `json:"homes"`
	Hooks              []codexEffectiveHook    `json:"hooks"`
	Diagnoses          []codexHookDiagnosis    `json:"diagnoses"`
	RecentToolFailures codexRecentToolFailures `json:"recent_tool_failures"`
	Warnings           []string                `json:"warnings,omitempty"`
	Errors             []codexHookSourceError  `json:"errors,omitempty"`
}

type codexProfileIdentity struct {
	DeclaredCodexHome  string                  `json:"declared_codex_home,omitempty"`
	AppServerCodexHome string                  `json:"app_server_codex_home"`
	WorkingDirectory   string                  `json:"working_directory"`
	AppServerUserAgent string                  `json:"app_server_user_agent,omitempty"`
	CodexExecutable    codexExecutableIdentity `json:"codex_executable"`
	TrunkHead          string                  `json:"trunk_head,omitempty"`
	Matches            bool                    `json:"matches"`
}

type codexLogStoreIdentity struct {
	RequestedPath string `json:"requested_path"`
	ActivePath    string `json:"active_path"`
	Exists        bool   `json:"exists"`
	ProfileMatch  bool   `json:"profile_match"`
}

type codexHomeObservation struct {
	Path      string `json:"path"`
	Exists    bool   `json:"exists"`
	Active    bool   `json:"active"`
	HasConfig bool   `json:"has_config"`
	HasLogDB  bool   `json:"has_log_db"`
	Status    string `json:"status"`
}

type codexEffectiveHook struct {
	Key          string                    `json:"key"`
	EventName    string                    `json:"event_name"`
	HandlerType  string                    `json:"handler_type"`
	Matcher      string                    `json:"matcher,omitempty"`
	Command      string                    `json:"command,omitempty"`
	SourcePath   string                    `json:"source_path"`
	Source       string                    `json:"source"`
	PluginID     string                    `json:"plugin_id,omitempty"`
	DisplayOrder int64                     `json:"display_order"`
	Precedence   string                    `json:"precedence"`
	Layer        string                    `json:"layer"`
	Enabled      bool                      `json:"enabled"`
	Managed      bool                      `json:"managed"`
	CurrentHash  string                    `json:"trust_hash"`
	TrustStatus  string                    `json:"trust_status"`
	Status       string                    `json:"status"`
	Executables  []codexExecutableIdentity `json:"executables,omitempty"`
}

type codexExecutableIdentity struct {
	Name    string `json:"name"`
	Path    string `json:"path,omitempty"`
	Exists  bool   `json:"exists"`
	SHA256  string `json:"sha256,omitempty"`
	Version string `json:"version,omitempty"`
	Build   string `json:"build,omitempty"`
	Stale   bool   `json:"stale"`
}

type codexHookDiagnosis struct {
	Type        string `json:"type"`
	Subject     string `json:"subject"`
	Message     string `json:"message"`
	Remediation string `json:"remediation"`
}

type codexHookSourceError struct {
	Path    string `json:"path,omitempty"`
	Message string `json:"message"`
}

type codexRecentToolFailures struct {
	LogPath        string `json:"log_path,omitempty"`
	RowWindow      int    `json:"row_window"`
	RouterErrors   int    `json:"router_errors"`
	RefusalSignals int    `json:"refusal_signals"`
	ParserErrors   int    `json:"parser_errors"`
	Timeouts       int    `json:"timeouts"`
	Interpretation string `json:"interpretation"`
}

type codexHookProfileQuery func(codexHookProfileQueryInput) (codexHookProfileBuildInput, error)

func runCodexHookProfileWith(stdout, stderr io.Writer, input codexHookProfileQueryInput, jsonOut bool, query codexHookProfileQuery) int {
	if query == nil {
		query = queryCodexHookProfile
	}
	built, err := query(input)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessiondiag: hook-profile probe failed: %s\n", strings.TrimSpace(err.Error()))
		return 2
	}
	report := buildCodexHookProfile(built)
	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(stderr, "fak sessiondiag: encode hook profile")
			return 2
		}
	} else {
		writeHookProfileReport(stdout, report)
	}
	if report.Verdict == hookProfileVerdictAction {
		return 1
	}
	return 0
}

func buildCodexHookProfile(input codexHookProfileBuildInput) codexHookProfileReport {
	report := codexHookProfileReport{
		Schema:          hookProfileSchemaVersion,
		ObservedAt:      input.ObservedAt.UTC(),
		Verdict:         hookProfileVerdictOK,
		ActiveCodexHome: input.ActiveCodexHome,
		ConfigPrecedence: []string{
			"session flags and command-line overrides",
			"workspace/project config, closest directory first",
			"selected profile config",
			"active CODEX_HOME user config",
			"system config",
			"built-in defaults",
		},
		HookMergeSemantics: "effective hooks are additive across loaded config and plugin sources; display_order is runtime execution order, not replacement precedence",
		RecentToolFailures: input.RecentToolFailures,
		Warnings:           append([]string(nil), input.Warnings...),
		Errors:             append([]codexHookSourceError(nil), input.Errors...),
	}
	report.Profile = codexProfileIdentity{
		DeclaredCodexHome:  input.DeclaredCodexHome,
		AppServerCodexHome: input.ActiveCodexHome,
		WorkingDirectory:   input.WorkingDirectory,
		AppServerUserAgent: input.AppServerUserAgent,
		CodexExecutable:    input.CodexExecutable,
		TrunkHead:          input.TrunkHead,
		Matches:            input.DeclaredCodexHome == "" || equalProfilePath(input.DeclaredCodexHome, input.ActiveCodexHome),
	}

	activeLogPath := input.ActiveLogDBPath
	if activeLogPath == "" && input.ActiveCodexHome != "" {
		activeLogPath = filepath.Join(input.ActiveCodexHome, "logs_2.sqlite")
	}
	requestedLogPath := input.LogDBPath
	if requestedLogPath == "" {
		requestedLogPath = activeLogPath
	}
	report.LogStore = codexLogStoreIdentity{
		RequestedPath: requestedLogPath,
		ActivePath:    activeLogPath,
		Exists:        fileExists(activeLogPath),
		ProfileMatch:  activeLogPath != "" && equalProfilePath(requestedLogPath, activeLogPath),
	}

	report.Homes = append([]codexHomeObservation(nil), input.Homes...)
	for i := range report.Homes {
		home := &report.Homes[i]
		switch {
		case home.Active:
			home.Status = hookProfileStatusReady
		case home.Exists:
			home.Status = hookProfileStatusShadowed
			report.Diagnoses = append(report.Diagnoses, codexHookDiagnosis{
				Type:        hookProfileStatusShadowed,
				Subject:     home.Path,
				Message:     "Codex home exists but was not selected by the app-server initialized under the current environment",
				Remediation: "launch Codex with the intended CODEX_HOME, or migrate/remove the inactive home after preserving credentials and state",
			})
		default:
			home.Status = hookProfileStatusUnknown
		}
	}

	if !report.Profile.Matches {
		report.Diagnoses = append(report.Diagnoses, codexHookDiagnosis{
			Type:        hookProfileStatusShadowed,
			Subject:     input.DeclaredCodexHome,
			Message:     "declared CODEX_HOME and the app-server's effective Codex home differ",
			Remediation: "restart the Codex process with the intended CODEX_HOME and rerun sessiondiag --hook-profile",
		})
	}
	if !report.LogStore.ProfileMatch {
		report.Diagnoses = append(report.Diagnoses, codexHookDiagnosis{
			Type:        hookProfileStatusShadowed,
			Subject:     requestedLogPath,
			Message:     "the requested diagnostic log store does not belong to the app-server's active Codex home",
			Remediation: "read logs_2.sqlite from active_codex_home, or pass --db explicitly when diagnosing a different profile",
		})
	}

	report.Hooks = append([]codexEffectiveHook(nil), input.Hooks...)
	sort.SliceStable(report.Hooks, func(i, j int) bool {
		return report.Hooks[i].DisplayOrder < report.Hooks[j].DisplayOrder
	})
	for i := range report.Hooks {
		hook := &report.Hooks[i]
		hook.Layer = classifyHookLayer(*hook, input.WorkingDirectory)
		hook.Precedence = fmt.Sprintf("display_order=%d; additive", hook.DisplayOrder)
		hook.Status, report.Diagnoses = diagnoseEffectiveHook(*hook, input.TrunkHead, report.Diagnoses)
	}
	for _, warning := range report.Warnings {
		report.Diagnoses = append(report.Diagnoses, codexHookDiagnosis{
			Type:        hookProfileStatusUnknown,
			Subject:     "codex app-server warning",
			Message:     warning,
			Remediation: "repair the named hook source, then rerun the app-server hooks/list probe",
		})
	}
	for _, sourceErr := range report.Errors {
		report.Diagnoses = append(report.Diagnoses, codexHookDiagnosis{
			Type:        hookProfileStatusUnknown,
			Subject:     hookFirstNonEmpty(sourceErr.Path, "codex hook source"),
			Message:     sourceErr.Message,
			Remediation: "repair or remove the unreadable hook source, then rerun sessiondiag --hook-profile",
		})
	}
	report.Diagnoses = append(report.Diagnoses, duplicateHostHandlerDiagnoses(report.Hooks)...)
	if len(report.Diagnoses) > 0 {
		report.Verdict = hookProfileVerdictAction
	}
	return report
}

func duplicateHostHandlerDiagnoses(hooks []codexEffectiveHook) []codexHookDiagnosis {
	byEvent := make(map[string][]codexEffectiveHook)
	for _, hook := range hooks {
		if !hook.Enabled || strings.TrimSpace(hook.EventName) == "" {
			continue
		}
		byEvent[hook.EventName] = append(byEvent[hook.EventName], hook)
	}
	var events []string
	for event, declarations := range byEvent {
		if len(declarations) > 1 {
			events = append(events, event)
		}
	}
	sort.Strings(events)
	diagnoses := make([]codexHookDiagnosis, 0, len(events))
	for _, event := range events {
		declarations := byEvent[event]
		keys := make([]string, 0, len(declarations))
		for _, declaration := range declarations {
			keys = append(keys, declaration.Key)
		}
		sort.Strings(keys)
		diagnoses = append(diagnoses, codexHookDiagnosis{
			Type: "duplicate-host-handlers", Subject: event,
			Message:     fmt.Sprintf("%d enabled handlers are independently visible to Codex for one lifecycle event: %s", len(declarations), strings.Join(keys, ", ")),
			Remediation: "fan component checks into one host-visible adapter and retain per-component receipts out of band",
		})
	}
	return diagnoses
}

func diagnoseEffectiveHook(hook codexEffectiveHook, trunkHead string, diagnoses []codexHookDiagnosis) (string, []codexHookDiagnosis) {
	add := func(kind, message, remediation string) {
		diagnoses = append(diagnoses, codexHookDiagnosis{
			Type:        kind,
			Subject:     hook.Key,
			Message:     message,
			Remediation: remediation,
		})
	}
	if !hook.Enabled {
		add(hookProfileStatusDisabled, "the effective hook is explicitly disabled", "enable it in /hooks or clear hooks.state.<key>.enabled")
		return hookProfileStatusDisabled, diagnoses
	}
	switch hook.TrustStatus {
	case "modified":
		add(hookProfileStatusStaleHash, "the current hook hash differs from the trusted hash", "review the command in /hooks and trust the new hash only if the change is expected")
		return hookProfileStatusStaleHash, diagnoses
	case "untrusted":
		add(hookProfileStatusUnknown, "the hook has no accepted trust hash", "review and trust the hook through /hooks")
		return hookProfileStatusUnknown, diagnoses
	case "trusted", "managed":
	default:
		add(hookProfileStatusUnknown, "the app-server returned an unrecognized trust state", "upgrade Codex or fak, then rerun the profile probe")
		return hookProfileStatusUnknown, diagnoses
	}
	if hook.HandlerType != "command" {
		add(hookProfileStatusUnknown, "executable identity is unavailable for a non-command hook handler", "inspect the named MCP/prompt/agent handler in /hooks")
		return hookProfileStatusUnknown, diagnoses
	}
	if len(hook.Executables) == 0 {
		add(hookProfileStatusUnknown, "the command was loaded but its executable identity could not be derived", "rewrite the hook to begin with or reference an absolute executable, then rerun the probe")
		return hookProfileStatusUnknown, diagnoses
	}
	hasRunnable := false
	hasStale := false
	for i := range hook.Executables {
		executable := &hook.Executables[i]
		if executable.Exists {
			hasRunnable = true
		}
		if executable.Build != "" && trunkHead != "" && !revisionMatches(executable.Build, trunkHead) {
			executable.Stale = true
			hasStale = true
		}
	}
	if !hasRunnable {
		add(hookProfileStatusMissingExecutable, "none of the command's derived executable identities exists", "install the named executable or update the hook command to an absolute existing path")
		return hookProfileStatusMissingExecutable, diagnoses
	}
	if hasStale {
		add(hookProfileStatusStaleBinary, "the hook resolves a fak executable whose embedded build does not match the repository tip", "rebuild/install fak from the intended trunk and verify PATH before restarting Codex")
		return hookProfileStatusStaleBinary, diagnoses
	}
	return hookProfileStatusReady, diagnoses
}

func queryCodexHookProfile(input codexHookProfileQueryInput) (codexHookProfileBuildInput, error) {
	if input.ObservedAt.IsZero() {
		input.ObservedAt = time.Now()
	}
	if input.RecentLogRows <= 0 {
		input.RecentLogRows = 20_000
	}
	if strings.TrimSpace(input.WorkingDirectory) == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return codexHookProfileBuildInput{}, err
		}
		input.WorkingDirectory = cwd
	}
	codexBin, err := resolveCodexBinary(input.CodexBin)
	if err != nil {
		return codexHookProfileBuildInput{}, err
	}
	snapshot, err := queryCodexAppServerHooks(codexBin, input.CodexHome, input.WorkingDirectory, 20*time.Second)
	if err != nil {
		return codexHookProfileBuildInput{}, err
	}

	activeHome := snapshot.CodexHome
	declaredHome := strings.TrimSpace(input.CodexHome)
	if declaredHome == "" {
		declaredHome = strings.TrimSpace(os.Getenv("CODEX_HOME"))
	}
	defaultHome := ""
	if userHome, err := os.UserHomeDir(); err == nil {
		defaultHome = filepath.Join(userHome, ".codex")
	}
	activeLogDB := filepath.Join(activeHome, "logs_2.sqlite")
	logPath := input.LogDBPath
	if strings.TrimSpace(logPath) == "" {
		logPath = activeLogDB
	}
	repoRoot := input.RepoRoot
	if repoRoot == "" {
		repoRoot = input.WorkingDirectory
	}
	trunkHead := gitHead(repoRoot)

	built := codexHookProfileBuildInput{
		ObservedAt:         input.ObservedAt,
		WorkingDirectory:   input.WorkingDirectory,
		DeclaredCodexHome:  declaredHome,
		ActiveCodexHome:    activeHome,
		DefaultCodexHome:   defaultHome,
		AppServerUserAgent: snapshot.UserAgent,
		CodexExecutable:    inspectExecutable(codexBin, ""),
		LogDBPath:          logPath,
		ActiveLogDBPath:    activeLogDB,
		TrunkHead:          trunkHead,
		Homes:              collectCodexHomes(activeHome, declaredHome, defaultHome),
		Warnings:           append([]string(nil), snapshot.Warnings...),
		Errors:             append([]codexHookSourceError(nil), snapshot.Errors...),
	}
	for _, hook := range snapshot.Hooks {
		effective := codexEffectiveHook{
			Key:          hook.Key,
			EventName:    hook.EventName,
			HandlerType:  hook.HandlerType,
			Matcher:      hook.Matcher,
			Command:      hook.Command,
			SourcePath:   hook.SourcePath,
			Source:       hook.Source,
			PluginID:     hook.PluginID,
			DisplayOrder: hook.DisplayOrder,
			Enabled:      hook.Enabled,
			Managed:      hook.Managed,
			CurrentHash:  hook.CurrentHash,
			TrustStatus:  hook.TrustStatus,
		}
		effective.Executables = resolveHookExecutables(effective)
		built.Hooks = append(built.Hooks, effective)
	}
	built.RecentToolFailures = queryRecentCodexToolFailures(activeLogDB, input.RecentLogRows)
	return built, nil
}

type codexAppServerSnapshot struct {
	CodexHome string
	UserAgent string
	Hooks     []codexAppServerHook
	Warnings  []string
	Errors    []codexHookSourceError
}

type codexAppServerHook struct {
	Key          string `json:"key"`
	EventName    string `json:"eventName"`
	HandlerType  string `json:"handlerType"`
	Matcher      string `json:"matcher"`
	Command      string `json:"command"`
	SourcePath   string `json:"sourcePath"`
	Source       string `json:"source"`
	PluginID     string `json:"pluginId"`
	DisplayOrder int64  `json:"displayOrder"`
	Enabled      bool   `json:"enabled"`
	Managed      bool   `json:"isManaged"`
	CurrentHash  string `json:"currentHash"`
	TrustStatus  string `json:"trustStatus"`
}

type codexAppServerEnvelope struct {
	ID     json.RawMessage `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func queryCodexAppServerHooks(codexBin, codexHome, cwd string, timeout time.Duration) (codexAppServerSnapshot, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, codexBin, "app-server", "--stdio")
	cmd.Dir = cwd
	if codexHome != "" {
		cmd.Env = replaceEnv(os.Environ(), "CODEX_HOME", codexHome)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return codexAppServerSnapshot{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return codexAppServerSnapshot{}, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return codexAppServerSnapshot{}, err
	}
	defer func() {
		_ = stdin.Close()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	lines := make(chan []byte, 32)
	scanErr := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			lines <- append([]byte(nil), scanner.Bytes()...)
		}
		close(lines)
		scanErr <- scanner.Err()
	}()
	writeJSONLine := func(value any) error {
		raw, err := json.Marshal(value)
		if err != nil {
			return err
		}
		raw = append(raw, '\n')
		_, err = stdin.Write(raw)
		return err
	}
	readID := func(id int) (codexAppServerEnvelope, error) {
		for {
			select {
			case <-ctx.Done():
				return codexAppServerEnvelope{}, fmt.Errorf("codex app-server response timeout: %w", ctx.Err())
			case line, ok := <-lines:
				if !ok {
					err := <-scanErr
					if err != nil {
						return codexAppServerEnvelope{}, err
					}
					return codexAppServerEnvelope{}, fmt.Errorf("codex app-server closed before response: %s", strings.TrimSpace(stderr.String()))
				}
				var envelope codexAppServerEnvelope
				if json.Unmarshal(line, &envelope) != nil || len(envelope.ID) == 0 {
					continue
				}
				var got int
				if json.Unmarshal(envelope.ID, &got) == nil && got == id {
					if envelope.Error != nil {
						return codexAppServerEnvelope{}, fmt.Errorf("codex app-server %d: %s", envelope.Error.Code, envelope.Error.Message)
					}
					return envelope, nil
				}
			}
		}
	}

	if err := writeJSONLine(map[string]any{
		"id":     1,
		"method": "initialize",
		"params": map[string]any{
			"clientInfo":   map[string]any{"name": "fak-sessiondiag", "version": "1"},
			"capabilities": map[string]any{"experimentalApi": true},
		},
	}); err != nil {
		return codexAppServerSnapshot{}, err
	}
	initializeEnvelope, err := readID(1)
	if err != nil {
		return codexAppServerSnapshot{}, err
	}
	var initialize struct {
		CodexHome string `json:"codexHome"`
		UserAgent string `json:"userAgent"`
	}
	if err := json.Unmarshal(initializeEnvelope.Result, &initialize); err != nil {
		return codexAppServerSnapshot{}, fmt.Errorf("decode Codex initialize response: %w", err)
	}
	if err := writeJSONLine(map[string]any{"method": "initialized", "params": map[string]any{}}); err != nil {
		return codexAppServerSnapshot{}, err
	}
	if err := writeJSONLine(map[string]any{
		"id":     2,
		"method": "hooks/list",
		"params": map[string]any{"cwds": []string{cwd}},
	}); err != nil {
		return codexAppServerSnapshot{}, err
	}
	hooksEnvelope, err := readID(2)
	if err != nil {
		return codexAppServerSnapshot{}, err
	}
	var hooksResult struct {
		Data []struct {
			Hooks    []codexAppServerHook   `json:"hooks"`
			Warnings []string               `json:"warnings"`
			Errors   []codexHookSourceError `json:"errors"`
		} `json:"data"`
	}
	if err := json.Unmarshal(hooksEnvelope.Result, &hooksResult); err != nil {
		return codexAppServerSnapshot{}, fmt.Errorf("decode Codex hooks/list response: %w", err)
	}
	snapshot := codexAppServerSnapshot{CodexHome: initialize.CodexHome, UserAgent: initialize.UserAgent}
	for _, entry := range hooksResult.Data {
		snapshot.Hooks = append(snapshot.Hooks, entry.Hooks...)
		snapshot.Warnings = append(snapshot.Warnings, entry.Warnings...)
		snapshot.Errors = append(snapshot.Errors, entry.Errors...)
	}
	return snapshot, nil
}

func resolveCodexBinary(explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		path, err := filepath.Abs(explicit)
		if err == nil && fileExists(path) {
			return path, nil
		}
		if resolved, lookErr := exec.LookPath(explicit); lookErr == nil {
			return resolved, nil
		}
		return "", fmt.Errorf("Codex executable not found")
	}
	for _, name := range []string{"codex.exe", "codex"} {
		if resolved, err := exec.LookPath(name); err == nil && strings.EqualFold(filepath.Ext(resolved), ".exe") {
			return resolved, nil
		}
	}
	if runtime.GOOS == "windows" {
		appData := strings.TrimSpace(os.Getenv("APPDATA"))
		pattern := filepath.Join(appData, "npm", "node_modules", "@openai", "codex", "node_modules", "@openai", "codex-win32-*", "vendor", "*", "bin", "codex.exe")
		if matches, _ := filepath.Glob(pattern); len(matches) > 0 {
			sort.Strings(matches)
			return matches[len(matches)-1], nil
		}
	}
	return "", fmt.Errorf("Codex executable not found; pass --codex-bin")
}

func collectCodexHomes(active, declared, defaultHome string) []codexHomeObservation {
	seen := map[string]bool{}
	var paths []string
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		key := comparablePath(path)
		if seen[key] {
			return
		}
		seen[key] = true
		paths = append(paths, path)
	}
	add(active)
	add(declared)
	add(defaultHome)
	if userHome, err := os.UserHomeDir(); err == nil {
		matches, _ := filepath.Glob(filepath.Join(userHome, ".codex*"))
		for _, path := range matches {
			if fileExists(filepath.Join(path, "config.toml")) ||
				fileExists(filepath.Join(path, "logs_2.sqlite")) ||
				fileExists(filepath.Join(path, "PROFILE-NAME.txt")) {
				add(path)
			}
		}
	}
	homes := make([]codexHomeObservation, 0, len(paths))
	for _, path := range paths {
		info, err := os.Stat(path)
		exists := err == nil && info.IsDir()
		homes = append(homes, codexHomeObservation{
			Path:      path,
			Exists:    exists,
			Active:    equalProfilePath(path, active),
			HasConfig: fileExists(filepath.Join(path, "config.toml")),
			HasLogDB:  fileExists(filepath.Join(path, "logs_2.sqlite")),
		})
	}
	sort.SliceStable(homes, func(i, j int) bool {
		if homes[i].Active != homes[j].Active {
			return homes[i].Active
		}
		if equalProfilePath(homes[i].Path, defaultHome) != equalProfilePath(homes[j].Path, defaultHome) {
			return equalProfilePath(homes[i].Path, defaultHome)
		}
		return comparablePath(homes[i].Path) < comparablePath(homes[j].Path)
	})
	return homes
}

func resolveHookExecutables(hook codexEffectiveHook) []codexExecutableIdentity {
	command := hook.Command
	var identities []codexExecutableIdentity
	addPath := func(name, path string) {
		if path == "" {
			identities = append(identities, codexExecutableIdentity{Name: name})
			return
		}
		identities = append(identities, inspectExecutable(path, name))
	}

	if hook.PluginID != "" &&
		(strings.Contains(command, `"$root/bin/dos-hook"`) || strings.Contains(command, `$root/bin/dos-hook`)) {
		pluginRoot := filepath.Dir(filepath.Dir(hook.SourcePath))
		addPath("dos-hook", filepath.Join(pluginRoot, "bin", "dos-hook"))
		return identities
	}
	for _, name := range []string{"fak", "dos", "python3", "python", "py", "powershell.exe", "pwsh", "bash", "sh"} {
		if !commandContainsExecutable(command, name) {
			continue
		}
		path, err := lookExecutable(name)
		if err != nil {
			addPath(name, "")
		} else {
			addPath(name, path)
		}
		if name == "fak" || name == "dos" {
			break
		}
	}
	return identities
}

func lookExecutable(name string) (string, error) {
	if filepath.IsAbs(name) && fileExists(name) {
		return name, nil
	}
	if runtime.GOOS == "windows" {
		candidates := []string{name}
		if filepath.Ext(name) == "" {
			candidates = append([]string{name + ".exe", name + ".cmd", name + ".bat"}, candidates...)
		}
		for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
			dir = strings.Trim(strings.TrimSpace(dir), `"`)
			if dir == "" {
				continue
			}
			for _, candidate := range candidates {
				path := filepath.Join(dir, candidate)
				if fileExists(path) {
					return path, nil
				}
			}
		}
		return "", exec.ErrNotFound
	}
	if path, err := exec.LookPath(name); err == nil {
		return path, nil
	}
	return "", exec.ErrNotFound
}

func inspectExecutable(path, name string) codexExecutableIdentity {
	if name == "" {
		name = filepath.Base(path)
	}
	identity := codexExecutableIdentity{Name: name, Path: path, Exists: fileExists(path)}
	if !identity.Exists {
		return identity
	}
	if raw, err := os.ReadFile(path); err == nil {
		sum := sha256.Sum256(raw)
		identity.SHA256 = "sha256:" + hex.EncodeToString(sum[:])
	}
	base := strings.ToLower(filepath.Base(path))
	if strings.HasPrefix(base, "fak") || strings.HasPrefix(strings.ToLower(name), "fak") {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, path, "version")
		raw, err := cmd.CombinedOutput()
		if err == nil {
			identity.Version, identity.Build = parseFakVersion(string(raw))
		}
	} else if strings.HasPrefix(base, "codex") {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		raw, err := exec.CommandContext(ctx, path, "--version").CombinedOutput()
		if err == nil {
			identity.Version = strings.TrimSpace(string(raw))
		}
	}
	return identity
}

func parseFakVersion(raw string) (version, build string) {
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "build:"):
			build = strings.TrimSpace(strings.TrimPrefix(line, "build:"))
		case version == "" && line != "":
			version = line
		}
	}
	return version, build
}

func commandContainsExecutable(command, name string) bool {
	lower := strings.ToLower(command)
	name = strings.ToLower(name)
	for _, separator := range []string{" ", "\t", "\n", "\r", ";", "|", "&", "(", ")", "{", "}", "'", "\""} {
		lower = strings.ReplaceAll(lower, separator, "\x00")
	}
	for _, token := range strings.Split(lower, "\x00") {
		token = strings.TrimSpace(token)
		if token == name || filepath.Base(token) == name {
			return true
		}
	}
	return false
}

func queryRecentCodexToolFailures(logPath string, rowWindow int) codexRecentToolFailures {
	result := codexRecentToolFailures{
		LogPath:        logPath,
		RowWindow:      rowWindow,
		Interpretation: "router ERROR rows are failed tool calls, not proof that a PreToolUse hook refused them; refusal_signals counts only explicit deny/refuse markers",
	}
	if !fileExists(logPath) || rowWindow <= 0 {
		return result
	}
	python, err := sessionDiagPython()
	if err != nil {
		return result
	}
	script := `import json,re,sqlite3,sys
p=sys.argv[1]; n=int(sys.argv[2]); c=sqlite3.connect("file:"+p.replace("\\","/")+"?mode=ro",uri=True,timeout=5)
rows=c.execute("select coalesce(feedback_log_body,'') from logs where id>(select max(id)-? from logs) and target='codex_core::tools::router' and level='ERROR'",(n,)).fetchall()
b=[r[0] for r in rows]
ref=re.compile(r'(refus|denied|permissionDecision.{0,30}deny|outcome.{0,30}refuse|PRESTAGED_PATH_OVERLAP|OFF_TRUNK|MERGE_IN_PROGRESS|POLICY_BLOCK)',re.I)
parser=re.compile(r'(ParserError|usage:)',re.I); timeout=re.compile(r'(timed out|Exit code: 124)',re.I)
print(json.dumps(dict(router_errors=len(b),refusal_signals=sum(bool(ref.search(x)) for x in b),parser_errors=sum(bool(parser.search(x)) for x in b),timeouts=sum(bool(timeout.search(x)) for x in b))))`
	raw, err := exec.Command(python, "-c", script, logPath, strconv.Itoa(rowWindow)).Output()
	if err != nil {
		return result
	}
	var counts struct {
		RouterErrors   int `json:"router_errors"`
		RefusalSignals int `json:"refusal_signals"`
		ParserErrors   int `json:"parser_errors"`
		Timeouts       int `json:"timeouts"`
	}
	if json.Unmarshal(raw, &counts) == nil {
		result.RouterErrors = counts.RouterErrors
		result.RefusalSignals = counts.RefusalSignals
		result.ParserErrors = counts.ParserErrors
		result.Timeouts = counts.Timeouts
	}
	return result
}

func writeHookProfileReport(w io.Writer, report codexHookProfileReport) {
	fmt.Fprintf(w, "CODEX HOOK PROFILE %s\n", report.Verdict)
	fmt.Fprintf(w, "active_home=%s declared_home=%s match=%t\n", report.ActiveCodexHome, report.Profile.DeclaredCodexHome, report.Profile.Matches)
	fmt.Fprintf(w, "codex=%s version=%s trunk=%s\n", report.Profile.CodexExecutable.Path, report.Profile.CodexExecutable.Version, report.Profile.TrunkHead)
	fmt.Fprintf(w, "log_store=%s profile_match=%t exists=%t\n", report.LogStore.RequestedPath, report.LogStore.ProfileMatch, report.LogStore.Exists)
	fmt.Fprintf(w, "hook_semantics=%s\n", report.HookMergeSemantics)
	for _, hook := range report.Hooks {
		fmt.Fprintf(w, "[%d] %s %s layer=%s enabled=%t trust=%s hash=%s status=%s\n",
			hook.DisplayOrder, hook.EventName, hook.Key, hook.Layer, hook.Enabled, hook.TrustStatus, hook.CurrentHash, hook.Status)
		fmt.Fprintf(w, "  command=%s\n", hook.Command)
		for _, executable := range hook.Executables {
			fmt.Fprintf(w, "  executable=%s path=%s exists=%t build=%s sha256=%s\n",
				executable.Name, executable.Path, executable.Exists, executable.Build, executable.SHA256)
		}
	}
	fmt.Fprintf(w, "recent_tool_failures router_errors=%d refusal_signals=%d parser_errors=%d timeouts=%d rows=%d\n",
		report.RecentToolFailures.RouterErrors,
		report.RecentToolFailures.RefusalSignals,
		report.RecentToolFailures.ParserErrors,
		report.RecentToolFailures.Timeouts,
		report.RecentToolFailures.RowWindow)
	for _, diagnosis := range report.Diagnoses {
		fmt.Fprintf(w, "- %s subject=%s: %s; fix=%s\n", diagnosis.Type, diagnosis.Subject, diagnosis.Message, diagnosis.Remediation)
	}
}

func classifyHookLayer(hook codexEffectiveHook, cwd string) string {
	switch hook.Source {
	case "plugin":
		return "plugin"
	case "project":
		projectRoot := filepath.Dir(filepath.Dir(hook.SourcePath))
		if equalProfilePath(projectRoot, cwd) {
			return "workspace"
		}
		return "inherited"
	case "user":
		return "user"
	case "appServerArgs":
		return "session"
	case "system", "mdm", "cloudRequirements", "cloudManagedConfig", "legacyManagedConfigFile", "legacyManagedConfigMdm":
		return "managed"
	default:
		return hookFirstNonEmpty(hook.Source, "unknown")
	}
}

func gitHead(root string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	raw, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(raw))
}

func revisionMatches(build, head string) bool {
	build = strings.TrimSpace(strings.ToLower(build))
	head = strings.TrimSpace(strings.ToLower(head))
	return build != "" && head != "" && (strings.HasPrefix(head, build) || strings.HasPrefix(build, head))
}

func replaceEnv(env []string, key, value string) []string {
	prefix := strings.ToUpper(key) + "="
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		if strings.HasPrefix(strings.ToUpper(entry), prefix) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, key+"="+value)
}

func equalProfilePath(left, right string) bool {
	if strings.TrimSpace(left) == "" || strings.TrimSpace(right) == "" {
		return false
	}
	return comparablePath(left) == comparablePath(right)
}

func comparablePath(path string) string {
	path = filepath.Clean(path)
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return path
}

func fileExists(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func hookFirstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
