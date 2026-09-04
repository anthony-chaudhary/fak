package policy

import (
	"encoding/json"
	"fmt"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// Tier constants formalizing the decoupled evaluation model.
const (
	TierFrozenCore         = "FROZEN_CORE"
	TierConvenienceSurface = "CONVENIENCE_SURFACE"
)

// Reason constants for decoupled evaluation decisions.
const (
	ReasonFrozenCoreSafetyViolation = "FROZEN_CORE_SAFETY_VIOLATION"
	ReasonConveniencePermitted      = "CONVENIENCE_PERMITTED"
	ReasonUnknownCommand            = "UNKNOWN_COMMAND"
)

// TierDecision is the structured adjudication outcome distinguishing physical
// safety floor decisions from developer convenience surface decisions.
type TierDecision struct {
	Allowed    bool   `json:"allowed"`
	Tier       string `json:"tier"`
	Reason     string `json:"reason"`
	Affordance string `json:"affordance,omitempty"`
}

// TieredEvaluator evaluates tool calls against a two-tier decoupled model:
// Tier 1 (Frozen Core) is the immutable physical safety floor gating SSRF,
// destructive root operations, out-of-tree writes, and secret exfiltration.
// Tier 2 (Convenience Surface) provides permissive developer tool allowances
// with affirmative guidance for unrecognised commands.
type TieredEvaluator struct {
	mu              sync.RWMutex
	allowedTools    map[string]bool
	allowedCommands map[string]bool
}

// DecoupledTierEvaluator is an alias for TieredEvaluator.
type DecoupledTierEvaluator = TieredEvaluator

// NewTieredEvaluator constructs a TieredEvaluator with default settings.
func NewTieredEvaluator() *TieredEvaluator {
	return &TieredEvaluator{
		allowedTools:    make(map[string]bool),
		allowedCommands: make(map[string]bool),
	}
}

// NewDecoupledTierEvaluator constructs a DecoupledTierEvaluator with default settings.
func NewDecoupledTierEvaluator() *TieredEvaluator {
	return NewTieredEvaluator()
}

// AllowTool registers an additional permitted tool on the convenience surface.
// This cannot amend or bypass the immutable frozen core safety floor.
func (e *TieredEvaluator) AllowTool(tool string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.allowedTools[strings.ToLower(strings.TrimSpace(tool))] = true
}

// AllowCommand registers an additional permitted command on the convenience surface.
// This cannot amend or bypass the immutable frozen core safety floor.
func (e *TieredEvaluator) AllowCommand(cmd string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.allowedCommands[strings.ToLower(strings.TrimSpace(cmd))] = true
}

// Evaluate evaluates a tool invocation against the decoupled tiers.
// Tier 1 (Frozen Core) is evaluated first and unconditionally; violations cannot
// be bypassed or amended by convenience rules. Tier 2 (Convenience Surface)
// permits developer tools and provides affirmative guidance for unknown commands.
func (e *TieredEvaluator) Evaluate(tool string, args any) TierDecision {
	details := parseCallDetails(tool, args)

	// Tier 1: Frozen Core (Immutable Safety Floor)
	if danger, reason, affordance := checkFrozenCoreSafety(details); danger {
		return TierDecision{
			Allowed:    false,
			Tier:       TierFrozenCore,
			Reason:     reason,
			Affordance: affordance,
		}
	}

	// Tier 2: Convenience Surface
	return e.evaluateConvenienceSurface(details)
}

type callDetails struct {
	tool      string
	toolLower string
	command   string
	path      string
	url       string
	raw       string
	data      map[string]any
	isShell   bool
	isWrite   bool
	isRead    bool
}

func parseCallDetails(tool string, args any) *callDetails {
	d := &callDetails{
		tool:      tool,
		toolLower: strings.ToLower(strings.TrimSpace(tool)),
		data:      make(map[string]any),
	}

	switch d.toolLower {
	case "bash", "sh", "shell", "shell_command", "powershell", "exec_command", "terminal", "zsh":
		d.isShell = true
	case "write", "write_file", "writefile", "edit", "edit_file", "editfile", "str_replace_editor", "create_file", "patch", "save_file", "append_file":
		d.isWrite = true
	case "read", "read_file", "readfile", "view", "cat":
		d.isRead = true
	}

	if args == nil {
		return d
	}

	switch v := args.(type) {
	case string:
		d.raw = v
		trimmed := strings.TrimSpace(v)
		if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
			_ = json.Unmarshal([]byte(trimmed), &d.data)
		}
		if len(d.data) == 0 {
			if d.isShell {
				d.command = trimmed
				d.data["command"] = trimmed
			} else if d.isWrite || d.isRead {
				d.path = trimmed
				d.data["path"] = trimmed
			} else if strings.Contains(trimmed, "http://") || strings.Contains(trimmed, "https://") {
				d.url = trimmed
				d.data["url"] = trimmed
			}
		}
	case []byte:
		d.raw = string(v)
		trimmed := strings.TrimSpace(d.raw)
		if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
			_ = json.Unmarshal(v, &d.data)
		}
		if len(d.data) == 0 {
			if d.isShell {
				d.command = trimmed
				d.data["command"] = trimmed
			} else if d.isWrite || d.isRead {
				d.path = trimmed
				d.data["path"] = trimmed
			}
		}
	case map[string]any:
		d.data = v
		b, _ := json.Marshal(v)
		d.raw = string(b)
	default:
		b, err := json.Marshal(v)
		if err == nil {
			d.raw = string(b)
			_ = json.Unmarshal(b, &d.data)
		}
	}

	// Extract command
	for _, k := range []string{"command", "cmd", "script", "input"} {
		if val, ok := d.data[k].(string); ok && val != "" {
			d.command = val
			break
		}
	}

	// Extract path
	for _, k := range []string{"filePath", "file_path", "path", "target", "destination", "dest", "file", "filename"} {
		if val, ok := d.data[k].(string); ok && val != "" {
			d.path = val
			break
		}
	}

	// Extract URL
	for _, k := range []string{"url", "uri", "endpoint", "host", "address", "target_url"} {
		if val, ok := d.data[k].(string); ok && val != "" {
			d.url = val
			break
		}
	}

	return d
}

var (
	reSSRFAddress       = regexp.MustCompile(`(?i)\b(169\.254\.169\.254|metadata\.google\.internal|fd00:ec2::254)\b`)
	reRmRfRoot          = regexp.MustCompile(`(?i)\brm\s+-[a-z]*[r][a-z]*[f][a-z]*\s+([/]|/\*|--no-preserve-root\s+[/]|[/]\s+--no-preserve-root)\b`)
	reRmFrRoot          = regexp.MustCompile(`(?i)\brm\s+-[a-z]*[f][a-z]*[r][a-z]*\s+([/]|/\*|--no-preserve-root\s+[/]|[/]\s+--no-preserve-root)\b`)
	reRmRecursiveForce  = regexp.MustCompile(`(?i)\brm\s+.*(-r|-R|--recursive)\s+.*(-f|--force)\s+([/]|/\*)\b`)
	reRmForceRecursive  = regexp.MustCompile(`(?i)\brm\s+.*(-f|--force)\s+.*(-r|-R|--recursive)\s+([/]|/\*)\b`)
	rePSDeleteRoot      = regexp.MustCompile(`(?i)\bRemove-Item\b.*(-Recurse|-Force)\b.*(-Recurse|-Force)\b.*([/]|C:\\[*]?)\b`)
	reOutRedirect       = regexp.MustCompile(`(?i)(?:>>?|tee\s+(?:-a\s+)?)\s*(?:/etc/|C:[\\/]Windows[\\/]|/var/|/usr/|/bin/|/sbin/|/boot/|/sys/|/dev/|/root/|\.\.[\\/])`)
	reOutOutputFlag     = regexp.MustCompile(`(?i)(?:-o\s+|--output[=\s]+)(?:/etc/|C:[\\/]Windows[\\/]|/var/|/usr/|/bin/|/sbin/|/boot/|/sys/|/dev/|/root/|\.\.[\\/])`)
	reOutCopyVerb       = regexp.MustCompile(`(?i)\b(cp|mv|install|rsync|ln)\b.*(?:\s/etc/|\sC:[\\/]Windows[\\/]|\s\.\.[\\/])`)
	reOutModifyingShell = regexp.MustCompile(`(?i)\b(echo|printf|cat)\b.*>>?\s*(?:/etc/|C:[\\/]Windows[\\/]|\.\.[\\/])`)
	reOutSystemTouch    = regexp.MustCompile(`(?i)\b(touch|sed\s+-i|rm)\b.*(?:/etc/|C:[\\/]Windows[\\/])`)
)

func checkFrozenCoreSafety(d *callDetails) (danger bool, reason string, affordance string) {
	// 1. Network SSRF / Cloud Metadata
	if isSSRFViolation(d) {
		return true, ReasonFrozenCoreSafetyViolation, "Immutable safety floor: network SSRF and cloud metadata access (169.254.169.254) are permanently refused"
	}

	// 2. Destructive Root Operations (rm -rf /)
	if isDestructiveRootViolation(d) {
		return true, ReasonFrozenCoreSafetyViolation, "Immutable safety floor: destructive root operations (rm -rf /) are permanently refused"
	}

	// 3. Out-of-tree writes (/etc/, C:\Windows\)
	if isOutOfTreeWriteViolation(d) {
		return true, ReasonFrozenCoreSafetyViolation, "Immutable safety floor: out-of-tree writes to system directories (/etc/, C:\\Windows\\) are permanently refused"
	}

	// 4. Secret Exfiltration
	if isSecretExfiltrationViolation(d) {
		return true, ReasonFrozenCoreSafetyViolation, "Immutable safety floor: secret and credential exfiltration attempts are permanently refused"
	}

	return false, "", ""
}

func isSSRFViolation(d *callDetails) bool {
	if host, _ := egressfloor.Classify(d.tool, d.data); host != "" {
		return true
	}
	if d.url != "" && (reSSRFAddress.MatchString(d.url) || strings.Contains(d.url, "169.254.169.254") || strings.Contains(d.url, "metadata.google.internal")) {
		return true
	}
	if d.isShell && (reSSRFAddress.MatchString(d.command) || strings.Contains(d.command, "169.254.169.254") || strings.Contains(d.command, "metadata.google.internal")) {
		return true
	}
	if strings.Contains(d.toolLower, "fetch") || strings.Contains(d.toolLower, "web") || strings.Contains(d.toolLower, "http") || strings.Contains(d.toolLower, "curl") {
		if reSSRFAddress.MatchString(d.raw) || strings.Contains(d.raw, "169.254.169.254") || strings.Contains(d.raw, "metadata.google.internal") {
			return true
		}
	}
	return false
}

func isDestructiveRootViolation(d *callDetails) bool {
	if !d.isShell {
		return false
	}
	cmd := strings.TrimSpace(d.command)
	if cmd == "" {
		return false
	}

	// Benign wrapper commands (echo, printf, grep, rg, git, docker) do not execute destructive root rm.
	firstWord := extractPrimaryBinary(cmd)
	if firstWord == "echo" || firstWord == "printf" || firstWord == "grep" || firstWord == "rg" || firstWord == "git" || firstWord == "docker" {
		return false
	}

	lower := strings.ToLower(cmd)

	// PowerShell Remove-Item recursive and forced targeting root
	if strings.Contains(lower, "remove-item") {
		hasRecurse := strings.Contains(lower, "-recurse") || strings.Contains(lower, " -r")
		hasForce := strings.Contains(lower, "-force") || strings.Contains(lower, " -f")
		if hasRecurse && hasForce {
			if strings.Contains(lower, " /") || strings.HasSuffix(lower, " /") || strings.Contains(lower, " c:\\") || strings.Contains(lower, " c:/") {
				return true
			}
		}
	}

	// POSIX rm with recursive and force targeting root
	if firstWord == "rm" || strings.Contains(cmd, " rm ") || strings.HasPrefix(cmd, "rm ") {
		fields := strings.Fields(cmd)
		hasRecurse := false
		hasForce := false
		targetsRoot := false

		for _, f := range fields {
			if f == "rm" || f == "sudo" || f == "time" || f == "nohup" || f == "xargs" || f == "env" {
				continue
			}
			if strings.HasPrefix(f, "-") {
				if strings.Contains(f, "r") || strings.Contains(f, "R") || f == "--recursive" {
					hasRecurse = true
				}
				if strings.Contains(f, "f") || strings.Contains(f, "F") || f == "--force" {
					hasForce = true
				}
				continue
			}
			if f == "/" || f == "/*" || strings.HasPrefix(f, "/etc") {
				targetsRoot = true
			}
		}

		if hasRecurse && hasForce && targetsRoot {
			return true
		}
	}

	return false
}

func isOutOfTreeWriteViolation(d *callDetails) bool {
	// Write/Edit file tool calls
	if d.isWrite {
		p := d.path
		if p == "" {
			p = d.raw
		}
		if isOutOfTreePath(p) {
			return true
		}
	}

	// Shell commands attempting writes
	if d.isShell {
		cmd := d.command
		if reOutRedirect.MatchString(cmd) || reOutOutputFlag.MatchString(cmd) || reOutCopyVerb.MatchString(cmd) || reOutModifyingShell.MatchString(cmd) || reOutSystemTouch.MatchString(cmd) {
			return true
		}
	}

	return false
}

func isOutOfTreePath(p string) bool {
	normalized := strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	clean := path.Clean(normalized)
	clean = strings.Trim(clean, "\"'")
	upper := strings.ToUpper(clean)

	if strings.HasPrefix(clean, "/etc/") || clean == "/etc" || strings.Contains(clean, "/etc/") {
		return true
	}
	if strings.HasPrefix(upper, "C:/WINDOWS/") || upper == "C:/WINDOWS" || strings.Contains(upper, "/WINDOWS/") {
		return true
	}
	for _, sys := range []string{"/var/", "/usr/", "/bin/", "/sbin/", "/boot/", "/sys/", "/dev/", "/root/"} {
		if strings.HasPrefix(clean, sys) {
			return true
		}
	}
	if strings.HasPrefix(clean, "../") || strings.Contains(clean, "/../") || clean == ".." {
		return true
	}

	return false
}

func isSecretExfiltrationViolation(d *callDetails) bool {
	if d.toolLower == "exfiltrate" || d.toolLower == "exfil" || d.toolLower == "secret_exfil" || d.toolLower == "leak_secrets" {
		return true
	}

	// Check if this is an outbound call attempting to send secrets
	isOutbound := strings.Contains(d.toolLower, "fetch") || strings.Contains(d.toolLower, "web") || strings.Contains(d.toolLower, "http") || strings.Contains(d.toolLower, "curl")
	if d.isShell {
		cmdLower := strings.ToLower(d.command)
		if strings.Contains(cmdLower, "curl") || strings.Contains(cmdLower, "wget") || strings.Contains(cmdLower, "invoke-webrequest") || strings.Contains(cmdLower, "invoke-restmethod") {
			isOutbound = true
		}
	}

	if isOutbound {
		findings := canon.Scan([]byte(d.raw))
		if findings.Secret {
			return true
		}
		// Match common secret prefixes directly
		for _, pattern := range canon.SecretPatterns {
			if pattern.MatchString(d.raw) {
				return true
			}
		}
	}

	return false
}

var safeDeveloperBinaries = map[string]bool{
	// Version control
	"git": true, "gh": true, "svn": true, "hg": true,
	// Compilers and interpreters
	"go": true, "gofmt": true, "golint": true, "golangci-lint": true,
	"node": true, "npm": true, "npx": true, "yarn": true, "pnpm": true, "bun": true, "deno": true,
	"tsc": true, "eslint": true, "prettier": true, "vitest": true, "jest": true, "mocha": true,
	"python": true, "python3": true, "py": true, "pip": true, "pip3": true,
	"pytest": true, "pytest-3": true, "pytest3": true, "poetry": true, "uv": true,
	"ruff": true, "black": true, "flake8": true, "mypy": true, "pylint": true,
	"cargo": true, "rustc": true, "rustfmt": true, "clippy": true,
	"gcc": true, "g++": true, "clang": true, "clang++": true, "make": true, "ninja": true, "cmake": true, "ctest": true,
	"java": true, "javac": true, "mvn": true, "gradle": true, "gradlew": true,
	"dotnet": true, "ruby": true, "gem": true, "bundle": true, "rake": true,
	// Common shell utilities
	"ls": true, "dir": true, "pwd": true, "cd": true, "cat": true, "head": true, "tail": true, "more": true, "less": true,
	"echo": true, "printf": true, "touch": true, "mkdir": true, "cp": true, "mv": true,
	"grep": true, "rg": true, "find": true, "findstr": true, "wc": true, "diff": true,
	"sort": true, "uniq": true, "cut": true, "tr": true, "sed": true, "awk": true,
	"tar": true, "zip": true, "unzip": true, "gzip": true, "gunzip": true,
	"which": true, "where": true, "type": true, "uname": true, "whoami": true, "date": true,
	"env": true, "printenv": true, "export": true, "set": true, "alias": true,
	"true": true, "false": true, "test": true, "[": true, "sleep": true, "clear": true, "chmod": true,
	// Container operations
	"docker": true, "docker-compose": true, "podman": true,
}

var permissiveDeveloperTools = map[string]bool{
	"read": true, "read_file": true, "readfile": true, "view": true,
	"edit": true, "edit_file": true, "editfile": true, "str_replace_editor": true,
	"write": true, "write_file": true, "writefile": true, "create_file": true,
	"glob": true, "list_files": true,
	"grep": true, "search_files": true,
	"git": true,
}

func (e *TieredEvaluator) evaluateConvenienceSurface(d *callDetails) TierDecision {
	e.mu.RLock()
	isExplicitTool := e.allowedTools[d.toolLower]
	e.mu.RUnlock()

	if isExplicitTool {
		return TierDecision{
			Allowed:    true,
			Tier:       TierConvenienceSurface,
			Reason:     ReasonConveniencePermitted,
			Affordance: fmt.Sprintf("Tool %q is explicitly permitted by convenience allowance", d.tool),
		}
	}

	if permissiveDeveloperTools[d.toolLower] {
		return TierDecision{
			Allowed:    true,
			Tier:       TierConvenienceSurface,
			Reason:     ReasonConveniencePermitted,
			Affordance: fmt.Sprintf("Developer tool %q is permitted on convenience surface", d.tool),
		}
	}

	if d.isShell {
		return e.evaluateShellConvenience(d)
	}

	// Unknown tool
	return TierDecision{
		Allowed:    false,
		Tier:       TierConvenienceSurface,
		Reason:     ReasonUnknownCommand,
		Affordance: fmt.Sprintf("Tool %q is not in the default convenience surface. Permitted developer tools: Read, Edit, Write, Glob, Grep, Git, and safe Shell commands. To use this tool, register it via AllowTool.", d.tool),
	}
}

func (e *TieredEvaluator) evaluateShellConvenience(d *callDetails) TierDecision {
	cmd := strings.TrimSpace(d.command)
	if cmd == "" {
		return TierDecision{
			Allowed:    true,
			Tier:       TierConvenienceSurface,
			Reason:     ReasonConveniencePermitted,
			Affordance: "Empty shell invocation permitted",
		}
	}

	// Check if all chained command segments are safe developer commands
	segments := splitCommandSegments(cmd)
	for _, seg := range segments {
		bin := extractPrimaryBinary(seg)
		if bin == "" {
			continue
		}

		e.mu.RLock()
		isExplicitCmd := e.allowedCommands[strings.ToLower(bin)]
		e.mu.RUnlock()

		if isExplicitCmd || safeDeveloperBinaries[strings.ToLower(bin)] {
			continue
		}

		// Unknown command binary
		return TierDecision{
			Allowed:    false,
			Tier:       TierConvenienceSurface,
			Reason:     ReasonUnknownCommand,
			Affordance: fmt.Sprintf("Command %q is not recognized in default convenience surface. Permitted developer commands include go, npm, cargo, python, git, and standard shell utilities. To use this command, register it via AllowCommand.", bin),
		}
	}

	return TierDecision{
		Allowed:    true,
		Tier:       TierConvenienceSurface,
		Reason:     ReasonConveniencePermitted,
		Affordance: "Safe developer shell execution permitted",
	}
}

func splitCommandSegments(cmd string) []string {
	// Simple delimiter splitting by ;, &&, ||, |, \n
	var segments []string
	curr := strings.Builder{}
	inSingle := false
	inDouble := false

	runes := []rune(cmd)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == '\'' && !inDouble {
			inSingle = !inSingle
			curr.WriteRune(r)
			continue
		}
		if r == '"' && !inSingle {
			inDouble = !inDouble
			curr.WriteRune(r)
			continue
		}
		if !inSingle && !inDouble {
			if r == ';' || r == '\n' {
				segments = append(segments, curr.String())
				curr.Reset()
				continue
			}
			if r == '|' {
				if i+1 < len(runes) && runes[i+1] == '|' {
					segments = append(segments, curr.String())
					curr.Reset()
					i++
					continue
				}
				segments = append(segments, curr.String())
				curr.Reset()
				continue
			}
			if r == '&' && i+1 < len(runes) && runes[i+1] == '&' {
				segments = append(segments, curr.String())
				curr.Reset()
				i++
				continue
			}
		}
		curr.WriteRune(r)
	}

	if curr.Len() > 0 {
		segments = append(segments, curr.String())
	}

	return segments
}

func extractPrimaryBinary(segment string) string {
	fields := strings.Fields(strings.TrimSpace(segment))
	if len(fields) == 0 {
		return ""
	}

	i := 0
	// Skip variable assignments e.g. FOO=bar
	for i < len(fields) && strings.Contains(fields[i], "=") && !strings.HasPrefix(fields[i], "-") {
		i++
	}

	// Skip common wrappers e.g. sudo, time, nohup, xargs, env, nice
	wrappers := map[string]bool{
		"sudo": true, "time": true, "nohup": true, "xargs": true, "env": true, "nice": true, "builtin": true,
	}

	for i < len(fields) {
		word := fields[i]
		// Skip flags on wrapper
		if strings.HasPrefix(word, "-") {
			i++
			continue
		}
		base := strings.ToLower(filepath.Base(word))
		if wrappers[base] {
			i++
			continue
		}
		return base
	}

	return ""
}
