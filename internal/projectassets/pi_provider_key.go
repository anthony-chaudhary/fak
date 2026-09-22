package projectassets

import (
	"regexp"
	"strings"
)

// Pi resolves a provider apiKey whose value starts with "!" by EXECUTING the rest as a
// shell command (see pi's dist/core/resolve-config-value.js:resolveConfigValueOrThrow).
// On Windows that command is run through `bash -c` (utils/shell.js:getShellConfig), with an
// `execSync` fallback, and a failure surfaces to the operator as:
//
//	Error: API key auth failed for provider "<id>": Failed to resolve API key for
//	provider "<id>" from shell command: <command>
//
// The fragile-but-common form is a PowerShell one-liner that reads a key file:
//
//	!powershell -NoProfile -ExecutionPolicy Bypass -Command \
//	  "(Get-Content -LiteralPath 'C:\Users\me\.fak\keys\hive.key' -Raw).Trim()"
//
// It has two independent failure modes, both observed in the wild:
//
//  1. `powershell.exe` is only on PATH via `%SystemRoot%\System32\WindowsPowerShell\v1.0`.
//     A process launched with a trimmed/minimal PATH (GUI launch, cloud-synced env, a child
//     spawned with a reduced environment) cannot find it: bash exits 127
//     (`powershell: command not found`).
//  2. The fallback leg is `cmd.exe`, whose `type` builtin rejects a forward-slash path
//     (`type C:/Users/.../hive.key` exits 1), so it cannot rescue case 1.
//
// The replacement is deliberately dumb: read the file with a command that exists in BOTH
// legs and has no PATH dependency beyond coreutils/cmd, and quote the path exactly as a
// single-quoted literal so `$`/backslash handling cannot mangle it. `cat "<path>"` works
// under bash and under cmd's `cat` on Git-for-Windows installs; the canonical shape is
// therefore a small, auditable template rather than an arbitrary command.
//
// piShellCommandKeyPattern matches only the narrow, known-fragile family: a leading "!",
// a PowerShell invocation, and a Get-Content read of a key file. Anything else (a custom
// command, an env template, a literal) is left byte-identical — this is a repair, not a
// policy.
var piShellCommandKeyPattern = regexp.MustCompile(
	`(?is)^!\s*(?:powershell|pwsh)(?:\.exe)?\b.*?\bGet-Content\b.*?-LiteralPath\s+(?:'([^']*)'|"([^"]*)"|(\S+))\s*.*$`,
)

// PiProviderKeyCommandIsFragile reports whether an apiKey value takes the fragile
// PowerShell-reads-a-file form that Pi must execute as a subprocess.
func PiProviderKeyCommandIsFragile(apiKey string) bool {
	return piShellCommandKeyPattern.MatchString(strings.TrimSpace(apiKey))
}

// NormalizePiProviderKeyCommand rewrites a fragile `!powershell ... Get-Content ... <file>`
// apiKey into the PATH-independent `!cat '<file>'` form, returning the replacement and true
// when a rewrite applies. Any other value — including an empty string, a plain literal, an
// `$ENV_VAR` template, or an unrecognized command — is returned unchanged with ok=false, so
// an operator's deliberate configuration is never clobbered.
//
// The path is emitted inside single quotes so nothing in it is subject to shell expansion;
// an embedded single quote (the one character that cannot appear in a single-quoted shell
// string) makes the value unrepairable and it is left alone rather than mis-quoted.
func NormalizePiProviderKeyCommand(apiKey string) (string, bool) {
	trimmed := strings.TrimSpace(apiKey)
	match := piShellCommandKeyPattern.FindStringSubmatch(trimmed)
	if match == nil {
		return apiKey, false
	}

	keyPath := ""
	for _, candidate := range match[1:4] {
		if candidate != "" {
			keyPath = candidate
			break
		}
	}
	keyPath = strings.TrimSpace(keyPath)
	if keyPath == "" || strings.ContainsAny(keyPath, "'\"") {
		return apiKey, false
	}

	replacement := "!cat '" + keyPath + "'"
	if replacement == trimmed {
		return apiKey, false
	}
	return replacement, true
}

// repairPiProviderKeyCommand normalizes a provider map's apiKey in place, reporting whether
// it changed. It is the repair half of EnsurePiProviderConfigForWindow: a config written by
// hand (or by an older tool) is converged to the PATH-independent form on the next
// `fak pi config` run, so the failure cannot silently return.
func repairPiProviderKeyCommand(provider map[string]interface{}) bool {
	rawKey, ok := provider["apiKey"].(string)
	if !ok {
		return false
	}
	replacement, changed := NormalizePiProviderKeyCommand(rawKey)
	if !changed {
		return false
	}
	provider["apiKey"] = replacement
	return true
}
