package projectassets

import (
	"regexp"
	"strings"
)

const (
	piHiveAPIBaseURL      = "https://api-cdn.thehive.ai/api/v3"
	piHiveAPIKeyReference = "${HIVE_API_KEY}"
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
// The managed Hive provider does not need a shell command. Fak-managed Pi launches inject
// HIVE_API_KEY into Pi's environment, and Pi expands ${HIVE_API_KEY} directly. This avoids
// shell selection entirely, including Windows machines where Pi finds WSL's bash first.
//
// piShellCommandKeyPattern matches only the narrow, known-fragile family: a leading "!",
// a PowerShell invocation, and a Get-Content read of a key file. Anything else (a custom
// command, an env template, a literal) is left byte-identical — this is a repair, not a
// policy.
var piShellCommandKeyPattern = regexp.MustCompile(
	`(?is)^!\s*(?:powershell|pwsh)(?:\.exe)?\b.*?\bGet-Content\b.*?-LiteralPath\s+(?:'([^']*)'|"([^"]*)"|(\S+))\s*.*$`,
)

var piCatKeyPattern = regexp.MustCompile(
	`(?is)^!\s*cat(?:\.exe)?\s+(?:'([^']*)'|"([^"]*)"|(\S+))\s*$`,
)

// PiProviderKeyCommandIsFragile reports whether an apiKey value takes the fragile
// PowerShell-reads-a-file form that Pi must execute as a subprocess.
func PiProviderKeyCommandIsFragile(apiKey string) bool {
	return piShellCommandKeyPattern.MatchString(strings.TrimSpace(apiKey))
}

// NormalizePiProviderKeyCommand rewrites a fragile `!powershell ... Get-Content ... <file>`
// apiKey into Pi's environment reference. Any other value is returned unchanged.
func NormalizePiProviderKeyCommand(apiKey string) (string, bool) {
	trimmed := strings.TrimSpace(apiKey)
	match := piShellCommandKeyPattern.FindStringSubmatch(trimmed)
	if match == nil || !piLegacyHiveKeyPath(firstPiKeyPath(match)) {
		return apiKey, false
	}
	return piHiveAPIKeyReference, true
}

func firstPiKeyPath(match []string) string {
	for _, candidate := range match[1:] {
		if candidate != "" {
			return strings.TrimSpace(candidate)
		}
	}
	return ""
}

func piLegacyHiveKeyPath(path string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(path), `\`, "/"))
	return normalized == ".fak/keys/hive.key" || strings.HasSuffix(normalized, "/.fak/keys/hive.key")
}

func piLegacyHiveCatCommand(apiKey string) bool {
	match := piCatKeyPattern.FindStringSubmatch(strings.TrimSpace(apiKey))
	return match != nil && piLegacyHiveKeyPath(firstPiKeyPath(match))
}

// repairPiProviderKeyCommand repairs only fak's managed Hive provider. Other endpoints and
// custom provider commands remain operator-owned. Legacy !cat values are repaired alongside
// the older PowerShell form so `fak pi config` cannot preserve the Windows failure.
func repairPiProviderKeyCommand(provider map[string]interface{}) bool {
	baseURL, _ := provider["baseUrl"].(string)
	if strings.TrimRight(strings.TrimSpace(baseURL), "/") != piHiveAPIBaseURL {
		return false
	}
	rawKey, ok := provider["apiKey"].(string)
	if !ok {
		return false
	}
	trimmed := strings.TrimSpace(rawKey)
	if trimmed == piHiveAPIKeyReference {
		return false
	}
	_, changed := NormalizePiProviderKeyCommand(rawKey)
	if !changed && !piLegacyHiveCatCommand(trimmed) {
		return false
	}
	provider["apiKey"] = piHiveAPIKeyReference
	return true
}
