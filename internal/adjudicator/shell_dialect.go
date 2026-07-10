package adjudicator

import "strings"

// defaultShellDialectDenyRegex is the one canonical spelling of the shipped
// cross-shell-dialect deny_regex (cmd/fak/guard-default-policy.json). Like the
// rm_rf / RCE-pipe rules it is RECOGNISED by this exact string and then decided
// STRUCTURALLY (commandLeadsWithPowerShellCmdlet) rather than by the raw regex —
// a raw `\bGet-Content\b` would false-positive on the cmdlet name appearing as an
// argument (`grep Get-Content file`) or inside quotes (`echo 'Get-Content'`). A
// policy that ships a different spelling is unaffected and keeps the raw-regex
// path. The enumerated alternation mirrors powerShellCmdlets so the policy file
// reads as documentation of what the rule catches. [#3941]
const defaultShellDialectDenyRegex = `(?i)\b(?:Select-Object|Get-ChildItem|Get-Content|Get-Item|Get-ItemProperty|Get-Process|Get-Command|Where-Object|ForEach-Object|Measure-Object|Sort-Object|Group-Object|Format-Table|Format-List|Out-File|Set-Content|Add-Content|New-Item|Remove-Item|Copy-Item|Move-Item|Rename-Item|Test-Path|Set-Location|Get-Location|Select-String|Write-Output|Write-Host|Invoke-WebRequest|Invoke-RestMethod)\b`

// powerShellCmdlets is the curated allow-set of PowerShell cmdlets that a model
// routinely submits to the POSIX Bash tool by mistake, keyed by their lowercased
// name (rceProgramBasename lowercases) → canonical spelling for the refusal hint.
// It is an EXPLICIT list, not a bare `Verb-Noun` shape match: a generic shape
// would deny a legitimately Verb-Noun-named POSIX binary. Every entry is a real
// cmdlet with no POSIX binary of the same name, so a command-word match is a
// wrong-shell error with certainty. Extend it as new confusions surface. [#3941]
var powerShellCmdlets = map[string]string{
	"select-object":     "Select-Object",
	"get-childitem":     "Get-ChildItem",
	"get-content":       "Get-Content",
	"get-item":          "Get-Item",
	"get-itemproperty":  "Get-ItemProperty",
	"get-process":       "Get-Process",
	"get-command":       "Get-Command",
	"where-object":      "Where-Object",
	"foreach-object":    "ForEach-Object",
	"measure-object":    "Measure-Object",
	"sort-object":       "Sort-Object",
	"group-object":      "Group-Object",
	"format-table":      "Format-Table",
	"format-list":       "Format-List",
	"out-file":          "Out-File",
	"set-content":       "Set-Content",
	"add-content":       "Add-Content",
	"new-item":          "New-Item",
	"remove-item":       "Remove-Item",
	"copy-item":         "Copy-Item",
	"move-item":         "Move-Item",
	"rename-item":       "Rename-Item",
	"test-path":         "Test-Path",
	"set-location":      "Set-Location",
	"get-location":      "Get-Location",
	"select-string":     "Select-String",
	"write-output":      "Write-Output",
	"write-host":        "Write-Host",
	"invoke-webrequest": "Invoke-WebRequest",
	"invoke-restmethod": "Invoke-RestMethod",
}

// isShellDialectArgRule reports whether pr is the shipped cross-shell-dialect
// deny_regex on a Bash command arg. Bash-scoped like isRmRfArgRule / isRCEPipeArgRule
// (EqualFold ⇒ the lowercase `bash` harness alias matches too); a differently-spelled
// or non-Bash rule keeps the raw-regex path.
func isShellDialectArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || !strings.EqualFold(pr.Tool, "Bash") {
		return false
	}
	if pr.Arg != "command" && pr.Arg != "cmd" {
		return false
	}
	return pr.Re.String() == defaultShellDialectDenyRegex
}

// commandLeadsWithPowerShellCmdlet reports whether any pipeline stage of cmd leads
// with a PowerShell cmdlet command word — the deterministic signature of a cmdlet
// submitted to the POSIX Bash tool, which fails `command not found` (exit 127)
// before doing anything. It mirrors commandHasRecursiveForcedDelete: it unwraps
// sh -c / $() / backticks via rceShellSources, tokenizes each source (a quoted word
// is a single token, never a command) via rceShellSegments, and resolves each
// stage's command word past env-assign / env / sudo / command (rceCommandWord).
// The cmdlet must sit AT the command-word position — matching only there is what
// keeps a cmdlet name used as an argument (`grep Select-Object file`) or as quoted
// text (`echo 'Select-Object'`) from tripping the rule, exactly as rm_rf/RCE avoid
// their quoted/arg false positives. Returns the canonical cmdlet spelling for the
// refusal hint and true on the first match, else ("", false).
func commandLeadsWithPowerShellCmdlet(cmd string) (string, bool) {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rceCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			if canon, ok := powerShellCmdlets[rceProgramBasename(seg.argv[i])]; ok {
				return canon, true
			}
		}
	}
	return "", false
}
