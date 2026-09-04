package policy

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// DangerousGotchas re-exports the adjudicator's gotchas catalog.
func DangerousGotchas() []adjudicator.DangerousGotcha {
	return adjudicator.DangerousGotchas()
}

// DefaultDangerousGotchasRules converts the dangerous gotchas catalog into
// equivalent ArgRule definitions for policy manifests on standard shell tools.
func DefaultDangerousGotchasRules() []ArgRule {
	var rules []ArgRule
	gotchas := adjudicator.DangerousGotchas()
	tools := []struct {
		tool string
		arg  string
	}{
		{"Bash", "command"},
		{"PowerShell", "command"},
		{"shell_command", "command"},
		{"exec_command", "cmd"},
	}

	for _, g := range gotchas {
		denyRegex := gotchaDefaultRegex(g)
		if denyRegex == "" {
			continue
		}
		for _, t := range tools {
			rules = append(rules, ArgRule{
				Tool:      t.tool,
				Arg:       t.arg,
				DenyRegex: denyRegex,
				Reason:    abi.ReasonName(abi.ReasonPolicyBlock),
				Fix:       g.Remedy,
			})
		}
	}
	return rules
}

// DangerousGotchasRulesForTool converts the dangerous gotchas catalog into
// ArgRule definitions for a specific tool and argument.
func DangerousGotchasRulesForTool(tool, arg string) []ArgRule {
	var rules []ArgRule
	for _, g := range adjudicator.DangerousGotchas() {
		denyRegex := gotchaDefaultRegex(g)
		if denyRegex == "" {
			continue
		}
		rules = append(rules, ArgRule{
			Tool:      tool,
			Arg:       arg,
			DenyRegex: denyRegex,
			Reason:    abi.ReasonName(abi.ReasonPolicyBlock),
			Fix:       g.Remedy,
		})
	}
	return rules
}

func gotchaDefaultRegex(g adjudicator.DangerousGotcha) string {
	switch g.Category {
	case adjudicator.GotchaDestructiveDeletion:
		return `\brm\s+-[A-Za-z]*[rRfF]|(?i)\bRemove-Item\b[^|;\n]*-(Recurse|Force)\b|\b(shred|truncate|srm)\b`
	case adjudicator.GotchaRawDisk:
		return `\bmkfs\b|\bdd\s+if=|>\s*/dev/sd|(?i)\b(Format-Volume|Clear-Disk|Initialize-Disk)\b`
	case adjudicator.GotchaHostShellEvasion:
		return `:\(\)\s*\{|\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(bash|sh|python|perl|ruby|node)\b|(?i)\b(Invoke-WebRequest|iwr|curl|wget|Invoke-RestMethod|irm)\b[^|]*\|[^|]*\b(iex|Invoke-Expression)\b`
	case adjudicator.GotchaPrivilegeEscalation:
		return `\b(sudo|doas)\b|(?i)\bStart-Process\b[^|;\n]*-Verb\s+RunAs\b`
	case adjudicator.GotchaInfraTeardown:
		return `\bterraform(?:\.exe)?\b[^|;&\n]*\bdestroy\b|\bkubectl\s+delete\s+all\b|\baws\s+s3\s+rb\b[^|;&\n]*--force\b`
	case adjudicator.GotchaSystemDisruption:
		return `\bkill\s+-(?:9|KILL|SIGKILL)\s+1\b|\b(?:pkill|killall)\b[^|;&\n]*\bsystemd\b`
	default:
		return ""
	}
}
