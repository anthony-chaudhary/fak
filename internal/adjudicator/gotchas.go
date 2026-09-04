package adjudicator

import (
	"regexp"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// GotchaCategory represents a danger class for high-risk system operations.
type GotchaCategory string

const (
	GotchaDestructiveDeletion GotchaCategory = "destructive_deletion"
	GotchaRawDisk             GotchaCategory = "raw_disk"
	GotchaHostShellEvasion    GotchaCategory = "host_shell_evasion"
	GotchaPrivilegeEscalation GotchaCategory = "privilege_escalation"
	GotchaInfraTeardown       GotchaCategory = "infra_teardown"
	GotchaSystemDisruption    GotchaCategory = "system_disruption"
)

// DangerousGotcha specifies a single dangerous operation definition and detector.
type DangerousGotcha struct {
	ID         string
	Category   GotchaCategory
	DenyRuleID string
	Summary    string
	Remedy     string
	Match      func(tool string, command string) (denied bool, detail string)
}

var (
	gotchasMu      sync.RWMutex
	gotchasCatalog []DangerousGotcha
)

func init() {
	ResetDangerousGotchas()
}

// ResetDangerousGotchas restores the default gotchas catalog.
func ResetDangerousGotchas() {
	gotchasMu.Lock()
	defer gotchasMu.Unlock()
	gotchasCatalog = append([]DangerousGotcha(nil), defaultGotchas...)
}

// RegisterDangerousGotcha registers an additional gotcha or overrides an existing one by ID.
func RegisterDangerousGotcha(g DangerousGotcha) {
	gotchasMu.Lock()
	defer gotchasMu.Unlock()
	for i, existing := range gotchasCatalog {
		if existing.ID == g.ID {
			gotchasCatalog[i] = g
			return
		}
	}
	gotchasCatalog = append(gotchasCatalog, g)
}

// DangerousGotchas returns a copy of the current gotchas catalog.
func DangerousGotchas() []DangerousGotcha {
	gotchasMu.RLock()
	defer gotchasMu.RUnlock()
	out := make([]DangerousGotcha, len(gotchasCatalog))
	copy(out, gotchasCatalog)
	return out
}

var defaultGotchas = []DangerousGotcha{
	{
		ID:         "destructive_deletion",
		Category:   GotchaDestructiveDeletion,
		DenyRuleID: abi.DenyRuleRmRf,
		Summary:    "Destructive file or directory deletion",
		Remedy:     "use repository scratchpad or target individual files without recursive forced flags",
		Match: func(tool, cmd string) (bool, string) {
			return matchDestructiveDeletions(cmd)
		},
	},
	{
		ID:         "raw_disk",
		Category:   GotchaRawDisk,
		DenyRuleID: abi.DenyRuleDeviceOp,
		Summary:    "Raw block device or storage destruction",
		Remedy:     "device and volume formatting operations require human operator approval",
		Match: func(tool, cmd string) (bool, string) {
			return matchRawDiskDestruction(cmd)
		},
	},
	{
		ID:         "host_shell_evasion",
		Category:   GotchaHostShellEvasion,
		DenyRuleID: abi.DenyRuleRCEPipe,
		Summary:    "Host shell evasion, pipe-to-shell RCE, or fork bomb",
		Remedy:     "download scripts to a file and review before executing; do not pipe remote content into shells",
		Match: func(tool, cmd string) (bool, string) {
			return matchHostShellEvasion(cmd)
		},
	},
	{
		ID:         "privilege_escalation",
		Category:   GotchaPrivilegeEscalation,
		DenyRuleID: abi.DenyRuleSudoLocal,
		Summary:    "Local privilege escalation",
		Remedy:     "run commands without privilege escalation or ask the operator to perform elevated setup",
		Match: func(tool, cmd string) (bool, string) {
			return matchPrivilegeEscalation(cmd)
		},
	},
	{
		ID:         "infra_teardown",
		Category:   GotchaInfraTeardown,
		DenyRuleID: abi.DenyRuleTerraformDestroy,
		Summary:    "Cloud infrastructure teardown",
		Remedy:     "preview infrastructure changes with terraform plan -destroy or kubectl diff/dry-run",
		Match: func(tool, cmd string) (bool, string) {
			return matchCloudInfraTeardown(cmd)
		},
	},
	{
		ID:         "system_disruption",
		Category:   GotchaSystemDisruption,
		DenyRuleID: "system_disruption",
		Summary:    "Critical system process disruption",
		Remedy:     "do not terminate system init (PID 1) or supervisor processes (systemd)",
		Match: func(tool, cmd string) (bool, string) {
			return matchSystemDisruption(cmd)
		},
	},
}

func matchDestructiveDeletions(cmd string) (bool, string) {
	ws, scratch := outOfTreeRoots()
	if commandHasUnsafeRecursiveForcedDelete(cmd, ws, scratch) {
		return true, "recursive or forced deletion (rm / Remove-Item)"
	}
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rmDeleteCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			prog := rceProgramBasename(seg.argv[i])
			switch prog {
			case "shred":
				targets := extractShredTargets(seg.argv[i+1:])
				if len(targets) == 0 || !deleteTargetsStrictlyInScratch(targets, ws, scratch) {
					return true, "shred"
				}
			case "truncate":
				targets := extractTruncateTargets(seg.argv[i+1:])
				if len(targets) == 0 || !deleteTargetsStrictlyInScratch(targets, ws, scratch) {
					return true, "truncate"
				}
			case "srm":
				targets := extractSrmTargets(seg.argv[i+1:])
				if len(targets) == 0 || !deleteTargetsStrictlyInScratch(targets, ws, scratch) {
					return true, "srm"
				}
			}
		}
	}
	return false, ""
}

func extractShredTargets(args []string) []string {
	var targets []string
	optionsDone := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !optionsDone && arg == "--" {
			optionsDone = true
			continue
		}
		if !optionsDone && strings.HasPrefix(arg, "-") {
			if (arg == "-n" || arg == "-s" || arg == "--random-source") && i+1 < len(args) {
				i++
			}
			continue
		}
		targets = append(targets, arg)
	}
	return targets
}

func extractTruncateTargets(args []string) []string {
	var targets []string
	optionsDone := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !optionsDone && arg == "--" {
			optionsDone = true
			continue
		}
		if !optionsDone && strings.HasPrefix(arg, "-") {
			if (arg == "-s" || arg == "-r" || arg == "--size" || arg == "--reference") && i+1 < len(args) {
				i++
			}
			continue
		}
		targets = append(targets, arg)
	}
	return targets
}

func extractSrmTargets(args []string) []string {
	var targets []string
	optionsDone := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !optionsDone && arg == "--" {
			optionsDone = true
			continue
		}
		if !optionsDone && strings.HasPrefix(arg, "-") {
			continue
		}
		targets = append(targets, arg)
	}
	return targets
}

func matchRawDiskDestruction(cmd string) (bool, string) {
	if commandPerformsDeviceOperation(cmd) {
		return true, "raw block device or storage operation"
	}
	return false, ""
}

var forkBombRe = regexp.MustCompile(`(?i)(?:^|[;&|\n\s])(?::|\w+)\s*\(\s*\)\s*\{\s*(?::|\w+)\s*\|\s*(?::|\w+)\s*&\s*\}\s*;\s*(?::|\w+)`)

func commandHasForkBomb(cmd string) bool {
	sources := rceShellSources(cmd)
	if len(sources) == 0 {
		return false
	}
	for _, src := range sources {
		segs := rceShellSegments(src)
		allInert := len(segs) > 0
		for _, seg := range segs {
			i := rceCommandWord(seg.argv)
			if i < 0 || !deviceOpInertHeads[rceProgramBasename(seg.argv[i])] {
				allInert = false
				break
			}
		}
		if allInert {
			continue
		}
		unquoted := stripQuotedLiterals(src)
		if forkBombRe.MatchString(unquoted) {
			return true
		}
		if forkBombRe.MatchString(src) && !hasOnlyQuotedForkBomb(src) {
			return true
		}
	}
	return false
}

func stripQuotedLiterals(s string) string {
	var b strings.Builder
	var quote byte
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if quote != 0 {
			if ch == '\\' && quote == '"' && i+1 < len(s) {
				i++
				continue
			}
			if ch == quote {
				quote = 0
			}
			continue
		}
		if ch == '\'' || ch == '"' {
			quote = ch
			continue
		}
		b.WriteByte(ch)
	}
	return b.String()
}

func hasOnlyQuotedForkBomb(s string) bool {
	unquoted := stripQuotedLiterals(s)
	return !forkBombRe.MatchString(unquoted)
}

func matchHostShellEvasion(cmd string) (bool, string) {
	if commandHasRemotePipeToInterpreter(cmd) {
		return true, "pipe-to-shell remote code execution"
	}
	if commandHasForkBomb(cmd) {
		return true, "fork bomb"
	}
	return false, ""
}

func matchPrivilegeEscalation(cmd string) (bool, string) {
	if word, ok := commandLocalEscalationWord(cmd); ok {
		return true, word
	}
	if commandInvokesRunAsElevation(cmd) {
		return true, "Start-Process -Verb RunAs"
	}
	return false, ""
}

func matchCloudInfraTeardown(cmd string) (bool, string) {
	if commandAppliesTerraformDestroy(cmd) {
		return true, "terraform destroy"
	}
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rceCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			prog := rceProgramBasename(seg.argv[i])
			switch prog {
			case "kubectl":
				if kubectlDeletesAll(seg.argv[i+1:]) {
					return true, "kubectl delete all"
				}
			case "aws":
				if awsS3RbForce(seg.argv[i+1:]) {
					return true, "aws s3 rb --force"
				}
			}
		}
	}
	return false, ""
}

func kubectlDeletesAll(args []string) bool {
	deleteFound := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-") {
			if (arg == "-n" || arg == "--namespace" || arg == "-f" || arg == "--filename" ||
				arg == "-l" || arg == "--selector" || arg == "--context") && i+1 < len(args) {
				i++
			}
			continue
		}
		if !deleteFound {
			if strings.ToLower(arg) == "delete" {
				deleteFound = true
			} else {
				return false
			}
			continue
		}
		lower := strings.ToLower(arg)
		if lower == "all" || strings.HasPrefix(lower, "all,") || strings.Contains(lower, ",all,") || strings.HasSuffix(lower, ",all") {
			return true
		}
	}
	return false
}

func awsS3RbForce(args []string) bool {
	s3Found := false
	rbFound := false
	hasForce := false
	for _, arg := range args {
		lower := strings.ToLower(arg)
		if lower == "--force" {
			hasForce = true
		}
		if strings.HasPrefix(arg, "-") {
			continue
		}
		if !s3Found {
			if lower == "s3" {
				s3Found = true
			} else {
				return false
			}
			continue
		}
		if !rbFound {
			if lower == "rb" {
				rbFound = true
			} else {
				return false
			}
			continue
		}
	}
	return s3Found && rbFound && hasForce
}

func matchSystemDisruption(cmd string) (bool, string) {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rceCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			prog := rceProgramBasename(seg.argv[i])
			switch prog {
			case "kill":
				if killTargetsPID1(seg.argv[i+1:]) {
					return true, "kill targeting PID 1"
				}
			case "pkill", "killall":
				if pkillTargetsSystemd(seg.argv[i+1:]) {
					return true, "pkill/killall targeting systemd"
				}
			}
		}
	}
	return false, ""
}

func killTargetsPID1(args []string) bool {
	optionsDone := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !optionsDone && arg == "--" {
			optionsDone = true
			continue
		}
		if !optionsDone && strings.HasPrefix(arg, "-") {
			if (arg == "-s" || arg == "--signal" || arg == "-p" || arg == "--pid") && i+1 < len(args) {
				i++
			}
			continue
		}
		if arg == "1" {
			return true
		}
	}
	return false
}

func pkillTargetsSystemd(args []string) bool {
	for _, arg := range args {
		lower := strings.ToLower(arg)
		if lower == "systemd" || strings.HasPrefix(lower, "systemd") || strings.Contains(lower, "/systemd") {
			return true
		}
	}
	return false
}

// EvalDangerousGotchas checks whether any extracted command string from args matches
// a registered dangerous gotcha.
func EvalDangerousGotchas(tool string, args map[string]any) (abi.Verdict, bool) {
	if args == nil {
		return abi.Verdict{}, false
	}
	gotchas := DangerousGotchas()
	if len(gotchas) == 0 {
		return abi.Verdict{}, false
	}
	for _, key := range []string{"command", "cmd", "code", "script", "input"} {
		v, ok := args[key]
		if !ok || v == nil {
			continue
		}
		cmd, ok := v.(string)
		if !ok || strings.TrimSpace(cmd) == "" {
			continue
		}
		for _, g := range gotchas {
			if g.Match == nil {
				continue
			}
			if denied, detail := g.Match(tool, cmd); denied {
				meta := map[string]string{
					"gotcha_id":       g.ID,
					"gotcha_category": string(g.Category),
				}
				if g.Remedy != "" {
					meta["remedy"] = g.Remedy
					meta["fix"] = g.Remedy
				}
				if g.DenyRuleID != "" {
					if rule, ok := abi.DenyRuleID(g.DenyRuleID); ok {
						meta[abi.MetaDenyRule] = rule
					} else {
						meta[abi.MetaDenyRule] = g.DenyRuleID
					}
				}
				claim := tool + "." + key + " dangerous_gotcha:" + g.ID
				if detail != "" {
					claim += " (" + detail + ")"
				}
				return abi.Verdict{
					Kind:    abi.VerdictDeny,
					Reason:  abi.ReasonPolicyBlock,
					By:      "monitor/gotchas",
					Payload: abi.WitnessPayload{Claim: claim},
					Meta:    meta,
				}, true
			}
		}
	}
	return abi.Verdict{}, false
}
