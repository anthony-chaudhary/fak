package agent

import (
	"regexp"
	"strconv"
	"strings"
)

// Effort tier constants defining model reasoning capacity allocations.
const (
	EffortTierNone     = "none"
	EffortTierLow      = "low"
	EffortTierMedium   = "medium"
	EffortTierBalanced = "balanced"
	EffortTierAdaptive = "adaptive"
	EffortTierHigh     = "high"

	BudgetTierNone            = 0
	BudgetTierLow             = 256
	BudgetTierMedium          = 1024
	BudgetTierHigh            = 2048
	BudgetBalancedRoutineTool = 0
	BudgetBalancedError       = 1536
	BudgetBalancedDefault     = 768

	ReasoningProfileDefault    = "default"
	ReasoningProfileBaseline   = "baseline"
	ReasoningProfileDeepReason = "deep-reason"
)

// TurnAssessment carries signals about the current turn's workload to titrate
// adaptive and balanced reasoning budgets.
type TurnAssessment struct {
	ToolName        string // tool name invoked, e.g. "read", "glob", "grep", "cat"
	IsRoutineTool   bool   // explicit indicator of routine tool inspection
	IsErrorRecovery bool   // explicit indicator of error recovery
	ErrorMessage    string // error message or failure text
}

// IsRoutine reports whether the assessment indicates routine tool inspection
// (e.g. read/glob/grep/cat/etc.) that benefits from zero thinking overhead.
func (ta TurnAssessment) IsRoutine() bool {
	if ta.IsRoutineTool {
		return true
	}
	return isRoutineToolName(ta.ToolName)
}

// IsError reports whether the assessment indicates error recovery (compiler error,
// test failure, panic, policy block) requiring higher thinking capacity.
func (ta TurnAssessment) IsError() bool {
	if ta.IsErrorRecovery {
		return true
	}
	if ta.ErrorMessage != "" {
		return true
	}
	lower := strings.ToLower(ta.ToolName)
	return strings.Contains(lower, "error") || strings.Contains(lower, "fail") || strings.Contains(lower, "panic")
}

var reExitStatus = regexp.MustCompile(`(?i)\b(?:exit status|exit code)\s*[:=]?\s*(-?[0-9]+)\b`)

func hasNonZeroExitStatus(s string) bool {
	matches := reExitStatus.FindAllStringSubmatch(s, -1)
	for _, m := range matches {
		if len(m) >= 2 {
			code, err := strconv.Atoi(m[1])
			if err == nil && code != 0 {
				return true
			}
		}
	}
	return false
}

func isRoutineToolName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "read", "glob", "grep", "cat", "view", "read_file", "list_dir", "find", "file_search":
		return true
	}
	return false
}

// AssessTranscriptTurn inspects the trailing tool turns of a message transcript to determine
// whether the turn is routine tool inspection, error recovery, or normal work.
// It scans contiguous trailing tool messages so multi-tool batches with errors are detected.
func AssessTranscriptTurn(messages []Message) (TurnAssessment, bool) {
	if len(messages) == 0 {
		return TurnAssessment{}, false
	}
	lastIdx := len(messages) - 1
	if messages[lastIdx].Role != RoleTool {
		return TurnAssessment{}, false
	}

	ta := TurnAssessment{
		ToolName: messages[lastIdx].Name,
	}

	// Scan contiguous trailing tool messages backward.
	allRoutine := true
	for i := lastIdx; i >= 0 && messages[i].Role == RoleTool; i-- {
		msg := messages[i]
		lower := strings.ToLower(msg.Content)
		isRoutine := isRoutineToolName(msg.Name)
		if !isRoutine {
			allRoutine = false
		}
		if isRoutine {
			continue
		}
		if strings.Contains(lower, "compiler error") ||
			strings.Contains(lower, "test failure") ||
			strings.Contains(lower, "panic:") ||
			strings.Contains(lower, "panic") ||
			strings.Contains(lower, "policy block") ||
			strings.Contains(lower, "policy_block") ||
			hasNonZeroExitStatus(lower) {
			ta.IsErrorRecovery = true
			ta.ErrorMessage = msg.Content
			return ta, true
		}
	}

	if allRoutine {
		ta.IsRoutineTool = true
	}
	return ta, true
}

// ResolveReasoningProfile resolves a named reasoning profile into an effort tier
// and a default thinking token budget ceiling.
// Routine tool turns under default/baseline profiles run at default/medium effort
// with routine turns titrated down to zero thinking overhead (<2s overhead).
// Deep-reason delegates on-demand high-effort reasoning for complex tasks.
func ResolveReasoningProfile(profile string) (effort string, budget int) {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case "", ReasoningProfileDefault, ReasoningProfileBaseline:
		return EffortTierMedium, BudgetTierMedium
	case ReasoningProfileDeepReason, "deepreason", "deep_reason":
		return EffortTierHigh, BudgetTierHigh
	case EffortTierHigh:
		return EffortTierHigh, BudgetTierHigh
	case EffortTierMedium:
		return EffortTierMedium, BudgetTierMedium
	case EffortTierLow:
		return EffortTierLow, BudgetTierLow
	case EffortTierNone:
		return EffortTierNone, BudgetTierNone
	default:
		return EffortTierMedium, BudgetTierMedium
	}
}

// ResolveReasoningProfileBudget resolves a named reasoning profile, optional explicit budget override,
// and turn assessments into an effective effort tier and token budget.
// Under default/baseline, routine inspection turns are clamped to BudgetBalancedRoutineTool (0)
// for <2s overhead, error recovery receives BudgetBalancedError (1536), and normal turns get BudgetTierMedium (1024).
// Deep-reason returns high effort and BudgetTierHigh (2048).
func ResolveReasoningProfileBudget(profile string, explicitBudget *int, turnContext ...TurnAssessment) (effort string, budget int) {
	effort, defaultBudget := ResolveReasoningProfile(profile)
	if explicitBudget != nil && *explicitBudget >= 0 {
		return effort, *explicitBudget
	}
	p := strings.ToLower(strings.TrimSpace(profile))
	if p == "" || p == ReasoningProfileDefault || p == ReasoningProfileBaseline {
		for _, ta := range turnContext {
			if ta.IsError() {
				return effort, BudgetBalancedError
			}
		}
		for _, ta := range turnContext {
			if ta.IsRoutine() {
				return effort, BudgetBalancedRoutineTool
			}
		}
	}
	return effort, defaultBudget
}

// IsRoutineToolName reports whether a tool name represents a routine inspection tool.
func IsRoutineToolName(name string) bool {
	return isRoutineToolName(name)
}

// IsRoutineTurn reports whether the trailing transcript turn is classified as a routine tool turn.
func IsRoutineTurn(messages []Message) bool {
	ta, ok := AssessTranscriptTurn(messages)
	if !ok {
		return false
	}
	return ta.IsRoutine() && !ta.IsError()
}

// ValidReasoningProfiles returns the supported reasoning profile names.
func ValidReasoningProfiles() []string {
	return []string{ReasoningProfileDefault, ReasoningProfileBaseline, ReasoningProfileDeepReason}
}

// IsValidReasoningProfile checks if the given profile string is recognized.
func IsValidReasoningProfile(profile string) bool {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case "", ReasoningProfileDefault, ReasoningProfileBaseline, ReasoningProfileDeepReason:
		return true
	default:
		return false
	}
}

// ResolveEffortBudget resolves the reasoning token budget given an effort tier,
// an optional explicit token budget override, and optional turn context assessments.
// Explicit budget wins if set and >= 0.
func ResolveEffortBudget(effort string, explicitBudget *int, turnContext ...TurnAssessment) int {
	if explicitBudget != nil && *explicitBudget >= 0 {
		return *explicitBudget
	}

	tier := strings.ToLower(strings.TrimSpace(effort))
	switch tier {
	case EffortTierNone:
		return BudgetTierNone
	case EffortTierLow:
		return BudgetTierLow
	case EffortTierMedium, ReasoningProfileDefault, ReasoningProfileBaseline:
		for _, ta := range turnContext {
			if ta.IsError() {
				return BudgetBalancedError
			}
		}
		for _, ta := range turnContext {
			if ta.IsRoutine() {
				return BudgetBalancedRoutineTool
			}
		}
		return BudgetTierMedium
	case EffortTierHigh, ReasoningProfileDeepReason:
		return BudgetTierHigh
	case EffortTierBalanced, EffortTierAdaptive:
		// Error recovery takes precedence over routine inspection.
		for _, ta := range turnContext {
			if ta.IsError() {
				return BudgetBalancedError
			}
		}
		for _, ta := range turnContext {
			if ta.IsRoutine() {
				return BudgetBalancedRoutineTool
			}
		}
		return BudgetBalancedDefault
	default:
		return 0
	}
}
