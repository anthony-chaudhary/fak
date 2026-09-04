package agentopt

import (
	"fmt"
	"regexp"
	"strings"
	"sync"
)

// Family 13: Model routing, cascades & reasoning effort optimization.
//
// IntraModelEffortRouter classifies incoming turn context into fine-grained
// reasoning effort tiers (none, low, medium, high) and operational categories.
// This enables dynamic adjustment of thinking token budgets per turn:
// high-effort planning, replanning, and error recovery receive full reasoning budgets,
// routine tool mechanics bypass reasoning to minimize latency and token spend,
// and diagnostic/verification turns scale reasoning according to complexity.

// EffortTier represents the reasoning effort level allocated to a model turn.
type EffortTier string

const (
	// EffortNone designates zero reasoning tokens for sub-second, routine execution.
	EffortNone EffortTier = "none"
	// EffortLow designates a minimal thinking budget for straightforward checks and verdicts.
	EffortLow EffortTier = "low"
	// EffortMedium designates moderate reasoning for multi-step diagnostics, diffs, and synthesis.
	EffortMedium EffortTier = "medium"
	// EffortHigh designates full reasoning capability for planning, decomposition, and error recovery.
	EffortHigh EffortTier = "high"
)

// String returns the string representation of the EffortTier.
func (e EffortTier) String() string {
	return string(e)
}

// ThinkingBudget returns the canonical thinking token budget for this tier.
func (e EffortTier) ThinkingBudget() int {
	switch e {
	case EffortNone:
		return 0
	case EffortLow:
		return 256
	case EffortMedium:
		return 1024
	case EffortHigh:
		return 2048
	default:
		return 0
	}
}

// ProviderEffort returns the provider-neutral effort label ("none", "low", "medium", "high").
func (e EffortTier) ProviderEffort() string {
	return string(e)
}

// ProviderRepresentation maps the EffortTier to provider-specific configuration strings.
func (e EffortTier) ProviderRepresentation(provider string) string {
	p := strings.ToLower(strings.TrimSpace(provider))
	switch p {
	case "anthropic":
		return string(e)
	case "openai":
		return string(e)
	case "gemini":
		return string(e)
	default:
		return string(e)
	}
}

// IsValid reports whether the effort tier is one of the recognized tiers.
func (e EffortTier) IsValid() bool {
	switch e {
	case EffortNone, EffortLow, EffortMedium, EffortHigh:
		return true
	default:
		return false
	}
}

// ParseEffortTier parses a string into an EffortTier.
func ParseEffortTier(s string) (EffortTier, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "none", "0", "off":
		return EffortNone, nil
	case "low":
		return EffortLow, nil
	case "medium", "med":
		return EffortMedium, nil
	case "high":
		return EffortHigh, nil
	default:
		return EffortNone, fmt.Errorf("unknown effort tier: %q", s)
	}
}

// OperationalCategory represents the task classification for an agent turn.
type OperationalCategory string

const (
	// CategoryPlanAndDecompose covers initial prompts, high-level queries, and replan instructions.
	CategoryPlanAndDecompose OperationalCategory = "plan_and_decompose"
	// CategoryRoutineToolInvocation covers straightforward file inspections, directory listings, and formatting.
	CategoryRoutineToolInvocation OperationalCategory = "routine_tool_invocation"
	// CategoryDiagnosticAndVerification covers evaluating tool outputs, test passes, linter checks, and diffs.
	CategoryDiagnosticAndVerification OperationalCategory = "diagnostic_and_verification"
	// CategoryErrorRecovery covers compiler errors, test failures, kernel policy blocks, and panic traces.
	CategoryErrorRecovery OperationalCategory = "error_recovery"
)

// String returns the string representation of the OperationalCategory.
func (c OperationalCategory) String() string {
	return string(c)
}

// TurnContext captures the state, signals, and inputs of an incoming agent turn.
type TurnContext struct {
	// TurnIndex is the 0-based turn counter within the session trajectory.
	TurnIndex int `json:"turn_index"`
	// Role is the message role ("user", "assistant", "tool", "system").
	Role string `json:"role,omitempty"`
	// Prompt is the user prompt or instruction for this turn.
	Prompt string `json:"prompt,omitempty"`
	// SystemPrompt is the active system instructions, if any.
	SystemPrompt string `json:"system_prompt,omitempty"`

	// ToolName is the primary tool invoked or returning output.
	ToolName string `json:"tool_name,omitempty"`
	// ToolArgs are arguments passed to the tool.
	ToolArgs map[string]any `json:"tool_args,omitempty"`
	// ToolOutput contains stdout/data returned by the tool.
	ToolOutput string `json:"tool_output,omitempty"`
	// ToolError contains stderr or execution error text.
	ToolError string `json:"tool_error,omitempty"`
	// ToolCalls lists one or more tool invocations planned or in flight.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ExitCode is the process exit code, if applicable.
	ExitCode int `json:"exit_code,omitempty"`

	// IsInitial explicitly marks this turn as the initial prompt of a task.
	IsInitial bool `json:"is_initial,omitempty"`
	// IsReplan explicitly marks this turn as an explicit replanning request.
	IsReplan bool `json:"is_replan,omitempty"`
	// HasError explicitly flags an active error or failure state.
	HasError bool `json:"has_error,omitempty"`

	// RecentErrors carries recent failure messages in the current trajectory.
	RecentErrors []string `json:"recent_errors,omitempty"`
	// Metadata holds arbitrary key-value context for custom inspection.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// TurnClassification provides the complete adjudication for an agent turn.
type TurnClassification struct {
	Effort         EffortTier          `json:"effort"`
	Category       OperationalCategory `json:"category"`
	ThinkingBudget int                 `json:"thinking_budget"`
	Reason         string              `json:"reason"`
}

// IntraModelEffortRouter routes and classifies turns into reasoning effort tiers.
type IntraModelEffortRouter struct {
	mu            sync.RWMutex
	customTools   map[string]OperationalCategory
	customBudgets map[EffortTier]int
	defaultEffort EffortTier
}

// RouterOption configures an IntraModelEffortRouter.
type RouterOption func(*IntraModelEffortRouter)

// WithCustomBudget overrides the thinking token budget for an effort tier.
func WithCustomBudget(tier EffortTier, budget int) RouterOption {
	return func(r *IntraModelEffortRouter) {
		r.customBudgets[tier] = budget
	}
}

// WithToolCategory overrides or registers an operational category for a specific tool.
func WithToolCategory(toolName string, category OperationalCategory) RouterOption {
	return func(r *IntraModelEffortRouter) {
		r.customTools[strings.ToLower(strings.TrimSpace(toolName))] = category
	}
}

// WithDefaultEffort sets a fallback effort tier for unclassified turns.
func WithDefaultEffort(tier EffortTier) RouterOption {
	return func(r *IntraModelEffortRouter) {
		r.defaultEffort = tier
	}
}

// NewIntraModelEffortRouter constructs an IntraModelEffortRouter with optional configuration.
func NewIntraModelEffortRouter(opts ...RouterOption) *IntraModelEffortRouter {
	r := &IntraModelEffortRouter{
		customTools:   make(map[string]OperationalCategory),
		customBudgets: make(map[EffortTier]int),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

var defaultRouter = NewIntraModelEffortRouter()

// DefaultIntraModelEffortRouter returns the shared default router instance.
func DefaultIntraModelEffortRouter() *IntraModelEffortRouter {
	return defaultRouter
}

// ClassifyTurn classifies a TurnContext using the default router.
func ClassifyTurn(input TurnContext) EffortTier {
	return defaultRouter.ClassifyTurn(input)
}

// Classify evaluates a TurnContext using the default router.
func Classify(input TurnContext) TurnClassification {
	return defaultRouter.Classify(input)
}

// RegisterToolCategory registers or overrides an operational category for a tool name.
func (r *IntraModelEffortRouter) RegisterToolCategory(toolName string, category OperationalCategory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.customTools[strings.ToLower(strings.TrimSpace(toolName))] = category
}

// RegisterBudget sets a custom thinking budget for an effort tier.
func (r *IntraModelEffortRouter) RegisterBudget(tier EffortTier, budget int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.customBudgets[tier] = budget
}

// ThinkingBudget returns the effective token budget for a tier under this router.
func (r *IntraModelEffortRouter) ThinkingBudget(tier EffortTier) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if b, ok := r.customBudgets[tier]; ok {
		return b
	}
	return tier.ThinkingBudget()
}

// ClassifyTurn evaluates a turn context and returns its EffortTier.
func (r *IntraModelEffortRouter) ClassifyTurn(input TurnContext) EffortTier {
	return r.Classify(input).Effort
}

// ThinkingBudgetForTurn evaluates a turn context and returns its thinking token budget.
func (r *IntraModelEffortRouter) ThinkingBudgetForTurn(input TurnContext) int {
	classification := r.Classify(input)
	return classification.ThinkingBudget
}

// Regex matchers for error, diagnostic, and planning detection.
var (
	rePanicOrCrash = regexp.MustCompile(`(?i)(panic:\s|fatal error:\s|runtime error:\s|Traceback \(most recent call last\):|SIGSEGV|SIGBUS|SIGABRT|Segmentation fault|NullPointerException)`)
	reGoroutine    = regexp.MustCompile(`goroutine \d+ \[`)

	reTestFailure = regexp.MustCompile(`(?i)(---\s*FAIL:|FAIL\t|\[FAIL\]|FAIL:\s|FAILED\s*\((failures|errors)=|test(s)? failed|assertion failed|assert\s+failed)`)
	reFailLine    = regexp.MustCompile(`(?m)^FAIL(\s|$)`)

	reCompilerError = regexp.MustCompile(`(?i)(syntax error:|undefined:\s|cannot use |cannot find package|cannot find module|compilation error|compile error|build failed|imported and not used|declared and not used|type error|error\[E\d+\]|error TS\d+|gcc:\s*error|clang:\s*error)`)

	rePolicyBlock = regexp.MustCompile(`(?i)(POLICY_BLOCK|DENY\s*\(POLICY_BLOCK\)|refused by policy|permission denied|access denied|guard refusal|EPERM|EACCES|CLAIM_UNWITNESSED|divergence detected|invariant violation)`)

	reTestPass = regexp.MustCompile(`(?i)(---\s*PASS:|PASS\t|\bPASS\b|ok\s+\S+\s+[0-9.]+s|All tests passed|Tests:\s*\d+\s*passed|0 failures,\s*0 errors|SUCCESS)`)

	reDiffMarkers = regexp.MustCompile(`(?i)(diff --git\s|---\s+[ab]\/|\+\+\+\s+[ab]\/|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@)`)

	reSynthesis = regexp.MustCompile(`(?i)(synthesiz(e|is)|summarize\s+(the\s+)?(findings|results|test|verification|diagnostic|root\s+cause)|evaluate\s+(the\s+)?(findings|results|verification|diagnostic)|review\s+(the\s+)?(findings|results|verification)|cross-validate|status\s+audit)`)

	rePlanning = regexp.MustCompile(`(?i)(\b(replan|plan|decompose|break\s+down|step-by-step|architecture|architect|strategy|roadmap|milestone|high-level|system\s+design)\b|formulate\s+a\s+plan|propose\s+a\s+plan|implementation\s+plan)`)
)

var defaultRoutineTools = map[string]bool{
	"read":           true,
	"read_file":      true,
	"readfile":       true,
	"cat":            true,
	"head":           true,
	"tail":           true,
	"open_file":      true,
	"file_read":      true,
	"view_file":      true,
	"view":           true,
	"glob":           true,
	"find":           true,
	"list_files":     true,
	"file_search":    true,
	"locate_files":   true,
	"grep":           true,
	"search":         true,
	"rg":             true,
	"ripgrep":        true,
	"ag":             true,
	"content_search": true,
	"listdir":        true,
	"list_dir":       true,
	"ls":             true,
	"dir":            true,
	"format":         true,
	"fmt":            true,
	"gofmt":          true,
	"prettier":       true,
	"black":          true,
	"ruff_format":    true,
	"format_code":    true,
	"autofmt":        true,
	"tidy":           true,
	"stat":           true,
	"file_info":      true,
	"inspect_file":   true,
	"wc":             true,
	"pwd":            true,
	"which":          true,
	"env":            true,
	"echo":           true,
}

var defaultLinterTools = map[string]bool{
	"lint":          true,
	"linter":        true,
	"golangci-lint": true,
	"ruff":          true,
	"eslint":        true,
	"flake8":        true,
	"pylint":        true,
	"clippy":        true,
	"vet":           true,
	"govet":         true,
}

// Classify classifies a TurnContext into an EffortTier and OperationalCategory with deterministic reasoning.
func (r *IntraModelEffortRouter) Classify(input TurnContext) TurnClassification {
	r.mu.RLock()
	customTools := make(map[string]OperationalCategory, len(r.customTools))
	for k, v := range r.customTools {
		customTools[k] = v
	}
	r.mu.RUnlock()

	normTool := strings.ToLower(strings.TrimSpace(input.ToolName))

	// 0. Check for empty turn context.
	if input.Prompt == "" && input.ToolName == "" && input.ToolOutput == "" &&
		input.ToolError == "" && len(input.ToolCalls) == 0 && !input.HasError &&
		!input.IsInitial && !input.IsReplan && len(input.RecentErrors) == 0 &&
		input.ExitCode == 0 {
		return TurnClassification{
			Effort:         EffortNone,
			Category:       CategoryRoutineToolInvocation,
			ThinkingBudget: r.ThinkingBudget(EffortNone),
			Reason:         "empty turn context defaults to no effort",
		}
	}

	// 1. Error Recovery / Discrepancy -> EffortHigh
	// Catches compiler errors, test failures, panic traces, kernel policy blocks, and explicit error flags.
	if isErr, reason := r.detectErrorRecovery(input); isErr {
		return TurnClassification{
			Effort:         EffortHigh,
			Category:       CategoryErrorRecovery,
			ThinkingBudget: r.ThinkingBudget(EffortHigh),
			Reason:         reason,
		}
	}

	// Check if a custom tool registration dictates category.
	if cat, ok := customTools[normTool]; ok {
		switch cat {
		case CategoryErrorRecovery:
			return TurnClassification{
				Effort:         EffortHigh,
				Category:       CategoryErrorRecovery,
				ThinkingBudget: r.ThinkingBudget(EffortHigh),
				Reason:         fmt.Sprintf("custom tool %q registered as error recovery", normTool),
			}
		case CategoryPlanAndDecompose:
			return TurnClassification{
				Effort:         EffortHigh,
				Category:       CategoryPlanAndDecompose,
				ThinkingBudget: r.ThinkingBudget(EffortHigh),
				Reason:         fmt.Sprintf("custom tool %q registered as plan and decompose", normTool),
			}
		case CategoryRoutineToolInvocation:
			return TurnClassification{
				Effort:         EffortNone,
				Category:       CategoryRoutineToolInvocation,
				ThinkingBudget: r.ThinkingBudget(EffortNone),
				Reason:         fmt.Sprintf("custom tool %q registered as routine tool invocation", normTool),
			}
		case CategoryDiagnosticAndVerification:
			return TurnClassification{
				Effort:         EffortMedium,
				Category:       CategoryDiagnosticAndVerification,
				ThinkingBudget: r.ThinkingBudget(EffortMedium),
				Reason:         fmt.Sprintf("custom tool %q registered as diagnostic and verification", normTool),
			}
		}
	}

	// 2. Plan & Decompose: explicit replan instruction -> EffortHigh
	if input.IsReplan {
		return TurnClassification{
			Effort:         EffortHigh,
			Category:       CategoryPlanAndDecompose,
			ThinkingBudget: r.ThinkingBudget(EffortHigh),
			Reason:         "explicit replan instruction requested",
		}
	}

	// 3. Synthesis & Diagnostic evaluation prompts -> EffortMedium
	// If the user prompt specifically asks to synthesize or evaluate verification findings.
	if strings.TrimSpace(input.Prompt) != "" && reSynthesis.MatchString(input.Prompt) {
		return TurnClassification{
			Effort:         EffortMedium,
			Category:       CategoryDiagnosticAndVerification,
			ThinkingBudget: r.ThinkingBudget(EffortMedium),
			Reason:         "synthesis prompt evaluating findings and diagnostic state",
		}
	}

	// 4. Routine Tool Invocation -> EffortNone
	// Straightforward tool calls (Read, Glob, Grep, ListDir, routine formatting, file inspection).
	if isRoutine, reason := r.detectRoutineTool(input); isRoutine {
		return TurnClassification{
			Effort:         EffortNone,
			Category:       CategoryRoutineToolInvocation,
			ThinkingBudget: r.ThinkingBudget(EffortNone),
			Reason:         reason,
		}
	}

	// 5. Diagnostic & Verification: test passes, linter checks, diff inspections -> EffortLow or EffortMedium
	if isDiag, effort, reason := r.detectDiagnostic(input); isDiag {
		return TurnClassification{
			Effort:         effort,
			Category:       CategoryDiagnosticAndVerification,
			ThinkingBudget: r.ThinkingBudget(effort),
			Reason:         reason,
		}
	}

	// 6. Plan & Decompose: Initial user prompt, high-level query, explicit planning instruction -> EffortHigh
	if isPlan, reason := r.detectPlanAndDecompose(input); isPlan {
		return TurnClassification{
			Effort:         EffortHigh,
			Category:       CategoryPlanAndDecompose,
			ThinkingBudget: r.ThinkingBudget(EffortHigh),
			Reason:         reason,
		}
	}

	// Fallback handling
	if r.defaultEffort != "" {
		return TurnClassification{
			Effort:         r.defaultEffort,
			Category:       CategoryPlanAndDecompose,
			ThinkingBudget: r.ThinkingBudget(r.defaultEffort),
			Reason:         "fallback to configured default effort",
		}
	}

	// If prompt exists, default to high effort planning; otherwise routine none.
	if strings.TrimSpace(input.Prompt) != "" {
		return TurnClassification{
			Effort:         EffortHigh,
			Category:       CategoryPlanAndDecompose,
			ThinkingBudget: r.ThinkingBudget(EffortHigh),
			Reason:         "unclassified user prompt defaults to plan and decompose",
		}
	}

	return TurnClassification{
		Effort:         EffortNone,
		Category:       CategoryRoutineToolInvocation,
		ThinkingBudget: r.ThinkingBudget(EffortNone),
		Reason:         "unclassified turn without prompt defaults to routine execution",
	}
}

// detectErrorRecovery checks for active errors, crashes, test failures, compiler failures, and policy blocks.
func (r *IntraModelEffortRouter) detectErrorRecovery(input TurnContext) (bool, string) {
	if input.HasError {
		return true, "explicit error flag set"
	}
	if input.ExitCode != 0 {
		return true, fmt.Sprintf("non-zero process exit code: %d", input.ExitCode)
	}
	if strings.TrimSpace(input.ToolError) != "" {
		return true, "tool execution error emitted"
	}
	if len(input.RecentErrors) > 0 {
		return true, fmt.Sprintf("trajectory carries %d recent errors", len(input.RecentErrors))
	}

	textToScan := input.ToolOutput + "\n" + input.ToolError

	// Panic & crash detection
	if rePanicOrCrash.MatchString(textToScan) || reGoroutine.MatchString(textToScan) {
		return true, "panic or runtime crash stack trace detected"
	}

	// Policy blocks and kernel refusals
	if rePolicyBlock.MatchString(textToScan) {
		return true, "kernel policy block or discrepancy detected"
	}

	// Test failure detection
	if reTestFailure.MatchString(textToScan) || reFailLine.MatchString(textToScan) {
		return true, "test failure trace detected"
	}

	// Compiler / build errors
	if reCompilerError.MatchString(textToScan) {
		return true, "compiler or build error detected"
	}

	// Check prompt for explicit error recovery instruction if prompt has error context
	if strings.TrimSpace(input.Prompt) != "" {
		pLower := strings.ToLower(input.Prompt)
		if strings.Contains(pLower, "fix compiler error") ||
			strings.Contains(pLower, "fix the build") ||
			strings.Contains(pLower, "fix test failure") ||
			strings.Contains(pLower, "debug the failure") ||
			strings.Contains(pLower, "recover from panic") {
			return true, "prompt requests error recovery and fix"
		}
		if rePanicOrCrash.MatchString(input.Prompt) || reCompilerError.MatchString(input.Prompt) {
			return true, "error or crash trace provided in prompt"
		}
	}

	return false, ""
}

// detectRoutineTool checks if the turn represents straightforward tool invocation.
func (r *IntraModelEffortRouter) detectRoutineTool(input TurnContext) (bool, string) {
	normTool := strings.ToLower(strings.TrimSpace(input.ToolName))

	if defaultRoutineTools[normTool] {
		return true, fmt.Sprintf("routine tool invocation: %s", normTool)
	}

	// Check bash command if tool is a shell/terminal tool
	if normTool == "bash" || normTool == "exec" || normTool == "sh" || normTool == "terminal" {
		if cmd, ok := input.ToolArgs["command"].(string); ok {
			if isRoutineBashCommand(cmd) {
				return true, fmt.Sprintf("routine shell command: %s", cmd)
			}
		}
	}

	// Check ToolCalls list
	if len(input.ToolCalls) > 0 {
		allRoutine := true
		for _, tc := range input.ToolCalls {
			cName := strings.ToLower(strings.TrimSpace(tc.Name))
			if !defaultRoutineTools[cName] {
				allRoutine = false
				break
			}
		}
		if allRoutine {
			return true, fmt.Sprintf("batch execution of %d routine tool calls", len(input.ToolCalls))
		}
	}

	// Routine file inspection prompt
	if strings.TrimSpace(input.Prompt) != "" && input.ToolName == "" && input.ToolOutput == "" {
		pLower := strings.ToLower(strings.TrimSpace(input.Prompt))
		if strings.HasPrefix(pLower, "read ") ||
			strings.HasPrefix(pLower, "cat ") ||
			strings.HasPrefix(pLower, "glob ") ||
			strings.HasPrefix(pLower, "grep ") ||
			strings.HasPrefix(pLower, "list files") ||
			strings.HasPrefix(pLower, "ls ") {
			return true, "routine file inspection prompt"
		}
	}

	return false, ""
}

// detectDiagnostic checks for test passes, linter checks, and diff inspections.
func (r *IntraModelEffortRouter) detectDiagnostic(input TurnContext) (bool, EffortTier, string) {
	normTool := strings.ToLower(strings.TrimSpace(input.ToolName))

	// 1. Diff inspections -> EffortMedium
	if normTool == "git_diff" || normTool == "diff" || normTool == "review_diff" {
		return true, EffortMedium, "diff tool inspection"
	}
	if normTool == "bash" || normTool == "exec" || normTool == "sh" {
		if cmd, ok := input.ToolArgs["command"].(string); ok {
			cLower := strings.ToLower(cmd)
			if strings.Contains(cLower, "git diff") && !strings.Contains(cLower, "--stat") {
				return true, EffortMedium, "git diff inspection"
			}
		}
	}
	if reDiffMarkers.MatchString(input.ToolOutput) {
		return true, EffortMedium, "unified diff patch inspection"
	}
	if strings.TrimSpace(input.Prompt) != "" {
		pLower := strings.ToLower(input.Prompt)
		if strings.Contains(pLower, "review diff") ||
			strings.Contains(pLower, "inspect diff") ||
			strings.Contains(pLower, "check diff") ||
			strings.Contains(pLower, "review changes") ||
			strings.Contains(pLower, "inspect git diff") {
			return true, EffortMedium, "prompt requests diff inspection"
		}
	}

	// 2. Linter checks -> EffortLow
	if defaultLinterTools[normTool] {
		return true, EffortLow, fmt.Sprintf("routine linter check: %s", normTool)
	}
	if normTool == "bash" || normTool == "exec" || normTool == "sh" {
		if cmd, ok := input.ToolArgs["command"].(string); ok {
			cLower := strings.ToLower(cmd)
			for linter := range defaultLinterTools {
				if strings.Contains(cLower, linter) {
					return true, EffortLow, fmt.Sprintf("linter invocation in shell: %s", linter)
				}
			}
		}
	}

	// 3. Test passes -> EffortLow
	if reTestPass.MatchString(input.ToolOutput) {
		return true, EffortLow, "test pass verification"
	}

	// 4. Benchmark evaluation -> EffortMedium
	if strings.Contains(input.ToolOutput, "ns/op") || strings.Contains(input.ToolOutput, "B/op") {
		return true, EffortMedium, "benchmark diagnostic evaluation"
	}

	return false, EffortNone, ""
}

// detectPlanAndDecompose checks for planning, initial prompts, and high-level queries.
func (r *IntraModelEffortRouter) detectPlanAndDecompose(input TurnContext) (bool, string) {
	if input.IsInitial {
		return true, "explicit initial task turn"
	}

	// TurnIndex 0 with a prompt represents the task entry point
	if input.TurnIndex == 0 && strings.TrimSpace(input.Prompt) != "" {
		return true, "initial user prompt decomposition"
	}

	if strings.TrimSpace(input.Prompt) != "" {
		if rePlanning.MatchString(input.Prompt) {
			return true, "prompt contains planning or decomposition directives"
		}

		// High-level query without active tool interaction
		if input.ToolName == "" && input.ToolOutput == "" && len(input.ToolCalls) == 0 {
			return true, "high-level query without tool execution"
		}
	}

	return false, ""
}

// isRoutineBashCommand reports whether a shell command is a standard read/format/find inspection.
func isRoutineBashCommand(cmd string) bool {
	c := strings.TrimSpace(cmd)
	if c == "" {
		return false
	}

	nonRoutineKeywords := []string{
		"go test", "npm test", "pytest", "cargo test", "make test",
		"go build", "cargo build", "make build", "npm run build",
		"golangci-lint", "ruff check", "eslint", "flake8", "clippy",
	}
	for _, nr := range nonRoutineKeywords {
		if strings.Contains(c, nr) {
			return false
		}
	}

	fields := strings.Fields(c)
	if len(fields) == 0 {
		return false
	}
	base := strings.ToLower(fields[0])

	switch base {
	case "ls", "cat", "head", "tail", "find", "grep", "rg", "ag", "wc", "pwd", "which", "echo":
		return true
	case "gofmt", "prettier", "black":
		return true
	case "git":
		if len(fields) > 1 {
			sub := strings.ToLower(fields[1])
			if sub == "status" || sub == "branch" {
				return true
			}
			if sub == "diff" && strings.Contains(c, "--stat") {
				return true
			}
		}
	}
	return false
}
