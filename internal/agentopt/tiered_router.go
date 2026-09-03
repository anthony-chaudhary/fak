package agentopt

import (
	"fmt"
	"strings"
	"sync"
)

// Family 13: Model routing, cascades & cost optimization.
//
// TieredModelRouter routes tool call tasks to appropriate model tiers (T0 fast/small,
// T1 normal, T2 frontier) based on tool characteristics, task complexity, and prompt
// context. Routine operations such as formatting, linting, simple read/grep, and
// trivial JSON extraction are routed to fast local or small models (T0) to optimize
// latency and token cost, while complex multi-file edits and architectural tasks
// are routed to higher tiers (T1/T2).

// ModelTier represents the performance and cost tier for model routing.
type ModelTier string

const (
	// TierT0 designates fast, lightweight, or local models optimized for routine operations.
	TierT0 ModelTier = "T0"
	// TierT1 designates standard production models for single-file edits and normal tool tasks.
	TierT1 ModelTier = "T1"
	// TierT2 designates frontier models for complex multi-file edits and architecture tasks.
	TierT2 ModelTier = "T2"

	// T0, T1, T2 aliases for convenience.
	T0 = TierT0
	T1 = TierT1
	T2 = TierT2
)

// TargetModelTier defines the destination model tier for a routed tool task.
type TargetModelTier = ModelTier

// TierClassification captures the classification result of a task or tool call.
type TierClassification struct {
	Tier            ModelTier       `json:"tier"`
	TargetModelTier TargetModelTier `json:"target_model_tier"`
	Reason          string          `json:"reason"`
}

// RouteChoice captures the routing result for a tool call task.
type RouteChoice struct {
	TargetModelTier TargetModelTier    `json:"target_model_tier"`
	TargetEndpoint  string             `json:"target_endpoint"`
	Classification  TierClassification `json:"classification"`
	Reason          string             `json:"reason"`
}

// TierClassifier evaluates tool calls, arguments, and prompts to determine task complexity tier.
type TierClassifier struct {
	mu            sync.RWMutex
	customT0Tools map[string]bool
	customT1Tools map[string]bool
	customT2Tools map[string]bool
}

// NewTierClassifier creates an initialized TierClassifier.
func NewTierClassifier() *TierClassifier {
	return &TierClassifier{
		customT0Tools: make(map[string]bool),
		customT1Tools: make(map[string]bool),
		customT2Tools: make(map[string]bool),
	}
}

// RegisterToolTier registers a custom tool name to a specific model tier.
func (c *TierClassifier) RegisterToolTier(toolName string, tier ModelTier) {
	c.mu.Lock()
	defer c.mu.Unlock()
	normalized := strings.ToLower(strings.TrimSpace(toolName))
	delete(c.customT0Tools, normalized)
	delete(c.customT1Tools, normalized)
	delete(c.customT2Tools, normalized)

	switch tier {
	case TierT0:
		c.customT0Tools[normalized] = true
	case TierT2:
		c.customT2Tools[normalized] = true
	default:
		c.customT1Tools[normalized] = true
	}
}

// ClassifyToolCall evaluates the tool name, arguments, and prompt, returning ModelTier, Reason.
func (c *TierClassifier) ClassifyToolCall(toolName string, args map[string]any, prompt string) (ModelTier, string) {
	classification := c.Classify(toolName, args, prompt)
	return classification.Tier, classification.Reason
}

// ClassifyCall evaluates a ToolCall instance and prompt, returning ModelTier, Reason.
func (c *TierClassifier) ClassifyCall(call ToolCall, prompt string) (ModelTier, string) {
	return c.ClassifyToolCall(call.Name, call.Args, prompt)
}

// Classify evaluates task complexity and tool call requirements, returning a structured TierClassification.
func (c *TierClassifier) Classify(toolName string, args map[string]any, prompt string) TierClassification {
	c.mu.RLock()
	normalizedTool := strings.ToLower(strings.TrimSpace(toolName))
	isCustomT0 := c.customT0Tools[normalizedTool]
	isCustomT1 := c.customT1Tools[normalizedTool]
	isCustomT2 := c.customT2Tools[normalizedTool]
	c.mu.RUnlock()

	if isCustomT0 {
		return TierClassification{
			Tier:            TierT0,
			TargetModelTier: TierT0,
			Reason:          fmt.Sprintf("registered custom T0 tool: %s", toolName),
		}
	}
	if isCustomT2 {
		return TierClassification{
			Tier:            TierT2,
			TargetModelTier: TierT2,
			Reason:          fmt.Sprintf("registered custom T2 tool: %s", toolName),
		}
	}
	if isCustomT1 {
		return TierClassification{
			Tier:            TierT1,
			TargetModelTier: TierT1,
			Reason:          fmt.Sprintf("registered custom T1 tool: %s", toolName),
		}
	}

	normalizedPrompt := strings.ToLower(strings.TrimSpace(prompt))

	// 1. Check for built-in T0 tool names (routine formatting, linting, simple read/grep, trivial JSON extraction)
	if isRoutineFormatTool(normalizedTool) {
		return TierClassification{
			Tier:            TierT0,
			TargetModelTier: TierT0,
			Reason:          fmt.Sprintf("routine formatting tool: %s", toolName),
		}
	}
	if isRoutineLintTool(normalizedTool) {
		return TierClassification{
			Tier:            TierT0,
			TargetModelTier: TierT0,
			Reason:          fmt.Sprintf("routine linting tool: %s", toolName),
		}
	}
	if isSimpleReadGrepTool(normalizedTool) {
		return TierClassification{
			Tier:            TierT0,
			TargetModelTier: TierT0,
			Reason:          fmt.Sprintf("simple read/grep tool: %s", toolName),
		}
	}
	if isTrivialJSONTool(normalizedTool) {
		return TierClassification{
			Tier:            TierT0,
			TargetModelTier: TierT0,
			Reason:          fmt.Sprintf("trivial JSON extraction tool: %s", toolName),
		}
	}

	// 2. Check command / execution args for command-based tools (bash, exec, sh, command, etc.)
	if cmd, ok := extractCommand(args); ok {
		cmdLower := strings.ToLower(strings.TrimSpace(cmd))
		if isRoutineFormatCommand(cmdLower) {
			return TierClassification{
				Tier:            TierT0,
				TargetModelTier: TierT0,
				Reason:          fmt.Sprintf("routine formatting command: %s", cmd),
			}
		}
		if isRoutineLintCommand(cmdLower) {
			return TierClassification{
				Tier:            TierT0,
				TargetModelTier: TierT0,
				Reason:          fmt.Sprintf("routine linting command: %s", cmd),
			}
		}
		if isSimpleReadGrepCommand(cmdLower) {
			return TierClassification{
				Tier:            TierT0,
				TargetModelTier: TierT0,
				Reason:          fmt.Sprintf("simple read/grep command: %s", cmd),
			}
		}
		if isTrivialJSONCommand(cmdLower) {
			return TierClassification{
				Tier:            TierT0,
				TargetModelTier: TierT0,
				Reason:          fmt.Sprintf("trivial JSON extraction command: %s", cmd),
			}
		}
	}

	// 3. Check for Architecture Tasks (T2 frontier)
	if keyword, ok := matchArchitectureKeyword(normalizedPrompt, normalizedTool); ok {
		return TierClassification{
			Tier:            TierT2,
			TargetModelTier: TierT2,
			Reason:          fmt.Sprintf("architecture task detected: %s", keyword),
		}
	}

	// 4. Check for Complex Multi-File Edits (T2 frontier)
	if multiFileReason, ok := checkMultiFileEdits(args, normalizedPrompt); ok {
		return TierClassification{
			Tier:            TierT2,
			TargetModelTier: TierT2,
			Reason:          multiFileReason,
		}
	}

	// 5. Check Prompt for Routine T0 Tasks when tool is generic, bash, or empty
	if normalizedTool == "" || isGenericRunnerTool(normalizedTool) {
		if reason, ok := matchRoutinePrompt(normalizedPrompt); ok {
			return TierClassification{
				Tier:            TierT0,
				TargetModelTier: TierT0,
				Reason:          reason,
			}
		}
	}

	// 6. Standard Task (T1)
	if isStandardEditTool(normalizedTool) {
		return TierClassification{
			Tier:            TierT1,
			TargetModelTier: TierT1,
			Reason:          fmt.Sprintf("standard single-file edit or code modification: %s", toolName),
		}
	}

	return TierClassification{
		Tier:            TierT1,
		TargetModelTier: TierT1,
		Reason:          "standard complexity: default tier",
	}
}

func isRoutineFormatTool(name string) bool {
	switch name {
	case "format", "fmt", "gofmt", "prettier", "black", "ruff_format",
		"format_code", "code_format", "lint_fix", "autofmt", "rustfmt", "clang-format":
		return true
	default:
		return false
	}
}

func isRoutineLintTool(name string) bool {
	switch name {
	case "lint", "linter", "eslint", "flake8", "golangci-lint", "ruff",
		"pylint", "rubocop", "checkstyle", "clippy", "vet", "govet":
		return true
	default:
		return false
	}
}

func isSimpleReadGrepTool(name string) bool {
	switch name {
	case "read", "read_file", "cat", "head", "tail", "grep",
		"search_files", "glob", "list_dir", "ls", "file_search", "find_files", "view_file":
		return true
	default:
		return false
	}
}

func isTrivialJSONTool(name string) bool {
	switch name {
	case "extract_json", "json_extract", "parse_json", "jq", "json_parse", "format_json":
		return true
	default:
		return false
	}
}

func extractCommand(args map[string]any) (string, bool) {
	if args == nil {
		return "", false
	}
	for _, key := range []string{"command", "cmd", "script"} {
		if val, ok := args[key]; ok {
			if s, isStr := val.(string); isStr && strings.TrimSpace(s) != "" {
				return s, true
			}
		}
	}
	return "", false
}

func isRoutineFormatCommand(cmd string) bool {
	patterns := []string{
		"gofmt", "prettier", "black ", "ruff format", "npm run format",
		"yarn format", "cargo fmt", "rustfmt", "clang-format",
	}
	for _, p := range patterns {
		if strings.Contains(cmd, p) {
			return true
		}
	}
	return false
}

func isRoutineLintCommand(cmd string) bool {
	patterns := []string{
		"golangci-lint", "eslint", "flake8", "ruff check", "npm run lint",
		"yarn lint", "cargo clippy", "cargo check", "go vet", "pylint", "rubocop",
	}
	for _, p := range patterns {
		if strings.Contains(cmd, p) {
			return true
		}
	}
	return false
}

func isSimpleReadGrepCommand(cmd string) bool {
	prefixes := []string{
		"cat ", "grep ", "rg ", "ls ", "find ", "head ", "tail ", "wc ", "tree ",
	}
	trimmed := strings.TrimSpace(cmd)
	if trimmed == "ls" {
		return true
	}
	for _, p := range prefixes {
		if strings.HasPrefix(trimmed, p) {
			return true
		}
	}
	return false
}

func isTrivialJSONCommand(cmd string) bool {
	return strings.Contains(cmd, "jq ") || strings.Contains(cmd, "python -m json.tool")
}

func matchArchitectureKeyword(prompt, tool string) (string, bool) {
	keywords := []string{
		"architecture",
		"architectural",
		"system design",
		"distributed consensus",
		"concurrency invariant",
		"concurrency model",
		"lock ordering",
		"security audit",
		"threat model",
		"protocol migration",
		"formal verification",
		"core redesign",
		"high-risk reasoning",
		"cross-subsystem protocol",
		"frozen abi",
	}
	for _, kw := range keywords {
		if strings.Contains(prompt, kw) || strings.Contains(tool, kw) {
			return kw, true
		}
	}
	return "", false
}

func checkMultiFileEdits(args map[string]any, prompt string) (string, bool) {
	if args != nil {
		for _, key := range []string{"files", "file_paths", "paths", "targets"} {
			if val, exists := args[key]; exists {
				switch items := val.(type) {
				case []string:
					if len(items) > 1 {
						return fmt.Sprintf("complex multi-file edit: %d files specified in %s", len(items), key), true
					}
				case []any:
					if len(items) > 1 {
						return fmt.Sprintf("complex multi-file edit: %d files specified in %s", len(items), key), true
					}
				}
			}
		}
	}

	multiFileKeywords := []string{
		"multi-file",
		"multifile",
		"multiple files",
		"across multiple files",
		"across the codebase",
		"cross-cutting",
		"system-wide refactor",
		"repository-wide",
		"repo-wide",
		"sweep all packages",
		"refactor across",
	}
	for _, kw := range multiFileKeywords {
		if strings.Contains(prompt, kw) {
			return fmt.Sprintf("complex multi-file edit task: %s", kw), true
		}
	}

	return "", false
}

func matchRoutinePrompt(prompt string) (string, bool) {
	formatPrompts := []string{
		"format code", "formatting", "run gofmt", "run prettier", "run black",
		"format this file", "format file", "autoformat", "indentation fix", "routine formatting",
	}
	for _, p := range formatPrompts {
		if strings.Contains(prompt, p) {
			return fmt.Sprintf("routine formatting task in prompt: %s", p), true
		}
	}

	lintPrompts := []string{
		"run lint", "linting", "run linter", "check lint", "eslint", "ruff check",
		"golangci-lint", "fix lint", "style warnings", "routine lint",
	}
	for _, p := range lintPrompts {
		if strings.Contains(prompt, p) {
			return fmt.Sprintf("routine linting task in prompt: %s", p), true
		}
	}

	readPrompts := []string{
		"read file", "read the file", "grep for", "search files", "list files",
		"inspect file", "view file", "find pattern", "simple read", "simple grep",
	}
	for _, p := range readPrompts {
		if strings.Contains(prompt, p) {
			return fmt.Sprintf("routine read/grep task in prompt: %s", p), true
		}
	}

	jsonPrompts := []string{
		"extract json", "json extraction", "parse json", "convert to json",
		"format json", "parse the json", "trivial json",
	}
	for _, p := range jsonPrompts {
		if strings.Contains(prompt, p) {
			return fmt.Sprintf("trivial JSON extraction task in prompt: %s", p), true
		}
	}

	return "", false
}

func isGenericRunnerTool(name string) bool {
	switch name {
	case "bash", "sh", "exec", "terminal", "command", "run", "cmd", "tool", "agent":
		return true
	default:
		return false
	}
}

func isStandardEditTool(name string) bool {
	switch name {
	case "edit", "write", "patch", "replace", "update", "test", "build", "git":
		return true
	default:
		return false
	}
}

// DefaultTierEndpoints returns default endpoints for each model tier.
func DefaultTierEndpoints() map[ModelTier]string {
	return map[ModelTier]string{
		TierT0: "http://127.0.0.1:8080/v1/models/t0-fast",
		TierT1: "http://127.0.0.1:8080/v1/models/t1-standard",
		TierT2: "http://127.0.0.1:8080/v1/models/t2-frontier",
	}
}

// TieredModelRouter routes tool call tasks to model endpoints according to complexity tiers.
type TieredModelRouter struct {
	mu         sync.RWMutex
	classifier *TierClassifier
	endpoints  map[ModelTier]string
}

// NewTieredModelRouter constructs a router with optional endpoint overrides.
func NewTieredModelRouter(endpoints map[ModelTier]string) *TieredModelRouter {
	return NewTieredModelRouterWithClassifier(NewTierClassifier(), endpoints)
}

// NewTieredModelRouterWithClassifier constructs a router with an explicit classifier and endpoints.
func NewTieredModelRouterWithClassifier(classifier *TierClassifier, endpoints map[ModelTier]string) *TieredModelRouter {
	if classifier == nil {
		classifier = NewTierClassifier()
	}
	ep := DefaultTierEndpoints()
	for k, v := range endpoints {
		if v != "" {
			ep[k] = v
		}
	}
	return &TieredModelRouter{
		classifier: classifier,
		endpoints:  ep,
	}
}

// RouteToolTask evaluates task complexity and tool call requirements,
// returning TargetModelTier, TargetEndpoint.
func (r *TieredModelRouter) RouteToolTask(toolName string, args map[string]any, prompt string) (TargetModelTier, string) {
	choice := r.Route(toolName, args, prompt)
	return choice.TargetModelTier, choice.TargetEndpoint
}

// RouteCall routes a ToolCall instance and prompt, returning TargetModelTier, TargetEndpoint.
func (r *TieredModelRouter) RouteCall(call ToolCall, prompt string) (TargetModelTier, string) {
	return r.RouteToolTask(call.Name, call.Args, prompt)
}

// Route evaluates the tool call task and returns a structured RouteChoice.
func (r *TieredModelRouter) Route(toolName string, args map[string]any, prompt string) RouteChoice {
	classification := r.classifier.Classify(toolName, args, prompt)

	r.mu.RLock()
	endpoint, ok := r.endpoints[classification.TargetModelTier]
	if !ok || endpoint == "" {
		endpoint = r.endpoints[TierT1]
	}
	r.mu.RUnlock()

	return RouteChoice{
		TargetModelTier: classification.TargetModelTier,
		TargetEndpoint:  endpoint,
		Classification:  classification,
		Reason:          classification.Reason,
	}
}

// RouteCallChoice returns a complete RouteChoice for a ToolCall.
func (r *TieredModelRouter) RouteCallChoice(call ToolCall, prompt string) RouteChoice {
	return r.Route(call.Name, call.Args, prompt)
}

// Classifier returns the underlying TierClassifier.
func (r *TieredModelRouter) Classifier() *TierClassifier {
	return r.classifier
}

// EndpointForTier returns the registered endpoint for the given tier.
func (r *TieredModelRouter) EndpointForTier(tier ModelTier) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.endpoints[tier]
}

// SetEndpoint updates the registered endpoint for a given tier.
func (r *TieredModelRouter) SetEndpoint(tier ModelTier, endpoint string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.endpoints[tier] = endpoint
}

// RegisterToolTier registers a custom tool name to a specific model tier on the router's classifier.
func (r *TieredModelRouter) RegisterToolTier(toolName string, tier ModelTier) {
	r.classifier.RegisterToolTier(toolName, tier)
}
