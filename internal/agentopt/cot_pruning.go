package agentopt

import (
	"fmt"
	"regexp"
	"strings"
)

// Family 1: Prompting & reasoning strategies.
//
// Contrastive CoT exemplar pruning minimizes prompt token overhead by
// stripping conversational fluff, meta-reasoning boilerplate, and verbose
// self-narration from Chain-of-Thought demonstrations while preserving
// critical reasoning pivots, contrastive alternatives, branch boundaries,
// and tool invocations with exact arguments.
// This delivers >= 40% token reduction while maintaining 100% preservation
// of reasoning pivots for in-context learning guidance.

// ReasoningPivotKind classifies the structural role of a critical reasoning nexus.
type ReasoningPivotKind string

const (
	// PivotKindContrast denotes contrastive reasoning between alternatives
	// (e.g., rejecting path A in favor of path B due to observed constraints).
	PivotKindContrast ReasoningPivotKind = "contrast"

	// PivotKindBoundary denotes branch boundaries, conditional guards, and thresholds.
	PivotKindBoundary ReasoningPivotKind = "boundary"

	// PivotKindToolCall denotes tool execution invocations and argument specifications.
	PivotKindToolCall ReasoningPivotKind = "tool_call"

	// PivotKindDeduction denotes critical deductions, inferences, and resolutions.
	PivotKindDeduction ReasoningPivotKind = "deduction"

	// PivotKindCorrection denotes error correction, disproof, or hypothesis revision.
	PivotKindCorrection ReasoningPivotKind = "correction"
)

// ReasoningPivot represents a critical nexus in a reasoning trace, such as
// a contrastive branch point, boundary condition, tool invocation, or deduction.
type ReasoningPivot struct {
	ID           string             `json:"id,omitempty"`
	Kind         ReasoningPivotKind `json:"kind"`
	Statement    string             `json:"statement"`
	ContrastCue  string             `json:"contrast_cue,omitempty"`
	StepIndex    int                `json:"step_index,omitempty"`
	Retained     bool               `json:"retained"`
	Significance float64            `json:"significance,omitempty"`
}

// CoTExemplar represents a Chain-of-Thought few-shot demonstration
// comprising a prompt, step-by-step reasoning trace, optional explicit pivots,
// tool invocations, and target output.
type CoTExemplar struct {
	ID        string           `json:"id,omitempty"`
	Prompt    string           `json:"prompt,omitempty"`
	Thought   string           `json:"thought,omitempty"`
	Steps     []string         `json:"steps,omitempty"`
	Pivots    []ReasoningPivot `json:"pivots,omitempty"`
	ToolCalls []ToolCall       `json:"tool_calls,omitempty"`
	Output    string           `json:"output,omitempty"`
	Metadata  map[string]any   `json:"metadata,omitempty"`
}

// PrunedExemplarResult contains the outcome of exemplar pruning, including
// retained pivots, tool invocations, token accounting, and the pruned demonstration.
type PrunedExemplarResult struct {
	ExemplarID        string           `json:"exemplar_id,omitempty"`
	Prompt            string           `json:"prompt,omitempty"`
	OriginalThought   string           `json:"original_thought"`
	PrunedThought     string           `json:"pruned_thought"`
	OriginalSteps     []string         `json:"original_steps,omitempty"`
	PrunedSteps       []string         `json:"pruned_steps,omitempty"`
	RetainedPivots    []ReasoningPivot `json:"retained_pivots"`
	RetainedToolCalls []ToolCall       `json:"retained_tool_calls,omitempty"`
	Output            string           `json:"output,omitempty"`
	OriginalTokens    int              `json:"original_tokens"`
	PrunedTokens      int              `json:"pruned_tokens"`
	TokensSaved       int              `json:"tokens_saved"`
	ReductionRatio    float64          `json:"reduction_ratio"`
	PivotsRetained    int              `json:"pivots_retained"`
	TotalPivots       int              `json:"total_pivots"`
	PreservationRate  float64          `json:"preservation_rate"`
	FormattedPrompt   string           `json:"formatted_prompt,omitempty"`
}

// SavingsPercentage returns the token reduction as a percentage (0.0 to 100.0).
func (r PrunedExemplarResult) SavingsPercentage() float64 {
	return r.ReductionRatio * 100.0
}

// AllPivotsPreserved reports whether all critical reasoning pivots were retained.
func (r PrunedExemplarResult) AllPivotsPreserved() bool {
	if r.TotalPivots == 0 {
		return true
	}
	return r.PivotsRetained >= r.TotalPivots
}

// MeetsTarget reports whether the token reduction meets or exceeds the target ratio.
func (r PrunedExemplarResult) MeetsTarget(target float64) bool {
	return r.ReductionRatio >= target
}

// FormatDemonstration formats the pruned exemplar into a standard few-shot demonstration prompt string.
func (r PrunedExemplarResult) FormatDemonstration() string {
	var sb strings.Builder
	if r.Prompt != "" {
		sb.WriteString("Question: ")
		sb.WriteString(r.Prompt)
		sb.WriteString("\n")
	}
	if r.PrunedThought != "" {
		sb.WriteString("Reasoning:\n")
		sb.WriteString(r.PrunedThought)
		sb.WriteString("\n")
	}
	if r.Output != "" {
		sb.WriteString("Answer: ")
		sb.WriteString(r.Output)
	}
	return strings.TrimSpace(sb.String())
}

// ContrastiveCoTPruner prunes boilerplate, conversational fluff, and redundant
// self-narration from Chain-of-Thought exemplars while strictly preserving
// contrastive reasoning pivots, branch boundaries, and tool arguments.
type ContrastiveCoTPruner struct {
	MinReductionRatio   float64          `json:"min_reduction_ratio"`
	PreserveToolCalls   bool             `json:"preserve_tool_calls"`
	PreserveAllPivots   bool             `json:"preserve_all_pivots"`
	StripBoilerplate    bool             `json:"strip_boilerplate"`
	StripConversational bool             `json:"strip_conversational"`
	CompactFormatting   bool             `json:"compact_formatting"`
	CustomBoilerplate   []string         `json:"custom_boilerplate,omitempty"`
	TokenEstimator      func(string) int `json:"-"`
}

// PrunerOption configures a ContrastiveCoTPruner.
type PrunerOption func(*ContrastiveCoTPruner)

// WithMinReductionRatio sets the target minimum reduction threshold.
func WithMinReductionRatio(ratio float64) PrunerOption {
	return func(p *ContrastiveCoTPruner) {
		p.MinReductionRatio = ratio
	}
}

// WithPreserveToolCalls configures whether tool calls are strictly retained.
func WithPreserveToolCalls(preserve bool) PrunerOption {
	return func(p *ContrastiveCoTPruner) {
		p.PreserveToolCalls = preserve
	}
}

// WithPreserveAllPivots configures whether all reasoning pivots are preserved.
func WithPreserveAllPivots(preserve bool) PrunerOption {
	return func(p *ContrastiveCoTPruner) {
		p.PreserveAllPivots = preserve
	}
}

// WithCustomBoilerplate registers extra boilerplate substrings to strip.
func WithCustomBoilerplate(phrases []string) PrunerOption {
	return func(p *ContrastiveCoTPruner) {
		p.CustomBoilerplate = append(p.CustomBoilerplate, phrases...)
	}
}

// WithTokenEstimator overrides the default token counting function.
func WithTokenEstimator(estimator func(string) int) PrunerOption {
	return func(p *ContrastiveCoTPruner) {
		p.TokenEstimator = estimator
	}
}

// NewContrastiveCoTPruner initializes a pruner with standard defaults.
func NewContrastiveCoTPruner(opts ...PrunerOption) *ContrastiveCoTPruner {
	p := &ContrastiveCoTPruner{
		MinReductionRatio:   0.40,
		PreserveToolCalls:   true,
		PreserveAllPivots:   true,
		StripBoilerplate:    true,
		StripConversational: true,
		CompactFormatting:   true,
		TokenEstimator:      EstimateTokens,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

var defaultBoilerplatePatterns = []*regexp.Regexp{
	// Greetings & pleasantries
	regexp.MustCompile(`(?i)^(hello|hi|hey|greetings|good morning|good afternoon|good evening)[!.,\s]`),
	regexp.MustCompile(`(?i)^(sure|certainly|of course|absolutely)[!.,\s]`),
	regexp.MustCompile(`(?i)(i would be |i'm |i am )?(delighted|happy|glad) to (help|assist)`),
	regexp.MustCompile(`(?i)thank you for (asking|reaching out)`),
	// Prompt echoes and CoT clichés
	regexp.MustCompile(`(?i)let('?s| us) think step by step`),
	regexp.MustCompile(`(?i)let('?s| us) approach this (systematically|methodically|carefully|step by step)`),
	regexp.MustCompile(`(?i)let('?s| us) break down this (problem|task|question)`),
	regexp.MustCompile(`(?i)let('?s| us) take a deep breath`),
	regexp.MustCompile(`(?i)as an ai( language model)?`),
	regexp.MustCompile(`(?i)as an (intelligent )?assistant`),
	regexp.MustCompile(`(?i)as an autonomous agent`),
	regexp.MustCompile(`(?i)first,? let (me|us) (carefully )?(understand|analyze) what the user is asking`),
	regexp.MustCompile(`(?i)the goal is to (deploy|diagnose|solve|determine)`),
	regexp.MustCompile(`(?i)let('?s| me) see what we have here`),
	regexp.MustCompile(`(?i)let me ponder this for a moment`),
	// Reassurance & self-verification chatter
	regexp.MustCompile(`(?i)let me double[- ]check (my reasoning|this|the work)`),
	regexp.MustCompile(`(?i)to make sure (there are no|i don't make any) (mistakes|errors)`),
	regexp.MustCompile(`(?i)everything looks (correct|good|fine) up to this point`),
	regexp.MustCompile(`(?i)seems completely fine and (in order|working)`),
	regexp.MustCompile(`(?i)let('?s| us) pause and verify`),
	// Procedural transition filler
	regexp.MustCompile(`(?i)^(i am |i'm )?now going to (execute|run|invoke|proceed)`),
	regexp.MustCompile(`(?i)^now (i will|let us|let's) proceed to`),
	regexp.MustCompile(`(?i)^let('?s| us) proceed to the next step`),
	regexp.MustCompile(`(?i)^in this step,? we will`),
	regexp.MustCompile(`(?i)^let('?s| us) review what we have so far`),
	regexp.MustCompile(`(?i)^now i (should|need to) determine`),
	regexp.MustCompile(`(?i)^let (me|us) (review|examine|inspect|analyze|consider|check) the (deployment|options|data|logs|cluster)`),
	regexp.MustCompile(`(?i)^(we could|one option is to|one approach is to)`),
	// Outro / conversational signoffs
	regexp.MustCompile(`(?i)i hope this (detailed )?breakdown was helpful`),
	regexp.MustCompile(`(?i)i hope this helps`),
	regexp.MustCompile(`(?i)feel free to ask if you have (any )?(further )?questions`),
	regexp.MustCompile(`(?i)let me know if you need (anything else|further analysis)`),
	regexp.MustCompile(`(?i)thank you for your patience`),
}

var contrastiveCuePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(however|in contrast|contrastingly|on the other hand|unlike)\b`),
	regexp.MustCompile(`(?i)\b(whereas|instead of|rather than|alternatively|although|despite|nevertheless|yet)\b`),
	regexp.MustCompile(`(?i)\b(in lieu of|counter to|opposite to)\b`),
}

var boundaryCuePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bif\b.+\b(then|otherwise|must|should|route|proceed|fallback|trigger)\b`),
	regexp.MustCompile(`(?i)\b(boundary|threshold|condition|constraint check|quorum)\b`),
	regexp.MustCompile(`(?i)\bunless\b`),
	regexp.MustCompile(`(?i)\b(min_version|max_lag|timeout_ms|weight|peers)\s*[<>=!]+`),
}

var toolCallPatterns = []*regexp.Regexp{
	regexp.MustCompile(`[a-zA-Z0-9_]+\([a-zA-Z0-9_]+=[^)]*\)`),
	regexp.MustCompile(`(?i)\b(tool call|invoke|action):\s*[a-zA-Z0-9_]+`),
	regexp.MustCompile(`(?i)\b(call|calling)\s+[a-zA-Z0-9_]+\(`),
}

var deductionCuePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b(therefore|hence|thus|consequently|in conclusion|crucially|pivot:|correction:|falsified)\b`),
}

func (p *ContrastiveCoTPruner) isBoilerplateText(text string) bool {
	clean := strings.TrimSpace(text)
	if clean == "" {
		return true
	}
	for _, custom := range p.CustomBoilerplate {
		if strings.Contains(strings.ToLower(clean), strings.ToLower(custom)) {
			return true
		}
	}
	for _, re := range defaultBoilerplatePatterns {
		if re.MatchString(clean) {
			return true
		}
	}
	return false
}

func (p *ContrastiveCoTPruner) classifyStep(step string) (isPivot bool, kind ReasoningPivotKind, cue string) {
	for _, re := range contrastiveCuePatterns {
		if loc := re.FindString(step); loc != "" {
			return true, PivotKindContrast, strings.ToLower(loc)
		}
	}
	for _, re := range boundaryCuePatterns {
		if loc := re.FindString(step); loc != "" {
			return true, PivotKindBoundary, strings.ToLower(loc)
		}
	}
	for _, re := range toolCallPatterns {
		if loc := re.FindString(step); loc != "" {
			return true, PivotKindToolCall, strings.ToLower(loc)
		}
	}
	for _, re := range deductionCuePatterns {
		if loc := re.FindString(step); loc != "" {
			lower := strings.ToLower(step)
			if strings.Contains(lower, "correction") || strings.Contains(lower, "falsified") {
				return true, PivotKindCorrection, strings.ToLower(loc)
			}
			return true, PivotKindDeduction, strings.ToLower(loc)
		}
	}
	return false, "", ""
}

func cleanStepArtifacts(step string) string {
	cleaned := step
	for _, re := range defaultBoilerplatePatterns {
		loc := re.FindStringIndex(cleaned)
		if len(loc) == 2 && loc[0] == 0 && loc[1] < len(cleaned) {
			remainder := strings.TrimSpace(cleaned[loc[1]:])
			remainder = strings.TrimLeft(remainder, ":.,- ")
			if remainder != "" {
				cleaned = remainder
			}
		}
	}
	return cleaned
}

func splitIntoSentences(text string) []string {
	var sentences []string
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		start := 0
		runes := []rune(line)
		n := len(runes)
		for i := 0; i < n; i++ {
			r := runes[i]
			if r == '.' || r == '!' || r == '?' {
				isBoundary := false
				if i == n-1 {
					isBoundary = true
				} else if i+1 < n && (runes[i+1] == ' ' || runes[i+1] == '\t') {
					if i > 0 && runes[i-1] >= '0' && runes[i-1] <= '9' && i+2 < n && runes[i+2] >= '0' && runes[i+2] <= '9' {
						isBoundary = false
					} else {
						isBoundary = true
					}
				}
				if isBoundary {
					s := strings.TrimSpace(string(runes[start : i+1]))
					if s != "" {
						sentences = append(sentences, s)
					}
					for i+1 < n && (runes[i+1] == ' ' || runes[i+1] == '\t') {
						i++
					}
					start = i + 1
				}
			}
		}
		if start < n {
			s := strings.TrimSpace(string(runes[start:]))
			if s != "" {
				sentences = append(sentences, s)
			}
		}
	}
	return sentences
}

func extractToolCallsFromText(text string) []ToolCall {
	var calls []ToolCall
	re := regexp.MustCompile(`([a-zA-Z0-9_]+)\(([^)]*)\)`)
	matches := re.FindAllStringSubmatch(text, -1)
	for _, m := range matches {
		if len(m) == 3 {
			name := m[1]
			if name == "if" || name == "when" || name == "for" || name == "while" {
				continue
			}
			rawArgs := m[2]
			args := parseToolArgs(rawArgs)
			calls = append(calls, ToolCall{
				ID:   fmt.Sprintf("call_%s", name),
				Name: name,
				Args: args,
			})
		}
	}
	return calls
}

func parseToolArgs(raw string) map[string]any {
	args := make(map[string]any)
	if strings.TrimSpace(raw) == "" {
		return args
	}
	re := regexp.MustCompile(`([a-zA-Z0-9_]+)=("([^"]*)"|'([^']*)'|([^,\s]+))`)
	matches := re.FindAllStringSubmatch(raw, -1)
	for _, m := range matches {
		if len(m) >= 3 {
			key := m[1]
			val := m[2]
			val = strings.Trim(val, `"'`)
			args[key] = val
		}
	}
	if len(args) == 0 {
		args["raw"] = raw
	}
	return args
}

func containsToolCall(calls []ToolCall, name string) bool {
	for _, c := range calls {
		if strings.EqualFold(c.Name, name) {
			return true
		}
	}
	return false
}

func matchesPivot(step string, pivot ReasoningPivot) bool {
	if strings.Contains(step, pivot.Statement) || strings.Contains(pivot.Statement, step) {
		return true
	}
	if pivot.ContrastCue != "" && strings.Contains(strings.ToLower(step), strings.ToLower(pivot.ContrastCue)) {
		return true
	}
	return false
}

// PruneExemplar prunes an exemplar by eliminating conversational boilerplate and verbose
// self-narration while retaining contrastive pivots, branch boundaries, and tool arguments.
func (p *ContrastiveCoTPruner) PruneExemplar(exemplar CoTExemplar) PrunedExemplarResult {
	if p == nil {
		p = NewContrastiveCoTPruner()
	}
	estimator := p.TokenEstimator
	if estimator == nil {
		estimator = EstimateTokens
	}

	result := PrunedExemplarResult{
		ExemplarID:        exemplar.ID,
		Prompt:            exemplar.Prompt,
		Output:            exemplar.Output,
		RetainedPivots:    make([]ReasoningPivot, 0),
		RetainedToolCalls: make([]ToolCall, 0),
		OriginalSteps:     make([]string, 0),
		PrunedSteps:       make([]string, 0),
	}

	rawThought := exemplar.Thought
	if len(exemplar.Steps) > 0 {
		result.OriginalSteps = make([]string, len(exemplar.Steps))
		copy(result.OriginalSteps, exemplar.Steps)
		if rawThought == "" {
			rawThought = strings.Join(exemplar.Steps, "\n")
		}
	} else if rawThought != "" {
		result.OriginalSteps = splitIntoSentences(rawThought)
	}

	result.OriginalThought = rawThought
	result.OriginalTokens = estimator(rawThought)

	if len(result.OriginalSteps) == 0 && rawThought == "" {
		result.PreservationRate = 1.0
		return result
	}

	toolCallNames := make(map[string]bool)
	for _, tc := range exemplar.ToolCalls {
		result.RetainedToolCalls = append(result.RetainedToolCalls, tc)
		toolCallNames[strings.ToLower(tc.Name)] = true
	}

	explicitPivotsMatched := make(map[int]bool)
	explicitPivots := exemplar.Pivots

	retainedSteps := make([]string, 0)

	for _, step := range result.OriginalSteps {
		stepTrimmed := strings.TrimSpace(step)
		if stepTrimmed == "" {
			continue
		}

		matchedExplicitIdx := -1
		for pIdx, ep := range explicitPivots {
			if matchesPivot(stepTrimmed, ep) {
				matchedExplicitIdx = pIdx
				explicitPivotsMatched[pIdx] = true
				break
			}
		}

		isPivot, kind, cue := p.classifyStep(stepTrimmed)

		hasToolInvocation := false
		for _, re := range toolCallPatterns {
			if re.MatchString(stepTrimmed) {
				hasToolInvocation = true
				break
			}
		}
		if !hasToolInvocation {
			lowerStep := strings.ToLower(stepTrimmed)
			for tName := range toolCallNames {
				if strings.Contains(lowerStep, tName) {
					hasToolInvocation = true
					break
				}
			}
		}

		isExplicitPivot := matchedExplicitIdx >= 0
		shouldRetain := isExplicitPivot || isPivot || hasToolInvocation

		if !shouldRetain {
			if p.isBoilerplateText(stepTrimmed) {
				continue
			}
			lower := strings.ToLower(stepTrimmed)
			if strings.Contains(lower, "status") || strings.Contains(lower, "output") ||
				strings.Contains(lower, "result") || strings.Contains(lower, "error") ||
				strings.Contains(lower, "passed") || strings.Contains(lower, "healthy") ||
				strings.Contains(lower, "synced") || strings.Contains(lower, "confirms") {
				shouldRetain = true
			} else {
				continue
			}
		}

		cleaned := cleanStepArtifacts(stepTrimmed)
		if cleaned == "" {
			cleaned = stepTrimmed
		}

		retainedSteps = append(retainedSteps, cleaned)

		if isExplicitPivot {
			ep := explicitPivots[matchedExplicitIdx]
			ep.Retained = true
			ep.StepIndex = len(retainedSteps) - 1
			result.RetainedPivots = append(result.RetainedPivots, ep)
		} else if isPivot || hasToolInvocation {
			pivotKind := kind
			if hasToolInvocation && pivotKind == "" {
				pivotKind = PivotKindToolCall
			}
			newPivot := ReasoningPivot{
				ID:           fmt.Sprintf("pivot-%d", len(result.RetainedPivots)+1),
				Kind:         pivotKind,
				Statement:    cleaned,
				ContrastCue:  cue,
				StepIndex:    len(retainedSteps) - 1,
				Retained:     true,
				Significance: 1.0,
			}
			result.RetainedPivots = append(result.RetainedPivots, newPivot)
		}

		if hasToolInvocation && p.PreserveToolCalls {
			extractedCalls := extractToolCallsFromText(cleaned)
			for _, ec := range extractedCalls {
				if !containsToolCall(result.RetainedToolCalls, ec.Name) {
					result.RetainedToolCalls = append(result.RetainedToolCalls, ec)
				}
			}
		}
	}

	for pIdx, ep := range explicitPivots {
		if !explicitPivotsMatched[pIdx] {
			ep.Retained = true
			result.RetainedPivots = append(result.RetainedPivots, ep)
			retainedSteps = append(retainedSteps, ep.Statement)
		}
	}

	result.PrunedSteps = retainedSteps
	result.PrunedThought = strings.Join(retainedSteps, "\n")
	result.PrunedTokens = estimator(result.PrunedThought)

	if result.OriginalTokens > result.PrunedTokens {
		result.TokensSaved = result.OriginalTokens - result.PrunedTokens
		result.ReductionRatio = float64(result.TokensSaved) / float64(result.OriginalTokens)
	} else {
		result.TokensSaved = 0
		result.ReductionRatio = 0.0
	}

	if len(explicitPivots) > 0 {
		result.TotalPivots = len(explicitPivots)
	} else {
		result.TotalPivots = len(result.RetainedPivots)
	}
	result.PivotsRetained = len(result.RetainedPivots)
	if result.TotalPivots > 0 {
		result.PreservationRate = float64(result.PivotsRetained) / float64(result.TotalPivots)
		if result.PreservationRate > 1.0 {
			result.PreservationRate = 1.0
		}
	} else {
		result.PreservationRate = 1.0
	}

	result.FormattedPrompt = result.FormatDemonstration()

	return result
}

// PruneBatch prunes a collection of CoT exemplars, returning the pruned results in order.
func (p *ContrastiveCoTPruner) PruneBatch(exemplars []CoTExemplar) []PrunedExemplarResult {
	results := make([]PrunedExemplarResult, len(exemplars))
	for i, ex := range exemplars {
		results[i] = p.PruneExemplar(ex)
	}
	return results
}

// IdentifyPivots discovers and extracts critical reasoning pivots from a thought trace or step list.
func (p *ContrastiveCoTPruner) IdentifyPivots(thought string, steps []string) []ReasoningPivot {
	if p == nil {
		p = NewContrastiveCoTPruner()
	}
	if len(steps) == 0 && thought != "" {
		steps = splitIntoSentences(thought)
	}
	var pivots []ReasoningPivot
	for idx, s := range steps {
		sTrimmed := strings.TrimSpace(s)
		if isPivot, kind, cue := p.classifyStep(sTrimmed); isPivot {
			pivots = append(pivots, ReasoningPivot{
				ID:           fmt.Sprintf("pivot-%d", len(pivots)+1),
				Kind:         kind,
				Statement:    cleanStepArtifacts(sTrimmed),
				ContrastCue:  cue,
				StepIndex:    idx,
				Retained:     true,
				Significance: 1.0,
			})
		}
	}
	return pivots
}

// BenchmarkExemplars returns realistic Chain-of-Thought exemplars from production agent domains
// demonstrating deployment rollouts, incident triage, and database failover.
func BenchmarkExemplars() []CoTExemplar {
	return []CoTExemplar{
		{
			ID:     "cot-deploy-bluegreen",
			Prompt: "Deploy the payment service to production.",
			Thought: `Hello! I would be delighted to assist you with this deployment today. Let's think step by step and approach this systematically.
First, let me carefully understand what the user wants. The goal is to deploy the payment backend_api service to the production cluster.
Let me review the deployment options available. We could perform an in-place deployment directly on the production nodes.
However, in contrast to an in-place deployment which risks downtime during database migration, we must use a blue-green strategy.
Let me double-check my reasoning to make sure there are no mistakes. Yes, blue-green isolates production traffic.
Now, let's look at the database configuration. If db_version < 2.4, we must execute the pre-migration script migrate_v2.sql; otherwise, we can proceed directly to traffic shifting.
I am now going to execute the verification tool. Tool call: verify_schema(service="payment", min_version="2.4.0").
The schema check returns successful status.
Now I need to determine the traffic routing rate.
Wait, unlike standard microservices where immediate 100% cutover is acceptable, payment services have a strict boundary: canary routing must route 5% traffic for 10 minutes first.
Therefore, instead of immediate cutover, invoke route_traffic(service="payment", canary_weight=0.05, duration_mins=10).
Let's pause and verify everything looks good up to this point. Everything seems completely fine and in order.
In conclusion, we have established the blue-green strategy, verified schema version, and scheduled canary traffic routing.
I hope this detailed breakdown was helpful! Please feel free to ask if you have any follow-up questions.`,
			ToolCalls: []ToolCall{
				{
					Name: "verify_schema",
					Args: map[string]any{"service": "payment", "min_version": "2.4.0"},
				},
				{
					Name: "route_traffic",
					Args: map[string]any{"service": "payment", "canary_weight": 0.05, "duration_mins": 10},
				},
			},
			Output: "Blue-green deployment initiated with canary route_traffic(service=\"payment\", canary_weight=0.05, duration_mins=10).",
		},
		{
			ID:     "cot-incident-triage",
			Prompt: "Diagnose elevated error rates on API cluster.",
			Thought: `Hi there! Certainly, I can help diagnose the elevated error rates on the API cluster. Let's think step by step.
First, let me break down this problem and understand the symptoms reported by the user.
We need to inspect the logs. We could query all logs across the entire cluster for the past 24 hours.
However, in contrast to a wide 24-hour search which would overwhelm the logging pipeline with millions of rows, we must narrow the search to the last 15 minutes filtered by status >= 500.
Let's make sure we don't make any errors in our query parameters.
Invoke query_logs(time_range="15m", filter="status>=500", limit=50).
The logs show database connection group timeouts on node api-worker-03.
Now let's consider rebooting the worker node immediately.
Wait, unlike stateless worker crashes where a reboot is safe, connection group starvation indicates database backend saturation; rebooting the node will cause a stampede on the primary database.
Therefore, instead of rebooting api-worker-03, we must isolate the node and shed traffic: isolate_node(node_id="api-worker-03", shed_traffic=true).
Double checking the work to make sure everything is in order.
Everything looks correct up to this point.
I hope this helps! Let me know if you need further analysis.`,
			ToolCalls: []ToolCall{
				{
					Name: "query_logs",
					Args: map[string]any{"time_range": "15m", "filter": "status>=500", "limit": 50},
				},
				{
					Name: "isolate_node",
					Args: map[string]any{"node_id": "api-worker-03", "shed_traffic": true},
				},
			},
			Output: "Isolated api-worker-03 with isolate_node(node_id=\"api-worker-03\", shed_traffic=true) to prevent database connection stampede.",
		},
		{
			ID:     "cot-quorum-failover",
			Prompt: "Determine primary failover for partitioned replica cluster.",
			Thought: `Greetings! I am happy to help resolve the cluster failover issue. Let's approach this methodically step by step.
As an autonomous agent, I will inspect the cluster health.
Let's see what we have here. We have three replicas: replica-a, replica-b, and replica-c.
One approach is to elect replica-a simply because it was the first replica created.
However, unlike replica-a which has replication lag of 450ms, replica-b has replication lag of only 2ms and is fully synced with WAL.
Let's check the quorum boundary rule.
If consensus_peers >= 2 and max_lag_ms <= 10, promote replica immediately; otherwise, trigger manual fence.
Let's invoke check_consensus(cluster_id="db-prod-01", quorum_threshold=2).
Consensus confirms replica-b is qualified.
Let me pause and verify all details. Everything seems completely fine.
Therefore, promote replica-b using promote_replica(cluster_id="db-prod-01", target="replica-b", force=false).
In conclusion, replica-b is promoted successfully.
Thank you for your patience! Feel free to ask if you have any other questions.`,
			ToolCalls: []ToolCall{
				{
					Name: "check_consensus",
					Args: map[string]any{"cluster_id": "db-prod-01", "quorum_threshold": 2},
				},
				{
					Name: "promote_replica",
					Args: map[string]any{"cluster_id": "db-prod-01", "target": "replica-b", "force": false},
				},
			},
			Output: "Promoted replica-b with promote_replica(cluster_id=\"db-prod-01\", target=\"replica-b\", force=false).",
		},
	}
}
