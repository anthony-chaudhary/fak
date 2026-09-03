package agentopt

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// Family 3: Self-improvement & feedback loops.
//
// Reflexion trace summarizer: verbal failure reflection into bounded memory.
// Synthesizes structured, compact verbal self-reflections (< 200 tokens)
// from failed agent execution attempts (tool errors, assertion failures, exception traces)
// and injects them as hard operational memory constraints into subsequent attempts.

const (
	// MaxReflexionTokens defines the strict token ceiling (< 200 tokens)
	// for verbal self-reflections injected into operational memory.
	MaxReflexionTokens = 200

	// DefaultReflexionTargetTokens defines the default token budget for compact reflections.
	DefaultReflexionTargetTokens = 150
)

// FailedAttemptInfo encapsulates diagnostic signals and trace data from a failed attempt.
type FailedAttemptInfo struct {
	AttemptIndex     int               `json:"attempt_index"`
	Goal             string            `json:"goal,omitempty"`
	Action           string            `json:"action,omitempty"`
	ToolName         string            `json:"tool_name,omitempty"`
	ToolArgs         string            `json:"tool_args,omitempty"`
	ToolError        string            `json:"tool_error,omitempty"`
	AssertionFailure string            `json:"assertion_failure,omitempty"`
	ExceptionTrace   string            `json:"exception_trace,omitempty"`
	Output           string            `json:"output,omitempty"`
	ExecutionLog     []string          `json:"execution_log,omitempty"`
	Details          map[string]string `json:"details,omitempty"`
}

// ReflexionRecord represents a structured, compact verbal self-reflection.
// Total tokens across the record and its constraint representation remain strictly < 200 tokens.
type ReflexionRecord struct {
	AttemptIndex       int    `json:"attempt_index"`
	FailedAction       string `json:"failed_action"`
	RootCause          string `json:"root_cause"`
	ConcreteMitigation string `json:"concrete_mitigation"`
	ConstraintRule     string `json:"constraint_rule"`
	EstimatedTokens    int    `json:"estimated_tokens"`
}

// String returns a compact representation of the reflection record.
func (r ReflexionRecord) String() string {
	return fmt.Sprintf("Attempt %d failed on '%s': %s (Rule: %s)", r.AttemptIndex, r.FailedAction, r.RootCause, r.ConstraintRule)
}

// Tokens returns the estimated token count of the record constraint.
func (r ReflexionRecord) Tokens() int {
	if r.EstimatedTokens > 0 {
		return r.EstimatedTokens
	}
	return EstimateTokens(r.String())
}

// FailureReflection couples a ReflexionRecord with its formatted operational memory constraint.
type FailureReflection struct {
	Record          ReflexionRecord `json:"record"`
	Constraint      string          `json:"constraint"`
	EstimatedTokens int             `json:"estimated_tokens"`
}

// OperationalMemory represents a bounded memory store capable of storing constraints.
type OperationalMemory interface {
	Set(key, val string) error
}

// ReflexionSummarizer synthesizes compact verbal reflections from failed attempts
// and formats/injects them as operational memory constraints.
type ReflexionSummarizer struct {
	maxTokens int
}

// ReflexionOption configures a ReflexionSummarizer.
type ReflexionOption func(*ReflexionSummarizer)

// WithMaxTokens sets a custom token ceiling for the summarizer (bounded to at most 200).
func WithMaxTokens(tokens int) ReflexionOption {
	return func(s *ReflexionSummarizer) {
		if tokens > 0 && tokens <= MaxReflexionTokens {
			s.maxTokens = tokens
		}
	}
}

// NewReflexionSummarizer creates a new ReflexionSummarizer with default bounds.
func NewReflexionSummarizer(opts ...ReflexionOption) *ReflexionSummarizer {
	s := &ReflexionSummarizer{
		maxTokens: DefaultReflexionTargetTokens,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// SummarizeFailure analyzes a failed attempt and extracts a compact verbal reflection record (< 200 tokens).
func (s *ReflexionSummarizer) SummarizeFailure(failedAttempt FailedAttemptInfo) ReflexionRecord {
	action := s.extractFailedAction(failedAttempt)
	rootCause, mitigation, rule := s.extractFailureDiagnostics(failedAttempt, action)

	attemptIdx := failedAttempt.AttemptIndex
	if attemptIdx <= 0 {
		attemptIdx = 1
	}

	record := ReflexionRecord{
		AttemptIndex:       attemptIdx,
		FailedAction:       action,
		RootCause:          rootCause,
		ConcreteMitigation: mitigation,
		ConstraintRule:     rule,
	}

	s.compactRecord(&record)
	return record
}

// FormatAsMemoryConstraint formats a ReflexionRecord into a structured string suitable
// for injection into operational memory as a hard constraint.
func (s *ReflexionSummarizer) FormatAsMemoryConstraint(record ReflexionRecord) string {
	return fmt.Sprintf(
		"[REFLEXION_CONSTRAINT_ATTEMPT_%d] Action: %s | Cause: %s | Mitigation: %s | Rule: %s",
		record.AttemptIndex,
		record.FailedAction,
		record.RootCause,
		record.ConcreteMitigation,
		record.ConstraintRule,
	)
}

// Reflect processes a failed attempt into a complete FailureReflection with formatted constraint.
func (s *ReflexionSummarizer) Reflect(failedAttempt FailedAttemptInfo) FailureReflection {
	rec := s.SummarizeFailure(failedAttempt)
	cstr := s.FormatAsMemoryConstraint(rec)
	toks := EstimateTokens(cstr)
	return FailureReflection{
		Record:          rec,
		Constraint:      cstr,
		EstimatedTokens: toks,
	}
}

// InjectMemoryConstraint writes the formatted constraint into an operational memory store.
func (s *ReflexionSummarizer) InjectMemoryConstraint(mem OperationalMemory, record ReflexionRecord) error {
	if mem == nil {
		return errors.New("operational memory cannot be nil")
	}
	key := fmt.Sprintf("reflexion_constraint_attempt_%d", record.AttemptIndex)
	val := s.FormatAsMemoryConstraint(record)
	return mem.Set(key, val)
}

// InjectWorkingMemory injects the reflection constraint directly into WorkingMemory.
func (s *ReflexionSummarizer) InjectWorkingMemory(wm *WorkingMemory, record ReflexionRecord) error {
	return s.InjectMemoryConstraint(wm, record)
}

// InjectReflection writes a FailureReflection directly into operational memory.
func (s *ReflexionSummarizer) InjectReflection(mem OperationalMemory, reflection FailureReflection) error {
	if mem == nil {
		return errors.New("operational memory cannot be nil")
	}
	key := fmt.Sprintf("reflexion_constraint_attempt_%d", reflection.Record.AttemptIndex)
	return mem.Set(key, reflection.Constraint)
}

// FormatOperationalMemoryBlock formats a series of reflection records into a unified
// constraint block for subsequent attempt prompts or execution context.
func (s *ReflexionSummarizer) FormatOperationalMemoryBlock(records []ReflexionRecord) string {
	if len(records) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("OPERATIONAL CONSTRAINTS FROM PRIOR FAILED ATTEMPTS:\n")
	for _, rec := range records {
		sb.WriteString(fmt.Sprintf("- Attempt %d: %s\n", rec.AttemptIndex, rec.ConstraintRule))
	}
	return strings.TrimSpace(sb.String())
}

// FromExecutionSteps constructs a FailedAttemptInfo from execution steps and an error output.
func FromExecutionSteps(attemptIndex int, goal string, steps []ExecutionStep, failureError string) FailedAttemptInfo {
	info := FailedAttemptInfo{
		AttemptIndex: attemptIndex,
		Goal:         goal,
		Output:       failureError,
	}
	if len(steps) > 0 {
		last := steps[len(steps)-1]
		info.ToolName = last.ToolName
		if last.Receipt != nil && last.Receipt.Error != "" {
			info.ToolError = last.Receipt.Error
		}
		if last.ToolOutput != "" && info.ToolError == "" {
			info.Output = last.ToolOutput
		}
	}
	return info
}

func (s *ReflexionSummarizer) extractFailedAction(attempt FailedAttemptInfo) string {
	if strings.TrimSpace(attempt.Action) != "" {
		return cleanSingleLine(attempt.Action, 60)
	}
	if strings.TrimSpace(attempt.ToolName) != "" {
		if strings.TrimSpace(attempt.ToolArgs) != "" {
			return cleanSingleLine(fmt.Sprintf("%s(%s)", attempt.ToolName, attempt.ToolArgs), 60)
		}
		return cleanSingleLine(attempt.ToolName, 40)
	}
	if len(attempt.ExecutionLog) > 0 {
		for i := len(attempt.ExecutionLog) - 1; i >= 0; i-- {
			line := strings.TrimSpace(attempt.ExecutionLog[i])
			if strings.HasPrefix(line, "Action:") || strings.HasPrefix(line, "Step:") || strings.HasPrefix(line, "Tool:") {
				return cleanSingleLine(line, 60)
			}
		}
		return cleanSingleLine(attempt.ExecutionLog[len(attempt.ExecutionLog)-1], 60)
	}
	if strings.TrimSpace(attempt.Goal) != "" {
		return cleanSingleLine("execute goal: "+attempt.Goal, 50)
	}
	return "unspecified action"
}

func (s *ReflexionSummarizer) extractFailureDiagnostics(attempt FailedAttemptInfo, action string) (rootCause, mitigation, rule string) {
	// 1. Explicit exception / panic trace
	if strings.TrimSpace(attempt.ExceptionTrace) != "" {
		cleanEx := extractKeyExceptionLine(attempt.ExceptionTrace)
		rootCause = fmt.Sprintf("Exception encountered: %s", cleanEx)
		mitigation, rule = synthesizeMitigationAndRule(cleanEx, action)
		return
	}

	// 2. Explicit assertion failure
	if strings.TrimSpace(attempt.AssertionFailure) != "" {
		cleanAssert := cleanSingleLine(attempt.AssertionFailure, 100)
		rootCause = fmt.Sprintf("Assertion violated: %s", cleanAssert)
		mitigation, rule = synthesizeMitigationAndRule(cleanAssert, action)
		return
	}

	// 3. Explicit tool error
	if strings.TrimSpace(attempt.ToolError) != "" {
		cleanErr := cleanSingleLine(attempt.ToolError, 100)
		rootCause = fmt.Sprintf("Tool failure: %s", cleanErr)
		mitigation, rule = synthesizeMitigationAndRule(cleanErr, action)
		return
	}

	// 4. Inspect Output or ExecutionLog for failure signals
	combined := attempt.Output
	if combined == "" && len(attempt.ExecutionLog) > 0 {
		combined = strings.Join(attempt.ExecutionLog, "\n")
	}

	if strings.TrimSpace(combined) != "" {
		if ex := findExceptionInText(combined); ex != "" {
			rootCause = fmt.Sprintf("Exception encountered: %s", ex)
			mitigation, rule = synthesizeMitigationAndRule(ex, action)
			return
		}
		if assertErr := findAssertionInText(combined); assertErr != "" {
			rootCause = fmt.Sprintf("Assertion violated: %s", assertErr)
			mitigation, rule = synthesizeMitigationAndRule(assertErr, action)
			return
		}
		if toolErr := findErrorInText(combined); toolErr != "" {
			rootCause = fmt.Sprintf("Execution error: %s", toolErr)
			mitigation, rule = synthesizeMitigationAndRule(toolErr, action)
			return
		}
	}

	// 5. Fallback generic failure reflection
	rootCause = "Execution did not achieve expected success conditions"
	mitigation = "Review preconditions and verify inputs before repeating the action."
	rule = fmt.Sprintf("DO NOT repeat '%s' without verifying preconditions.", action)
	return
}

func synthesizeMitigationAndRule(diagnostic, action string) (mitigation, rule string) {
	lower := strings.ToLower(diagnostic)

	switch {
	case strings.Contains(lower, "permission denied") || strings.Contains(lower, "access denied") || strings.Contains(lower, "eacces") || strings.Contains(lower, "unauthorized"):
		mitigation = "Verify target file permissions and operational privileges before retrying."
		rule = "DO NOT access paths lacking required permissions; MUST verify access rights beforehand."

	case strings.Contains(lower, "no such file") || strings.Contains(lower, "not found") || strings.Contains(lower, "enoent") || strings.Contains(lower, "command not found") || strings.Contains(lower, "404"):
		mitigation = "Confirm target file path or executable exists prior to invocation."
		rule = "DO NOT execute actions against unverified paths; MUST confirm target presence first."

	case strings.Contains(lower, "timeout") || strings.Contains(lower, "timed out") || strings.Contains(lower, "deadline exceeded"):
		mitigation = "Reduce workload batch size or configure an extended deadline."
		rule = "DO NOT dispatch unbounded requests without timeout bounds or payload batching."

	case strings.Contains(lower, "syntax") || strings.Contains(lower, "parse") || strings.Contains(lower, "invalid json") || strings.Contains(lower, "unexpected token") || strings.Contains(lower, "unmarshal"):
		mitigation = "Validate syntax formatting and argument schema prior to invocation."
		rule = "DO NOT submit malformed syntax payloads; MUST validate arguments against expected schema."

	case strings.Contains(lower, "index out of range") || strings.Contains(lower, "out of bounds") || strings.Contains(lower, "slice bounds"):
		mitigation = "Add collection bounds check prior to indexing elements."
		rule = "DO NOT index collections without length checks; MUST enforce bounds verification."

	case strings.Contains(lower, "nil pointer") || strings.Contains(lower, "null pointer") || strings.Contains(lower, "nonetype") || strings.Contains(lower, "nullpointer"):
		mitigation = "Ensure target reference is initialized and non-nil before accessing properties."
		rule = "DO NOT dereference unverified references; MUST check for non-nil status."

	case strings.Contains(lower, "assertion") || strings.Contains(lower, "expected") || strings.Contains(lower, "mismatch"):
		mitigation = "Align output structure and properties with target verification criteria."
		rule = "DO NOT produce outputs deviating from asserted contract; MUST satisfy expected output criteria."

	case strings.Contains(lower, "rate limit") || strings.Contains(lower, "429") || strings.Contains(lower, "too many requests"):
		mitigation = "Throttle invocation frequency and apply exponential backoff delay."
		rule = "DO NOT execute rapid request bursts against rate-limited endpoints; MUST throttle calls."

	default:
		mitigation = "Inspect input arguments and handle edge conditions before repeating action."
		rule = fmt.Sprintf("DO NOT re-attempt '%s' with identical inputs; MUST adjust execution parameters.", action)
	}
	return
}

func extractKeyExceptionLine(trace string) string {
	lines := strings.Split(trace, "\n")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if strings.HasPrefix(l, "panic:") {
			return cleanSingleLine(l, 100)
		}
	}
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" {
			continue
		}
		if strings.Contains(l, "Error:") || strings.Contains(l, "Exception:") {
			return cleanSingleLine(l, 100)
		}
	}
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" {
			return cleanSingleLine(l, 100)
		}
	}
	return "unknown runtime exception"
}

var (
	panicRegex     = regexp.MustCompile(`(?i)(panic:\s*[^\n]+)`)
	exceptionRegex = regexp.MustCompile(`(?i)([a-z0-9_.]*(?:exception|error):\s*[^\n]+)`)
	assertionRegex = regexp.MustCompile(`(?i)(assert(?:ion)?(?:\s*failed|\s*error)?:\s*[^\n]+)`)
)

func findExceptionInText(text string) string {
	if m := panicRegex.FindString(text); m != "" {
		return cleanSingleLine(m, 100)
	}
	if m := exceptionRegex.FindString(text); m != "" {
		return cleanSingleLine(m, 100)
	}
	return ""
}

func findAssertionInText(text string) string {
	if m := assertionRegex.FindString(text); m != "" {
		return cleanSingleLine(m, 100)
	}
	return ""
}

func findErrorInText(text string) string {
	lines := strings.Split(text, "\n")
	for _, l := range lines {
		l = strings.TrimSpace(l)
		lower := strings.ToLower(l)
		if strings.HasPrefix(lower, "error:") || strings.HasPrefix(lower, "failed:") || strings.Contains(lower, "exit status") {
			return cleanSingleLine(l, 100)
		}
	}
	return ""
}

func cleanSingleLine(text string, maxLen int) string {
	text = strings.ReplaceAll(text, "\r", " ")
	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.Join(strings.Fields(text), " ")
	if len(text) <= maxLen {
		return text
	}
	if maxLen <= 3 {
		return text[:maxLen]
	}
	return text[:maxLen-3] + "..."
}

func (s *ReflexionSummarizer) compactRecord(rec *ReflexionRecord) {
	budget := s.maxTokens
	if budget <= 0 || budget > MaxReflexionTokens {
		budget = MaxReflexionTokens
	}

	formatted := s.FormatAsMemoryConstraint(*rec)
	toks := EstimateTokens(formatted)
	if toks < budget {
		rec.EstimatedTokens = toks
		return
	}

	rec.RootCause = cleanSingleLine(rec.RootCause, 80)
	rec.ConcreteMitigation = cleanSingleLine(rec.ConcreteMitigation, 80)
	rec.ConstraintRule = cleanSingleLine(rec.ConstraintRule, 80)
	rec.FailedAction = cleanSingleLine(rec.FailedAction, 40)

	formatted = s.FormatAsMemoryConstraint(*rec)
	toks = EstimateTokens(formatted)
	if toks >= budget {
		rec.RootCause = cleanSingleLine(rec.RootCause, 50)
		rec.ConcreteMitigation = cleanSingleLine(rec.ConcreteMitigation, 50)
		rec.ConstraintRule = cleanSingleLine(rec.ConstraintRule, 50)
		rec.FailedAction = cleanSingleLine(rec.FailedAction, 30)
		formatted = s.FormatAsMemoryConstraint(*rec)
		toks = EstimateTokens(formatted)
	}
	rec.EstimatedTokens = toks
}
