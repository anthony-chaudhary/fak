// Package ciyaml provides structural CI YAML linting, validation, and DAG dependency resolution.
//
// Invariant: workflow jobs form a directed acyclic graph with deterministic execution order.
// Invariant: line scanning strictly isolates block scalars and comments from structural tokens.
// Guard: fail-closed on duplicate keys.
package ciyaml

import (
	"fmt"
	"strings"
)

// ViolationSeverity represents the severity level of a lint violation.
type ViolationSeverity string

const (
	// SeverityError indicates a critical violation that prevents safe parsing or execution.
	SeverityError ViolationSeverity = "error"
	// SeverityWarning indicates a non-critical deviation or recommendation.
	SeverityWarning ViolationSeverity = "warning"
)

// Violation describes a structural defect, syntax issue, or constraint breach in CI YAML.
type Violation struct {
	Line     int               `json:"line"`
	Column   int               `json:"column"`
	Message  string            `json:"message"`
	Rule     string            `json:"rule"`
	Severity ViolationSeverity `json:"severity"`
}

// LintError represents an aggregated error containing one or more lint violations.
type LintError struct {
	Violations []Violation
}

// Error returns a formatted representation of the lint violations.
func (e *LintError) Error() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("ciyaml: %d lint violation(s) found:\n", len(e.Violations)))
	for _, v := range e.Violations {
		sb.WriteString(fmt.Sprintf("  line %d, col %d [%s] %s: %s\n", v.Line, v.Column, v.Severity, v.Rule, v.Message))
	}
	return sb.String()
}

// Workflow represents a top-level CI workflow definition.
type Workflow struct {
	Name string         `json:"name"`
	On   []string       `json:"on,omitempty"`
	Jobs map[string]Job `json:"jobs"`
}

// Job represents a single execution job in a workflow, including steps and dependencies.
type Job struct {
	ID        string   `json:"id"`
	Name      string   `json:"name,omitempty"`
	RunsOn    string   `json:"runs_on,omitempty"`
	Needs     []string `json:"needs,omitempty"`
	DependsOn []string `json:"depends_on,omitempty"`
	Steps     []Step   `json:"steps,omitempty"`
}

// Step represents an individual execution step within a job.
type Step struct {
	ID   string            `json:"id,omitempty"`
	Name string            `json:"name,omitempty"`
	Uses string            `json:"uses,omitempty"`
	Run  string            `json:"run,omitempty"`
	With map[string]string `json:"with,omitempty"`
	Env  map[string]string `json:"env,omitempty"`
}

// HasErrors returns true if any violation in the slice has SeverityError.
func HasErrors(violations []Violation) bool {
	for _, v := range violations {
		if v.Severity == SeverityError {
			return true
		}
	}
	return false
}

type scope struct {
	indent     int
	isListItem bool
	keys       map[string]int
	stepKeys   map[string]int
}

// Lint performs structural line scanning and validation on CI YAML content.
// It detects tabs in indentation or structure, unclosed single/double quotes,
// inconsistent indentation levels, and duplicate keys (jobs, steps, or mappings).
// Comments (#) and block scalars (| or >) are isolated and ignored during structural inspection.
//
// Invariant: line scanning strictly isolates block scalars and comments from structural tokens.
// Guard: fail-closed on duplicate keys.
func Lint(content string) ([]Violation, error) {
	lines := splitLines(content)
	var violations []Violation

	inBlockScalar := false
	blockScalarHeaderIndent := 0

	indentStack := []int{0}
	scopeStack := []*scope{
		{
			indent:   0,
			keys:     make(map[string]int),
			stepKeys: make(map[string]int),
		},
	}

	jobKeys := make(map[string]int)
	stepIDs := make(map[string]map[string]int)
	currentJob := ""
	currentSection := ""

	for lineIdx, rawLine := range lines {
		lineNum := lineIdx + 1

		// 1. Check if currently inside a block scalar
		if inBlockScalar {
			trimmedRaw := strings.TrimSpace(rawLine)
			if trimmedRaw == "" {
				// Blank line inside block scalar: continuation
				continue
			}
			leadingSpaces := countLeadingSpaces(rawLine)
			if leadingSpaces > blockScalarHeaderIndent {
				// Indented deeper than header: content of block scalar
				continue
			}
			// Dedented to <= header indent: block scalar ends
			inBlockScalar = false
		}

		// 2. Tab checks outside block scalar
		if tabIdx := strings.Index(rawLine, "\t"); tabIdx != -1 {
			leadingSpaces := countLeadingSpaces(rawLine)
			if tabIdx <= leadingSpaces {
				violations = append(violations, Violation{
					Line:     lineNum,
					Column:   tabIdx + 1,
					Message:  "tab character used for indentation; YAML requires spaces",
					Rule:     "tabs-forbidden",
					Severity: SeverityError,
				})
			} else {
				// Check if tab is outside quotes
				cleanBeforeTab, _, _ := scanQuotesAndStripComment(rawLine[:tabIdx])
				if !isInsideQuotes(cleanBeforeTab) {
					violations = append(violations, Violation{
						Line:     lineNum,
						Column:   tabIdx + 1,
						Message:  "tab character found in YAML structure",
						Rule:     "tabs-forbidden",
						Severity: SeverityError,
					})
				}
			}
		}

		// 3. Scan quotes and strip comments
		clean, unclosedQuote, quoteCol := scanQuotesAndStripComment(rawLine)
		if unclosedQuote != 0 {
			violations = append(violations, Violation{
				Line:     lineNum,
				Column:   quoteCol,
				Message:  fmt.Sprintf("unclosed %c quote on line %d", unclosedQuote, lineNum),
				Rule:     "unclosed-quote",
				Severity: SeverityError,
			})
		}

		trimmed := strings.TrimSpace(clean)
		if trimmed == "" {
			// Blank or comment-only line
			continue
		}

		indent := countLeadingSpaces(rawLine)

		// 4. Block scalar header detection
		if isBlockScalarHeader(trimmed) {
			inBlockScalar = true
			blockScalarHeaderIndent = indent
		}

		// 5. Indentation consistency check
		if indent%2 != 0 {
			violations = append(violations, Violation{
				Line:     lineNum,
				Column:   indent + 1,
				Message:  fmt.Sprintf("inconsistent indentation: %d spaces (expected multiple of 2)", indent),
				Rule:     "indentation-consistency",
				Severity: SeverityError,
			})
		}

		topIndent := indentStack[len(indentStack)-1]
		if indent > topIndent {
			indentStack = append(indentStack, indent)
		} else if indent < topIndent {
			for len(indentStack) > 1 && indentStack[len(indentStack)-1] > indent {
				indentStack = indentStack[:len(indentStack)-1]
			}
			if indentStack[len(indentStack)-1] != indent {
				violations = append(violations, Violation{
					Line:     lineNum,
					Column:   indent + 1,
					Message:  fmt.Sprintf("inconsistent indentation: dedent level %d does not match any outer block %v", indent, indentStack),
					Rule:     "indentation-consistency",
					Severity: SeverityError,
				})
			}
		}

		// 6. Section and job context tracking
		if indent == 0 {
			if strings.HasPrefix(trimmed, "jobs:") {
				currentSection = "jobs"
			} else if strings.HasPrefix(trimmed, "on:") {
				currentSection = "on"
			} else if strings.HasPrefix(trimmed, "name:") {
				currentSection = ""
			}
		}

		// 7. Duplicate key detection
		// Guard: fail-closed on duplicate keys
		isListItem := strings.HasPrefix(trimmed, "-")
		keyContent := trimmed
		if isListItem {
			keyContent = strings.TrimSpace(trimmed[1:])
		}

		colonIdx := findColonOutsideQuotes(keyContent)
		if colonIdx != -1 {
			key := strings.TrimSpace(keyContent[:colonIdx])
			key = strings.Trim(key, `"'`)

			if key != "" {
				// Manage scope stack according to indentation
				for len(scopeStack) > 1 && scopeStack[len(scopeStack)-1].indent > indent {
					scopeStack = scopeStack[:len(scopeStack)-1]
				}

				if isListItem {
					// Popping previous list item at the same indent
					if len(scopeStack) > 0 && scopeStack[len(scopeStack)-1].indent == indent && scopeStack[len(scopeStack)-1].isListItem {
						scopeStack = scopeStack[:len(scopeStack)-1]
					}
					newScope := &scope{
						indent:     indent,
						isListItem: true,
						keys:       make(map[string]int),
						stepKeys:   make(map[string]int),
					}
					scopeStack = append(scopeStack, newScope)
				} else if len(scopeStack) == 0 || scopeStack[len(scopeStack)-1].indent < indent {
					newScope := &scope{
						indent:     indent,
						isListItem: false,
						keys:       make(map[string]int),
						stepKeys:   make(map[string]int),
					}
					scopeStack = append(scopeStack, newScope)
				}

				currScope := scopeStack[len(scopeStack)-1]

				// Check duplicate within the immediate mapping scope
				if prevLine, exists := currScope.keys[key]; exists {
					violations = append(violations, Violation{
						Line:     lineNum,
						Column:   indent + 1,
						Message:  fmt.Sprintf("duplicate key %q (previously defined on line %d)", key, prevLine),
						Rule:     "duplicate-key",
						Severity: SeverityError,
					})
				} else {
					currScope.keys[key] = lineNum
				}

				// Check duplicate step keys inside a step list item
				for sIdx := len(scopeStack) - 1; sIdx >= 0; sIdx-- {
					sc := scopeStack[sIdx]
					if sc.isListItem {
						if prevLine, exists := sc.stepKeys[key]; exists && prevLine != lineNum {
							violations = append(violations, Violation{
								Line:     lineNum,
								Column:   indent + 1,
								Message:  fmt.Sprintf("duplicate step key %q (previously defined on line %d)", key, prevLine),
								Rule:     "duplicate-step-key",
								Severity: SeverityError,
							})
						} else {
							sc.stepKeys[key] = lineNum
						}
						break
					}
				}

				// Check duplicate job keys under jobs: section (at indent 2)
				if currentSection == "jobs" && indent == 2 && !isListItem {
					currentJob = key
					if prevLine, exists := jobKeys[key]; exists {
						violations = append(violations, Violation{
							Line:     lineNum,
							Column:   indent + 1,
							Message:  fmt.Sprintf("duplicate job key %q (previously defined on line %d)", key, prevLine),
							Rule:     "duplicate-job-key",
							Severity: SeverityError,
						})
					} else {
						jobKeys[key] = lineNum
					}
				}

				// Check duplicate step ID in current job
				if key == "id" && currentJob != "" {
					val := strings.TrimSpace(keyContent[colonIdx+1:])
					val = cleanQuotedString(val)
					if val != "" {
						if stepIDs[currentJob] == nil {
							stepIDs[currentJob] = make(map[string]int)
						}
						if prevLine, exists := stepIDs[currentJob][val]; exists {
							violations = append(violations, Violation{
								Line:     lineNum,
								Column:   indent + 1,
								Message:  fmt.Sprintf("duplicate step ID %q in job %q (previously defined on line %d)", val, currentJob, prevLine),
								Rule:     "duplicate-step-id",
								Severity: SeverityError,
							})
						} else {
							stepIDs[currentJob][val] = lineNum
						}
					}
				}
			}
		}
	}

	return violations, nil
}

// ParseWorkflow parses CI YAML content into a structured Workflow.
// It verifies lint rules first, failing closed on any error-severity violation.
//
// Invariant: workflow jobs form a directed acyclic graph with deterministic execution order.
// Guard: fail-closed on duplicate keys.
func ParseWorkflow(content string) (*Workflow, error) {
	violations, err := Lint(content)
	if err != nil {
		return nil, fmt.Errorf("ciyaml: lint scan error: %w", err)
	}
	if HasErrors(violations) {
		return nil, &LintError{Violations: violations}
	}

	w := &Workflow{
		Jobs: make(map[string]Job),
	}

	lines := splitLines(content)
	n := len(lines)

	var currentSection string
	var currentJobID string
	var currentJob *Job
	var currentStep *Step

	i := 0
	for i < n {
		rawLine := lines[i]
		clean, _, _ := scanQuotesAndStripComment(rawLine)
		trimmed := strings.TrimSpace(clean)
		if trimmed == "" {
			i++
			continue
		}

		indent := countLeadingSpaces(rawLine)

		// Top-level declarations
		if indent == 0 {
			if strings.HasPrefix(trimmed, "name:") {
				val := strings.TrimSpace(trimmed[len("name:"):])
				w.Name = cleanQuotedString(val)
				currentSection = ""
				i++
				continue
			}
			if strings.HasPrefix(trimmed, "on:") {
				currentSection = "on"
				val := strings.TrimSpace(trimmed[len("on:"):])
				if val != "" {
					w.On = parseFlowOrScalarList(val)
				}
				i++
				continue
			}
			if strings.HasPrefix(trimmed, "jobs:") {
				currentSection = "jobs"
				if currentJob != nil {
					if currentStep != nil {
						currentJob.Steps = append(currentJob.Steps, *currentStep)
						currentStep = nil
					}
					w.Jobs[currentJobID] = *currentJob
					currentJob = nil
					currentJobID = ""
				}
				i++
				continue
			}
		}

		if currentSection == "on" && indent > 0 {
			if strings.HasPrefix(trimmed, "-") {
				val := strings.TrimSpace(trimmed[1:])
				val = cleanQuotedString(val)
				if val != "" {
					w.On = append(w.On, val)
				}
			}
			i++
			continue
		}

		if currentSection == "jobs" {
			// Job definition under jobs: (indent == 2)
			if indent == 2 && strings.HasSuffix(trimmed, ":") && !strings.HasPrefix(trimmed, "-") {
				if currentJob != nil {
					if currentStep != nil {
						currentJob.Steps = append(currentJob.Steps, *currentStep)
						currentStep = nil
					}
					w.Jobs[currentJobID] = *currentJob
				}
				jobID := strings.TrimSuffix(trimmed, ":")
				jobID = cleanQuotedString(jobID)
				currentJobID = jobID
				currentJob = &Job{
					ID: jobID,
				}
				currentStep = nil
				i++
				continue
			}

			if currentJob != nil && indent >= 4 {
				colonIdx := findColonOutsideQuotes(trimmed)

				// Step item line: "- name: ..." or "- run: ..."
				if strings.HasPrefix(trimmed, "-") {
					if currentStep != nil {
						currentJob.Steps = append(currentJob.Steps, *currentStep)
					}
					currentStep = &Step{}
					stepContent := strings.TrimSpace(trimmed[1:])
					if stepColon := findColonOutsideQuotes(stepContent); stepColon != -1 {
						k := strings.TrimSpace(stepContent[:stepColon])
						v := strings.TrimSpace(stepContent[stepColon+1:])
						assignStepField(currentStep, k, v)
						if isBlockScalarIndicator(v) {
							scalarContent, nextI := consumeBlockScalar(lines, i+1, indent)
							setStepScalar(currentStep, k, scalarContent)
							i = nextI
							continue
						}
					}
					i++
					continue
				}

				// Properties of current step
				if currentStep != nil && indent > 4 {
					if colonIdx != -1 {
						k := strings.TrimSpace(trimmed[:colonIdx])
						v := strings.TrimSpace(trimmed[colonIdx+1:])
						assignStepField(currentStep, k, v)
						if isBlockScalarIndicator(v) {
							scalarContent, nextI := consumeBlockScalar(lines, i+1, indent)
							setStepScalar(currentStep, k, scalarContent)
							i = nextI
							continue
						}
					}
					i++
					continue
				}

				// Job-level properties
				if colonIdx != -1 {
					k := strings.TrimSpace(trimmed[:colonIdx])
					v := strings.TrimSpace(trimmed[colonIdx+1:])

					switch k {
					case "name":
						currentJob.Name = cleanQuotedString(v)
					case "runs-on":
						currentJob.RunsOn = cleanQuotedString(v)
					case "needs":
						if v != "" {
							currentJob.Needs = parseFlowOrScalarList(v)
						} else {
							items, nextI := consumeListItems(lines, i+1, indent)
							currentJob.Needs = items
							i = nextI
							continue
						}
					case "depends_on":
						if v != "" {
							currentJob.DependsOn = parseFlowOrScalarList(v)
						} else {
							items, nextI := consumeListItems(lines, i+1, indent)
							currentJob.DependsOn = items
							i = nextI
							continue
						}
					}
				}
			}
		}

		i++
	}

	if currentJob != nil {
		if currentStep != nil {
			currentJob.Steps = append(currentJob.Steps, *currentStep)
		}
		w.Jobs[currentJobID] = *currentJob
	}

	return w, nil
}

func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return strings.Split(s, "\n")
}

func countLeadingSpaces(s string) int {
	count := 0
	for _, r := range s {
		if r == ' ' {
			count++
		} else {
			break
		}
	}
	return count
}

func isInsideQuotes(s string) bool {
	inSingle := false
	inDouble := false
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if inSingle {
			if r == '\'' {
				if i+1 < len(runes) && runes[i+1] == '\'' {
					i++
					continue
				}
				inSingle = false
			}
			continue
		}
		if inDouble {
			if r == '\\' {
				i++
				continue
			}
			if r == '"' {
				inDouble = false
			}
			continue
		}
		if r == '\'' {
			inSingle = true
		} else if r == '"' {
			inDouble = true
		}
	}
	return inSingle || inDouble
}

func scanQuotesAndStripComment(line string) (clean string, unclosedQuote rune, quoteCol int) {
	inSingle := false
	inDouble := false
	singleStartCol := 0
	doubleStartCol := 0

	runes := []rune(line)
	n := len(runes)
	commentStart := -1

	for i := 0; i < n; i++ {
		r := runes[i]

		if inSingle {
			if r == '\'' {
				if i+1 < n && runes[i+1] == '\'' {
					i++
					continue
				}
				inSingle = false
			}
			continue
		}

		if inDouble {
			if r == '\\' {
				i++
				continue
			}
			if r == '"' {
				inDouble = false
			}
			continue
		}

		if r == '\'' {
			inSingle = true
			singleStartCol = i + 1
			continue
		}
		if r == '"' {
			inDouble = true
			doubleStartCol = i + 1
			continue
		}
		if r == '#' {
			commentStart = i
			break
		}
	}

	if inSingle {
		return string(runes), '\'', singleStartCol
	}
	if inDouble {
		return string(runes), '"', doubleStartCol
	}

	if commentStart != -1 {
		return string(runes[:commentStart]), 0, 0
	}
	return line, 0, 0
}

func findColonOutsideQuotes(s string) int {
	inSingle := false
	inDouble := false
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if inSingle {
			if r == '\'' {
				if i+1 < len(runes) && runes[i+1] == '\'' {
					i++
					continue
				}
				inSingle = false
			}
			continue
		}
		if inDouble {
			if r == '\\' {
				i++
				continue
			}
			if r == '"' {
				inDouble = false
			}
			continue
		}
		if r == '\'' {
			inSingle = true
			continue
		}
		if r == '"' {
			inDouble = true
			continue
		}
		if r == ':' {
			return i
		}
	}
	return -1
}

func isBlockScalarHeader(trimmed string) bool {
	idx := findColonOutsideQuotes(trimmed)
	if idx == -1 {
		if strings.HasPrefix(trimmed, "-") {
			after := strings.TrimSpace(trimmed[1:])
			return isBlockScalarIndicator(after)
		}
		return false
	}
	after := strings.TrimSpace(trimmed[idx+1:])
	return isBlockScalarIndicator(after)
}

func isBlockScalarIndicator(s string) bool {
	if len(s) == 0 {
		return false
	}
	first := s[0]
	if first != '|' && first != '>' {
		return false
	}
	for i := 1; i < len(s); i++ {
		c := s[i]
		if c != '+' && c != '-' && (c < '0' || c > '9') {
			return false
		}
	}
	return true
}

func consumeBlockScalar(lines []string, startIdx int, headerIndent int) (string, int) {
	var collected []string
	i := startIdx
	for i < len(lines) {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			collected = append(collected, "")
			i++
			continue
		}
		indent := countLeadingSpaces(line)
		if indent <= headerIndent {
			break
		}
		collected = append(collected, strings.TrimRight(line, "\r\n"))
		i++
	}
	return strings.Join(collected, "\n"), i
}

func consumeListItems(lines []string, startIdx int, parentIndent int) ([]string, int) {
	var items []string
	i := startIdx
	for i < len(lines) {
		line := lines[i]
		clean, _, _ := scanQuotesAndStripComment(line)
		trimmed := strings.TrimSpace(clean)
		if trimmed == "" {
			i++
			continue
		}
		indent := countLeadingSpaces(line)
		if indent <= parentIndent {
			break
		}
		if strings.HasPrefix(trimmed, "-") {
			val := strings.TrimSpace(trimmed[1:])
			val = cleanQuotedString(val)
			if val != "" {
				items = append(items, val)
			}
		}
		i++
	}
	return items, i
}

func cleanQuotedString(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func parseFlowOrScalarList(s string) []string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		inner := s[1 : len(s)-1]
		parts := strings.Split(inner, ",")
		var result []string
		for _, p := range parts {
			item := cleanQuotedString(p)
			if item != "" {
				result = append(result, item)
			}
		}
		return result
	}
	item := cleanQuotedString(s)
	if item != "" {
		return []string{item}
	}
	return nil
}

func assignStepField(s *Step, key, val string) {
	val = cleanQuotedString(val)
	switch key {
	case "id":
		s.ID = val
	case "name":
		s.Name = val
	case "uses":
		s.Uses = val
	case "run":
		s.Run = val
	}
}

func setStepScalar(s *Step, key, val string) {
	if key == "run" {
		s.Run = val
	}
}
