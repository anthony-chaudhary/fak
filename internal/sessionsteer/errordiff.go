package sessionsteer

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// SynthesizedDiff holds the isolated root compiler error and actionable diff.
// Raw compiler errors can emit hundreds of lines of cascading type errors;
// SynthesizedDiff isolates the first/root error and provides an actionable
// unified line diff for autoregressive model consumption (Issue #11065).
type SynthesizedDiff struct {
	Compiler        string `json:"compiler"`         // "go", "tsc", "cargo", "generic"
	FilePath        string `json:"file_path"`        // Relative or absolute path to the file
	Line            int    `json:"line"`             // 1-based line number
	Column          int    `json:"column"`           // 1-based column number
	RootCause       string `json:"root_cause"`       // Concise description of root cause
	OffendingToken  string `json:"offending_token"`  // Identifier or symbol that triggered the error
	SuggestedFix    string `json:"suggested_fix"`    // Actionable instruction or code suggestion
	OriginalSnippet string `json:"original_snippet"` // Extracted source snippet if present in compiler output
	FormattedDiff   string `json:"formatted_diff"`   // Actionable unified line diff
	CascadingCount  int    `json:"cascading_count"`  // Number of subsequent cascading errors suppressed
}

// String returns the unified diff.
func (d *SynthesizedDiff) String() string {
	if d == nil {
		return ""
	}
	return d.FormattedDiff
}

// Summary returns a one-line human-readable summary of the synthesized diff.
func (d *SynthesizedDiff) Summary() string {
	if d == nil {
		return ""
	}
	return fmt.Sprintf("[%s] %s:%d:%d: %s (fix: %s)", d.Compiler, d.FilePath, d.Line, d.Column, d.RootCause, d.SuggestedFix)
}

// ErrNoCompilerError is returned when no recognizable compiler error can be parsed.
var ErrNoCompilerError = errors.New("sessionsteer: no recognizable compiler error found in output")

// SynthesizeErrorDiff isolates the root compiler error from raw compiler output
// (Go, TypeScript, Cargo/Rustc, or generic) and synthesizes an actionable unified diff.
func SynthesizeErrorDiff(compilerOutput string) (*SynthesizedDiff, error) {
	trimmed := strings.TrimSpace(compilerOutput)
	if trimmed == "" {
		return nil, errors.New("sessionsteer: empty compiler output")
	}

	// 1. Try Cargo / Rustc first if rust error patterns are present.
	if diff, ok := parseCargoOutput(trimmed); ok {
		return diff, nil
	}

	// 2. Try TypeScript / tsc if TS error markers are present.
	if diff, ok := parseTypeScriptOutput(trimmed); ok {
		return diff, nil
	}

	// 3. Try Go compiler output.
	if diff, ok := parseGoOutput(trimmed); ok {
		return diff, nil
	}

	// 4. Try generic compiler fallback (file:line:col: error).
	if diff, ok := parseGenericOutput(trimmed); ok {
		return diff, nil
	}

	return nil, ErrNoCompilerError
}

// -----------------------------------------------------------------------------
// Cargo / Rustc Parser
// -----------------------------------------------------------------------------

var (
	cargoErrorHeaderRegex = regexp.MustCompile(`(?m)^error(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>[^\n]+)`)
	cargoLocationRegex    = regexp.MustCompile(`(?m)^\s*-->\s*(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+)`)
	cargoSnippetRegex     = regexp.MustCompile(`(?m)^\s*(?P<line>\d+)\s*\|\s?(?P<src>.*)$`)
)

func parseCargoOutput(output string) (*SynthesizedDiff, bool) {
	// Must have error header and location arrow
	locMatch := cargoLocationRegex.FindStringSubmatchIndex(output)
	if locMatch == nil {
		return nil, false
	}

	headerMatches := cargoErrorHeaderRegex.FindAllStringSubmatchIndex(output, -1)
	if len(headerMatches) == 0 {
		return nil, false
	}

	// Count real rustc errors (excluding "could not compile", "aborting due to")
	totalErrors := 0
	for _, hm := range headerMatches {
		msg := output[hm[4]:hm[5]]
		if !strings.HasPrefix(msg, "could not compile") && !strings.HasPrefix(msg, "aborting due to") {
			totalErrors++
		}
	}

	// First error header
	firstHM := headerMatches[0]
	var code string
	if firstHM[2] >= 0 && firstHM[3] >= 0 {
		code = output[firstHM[2]:firstHM[3]]
	}
	rawMsg := strings.TrimSpace(output[firstHM[4]:firstHM[5]])

	// Normalize backticks to single quotes for consistency: `foo` -> 'foo'
	normalizedMsg := strings.ReplaceAll(rawMsg, "`", "'")

	// Location info
	locSubmatches := cargoLocationRegex.FindStringSubmatch(output)
	filePath := locSubmatches[1]
	line, _ := strconv.Atoi(locSubmatches[2])
	col, _ := strconv.Atoi(locSubmatches[3])

	// Look for snippet line matching line number
	var snippet string
	snippetMatches := cargoSnippetRegex.FindAllStringSubmatch(output, -1)
	for _, sm := range snippetMatches {
		sLine, _ := strconv.Atoi(sm[1])
		if sLine == line {
			snippet = sm[2]
			break
		}
	}

	// Form root cause
	var rootCause string
	if code != "" {
		rootCause = fmt.Sprintf("%s: %s", code, normalizedMsg)
	} else {
		rootCause = normalizedMsg
	}

	// Token & fix extraction
	token, fix := deriveCargoFix(normalizedMsg)

	cascadingCount := 0
	if totalErrors > 1 {
		cascadingCount = totalErrors - 1
	}

	diff := &SynthesizedDiff{
		Compiler:        "cargo",
		FilePath:        strings.TrimPrefix(filePath, "./"),
		Line:            line,
		Column:          col,
		RootCause:       rootCause,
		OffendingToken:  token,
		SuggestedFix:    fix,
		OriginalSnippet: snippet,
		CascadingCount:  cascadingCount,
	}
	diff.FormattedDiff = formatUnifiedDiff(diff.FilePath, diff.Line, diff.Column, diff.RootCause, diff.OriginalSnippet, diff.SuggestedFix)
	return diff, true
}

func deriveCargoFix(msg string) (string, string) {
	// cannot find value 'baz' in this scope
	reVal := regexp.MustCompile(`cannot find value '([^']+)' in this scope`)
	if m := reVal.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("bring '%s' into scope or define it", token)
	}

	// cannot find function 'baz' in this scope
	reFn := regexp.MustCompile(`cannot find function '([^']+)' in this scope`)
	if m := reFn.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("define function '%s' or import it into scope", token)
	}

	// cannot find type 'baz' in this scope
	reType := regexp.MustCompile(`cannot find type '([^']+)' in this scope`)
	if m := reType.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("define type '%s' or import it into scope", token)
	}

	// cannot find macro 'baz' in this scope
	reMacro := regexp.MustCompile(`cannot find macro '([^']+)' in this scope`)
	if m := reMacro.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("import macro '%s' or define it", token)
	}

	// mismatched types
	if strings.Contains(msg, "mismatched types") {
		return "", "match expected type or convert value"
	}

	// unused variable: 'foo'
	reUnused := regexp.MustCompile(`unused variable:\s*'([^']+)'`)
	if m := reUnused.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("prefix with underscore '_%s' or remove it", token)
	}

	// generic quoted token
	reQuoted := regexp.MustCompile(`'([^']+)'`)
	if m := reQuoted.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("fix issue with '%s': %s", token, msg)
	}

	return "", fmt.Sprintf("fix rustc error: %s", msg)
}

// -----------------------------------------------------------------------------
// TypeScript Parser
// -----------------------------------------------------------------------------

var (
	tscErrorStandardRegex = regexp.MustCompile(`(?m)^([^\s:(]+?):(\d+):(\d+)\s*-\s*error\s+(TS\d+):\s*(.+)$`)
	tscErrorVsRegex       = regexp.MustCompile(`(?m)^([^\s:(]+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)$`)
	tscSnippetRegex       = regexp.MustCompile(`(?m)^\s*(\d+)\s+(.*)$`)
)

func parseTypeScriptOutput(output string) (*SynthesizedDiff, bool) {
	var matches [][]int
	var submatches []string
	var isStandard bool

	if m := tscErrorStandardRegex.FindAllStringSubmatchIndex(output, -1); len(m) > 0 {
		matches = m
		submatches = tscErrorStandardRegex.FindStringSubmatch(output)
		isStandard = true
	} else if m := tscErrorVsRegex.FindAllStringSubmatchIndex(output, -1); len(m) > 0 {
		matches = m
		submatches = tscErrorVsRegex.FindStringSubmatch(output)
		isStandard = false
	} else {
		return nil, false
	}

	totalErrors := len(matches)
	cascadingCount := 0
	if totalErrors > 1 {
		cascadingCount = totalErrors - 1
	}

	filePath := submatches[1]
	line, _ := strconv.Atoi(submatches[2])
	col, _ := strconv.Atoi(submatches[3])
	code := submatches[4]
	msg := strings.TrimSpace(submatches[5])

	rootCause := fmt.Sprintf("%s: %s", code, msg)
	token, fix := deriveTypeScriptFix(code, msg)

	// Look for snippet line if tsc printed code
	var snippet string
	if isStandard {
		for _, sm := range tscSnippetRegex.FindAllStringSubmatch(output, -1) {
			sLine, _ := strconv.Atoi(sm[1])
			if sLine == line {
				snippet = sm[2]
				break
			}
		}
	}

	diff := &SynthesizedDiff{
		Compiler:        "tsc",
		FilePath:        strings.TrimPrefix(filePath, "./"),
		Line:            line,
		Column:          col,
		RootCause:       rootCause,
		OffendingToken:  token,
		SuggestedFix:    fix,
		OriginalSnippet: snippet,
		CascadingCount:  cascadingCount,
	}
	diff.FormattedDiff = formatUnifiedDiff(diff.FilePath, diff.Line, diff.Column, diff.RootCause, diff.OriginalSnippet, diff.SuggestedFix)
	return diff, true
}

func deriveTypeScriptFix(code, msg string) (string, string) {
	// TS2304: Cannot find name 'bar'.
	reName := regexp.MustCompile(`Cannot find name '([^']+)'.?`)
	if m := reName.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("declare or import '%s'", token)
	}

	// TS2322: Type 'number' is not assignable to type 'string'.
	reAssign := regexp.MustCompile(`Type '([^']+)' is not assignable to type '([^']+)'.?`)
	if m := reAssign.FindStringSubmatch(msg); len(m) > 1 {
		fromType := m[1]
		toType := m[2]
		return fromType, fmt.Sprintf("cast or convert expression from '%s' to '%s'", fromType, toType)
	}

	// TS2339: Property 'calculate' does not exist on type 'MathUtils'.
	reProp := regexp.MustCompile(`Property '([^']+)' does not exist on type '([^']+)'.?`)
	if m := reProp.FindStringSubmatch(msg); len(m) > 1 {
		prop := m[1]
		typeName := m[2]
		return prop, fmt.Sprintf("add property '%s' to type '%s' or check property name", prop, typeName)
	}

	// TS2345: Argument of type 'A' is not assignable to parameter of type 'B'.
	reArg := regexp.MustCompile(`Argument of type '([^']+)' is not assignable to parameter of type '([^']+)'.?`)
	if m := reArg.FindStringSubmatch(msg); len(m) > 1 {
		argType := m[1]
		paramType := m[2]
		return argType, fmt.Sprintf("pass an argument of type '%s' instead of '%s'", paramType, argType)
	}

	// Generic quoted token
	reQuoted := regexp.MustCompile(`'([^']+)'`)
	if m := reQuoted.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("resolve TypeScript %s regarding '%s'", code, token)
	}

	return "", fmt.Sprintf("resolve TypeScript %s: %s", code, msg)
}

// -----------------------------------------------------------------------------
// Go Compiler Parser
// -----------------------------------------------------------------------------

var goErrorRegex = regexp.MustCompile(`(?m)^(?:\./)?([^\s:]+\.go):(\d+)(?::(\d+))?:\s*(.+)$`)

func parseGoOutput(output string) (*SynthesizedDiff, bool) {
	allMatches := goErrorRegex.FindAllStringSubmatch(output, -1)
	if len(allMatches) == 0 {
		return nil, false
	}

	totalErrors := len(allMatches)
	cascadingCount := 0
	if totalErrors > 1 {
		cascadingCount = totalErrors - 1
	}

	firstMatch := allMatches[0]
	filePath := firstMatch[1]
	line, _ := strconv.Atoi(firstMatch[2])
	col := 1
	if firstMatch[3] != "" {
		col, _ = strconv.Atoi(firstMatch[3])
	}
	msg := strings.TrimSpace(firstMatch[4])

	token, fix := deriveGoFix(msg)

	diff := &SynthesizedDiff{
		Compiler:       "go",
		FilePath:       strings.TrimPrefix(filePath, "./"),
		Line:           line,
		Column:         col,
		RootCause:      msg,
		OffendingToken: token,
		SuggestedFix:   fix,
		CascadingCount: cascadingCount,
	}
	diff.FormattedDiff = formatUnifiedDiff(diff.FilePath, diff.Line, diff.Column, diff.RootCause, "", diff.SuggestedFix)
	return diff, true
}

func deriveGoFix(msg string) (string, string) {
	// undefined: foo
	reUndef := regexp.MustCompile(`undefined:\s*([^\s,]+)`)
	if m := reUndef.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("declare or import '%s'", token)
	}

	// cannot use x (variable of type int) as type string
	// cannot use x (variable of type int) as string in assignment
	reCannotUseTyped := regexp.MustCompile(`cannot use (.+?) \((?:variable of type|constant of type|type|untyped)\s+([^)]+)\)\s+as\s+(?:type\s+)?([^\s]+)`)
	if m := reCannotUseTyped.FindStringSubmatch(msg); len(m) > 3 {
		token := m[1]
		fromType := m[2]
		toType := m[3]
		return token, fmt.Sprintf("convert '%s' to %s or change its type from %s", token, toType, fromType)
	}

	// cannot use x as type string
	reCannotUse := regexp.MustCompile(`cannot use (.+?)\s+as\s+(?:type\s+)?([^\s]+)`)
	if m := reCannotUse.FindStringSubmatch(msg); len(m) > 2 {
		token := m[1]
		toType := m[2]
		return token, fmt.Sprintf("convert '%s' to type %s", token, toType)
	}

	// imported and not used: "fmt" or "fmt" imported and not used
	reImported := regexp.MustCompile(`(?:imported and not used:\s*"([^"]+)"|"([^"]+)"\s*imported and not used)`)
	if m := reImported.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		if token == "" && len(m) > 2 {
			token = m[2]
		}
		return token, fmt.Sprintf("remove unused import %q or use blank identifier _", token)
	}

	// not enough arguments in call to add
	reNotEnoughArgs := regexp.MustCompile(`not enough arguments in call to (\S+)`)
	if m := reNotEnoughArgs.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("provide required arguments in call to %s", token)
	}

	// too many arguments in call to add
	reTooManyArgs := regexp.MustCompile(`too many arguments in call to (\S+)`)
	if m := reTooManyArgs.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		return token, fmt.Sprintf("remove extra arguments in call to %s", token)
	}

	// type Foo has no field or method Bar
	reNoField := regexp.MustCompile(`type (\S+) has no field or method (\S+)`)
	if m := reNoField.FindStringSubmatch(msg); len(m) > 2 {
		typeName := m[1]
		token := m[2]
		return token, fmt.Sprintf("check field or method '%s' on type %s", token, typeName)
	}

	// syntax error: unexpected semicolon...
	reSyntax := regexp.MustCompile(`syntax error:\s*(.+)`)
	if m := reSyntax.FindStringSubmatch(msg); len(m) > 1 {
		detail := m[1]
		return "syntax error", fmt.Sprintf("correct syntax: %s", detail)
	}

	// General quoted token
	reQuoted := regexp.MustCompile(`'([^']+)'|"([^"]+)"`)
	if m := reQuoted.FindStringSubmatch(msg); len(m) > 1 {
		token := m[1]
		if token == "" && len(m) > 2 {
			token = m[2]
		}
		return token, fmt.Sprintf("fix issue with '%s': %s", token, msg)
	}

	return "", fmt.Sprintf("fix compiler error: %s", msg)
}

// -----------------------------------------------------------------------------
// Generic Compiler Fallback
// -----------------------------------------------------------------------------

var genericErrorRegex = regexp.MustCompile(`(?m)^(?:\./)?([^\s:]+):(\d+)(?::(\d+))?:\s*(?:(?:error|warning):\s*)?(.+)$`)

func parseGenericOutput(output string) (*SynthesizedDiff, bool) {
	allMatches := genericErrorRegex.FindAllStringSubmatch(output, -1)
	if len(allMatches) == 0 {
		return nil, false
	}

	totalErrors := len(allMatches)
	cascadingCount := 0
	if totalErrors > 1 {
		cascadingCount = totalErrors - 1
	}

	firstMatch := allMatches[0]
	filePath := firstMatch[1]
	line, _ := strconv.Atoi(firstMatch[2])
	col := 1
	if firstMatch[3] != "" {
		col, _ = strconv.Atoi(firstMatch[3])
	}
	msg := strings.TrimSpace(firstMatch[4])

	var token string
	reQuoted := regexp.MustCompile(`'([^']+)'|` + "`" + `([^` + "`" + `]+)` + "`" + `|"([^"]+)"`)
	if m := reQuoted.FindStringSubmatch(msg); len(m) > 1 {
		for i := 1; i < len(m); i++ {
			if m[i] != "" {
				token = m[i]
				break
			}
		}
	}

	var fix string
	if token != "" {
		fix = fmt.Sprintf("resolve issue with '%s': %s", token, msg)
	} else {
		fix = fmt.Sprintf("resolve error at %s:%d: %s", filePath, line, msg)
	}

	diff := &SynthesizedDiff{
		Compiler:       "generic",
		FilePath:       strings.TrimPrefix(filePath, "./"),
		Line:           line,
		Column:         col,
		RootCause:      msg,
		OffendingToken: token,
		SuggestedFix:   fix,
		CascadingCount: cascadingCount,
	}
	diff.FormattedDiff = formatUnifiedDiff(diff.FilePath, diff.Line, diff.Column, diff.RootCause, "", diff.SuggestedFix)
	return diff, true
}

// -----------------------------------------------------------------------------
// Unified Diff Formatting
// -----------------------------------------------------------------------------

func formatUnifiedDiff(filePath string, line, col int, rootCause, snippet, suggestedFix string) string {
	cleanPath := strings.TrimPrefix(filePath, "./")
	var b strings.Builder
	b.WriteString(fmt.Sprintf("--- a/%s\n", cleanPath))
	b.WriteString(fmt.Sprintf("+++ b/%s\n", cleanPath))
	b.WriteString(fmt.Sprintf("@@ -%d,1 +%d,1 @@\n", line, line))

	if snippet != "" {
		indent := getLeadingWhitespace(snippet)
		b.WriteString(fmt.Sprintf("-%s\n", snippet))
		b.WriteString(fmt.Sprintf("+%s// fix: %s\n", indent, suggestedFix))
	} else {
		if col > 0 {
			b.WriteString(fmt.Sprintf("- // line %d:%d: %s\n", line, col, rootCause))
		} else {
			b.WriteString(fmt.Sprintf("- // line %d: %s\n", line, rootCause))
		}
		b.WriteString(fmt.Sprintf("+ // fix: %s\n", suggestedFix))
	}

	return b.String()
}

func getLeadingWhitespace(s string) string {
	for i, r := range s {
		if r != ' ' && r != '\t' {
			return s[:i]
		}
	}
	return ""
}
