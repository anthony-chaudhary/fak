package architest

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// dosClosedCategories defines the five closed DOS categories for reason tokens.
var dosClosedCategories = map[string]bool{
	"MISROUTE":      true,
	"OPERATOR_GATE": true,
	"STALE_CLAIM":   true,
	"TRUE_DRAIN":    true,
	"UNCLASSIFIED":  true,
}

// dosReason represents a parsed [reasons.<TOKEN>] block.
type dosReason struct {
	token       string
	category    string
	hasCategory bool
	summary     string
	hasSummary  bool
	fix         string
	hasFix      bool
	refusal     bool
	hasRefusal  bool
}

// stripTOMLCommentWithEscapes strips trailing '#' comments outside of quoted strings.
func stripTOMLCommentWithEscapes(line string) string {
	var sb strings.Builder
	inQuote := false
	var quoteChar byte
	escaped := false

	for i := 0; i < len(line); i++ {
		c := line[i]
		if inQuote {
			sb.WriteByte(c)
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == quoteChar {
				inQuote = false
			}
			continue
		}

		if c == '"' || c == '\'' {
			inQuote = true
			quoteChar = c
			sb.WriteByte(c)
		} else if c == '#' {
			break
		} else {
			sb.WriteByte(c)
		}
	}
	return sb.String()
}

// unquoteTOMLString removes outer quotes and decodes basic escape sequences.
func unquoteTOMLString(s string) (string, error) {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		unq, err := strconv.Unquote(s)
		if err == nil {
			return unq, nil
		}
		// Fallback for strings that might have raw characters
		return strings.Trim(s, "\""), nil
	}
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return s[1 : len(s)-1], nil
	}
	return s, nil
}

// parseTOMLStringArray extracts quoted string elements from an array string like `["a", "b"]`.
func parseTOMLStringArray(s string) ([]string, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		return nil, fmt.Errorf("expected array enclosed in brackets: %q", s)
	}
	inner := strings.TrimSpace(s[1 : len(s)-1])
	if inner == "" {
		return []string{}, nil
	}

	var elements []string
	inQuote := false
	var quoteChar byte
	escaped := false
	start := -1

	for i := 0; i < len(inner); i++ {
		c := inner[i]
		if inQuote {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == quoteChar {
				inQuote = false
				val, err := unquoteTOMLString(inner[start : i+1])
				if err != nil {
					return nil, err
				}
				elements = append(elements, val)
				start = -1
			}
			continue
		}

		if c == '"' || c == '\'' {
			inQuote = true
			quoteChar = c
			start = i
		}
	}
	if inQuote {
		return nil, fmt.Errorf("unclosed quote in array: %s", s)
	}
	return elements, nil
}

// validateDOSTomlContent parses and strictly validates the dos.toml schema.
func validateDOSTomlContent(content string) (int, int, []string) {
	var errs []string

	lines := strings.Split(content, "\n")
	currentSection := ""
	reasons := map[string]*dosReason{}
	var reasonOrder []string

	lanesTreesFound := false
	lanesTreesCount := 0
	lanesTrees := map[string][]string{}

	for lineIdx, rawLine := range lines {
		lineNum := lineIdx + 1
		clean := strings.TrimSpace(stripTOMLCommentWithEscapes(rawLine))
		if clean == "" {
			continue
		}

		// Check for section header
		if strings.HasPrefix(clean, "[") && strings.HasSuffix(clean, "]") && !strings.Contains(clean, "=") {
			currentSection = clean
			if currentSection == "[lanes.trees]" {
				lanesTreesFound = true
			} else if strings.HasPrefix(currentSection, "[reasons.") {
				token := strings.TrimSuffix(strings.TrimPrefix(currentSection, "[reasons."), "]")
				token = strings.TrimSpace(strings.Trim(token, "\"'"))
				if token == "" {
					errs = append(errs, fmt.Sprintf("line %d: empty reason token in section header %s", lineNum, currentSection))
				} else {
					if _, exists := reasons[token]; exists {
						errs = append(errs, fmt.Sprintf("line %d: duplicate reason section [reasons.%s]", lineNum, token))
					} else {
						reasons[token] = &dosReason{token: token}
						reasonOrder = append(reasonOrder, token)
					}
				}
			}
			continue
		}

		// Inside [reasons.<TOKEN>]
		if strings.HasPrefix(currentSection, "[reasons.") {
			token := strings.TrimSuffix(strings.TrimPrefix(currentSection, "[reasons."), "]")
			token = strings.TrimSpace(strings.Trim(token, "\"'"))
			r := reasons[token]
			if r == nil {
				continue
			}

			key, val, ok := strings.Cut(clean, "=")
			if !ok {
				continue
			}
			key = strings.TrimSpace(key)
			val = strings.TrimSpace(val)

			switch key {
			case "category":
				r.hasCategory = true
				unq, err := unquoteTOMLString(val)
				if err != nil {
					errs = append(errs, fmt.Sprintf("line %d: reason %s category malformed: %v", lineNum, token, err))
				} else {
					r.category = unq
				}
			case "summary":
				r.hasSummary = true
				unq, err := unquoteTOMLString(val)
				if err != nil {
					errs = append(errs, fmt.Sprintf("line %d: reason %s summary malformed: %v", lineNum, token, err))
				} else {
					r.summary = unq
				}
			case "fix":
				r.hasFix = true
				unq, err := unquoteTOMLString(val)
				if err != nil {
					errs = append(errs, fmt.Sprintf("line %d: reason %s fix malformed: %v", lineNum, token, err))
				} else {
					r.fix = unq
				}
			case "refusal":
				r.hasRefusal = true
				valLower := strings.ToLower(val)
				if valLower == "true" {
					r.refusal = true
				} else if valLower == "false" {
					r.refusal = false
				} else {
					errs = append(errs, fmt.Sprintf("line %d: reason %s invalid refusal boolean %q", lineNum, token, val))
				}
			}
			continue
		}

		// Inside [lanes.trees]
		if currentSection == "[lanes.trees]" {
			key, val, ok := strings.Cut(clean, "=")
			if !ok {
				continue
			}
			lane := strings.TrimSpace(key)
			val = strings.TrimSpace(val)
			if lane == "" {
				errs = append(errs, fmt.Sprintf("line %d: empty lane name in [lanes.trees]", lineNum))
				continue
			}
			if _, exists := lanesTrees[lane]; exists {
				errs = append(errs, fmt.Sprintf("line %d: duplicate lane %q in [lanes.trees]", lineNum, lane))
				continue
			}
			globs, err := parseTOMLStringArray(val)
			if err != nil {
				errs = append(errs, fmt.Sprintf("line %d: lane %q tree configuration malformed: %v", lineNum, lane, err))
				continue
			}
			if len(globs) == 0 {
				errs = append(errs, fmt.Sprintf("line %d: lane %q tree has empty glob list", lineNum, lane))
				continue
			}
			for _, g := range globs {
				if strings.TrimSpace(g) == "" {
					errs = append(errs, fmt.Sprintf("line %d: lane %q contains empty glob string", lineNum, lane))
				}
			}
			lanesTrees[lane] = globs
			lanesTreesCount++
			continue
		}
	}

	// 1. Verify [lanes.trees] exists and has valid configurations
	if !lanesTreesFound {
		errs = append(errs, "missing required section [lanes.trees]")
	} else if lanesTreesCount == 0 {
		errs = append(errs, "[lanes.trees] section exists but declares zero lane trees")
	}

	// Verify reason count
	if len(reasons) == 0 {
		errs = append(errs, "dos.toml declares no [reasons.*] blocks")
	}

	// 2. Verify reasons
	for _, token := range reasonOrder {
		r := reasons[token]
		// Category check
		if !r.hasCategory {
			errs = append(errs, fmt.Sprintf("reason %s: missing required field 'category'", token))
		} else if !dosClosedCategories[r.category] {
			errs = append(errs, fmt.Sprintf("reason %s: invalid category %q (must be one of MISROUTE, OPERATOR_GATE, STALE_CLAIM, TRUE_DRAIN, UNCLASSIFIED)", token, r.category))
		}

		// Summary check
		if !r.hasSummary || strings.TrimSpace(r.summary) == "" {
			errs = append(errs, fmt.Sprintf("reason %s: summary must be defined and non-empty", token))
		}

		// Fix check if refusal = true
		if r.refusal {
			if !r.hasFix || strings.TrimSpace(r.fix) == "" {
				errs = append(errs, fmt.Sprintf("reason %s: refusal is true, but fix is missing or empty", token))
			}
		}
	}

	return len(reasons), lanesTreesCount, errs
}

// findRepoRoot locates the root of the repo containing dos.toml.
func findRepoRoot(t *testing.T) string {
	t.Helper()
	if _, self, _, ok := runtime.Caller(0); ok {
		dir := filepath.Dir(self)
		for i := 0; i < 10; i++ {
			if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	wd, err := os.Getwd()
	if err == nil {
		dir := wd
		for i := 0; i < 10; i++ {
			if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	t.Fatal("could not locate repository root with dos.toml")
	return ""
}

// TestDOSTomlSchemaStrict validates that dos.toml satisfies strict schema and closed-category requirements.
func TestDOSTomlSchemaStrict(t *testing.T) {
	root := findRepoRoot(t)
	dosTomlPath := filepath.Join(root, "dos.toml")

	raw, err := os.ReadFile(dosTomlPath)
	if err != nil {
		t.Fatalf("read dos.toml at %s: %v", dosTomlPath, err)
	}

	reasonsCount, lanesCount, errs := validateDOSTomlContent(string(raw))
	if len(errs) > 0 {
		sort.Strings(errs)
		t.Errorf("dos.toml schema validation failed with %d error(s):\n%s", len(errs), strings.Join(errs, "\n"))
	}
	if reasonsCount < 50 {
		t.Errorf("expected at least 50 reasons in dos.toml, got %d", reasonsCount)
	}
	if lanesCount < 50 {
		t.Errorf("expected at least 50 lane trees in dos.toml, got %d", lanesCount)
	}
	t.Logf("validated %d reasons and %d lane trees in dos.toml", reasonsCount, lanesCount)
}

// TestDOSTomlSchemaStrictRejectsViolations verifies that validateDOSTomlContent detects invalid configurations.
func TestDOSTomlSchemaStrictRejectsViolations(t *testing.T) {
	tests := []struct {
		name        string
		toml        string
		errContains string
	}{
		{
			name: "missing category",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
refusal = true
summary = "Test summary"
fix = "Test fix"
`,
			errContains: "missing required field 'category'",
		},
		{
			name: "invalid category",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
category = "CUSTOM_CATEGORY"
refusal = true
summary = "Test summary"
fix = "Test fix"
`,
			errContains: "invalid category \"CUSTOM_CATEGORY\"",
		},
		{
			name: "empty summary",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
category = "MISROUTE"
refusal = true
summary = ""
fix = "Test fix"
`,
			errContains: "summary must be defined and non-empty",
		},
		{
			name: "missing summary",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
category = "MISROUTE"
refusal = true
fix = "Test fix"
`,
			errContains: "summary must be defined and non-empty",
		},
		{
			name: "refusal true without fix",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
category = "OPERATOR_GATE"
refusal = true
summary = "Test summary"
`,
			errContains: "refusal is true, but fix is missing or empty",
		},
		{
			name: "refusal true with empty fix",
			toml: `[lanes.trees]
lane1 = ["internal/lane1/**"]

[reasons.TEST_REASON]
category = "OPERATOR_GATE"
refusal = true
summary = "Test summary"
fix = "   "
`,
			errContains: "refusal is true, but fix is missing or empty",
		},
		{
			name: "missing lanes.trees",
			toml: `[reasons.TEST_REASON]
category = "OPERATOR_GATE"
refusal = false
summary = "Test summary"
`,
			errContains: "missing required section [lanes.trees]",
		},
		{
			name: "empty lane tree glob list",
			toml: `[lanes.trees]
lane1 = []

[reasons.TEST_REASON]
category = "OPERATOR_GATE"
refusal = false
summary = "Test summary"
`,
			errContains: "has empty glob list",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, errs := validateDOSTomlContent(tc.toml)
			found := false
			for _, e := range errs {
				if strings.Contains(e, tc.errContains) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("expected error containing %q, got: %v", tc.errContains, errs)
			}
		})
	}
}
