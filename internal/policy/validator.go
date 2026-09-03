package policy

import (
	"errors"
	"fmt"
	"regexp/syntax"
	"strings"
)

// ErrInvalidRegexPattern indicates a regular expression pattern is syntactically
// invalid or unsafe (e.g. vulnerable to catastrophic ReDoS backtracking).
var ErrInvalidRegexPattern = errors.New("invalid or unsafe regex pattern")

// ValidateRegexSafety validates a regular expression pattern for syntactic validity
// and ReDoS safety. It checks:
//  1. Raw pattern for unsupported/unsafe constructs (lookaheads, lookbehinds, backreferences).
//  2. Syntax parsing via regexp/syntax.Parse with ClassNL|Perl flags.
//  3. AST analysis for nested repetitions (e.g. (a+)+, (a*)*, ([0-9]+)+, (a|b+)+).
//  4. AST analysis for consecutive overlapping repetitions (e.g. .*.*, a+a+).
func ValidateRegexSafety(pattern string) error {
	if err := checkRawPatternSafety(pattern); err != nil {
		return err
	}

	re, err := syntax.Parse(pattern, syntax.ClassNL|syntax.Perl)
	if err != nil {
		return fmt.Errorf("%w: syntax error: %v", ErrInvalidRegexPattern, err)
	}

	return checkASTSafety(re)
}

// checkRawPatternSafety inspects the unparsed pattern for lookaheads, lookbehinds,
// and backreferences, respecting escape sequences.
func checkRawPatternSafety(pattern string) error {
	inCharClass := false
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		if c == '\\' {
			if i+1 < len(pattern) {
				next := pattern[i+1]
				if !inCharClass && next >= '1' && next <= '9' {
					return fmt.Errorf("%w: backreference not supported (\\%c)", ErrInvalidRegexPattern, next)
				}
				i++ // skip escaped character
				continue
			}
			continue
		}
		if c == '[' && !inCharClass {
			inCharClass = true
			continue
		}
		if c == ']' && inCharClass {
			inCharClass = false
			continue
		}
		if !inCharClass {
			if strings.HasPrefix(pattern[i:], "(?=") {
				return fmt.Errorf("%w: positive lookahead is not supported: (?=...)", ErrInvalidRegexPattern)
			}
			if strings.HasPrefix(pattern[i:], "(?!") {
				return fmt.Errorf("%w: negative lookahead is not supported: (?!...)", ErrInvalidRegexPattern)
			}
			if strings.HasPrefix(pattern[i:], "(?<=") {
				return fmt.Errorf("%w: positive lookbehind is not supported: (?<=...)", ErrInvalidRegexPattern)
			}
			if strings.HasPrefix(pattern[i:], "(?<!") {
				return fmt.Errorf("%w: negative lookbehind is not supported: (?<!...)", ErrInvalidRegexPattern)
			}
		}
	}
	return nil
}

// checkASTSafety recursively inspects the parsed regexp AST for catastrophic backtracking.
func checkASTSafety(re *syntax.Regexp) error {
	if re == nil {
		return nil
	}

	if isMultiRepetition(re) {
		if len(re.Sub) > 0 {
			// Nested repetition check
			if sub := findInnerRepetition(re.Sub[0]); sub != nil {
				return fmt.Errorf("%w: nested repetition: %s contains %s", ErrInvalidRegexPattern, re.Op, sub.Op)
			}
		}
	}

	if re.Op == syntax.OpConcat {
		if err := checkConcatOverlaps(re.Sub); err != nil {
			return err
		}
	}

	for _, sub := range re.Sub {
		if err := checkASTSafety(sub); err != nil {
			return err
		}
	}

	return nil
}

// isMultiRepetition reports whether an operator can match its subexpression more than once.
func isMultiRepetition(re *syntax.Regexp) bool {
	if re == nil {
		return false
	}
	switch re.Op {
	case syntax.OpStar, syntax.OpPlus:
		return true
	case syntax.OpRepeat:
		return re.Max == -1 || re.Max > 1
	default:
		return false
	}
}

// findInnerRepetition finds an inner repetition or quest within a subexpression
// that would cause catastrophic backtracking if repeated.
func findInnerRepetition(re *syntax.Regexp) *syntax.Regexp {
	if re == nil {
		return nil
	}
	switch re.Op {
	case syntax.OpStar, syntax.OpPlus, syntax.OpQuest:
		return re
	case syntax.OpRepeat:
		return re
	case syntax.OpCapture:
		if len(re.Sub) > 0 {
			return findInnerRepetition(re.Sub[0])
		}
		return nil
	case syntax.OpConcat, syntax.OpAlternate:
		for _, sub := range re.Sub {
			if found := findInnerRepetition(sub); found != nil {
				return found
			}
		}
		return nil
	default:
		return nil
	}
}

// unwrapCapture strips wrapping OpCapture nodes.
func unwrapCapture(re *syntax.Regexp) *syntax.Regexp {
	for re != nil && re.Op == syntax.OpCapture && len(re.Sub) > 0 {
		re = re.Sub[0]
	}
	return re
}

// checkConcatOverlaps checks for consecutive multi-repetitions that can match overlapping characters.
func checkConcatOverlaps(subs []*syntax.Regexp) error {
	var nonZero []*syntax.Regexp
	for _, sub := range subs {
		if !isZeroWidth(sub.Op) {
			nonZero = append(nonZero, sub)
		}
	}
	for i := 0; i+1 < len(nonZero); i++ {
		first := unwrapCapture(nonZero[i])
		second := unwrapCapture(nonZero[i+1])
		if isMultiRepetition(first) && isMultiRepetition(second) {
			set1 := matchableRuneSet(first)
			set2 := matchableRuneSet(second)
			if set1.overlaps(set2) {
				return fmt.Errorf("%w: consecutive overlapping repetitions (%s and %s) can cause catastrophic backtracking", ErrInvalidRegexPattern, first.Op, second.Op)
			}
		}
	}
	return nil
}

// isZeroWidth reports whether an operator matches zero characters.
func isZeroWidth(op syntax.Op) bool {
	switch op {
	case syntax.OpBeginLine, syntax.OpEndLine,
		syntax.OpBeginText, syntax.OpEndText,
		syntax.OpWordBoundary, syntax.OpNoWordBoundary,
		syntax.OpEmptyMatch:
		return true
	default:
		return false
	}
}

type runeRange struct {
	lo, hi rune
}

type runeSet struct {
	anyChar bool
	ranges  []runeRange
}

func (s runeSet) isEmpty() bool {
	return !s.anyChar && len(s.ranges) == 0
}

func (s runeSet) overlaps(other runeSet) bool {
	if s.isEmpty() || other.isEmpty() {
		return false
	}
	if s.anyChar || other.anyChar {
		return true
	}
	for _, r1 := range s.ranges {
		for _, r2 := range other.ranges {
			if r1.lo <= r2.hi && r2.lo <= r1.hi {
				return true
			}
		}
	}
	return false
}

func matchableRuneSet(re *syntax.Regexp) runeSet {
	if re == nil {
		return runeSet{}
	}
	switch re.Op {
	case syntax.OpAnyChar, syntax.OpAnyCharNotNL:
		return runeSet{anyChar: true}
	case syntax.OpLiteral:
		if len(re.Rune) > 0 {
			r := re.Rune[0]
			return runeSet{ranges: []runeRange{{lo: r, hi: r}}}
		}
		return runeSet{}
	case syntax.OpCharClass:
		var ranges []runeRange
		for i := 0; i+1 < len(re.Rune); i += 2 {
			ranges = append(ranges, runeRange{lo: re.Rune[i], hi: re.Rune[i+1]})
		}
		return runeSet{ranges: ranges}
	case syntax.OpCapture, syntax.OpStar, syntax.OpPlus, syntax.OpQuest, syntax.OpRepeat:
		if len(re.Sub) > 0 {
			return matchableRuneSet(re.Sub[0])
		}
		return runeSet{}
	case syntax.OpConcat:
		for _, sub := range re.Sub {
			if !isZeroWidth(sub.Op) {
				return matchableRuneSet(sub)
			}
		}
		return runeSet{}
	case syntax.OpAlternate:
		var combined runeSet
		for _, sub := range re.Sub {
			s := matchableRuneSet(sub)
			if s.anyChar {
				combined.anyChar = true
			}
			combined.ranges = append(combined.ranges, s.ranges...)
		}
		return combined
	default:
		return runeSet{}
	}
}
