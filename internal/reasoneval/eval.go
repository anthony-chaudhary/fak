// Package reasoneval provides deterministic rule scoring for reasoning
// benchmarks whose answers do not require a model judge.
//
// The AIME and GPQA extraction order adapts DeepInfer's Apache-2.0 evaluation
// rules at deepseek-ai/DeepInfer@9bd9cf243fc53ef9afbf8f04d391d989117d6ab9:
// eval/aime/bench.py:73-148 and eval/gpqa/bench.py:111-134. HLE remains outside
// this rule scorer because eval/score.py:43-55 requires an external judge.
// AIME last-complete-box selection and numeric normalization also follow
// deepseek-ai/DeepInfer@553741557a57df44a46fb3ab72b6aafeaf945337.
package reasoneval

import (
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// RuleResult is a source-labelled deterministic benchmark score.
type RuleResult struct {
	Prediction string `json:"prediction"`
	Source     string `json:"source"`
	Correct    bool   `json:"correct"`
}

// ErrJudgeRequired reports that deterministic rule scoring is not authoritative
// for the selected benchmark.
var ErrJudgeRequired = errors.New("benchmark requires an external judge")

// GPQA follows Python's Unicode regex classes at DeepInfer@553741557a57df44a46fb3ab72b6aafeaf945337.
// Python whitespace includes the separators and U+001C..U+001F, unlike Go's \s.
const gpqaPythonSpace = `[\t-\r\x1c-\x1f\x85\p{Z}]`

var (
	// Python \w is letters, all numbers (not just Nd), and underscore. Consuming
	// its complement after A-D emulates \b here without losing a following line:
	// the next ^ match starts after, not on, a consumed newline.
	// Disable case folding only at the boundary: U+0345 folds to a letter but is
	// not a Python word character, even in a case-insensitive expression.
	gpqaAnswerLine = regexp.MustCompile(`(?im)^` + gpqaPythonSpace + `*ANSWER` + gpqaPythonSpace + `*[:：]` + gpqaPythonSpace + `*\(?([A-D])(?-i:[^\p{L}\p{N}_]|\z)`)
	gpqaBoxed      = regexp.MustCompile(`\\boxed\{` + gpqaPythonSpace + `*\(?([A-D])\)?` + gpqaPythonSpace + `*\}`)
	gpqaStandalone = regexp.MustCompile(`[A-D]`)
)

// ScoreRule extracts and scores one AIME or GPQA prediction. HLE fails closed
// with ErrJudgeRequired instead of manufacturing a rule-based score.
func ScoreRule(benchmark, text, gold string) (RuleResult, error) {
	switch strings.ToLower(strings.TrimSpace(benchmark)) {
	case "aime":
		prediction, source := extractAIME(text)
		return RuleResult{
			Prediction: prediction,
			Source:     source,
			Correct:    prediction != "" && normalizeAIME(prediction) == normalizeAIME(gold),
		}, nil
	case "gpqa":
		prediction, source := extractGPQA(text)
		return RuleResult{
			Prediction: prediction,
			Source:     source,
			Correct:    prediction != "" && strings.EqualFold(prediction, strings.TrimSpace(gold)),
		}, nil
	case "hle":
		return RuleResult{}, ErrJudgeRequired
	default:
		return RuleResult{}, fmt.Errorf("unsupported benchmark %q", benchmark)
	}
}

func extractAIME(text string) (string, string) {
boxedSearch:
	for offset := len(text); offset > 0; {
		idx := strings.LastIndex(text[:offset], `\boxed`)
		if idx < 0 {
			break
		}
		open := strings.IndexByte(text[idx+len(`\boxed`):], '{')
		if open >= 0 {
			open += idx + len(`\boxed`)
			depth := 0
			for pos := open; pos < len(text); pos++ {
				switch text[pos] {
				case '{':
					depth++
				case '}':
					depth--
					if depth == 0 {
						if answer := normalizeAIME(text[open+1 : pos]); answer != "" {
							return answer, "boxed"
						}
						// A complete empty box supersedes earlier boxed answers.
						break boxedSearch
					}
				}
			}
		}
		offset = idx
	}

	tail := reasoningTail(text)
	spans := unicodeDecimalSpans(tail)
	for i := len(spans) - 1; i >= 0; i-- {
		span := spans[i]
		if pythonWordOrDotBefore(tail, span.start) || pythonWordOrDotAfter(tail, span.end) {
			continue
		}
		return normalizeAIME(tail[span.start:span.end]), "tail_number"
	}
	return "", "none"
}

type byteSpan struct {
	start int
	end   int
}

// unicodeDecimalSpans mirrors Python's -?\d{1,3}: Go's regexp \d is ASCII-only,
// so the span scan uses the Unicode Decimal_Digit table explicitly.
func unicodeDecimalSpans(text string) []byteSpan {
	var spans []byteSpan
	for at := 0; at < len(text); {
		r, size := utf8.DecodeRuneInString(text[at:])
		if !unicode.Is(unicode.Digit, r) {
			at += size
			continue
		}
		start := at
		if at > 0 {
			previous, previousSize := utf8.DecodeLastRuneInString(text[:at])
			if previous == '-' {
				start -= previousSize
			}
		}
		digits := 0
		end := at
		for end < len(text) {
			digit, digitSize := utf8.DecodeRuneInString(text[end:])
			if !unicode.Is(unicode.Digit, digit) {
				break
			}
			digits++
			end += digitSize
		}
		if digits <= 3 {
			spans = append(spans, byteSpan{start: start, end: end})
		}
		at = end
	}
	return spans
}

func normalizeAIME(answer string) string {
	answer = strings.TrimSpace(strings.Trim(strings.TrimSpace(answer), "$"))
	replacer := strings.NewReplacer(`\left`, "", `\right`, "", `\!`, "", `\,`, "", `\;`, "", "{,}", "", ",", "", " ", "")
	answer = strings.TrimRight(replacer.Replace(answer), ".")
	answer = strings.TrimPrefix(answer, "+")
	if ascii, ok := decimalASCII(answer); ok {
		// Canonicalize validated decimal digits without a machine-int limit.
		negative := strings.HasPrefix(ascii, "-")
		digits := strings.TrimLeft(strings.TrimPrefix(ascii, "-"), "0")
		if digits == "" {
			return "0"
		}
		if negative {
			return "-" + digits
		}
		return digits
	}
	return strings.ToLower(answer)
}

func decimalASCII(value string) (string, bool) {
	var out strings.Builder
	if strings.HasPrefix(value, "-") {
		out.WriteByte('-')
		value = value[1:]
	}
	if value == "" {
		return "", false
	}
	for _, r := range value {
		digit, ok := unicodeDecimalValue(r)
		if !ok {
			return "", false
		}
		out.WriteByte(byte('0' + digit))
	}
	return out.String(), true
}

func unicodeDecimalValue(r rune) (int, bool) {
	for _, item := range unicode.Digit.R16 {
		if uint32(r) < uint32(item.Lo) || uint32(r) > uint32(item.Hi) || (uint32(r)-uint32(item.Lo))%uint32(item.Stride) != 0 {
			continue
		}
		return int((uint32(r)-uint32(item.Lo))/uint32(item.Stride)) % 10, true
	}
	for _, item := range unicode.Digit.R32 {
		if uint32(r) < item.Lo || uint32(r) > item.Hi || (uint32(r)-item.Lo)%item.Stride != 0 {
			continue
		}
		return int((uint32(r)-item.Lo)/item.Stride) % 10, true
	}
	return 0, false
}

func extractGPQA(text string) (string, string) {
	if matches := gpqaAnswerLine.FindAllStringSubmatch(text, -1); len(matches) > 0 {
		return strings.ToUpper(matches[len(matches)-1][1]), "answer_line"
	}
	if matches := gpqaBoxed.FindAllStringSubmatch(text, -1); len(matches) > 0 {
		return strings.ToUpper(matches[len(matches)-1][1]), "boxed"
	}
	tail := reasoningTail(text)
	for matches := gpqaStandalone.FindAllStringIndex(tail, -1); len(matches) > 0; matches = matches[:len(matches)-1] {
		match := matches[len(matches)-1]
		if asciiLetterBefore(tail, match[0]) || asciiLetterAfter(tail, match[1]) {
			continue
		}
		return strings.ToUpper(tail[match[0]:match[1]]), "tail_letter"
	}
	return "", "none"
}

func reasoningTail(text string) string {
	if idx := strings.LastIndex(text, "</think>"); idx >= 0 {
		return text[idx+len("</think>"):]
	}
	runes := []rune(text)
	if len(runes) > 400 {
		return string(runes[len(runes)-400:])
	}
	return text
}

func pythonWordOrDotBefore(text string, at int) bool {
	if at == 0 {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(text[:at])
	return r == '.' || r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

func pythonWordOrDotAfter(text string, at int) bool {
	if at == len(text) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(text[at:])
	return r == '.' || r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

func asciiLetterBefore(text string, at int) bool {
	return at > 0 && isASCIIAlpha(text[at-1])
}

func asciiLetterAfter(text string, at int) bool {
	return at < len(text) && isASCIIAlpha(text[at])
}

func isASCIIAlpha(b byte) bool {
	return b >= 'A' && b <= 'Z' || b >= 'a' && b <= 'z'
}
