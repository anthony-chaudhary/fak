package livecodebench

import "strings"

// CodeExtraction is the result of pulling the gradeable code out of ONE raw model
// generation. NoCode is an explicit verdict — the output carried nothing to grade
// (empty, prose-only, or an empty fenced block) — kept distinct from an extracted
// empty string so the grader can score "no code" as a miss instead of crashing on
// it or silently treating it as a pass.
type CodeExtraction struct {
	// Code is the extracted program, ready to hand to the grader. Internal
	// whitespace and indentation are preserved verbatim (code is significant); it
	// is empty when NoCode is true.
	Code string
	// Language is the fenced language tag, lower-cased ("python", "cpp", ...), or
	// "" when the answer was fenced without a tag. Meaningless when NoCode is true.
	Language string
	// NoCode is the explicit no-gradeable-code verdict.
	NoCode bool
}

// ExtractCode pulls the final gradeable code block out of a raw model generation,
// mirroring the lcb_runner fence convention: a model reasons in prose and emits
// its answer as a fenced code block, so the LAST closed ``` fence wins over any
// earlier ones (a self-repair or worked-example block never shadows the final
// answer). Behaviour by shape:
//
//   - fenced with a language tag (```python\n...\n```): the tag is captured into
//     Language and stripped from Code.
//   - fenced without a tag (```\n...\n```): Code is the block, Language is "".
//   - multi-block: only the last closed fence pair is returned.
//   - starter-merge: when starter (the problem's starter_code) is non-empty and
//     its signature is absent from the extracted block, starter is prepended so
//     the grader sees the complete program; when the block already carries the
//     signature it is left untouched (no duplication).
//   - empty / prose-only / an empty fenced block: NoCode, so the grader records an
//     explicit miss rather than crashing or scoring garbage as a pass.
//
// Unfenced prose is deliberately NOT salvaged as code: under the LCB convention a
// model that does not fence its answer has not produced a gradeable one, and
// guessing would fabricate a pass. starter is the problem's starter_code ("" for a
// pure code-generation problem).
func ExtractCode(raw, starter string) CodeExtraction {
	lines := strings.Split(raw, "\n")
	var fences []int
	for i, ln := range lines {
		if strings.Contains(ln, "```") {
			fences = append(fences, i)
		}
	}
	// Need a closed pair: fewer than two fence markers is not a gradeable block.
	if len(fences) < 2 {
		return CodeExtraction{NoCode: true}
	}
	open, close := fences[len(fences)-2], fences[len(fences)-1]
	lang := fenceLanguage(lines[open])
	code := strings.Join(lines[open+1:close], "\n")
	if strings.TrimSpace(code) == "" {
		return CodeExtraction{NoCode: true, Language: lang}
	}
	return CodeExtraction{Code: mergeStarter(code, starter), Language: lang}
}

// fenceLanguage returns the lower-cased language tag on an opening fence line
// ("```python" -> "python"), or "" when the fence carries no single-token tag.
func fenceLanguage(fenceLine string) string {
	tag := strings.TrimPrefix(strings.TrimSpace(fenceLine), "```")
	fields := strings.Fields(tag)
	if len(fields) == 0 {
		return ""
	}
	return strings.ToLower(fields[0])
}

// mergeStarter prepends starter to code when the problem supplies starter_code the
// model completed but did not echo. The signature test is the starter's first
// non-empty line: if the extracted block already contains it the block is a full
// solution and is returned unchanged; otherwise the block is a bare completion and
// starter is prepended so the grader compiles a whole program.
func mergeStarter(code, starter string) string {
	starter = strings.TrimRight(starter, "\n")
	if strings.TrimSpace(starter) == "" {
		return code
	}
	if sig := firstNonEmptyLine(starter); sig == "" || strings.Contains(code, sig) {
		return code
	}
	return starter + "\n" + code
}

// firstNonEmptyLine returns the first line of s with non-whitespace content,
// trimmed, or "" when s is blank.
func firstNonEmptyLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(ln); t != "" {
			return t
		}
	}
	return ""
}
