package egressfloor

import (
	"regexp"
	"strings"
)

// ResultQuarantine is the structural verdict for untrusted tool output. It is
// deliberately independent of model interpretation: matched control-like text
// is preserved as evidence but wrapped so it cannot be mistaken for trusted
// instructions by the next turn.
type ResultQuarantine struct {
	Quarantined bool
	Reason      string
	Output      string
}

var resultInjectionPatterns = []struct {
	reason string
	re     *regexp.Regexp
}{
	{"instruction_override", regexp.MustCompile(`(?i)\b(ignore|disregard|forget)\b.{0,48}\b(previous|prior|above|system|developer)\b.{0,24}\b(instruction|prompt|message)s?\b`)},
	{"role_impersonation", regexp.MustCompile(`(?im)^\s*(system|developer|assistant)\s*:`)},
	{"prompt_delimiter", regexp.MustCompile(`(?i)<\|?(system|developer|assistant)(?:_message)?\|?>|\[/?INST\]`)},
	{"secret_exfiltration", regexp.MustCompile(`(?i)\b(reveal|print|send|exfiltrat(?:e|ion))\b.{0,48}\b(secret|token|credential|api[_ -]?key|system prompt)\b`)},
	{"tool_coercion", regexp.MustCompile(`(?i)\b(run|execute|call|invoke)\b.{0,32}\b(tool|shell|command|powershell|bash)\b`)},
}

// QuarantineResult structurally screens untrusted tool output at the result
// seam. Benign output is returned byte-for-byte; suspicious output is enclosed
// in an explicit untrusted-data boundary and carries a stable reason.
func QuarantineResult(output string) ResultQuarantine {
	for _, pattern := range resultInjectionPatterns {
		if pattern.re.MatchString(output) {
			return ResultQuarantine{
				Quarantined: true,
				Reason:      pattern.reason,
				Output: "<fak-untrusted-tool-output quarantined=\"true\" reason=\"" + pattern.reason + "\">\n" +
					strings.ReplaceAll(output, "</fak-untrusted-tool-output>", "&lt;/fak-untrusted-tool-output&gt;") +
					"\n</fak-untrusted-tool-output>",
			}
		}
	}
	return ResultQuarantine{Output: output}
}
