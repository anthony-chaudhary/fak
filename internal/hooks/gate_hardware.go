package hooks

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	hardwareDGXWordRE = regexp.MustCompile(`\bDGX\b`)
	hardwareSXM4RE    = regexp.MustCompile(`\bSXM4\b`)
	hardwareDGXNRE    = regexp.MustCompile(`(?i)\bdgx[0-9]+`)
)

// ScanMessageHardwareTells ports the hard-tell subset of
// tools/scrub_hardware_names.py --audit-message for the commit-msg hook. It
// scans raw kept message text: comment lines and the git scissors preview are
// excluded, but code spans are not masked because a private node label in a
// commit subject/body is still a leak even inside backticks.
func ScanMessageHardwareTells(msg string) []Finding {
	var findings []Finding
	for i, line := range strings.Split(msg, "\n") {
		if strings.HasPrefix(line, "# ------------------------ >8") {
			break
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if hardwareLineHasTell(line) {
			findings = append(findings, Finding{
				Gate:   "HARDWARE_TELL",
				Line:   i + 1,
				Detail: preview(line),
			})
		}
	}
	return findings
}

func hardwareLineHasTell(line string) bool {
	if hardwareDGXWordRE.MatchString(line) || hardwareSXM4RE.MatchString(line) {
		return true
	}
	for _, loc := range hardwareDGXNRE.FindAllStringIndex(line, -1) {
		if hardwareDGXNBoundaryOK(line[loc[1]:]) {
			return true
		}
	}
	return false
}

func hardwareDGXNBoundaryOK(rest string) bool {
	if rest == "" {
		return true
	}
	r, sz := utf8.DecodeRuneInString(rest)
	if r == utf8.RuneError && sz == 0 {
		return true
	}
	if r == '-' || r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
		return false
	}
	if r == '.' {
		next := rest[sz:]
		if next == "" {
			return true
		}
		nr, _ := utf8.DecodeRuneInString(next)
		return !(unicode.IsLetter(nr) || unicode.IsDigit(nr))
	}
	return true
}
