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
	hardwareDA33RE    = regexp.MustCompile(`(?i)\bda33`)
	hardwareFenceRE   = regexp.MustCompile(`^\s*(` + "```" + `|~~~)`)

	hardwareMaskREs = []*regexp.Regexp{
		regexp.MustCompile("`[^`]*`"),
		regexp.MustCompile(`\]\([^)]*\)`),
		regexp.MustCompile(`https?://[^\s\x00]+`),
		regexp.MustCompile(`[\w./\\-]+\.(?:md|json|go|py|sh|txt|png|svg|jpg|ya?ml|toml|csv|html)\b`),
	}
)

func gateHardwareTell(d *StagedDiff) ([]Finding, error) {
	selfRef := map[string]bool{
		"tools/scrub_hardware_names.py":      true,
		"tools/scrub_hardware_names_test.py": true,
		"tools/check_hardware_tells.py":      true,
		"tools/check_hardware_tells_test.py": true,
		"PUBLIC-SCRUB-POLICY.md":             true,
	}
	var findings []Finding
	for _, path := range d.StagedPaths {
		path = strings.ReplaceAll(path, "\\", "/")
		if !strings.HasSuffix(path, ".md") || selfRef[path] {
			continue
		}
		added := map[string]bool{}
		for _, line := range d.AddedByFile[path] {
			added[strings.TrimSpace(line.Text)] = true
		}
		if len(added) == 0 {
			continue
		}
		if d.run == nil {
			continue
		}
		blob, code, err := d.run(d.ctx, d.Root, "show", ":"+path)
		if err != nil || code != 0 {
			continue
		}
		for _, hit := range residualHardwareDocHits(blob) {
			if added[strings.TrimSpace(hit.Detail)] {
				findings = append(findings, Finding{
					Gate:   "HARDWARE_TELL",
					File:   path,
					Line:   hit.Line,
					Detail: preview(hit.Detail),
				})
			}
		}
	}
	return findings, nil
}

func residualHardwareDocHits(content string) []Finding {
	var findings []Finding
	inFence := false
	for i, line := range strings.Split(content, "\n") {
		line = strings.TrimSuffix(line, "\r")
		if hardwareFenceRE.MatchString(line) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		if hardwareDocLineHasTell(line) {
			findings = append(findings, Finding{Gate: "HARDWARE_TELL", Line: i + 1, Detail: line})
		}
	}
	return findings
}

func hardwareDocLineHasTell(line string) bool {
	return hardwareLineHasTell(maskHardwareDocLine(line))
}

func maskHardwareDocLine(line string) string {
	line = maskBracketedPathishLabels(line)
	for _, rx := range hardwareMaskREs {
		line = rx.ReplaceAllString(line, "\x00")
	}
	return line
}

func maskBracketedPathishLabels(line string) string {
	var b strings.Builder
	for i := 0; i < len(line); {
		if line[i] != '[' {
			b.WriteByte(line[i])
			i++
			continue
		}
		end := i + 1
		pathish := false
		valid := false
		for ; end < len(line); end++ {
			c := line[end]
			if c == ']' {
				valid = true
				break
			}
			if c == 0 || c == ' ' || c == '\t' || c == '\r' || c == '\n' {
				break
			}
			if c == '-' || c == '/' || c == '_' || c == '.' {
				pathish = true
			}
		}
		if valid && pathish {
			b.WriteByte(0)
			i = end + 1
			continue
		}
		b.WriteByte(line[i])
		i++
	}
	return b.String()
}

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
	for _, loc := range hardwareDA33RE.FindAllStringIndex(line, -1) {
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
