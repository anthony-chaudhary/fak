package witness

import (
	"encoding/json"
	"regexp"
	"strings"
)

const symptomExcerptMax = 240

// SymptomFailureExcerpt is a bounded one-line excerpt of what failed in the last
// candidate run that refuted the witness ("" otherwise). It is kept out of the
// closed detail so detail consumers never receive test output.
func (r *Resolver) SymptomFailureExcerpt() string {
	return r.symptomExcerpt
}

// symptomFailureExcerpt names what failed in a `go test -json` run, so a refusal
// separates a red test from a build or environment failure: the first failing
// test with its first diagnostic line, else the first compiler or setup error.
func symptomFailureExcerpt(out string) string {
	firstFailed := ""
	testOutput := map[string][]string{}
	var loose []string
	for _, line := range strings.Split(out, "\n") {
		var event goTestJSONEvent
		if json.Unmarshal([]byte(line), &event) != nil {
			if t := strings.TrimSpace(line); t != "" {
				loose = append(loose, t)
			}
			continue
		}
		if event.Test == "" {
			if t := strings.TrimSpace(event.Output); t != "" {
				loose = append(loose, t)
			}
			continue
		}
		switch event.Action {
		case "output":
			if t := strings.TrimSpace(event.Output); t != "" && !symptomExcerptNoise(t) {
				testOutput[event.Test] = append(testOutput[event.Test], t)
			}
		case "fail":
			if firstFailed == "" {
				firstFailed = event.Test
			}
		}
	}
	if firstFailed != "" {
		excerpt := "FAIL " + firstFailed
		if lines := testOutput[firstFailed]; len(lines) > 0 {
			excerpt += ": " + lines[0]
		}
		return capSymptomExcerpt(excerpt)
	}
	for _, line := range loose {
		if goCompileErrorLine.MatchString(line) || strings.Contains(line, "[build failed]") || strings.Contains(line, "[setup failed]") || strings.HasPrefix(line, "panic:") {
			return capSymptomExcerpt(line)
		}
	}
	return ""
}

var goCompileErrorLine = regexp.MustCompile(`\.go:\d+(?::\d+)?: `)

func symptomExcerptNoise(line string) bool {
	for _, prefix := range []string{"=== RUN", "=== PAUSE", "=== CONT", "--- FAIL", "--- PASS", "--- SKIP", "FAIL", "PASS", "ok "} {
		if strings.HasPrefix(line, prefix) {
			return true
		}
	}
	return false
}

func capSymptomExcerpt(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > symptomExcerptMax {
		s = s[:symptomExcerptMax] + "..."
	}
	return s
}
