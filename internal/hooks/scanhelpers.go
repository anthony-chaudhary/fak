package hooks

import "strings"

// scanhelpers.go — small shared helpers factored out of near-identical scan loops that
// recur across the gate files. Behavior-preserving only: each helper's body is the exact
// code it replaces, unchanged.

// eachCommitMessageLine walks the commit message lines a git hook actually keeps: it stops
// at git's scissors block (the "# ------------------------ >8 ..." cut line the content gate
// owns) and skips comment lines git strips from the final message, then calls fn with the
// 1-based line number and text for every remaining line. Shared by ScanMessageHardwareTells
// and ScanMessageNeedles, whose message-line preambles were byte-identical.
func eachCommitMessageLine(msg string, fn func(i int, line string)) {
	for i, line := range strings.Split(msg, "\n") {
		if strings.HasPrefix(line, "# ------------------------ >8") {
			break
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		fn(i+1, line)
	}
}

// publicLeakLineFindings scans one line for a PUBLIC_LEAK: a needle substring hit
// (case-insensitive) or an auditRegexes match. file is attached to each finding ("" when the
// scan has no file, e.g. a commit message or an outbound payload). Shared by
// ScanMessageNeedles, scanPublicLeakTree, and ScanOutboundText — the three PUBLIC_LEAK
// needle/regex line scanners, whose loop bodies were otherwise identical.
func publicLeakLineFindings(line string, lineNo int, file string, needles []string) []Finding {
	var findings []Finding
	ll := strings.ToLower(line)
	for _, n := range needles {
		if strings.Contains(ll, strings.ToLower(n)) {
			findings = append(findings, Finding{Gate: "PUBLIC_LEAK", File: file, Line: lineNo, Detail: "[" + n + "]  " + preview(line)})
		}
	}
	for _, rx := range auditRegexes {
		if auditRegexMatches(rx.re, rx.label, line) {
			findings = append(findings, Finding{Gate: "PUBLIC_LEAK", File: file, Line: lineNo, Detail: "[" + rx.label + "]  " + preview(line)})
		}
	}
	return findings
}
