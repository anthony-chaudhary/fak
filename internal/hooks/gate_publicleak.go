package hooks

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// gate_publicleak.go — the PUBLIC_LEAK gate, a byte-faithful port of
// tools/scrub_public_copy.py's --audit-staged path (_scan_added_lines + _effective_audit_needles).
// It substring-matches added lines (case-insensitive) against a redact-needle list and two
// case-sensitive regexes (live Slack token, GCP service-account email), skipping self-referential
// files. The effective needle list = the base AUDIT_NEEDLES unioned with an optional gitignored
// sidecar JSON (so the operator's private identity tier can extend it without committing it).

// auditNeedles is the verbatim base list from scrub_public_copy.py L304-326. The repeated private
// address entries are kept exactly as in the source (de-duped at match time anyway). The Windows
// user-path entries de-escape to the same runtime strings as the Python literals.
var auditNeedles = []string{
	privateAddressNeedle(),
	privateAddressNeedle(),
	privateAddressNeedle(),
	"/Users/" + "anth" + "ony",
	`Users\` + "antho",
	`Users\\` + "antho",
	"GitHub/" + "Benchmark",
	"Documents/" + "GitHub/" + "Benchmark",
	"node-" + "agent-" + "netra",
	"node-" + "windows-a",
	"node-" + "desktop-b",
	".claude-" + "agent",
	"sam" + "sung",
}

func privateAddressNeedle() string { return "100" + ".64.0.10" }

// auditRegexes are applied CASE-SENSITIVELY to the raw added line (scrub_public_copy.py L369-380).
var auditRegexes = []struct {
	re    *regexp.Regexp
	label string
}{
	{regexp.MustCompile(`xox[bp]-\d{8,}-\d{8,}-[A-Za-z0-9]{16,}`), "live Slack token (xoxb/xoxp)"},
	{regexp.MustCompile(`[a-z0-9](?:[a-z0-9-]*[a-z0-9])?@[a-z0-9-]+\.iam\.gserviceaccount\.com`), "GCP service-account email"},
}

// selfReferentialLeak — files exempt from the needle scan (scrub_public_copy.py L463-467), path
// normalized to forward slashes. gate_publicleak.go is added because it DEFINES the needle list
// as source (auditNeedles) — the exact analog of exempting tools/scrub_public_copy.py, which
// holds the Python AUDIT_NEEDLES. (The test files construct their needle fixtures at runtime, so
// they carry no literal needle and need no exemption.)
var selfReferentialLeak = map[string]bool{
	"PUBLIC-SCRUB-POLICY.md":            true,
	"tools/scrub_public_copy.py":        true,
	"tools/githooks/pre-commit":         true,
	"internal/hooks/gate_publicleak.go": true,
}

// privateNeedlesRel is the optional gitignored sidecar that extends the needle list at runtime
// (scrub_public_copy.py L392).
const privateNeedlesRel = "tools/_registry/scrub_needles.private.json"

func gatePublicLeak(d *StagedDiff) ([]Finding, error) {
	needles := effectiveAuditNeedles(d)
	var findings []Finding
	for _, f := range d.sortedFiles() {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfReferentialLeak[norm] {
			continue
		}
		for _, al := range d.AddedByFile[f] {
			payloadL := strings.ToLower(al.Text)
			for _, n := range needles {
				if strings.Contains(payloadL, strings.ToLower(n)) {
					findings = append(findings, Finding{
						Gate: "PUBLIC_LEAK", File: f, Line: al.New,
						Detail: "[" + n + "]  " + preview(al.Text),
					})
				}
			}
			for _, rx := range auditRegexes {
				if rx.re.MatchString(al.Text) {
					findings = append(findings, Finding{
						Gate: "PUBLIC_LEAK", File: f, Line: al.New,
						Detail: "[" + rx.label + "]  " + preview(al.Text),
					})
				}
			}
		}
	}
	return findings, nil
}

// effectiveAuditNeedles unions the base list with the sidecar JSON's audit_needles +
// export_audit_needles (scrub_public_copy.py _effective_audit_needles L521-556): base list
// first, then any new extras in encounter order, de-duped. A missing/malformed sidecar yields
// the base list byte-identically.
func effectiveAuditNeedles(d *StagedDiff) []string {
	out := append([]string(nil), auditNeedles...)
	b, ok := d.FileBytes(privateNeedlesRel)
	if !ok {
		return out
	}
	var priv struct {
		AuditNeedles       []string `json:"audit_needles"`
		ExportAuditNeedles []string `json:"export_audit_needles"`
	}
	if err := json.Unmarshal(b, &priv); err != nil {
		return out
	}
	seen := map[string]bool{}
	for _, n := range auditNeedles {
		seen[n] = true
	}
	extras := append(append([]string(nil), priv.AuditNeedles...), priv.ExportAuditNeedles...)
	for _, n := range extras {
		if n != "" && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}

// messageTrailerRe matches the git identity trailers (DCO sign-off, co-author, …) that
// scrub_public_copy.py exempts from the commit-message PUBLIC_LEAK scan (its trailer_re):
// a `Signed-off-by: name <user@org>` legitimately carries an identity-tier needle (the org
// domain) and is a structured trailer, not prose — flagging it would refuse every signed
// commit. Kept in lockstep with the Python list so the gate and the checker cannot drift.
var messageTrailerRe = regexp.MustCompile(
	`(?i)^(Signed-off-by|Co-authored-by|Acked-by|Reviewed-by|Reported-by|` +
		`Suggested-by|Tested-by|Cc|Helped-by|Reported-and-tested-by):\s`)

// needlesWithSidecar returns the base audit needles unioned with the optional gitignored
// sidecar under root — the runtime twin of effectiveAuditNeedles for callers that have no
// StagedDiff (commit messages, outbound payloads). root == "" skips the sidecar (the base
// needles still apply); a missing/malformed sidecar yields the base list byte-identically.
func needlesWithSidecar(root string) []string {
	needles := append([]string(nil), auditNeedles...)
	if root == "" {
		return needles
	}
	b, err := readFileRel(root, privateNeedlesRel)
	if err != nil {
		return needles
	}
	var priv struct {
		AuditNeedles       []string `json:"audit_needles"`
		ExportAuditNeedles []string `json:"export_audit_needles"`
	}
	if json.Unmarshal(b, &priv) != nil {
		return needles
	}
	seen := map[string]bool{}
	for _, n := range needles {
		seen[n] = true
	}
	for _, n := range append(priv.AuditNeedles, priv.ExportAuditNeedles...) {
		if n != "" && !seen[n] {
			seen[n] = true
			needles = append(needles, n)
		}
	}
	return needles
}

// ScanMessageNeedles ports scrub_public_copy.py --audit-message: the SAME needle/regex scan over
// the lines of a commit message (the commit-msg hook's PUBLIC_LEAK gate). A message line carries
// no file, so File is "" and Line is the 1-based message line number. Like the Python twin it
// skips git's scissors block, comment lines, and identity trailers (see messageTrailerRe).
func ScanMessageNeedles(msg string, root string) []Finding {
	needles := needlesWithSidecar(root)
	var findings []Finding
	for i, line := range strings.Split(msg, "\n") {
		// Mirror scrub_public_copy.py's message scanner so the Go gate and the Python
		// checker cannot drift: stop at git's scissors line (the to-be-stripped diff
		// preview the content gate owns), skip comment lines git strips from the final
		// message, and skip identity trailers (DCO sign-off / co-author) — a needle in a
		// `Signed-off-by: name <user@org>` is identity metadata, not a leak, so scanning
		// it would refuse every signed commit.
		if strings.HasPrefix(line, "# ------------------------ >8") {
			break
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if messageTrailerRe.MatchString(line) {
			continue
		}
		ll := strings.ToLower(line)
		for _, n := range needles {
			if strings.Contains(ll, strings.ToLower(n)) {
				findings = append(findings, Finding{Gate: "PUBLIC_LEAK", Line: i + 1, Detail: "[" + n + "]  " + preview(line)})
			}
		}
		for _, rx := range auditRegexes {
			if rx.re.MatchString(line) {
				findings = append(findings, Finding{Gate: "PUBLIC_LEAK", Line: i + 1, Detail: "[" + rx.label + "]  " + preview(line)})
			}
		}
	}
	return findings
}

func readFileRel(root, rel string) ([]byte, error) {
	return os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
}

// preview trims an added line to the 80-char window the Python report used.
func preview(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 80 {
		return s[:80]
	}
	return s
}
