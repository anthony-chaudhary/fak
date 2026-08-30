package hooks

import (
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/egresslist"
)

// gate_publicleak.go — the PUBLIC_LEAK gate, a byte-faithful port of
// tools/scrub_public_copy.py's --audit-staged path (_scan_added_lines + _effective_audit_needles).
// It substring-matches added lines (case-insensitive) against a redact-needle list and two
// case-sensitive regexes (live Slack token, GCP service-account email), skipping self-referential
// files. The effective needle list = the base AUDIT_NEEDLES unioned with an optional gitignored
// sidecar JSON (so the operator's private identity tier can extend it without committing it).

// auditNeedles is the byte-faithful base list from scrub_public_copy.py's AUDIT_NEEDLES. The
// repeated private address entries are kept exactly as in the source (de-duped at match time
// anyway). The Windows user-path entries de-escape to the same runtime strings as the Python
// literals.
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
	"ca" + "ma",
	"sar" + "onic",
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
	{regexp.MustCompile(`(?i)\b(?:lab[-_ ])?dgx[0-9]+\b`), "private GPU host alias (dgxN)"},
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
	// scanned is this gate's candidate denominator (#5602): the staged files it actually read,
	// counted here rather than re-derived, so the number can never disagree with the set judged.
	scanned := 0
	for _, f := range d.sortedFiles() {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfReferentialLeak[norm] {
			continue
		}
		scanned++
		findings = append(findings, publicLeakLineFindings(norm, 0, norm, needles)...)
		pinned := isPinnedUpstreamArtifact(d, norm)
		for _, al := range d.AddedByFile[f] {
			if pinned && egresslist.IsUpstreamRuleLine(al.Text) {
				continue
			}
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
	d.NoteCandidates("PUBLIC_LEAK", scanned, "staged file(s) scanned")
	return findings, nil
}

// isPinnedUpstreamArtifact reports whether the STAGED bytes of `rel` are the checksum-pinned
// rendering that the manifest beside them records for that list (#5405). It is the one
// condition under which gatePublicLeak will skip an added line, and it delegates the whole
// definition to internal/egresslist — the leaf that OWNS the artifact grammar — rather than
// re-deriving it here, so the exemption cannot drift away from what the generator emits.
//
// WHY AN EXEMPTION IS SAFE HERE AND NOWHERE ELSE. A community ad/telemetry filter feed names
// vendor brands by construction: naming a host is how you block it. That collides head-on with
// an affiliation needle, and every other resolution is worse — dropping the rules loses them on
// the next refresh, and rewriting them at export makes the published fak block DIFFERENT hosts
// than the private one, which is silent divergence in a security artifact.
//
// The load-bearing condition is the PIN, not the path. A path allowlist would exempt anything
// dropped in the directory; a checksum means a hand-smuggled line changes the hash, so a leak
// cannot ride in under the exemption without re-pinning the manifest in the same reviewable
// diff. IsUpstreamRuleLine then narrows it to the single line SHAPE the generator emits for feed
// data, so a needle in the artifact's `!` prose header — the one place a human can write a claim
// into a generated file — is still a leak.
//
// Both file reads go through d.FileBytes, which resolves the STAGED blob: the bytes that would
// actually land, not whatever is loose on disk. Every failure mode (not a declared artifact path,
// missing artifact, missing manifest) returns false and the line is scanned as usual — fail-closed
// is the only correct direction, since a false negative costs one FLEET_ALLOW_LEAK override while
// a false positive silently publishes a leak.
func isPinnedUpstreamArtifact(d *StagedDiff, rel string) bool {
	if _, ok := egresslist.ArtifactName(rel); !ok {
		return false
	}
	artifact, ok := d.FileBytes(rel)
	if !ok {
		return false
	}
	// Resolved BESIDE the artifact, so this works unchanged whether the tree is rooted at the
	// repo or nested under the export layout's `fak/` prefix — the same tolerance ArtifactName has.
	manifest, ok := d.FileBytes(path.Join(path.Dir(rel), egresslist.ManifestFile))
	if !ok {
		return false
	}
	return egresslist.IsPinnedArtifact(rel, artifact, manifest)
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
	// Mirror scrub_public_copy.py's message scanner so the Go gate and the Python checker
	// cannot drift: eachCommitMessageLine already stops at git's scissors line and skips
	// comment lines; skip identity trailers (DCO sign-off / co-author) here too — a needle
	// in a `Signed-off-by: name <user@org>` is identity metadata, not a leak, so scanning
	// it would refuse every signed commit.
	eachCommitMessageLine(msg, func(i int, line string) {
		if messageTrailerRe.MatchString(line) {
			return
		}
		findings = append(findings, publicLeakLineFindings(line, i, "", needles)...)
	})
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
