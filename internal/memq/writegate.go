package memq

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// Write-time artifact-claim probe (#2431). The notes backend re-verifies a note's
// concrete claims at READ time (Materialize / Verify); ProbeAtWrite is the WRITE
// arm — run before a memory note lands so a self-reported artifact (a commit SHA,
// a repo path, a CLI flag the session narrated) is probed against the live
// checkout AT THE DOOR, and the verdict is stamped into the note's frontmatter as
// trusted structure the memoryread grammar carries — never prose.
//
// A failing claim does NOT block the write: the observation may still be useful,
// so the note lands, but stamped unverified-at-write with the failing claim named.
// A note whose every checkable claim verifies carries verified-at-write with the
// probe witness. Recall then renders an unverified-at-write note hedged by default
// (HedgeReason) — even when a later live re-check is inconclusive, the doubt
// recorded at the door survives to the table. This closes the loop the read gate
// opened: garbage is labeled at the door and re-checked at the table (#2077).
//
// A note with no checkable claim gets no write verdict — nothing was probed, so
// nothing is stamped; the read path already hedges a claim-free note (empty
// findings), exactly as before.
const (
	// WriteVerifyKey is the top-level frontmatter field the probe stamps. The read
	// side keys its hedge on this structure, not on the note's prose.
	WriteVerifyKey = "write_verify"
	// WriteVerifyDetailKey names the failing claim (unverified) or the probed
	// claims (verified) so a reader can see what was checked and why.
	WriteVerifyDetailKey = "write_verify_detail"

	// WriteVerified stamps a note whose every checkable claim verified at write.
	WriteVerified = "verified-at-write"
	// WriteUnverified stamps a note with at least one claim that did not verify.
	WriteUnverified = "unverified-at-write"
)

// ProbeAtWrite runs the artifact verifier over a memory note's concrete claims
// BEFORE it lands and returns the note with a write_verify frontmatter stamp plus
// the per-claim findings. A nil verifier uses recall.DefaultArtifactVerifier (the
// same default the read gate uses). When the note names no checkable artifact the
// input is returned unstamped and findings is empty — there was nothing to prove.
func ProbeAtWrite(ctx context.Context, raw []byte, verifier recall.ArtifactVerifier) ([]byte, []recall.ArtifactFinding) {
	if verifier == nil {
		verifier = recall.DefaultArtifactVerifier
	}
	body := memoryread.StripFrontmatter(string(raw))
	claims := recall.ExtractArtifactClaims(body)
	if len(claims) == 0 {
		return append([]byte(nil), raw...), nil
	}
	findings := verifier(ctx, claims)
	status, detail := writeVerdict(findings)
	return []byte(stampWriteVerify(string(raw), status, detail)), findings
}

// writeVerdict folds the per-claim findings into a single write-time verdict: any
// non-fresh finding makes the note unverified-at-write and the detail names the
// failing claims (sorted, deterministic); all-fresh makes it verified-at-write and
// the detail witnesses the probed claims.
func writeVerdict(findings []recall.ArtifactFinding) (status, detail string) {
	var bad, ok []string
	for _, f := range findings {
		label := fmt.Sprintf("%s %q", f.Claim.Kind, f.Claim.Value)
		if f.Status == recall.ArtifactFresh {
			ok = append(ok, label)
			continue
		}
		full := label + " " + string(f.Status)
		if d := strings.TrimSpace(f.Detail); d != "" {
			full += ": " + d
		}
		bad = append(bad, full)
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return WriteUnverified, strings.Join(bad, "; ")
	}
	sort.Strings(ok)
	return WriteVerified, strings.Join(ok, "; ")
}

var (
	writeVerifyStatusRE = regexp.MustCompile(`(?m)^write_verify:\s*(\S+)`)
	writeVerifyDetailRE = regexp.MustCompile(`(?m)^write_verify_detail:\s*(.+?)\s*$`)
	writeVerifyLineRE   = regexp.MustCompile(`(?m)^(?:write_verify|write_verify_detail):.*\n?`)
)

// stampWriteVerify writes the verdict into the note's frontmatter, replacing any
// prior write_verify stamp so re-probing is idempotent. A note without a
// frontmatter block gets a fresh one; the body is never touched.
func stampWriteVerify(raw, status, detail string) string {
	if status == "" {
		return raw
	}
	stamp := fmt.Sprintf("%s: %s\n%s: %s", WriteVerifyKey, status, WriteVerifyDetailKey, yamlInline(detail))
	if !strings.HasPrefix(raw, "---") {
		return "---\n" + stamp + "\n---\n\n" + raw
	}
	end := strings.Index(raw[3:], "\n---")
	if end == -1 { // malformed block — prepend a fresh one, leave the rest intact
		return "---\n" + stamp + "\n---\n\n" + raw
	}
	header := strings.TrimRight(writeVerifyLineRE.ReplaceAllString(raw[:3+end], ""), "\n")
	rest := raw[3+end:] // begins with "\n---"
	return header + "\n" + stamp + rest
}

// yamlInline renders a detail string as a double-quoted single-line YAML scalar so
// a colon or quote in a probe detail cannot break the frontmatter grammar.
func yamlInline(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", " ")
	return "\"" + s + "\""
}

// writeVerifyStamp reads the write-time verdict + detail from a note's frontmatter,
// returning empty strings when no stamp is present (lexical and tolerant, like
// parseNoteMeta).
func writeVerifyStamp(raw string) (status, detail string) {
	if !strings.HasPrefix(raw, "---") {
		return "", ""
	}
	end := strings.Index(raw[3:], "\n---")
	if end == -1 {
		return "", ""
	}
	front := raw[:end+3]
	if m := writeVerifyStatusRE.FindStringSubmatch(front); m != nil {
		status = m[1]
	}
	if m := writeVerifyDetailRE.FindStringSubmatch(front); m != nil {
		detail = strings.Trim(strings.TrimSpace(m[1]), "\"")
	}
	return status, detail
}

// HedgeReason reports why a note must render hedged (reduced trust), or "" when it
// renders plainly. A note stamped unverified-at-write at the door hedges
// regardless of a later live re-check — the recorded doubt survives even when the
// live probe is inconclusive. Otherwise a note with no checkable claim hedges
// (nothing was ever proved), and a note whose live claims no longer all verify
// hedges with the offending claim named. A verified-at-write note whose claims
// still verify renders plainly.
//
// The write-time stamp is read back from the note's source file (its frontmatter
// is stripped off the in-memory cell at scan) so this gate composes with the
// read-only backend without re-scanning the store.
func (b *NotesBackend) HedgeReason(ctx context.Context, id string) (string, error) {
	cell, body, err := b.lookup(id)
	if err != nil {
		return "", err
	}
	if status, detail := b.writeStampFor(cell); status == WriteUnverified {
		reason := "unverified at write"
		if detail != "" {
			reason += ": " + detail
		}
		return reason, nil
	}
	findings := b.verifyFindings(ctx, cell, body)
	if len(findings) == 0 {
		return "no checkable artifact claim", nil
	}
	for _, f := range findings {
		if f.Status != recall.ArtifactFresh {
			reason := fmt.Sprintf("live claim %s %q %s", f.Claim.Kind, f.Claim.Value, f.Status)
			if d := strings.TrimSpace(f.Detail); d != "" {
				reason += ": " + d
			}
			return reason, nil
		}
	}
	return "", nil
}

// writeStampFor reads the write-time verdict + detail from a scanned note's source
// file. It reuses the source_path the backend already records at scan; a note with
// no recorded path or an unreadable file yields no stamp (empty strings), never an
// error — the read gate degrades to its live-verify behavior.
func (b *NotesBackend) writeStampFor(cell Cell) (status, detail string) {
	path := cell.Attrs["source_path"]
	if path == "" {
		return "", ""
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", ""
	}
	return writeVerifyStamp(string(raw))
}
