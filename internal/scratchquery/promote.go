// promote.go defines the CLOSED promotion-candidate taxonomy (deliverable A of
// issue #13069) and the deterministic matcher that flags parked scratchpad
// artifacts looking like unfiled SESSION FLAGS before the reap window closes.
//
// The problem this closes (fak#2344 reap vs fak#2345 strand): the scratchpad
// janitor's only liveness signal is "a resumable session points at it". It has
// no notion of "the content is worth keeping", so it can reap an unreferenced
// session whose sole copy of a session flag was parked in its scratchpad. This
// query is the promotion-first read that lets an agent promote such content to
// an issue BEFORE the reap window closes. It is read-only and advisory: it never
// files an issue and never deletes anything.
//
// TAXONOMY (closed; the same names the private run-end fold keys on ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â see
// docs/SESSION-FLAGS-TO-ISSUES.md in the companion repo):
//
//   - unfinished-spine       the smallest end-to-end path could not ship.
//   - blocked                a real blocker in another lane / missing authority.
//   - discovered-edge-case   an edge case, failure path, or hazard found while working.
//   - deferred-caveat        a known limitation the work disclosed (not yet / simulated).
//   - next-checkable-step    the concrete forward step the run ended on.
//
// A candidate is classified by an EXPLICIT marker first (a leading comment block
// or filename carrying one of the marker phrases below), so the query is
// deterministic and closed: it does not guess intent from arbitrary prose. An
// artifact can opt out of the surface with the `promote:keep` directive.
package scratchquery

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// CandidateKind is the closed promotion-candidate taxonomy. A value outside this
// set is never produced by Classify.
type CandidateKind string

const (
	// KindUnfinishedSpine: the smallest working end-to-end path could not ship.
	KindUnfinishedSpine CandidateKind = "unfinished-spine"
	// KindBlocked: a real blocker in another lane, a peer's tree, or missing authority.
	KindBlocked CandidateKind = "blocked"
	// KindDiscoveredEdgeCase: an edge case, failure path, or hazard found while working.
	KindDiscoveredEdgeCase CandidateKind = "discovered-edge-case"
	// KindDeferredCaveat: a known limitation disclosed (not yet / simulated / degraded).
	KindDeferredCaveat CandidateKind = "deferred-caveat"
	// KindNextCheckableStep: the concrete forward step the run ended on.
	KindNextCheckableStep CandidateKind = "next-checkable-step"
)

// CandidateKinds is the closed set in taxonomy order. It is the authority a
// caller (or a test) iterates to prove the vocabulary is closed.
var CandidateKinds = []CandidateKind{
	KindUnfinishedSpine,
	KindBlocked,
	KindDiscoveredEdgeCase,
	KindDeferredCaveat,
	KindNextCheckableStep,
}

// KeepDirective suppresses promotion of an artifact whose leading content (or
// filename) carries it ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the artifact author's explicit "do not promote me".
const KeepDirective = "promote:keep"

// promoteHeaderBytes bounds how much of an artifact's head Classify reads, the
// same fail-closed discipline scratchmark applies to a declarative header.
const promoteHeaderBytes = 16 * 1024

// markerRules is the CLOSED marker table: each rule names one kind and the
// case-insensitive phrases that declare it. A match on any phrase classifies the
// artifact as that kind. Rules are evaluated in CandidateKinds order, so an
// artifact that declares several resolves to the earliest taxonomy entry (a
// deterministic, documented tie-break).
var markerRules = []struct {
	kind    CandidateKind
	markers []string
}{
	{KindUnfinishedSpine, []string{"unfinished-spine", "unfinished spine", "spine could not ship", "spine did not ship", "end-to-end path could not"}},
	{KindBlocked, []string{"blocked:", "blocked by", "blocker:", "cannot proceed", "waiting on", "quarantine"}},
	{KindDiscoveredEdgeCase, []string{"discovered-edge-case", "discovered edge case", "edge case", "failure path", "concurrency hazard", "soak requirement", "found while working"}},
	{KindDeferredCaveat, []string{"deferred-caveat", "deferred caveat", "deferred:", "not yet", "sw-verified", "simulated only", "degraded path", "disclosed fallback"}},
	{KindNextCheckableStep, []string{"next-checkable-step", "next checkable step", "next step:", "next steps", "follow-up", "followup", "left to do", "still remaining", "todo:"}},
}

// Classify resolves one scratchpad artifact's leading content (and filename) to
// at most one CandidateKind. It returns ("", false) when the artifact is not a
// promotion candidate: no marker matched, the file is empty/unreadable, it is
// binary, or it carries the KeepDirective. Filename markers are honored too, so
// `edge-case-*.md` is caught even before its body is read.
func Classify(src Artifact) (CandidateKind, bool) {
	if src.Kept {
		return "", false
	}
	// Normalize separators so a hyphen/underscore filename channel
	// (`edge-case-*.md`) matches the same phrase as prose (`edge case`). Markers
	// themselves are stored space-separated, so ONE normalization covers both.
	haystack := normalizeMarkers(src.Name + "\n" + src.Header)
	if strings.Contains(haystack, KeepDirective) {
		return "", false
	}
	for _, rule := range markerRules {
		for _, marker := range rule.markers {
			if strings.Contains(haystack, normalizeMarkers(marker)) {
				return rule.kind, true
			}
		}
	}
	return "", false
}

// normalizeMarkers lowercases text and folds hyphen/underscore separators to
// spaces so a filename marker and a prose marker resolve through one table. A
// marker that needs a colon keeps it (`edge case:` is distinct from a bare
// mention); this only removes the spelling drift between `-` and ` `.
func normalizeMarkers(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, "-", " ")
	s = strings.ReplaceAll(s, "_", " ")
	return s
}

// Artifact is the bounded, content-only view of one parked scratchpad file that
// Classify reasons over. It is built by ScanPromotable; callers never hand-build
// one from unbounded input.
type Artifact struct {
	// Path is scratchpad-relative (slash-separated), the stable identity.
	Path string
	// Name is the file's base name, a filename-level marker channel.
	Name string
	// Header is at most promoteHeaderBytes of the file's head, lowercased by
	// Classify itself (stored raw here).
	Header string
	// Bytes is the file's full size (for the candidate receipt, never the read).
	Bytes int
	// Kept reports the file is unreadable/binary and therefore not classifiable.
	Kept bool
}

// Promotable is one promotion candidate: the artifact that looks like a session
// flag, the closed kind it matched, and the session context it was found under.
type Promotable struct {
	Kind        CandidateKind `json:"kind"`
	Session     string        `json:"session"`
	Scratchpad  string        `json:"scratchpad"`
	Path        string        `json:"file"`
	Bytes       int           `json:"bytes"`
	Diagnosis   any           `json:"diagnosis,omitempty"`
	DiagnosisID string        `json:"session_id,omitempty"`
}

// PromotableResult is the deterministic answer to one promotion scan.
type PromotableResult struct {
	Root       string       `json:"root"`
	Candidates []Promotable `json:"candidates"`
	Scanned    int          `json:"scanned"`
	Skipped    int          `json:"skipped"`
}

// ScanPromotable walks one session's scratchpad directory and returns every
// parked artifact that Classify flags as a promotion candidate. Read-only: it
// never files, mutates, or deletes. A missing/empty scratchpad is a clean empty
// result. Results are sorted by (kind, file) so two runs agree byte-for-byte.
func ScanPromotable(scratchpad, session string) (PromotableResult, error) {
	result := PromotableResult{Root: scratchpad, Candidates: []Promotable{}}
	if strings.TrimSpace(scratchpad) == "" {
		return result, fmt.Errorf("scratchpad path is required")
	}
	info, err := os.Stat(scratchpad)
	if err != nil {
		if os.IsNotExist(err) {
			return result, nil
		}
		return result, fmt.Errorf("stat scratchpad %q: %w", scratchpad, err)
	}
	if !info.IsDir() {
		return PromotableResult{Root: scratchpad, Candidates: []Promotable{}}, nil
	}

	files, err := scratchFiles(scratchpad)
	if err != nil {
		return result, err
	}
	for _, rel := range files {
		full := filepath.Join(scratchpad, filepath.FromSlash(rel))
		fi, err := os.Stat(full)
		if err != nil {
			continue
		}
		if fi.Size() > MaxFileBytes {
			result.Skipped++
			continue
		}
		header, err := readHeader(full)
		if err != nil {
			result.Skipped++
			continue
		}
		if looksBinary(header) {
			result.Skipped++ // a binary artifact never carries a declarative flag marker
			continue
		}
		result.Scanned++
		artifact := Artifact{
			Path:   rel,
			Name:   filepath.Base(rel),
			Header: string(header),
			Bytes:  int(fi.Size()),
		}
		kind, ok := Classify(artifact)
		if !ok {
			continue
		}
		result.Candidates = append(result.Candidates, Promotable{
			Kind:       kind,
			Session:    session,
			Scratchpad: scratchpad,
			Path:       rel,
			Bytes:      artifact.Bytes,
		})
	}
	sortPromotable(result.Candidates)
	return result, nil
}

// readHeader reads at most promoteHeaderBytes of a file's head. An unreadable
// file is an error (the caller counts it skipped); a binary file is reported by
// Classify as non-flag because the NUL byte keeps no marker from matching.
func readHeader(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(io.LimitReader(f, promoteHeaderBytes))
}

// sortPromotable orders candidates by (kind, session, file), the deterministic
// order two runs over the same tree agree on.
func sortPromotable(candidates []Promotable) {
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Kind != candidates[j].Kind {
			return candidates[i].Kind < candidates[j].Kind
		}
		if candidates[i].Session != candidates[j].Session {
			return candidates[i].Session < candidates[j].Session
		}
		return candidates[i].Path < candidates[j].Path
	})
}
