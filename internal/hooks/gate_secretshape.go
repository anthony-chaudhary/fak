package hooks

import (
	"path"
	"regexp"
	"strings"
)

// catches operator-leak SHAPES the literal needle list (PUBLIC_LEAK) misses: a Windows/macOS
// home path with a real username, an msl-* host, a *.lab host. Scans added lines of text files.

var (
	// check_secret_shapes.py L33-36, verbatim. Go's regexp is RE2; these are plain enough to
	// translate directly. (?i) replaces re.IGNORECASE on the two host patterns.
	opPathRE = regexp.MustCompile(`(?:^|[^A-Za-z0-9._-])([A-Za-z]:[\\/][Uu]sers[\\/]([A-Za-z][A-Za-z0-9._-]{1,}))`)
	// macOS home path: case-insensitive on `Users` and anchored on a path-segment boundary,
	// matching the Python MAC_PATH. The boundary is a capture group, not a lookbehind, because
	// RE2 has none -- group 1 is the path, group 2 the username. See the Python comment for why
	// both properties are load-bearing (a lowercase `/users/<operator>` literal in an exempt
	// _test.go fixture was published while this gate reported the tree clean).
	macPathRE = regexp.MustCompile(`(?:^|[^A-Za-z0-9._-])(/[Uu][Ss][Ee][Rr][Ss]/([A-Za-z][A-Za-z0-9._-]{1,}))`)
	mslHostRE = regexp.MustCompile(`(?i)\bmsl-[a-z0-9][a-z0-9-]*`)
	// The first label must be at least 3 characters, matching the Python LAB_HOST. A 1-2
	// character DNS label is not a hostname convention, and it is exactly the shape a Go
	// field selector takes: the tree carries `tt.lab` (a struct field in internal/devindex),
	// which a 1+ label rule reports as an internal host on every audit.
	labHostRE = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9.-]{2,}\.lab\b`)
	// Site/cluster CODENAME family, matching the Python CODENAME_HOST. The prefix is assembled
	// from fragments so this file carries no contiguous codename literal. Hardware CLASSES
	// (`h200`, `a100`) are NOT matched -- only the site codename.
	codenamePrefix = "rt" + "p"
	codenameRE     = regexp.MustCompile(`(?i)\b` + codenamePrefix + `-[a-z0-9][a-z0-9-]*`)
)

// isPublicLLMRepo reports whether a codename match is the PUBLIC `<prefix>-llm` third-party
// repo cited in docs/research. The Python rule wants a `(?!llm\b)` lookahead here, but RE2 has
// no lookahead at all -- `regexp.MustCompile` on `(?!` PANICS at package init (Go's parser
// accepts only `i m s U - : )` after `(?`). Since the two implementations are parity-tested on
// the same grammar, the guard lives in this shared post-match predicate instead.
//
// The predicate is deliberately ASCII-only, where `\b` would be Unicode-aware in Python and
// ASCII in RE2. A non-ASCII byte after `llm` therefore counts as a WORD character, so the
// guard FAILS SAFE: an ambiguous `rtp-llm<non-ascii>` is flagged, not exempted.
func isPublicLLMRepo(line string, m []int) bool {
	rest := line[m[0]+len(codenamePrefix)+1:]
	if len(rest) < 3 || !strings.EqualFold(rest[:3], "llm") {
		return false
	}
	if len(rest) == 3 {
		return true
	}
	c := rest[3]
	return c < 0x80 && !(c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z')
}

// placeholderUsers — a home-path username in this set is a documentation placeholder, not a
// leak (check_secret_shapes.py PLACEHOLDER_USERS), compared lower-cased. The second group is
// the generic FIXTURE-name class: the names a test or a doc uses when it needs *some* home
// path and does not mean anyone. Without them the whole-tree audit reports ~100 findings on a
// clean tree, which is what makes a hard tree gate unusable. Keep in lockstep with the Python.
var placeholderUsers = map[string]bool{
	"user": true, "you": true, "runner": true, "public": true, "default": true,
	"all": true, "username": true, "administrator": true, "guest": true,
	"youruser": true, "name": true, "someone": true,
	"dev": true, "devuser": true, "test": true, "example": true, "private": true,
	"operator": true, "alice": true, "foo": true, "bench": true, "me": true,
}

// selfRefShape — files skipped entirely (check_secret_shapes.py L39-44), path forward-slashed.
// The two shape-defining gate sources are added (the analog of the Python exempting itself); the
// test files construct their shape fixtures at runtime, so they carry no literal shape.
var selfRefShape = map[string]bool{
	"PUBLIC-SCRUB-POLICY.md":             true,
	"tools/scrub_public_copy.py":         true,
	"tools/check_secret_shapes.py":       true,
	"tools/operator_path_needle_test.py": true,
	"internal/hooks/gate_secretshape.go": true,
	"internal/hooks/gate_publicleak.go":  true,
}

// generatedCorpus — machine-generated third-party corpora (docs/research/inventory/README.md:
// "generated from pinned local checkouts with `fak study-inventory`"). These are verbatim
// scrapes of UPSTREAM projects, so a home path inside one is a THIRD-PARTY developer's, not
// the operator's -- the shape this gate exists to catch. Enumerated as a glob list, not a
// pattern, so widening it is a reviewed edit (the same bar as adding a needle). The
// literal-needle leak gate still scans these files; only the SHAPE heuristic stands down.
var generatedCorpus = []string{"docs/research/inventory/*.jsonl"}

// isGeneratedCorpus reports whether a forward-slashed repo-relative path is a generated
// third-party corpus. Mirrors check_secret_shapes.py _is_generated_corpus.
func isGeneratedCorpus(norm string) bool {
	for _, pat := range generatedCorpus {
		if ok, err := path.Match(pat, norm); err == nil && ok {
			return true
		}
	}
	return false
}

// textExt — only these extensions are scanned (check_secret_shapes.py L45-46).
var textExt = map[string]bool{
	".md": true, ".txt": true, ".go": true, ".py": true, ".json": true, ".jsonl": true,
	".sh": true, ".ps1": true, ".yml": true, ".yaml": true, ".toml": true, ".cff": true, ".html": true,
}

func gateSecretShape(d *StagedDiff) ([]Finding, error) {
	seen := map[string]bool{} // dedupe on (file, hit) like the Python report
	var findings []Finding
	// This gate's own filter is narrower than "every staged file": it judges TEXT files only.
	// Counting after that filter is what makes "scanned forty" distinguishable from "scanned
	// none" for a security gate whose clean payload is otherwise identical either way (#5602).
	scanned := 0
	for _, f := range d.sortedFiles() {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfRefShape[norm] || isGeneratedCorpus(norm) {
			continue
		}
		if !textExt[lowerExt(norm)] {
			continue
		}
		scanned++
		for _, al := range d.AddedByFile[f] {
			findings = append(findings, shapeHitFindings(seen, f, al.New, al.Text)...)
		}
	}
	d.NoteCandidates("SECRET_SHAPE", scanned, "staged text file(s) scanned")
	return findings, nil
}

type shapeHit struct{ shape, text string }

// shapeHitFindings ports the SECRET_SHAPE dedup-and-emit inner loop shared by the staged gate
// (per added line, with its new-file line number) and its tree twin (whole-file body, Line 0):
// scan text for leak shapes, dedupe each hit on (file, hit-text) via the caller's seen map, and
// emit a SECRET_SHAPE finding for every first sighting. The line number is the only per-mode input.
func shapeHitFindings(seen map[string]bool, file string, line int, text string) []Finding {
	var findings []Finding
	for _, hit := range scanShapes(text) {
		key := file + "\x00" + hit.text
		if seen[key] {
			continue
		}
		seen[key] = true
		findings = append(findings, Finding{
			Gate: "SECRET_SHAPE", File: file, Line: line,
			Detail: "[" + hit.shape + "] " + hit.text,
		})
	}
	return findings
}

// scanShapes ports _scan_text (check_secret_shapes.py L58-71).
func scanShapes(line string) []shapeHit {
	var hits []shapeHit
	// Both home-path families (Windows C:\Users\X and macOS /Users/X) share the same per-match
	// decision: flag the captured username unless it is a documentation placeholder.
	hits = append(hits, operatorPathHits(opPathRE, line)...)
	hits = append(hits, operatorPathHits(macPathRE, line)...)
	for _, m := range mslHostRE.FindAllString(line, -1) {
		hits = append(hits, shapeHit{"internal-host", m})
	}
	for _, m := range labHostRE.FindAllString(line, -1) {
		if !strings.HasSuffix(strings.ToLower(m), "example.lab") {
			hits = append(hits, shapeHit{"internal-host", m})
		}
	}
	for _, m := range codenameRE.FindAllStringIndex(line, -1) {
		if !isPublicLLMRepo(line, m) {
			hits = append(hits, shapeHit{"site-codename", line[m[0]:m[1]]})
		}
	}
	return hits
}

// operatorPathHits runs a home-path regex and emits an operator-path shapeHit for every match
// whose username is not a documentation placeholder. Both home-path forms share one group
// structure: group 1 is the path (reported), group 2 the username (judged). The leading
// path-segment boundary is a capture group rather than a lookbehind because RE2 has none.
func operatorPathHits(re *regexp.Regexp, line string) []shapeHit {
	var hits []shapeHit
	for _, m := range re.FindAllStringSubmatch(line, -1) {
		if !placeholderUsers[strings.ToLower(m[2])] {
			hits = append(hits, shapeHit{"operator-path", m[1]})
		}
	}
	return hits
}

func lowerExt(path string) string {
	i := strings.LastIndex(path, ".")
	if i < 0 {
		return ""
	}
	return strings.ToLower(path[i:])
}
