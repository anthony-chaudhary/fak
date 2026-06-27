package hooks

import (
	"regexp"
	"strings"
)

// gate_secretshape.go — the SECRET_SHAPE gate, a port of tools/check_secret_shapes.py. It
// catches operator-leak SHAPES the literal needle list (PUBLIC_LEAK) misses: a Windows/macOS
// home path with a real username, an msl-* host, a *.lab host. Scans added lines of text files.

var (
	// check_secret_shapes.py L33-36, verbatim. Go's regexp is RE2; these are plain enough to
	// translate directly. (?i) replaces re.IGNORECASE on the two host patterns.
	opPathRE  = regexp.MustCompile(`[A-Za-z]:[\\/][Uu]sers[\\/]([A-Za-z][A-Za-z0-9._-]{1,})`)
	macPathRE = regexp.MustCompile(`/Users/([A-Za-z][A-Za-z0-9._-]{1,})`)
	mslHostRE = regexp.MustCompile(`(?i)\bmsl-[a-z0-9][a-z0-9-]*`)
	labHostRE = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9.-]*\.lab\b`)
)

// placeholderUsers — a home-path username in this set is a documentation placeholder, not a
// leak (check_secret_shapes.py L29-32), compared lower-cased.
var placeholderUsers = map[string]bool{
	"user": true, "you": true, "runner": true, "public": true, "default": true,
	"all": true, "username": true, "administrator": true, "guest": true,
	"youruser": true, "name": true, "someone": true,
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

// textExt — only these extensions are scanned (check_secret_shapes.py L45-46).
var textExt = map[string]bool{
	".md": true, ".txt": true, ".go": true, ".py": true, ".json": true, ".jsonl": true,
	".sh": true, ".ps1": true, ".yml": true, ".yaml": true, ".toml": true, ".cff": true, ".html": true,
}

func gateSecretShape(d *StagedDiff) ([]Finding, error) {
	seen := map[string]bool{} // dedupe on (file, hit) like the Python report
	var findings []Finding
	for _, f := range d.sortedFiles() {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfRefShape[norm] {
			continue
		}
		if !textExt[lowerExt(norm)] {
			continue
		}
		for _, al := range d.AddedByFile[f] {
			for _, hit := range scanShapes(al.Text) {
				key := f + "\x00" + hit.text
				if seen[key] {
					continue
				}
				seen[key] = true
				findings = append(findings, Finding{
					Gate: "SECRET_SHAPE", File: f, Line: al.New,
					Detail: "[" + hit.shape + "] " + hit.text,
				})
			}
		}
	}
	return findings, nil
}

type shapeHit struct{ shape, text string }

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
	return hits
}

// operatorPathHits runs a home-path regex (capture group 1 = username) and emits an
// operator-path shapeHit for every match whose username is not a documentation placeholder.
func operatorPathHits(re *regexp.Regexp, line string) []shapeHit {
	var hits []shapeHit
	for _, m := range re.FindAllStringSubmatch(line, -1) {
		if !placeholderUsers[strings.ToLower(m[1])] {
			hits = append(hits, shapeHit{"operator-path", m[0]})
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
