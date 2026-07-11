package wiki

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// citationRE matches a markdown code citation of the DeepWiki shape
// `[<path>:<start>[-<end>]](<target>)` — e.g. `[internal/gateway/gateway.go:120-140]()`.
// The target (group 4) is ignored: DeepWiki emits an empty `()`, and whether it is
// empty or a URL, the load-bearing operands are the path and the line span.
var citationRE = regexp.MustCompile(`\[([^\[\]\n]+?):(\d+)(?:-(\d+))?\]\(([^)\n]*)\)`)

// pathExtRE recognises a trailing file extension (".go", ".tsx", ".py"). Together
// with a "/" test it guards against reading a non-path bracket like `[Step:2]()`
// as a code citation — a citation resolver must not manufacture danglers out of
// ordinary prose that happens to use the `[word:number]` shape.
var pathExtRE = regexp.MustCompile(`\.[A-Za-z0-9]+$`)

// DanglerReason is the closed set of ways a code citation fails to resolve.
type DanglerReason string

const (
	// ReasonMissingFile: the cited path does not exist under the tree (or names a
	// directory, not a file).
	ReasonMissingFile DanglerReason = "missing-file"
	// ReasonLineOutOfRange: the file exists but the cited line span falls outside
	// [1, lineCount] (or end < start).
	ReasonLineOutOfRange DanglerReason = "line-out-of-range"
)

// Dangler is one code citation that failed to resolve against the working tree —
// the anti-hallucination finding L3 exists to surface. It carries enough to point
// a reader (or a CI gate) at the exact offending cite.
type Dangler struct {
	Path    string        `json:"path"`
	Start   int           `json:"start"`
	End     int           `json:"end,omitempty"` // 0 when the cite named a single line
	Reason  DanglerReason `json:"reason"`
	Lines   int           `json:"lines,omitempty"` // the file's actual line count (range failures)
	Line    int           `json:"line"`            // 1-based line in the markdown where the cite appears
	Raw     string        `json:"raw"`             // the citation text as written
	Section string        `json:"section,omitempty"`
}

// VerifyCitations parses every `[path:line]` code citation out of the markdown and
// returns the ones that fail to resolve against the tree rooted at root: a missing
// file or an out-of-bounds line span. A page with no code citations returns nil
// (that is not itself a failure — L7 measures citation DENSITY separately).
//
// It is deterministic and side-effect-free: it only reads the cited files to count
// their lines. Citations whose path does not look like a file (no "/" and no
// extension) are skipped rather than flagged, so ordinary `[word:2]` prose does not
// masquerade as a broken code cite.
func VerifyCitations(root string, markdown []byte) []Dangler {
	var out []Dangler
	lineCounts := map[string]int{} // memoized per cited path within this page
	section := ""
	for i, raw := range bytes.Split(markdown, []byte("\n")) {
		line := string(raw)
		if h := headingText(line); h != "" {
			section = h
		}
		for _, m := range citationRE.FindAllStringSubmatch(line, -1) {
			rawPath := strings.TrimSpace(m[1])
			if !looksLikePath(rawPath) {
				continue
			}
			start, _ := strconv.Atoi(m[2])
			end := 0
			if m[3] != "" {
				end, _ = strconv.Atoi(m[3])
			}
			d := Dangler{
				Path:    rawPath,
				Start:   start,
				End:     end,
				Line:    i + 1,
				Raw:     m[0],
				Section: section,
			}
			n, ok := resolveLines(root, rawPath, lineCounts)
			if !ok {
				d.Reason = ReasonMissingFile
				out = append(out, d)
				continue
			}
			if !inRange(start, end, n) {
				d.Reason = ReasonLineOutOfRange
				d.Lines = n
				out = append(out, d)
			}
		}
	}
	return out
}

// resolveLines returns the line count of the cited file under root, memoized in
// counts. ok is false when the path is missing or is a directory. A negative
// sentinel (-1) memoizes an unresolvable path so a page that cites it many times
// stats the tree once.
func resolveLines(root, relPath string, counts map[string]int) (n int, ok bool) {
	if c, seen := counts[relPath]; seen {
		if c < 0 {
			return 0, false
		}
		return c, true
	}
	full := filepath.Join(root, filepath.FromSlash(relPath))
	info, err := os.Stat(full)
	if err != nil || info.IsDir() {
		counts[relPath] = -1
		return 0, false
	}
	b, err := os.ReadFile(full)
	if err != nil {
		counts[relPath] = -1
		return 0, false
	}
	c := countLines(b)
	counts[relPath] = c
	return c, true
}

// inRange reports whether the cited [start,end] span sits within a file of n
// lines. A single-line cite (end==0) checks start alone. start must be >= 1.
func inRange(start, end, n int) bool {
	if start < 1 || start > n {
		return false
	}
	if end == 0 {
		return true
	}
	return end >= start && end <= n
}

// countLines counts the lines in b the way an editor numbers them: a trailing
// newline does not add a phantom empty final line, but a file with no trailing
// newline still counts its last line. Empty input is 0 lines.
func countLines(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	n := bytes.Count(b, []byte("\n"))
	if b[len(b)-1] != '\n' {
		n++ // last line has no terminating newline
	}
	return n
}

// looksLikePath guards the citation regex: a real code cite's path has a directory
// separator or a file extension. `[Overview:1]()` (no "/" , no ".ext") is prose.
func looksLikePath(s string) bool {
	return strings.Contains(s, "/") || pathExtRE.MatchString(s)
}

// headingText returns the text of a markdown ATX heading line ("## Foo" -> "Foo"),
// or "" for a non-heading. Used only to attribute a dangler to its section.
func headingText(line string) string {
	t := strings.TrimSpace(line)
	if !strings.HasPrefix(t, "#") {
		return ""
	}
	return strings.TrimSpace(strings.TrimLeft(t, "#"))
}
