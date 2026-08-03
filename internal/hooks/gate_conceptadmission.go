package hooks

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var conceptIdentRE = regexp.MustCompile(`[A-Za-z][A-Za-z0-9_]*`)

// conceptSimpleEscapes are the one-letter Go escapes, plus the three that quote a
// delimiter. Each stands for a single character of the compiled string, so its letter
// belongs to the escape and never to the identifier that follows it.
const conceptSimpleEscapes = `abfnrtv\'"`

// conceptEscapeLen reports the length of the Go escape sequence at the head of s, or 0
// when the backslash starts no valid escape and is therefore an ordinary character.
func conceptEscapeLen(s string) int {
	hex := func(i, n int) int {
		if len(s) < i+n {
			return 0
		}
		for _, c := range []byte(s[i : i+n]) {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
				return 0
			}
		}
		return i + n
	}
	if len(s) < 2 || s[0] != '\\' {
		return 0
	}
	switch c := s[1]; {
	case strings.IndexByte(conceptSimpleEscapes, c) >= 0:
		return 2
	case c >= '0' && c <= '7':
		if n := hex(1, 3); n > 0 && s[2] <= '7' && s[3] <= '7' {
			return n
		}
	case c == 'x':
		return hex(2, 2)
	case c == 'u':
		return hex(2, 4)
	case c == 'U':
		return hex(2, 8)
	}
	return 0
}

// conceptScanLine blanks every Go escape sequence inside an interpreted string or rune
// literal so the identifier after an escape is tokenized on its own, and reports alongside
// the text which bytes sit in CODE position -- outside every literal and every comment.
//
// Reading the raw source text instead welds the escape's letter onto the next word: an
// ordinary header literal joining the columns DRIFT, AGE_H, SESSION and LEAVES with tab
// escapes yielded a token spelling a stray leading "t" in front of the column name, a symbol
// that exists nowhere in the tree, and the gate then blocked the commit until someone
// classified it (#5529). The artifact is spelled out only in the regression test, never
// here: a comment quoting it verbatim would re-mint it, since a backslash in a comment is a
// literal character rather than an escape. That is also why raw strings and comments are
// left exactly as the extractor has always read them -- eating a would-be escape there would
// swallow the first letter of a real word (Users -> sers).
//
// The code mask is what separates a declaration from a mention. Only a byte in code position
// can begin a declaration, so the two reference shapes below are recognised in prose and in
// literals, where nothing is ever declared, and left untouched in code (#5533).
func conceptScanLine(s string) (string, []bool) {
	const (
		code = iota
		interpreted
		runeLit
		rawLit
		block
	)
	b, state := []byte(s), code
	isCode := make([]bool, len(b))
	for i := 0; i < len(b); i++ {
		switch state {
		case code:
			isCode[i] = true
			switch {
			case b[i] == '"':
				state = interpreted
			case b[i] == '\'':
				state = runeLit
			case b[i] == '`':
				state = rawLit
			case b[i] == '/' && i+1 < len(b) && b[i+1] == '/':
				return string(b), isCode // the rest of the line is prose
			case b[i] == '/' && i+1 < len(b) && b[i+1] == '*':
				state, i = block, i+1
			}
		case rawLit:
			if b[i] == '`' {
				state = code
			}
		case block:
			if b[i] == '*' && i+1 < len(b) && b[i+1] == '/' {
				state, i = code, i+1
			}
		case interpreted, runeLit:
			if b[i] == '"' && state == interpreted || b[i] == '\'' && state == runeLit {
				state = code
				continue
			}
			n := conceptEscapeLen(s[i:])
			if n == 0 {
				continue
			}
			for j := i; j < i+n; j++ {
				b[j] = ' '
			}
			i += n - 1
		}
	}
	return string(b), isCode
}

// conceptPathSuffixes are the extensions a repo path ends in. A run of identifier characters
// immediately followed by a dot and one of these is the stem of a PATH: the tree holds a file
// by that name, never a declaration by it. Reading such a stem as an introduced identifier
// blocked commits on names declared nowhere in the tree, and the only ways past were to
// position a concept that does not exist or to classify a non-symbol as incidental -- both of
// which write a permanent corpus row and inflate the discovered-token denominator of the
// figure the scorecard reports (#5533).
var conceptPathSuffixes = map[string]bool{
	"go": true, "json": true, "md": true, "ps1": true, "py": true,
	"sh": true, "toml": true, "txt": true, "yaml": true, "yml": true,
}

// conceptTestNameRE matches the spelling Go REQUIRES of a test entry point. Such a function
// is only ever run from a file the scan below refuses to admit in the first place, so a name
// of this shape written in prose or inside a literal is a reference into a scope this gate
// does not own -- not a symbol the quoting file introduces. Naming the test that witnesses
// the code above it is good practice and used to cost a permanent corpus row (#5533). In code
// position the name is left alone: a non-test file may legally declare a function so named.
var conceptTestNameRE = regexp.MustCompile(`^(?:Test|Benchmark|Fuzz|Example)[A-Z]`)

// conceptFileSuffixAt reports the extension beginning at i when text[i] is the dot of a path,
// and "" when it is not one. Two deliberate narrowings keep a method call from reading as a
// path: the run after the dot is taken whole, so a selector spelled x.gopher is not misread as
// an extension plus a tail; and the match is CASE-SENSITIVE, because a repo path spells its
// extension in lower case while an exported method is capitalised. Folding case would have
// read `x.JSON()` -- and, worse, the errgroup idiom `g.Go(...)` -- as a file name and dropped
// the receiver, which in code position is a symbol like any other.
func conceptFileSuffixAt(text string, i int) string {
	if i >= len(text) || text[i] != '.' {
		return ""
	}
	j := i + 1
	for j < len(text) {
		c := text[j]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
			break
		}
		j++
	}
	if ext := text[i+1 : j]; conceptPathSuffixes[ext] {
		return ext
	}
	return ""
}

type admissionMeta struct {
	Families []struct {
		ID      string   `json:"id"`
		Roots   []string `json:"roots"`
		Ignore  []string `json:"ignore"`
		Exclude []string `json:"exclude"`
	} `json:"families"`
}
type admissionRows struct {
	Rows []struct{ ID, Family, Grounding string } `json:"rows"`
}

func admissionToken(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// admissionLeaf derives the leaf name from a gated source path
// (internal/<leaf>/... or cmd/<leaf>/...) so the remedy can name the
// per-leaf rows-<leaf>.json shard.
func admissionLeaf(p string) string {
	if segs := strings.SplitN(p, "/", 3); len(segs) >= 3 {
		return segs[1]
	}
	return "leaf"
}

func gateConceptAdmission(d *StagedDiff) ([]Finding, error) {
	metaBytes, ok := d.FileBytes("tools/concept_disambiguation_scorecard.data/_meta.json")
	if !ok {
		return nil, nil
	}
	var meta admissionMeta
	if json.Unmarshal(metaBytes, &meta) != nil {
		return nil, nil
	} // semantic gate owns malformed data
	positioned := map[string]bool{}
	// Read every row through the index-aware accessor, including a newly staged
	// row file that does not exist in HEAD or a peer-dirty workspace.
	paths := append([]string{}, d.IndexPaths...)
	if len(paths) == 0 {
		matches, _ := filepath.Glob(filepath.Join(d.Root, "tools", "concept_disambiguation_scorecard.data", "rows-*.json"))
		for _, abs := range matches {
			rel, _ := filepath.Rel(d.Root, abs)
			paths = append(paths, filepath.ToSlash(rel))
		}
	}
	for _, rel := range paths {
		if !strings.HasPrefix(rel, "tools/concept_disambiguation_scorecard.data/rows-") || !strings.HasSuffix(rel, ".json") {
			continue
		}
		b, exists := d.FileBytes(rel)
		if !exists {
			continue
		}
		var doc admissionRows
		if json.Unmarshal(b, &doc) != nil {
			continue
		}
		for _, r := range doc.Rows {
			positioned[admissionToken(r.Family)+"\x00"+admissionToken(r.Grounding)] = true
		}
	}
	type family struct {
		id                string
		roots, classified map[string]bool
	}
	var fams []family
	for _, f := range meta.Families {
		x := family{id: f.ID, roots: map[string]bool{}, classified: map[string]bool{}}
		for _, r := range f.Roots {
			x.roots[admissionToken(r)] = true
		}
		for _, v := range append(append([]string{}, f.Ignore...), f.Exclude...) {
			x.classified[admissionToken(v)] = true
		}
		fams = append(fams, x)
	}
	seen := map[string]bool{}
	var out []Finding
	for path, lines := range d.AddedByFile {
		p := filepath.ToSlash(path)
		if !(strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/")) || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			continue
		}
		for _, line := range lines {
			text, isCode := conceptScanLine(line.Text)
			for _, m := range conceptIdentRE.FindAllStringIndex(text, -1) {
				raw := text[m[0]:m[1]]
				inCode := m[0] < len(isCode) && isCode[m[0]]
				// A path stem is not a symbol. In code position only ".go" can be trusted to
				// be an extension: `go` is a reserved word, so `x.go` can never be a package
				// or field selector, while `x.json` can be -- and suppressing that receiver
				// would drop a real symbol.
				if ext := conceptFileSuffixAt(text, m[1]); ext != "" && (!inCode || ext == "go") {
					continue
				}
				if !inCode && conceptTestNameRE.MatchString(raw) {
					continue
				}
				tok := admissionToken(raw)
				if tok == "" {
					continue
				}
				for _, f := range fams {
					matchesRoot := false
					for root := range f.roots {
						if strings.Contains(tok, root) {
							matchesRoot = true
							break
						}
					}
					key := admissionToken(f.id) + "\x00" + tok
					if !matchesRoot || seen[key] || positioned[key] || f.classified[tok] {
						continue
					}
					// A rename/move or another use of an established token is not a new
					// corpus admission: only tokens absent from committed HEAD qualify.
					baseTree := "HEAD"
					if strings.HasSuffix(d.Treeish, ":") && d.Treeish != ":" {
						tip := strings.TrimSuffix(d.Treeish, ":")
						if out, code, _ := d.run(d.ctx, d.Root, "rev-parse", tip+"^"); code == 0 && strings.TrimSpace(out) != "" {
							baseTree = strings.TrimSpace(out)
						}
					}
					grep, code, _ := d.run(d.ctx, d.Root, "grep", "-I", "-w", "-e", raw, baseTree, "--", "internal", "cmd")
					if code == 0 && strings.TrimSpace(grep) != "" {
						continue
					}
					seen[key] = true
					// Suggest the per-leaf rows shard (--row-file) rather than the shared
					// glossary/scorecard files: on a shared trunk the bare command plans
					// writes into peer-contended files, while a new rows-<leaf>.json is
					// lane-local and clears this gate on its own (#5104).
					leaf := admissionLeaf(p)
					cmd := fmt.Sprintf("fak concept position --id %s --canonical %q --family %s --definition TEXT --distinction TEXT --kind symbol --grounding %s --grounding-kind symbol --row-file rows-%s.json --distinct-from SIBLING_ID", tok, raw, f.id, raw, leaf)
					// Both remedy commands write the corpus in the WORKTREE, and this gate reads
					// the INDEX (the accessor above resolves the staged blob). `fak commit --path`
					// stages only the paths it is handed, so running the command and re-committing
					// the same code paths reports this finding again, byte for byte -- the cure
					// looks applied and nothing changes. Name the corpus paths the commit has to
					// carry, in full, so the printed remedy is sufficient on its own (#5534).
					stage := fmt.Sprintf("--path tools/concept_disambiguation_scorecard.data/_meta.json --path tools/concept_disambiguation_scorecard.data/rows-%s.json", leaf)
					detail := fmt.Sprintf("family=%s token=%s introduced at %s:%d is neither positioned nor classified; run `%s` (or `fak concept classify --family %s --token %s --category incidental --reason TEXT`), then add `%s` to the SAME `fak commit` -- this gate reads the git index, not the worktree, so a corpus write left unstaged reports this finding unchanged; landing the corpus as its own commit instead trips CONCEPT_FRESHNESS, whose generated markdown has to be regenerated from a clean-room export of the commit rather than from the dirty worktree", f.id, raw, p, line.New, cmd, f.id, raw, stage)
					out = append(out, Finding{Gate: "CONCEPT_ADMISSION", File: p, Line: line.New, Detail: detail})
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File == out[j].File {
			return out[i].Line < out[j].Line
		}
		return out[i].File < out[j].File
	})
	return out, nil
}

// CheckConceptAdmission exposes the shared staged/range admission decision.
func CheckConceptAdmission(d *StagedDiff) ([]Finding, error) { return gateConceptAdmission(d) }
