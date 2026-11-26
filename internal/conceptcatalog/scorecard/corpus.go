package scorecard

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

var skipDir = map[string]bool{
	".git": true, ".cache": true, "node_modules": true, "testdata": true, "_registry": true,
	"__pycache__": true, ".pytest_cache": true, ".ruff_cache": true, "vendor": true,
	".dispatch-runs": true, ".goal-runs": true,
}

var rootDocs = []string{"CLAIMS.md", "ARCHITECTURE.md", "README.md", "AGENTS.md", "INDEX.md", "GLOSSARY.md"}

const maxCorpusBytes = 512 * 1024

// Path globbing is case-insensitive on Windows, as pathlib's is.
func hasSuffixGlob(name, suffix string) bool {
	if runtime.GOOS == "windows" {
		return len(name) >= len(suffix) && strings.EqualFold(name[len(name)-len(suffix):], suffix)
	}
	return strings.HasSuffix(name, suffix)
}

func pathHasSkipPart(p string) bool {
	p = p[len(filepath.VolumeName(p)):]
	for _, part := range strings.FieldsFunc(p, func(r rune) bool { return r == '/' || r == filepath.Separator }) {
		if skipDir[part] {
			return true
		}
	}
	return false
}

// walkTree visits every entry beneath dir without following directory links,
// reporting each entry's path and whether it is (or links to) a directory.
func walkTree(dir string, visit func(path string, isDir bool)) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		p := filepath.Join(dir, e.Name())
		switch {
		case e.IsDir():
			visit(p, true)
			walkTree(p, visit)
		case e.Type()&fs.ModeSymlink != 0:
			st, err := os.Stat(p)
			visit(p, err == nil && st.IsDir())
		default:
			visit(p, false)
		}
	}
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// identTokens calls emit with the normalized token of every [A-Za-z][A-Za-z0-9_]* run.
func identTokens(b []byte, emit func(string)) {
	i := 0
	for i < len(b) {
		c := b[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
			i++
			continue
		}
		j := i + 1
		for j < len(b) && (isAlnumASCII(b[j]) || b[j] == '_') {
			j++
		}
		emit(normIdent(b[i:j]))
		i = j
	}
}

func normIdent(b []byte) string {
	out := make([]byte, 0, len(b))
	for _, c := range b {
		switch {
		case c >= 'A' && c <= 'Z':
			out = append(out, c+'a'-'A')
		case c == '_':
		default:
			out = append(out, c)
		}
	}
	return string(out)
}

// headingIdents emits the identifier tokens of every Markdown heading line
// (`^#{1,4}\s+(.*)$` per str.splitlines line).
func headingIdents(text []byte, emit func(string)) {
	for _, line := range pySplitLines(string(text)) {
		k := 0
		for k < len(line) && line[k] == '#' {
			k++
		}
		if k < 1 || k > 4 || k >= len(line) {
			continue
		}
		r, _ := utf8.DecodeRuneInString(line[k:])
		if !pyIsSpace(r) {
			continue
		}
		identTokens([]byte(line[k:]), emit)
	}
}

func readCorpusFile(p string) ([]byte, bool) {
	st, err := os.Stat(p)
	if err != nil || st.IsDir() || st.Size() > maxCorpusBytes {
		return nil, false
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, false
	}
	return b, true
}

// LoadTree collects the production-corpus facts the KPIs verify rows against:
// identifier presence across non-test Go files under internal/ and cmd/,
// directory and doc-heading tokens, the verbatim Go source for wire keys, and
// the CLI reference's verb tokens.
func LoadTree(root string) *Tree {
	root = filepath.Clean(root)
	structural := map[string]bool{}
	var goFiles []string
	rootSkipped := pathHasSkipPart(root)
	for _, base := range []string{"internal", "cmd"} {
		bdir := filepath.Join(root, base)
		if !isDir(bdir) {
			continue
		}
		walkTree(bdir, func(p string, dir bool) {
			if rootSkipped || pathHasSkipPart(p[len(root):]) {
				return
			}
			name := filepath.Base(p)
			if dir {
				structural[normToken(name)] = true
				return
			}
			if hasSuffixGlob(name, ".go") && !strings.HasSuffix(name, "_test.go") {
				goFiles = append(goFiles, p)
			}
		})
	}
	sort.Strings(goFiles)

	type fileTokens struct {
		raw    []byte
		tokens map[string]bool
	}
	results := make([]fileTokens, len(goFiles))
	var wg sync.WaitGroup
	next := make(chan int)
	workers := max(1, runtime.NumCPU())
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				raw, ok := readCorpusFile(goFiles[i])
				if !ok {
					continue
				}
				seen := map[string]bool{}
				identTokens(raw, func(t string) { seen[t] = true })
				results[i] = fileTokens{raw: raw, tokens: seen}
			}
		}()
	}
	for i := range goFiles {
		next <- i
	}
	close(next)
	wg.Wait()

	symFiles := map[string]int{}
	var blob bytes.Buffer
	first := true
	for _, ft := range results {
		if ft.tokens == nil {
			continue
		}
		for t := range ft.tokens {
			symFiles[t]++
		}
		if !first {
			blob.WriteByte(0)
		}
		first = false
		blob.Write(ft.raw)
	}

	var docFiles []string
	if ddir := filepath.Join(root, "docs"); isDir(ddir) {
		walkTree(ddir, func(p string, dir bool) {
			if dir || rootSkipped || pathHasSkipPart(p[len(root):]) {
				return
			}
			if hasSuffixGlob(filepath.Base(p), ".md") {
				docFiles = append(docFiles, p)
			}
		})
	}
	for _, rd := range rootDocs {
		p := filepath.Join(root, rd)
		if _, err := os.Stat(p); err == nil {
			docFiles = append(docFiles, p)
		}
	}
	for _, dp := range docFiles {
		text, ok := readCorpusFile(dp)
		if !ok {
			continue
		}
		headingIdents(text, func(t string) { structural[t] = true })
	}

	docVerbs := map[string]bool{}
	if cr, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(CLIRefRel))); err == nil {
		i := 0
		for i < len(cr) {
			if !(isAlnumASCII(cr[i]) || cr[i] == '-') {
				i++
				continue
			}
			j := i
			for j < len(cr) && (isAlnumASCII(cr[j]) || cr[j] == '-') {
				j++
			}
			docVerbs[normToken(string(cr[i:j]))] = true
			i = j
		}
	}

	goBlob := blob.Bytes()
	return &Tree{
		SymFiles:   symFiles,
		Structural: structural,
		InTree: func(tok string) bool {
			if tok == "" {
				return false
			}
			if _, ok := symFiles[tok]; ok {
				return true
			}
			return structural[tok]
		},
		WirePresent: func(key string) bool {
			return key != "" && !strings.Contains(key, "\x00") && bytes.Contains(goBlob, []byte(key))
		},
		Exists: func(p string) bool {
			if p == "" {
				return false
			}
			_, err := os.Stat(joinWorkspace(root, p))
			return err == nil
		},
		DocVerbs: docVerbs,
	}
}

// joinWorkspace is pathlib's root / p: an absolute p replaces root, and on
// Windows a rooted p keeps root's drive.
func joinWorkspace(root, p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	if runtime.GOOS == "windows" && (strings.HasPrefix(p, "/") || strings.HasPrefix(p, `\`)) {
		return filepath.VolumeName(root) + filepath.FromSlash(p)
	}
	return filepath.Join(root, filepath.FromSlash(p))
}

// LoadData merges the modular data directory: _meta.json contributes meta and
// families, every other *.json contributes its rows (tagged with _source_file).
func LoadData(dir string) (*Data, string) {
	if !isDir(dir) {
		return nil, "missing data directory: " + dir
	}
	metaV, msg := readJSONFile(filepath.Join(dir, "_meta.json"))
	if msg != "" {
		return nil, msg
	}
	metaDoc, ok := metaV.(*Object)
	if !ok {
		return nil, "_meta.json is not a JSON object"
	}
	out := &Data{Meta: orEmptyObject(get(metaDoc, "meta")), Families: orEmptyList(get(metaDoc, "families")), Rows: []*Object{}}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Sprintf("cannot list %s: %v", filepath.Base(dir), err)
	}
	var names []string
	for _, e := range entries {
		if hasSuffixGlob(e.Name(), ".json") {
			names = append(names, e.Name())
		}
	}
	sortPathNames(names)
	for _, name := range names {
		if strings.HasPrefix(name, "_") {
			continue
		}
		doc, msg := readJSONFile(filepath.Join(dir, name))
		if msg != "" {
			return nil, msg
		}
		docObj, _ := doc.(*Object)
		for _, r := range iterValue(orEmptyList(get(docObj, "rows"))) {
			if o, ok := r.(*Object); ok {
				if _, has := o.Get("_source_file"); !has {
					o.Set("_source_file", name)
				}
				out.Rows = append(out.Rows, o)
			}
		}
	}
	return out, ""
}

// sortPathNames orders sibling file names the way sorted(Path...) does:
// case-insensitively on Windows.
func sortPathNames(names []string) {
	if runtime.GOOS == "windows" {
		sort.SliceStable(names, func(i, j int) bool { return strings.ToLower(names[i]) < strings.ToLower(names[j]) })
		return
	}
	sort.Strings(names)
}

func orEmptyObject(v Value) Value {
	if truthy(v) {
		return v
	}
	return newObject()
}

func readJSONFile(p string) (Value, string) {
	b, err := os.ReadFile(p)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Sprintf("cannot parse %s: [Errno 2] No such file or directory: %s", filepath.Base(p), pyStrString(p))
		}
		return nil, fmt.Sprintf("cannot parse %s: %v", filepath.Base(p), err)
	}
	if !utf8.Valid(b) {
		return nil, fmt.Sprintf("cannot parse %s: 'utf-8' codec can't decode the file", filepath.Base(p))
	}
	v, err := DecodeValue(b)
	if err != nil {
		return nil, fmt.Sprintf("cannot parse %s: %v", filepath.Base(p), err)
	}
	return v, ""
}
