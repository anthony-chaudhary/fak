// Package generalizationdebt detects production implementations whose shape is
// coupled to one model, backend, or provider instead of an interface or registry.
package generalizationdebt

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

const Schema = "fak-generalization-debt-scorecard/1"

type Disposition string

const (
	DispositionAccidental        Disposition = "accidental_unaccepted"
	DispositionAcceptedTemporary Disposition = "accepted_temporary"
)

type InterestBand string

const (
	InterestLow      InterestBand = "low"
	InterestModerate InterestBand = "moderate"
	InterestHigh     InterestBand = "high"
	InterestCritical InterestBand = "critical"
)

type AcceptedDebt struct {
	Rationale string `json:"rationale"`
	Owner     string `json:"owner"`
	ExitGate  string `json:"exit_gate"`
}

type Interest struct {
	Band    InterestBand `json:"band"`
	Rate    string       `json:"rate"`
	Drivers []string     `json:"drivers"`
}

type Finding struct {
	Path        string        `json:"path"`
	Line        int           `json:"line"`
	Kind        string        `json:"kind"`
	Subject     string        `json:"subject"`
	Disposition Disposition   `json:"disposition"`
	Accepted    *AcceptedDebt `json:"accepted,omitempty"`
	Interest    Interest      `json:"interest"`
	Evidence    string        `json:"evidence"`
	DebtPoints  int           `json:"debt_points"`
}

type Totals struct {
	Findings             int `json:"findings"`
	DebtPoints           int `json:"debt_points"`
	AccidentalUnaccepted int `json:"accidental_unaccepted"`
	AcceptedTemporary    int `json:"accepted_temporary"`
}

type Report struct {
	Schema   string    `json:"schema"`
	Totals   Totals    `json:"totals"`
	Findings []Finding `json:"findings"`
}

var acceptedRE = regexp.MustCompile(`fak:generalization-debt\s+accepted_temporary\s+rationale=("(?:[^"\\]|\\.)*")\s+owner=("(?:[^"\\]|\\.)*")\s+exit_gate=("(?:[^"\\]|\\.)*")\s*$`)

type specificity struct {
	name    string
	kind    string
	aliases [][]string
}

var specificities = []specificity{
	{name: "anthropic", kind: "provider_specific", aliases: [][]string{{"anthropic"}}},
	{name: "openai", kind: "provider_specific", aliases: [][]string{{"open", "ai"}, {"openai"}}},
	{name: "azure_openai", kind: "provider_specific", aliases: [][]string{{"azure", "open", "ai"}, {"azure", "openai"}}},
	{name: "bedrock", kind: "provider_specific", aliases: [][]string{{"bedrock"}}},
	{name: "cohere", kind: "provider_specific", aliases: [][]string{{"cohere"}}},
	{name: "vertex", kind: "provider_specific", aliases: [][]string{{"vertex"}}},
	{name: "claude", kind: "model_specific", aliases: [][]string{{"claude"}}},
	{name: "gemini", kind: "model_specific", aliases: [][]string{{"gemini"}}},
	{name: "mistral", kind: "model_specific", aliases: [][]string{{"mistral"}}},
	{name: "qwen3_8", kind: "model_specific", aliases: [][]string{{"qwen3", "8"}, {"qwen38"}}},
	{name: "llama_cpp", kind: "backend_specific", aliases: [][]string{{"llama", "cpp"}, {"llamacpp"}}},
	{name: "ollama", kind: "backend_specific", aliases: [][]string{{"ollama"}}},
	{name: "tensorrt", kind: "backend_specific", aliases: [][]string{{"tensorrt"}}},
	{name: "vllm", kind: "backend_specific", aliases: [][]string{{"vllm"}}},
}

// Scan walks root and returns a stable report. It deliberately ignores tests,
// examples, generated code, scratch trees, and historical Qwen3.6 artifacts.
func Scan(root string) (Report, error) {
	var findings []Finding
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel != "." && excludedDir(rel, d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") || excludedPath(rel) {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if generated(data) {
			return nil
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, path, data, parser.ParseComments)
		if err != nil {
			return fmt.Errorf("parse %s: %w", rel, err)
		}
		findings = append(findings, scanFile(fset, file, rel)...)
		return nil
	})
	if err != nil {
		return Report{}, err
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].Path != findings[j].Path {
			return findings[i].Path < findings[j].Path
		}
		if findings[i].Line != findings[j].Line {
			return findings[i].Line < findings[j].Line
		}
		if findings[i].Kind != findings[j].Kind {
			return findings[i].Kind < findings[j].Kind
		}
		return findings[i].Subject < findings[j].Subject
	})
	r := Report{Schema: Schema, Findings: findings}
	if r.Findings == nil {
		r.Findings = []Finding{}
	}
	for _, f := range findings {
		r.Totals.Findings++
		r.Totals.DebtPoints += f.DebtPoints
		if f.Disposition == DispositionAcceptedTemporary {
			r.Totals.AcceptedTemporary++
		} else {
			r.Totals.AccidentalUnaccepted++
		}
	}
	return r, nil
}

func excludedDir(rel, name string) bool {
	name = strings.ToLower(name)
	if strings.HasPrefix(name, ".") || name == "_scratch" || name == "testdata" || name == "examples" || name == "docs" || name == "vendor" {
		return true
	}
	return historicalQwen36Path(rel)
}

func excludedPath(rel string) bool {
	p := "/" + strings.ToLower(filepath.ToSlash(rel)) + "/"
	return strings.Contains(p, "/testdata/") || strings.Contains(p, "/examples/") || strings.Contains(p, "/docs/") || historicalQwen36Path(rel)
}

func historicalQwen36Path(rel string) bool {
	for _, part := range strings.FieldsFunc(strings.ToLower(filepath.ToSlash(rel)), func(r rune) bool {
		return r == '/' || r == '-' || r == '_' || r == '.'
	}) {
		if part == "qwen36" {
			return true
		}
	}
	lower := strings.ToLower(filepath.ToSlash(rel))
	return strings.Contains(lower, "qwen3.6") || strings.Contains(lower, "qwen3_6") || strings.Contains(lower, "qwen3-6")
}

func generated(data []byte) bool {
	s := bufio.NewScanner(bytes.NewReader(data))
	for n := 0; n < 20 && s.Scan(); n++ {
		line := strings.ToLower(s.Text())
		if strings.Contains(line, "code generated") && strings.Contains(line, "do not edit") {
			return true
		}
	}
	return false
}

func scanFile(fset *token.FileSet, file *ast.File, rel string) []Finding {
	var out []Finding
	for _, decl := range file.Decls {
		if genericDeclaration(decl) {
			continue
		}
		terms, kinds := matchedSpecific(decl)
		if len(terms) == 0 {
			continue
		}
		start := fset.Position(decl.Pos())
		subject := declarationSubject(decl)
		points := 8
		drivers := []string{"implementation_specificity"}
		if len(terms) > 1 {
			points += 2
			drivers = append(drivers, "unsupported_variant_count")
		}
		if ast.IsExported(subject) {
			points += 2
			drivers = append(drivers, "blast_radius")
		}
		accepted := parseAccepted(declDoc(decl))
		disposition := DispositionAccidental
		if accepted != nil {
			disposition = DispositionAcceptedTemporary
			drivers = append(drivers, "explicit_retirement_gate")
		}
		sort.Strings(drivers)
		kind := "specific_implementation"
		if len(kinds) == 1 {
			kind = kinds[0]
		}
		out = append(out, Finding{
			Path: rel, Line: start.Line, Kind: kind, Subject: subject,
			Disposition: disposition, Accepted: accepted, Interest: interest(points, drivers),
			Evidence: strings.Join(terms, ","), DebtPoints: points,
		})
	}
	return out
}

// genericDeclaration recognizes the abstraction shape itself, rather than
// searching declaration text. A comment or local variable named registry must
// not launder a concrete implementation into zero debt.
func genericDeclaration(decl ast.Decl) bool {
	if gen, ok := decl.(*ast.GenDecl); ok && gen.Tok == token.TYPE {
		for _, spec := range gen.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok {
				if _, ok := ts.Type.(*ast.InterfaceType); ok {
					return true
				}
			}
		}
	}
	words := identifierWords(declarationSubject(decl))
	for _, word := range words {
		switch word {
		case "registry", "register", "registration", "factory", "adapter":
			return true
		}
	}
	return false
}

func matchedSpecific(decl ast.Decl) ([]string, []string) {
	found := make(map[string]string)
	ast.Inspect(decl, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CommentGroup, *ast.Comment:
			return false
		case *ast.Ident:
			matchWords(identifierWords(x.Name), found)
		case *ast.BasicLit:
			if x.Kind == token.STRING {
				if value, err := strconv.Unquote(x.Value); err == nil {
					matchWords(identifierWords(value), found)
				}
			}
		}
		return true
	})
	terms := make([]string, 0, len(found))
	kindSet := make(map[string]struct{})
	for term, kind := range found {
		terms = append(terms, term)
		kindSet[kind] = struct{}{}
	}
	sort.Strings(terms)
	kinds := make([]string, 0, len(kindSet))
	for kind := range kindSet {
		kinds = append(kinds, kind)
	}
	sort.Strings(kinds)
	return terms, kinds
}

func matchWords(words []string, found map[string]string) {
	for _, spec := range specificities {
		for _, alias := range spec.aliases {
			if containsSequence(words, alias) {
				found[spec.name] = spec.kind
				break
			}
		}
	}
}

func containsSequence(words, sequence []string) bool {
	for i := 0; i+len(sequence) <= len(words); i++ {
		match := true
		for j := range sequence {
			if words[i+j] != sequence[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func identifierWords(s string) []string {
	var words []string
	var current []rune
	flush := func() {
		if len(current) > 0 {
			words = append(words, strings.ToLower(string(current)))
			current = current[:0]
		}
	}
	runes := []rune(s)
	for i, r := range runes {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			flush()
			continue
		}
		if len(current) > 0 && unicode.IsUpper(r) {
			prev := runes[i-1]
			nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
			if unicode.IsLower(prev) || unicode.IsDigit(prev) || nextLower {
				flush()
			}
		}
		current = append(current, r)
	}
	flush()
	return words
}

func declarationSubject(d ast.Decl) string {
	switch x := d.(type) {
	case *ast.FuncDecl:
		return x.Name.Name
	case *ast.GenDecl:
		for _, s := range x.Specs {
			switch v := s.(type) {
			case *ast.TypeSpec:
				return v.Name.Name
			case *ast.ValueSpec:
				if len(v.Names) > 0 {
					return v.Names[0].Name
				}
			}
		}
	}
	return "declaration"
}

func declDoc(d ast.Decl) *ast.CommentGroup {
	switch x := d.(type) {
	case *ast.FuncDecl:
		return x.Doc
	case *ast.GenDecl:
		return x.Doc
	}
	return nil
}

func parseAccepted(doc *ast.CommentGroup) *AcceptedDebt {
	if doc == nil {
		return nil
	}
	for _, c := range doc.List {
		m := acceptedRE.FindStringSubmatch(strings.TrimSpace(strings.TrimPrefix(c.Text, "//")))
		if m == nil {
			continue
		}
		vals := make([]string, 3)
		ok := true
		for i := range vals {
			vals[i], _ = strconv.Unquote(m[i+1])
			if strings.TrimSpace(vals[i]) == "" {
				ok = false
			}
		}
		if ok {
			return &AcceptedDebt{Rationale: vals[0], Owner: vals[1], ExitGate: vals[2]}
		}
	}
	return nil
}

func interest(points int, drivers []string) Interest {
	var band InterestBand
	var rate string
	switch {
	case points >= 10:
		band, rate = InterestCritical, "compounding"
	case points >= 8:
		band, rate = InterestHigh, "accelerating"
	case points >= 5:
		band, rate = InterestModerate, "elevated"
	default:
		band, rate = InterestLow, "baseline"
	}
	return Interest{Band: band, Rate: rate, Drivers: drivers}
}
