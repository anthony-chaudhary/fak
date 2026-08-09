package enumlint

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// discover.go is THE discovery pass — the one AST walker that turns Go source
// into closed enumerations. Both this package's exhaustiveness rules (#5935) and
// the closed-vocabulary drift check (#5649) read enumerations through it, which
// is the whole reason it is exported. Two walkers would be two answers to "what
// are this type's constants?", and the first time they disagreed the drift check
// and the exhaustiveness check would each certify the half the other missed.

// Member is one constant of a closed enumeration, located precisely enough to
// fix a finding without a search.
type Member struct {
	// Name is the constant's identifier as written.
	Name string
	// File is the repo-relative, slash-separated path of the declaration.
	File string
	// Line is the 1-based line of the declaration.
	Line int
	// Val is the basic literal the constant is declared with, unquoted. It is
	// "" for an IOTA MEMBER — a constant whose value comes from the const
	// block's iota run rather than from a literal of its own. Callers that
	// synthesise a new member (the realsite mutation test) need the difference:
	// an iota member is a bare name, a valued member needs a literal.
	Val string
}

// IsIota reports whether the member takes its value from an iota run rather
// than from a literal of its own.
func (m Member) IsIota() bool { return m.Val == "" }

// Enum is one discovered closed enumeration: a defined type over a basic kind,
// plus every package-level constant declared with it.
type Enum struct {
	// Name is the defined type's name, unqualified.
	Name string
	// Kind is the underlying basic kind ("string", "int", ...).
	Kind string
	// Pkg is the repo-relative, slash-separated directory that declares it.
	Pkg string
	// File and Line locate the `type X ...` declaration itself.
	File string
	Line int
	// Members are the constants, in declaration order.
	Members []Member

	byName map[string]bool
}

// Has reports whether name is one of the enumeration's constants.
func (e *Enum) Has(name string) bool { return e.byName[name] }

// Names returns the members' names in declaration order.
func (e *Enum) Names() []string {
	out := make([]string, 0, len(e.Members))
	for _, m := range e.Members {
		out = append(out, m.Name)
	}
	return out
}

// Missing returns the members not present in named, in declaration order.
func (e *Enum) Missing(named map[string]bool) []Member {
	var out []Member
	for _, m := range e.Members {
		if !named[m.Name] {
			out = append(out, m)
		}
	}
	return out
}

// basicKinds is the underlying-type allowlist, and its NARROWNESS is the whole
// scope decision of this package.
//
// Only a defined type over one of these kinds can have a package-level const
// block, and only a const block makes a type CLOSED. A defined type over a
// struct, a slice, a map or a func is an open set — there is no enumeration of
// its values to be exhaustive over — so admitting one would not widen coverage,
// it would manufacture findings against types that have no const block to be
// closed against. Every such "finding" would be unfixable except by exempting
// it, and a gate whose only remedy is an exemption is a gate that gets turned
// off. `float32`/`float64` are excluded for the same reason with an extra edge:
// they are legal const kinds but fak declares no float enumerations, and
// admitting them would put float equality into the case-matching path.
//
// `bool` is excluded too, and deliberately: a defined bool type has exactly two
// values and a two-arm switch on it is an `if`, not an enumeration.
var basicKinds = map[string]bool{
	"string": true,
	"int":    true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"byte": true, "rune": true,
}

// BasicKinds returns the underlying-kind allowlist, sorted. Exported so a
// caller (and this package's own doc test) can state the scope boundary rather
// than restate it by hand.
func BasicKinds() []string {
	out := make([]string, 0, len(basicKinds))
	for k := range basicKinds {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// MinMembers is the smallest const block this package will treat as an
// enumeration. A one-constant type is a sentinel, not a set, and demanding
// exhaustiveness over it would fire on every named string type in the tree that
// happens to carry a single default.
const MinMembers = 2

// Package is one directory's worth of parsed Go plus everything discovered in
// it. It is the unit both #5935's rules and #5649's drift check work over.
type Package struct {
	// Dir is the repo-relative, slash-separated directory.
	Dir string
	// Enums are the discovered enumerations, by type name.
	Enums map[string]*Enum

	root    string
	fset    *token.FileSet
	files   []*ast.File
	structs map[string]structEnumField
	// exemptLines records the line ranges an in-place //enumlint:exempt
	// directive covers, per repo-relative file.
	exemptLines map[string][]lineRange
}

type lineRange struct{ lo, hi int }

// structEnumField is a struct's single enum-typed field. See enumFieldOf for
// why "single" is load-bearing.
type structEnumField struct {
	typeName string // the enum type's name
	index    int    // positional index of the field within the struct
	field    string // the field's name, for keyed composite literals
}

// EnumNames returns the package's enumeration names, sorted, so a caller
// iterates deterministically.
func (p *Package) EnumNames() []string {
	out := make([]string, 0, len(p.Enums))
	for n := range p.Enums {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// MemberCount is the total number of constants across the package's
// enumerations — the census number a caller asserts on before believing a
// zero-finding verdict.
func (p *Package) MemberCount() int {
	n := 0
	for _, e := range p.Enums {
		n += len(e.Members)
	}
	return n
}

// Discover walks root and returns every package that declares at least one
// closed enumeration, plus a list of "<path>: <error>" for each file that would
// not parse.
//
// A parse failure is REPORTED, never absorbed. This tree is shared by ~20 live
// sessions, so a peer's half-written file is routine and failing the whole walk
// on someone else's editor state would make this gate the fleet's problem
// rather than the tree's. But a skipped file must never read as a checked one,
// which is the difference between "no findings" and "did not look".
func Discover(root string, cfg Config) ([]*Package, []string, error) {
	cfg = cfg.withDefaults()
	skip := map[string]bool{}
	for _, d := range cfg.SkipDirs {
		skip[d] = true
	}

	includeTop := map[string]bool{}
	for _, d := range cfg.IncludeTopDirs {
		includeTop[d] = true
	}
	var dirs []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		if path != root && len(includeTop) > 0 && filepath.Dir(path) == filepath.Clean(root) && !includeTop[d.Name()] {
			return fs.SkipDir
		}
		if path != root && (skip[d.Name()] || strings.HasPrefix(d.Name(), ".")) {
			return fs.SkipDir
		}
		dirs = append(dirs, path)
		return nil
	})
	if err != nil {
		return nil, nil, &ScanError{Op: "walk", Path: root, Err: err}
	}
	sort.Strings(dirs)

	var (
		pkgs     []*Package
		unparsed []string
	)
	for _, dir := range dirs {
		p, u, err := LoadPackage(root, dir, cfg.IncludeTestFiles)
		unparsed = append(unparsed, u...)
		if err != nil {
			return nil, unparsed, err
		}
		if p == nil {
			continue
		}
		pkgs = append(pkgs, p)
	}
	return pkgs, unparsed, nil
}

// LoadPackage parses one directory and discovers its enumerations. It returns
// (nil, unparsed, nil) when the directory holds no Go files or no enumeration —
// "nothing here" is not an error.
//
// includeTests reads _test.go files too. A tree scan wants them ON: fak's
// hand-written coverage assertions (wipref's TestClassifyVocabulary, the shape
// this whole package generalises) live in test files, and a scan that skipped
// them would report clean on exactly the sites the ticket is about.
func LoadPackage(root, dir string, includeTests bool) (*Package, []string, error) {
	var unparsed []string
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, unparsed, &ScanError{Op: "readdir", Path: dir, Err: err}
	}
	rel, err := filepath.Rel(root, dir)
	if err != nil {
		return nil, unparsed, &ScanError{Op: "relativise", Path: dir, Err: err}
	}
	if rel == "." {
		rel = ""
	}
	p := &Package{
		Dir:         filepath.ToSlash(rel),
		Enums:       map[string]*Enum{},
		root:        root,
		fset:        token.NewFileSet(),
		structs:     map[string]structEnumField{},
		exemptLines: map[string][]lineRange{},
	}
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		if !includeTests && strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		full := filepath.Join(dir, e.Name())
		f, err := parser.ParseFile(p.fset, full, nil, parser.ParseComments)
		if err != nil {
			unparsed = append(unparsed, p.relPath(full)+": "+err.Error())
			continue
		}
		p.files = append(p.files, f)
		p.collectExemptDirectives(f)
	}
	if len(p.files) == 0 {
		return nil, unparsed, nil
	}
	p.collectTypes()
	p.collectConsts()
	for name, e := range p.Enums {
		if len(e.Members) < MinMembers {
			delete(p.Enums, name)
		}
	}
	if len(p.Enums) == 0 {
		return nil, unparsed, nil
	}
	return p, unparsed, nil
}

// collectTypes finds the candidate enumeration types, then — once every enum
// name is known — every struct that has EXACTLY ONE enum-typed field.
//
// The second half is what lets the composite-literal rule see through a
// vocabulary of structs. A `[]RefusalReason` whose RefusalReason carries a
// `Code RefusalCode` is enumerating RefusalCode one field deep; a rule that
// only understood `[]RefusalCode` would report that site clean, and that shape
// is the one fak writes most.
func (p *Package) collectTypes() {
	// Pass 1: every defined type over an allowlisted basic kind.
	for _, f := range p.files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || ts.Assign.IsValid() { // a type alias is not a defined type
					continue
				}
				id, ok := ts.Type.(*ast.Ident)
				if !ok || !basicKinds[id.Name] {
					continue
				}
				file, line := p.pos(ts.Name)
				p.Enums[ts.Name.Name] = &Enum{
					Name:   ts.Name.Name,
					Kind:   id.Name,
					Pkg:    p.Dir,
					File:   file,
					Line:   line,
					byName: map[string]bool{},
				}
			}
		}
	}
	// Pass 2: structs, now that every enum name is known.
	for _, f := range p.files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || ts.Assign.IsValid() {
					continue
				}
				st, ok := ts.Type.(*ast.StructType)
				if !ok {
					continue
				}
				if sf, ok := p.enumFieldOf(st); ok {
					p.structs[ts.Name.Name] = sf
				}
			}
		}
	}
}

// enumFieldOf returns the struct's single enum-typed field, if it has exactly
// one.
//
// "Exactly one" is the entire safety argument. With two enum-typed fields there
// is no single set a literal of that struct is enumerating, so the literal's
// obligation is ambiguous and this package declines to guess rather than pick
// the first field and be confidently wrong.
//
// It takes an *ast.StructType rather than a type name so it also reads the
// ANONYMOUS struct — `[]struct{ kind Kind; wire string }{...}` is a table shape
// fak writes constantly, and a named-types-only reader would report every one of
// them clean.
func (p *Package) enumFieldOf(st *ast.StructType) (structEnumField, bool) {
	if st.Fields == nil {
		return structEnumField{}, false
	}
	var hits []structEnumField
	idx := 0
	for _, fld := range st.Fields.List {
		n := len(fld.Names)
		if n == 0 {
			n = 1 // an embedded field still occupies a position
		}
		tn := typeName(fld.Type)
		if _, isEnum := p.Enums[tn]; isEnum {
			for i, nm := range fld.Names {
				hits = append(hits, structEnumField{typeName: tn, index: idx + i, field: nm.Name})
			}
			if len(fld.Names) == 0 {
				hits = append(hits, structEnumField{typeName: tn, index: idx, field: tn})
			}
		}
		idx += n
	}
	if len(hits) != 1 {
		return structEnumField{}, false
	}
	return hits[0], true
}

// collectConsts attaches every package-level constant to its enumeration.
//
// The trap that makes this thirty lines rather than five: IN AN IOTA BLOCK THE
// TYPE APPEARS ON THE FIRST ValueSpec ONLY.
//
//	const (
//	        KindUnknown Kind = iota // <- typed
//	        KindNewAsk              // <- ValueSpec.Type is nil
//	        KindMechanical          // <- ValueSpec.Type is nil
//	)
//
// A walker that reads each spec's own Type therefore discovers ONE member of
// turnkind.Kind instead of four, and every exhaustiveness assertion over it
// passes while checking almost nothing. That failure has exactly the shape of
// the defect this package exists to catch, so it would have been invisible: the
// suite green and the linter blind. The last-seen type is carried forward
// across specs within one GenDecl, and TestIotaBlockYieldsEveryMember pins it.
func (p *Package) collectConsts() {
	for _, f := range p.files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			iotaRun := isIotaOnlyGroup(gd)
			carried := "" // the type declared on the most recent typed spec
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				switch {
				case vs.Type != nil:
					carried = typeName(vs.Type)
				case len(vs.Values) > 0 && !iotaRun:
					// An untyped spec carrying its own value, in a block that is
					// not an iota run, belongs to no carried type. Without this
					// a mixed const block would absorb unrelated constants into
					// whichever enum was named above them.
					carried = ""
				}
				e, ok := p.Enums[carried]
				if !ok {
					continue
				}
				for i, nm := range vs.Names {
					if nm.Name == "_" {
						continue
					}
					file, line := p.pos(nm)
					val := ""
					if i < len(vs.Values) {
						val = basicLit(vs.Values[i])
					}
					e.Members = append(e.Members, Member{Name: nm.Name, File: file, Line: line, Val: val})
					e.byName[nm.Name] = true
				}
			}
		}
	}
}

// isIotaOnlyGroup reports whether a const block's FIRST spec uses iota, in which
// case later untyped specs continue the run and inherit the carried type.
func isIotaOnlyGroup(gd *ast.GenDecl) bool {
	for _, spec := range gd.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for _, v := range vs.Values {
			if mentionsIota(v) {
				return true
			}
		}
		return false // only the first spec decides
	}
	return false
}

func mentionsIota(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == "iota" {
			found = true
		}
		return !found
	})
	return found
}

// typeName reduces a type expression to the bare identifier this package
// reasons about. A qualified name (turnkind.Kind) reduces to its selector, so an
// EXTERNAL test package's `[]turnkind.Kind` literal is understood as a literal
// of Kind — fak's hand-written coverage tables routinely live in `package
// foo_test`, and dropping the qualified form would skip every one of them.
func typeName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		return t.Sel.Name
	case *ast.StarExpr:
		return typeName(t.X)
	case *ast.ParenExpr:
		return typeName(t.X)
	}
	return ""
}

// identName is typeName's value-expression twin: the constant an expression
// names, or "" when the expression is not a plain constant reference.
func identName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		if _, ok := t.X.(*ast.Ident); ok {
			return t.Sel.Name
		}
	case *ast.ParenExpr:
		return identName(t.X)
	}
	return ""
}

func basicLit(e ast.Expr) string {
	bl, ok := e.(*ast.BasicLit)
	if !ok {
		return ""
	}
	if bl.Kind == token.STRING {
		if s, err := strconv.Unquote(bl.Value); err == nil {
			return s
		}
	}
	return bl.Value
}

func (p *Package) relPath(full string) string {
	rel, err := filepath.Rel(p.root, full)
	if err != nil {
		return filepath.ToSlash(full)
	}
	return filepath.ToSlash(rel)
}

func (p *Package) pos(n ast.Node) (string, int) {
	pos := p.fset.Position(n.Pos())
	return p.relPath(pos.Filename), pos.Line
}
