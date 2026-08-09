package devindex

// verbsurface.go — the SOURCE-derived `fak` verb surface (#5934, epic #5949): every
// verb and sub-verb the binary routes, what implements it, whether it appears in
// fak's own help wall, and — the column this generator exists for — what it REFUSES,
// by closed reason code.
//
// Two design constraints, both borrowed from tb's tools/verbsdoc (their IDX-03 /
// #915) because both were learned from a failure rather than chosen:
//
//  1. The tree is read from SOURCE, never scraped from `--help`. A sub-verb missing
//     from its own help text is precisely the drift this page exists to catch, and a
//     scraper would inherit the omission and call it complete. It would also measure
//     the binary you happened to build, not the tree you are about to commit. Every
//     row below comes out of go/ast over cmd/fak/*.go.
//  2. A blank preconditions cell is impossible by construction. PreState (see
//     verbsurface_refusals.go) is a closed five-state enum whose ZERO VALUE is
//     UNVERIFIED, and the count of unverified rows is printed in the generated page
//     itself. A generated index that hides its own gaps reads as complete coverage,
//     which is worse than no index.
//
// Lane note: this is the internal/devindex half. The `fak verb-surface` shell
// (check / -write / -print) is the cmd/ half, and internal/hooks/gate_verbsurface.go
// is the gate-family half.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// vsCmdDir is the package the surface is derived from. It is a constant rather than a
// parameter because a reader pointed at the wrong directory parses nothing, finds
// nothing, and reports an EMPTY command tree as a clean one — see vsFileFloor.
const vsCmdDir = "cmd/fak"

// vsFileFloor is a floor on the parsed non-test file count of cmd/fak.
//
// A generator whose input silently became empty is not green, it is blind: every
// containment check downstream passes against an empty set. cmd/fak carries ~840
// non-test files today, so 300 is a floor with room for a large deletion and no room
// for "the glob matched nothing".
const vsFileFloor = 300

// vsMaxWords bounds command-path depth. Three is measured against the tree, not
// chosen: `fak worktree worker prepare` is the deepest real path, and an unbounded
// walk would follow a helper that happens to switch on argv[0] for a reason that is
// not a sub-verb at all.
const vsMaxWords = 3

// vsReachDepth bounds the refusal walk. Past four calls out of a handler the walk
// stops answering "what does this verb check" and starts answering "what does this
// binary contain".
const vsReachDepth = 4

// ---------------------------------------------------------------- the parsed package

// vsPkg is cmd/fak parsed once: every non-test file, plus the indexes the extractors
// need — function name to declaration, each file's import map (which turns a bare
// `safecommit.ReasonOffTrunk` selector into internal/safecommit), and each file's
// repo-relative path so a claim in the page cites a file:line a reader can open.
type vsPkg struct {
	fset    *token.FileSet
	files   []*ast.File
	names   []string // declared func names, sorted — the deterministic iteration order
	funcs   map[string]*ast.FuncDecl
	fileOf  map[string]*ast.File
	pathOf  map[*ast.File]string
	imports map[*ast.File]map[string]string
	// verbTokenParams maps a function name to the parameter names it receives
	// os.Args[1] in. It is what lets `dispatchPrimaryVerb(os.Args[1], …)`'s
	// `switch name` read as a verb dispatch switch without naming it by hand.
	verbTokenParams map[string]map[string]bool
	// helpText is every string literal declared in cmd/fak's help/usage sources,
	// concatenated. It is the binary's OWN help wall, read from source rather than
	// executed — the point being to compare the two and print the difference.
	helpText string
}

// vsLoadCmd parses every non-test .go file under <root>/cmd/fak.
func vsLoadCmd(root string) (*vsPkg, error) {
	dir := filepath.Join(root, filepath.FromSlash(vsCmdDir))
	entries, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, err
	}
	sort.Strings(entries)
	p := &vsPkg{
		fset:            token.NewFileSet(),
		funcs:           map[string]*ast.FuncDecl{},
		fileOf:          map[string]*ast.File{},
		pathOf:          map[*ast.File]string{},
		imports:         map[*ast.File]map[string]string{},
		verbTokenParams: map[string]map[string]bool{},
	}
	var help []string
	for _, name := range entries {
		base := filepath.Base(name)
		if strings.HasSuffix(base, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			return nil, err
		}
		f, err := parser.ParseFile(p.fset, name, src, parser.ParseComments)
		if err != nil {
			// One unparseable file must not silently shrink the surface: a partial
			// tree renders as a smaller tool, which is the failure this generator is
			// supposed to make impossible.
			return nil, fmt.Errorf("%s/%s: %w", vsCmdDir, base, err)
		}
		p.files = append(p.files, f)
		p.pathOf[f] = vsCmdDir + "/" + base
		p.imports[f] = vsImportMap(f)
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Body == nil {
				continue
			}
			p.funcs[fd.Name.Name] = fd
			p.fileOf[fd.Name.Name] = f
		}
		if vsIsHelpSource(base) {
			help = append(help, vsStringLiterals(f)...)
		}
	}
	if len(p.files) < vsFileFloor {
		return nil, fmt.Errorf("parsed only %d non-test file(s) under %s: the package is far bigger "+
			"than that, so the reader is pointed at the wrong tree and would report an EMPTY command "+
			"surface as a complete one", len(p.files), vsCmdDir)
	}
	p.names = make([]string, 0, len(p.funcs))
	for name := range p.funcs {
		p.names = append(p.names, name)
	}
	sort.Strings(p.names)
	p.helpText = strings.Join(help, "\n")
	p.indexVerbTokenParams()
	return p, nil
}

// vsIsHelpSource names the files that hold fak's own help wall. `fak help --full`
// prints usageCoreText+usageOpsText+usageScorecardText from usage.go, and help.go
// carves per-verb sections out of the same wall.
func vsIsHelpSource(base string) bool {
	return base == "usage.go" || base == "help.go"
}

func vsImportMap(f *ast.File) map[string]string {
	out := map[string]string{}
	for _, imp := range f.Imports {
		path, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			continue
		}
		local := path[strings.LastIndex(path, "/")+1:]
		if imp.Name != nil {
			local = imp.Name.Name
		}
		out[local] = path
	}
	return out
}

func (p *vsPkg) site(f *ast.File, pos token.Pos) string {
	return fmt.Sprintf("%s:%d", p.pathOf[f], p.fset.Position(pos).Line)
}

// indexVerbTokenParams records, for every call `f(…, os.Args[1], …)` to a package-main
// function, which of f's parameters the verb token lands in.
//
// This is what generalizes the reader past `switch os.Args[1]`. main() dispatches part
// of the surface itself and hands the rest to `dispatchPrimaryVerb(os.Args[1], …)`,
// whose `switch name` is a verb dispatch switch by virtue of that call and nothing
// else. Naming `dispatchPrimaryVerb` in a table here would work exactly until the next
// extraction moved it — the same hand-list rot this page exists to end.
func (p *vsPkg) indexVerbTokenParams() {
	for _, name := range p.names {
		ast.Inspect(p.funcs[name].Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok {
				return true
			}
			callee, ok := p.funcs[id.Name]
			if !ok || callee.Type.Params == nil {
				return true
			}
			params := vsParamNames(callee)
			for i, arg := range call.Args {
				if i >= len(params) || !vsIsOSArgs1(arg) || params[i] == "" || params[i] == "_" {
					continue
				}
				if p.verbTokenParams[id.Name] == nil {
					p.verbTokenParams[id.Name] = map[string]bool{}
				}
				p.verbTokenParams[id.Name][params[i]] = true
			}
			return true
		})
	}
}

// vsParamNames flattens a parameter list to one name per positional argument, so
// `func f(a, b string, c []string)` reads as ["a","b","c"].
func vsParamNames(fd *ast.FuncDecl) []string {
	var out []string
	if fd.Type.Params == nil {
		return out
	}
	for _, field := range fd.Type.Params.List {
		if len(field.Names) == 0 {
			out = append(out, "")
			continue
		}
		for _, n := range field.Names {
			out = append(out, n.Name)
		}
	}
	return out
}

// vsArgSliceParam names the function's first []string parameter — the residual argv a
// sub-verb dispatcher indexes at 0.
func vsArgSliceParam(fd *ast.FuncDecl) string {
	if fd.Type.Params == nil {
		return ""
	}
	for _, field := range fd.Type.Params.List {
		arr, ok := field.Type.(*ast.ArrayType)
		if !ok || arr.Len != nil {
			continue
		}
		id, ok := arr.Elt.(*ast.Ident)
		if !ok || id.Name != "string" || len(field.Names) == 0 {
			continue
		}
		return field.Names[0].Name
	}
	return ""
}

func vsIsOSArgs1(e ast.Expr) bool {
	idx, ok := e.(*ast.IndexExpr)
	if !ok {
		return false
	}
	sel, ok := idx.X.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Args" {
		return false
	}
	pkg, ok := sel.X.(*ast.Ident)
	if !ok || pkg.Name != "os" {
		return false
	}
	lit, ok := idx.Index.(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value == "1"
}

// vsIsIndexZero reports whether e is `<name>[0]`.
func vsIsIndexZero(e ast.Expr, name string) bool {
	if name == "" {
		return false
	}
	idx, ok := e.(*ast.IndexExpr)
	if !ok {
		return false
	}
	id, ok := idx.X.(*ast.Ident)
	if !ok || id.Name != name {
		return false
	}
	lit, ok := idx.Index.(*ast.BasicLit)
	return ok && lit.Kind == token.INT && lit.Value == "0"
}

// ---------------------------------------------------------------- the command tree

// SurfaceLeaf is one invocable command path — `fak commit`, `fak accounts add`,
// `fak worktree worker prepare`. The leaf, not the verb, is the unit this page has to
// be complete over: a doc that lists verbs and is silent about sub-verbs is exactly
// the state that sends a reader to a bespoke script when the verb they needed already
// existed one word deeper.
type SurfaceLeaf struct {
	Words    []string `json:"words"`
	Aliases  []string `json:"aliases,omitempty"`
	Synopsis string   `json:"synopsis,omitempty"`
	Fn       string   `json:"fn,omitempty"`
	Origin   string   `json:"origin"`
	Planned  bool     `json:"planned,omitempty"`
	Tier     VerbTier `json:"tier,omitempty"`
	Packages []string `json:"packages,omitempty"`
	// InHelp is whether fak's own help wall mentions this path. It is derived from
	// the help SOURCE, never from running the binary, so the comparison is between
	// two readings of the same tree rather than between a tree and a build.
	InHelp bool       `json:"in_help"`
	Pre    SurfacePre `json:"pre"`
}

// Path is what an operator types.
func (l SurfaceLeaf) Path() string { return "fak " + strings.Join(l.Words, " ") }

// Key is the path without the binary name — the map key the curated tables use.
func (l SurfaceLeaf) Key() string { return strings.Join(l.Words, " ") }

// Depth is the word count: 1 for a verb, 2+ for a sub-verb.
func (l SurfaceLeaf) Depth() int { return len(l.Words) }

// IsVerb reports whether this leaf is a top-level verb.
func (l SurfaceLeaf) IsVerb() bool { return len(l.Words) == 1 }

// Verb is the top-level verb this leaf hangs off.
func (l SurfaceLeaf) Verb() string { return l.Words[0] }

// VerbSurface is the whole derived command tree plus the provenance of the refusal
// vocabulary it was graded against.
type VerbSurface struct {
	Leaves []SurfaceLeaf `json:"leaves"`
	// Lexicon is the closed refusal vocabulary the REFUSES column is resolved
	// against, with each code's declaring site. It is derived, never hand-listed.
	Lexicon *ReasonLexicon `json:"-"`
	// Files is the count of cmd/fak non-test files parsed — printed on the page so
	// a shrunken surface is attributable to a shrunken input rather than mysterious.
	Files int `json:"files"`
}

// vsRow is one derived dispatch arm before it becomes a leaf.
type vsRow struct {
	name    string
	aliases []string
	fn      string
	origin  string
}

// ExtractVerbSurface walks cmd/fak's own dispatch switches and returns the complete
// command surface with its refusal column resolved.
//
// root is the repository root. The refusal lexicon is derived from the same tree
// (dos.toml's closed [reasons.*] registry plus every reason-shaped constant declared
// under internal/ and cmd/fak), so the page never asserts a code the tree does not
// declare.
func ExtractVerbSurface(root string) (*VerbSurface, error) {
	p, err := vsLoadCmd(root)
	if err != nil {
		return nil, err
	}
	lex, err := BuildReasonLexicon(root)
	if err != nil {
		return nil, err
	}
	if len(lex.Codes) < vsLexiconFloor {
		return nil, fmt.Errorf("the refusal lexicon resolved only %d code(s); fak's closed vocabulary is "+
			"far larger, so the REFUSES column would render EMPTY and read as \"these verbs refuse "+
			"nothing\" — repair the lexicon reader, do not lower the floor", len(lex.Codes))
	}

	top := vsTopRows(p)
	if len(top) < vsVerbFloor {
		return nil, fmt.Errorf("read only %d verb(s) out of %s's dispatch switches; the surface is far "+
			"bigger, so the reader no longer matches the dispatch shape", len(top), vsCmdDir+"/main.go")
	}

	s := &VerbSurface{Lexicon: lex, Files: len(p.files)}
	seen := map[string]bool{}
	var add func(prefix []string, r vsRow, avoid map[string]bool)
	add = func(prefix []string, r vsRow, avoid map[string]bool) {
		words := append(append([]string{}, prefix...), r.name)
		key := strings.Join(words, " ")
		if seen[key] {
			return
		}
		seen[key] = true
		s.Leaves = append(s.Leaves, SurfaceLeaf{
			Words:   words,
			Aliases: r.aliases,
			Fn:      r.fn,
			Origin:  r.origin,
			Planned: r.fn == "",
		})
		if r.fn == "" || len(words) >= vsMaxWords {
			return
		}
		next := map[string]bool{r.fn: true}
		for k := range avoid {
			next[k] = true
		}
		for _, sub := range vsSubRows(p, r.fn, avoid) {
			add(words, sub, next)
		}
	}

	// Every top-level handler is off-limits to every OTHER verb's sub-verb walk:
	// without it, a dispatcher that calls a sibling's handler mints that sibling's
	// sub-verbs a second time under the wrong parent.
	topHandlers := map[string]bool{}
	for _, r := range top {
		if r.fn != "" {
			topHandlers[r.fn] = true
		}
	}
	for _, r := range top {
		add(nil, r, topHandlers)
	}

	handlers := map[string]bool{}
	for _, l := range s.Leaves {
		if l.Fn != "" {
			handlers[l.Fn] = true
		}
	}
	for i := range s.Leaves {
		l := &s.Leaves[i]
		if l.IsVerb() {
			l.Tier = tierFor(l.Verb())
			if v, ok := manifestVerbByName(l.Verb()); ok {
				l.Synopsis = v.Synopsis
			}
		}
		if l.Synopsis == "" {
			l.Synopsis = vsDerivedSynopsis(p, *l)
		}
		l.InHelp = vsInHelp(p.helpText, l.Words)
		if l.Fn == "" {
			l.Pre = SurfacePre{State: PreNotApplicable,
				Notes: []string{"dispatched with no handler function this reader can name"}}
			continue
		}
		reach := vsReachable(p, l.Fn, handlers)
		l.Packages = vsInternalPackages(p, reach)
		l.Pre = vsPreconditions(p, *l, reach, lex)
	}
	sort.SliceStable(s.Leaves, func(i, j int) bool { return s.Leaves[i].Key() < s.Leaves[j].Key() })
	return s, nil
}

// vsVerbFloor is a floor on the derived top-level verb count, for the same reason
// vsFileFloor is a floor on the input: an extractor that stops seeing the dispatch
// switch derives zero verbs, has nothing to render, and produces a page that reads as
// an accurate map of a much smaller tool.
const vsVerbFloor = 120

// vsTopRows derives every top-level verb from the dispatch switches in cmd/fak.
//
// It does not name those switches. A switch is a verb dispatch switch when its tag is
// os.Args[1], or a parameter some caller passes os.Args[1] into, or a local bound to
// one of those — which is a property of the source, not of a list kept here.
func vsTopRows(p *vsPkg) []vsRow {
	var out []vsRow
	seen := map[string]bool{}
	for _, name := range p.names {
		decl := p.funcs[name]
		tok := vsTokenSet(decl, p.verbTokenParams[name], vsIsOSArgs1)
		if len(tok) == 0 && !vsBodyMentionsOSArgs1(decl) {
			continue
		}
		for _, r := range vsDispatchRows(p, name, func(e ast.Expr) bool {
			return vsIsOSArgs1(e) || vsIdentIn(e, tok)
		}) {
			if seen[r.name] {
				continue
			}
			seen[r.name] = true
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

// vsSubRows derives one verb's sub-verbs: the first function reachable from its
// handler (breadth first, the handler itself included) that dispatches on its own
// residual argv[0].
//
// Taking only the FIRST such function is what keeps depth honest. `fak worktree`
// routes through cmdWorktreeVerb (which switches on argv[0] for `worker`) into
// cmdWorktreeWorker (which switches on argv[0] for `prepare`/`land`/`reap`); folding
// both into one row set would report `prepare` as a sub-verb of `worktree`. The
// recursion reaches it at the right depth instead.
func vsSubRows(p *vsPkg, fn string, avoid map[string]bool) []vsRow {
	seen := map[string]bool{fn: true}
	frontier := []string{fn}
	for depth := 0; depth < vsReachDepth && len(frontier) > 0; depth++ {
		var next []string
		for _, cur := range frontier {
			decl, ok := p.funcs[cur]
			if !ok {
				continue
			}
			slice := vsArgSliceParam(decl)
			if slice != "" {
				isZero := func(e ast.Expr) bool { return vsIsIndexZero(e, slice) }
				tok := vsTokenSet(decl, nil, isZero)
				rows := vsDispatchRows(p, cur, func(e ast.Expr) bool {
					return isZero(e) || vsIdentIn(e, tok)
				})
				if len(rows) > 0 {
					return rows
				}
			}
			for _, callee := range vsCallees(p, decl) {
				if seen[callee] || avoid[callee] {
					continue
				}
				seen[callee] = true
				next = append(next, callee)
			}
		}
		frontier = next
	}
	return nil
}

// vsTokenSet names the locals a dispatcher binds the verb token to, so
// `sub, rest := argv[0], argv[1:]` followed by `switch sub` reads the same as
// `switch argv[0]` — the shape `fak accounts` uses for its ~30 sub-verbs.
//
// Only a name assigned EXACTLY ONCE, and only ever from the token, is accepted. A
// rebound name has stopped being a synonym and minting sub-verbs off it would invent
// command paths the binary does not have. That is deliberately the conservative
// direction: a missed alias understates the page and the floors catch it, whereas an
// invented one documents a verb nobody can run.
func vsTokenSet(decl *ast.FuncDecl, params map[string]bool, isToken func(ast.Expr) bool) map[string]bool {
	out := map[string]bool{}
	for name := range params {
		out[name] = true
	}
	if decl.Body == nil {
		return out
	}
	fromToken := map[string]int{}
	fromElse := map[string]bool{}
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for i, lhs := range as.Lhs {
			id, ok := lhs.(*ast.Ident)
			if !ok || id.Name == "_" {
				continue
			}
			if isToken(as.Rhs[i]) || (params != nil && vsIdentIn(as.Rhs[i], params)) {
				fromToken[id.Name]++
				continue
			}
			fromElse[id.Name] = true
		}
		return true
	})
	for name, n := range fromToken {
		if n == 1 && !fromElse[name] {
			out[name] = true
		}
	}
	return out
}

func vsIdentIn(e ast.Expr, set map[string]bool) bool {
	if len(set) == 0 {
		return false
	}
	id, ok := e.(*ast.Ident)
	return ok && set[id.Name]
}

func vsBodyMentionsOSArgs1(decl *ast.FuncDecl) bool {
	found := false
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		if found {
			return false
		}
		if e, ok := n.(ast.Expr); ok && vsIsOSArgs1(e) {
			found = true
			return false
		}
		return true
	})
	return found
}

// vsDispatchRows reads the `case "x":` labels of every switch in fn whose tag is the
// verb token, plus every `token == "x"` comparison, and pairs each with the handler
// its clause calls.
//
// Scoping to the NAMED function matters: cmd/fak/worktree_worker.go holds two argv[0]
// switches, and a file-wide walk would report `prepare` and `land` — leaves of
// `fak worktree worker` — as leaves of `fak worktree`.
func vsDispatchRows(p *vsPkg, fn string, isToken func(ast.Expr) bool) []vsRow {
	decl, ok := p.funcs[fn]
	if !ok || decl.Body == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []vsRow
	add := func(labels []ast.Expr, body ast.Node, origin string) {
		var words []string
		for _, e := range labels {
			s, ok := vsStringValue(e)
			if ok && s != "" {
				words = append(words, s)
			}
		}
		// A clause spelled only in flag form (`case "-h", "--help":`) or only as help
		// is not a verb; a clause whose FIRST spelling is a flag but which also
		// carries a bare word (`case "-h", "--help", "help":`) is the `help` verb
		// under two flag aliases.
		var name string
		var aliases []string
		for _, w := range words {
			if strings.HasPrefix(w, "-") {
				aliases = append(aliases, w)
				continue
			}
			if name == "" {
				name = w
				continue
			}
			aliases = append(aliases, w)
		}
		if name == "" || seen[name] {
			return
		}
		seen[name] = true
		out = append(out, vsRow{name: name, aliases: aliases,
			fn: vsFirstHandler(p, body, fn), origin: origin})
	}
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.SwitchStmt:
			if n.Tag == nil || !isToken(n.Tag) {
				return true
			}
			for _, stmt := range n.Body.List {
				clause, ok := stmt.(*ast.CaseClause)
				if !ok || len(clause.List) == 0 {
					continue
				}
				add(clause.List, clause, "`case` arm of the dispatch switch in "+fn+"()")
			}
		case *ast.IfStmt:
			if lits := vsTokenLiterals(n.Cond, isToken); len(lits) > 0 {
				add(lits, n.Body, "`if` arm of the dispatch in "+fn+"()")
			}
		}
		return true
	})
	return out
}

// vsTokenLiterals collects the literal in every `<token> == "x"` inside cond.
func vsTokenLiterals(cond ast.Expr, isToken func(ast.Expr) bool) []ast.Expr {
	var out []ast.Expr
	ast.Inspect(cond, func(n ast.Node) bool {
		be, ok := n.(*ast.BinaryExpr)
		if !ok || be.Op != token.EQL {
			return true
		}
		if isToken(be.X) {
			out = append(out, be.Y)
		}
		return true
	})
	return out
}

// vsFirstHandler names the package-main function a dispatch clause hands off to.
// `cmd`- and `run`-prefixed names win over any other callee, so a clause that stamps a
// usage record or wraps the call in an observer still resolves to the real handler.
func vsFirstHandler(p *vsPkg, body ast.Node, self string) string {
	var cmdName, runName, any string
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name == self {
			return true
		}
		if _, known := p.funcs[id.Name]; !known {
			return true
		}
		if cmdName == "" && strings.HasPrefix(id.Name, "cmd") {
			cmdName = id.Name
		}
		if runName == "" && strings.HasPrefix(id.Name, "run") {
			runName = id.Name
		}
		if any == "" {
			any = id.Name
		}
		return true
	})
	switch {
	case cmdName != "":
		return cmdName
	case runName != "":
		return runName
	}
	return any
}

func vsCallees(p *vsPkg, decl *ast.FuncDecl) []string {
	var out []string
	seen := map[string]bool{}
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || seen[id.Name] {
			return true
		}
		if _, known := p.funcs[id.Name]; !known {
			return true
		}
		seen[id.Name] = true
		out = append(out, id.Name)
		return true
	})
	return out
}

// ---------------------------------------------------------------- reachability

// vsReachable returns fn plus every package-main function it can call, stopping at any
// function that is another leaf's handler — that leaf owns its refusals, and crediting
// them here would report `fak accounts` as refusing what `fak accounts add` refuses.
func vsReachable(p *vsPkg, fn string, handlers map[string]bool) []string {
	seen := map[string]bool{fn: true}
	order := []string{fn}
	frontier := []string{fn}
	for depth := 0; depth < vsReachDepth && len(frontier) > 0; depth++ {
		var next []string
		for _, cur := range frontier {
			decl, ok := p.funcs[cur]
			if !ok {
				continue
			}
			for _, callee := range vsCallees(p, decl) {
				if seen[callee] {
					continue
				}
				if callee != fn && handlers[callee] {
					continue
				}
				seen[callee] = true
				order = append(order, callee)
				next = append(next, callee)
			}
		}
		frontier = next
	}
	return order
}

// vsInternalPackages names the internal/ packages a leaf's implementation reaches,
// resolved through each file's OWN import map so an import a file carries for some
// other verb is not credited here.
func vsInternalPackages(p *vsPkg, reach []string) []string {
	seen := map[string]bool{}
	for _, fn := range reach {
		decl, ok := p.funcs[fn]
		if !ok {
			continue
		}
		imports := p.imports[p.fileOf[fn]]
		ast.Inspect(decl.Body, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			id, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			if path, ok := imports[id.Name]; ok && strings.Contains(path, "/internal/") {
				seen[path[strings.Index(path, "internal/"):]] = true
			}
			return true
		})
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------- prose + help

// vsStringValue evaluates a string literal, including the `"a" + "b"` chains fak's
// long help strings are written as.
func vsStringValue(e ast.Expr) (string, bool) {
	switch v := e.(type) {
	case *ast.BasicLit:
		if v.Kind != token.STRING {
			return "", false
		}
		s, err := strconv.Unquote(v.Value)
		return s, err == nil
	case *ast.BinaryExpr:
		if v.Op != token.ADD {
			return "", false
		}
		l, lok := vsStringValue(v.X)
		r, rok := vsStringValue(v.Y)
		return l + r, lok && rok
	}
	return "", false
}

func vsStringLiterals(f *ast.File) []string {
	var out []string
	ast.Inspect(f, func(n ast.Node) bool {
		lit, ok := n.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		if s, err := strconv.Unquote(lit.Value); err == nil {
			out = append(out, s)
		}
		return true
	})
	return out
}

// vsInHelp reports whether fak's own help wall mentions this command path.
//
// A verb is present when the wall names `fak <verb>`. A sub-verb is present when the
// block introduced by its parent's `fak <verb>` line — that line plus the indented
// continuation under it — carries the sub-verb as a standalone word, which is how the
// wall writes them (`fak launch [install|uninstall|default|…]`). The comparison is
// source-against-source: both sides of it are read out of cmd/fak, so a drift is a
// drift in the tree and never an artifact of which binary happened to be on PATH.
func vsInHelp(help string, words []string) bool {
	blocks := vsHelpBlocks(help, words[0])
	if len(blocks) == 0 {
		return false
	}
	if len(words) == 1 {
		return true
	}
	for _, b := range blocks {
		all := true
		for _, w := range words[1:] {
			if !vsHasWord(b, w) {
				all = false
				break
			}
		}
		if all {
			return true
		}
	}
	return false
}

// vsHelpBlocks returns each `fak <verb>` entry of the help wall: the line that names
// it plus every following line more indented than it.
func vsHelpBlocks(help, verb string) []string {
	lines := strings.Split(help, "\n")
	var out []string
	for i, ln := range lines {
		trimmed := strings.TrimLeft(ln, " \t")
		if !strings.HasPrefix(trimmed, "fak "+verb) {
			continue
		}
		rest := trimmed[len("fak "+verb):]
		if rest != "" && !strings.ContainsAny(rest[:1], " \t") {
			continue // `fak commitfoo` is not `fak commit`
		}
		indent := len(ln) - len(trimmed)
		var b strings.Builder
		b.WriteString(rest)
		for _, next := range lines[i+1:] {
			t := strings.TrimLeft(next, " \t")
			if strings.TrimSpace(next) == "" {
				break
			}
			if len(next)-len(t) <= indent && strings.HasPrefix(t, "fak ") {
				break
			}
			b.WriteString("\n" + t)
		}
		out = append(out, b.String())
	}
	return out
}

// vsHasWord reports whether s carries word as a standalone token, so `list` matches
// `[list|show]` and `--listen` does not.
func vsHasWord(s, word string) bool {
	for i := 0; i+len(word) <= len(s); i++ {
		if s[i:i+len(word)] != word {
			continue
		}
		if i > 0 && vsWordByte(s[i-1]) {
			continue
		}
		if i+len(word) < len(s) && vsWordByte(s[i+len(word)]) {
			continue
		}
		return true
	}
	return false
}

func vsWordByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '-' || b == '_'
}

// vsDerivedSynopsis finds a purpose for a path whose `case "x":` label cannot carry
// one. Three sources, in order of authority, and each is prose an author wrote ABOUT
// this path rather than text found near it: the help wall's own entry, the handler's
// doc comment, then nothing.
//
// Falling through to "" is deliberate and renders as an explicit absence. An invented
// purpose — the sub-verb name re-spelled as a sentence — would be worse than a blank,
// because a reader cannot tell a generator's guess from a source claim, and this
// page's whole argument is that it only repeats claims.
func vsDerivedSynopsis(p *vsPkg, l SurfaceLeaf) string {
	if s := vsHelpSynopsis(p.helpText, l.Words); s != "" {
		return s
	}
	decl, ok := p.funcs[l.Fn]
	if !ok || decl.Doc == nil {
		return ""
	}
	text := vsCollapse(decl.Doc.Text())
	for _, stutter := range []string{
		l.Fn + " is the ", l.Fn + " is ", l.Fn + " runs ", l.Fn + " implements ", l.Fn + " ",
	} {
		if strings.HasPrefix(text, stutter) {
			text = strings.TrimPrefix(text, stutter)
			break
		}
	}
	return vsFirstSentence(text)
}

// vsHelpSynopsis pulls the parenthesised gloss the help wall writes under a verb.
func vsHelpSynopsis(help string, words []string) string {
	if len(words) != 1 {
		return ""
	}
	for _, b := range vsHelpBlocks(help, words[0]) {
		i := strings.Index(b, "(")
		if i < 0 {
			continue
		}
		gloss := vsCollapse(b[i+1:])
		gloss = strings.TrimSuffix(gloss, ")")
		if s := vsFirstSentence(gloss); s != "" {
			return s
		}
	}
	return ""
}

func vsCollapse(s string) string { return strings.Join(strings.Fields(s), " ") }

// vsFirstSentence cuts at the first sentence end that is not an abbreviation or a
// version number, then hard-caps the length a table cell can hold.
func vsFirstSentence(s string) string {
	s = vsCollapse(s)
	if s == "" {
		return ""
	}
	for i := 0; i+1 < len(s); i++ {
		if s[i] != '.' && s[i] != ';' {
			continue
		}
		if s[i+1] != ' ' {
			continue
		}
		if i > 0 && s[i-1] >= '0' && s[i-1] <= '9' {
			continue
		}
		// A single letter before the dot is an abbreviation, not a sentence end:
		// without this, "raw artifacts (e.g. two engines…" cuts to "raw artifacts (e.g".
		if i >= 2 && vsIsLetter(s[i-1]) && (s[i-2] == ' ' || s[i-2] == '.' || s[i-2] == '(') {
			continue
		}
		s = s[:i]
		break
	}
	if len(s) > 200 {
		s = strings.TrimSpace(s[:200]) + "…"
	}
	return vsEscapeCell(s)
}

func vsIsLetter(b byte) bool { return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' }

// vsEscapeCell makes a string safe inside a markdown table cell.
func vsEscapeCell(s string) string {
	s = strings.ReplaceAll(s, "|", `\|`)
	return strings.ReplaceAll(s, "\n", " ")
}

// Markdown renders the generated operator page. Coverage counts are in the
// artifact itself so a reader can distinguish known preconditions from gaps.
func (s *VerbSurface) Markdown() []byte {
	var b strings.Builder
	b.WriteString("# fak verb surface (generated)\n\n")
	b.WriteString("> Generated from Go source by `go run ./cmd/verbsdoc`; do not edit.\n\n")
	unverified := 0
	missingHelp := 0
	for _, l := range s.Leaves {
		if l.Pre.State == PreUnverified {
			unverified++
		}
		if !l.InHelp {
			missingHelp++
		}
	}
	fmt.Fprintf(&b, "parsed files: %d  \nrows: %d  \nunverified rows: %d / %d  \nsource-only rows absent from help: %d\n\n", s.Files, len(s.Leaves), unverified, len(s.Leaves), missingHelp)
	b.WriteString("| VERB | PURPOSE | IMPLEMENTS | DOC | PRECONDITION | REFUSES | HELP |\n")
	b.WriteString("|---|---|---|---|---|---|---|\n")
	for _, l := range s.Leaves {
		impl := l.Fn
		if len(l.Packages) > 0 {
			impl += " / " + strings.Join(l.Packages, ", ")
		}
		codes := "—"
		if len(l.Pre.Codes) > 0 {
			codes = "`" + strings.Join(l.Pre.Codes, "`, `") + "`"
		}
		help := "yes"
		if !l.InHelp {
			help = "**SOURCE ONLY**"
		}
		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s | %s | %s |\n", l.Path(), vsEscapeCell(l.Synopsis), vsEscapeCell(impl), vsEscapeCell(l.Origin), l.Pre.State.String(), codes, help)
	}
	return []byte(b.String())
}
