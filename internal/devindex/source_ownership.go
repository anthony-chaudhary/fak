package devindex

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const DevHandoffManifestPath = "internal/devhandoff/commands_gen.go"

// SourceClass is the fail-closed classification of one executable source
// component. New motion out of runtime is admitted only from DevOnly; already
// dev-owned commands remain inventoryable even when they use dev-only mechanisms.
type SourceClass string

const (
	SourceRuntime   SourceClass = "runtime"
	SourceDevOnly   SourceClass = "dev-only"
	SourceMixed     SourceClass = "mixed"
	SourceHazardous SourceClass = "hazardous"
)

// SourceOwnership is the generated authority shared by fak-dev dispatch and
// runtime's refusal/handoff boundary.
type SourceOwnership struct {
	Name           string
	Aliases        []string
	Owner          CommandOwner
	Handler        string
	SourceOrigin   string
	DispatchTarget string
	Class          SourceClass
}

// ExtractDevSourceOwnership derives ownership from the real fak-dev switch and
// resolves each handler back to its declaration using the same AST package model
// as verb-surface extraction. Non-dev cases are ignored; ambiguous or hazardous
// dev cases fail closed instead of entering the generated runtime boundary.
func ExtractDevSourceOwnership(root string) ([]SourceOwnership, error) {
	dispatchPkg, err := vsLoadPackage(root, "cmd/fak-dev", 1)
	if err != nil {
		return nil, err
	}
	implPkg, err := vsLoadPackage(root, "internal/devcmd", 1)
	if err != nil {
		return nil, err
	}
	run := dispatchPkg.funcs["run"]
	if run == nil {
		return nil, fmt.Errorf("cmd/fak-dev run handler not found")
	}
	var dispatch *ast.SwitchStmt
	ast.Inspect(run.Body, func(n ast.Node) bool {
		if dispatch != nil {
			return false
		}
		sw, ok := n.(*ast.SwitchStmt)
		if ok {
			dispatch = sw
			return false
		}
		return true
	})
	if dispatch == nil {
		return nil, fmt.Errorf("cmd/fak-dev run has no dispatch switch")
	}

	var rows []SourceOwnership
	seen := map[string]bool{}
	for _, stmt := range dispatch.Body.List {
		clause, ok := stmt.(*ast.CaseClause)
		if !ok || len(clause.List) == 0 {
			continue
		}
		var names []string
		for _, expr := range clause.List {
			lit, ok := expr.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return nil, fmt.Errorf("%s: non-literal fak-dev command case is hazardous", dispatchPkg.site(dispatchPkg.fileOf["run"], expr.Pos()))
			}
			name, err := strconv.Unquote(lit.Value)
			if err != nil {
				return nil, err
			}
			names = append(names, name)
		}
		devNames := names[:0]
		for _, name := range names {
			if tier, known := TierOf(name); known && tier == TierDev {
				devNames = append(devNames, name)
			}
		}
		if len(devNames) == 0 {
			continue
		}
		if len(devNames) != len(names) {
			return nil, fmt.Errorf("fak-dev case %q mixes dev and non-dev spellings", strings.Join(names, ","))
		}
		handler, source, handlerPkg, handlerName, err := resolveDevCaseHandler(clause, dispatchPkg, implPkg)
		if err != nil {
			return nil, fmt.Errorf("fak-dev case %q: %w", strings.Join(names, ","), err)
		}
		class := classifyExtractionCandidate(handlerPkg, handlerName, source, OwnerDev)
		row := SourceOwnership{
			Name:           names[0],
			Aliases:        append([]string(nil), names[1:]...),
			Owner:          OwnerDev,
			Handler:        handler,
			SourceOrigin:   source,
			DispatchTarget: "fak-dev",
			Class:          class,
		}
		for _, name := range names {
			if seen[name] {
				return nil, fmt.Errorf("duplicate fak-dev command %q", name)
			}
			seen[name] = true
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })
	return rows, nil
}

// classifyExtractionCandidate applies motion hazards only at the runtime
// ownership boundary. An implementation already rooted in internal/devcmd or
// cmd/fak-dev is not a candidate for extraction from runtime, so mechanisms such
// as an embedded dev policy do not make its exhaustive handoff row disappear.
func classifyExtractionCandidate(pkg *vsPkg, handler, source string, owner CommandOwner) SourceClass {
	if owner == OwnerDev || !strings.HasPrefix(filepath.ToSlash(source), "cmd/fak/") {
		return SourceDevOnly
	}
	if len(reachableComponentHazards(pkg, handler)) != 0 {
		return SourceHazardous
	}
	return SourceDevOnly
}

func resolveDevCaseHandler(clause *ast.CaseClause, dispatchPkg, implPkg *vsPkg) (string, string, *vsPkg, string, error) {
	type match struct {
		name string
		path string
		pkg  *vsPkg
		fn   string
		// inGuard marks a call lexically inside an if statement in the case
		// body — a subcommand route (`issue inventory`) that early-returns,
		// not the case's owning handler. The unconditional call owns the row.
		inGuard bool
	}
	var matches []match
	// Collect every call nested under an IfStmt so a match can be classified
	// as a guarded subcommand route instead of the case's handler.
	guardCalls := map[*ast.CallExpr]bool{}
	ast.Inspect(clause, func(n ast.Node) bool {
		if _, ok := n.(*ast.IfStmt); !ok {
			return true
		}
		ast.Inspect(n, func(inner ast.Node) bool {
			if call, ok := inner.(*ast.CallExpr); ok {
				guardCalls[call] = true
			}
			return true
		})
		return true
	})
	ast.Inspect(clause, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch fun := call.Fun.(type) {
		case *ast.SelectorExpr:
			pkg, ok := fun.X.(*ast.Ident)
			if !ok || pkg.Name != "devcmd" || implPkg.funcs[fun.Sel.Name] == nil {
				return true
			}
			file := implPkg.fileOf[fun.Sel.Name]
			matches = append(matches, match{"devcmd." + fun.Sel.Name, implPkg.pathOf[file], implPkg, fun.Sel.Name, guardCalls[call]})
		case *ast.Ident:
			if dispatchPkg.funcs[fun.Name] == nil {
				return true
			}
			file := dispatchPkg.fileOf[fun.Name]
			matches = append(matches, match{fun.Name, dispatchPkg.pathOf[file], dispatchPkg, fun.Name, guardCalls[call]})
		}
		return true
	})
	primary := matches
	if len(matches) > 1 {
		var unguarded []match
		for _, m := range matches {
			if !m.inGuard {
				unguarded = append(unguarded, m)
			}
		}
		// Filter only when the guarded/unguarded split isolates exactly one
		// owner; otherwise the case is genuinely ambiguous and must refuse.
		if len(unguarded) == 1 {
			primary = unguarded
		}
	}
	if len(primary) != 1 {
		return "", "", nil, "", fmt.Errorf("expected exactly one resolvable handler, found %d", len(matches))
	}
	return primary[0].name, primary[0].path, primary[0].pkg, primary[0].fn, nil
}

// reachableComponentHazards walks the same-package declaration call component,
// scanning the complete source file for every reached declaration. Direct calls
// to package functions add their declarations transitively. Selector calls are
// import/object boundaries and are deliberately not followed without Go type
// information, so a leaf's private implementation cannot contaminate its caller.
func reachableComponentHazards(pkg *vsPkg, handler string) []string {
	if pkg.funcs[handler] == nil || pkg.fileOf[handler] == nil {
		return []string{"unresolved-handler"}
	}
	seenFuncs := map[string]bool{}
	seenFiles := map[*ast.File]bool{}
	queue := []string{handler}
	var hazards []string
	for len(queue) != 0 {
		name := queue[0]
		queue = queue[1:]
		if seenFuncs[name] {
			continue
		}
		seenFuncs[name] = true
		file := pkg.fileOf[name]
		if !seenFiles[file] {
			seenFiles[file] = true
			path := pkg.pathOf[file]
			for _, hazard := range sourceHazards(file) {
				hazards = append(hazards, hazard+" ("+path+")")
			}
		}
		ast.Inspect(pkg.funcs[name].Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok || pkg.funcs[id.Name] == nil {
				return true
			}
			if !seenFuncs[id.Name] {
				queue = append(queue, id.Name)
			}
			return true
		})
	}
	sort.Strings(hazards)
	return compactStrings(hazards)
}

func sourceHazards(file *ast.File) []string {
	var hazards []string
	for _, imp := range file.Imports {
		path, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			continue
		}
		switch path {
		case "C":
			hazards = append(hazards, "cgo")
		case "reflect":
			hazards = append(hazards, "reflection")
		}
	}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Recv == nil && fn.Name.Name == "init" {
			hazards = append(hazards, "init")
		}
	}
	for _, group := range file.Comments {
		for _, comment := range group.List {
			if strings.Contains(comment.Text, "//go:linkname") {
				hazards = append(hazards, "linkname")
			}
			if strings.Contains(comment.Text, "//go:embed") {
				hazards = append(hazards, "embed")
			}
		}
	}
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Executable" {
			return true
		}
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == "os" {
			hazards = append(hazards, "self-exec")
		}
		return true
	})
	sort.Strings(hazards)
	return compactStrings(hazards)
}

func compactStrings(in []string) []string {
	out := in[:0]
	for _, s := range in {
		if len(out) == 0 || out[len(out)-1] != s {
			out = append(out, s)
		}
	}
	return out
}

// RenderDevHandoffManifest renders the one checked-in command authority consumed
// by runtime fak. The rows come from the fak-dev AST; editing the dispatcher
// without regenerating this file is a deterministic test failure.
func RenderDevHandoffManifest(root string) ([]byte, error) {
	rows, err := ExtractDevSourceOwnership(root)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	buf.WriteString("// Code generated by `fak-dev index ownership --write-manifest`; DO NOT EDIT.\n\n")
	buf.WriteString("package devhandoff\n\n")
	buf.WriteString("var Commands = []Command{\n")
	for _, row := range rows {
		fmt.Fprintf(&buf, "\t{Name: %q,", row.Name)
		if len(row.Aliases) != 0 {
			buf.WriteString(" Aliases: []string{")
			for i, alias := range row.Aliases {
				if i != 0 {
					buf.WriteString(", ")
				}
				fmt.Fprintf(&buf, "%q", alias)
			}
			buf.WriteString("},")
		}
		fmt.Fprintf(&buf, " Owner: %q, Handler: %q, SourceOrigin: %q, DispatchTarget: %q, SourceClass: %q},\n",
			row.Owner, row.Handler, row.SourceOrigin, row.DispatchTarget, row.Class)
	}
	buf.WriteString("}\n")
	return format.Source(buf.Bytes())
}

func WriteDevHandoffManifest(root string) error {
	data, err := RenderDevHandoffManifest(root)
	if err != nil {
		return err
	}
	path := filepath.Join(root, filepath.FromSlash(DevHandoffManifestPath))
	return os.WriteFile(path, data, 0o644)
}
