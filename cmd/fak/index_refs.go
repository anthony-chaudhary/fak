package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func indexRefs(stdout, stderr io.Writer, root string, args []string, asJSON bool, limit int) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "fak index refs: needs exactly one target <pkg>.<Symbol>")
		return 2
	}
	target, err := devindex.ParseSymbolID(args[0])
	if err != nil {
		fmt.Fprintf(stderr, "fak index refs: %v\n", err)
		return 2
	}
	pkgs, err := indexRefsListPackages(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak index refs: %v\n", err)
		return 1
	}
	graph := make([]devindex.PackageImports, 0, len(pkgs))
	for _, p := range pkgs {
		graph = append(graph, devindex.PackageImports{
			ImportPath:   p.ImportPath,
			Imports:      append([]string(nil), p.Imports...),
			TestImports:  append([]string(nil), p.TestImports...),
			XTestImports: append([]string(nil), p.XTestImports...),
		})
	}
	refs := collectIndexSymbolRefs(pkgs, target)
	result := devindex.BlastRadius(graph, refs, target)
	if limit > 0 && len(result.Packages) > limit {
		result.Packages = result.Packages[:limit]
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, result, "fak index refs")
	}
	if len(result.Packages) == 0 {
		fmt.Fprintf(stdout, "no dependents for %s\n", target.String())
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, row := range result.Packages {
		kind := "transitive"
		if row.Direct {
			kind = "direct"
		}
		fmt.Fprintf(tw, "%s\t%s\tdistance=%d\n", row.ImportPath, kind, row.Distance)
	}
	return flushTab(tw, stderr, "fak index refs")
}

func indexRefsListPackages(root string) ([]goPkg, error) {
	cmd := exec.Command("go", "list", "-e", "-json", "./...")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	runErr := cmd.Run()
	pkgs, err := parseGoListPackages(&out)
	if err != nil {
		return nil, err
	}
	if len(pkgs) == 0 {
		if runErr != nil {
			return nil, fmt.Errorf("go list produced no packages: %w", runErr)
		}
		return nil, fmt.Errorf("go list produced no packages")
	}
	return pkgs, nil
}

func parseGoListPackages(r io.Reader) ([]goPkg, error) {
	var pkgs []goPkg
	dec := json.NewDecoder(r)
	for {
		var p goPkg
		if err := dec.Decode(&p); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("parsing go list json: %w", err)
		}
		pkgs = append(pkgs, p)
	}
	return pkgs, nil
}

func collectIndexSymbolRefs(pkgs []goPkg, target devindex.SymbolID) []devindex.SymbolReference {
	targetName := packageNameForTarget(pkgs, target.Package)
	var refs []devindex.SymbolReference
	seen := map[string]bool{}
	for _, p := range pkgs {
		for _, file := range packageGoFiles(p) {
			names, dotImport := importedNamesForTarget(file.path, target.Package, targetName)
			if len(names) == 0 && !dotImport {
				continue
			}
			if fileReferencesSymbol(file.path, names, dotImport, target.Symbol) {
				key := p.ImportPath + "\x00" + strconv.FormatBool(file.test)
				if seen[key] {
					continue
				}
				seen[key] = true
				refs = append(refs, devindex.SymbolReference{
					FromPackage: p.ImportPath,
					Target:      target,
					Test:        file.test,
				})
			}
		}
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].FromPackage != refs[j].FromPackage {
			return refs[i].FromPackage < refs[j].FromPackage
		}
		return !refs[i].Test && refs[j].Test
	})
	return refs
}

type indexPackageFile struct {
	path string
	test bool
}

func packageGoFiles(p goPkg) []indexPackageFile {
	var out []indexPackageFile
	add := func(names []string, test bool) {
		for _, name := range names {
			if p.Dir == "" || name == "" {
				continue
			}
			out = append(out, indexPackageFile{path: filepath.Join(p.Dir, name), test: test})
		}
	}
	add(p.GoFiles, false)
	add(p.TestGoFiles, true)
	add(p.XTestGoFiles, true)
	return out
}

func packageNameForTarget(pkgs []goPkg, importPath string) string {
	for _, p := range pkgs {
		if p.ImportPath == importPath && p.Name != "" {
			return p.Name
		}
	}
	base := importPath
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	return strings.ReplaceAll(base, "-", "_")
}

func importedNamesForTarget(path, targetPackage, defaultName string) (map[string]bool, bool) {
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
	if err != nil {
		return nil, false
	}
	names := map[string]bool{}
	dotImport := false
	for _, spec := range file.Imports {
		imp, err := strconv.Unquote(spec.Path.Value)
		if err != nil || imp != targetPackage {
			continue
		}
		switch {
		case spec.Name == nil:
			if defaultName != "" {
				names[defaultName] = true
			}
		case spec.Name.Name == ".":
			dotImport = true
		case spec.Name.Name == "_":
			continue
		default:
			names[spec.Name.Name] = true
		}
	}
	if len(names) == 0 {
		names = nil
	}
	return names, dotImport
}

func fileReferencesSymbol(path string, importNames map[string]bool, dotImport bool, symbol string) bool {
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		return false
	}
	found := false
	ast.Inspect(file, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.SelectorExpr:
			if x.Sel == nil || x.Sel.Name != symbol {
				return true
			}
			id, ok := x.X.(*ast.Ident)
			if ok && importNames[id.Name] {
				found = true
				return false
			}
		case *ast.Ident:
			if dotImport && x.Name == symbol {
				found = true
				return false
			}
		}
		return true
	})
	return found
}
