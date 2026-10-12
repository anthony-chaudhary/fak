package compute

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// fak-test:runtime fast est=3s — parses this package's non-test sources; no device,
// no cgo, no build tag.

const vulkanDeviceMutex = "vulkanMu"

// muExternal types a value whose type lives in another package; its methods cannot
// reach this package's unexported mutex.
const muExternal = "."

// TestVulkanMuIsNeverReacquiredWhileHeld is the device-free guard for the vulkanMu
// self-deadlock class. sync.Mutex is not reentrant, so any call made while vulkanMu
// is held that can reach vulkanMu.Lock() hangs the process forever: MatMul probing
// the performance-query surface under the lock wedged `fak serve` warmup at layer 0
// once a kernel observer was attached. Every source file is parsed regardless of
// build tag, so the vulkan-tagged cgo files are audited on hosts without a device.
func TestVulkanMuIsNeverReacquiredWhileHeld(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || filepath.Ext(name) != ".go" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	audit := newMuAudit(fset, files)
	if audit.lockSites == 0 {
		t.Fatalf("no %s.Lock() call found in %d files; the audit would pass vacuously", vulkanDeviceMutex, len(files))
	}
	if got := audit.reacquisitions(); len(got) != 0 {
		lines := make([]string, len(got))
		for i, r := range got {
			lines[i] = r.String()
		}
		t.Fatalf("%d call(s) made while %s is held can lock it again (self-deadlock):\n%s",
			len(got), vulkanDeviceMutex, strings.Join(lines, "\n"))
	}
}

// TestVulkanMuLockOrderAuditCatchesReentry pins the audit itself on the exact
// regression shape (a probe reached through an interface while the lock is held)
// and on the two sanctioned shapes: probe before Lock, and call after an Unlock
// inside the deferred closure.
func TestVulkanMuLockOrderAuditCatchesReentry(t *testing.T) {
	const src = `package fixture

import "sync"

var vulkanMu sync.Mutex

type probe interface{ Supported() bool }
type dev struct{ inner *leaf }
type leaf struct{}

func (l *leaf) Validate() bool { return true }
func (d *dev) Supported() bool { vulkanMu.Lock(); defer vulkanMu.Unlock(); return true }
func observe(p probe) bool    { return p.Supported() }

func (d *dev) Reentrant() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	_ = observe(d)
}

func (d *dev) ProbeFirst() {
	ok := observe(d)
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	_ = ok && d.inner.Validate()
}

func (d *dev) ReleaseThenCall() {
	vulkanMu.Lock()
	defer func() {
		vulkanMu.Unlock()
		_ = observe(d)
	}()
}

func (d *dev) branchLocked() {
	if d.Supported() {
		return
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "fixture.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range newMuAudit(fset, []*ast.File{f}).reacquisitions() {
		got = append(got, r.Caller+"->"+r.Callee)
	}
	want := []string{"dev.Reentrant->observe", "dev.branchLocked->dev.Supported"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("reacquisitions = %q; want %q", got, want)
	}
}

type muFunc struct {
	name, recv string
	decl       *ast.FuncDecl
	imports    map[string]bool
}

func (f *muFunc) String() string {
	if f.recv == "" {
		return f.name
	}
	return f.recv + "." + f.name
}

type muReacquisition struct {
	Pos            token.Position
	Caller, Callee string
}

func (r muReacquisition) String() string {
	return r.Pos.Filename + ":" + strconv.Itoa(r.Pos.Line) + ": " + r.Caller + " holds " + vulkanDeviceMutex + " and calls " + r.Callee + ", which can lock it"
}

// muAudit is a conservative, name-and-shape call graph over one package. A call
// resolves through the static type of its receiver when that type is a package
// struct reached from the receiver, a parameter, a composite literal, or a struct
// field; otherwise it resolves to every package method of that name, which is how an
// interface call reaches the backend that implements it.
type muAudit struct {
	fset      *token.FileSet
	funcs     []*muFunc
	byName    map[string][]*muFunc
	byRecv    map[string]map[string][]*muFunc
	fields    map[string]map[string]string
	ifaces    map[string]bool
	acquires  map[*muFunc]bool
	lockSites int
}

func newMuAudit(fset *token.FileSet, files []*ast.File) *muAudit {
	a := &muAudit{
		fset:     fset,
		byName:   map[string][]*muFunc{},
		byRecv:   map[string]map[string][]*muFunc{},
		fields:   map[string]map[string]string{},
		ifaces:   map[string]bool{},
		acquires: map[*muFunc]bool{},
	}
	for _, file := range files {
		imports := map[string]bool{"C": true}
		for _, im := range file.Imports {
			path, _ := strconv.Unquote(im.Path.Value)
			name := path[strings.LastIndex(path, "/")+1:]
			if im.Name != nil {
				name = im.Name.Name
			}
			imports[name] = true
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Body == nil {
					continue
				}
				f := &muFunc{name: d.Name.Name, decl: d, imports: imports}
				if d.Recv != nil && len(d.Recv.List) == 1 {
					f.recv = muTypeName(d.Recv.List[0].Type)
				}
				a.funcs = append(a.funcs, f)
				a.byName[f.name] = append(a.byName[f.name], f)
				if f.recv != "" {
					if a.byRecv[f.recv] == nil {
						a.byRecv[f.recv] = map[string][]*muFunc{}
					}
					a.byRecv[f.recv][f.name] = append(a.byRecv[f.recv][f.name], f)
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}
					switch tt := ts.Type.(type) {
					case *ast.InterfaceType:
						a.ifaces[ts.Name.Name] = true
					case *ast.StructType:
						fields := map[string]string{}
						for _, field := range tt.Fields.List {
							typ := muTypeName(field.Type)
							for _, n := range field.Names {
								fields[n.Name] = typ
							}
							if len(field.Names) == 0 && typ != "" {
								fields[typ] = typ
							}
						}
						a.fields[ts.Name.Name] = fields
					}
				}
			}
		}
	}
	for _, f := range a.funcs {
		ast.Inspect(f.decl.Body, func(n ast.Node) bool {
			if _, ok := n.(*ast.GoStmt); ok {
				return false
			}
			if isMuCall(n, "Lock") {
				a.acquires[f] = true
				a.lockSites++
			}
			return true
		})
	}
	for changed := true; changed; {
		changed = false
		for _, f := range a.funcs {
			if a.acquires[f] {
				continue
			}
			env := a.env(f)
			ast.Inspect(f.decl.Body, func(n ast.Node) bool {
				if _, ok := n.(*ast.GoStmt); ok || a.acquires[f] {
					return false
				}
				if call, ok := n.(*ast.CallExpr); ok {
					for _, g := range a.resolve(f, env, call) {
						if a.acquires[g] {
							a.acquires[f], changed = true, true
							return false
						}
					}
				}
				return true
			})
		}
	}
	return a
}

// reacquisitions lists every call that can reach vulkanMu.Lock() while the caller
// holds it: statements after a Lock until an Unlock in the same statement list, and
// the whole body of a *Locked helper (whose contract is that the caller holds it).
func (a *muAudit) reacquisitions() []muReacquisition {
	seen := map[string]bool{}
	var out []muReacquisition
	for _, f := range a.funcs {
		env := a.env(f)
		report := func(call *ast.CallExpr, callee string) {
			r := muReacquisition{Pos: a.fset.Position(call.Pos()), Caller: f.String(), Callee: callee}
			if key := r.String(); !seen[key] {
				seen[key] = true
				out = append(out, r)
			}
		}
		if strings.HasSuffix(f.name, "Locked") {
			a.heldStmts(f, env, f.decl.Body.List, report)
		}
		ast.Inspect(f.decl.Body, func(n ast.Node) bool {
			var list []ast.Stmt
			switch n := n.(type) {
			case *ast.BlockStmt:
				list = n.List
			case *ast.CaseClause:
				list = n.Body
			case *ast.CommClause:
				list = n.Body
			}
			for i, s := range list {
				if es, ok := s.(*ast.ExprStmt); ok && isMuCall(es.X, "Lock") {
					a.heldStmts(f, env, list[i+1:], report)
				}
			}
			return true
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Pos.Filename != out[j].Pos.Filename {
			return out[i].Pos.Filename < out[j].Pos.Filename
		}
		return out[i].Pos.Offset < out[j].Pos.Offset
	})
	return out
}

func (a *muAudit) heldStmts(f *muFunc, env map[string]string, stmts []ast.Stmt, report func(*ast.CallExpr, string)) {
	for _, s := range stmts {
		if es, ok := s.(*ast.ExprStmt); ok && isMuCall(es.X, "Unlock") {
			return
		}
		a.heldNode(f, env, s, report)
	}
}

func (a *muAudit) heldNode(f *muFunc, env map[string]string, root ast.Node, report func(*ast.CallExpr, string)) {
	ast.Inspect(root, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.GoStmt:
			return false
		case *ast.BlockStmt:
			a.heldStmts(f, env, n.List, report)
			return false
		case *ast.CaseClause:
			for _, e := range n.List {
				a.heldNode(f, env, e, report)
			}
			a.heldStmts(f, env, n.Body, report)
			return false
		case *ast.CommClause:
			if n.Comm != nil {
				a.heldNode(f, env, n.Comm, report)
			}
			a.heldStmts(f, env, n.Body, report)
			return false
		case *ast.CallExpr:
			if isMuCall(n, "Lock") {
				report(n, vulkanDeviceMutex+".Lock")
				return true
			}
			if isMuCall(n, "Unlock") || isMuCall(n, "TryLock") {
				return true
			}
			for _, g := range a.resolve(f, env, n) {
				if a.acquires[g] {
					report(n, g.String())
				}
			}
		}
		return true
	})
}

// env maps the identifiers whose package struct type is evident from syntax alone.
func (a *muAudit) env(f *muFunc) map[string]string {
	env := map[string]string{}
	bind := func(names []*ast.Ident, typ string) {
		for _, n := range names {
			if typ != "" && n.Name != "_" {
				env[n.Name] = typ
			}
		}
	}
	if d := f.decl; d.Recv != nil {
		for _, field := range d.Recv.List {
			bind(field.Names, f.recv)
		}
	}
	if d := f.decl; d.Type.Params != nil {
		for _, field := range d.Type.Params.List {
			bind(field.Names, muTypeName(field.Type))
		}
	}
	ast.Inspect(f.decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			if n.Tok != token.DEFINE || len(n.Lhs) != len(n.Rhs) {
				return true
			}
			for i, rhs := range n.Rhs {
				if u, ok := rhs.(*ast.UnaryExpr); ok && u.Op == token.AND {
					rhs = u.X
				}
				if cl, ok := rhs.(*ast.CompositeLit); ok {
					if id, ok := n.Lhs[i].(*ast.Ident); ok {
						bind([]*ast.Ident{id}, muTypeName(cl.Type))
					}
				}
			}
		case *ast.ValueSpec:
			bind(n.Names, muTypeName(n.Type))
		}
		return true
	})
	return env
}

func (a *muAudit) typeOf(env map[string]string, e ast.Expr) string {
	switch e := e.(type) {
	case *ast.Ident:
		return env[e.Name]
	case *ast.ParenExpr:
		return a.typeOf(env, e.X)
	case *ast.StarExpr:
		return a.typeOf(env, e.X)
	case *ast.SelectorExpr:
		if t := a.typeOf(env, e.X); t == muExternal {
			return muExternal
		} else if t != "" {
			return a.fields[t][e.Sel.Name]
		}
	}
	return ""
}

func (a *muAudit) resolve(f *muFunc, env map[string]string, call *ast.CallExpr) []*muFunc {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		var out []*muFunc
		for _, g := range a.byName[fun.Name] {
			if g.recv == "" {
				out = append(out, g)
			}
		}
		return out
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok && f.imports[id.Name] && env[id.Name] == "" {
			return nil
		}
		t := a.typeOf(env, fun.X)
		if t == muExternal {
			return nil
		}
		if t != "" && !a.ifaces[t] {
			if ms := a.byRecv[t][fun.Sel.Name]; len(ms) > 0 {
				return ms
			}
			if _, isStruct := a.fields[t]; isStruct && !muHasEmbedded(a.fields[t]) {
				return nil
			}
		}
		var out []*muFunc
		for _, g := range a.byName[fun.Sel.Name] {
			if g.recv != "" {
				out = append(out, g)
			}
		}
		return out
	}
	return nil
}

func muHasEmbedded(fields map[string]string) bool {
	for name, typ := range fields {
		if name == typ {
			return true
		}
	}
	return false
}

func isMuCall(n ast.Node, method string) bool {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != method {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == vulkanDeviceMutex
}

func muTypeName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.StarExpr:
		return muTypeName(t.X)
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr:
		return muTypeName(t.X)
	case *ast.IndexListExpr:
		return muTypeName(t.X)
	case *ast.SelectorExpr:
		return muExternal
	}
	return ""
}
