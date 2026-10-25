package quantmeta

import (
	"errors"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func registryUnparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

// enumSource evaluates original constant declarations with Go's type checker,
// preserving iota, aliases and expressions. It does not import either engine
// package into the neutral package or compile their unrelated implementations.
func enumSource(t *testing.T, path, target string) (*types.Package, map[string]string) {
	t.Helper()
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	// The loader also owns one private typed encoding outside the shared enum.
	// Include its original constant declarations in the same Go type-check, not
	// a guessed numeric alias or an exemption from coverage.
	declarationFiles := map[string]bool{filepath.Base(path): true}
	if target == "TensorType" {
		extra := "v41_engram_rows.go"
		ef, err := parser.ParseFile(fs, filepath.Join(filepath.Dir(path), extra), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range ef.Decls {
			if group, ok := decl.(*ast.GenDecl); ok && (group.Tok == token.CONST || group.Tok == token.TYPE) {
				f.Decls = append(f.Decls, group)
			}
		}
		declarationFiles[extra] = true
	}
	out := &ast.File{Name: ast.NewIdent("enumfixture")}
	typeSpecs := map[string]*ast.TypeSpec{}
	for _, decl := range f.Decls {
		if g, ok := decl.(*ast.GenDecl); ok && g.Tok == token.TYPE {
			for _, spec := range g.Specs {
				st := spec.(*ast.TypeSpec)
				typeSpecs[st.Name.Name] = st
			}
		}
	}
	var scalar func(string, map[string]bool) bool
	scalar = func(name string, seen map[string]bool) bool {
		switch name {
		case "bool", "string", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "byte", "rune", "float32", "float64", "complex64", "complex128":
			return true
		}
		if seen[name] {
			return false
		}
		seen[name] = true
		st := typeSpecs[name]
		if st == nil {
			return false
		}
		id, ok := registryUnparen(st.Type).(*ast.Ident)
		return ok && scalar(id.Name, seen)
	}
	for _, decl := range f.Decls {
		g, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		if g.Tok == token.CONST {
			out.Decls = append(out.Decls, g)
		}
		if g.Tok == token.TYPE {
			for _, spec := range g.Specs {
				s := spec.(*ast.TypeSpec)
				if scalar(s.Name.Name, map[string]bool{}) {
					out.Decls = append(out.Decls, &ast.GenDecl{Tok: token.TYPE, Specs: []ast.Spec{s}})
				}
			}
		}
	}
	pkg, err := new(types.Config).Check("enumfixture", fs, []*ast.File{out}, nil)
	if err != nil {
		t.Fatalf("evaluate %s: %v", path, err)
	}
	obj := pkg.Scope().Lookup(target)
	if obj == nil {
		t.Fatalf("missing enum type %s", target)
	}
	values := map[string]string{}
	for _, name := range pkg.Scope().Names() {
		c, ok := pkg.Scope().Lookup(name).(*types.Const)
		if ok && types.Identical(c.Type(), obj.Type()) {
			values[name] = c.Val().ExactString()
		}
	}
	if len(values) == 0 {
		t.Fatalf("empty %s enum", target)
	}
	// Fail closed if the enum's declaration surface expands beyond its defining
	// file. This includes typed constants, aliases, conversions and transitive
	// alias chains: the first link must mention the enum or a known constant.
	// All production files (including other platform tags) are inspected; none
	// of their implementation imports need to be compiled into quantmeta.
	dir := filepath.Dir(path)
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{target: true}
	for _, name := range pkg.Scope().Names() {
		if tn, ok := pkg.Scope().Lookup(name).(*types.TypeName); ok && types.Identical(tn.Type(), obj.Type()) {
			names[name] = true
		}
	}
	for name := range values {
		names[name] = true
	}
	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".go") || strings.HasSuffix(file.Name(), "_test.go") || declarationFiles[file.Name()] {
			continue
		}
		other, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, file.Name()), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range other.Decls {
			group, ok := decl.(*ast.GenDecl)
			if !ok || (group.Tok != token.CONST && group.Tok != token.TYPE) {
				continue
			}
			if group.Tok == token.TYPE {
				for _, spec := range group.Specs {
					alias := spec.(*ast.TypeSpec)
					if id, ok := registryUnparen(alias.Type).(*ast.Ident); ok && names[id.Name] {
						t.Errorf("%s type alias escaped source-bound file into %s; extend explicit coverage", target, file.Name())
					}
				}
				continue
			}
			ast.Inspect(group, func(n ast.Node) bool {
				if id, ok := n.(*ast.Ident); ok && names[id.Name] {
					t.Errorf("%s enum declaration escaped source-bound file into %s; extend explicit coverage", target, file.Name())
				}
				return true
			})
		}
	}
	return pkg, values
}

func TestRegistryEnumCoverage(t *testing.T) {
	// fak:test runtime=small est=1s
	if err := Validate(); err != nil {
		t.Fatal(err)
	}
	_, gg := enumSource(t, filepath.Join("..", "ggufload", "gguf.go"), "TensorType")
	wantGG := map[string]string{}
	for name, v := range ggufEnum {
		wantGG[name] = constant.MakeUint64(uint64(v)).ExactString()
	}
	if !reflect.DeepEqual(gg, wantGG) {
		t.Fatalf("GGUF enum coverage drift: actual=%v registry=%v", gg, wantGG)
	}
	_, dt := enumSource(t, filepath.Join("..", "compute", "compute.go"), "Dtype")
	wantDT := map[string]string{}
	for name, v := range dtypeEnum {
		wantDT[name] = constant.MakeUint64(uint64(v)).ExactString()
	}
	if !reflect.DeepEqual(dt, wantDT) {
		t.Fatalf("Dtype enum coverage drift: actual=%v registry=%v", dt, wantDT)
	}
	_, fm := enumSource(t, "contract.go", "Format")
	actualFormats := map[Format]bool{}
	for _, v := range fm {
		lit, err := parser.ParseExpr(v)
		if err != nil {
			t.Fatal(err)
		}
		s := lit.(*ast.BasicLit)
		tv, err := types.Eval(token.NewFileSet(), nil, token.NoPos, s.Value)
		if err != nil {
			t.Fatal(err)
		}
		actualFormats[Format(constant.StringVal(tv.Value))] = true
	}
	if !reflect.DeepEqual(actualFormats, knownFormats) {
		t.Fatalf("Format enum coverage drift: actual=%v known=%v", actualFormats, knownFormats)
	}
	// A future newly declared value must fail unless mapped or unsupported. The
	// independently parsed complete enum universe prevents self-confirming lists.
}

func TestRegistryMissingEntryFailsClosed(t *testing.T) {
	// fak:test runtime=small est=1s
	entries := Entries()
	for i := range entries {
		missing := append([]DtypeDescriptor{}, entries[:i]...)
		missing = append(missing, entries[i+1:]...)
		if err := validateRegistry(missing); err == nil {
			t.Fatalf("removing %s silently passed coverage", entries[i].Token)
		}
	}
	bad := Entries()
	bad[0].BlockBytes++
	if validateRegistry(bad) == nil {
		t.Fatal("inconsistent bytes accepted")
	}
	bad = Entries()
	bad[1].Token = bad[0].Token
	if validateRegistry(bad) == nil {
		t.Fatal("duplicate token accepted")
	}
	bad = Entries()
	bad[0].GGUF = 999
	if validateRegistry(bad) == nil {
		t.Fatal("unknown GGUF binding accepted")
	}
	bad = Entries()
	bad[0].Dtype = 3
	if validateRegistry(bad) == nil {
		t.Fatal("mapped-and-unsupported binding accepted")
	}
}

func TestRegistryRoundTripAndRefusals(t *testing.T) {
	// fak:test runtime=small est=1s
	for _, entry := range Entries() {
		t.Run(entry.Token, func(t *testing.T) {
			if entry.GGUF >= 0 {
				got, err := FromGGUF(uint32(entry.GGUF))
				if err != nil || got != entry {
					t.Fatalf("GGUF round trip: %+v %v", got, err)
				}
			}
			if entry.Dtype >= 0 {
				got, err := FromDtype(uint8(entry.Dtype))
				if err != nil || got != entry {
					t.Fatalf("Dtype round trip: %+v %v", got, err)
				}
			}
			if entry.Format != "" {
				got, err := FromFormat(entry.Format)
				if err == nil {
					if got != entry {
						t.Fatalf("Format selected wrong layout: %+v", got)
					}
				} else if !errors.Is(err, ErrAmbiguousFormat) && !errors.Is(err, ErrUnsupportedFormat) {
					t.Fatal(err)
				}
			}
		})
	}
	for id := range unsupportedGGUF {
		if _, err := FromGGUF(id); !errors.Is(err, ErrUnsupportedFormat) {
			t.Fatalf("GGUF %d: %v", id, err)
		}
	}
	for id := range unsupportedDtype {
		if _, err := FromDtype(id); !errors.Is(err, ErrUnsupportedFormat) {
			t.Fatalf("Dtype %d: %v", id, err)
		}
	}
	for f := range knownFormats {
		_, err := FromFormat(f)
		if err != nil && !errors.Is(err, ErrUnsupportedFormat) && !errors.Is(err, ErrAmbiguousFormat) {
			t.Fatalf("Format %q: %v", f, err)
		}
	}
	for _, f := range []Format{FormatInt4, FormatCodebook, FormatTernary} {
		if _, err := FromFormat(f); !errors.Is(err, ErrAmbiguousFormat) {
			t.Fatalf("coarse %q selected arbitrary layout: %v", f, err)
		}
	}
	if _, err := FromGGUF(255); !errors.Is(err, ErrUnknownFormat) {
		t.Fatal(err)
	}
	if _, err := FromDtype(255); !errors.Is(err, ErrUnknownFormat) {
		t.Fatal(err)
	}
	if _, err := FromFormat("future"); !errors.Is(err, ErrUnknownFormat) {
		t.Fatal(err)
	}
	// Same numeric value in distinct vocabularies must never share a fallback.
	if got, err := FromGGUF(3); err != nil || got.Token != "gguf:q4_1" {
		t.Fatalf("GGUF domain: %+v %v", got, err)
	}
	if _, err := FromDtype(3); !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("HAL Q8_0 confused with GGUF3: %v", err)
	}
	for _, id := range []uint32{42, 142} {
		e, err := FromGGUF(id)
		if err != nil || e.BlockSize != 128 || e.BlockBytes != 34 {
			t.Fatalf("GGUF ternary geometry: %+v %v", e, err)
		}
	}
	if _, err := FromDtype(10); !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatal("dynamic HAL ternary mapped to fixed GGUF layout")
	}
	copy := Entries()
	copy[0].Token = "corrupt"
	if Entries()[0].Token == "corrupt" {
		t.Fatal("caller mutated registry")
	}
}

func TestRegistryGeometryMatchesSource(t *testing.T) {
	// fak:test runtime=small est=1s
	pkg, _ := enumSource(t, filepath.Join("..", "ggufload", "gguf.go"), "TensorType")
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, filepath.Join("..", "ggufload", "gguf_type_traits.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	traits := map[uint32][2]int{}
	ast.Inspect(f, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || vs.Names[0].Name != "ggufTypeTraits" {
			return true
		}
		table, ok := vs.Values[0].(*ast.CompositeLit)
		if !ok {
			t.Fatal("trait table changed")
		}
		for _, el := range table.Elts {
			kv := el.(*ast.KeyValueExpr)
			name := kv.Key.(*ast.Ident).Name
			id := pkg.Scope().Lookup(name).(*types.Const)
			u, ok := constant.Uint64Val(id.Val())
			if !ok {
				t.Fatal(name)
			}
			fields := kv.Value.(*ast.CompositeLit).Elts
			var pair [2]int
			for i := range pair {
				switch e := fields[i].(type) {
				case *ast.Ident:
					c, ok := pkg.Scope().Lookup(e.Name).(*types.Const)
					if !ok {
						t.Fatal(e.Name)
					}
					v, ok := constant.Int64Val(c.Val())
					if !ok {
						t.Fatal(e.Name)
					}
					pair[i] = int(v)
				case *ast.BasicLit:
					v, err := types.Eval(fs, pkg, token.NoPos, e.Value)
					if err != nil {
						t.Fatal(err)
					}
					x, ok := constant.Int64Val(v.Value)
					if !ok {
						t.Fatal(e.Value)
					}
					pair[i] = int(x)
				default:
					t.Fatal("unsupported trait expression; update explicit geometry witness")
				}
			}
			traits[uint32(u)] = pair
		}
		return false
	})
	for _, e := range Entries() {
		if e.GGUF >= 0 {
			got, ok := traits[uint32(e.GGUF)]
			if !ok || got != [2]int{e.BlockSize, e.BlockBytes} {
				t.Fatalf("%s geometry drift: source=%v descriptor=%d/%d", e.Token, got, e.BlockSize, e.BlockBytes)
			}
		}
	}
	// Production stays neutral: no engine imports or dispatch changes.
	src, err := os.ReadFile("registry.go")
	if err != nil {
		t.Fatal(err)
	}
	rf, err := parser.ParseFile(token.NewFileSet(), "registry.go", src, parser.ImportsOnly)
	if err != nil {
		t.Fatal(err)
	}
	for _, im := range rf.Imports {
		if strings.Contains(im.Path.Value, "github.com/") {
			t.Fatalf("neutral registry gained engine import %s", im.Path.Value)
		}
	}
}

func TestRegistryCodeAndHALLayoutSemantics(t *testing.T) {
	// fak:test runtime=small est=1s
	// Independent semantic expectations from ggufload's dequantizers: Q4_1,
	// Q5_1 and Q2/Q4/Q5_K add/subtract affine minima; Q*_0 and Q3/Q6_K
	// recenter signed codes. IQ2/3 use signed grid selectors; IQ4 uses signed
	// nonlinear codebooks without an affine minimum. IQ1's offset grid is
	// deliberately unsupported. MXFP4 is E2M1 plus E8M0, not NVIDIA NVFP4.
	// Bits is payload precision, NOT effective bits including scales/sign grids.
	expected := map[uint32]struct {
		bits      int
		symmetric bool
	}{
		0: {32, true}, 1: {16, true}, 30: {16, true}, 2: {4, true}, 3: {4, false}, 6: {5, true}, 7: {5, false}, 8: {8, true},
		10: {2, false}, 11: {3, true}, 12: {4, false}, 13: {5, false}, 14: {6, true}, 16: {2, true}, 17: {2, true}, 18: {3, true}, 20: {4, true}, 21: {3, true}, 22: {2, true}, 23: {4, true}, 39: {4, true}, 42: {2, true}, 142: {2, true},
	}
	for id, want := range expected {
		got, err := FromGGUF(id)
		if err != nil || got.Bits != want.bits || got.Symmetric != want.symmetric {
			t.Fatalf("GGUF%d payload/minimum semantics: %+v %v", id, got, err)
		}
	}
	// HAL K/IQ buffers are packed GGUF blocks, not the separately scaled Q8/Q2
	// constructors. Pin the actual packing contracts independently of registry
	// self round trips. Changed source contracts demand explicit reconciliation.
	sources := map[string][]string{
		"quant_rawiq.go":   {"verbatim GGUF super-block stream", "const rawIQSuper = 256", "const iq4xsSuperBlock = 2 + 2 + 4 + 128", "const iq3xxsSuperBlock = 2 + 64 + 32", "const iq2sSuperBlock = 2 + 32 + 32 + 8 + 8", "const iq3sSuperBlock = 2 + 64 + 8 + 32 + 4", "const iq2xsSuperBlock = 2 + 64 + 8"},
		"quant_q2k.go":     {"const q2kSuperBlock = 84", "const q2kSuper = 256"},
		"quant_q3k.go":     {"const q3kSuperBlock = 110", "const q3kSuper = 256"},
		"quant_q4k.go":     {"const q4kSuperBlock = 2 + 2 + 12 + q4kSuper/2", "const q4kSuper = 256"},
		"quant_iq2_xxs.go": {"const iq2xxsSuperBlock = 2 + 32*2", "const iq2xxsSuper = 256"},
		"cpuref.go":        {"case Q5_K, Q6_K:", "bs := 176", "bs = 210", "rowBytes := (in / 256) * bs", "blk := w.Quant.Block", "wc, ws := c.i8(w), w.Quant.Scale"},
	}
	for file, contracts := range sources {
		src, err := os.ReadFile(filepath.Join("..", "compute", file))
		if err != nil {
			t.Fatal(err)
		}
		for _, contract := range contracts {
			if !strings.Contains(string(src), contract) {
				t.Fatalf("HAL layout contract changed in %s: %q", file, contract)
			}
		}
	}
	for dtype, want := range map[uint8][2]int{7: {256, 144}, 8: {256, 176}, 9: {256, 210}, 11: {256, 84}, 12: {256, 110}, 13: {256, 98}, 14: {256, 110}, 15: {256, 66}, 17: {256, 136}, 18: {256, 82}, 19: {256, 74}} {
		got, err := FromDtype(dtype)
		if err != nil || [2]int{got.BlockSize, got.BlockBytes} != want {
			t.Fatalf("HAL%d packed layout: %+v %v", dtype, got, err)
		}
	}
}
