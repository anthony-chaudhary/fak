package hooks

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"sort"
	"strings"
)

// gate_cartbeforehorse.go is the commit-boundary nudge for the spine-first order
// in docs/spine-first-defaults.md. It looks only at a newly introduced
// internal/<leaf> package and asks one narrow question: is this commit adding a
// downstream proof/performance artifact before it carries an ordinary test that
// drives the leaf's applied production path?
//
// This is deliberately not a title or prose classifier. Candidate leaves come
// from newly added internal/<leaf> paths, downstream work comes from structural
// paths plus Go test declarations (Benchmark/Fuzz/etc.), and the spine witness is
// an ordinary Test function that references a production declaration in the full
// HEAD-plus-staged tree. Ordinary work in an existing leaf, docs-only commits,
// and safety/fail-closed proof stay outside the finding set. An explicit staged
// `Spine-witness: <command or receipt>` attestation is the narrow escape for a
// runnable applied path whose witness cannot live as an ordinary Go test.

const (
	cartBeforeHorseGate         = "CART_BEFORE_HORSE"
	cartBeforeHorseWitnessToken = "spine-witness:"
)

type cartLeaf struct {
	name  string
	paths []string
}

// gateCartBeforeHorse emits at most one finding per new leaf, sorted by leaf.
// A correctly sequenced new leaf and a commit with no new leaf both stay quiet.
func gateCartBeforeHorse(d *StagedDiff) ([]Finding, error) {
	leaves := cartBeforeHorseNewLeaves(d)
	d.NoteCandidates(cartBeforeHorseGate, len(leaves), "new internal/<leaf>/ package(s)")
	if len(leaves) == 0 {
		return nil, nil
	}

	var findings []Finding
	for _, leaf := range leaves {
		var carts []string
		for _, p := range leaf.paths {
			body, ok := d.FileBytes(p)
			if cartBeforeHorseArtifact(p, body, ok) {
				carts = append(carts, p)
			}
		}
		if len(carts) == 0 || hasSpineWitnessAttestation(d, leaf.name) || leafHasAppliedSpineTest(d, leaf) {
			continue
		}
		sort.Strings(carts)
		findings = append(findings, Finding{
			Gate: cartBeforeHorseGate,
			File: carts[0],
			Detail: fmt.Sprintf(
				"new leaf internal/%s stages downstream work (%s) before a captured ordinary spine test drives its applied production path — ship the smallest runnable path and witness first; see docs/spine-first-defaults.md. Add a Test... that references the real leaf API, or stage `Spine-witness: <command or receipt>` for an equivalent applied run. (advisory; FLEET_CART_BEFORE_HORSE_GUARD=block enforces, ALLOW_CART_BEFORE_HORSE=1 skips once)",
				leaf.name, cartPathSummary(carts)),
		})
	}
	return findings, nil
}

// cartBeforeHorseNewLeaves derives the candidate denominator from newly added
// paths below internal/<leaf>/. A leaf with any non-added path in the landing
// tree already existed and is intentionally quiet: this nudge is for creating
// the horse, not for judging ordinary maintenance in a mature package.
func cartBeforeHorseNewLeaves(d *StagedDiff) []cartLeaf {
	added := map[string]bool{}
	for _, raw := range d.AddedPaths {
		if p := normCartPath(raw); p != "" {
			added[p] = true
		}
	}

	candidates := map[string]bool{}
	for p := range added {
		if leaf, ok := internalLeafOf(p); ok {
			candidates[leaf] = true
		}
	}

	// Hand-built unit fixtures may omit IndexPaths. In that case the staged path
	// set is the best complete-tree witness available and consists entirely of
	// the files the fixture intentionally supplied.
	allPaths := d.IndexPaths
	if len(allPaths) == 0 {
		allPaths = d.StagedPaths
	}

	var names []string
	for leaf := range candidates {
		prefix := "internal/" + leaf + "/"
		existed := false
		for _, raw := range allPaths {
			p := normCartPath(raw)
			if strings.HasPrefix(p, prefix) && !added[p] {
				existed = true
				break
			}
		}
		if !existed {
			names = append(names, leaf)
		}
	}
	sort.Strings(names)

	out := make([]cartLeaf, 0, len(names))
	for _, leaf := range names {
		prefix := "internal/" + leaf + "/"
		seen := map[string]bool{}
		var paths []string
		for _, set := range [][]string{allPaths, d.StagedPaths, d.AddedPaths} {
			for _, raw := range set {
				p := normCartPath(raw)
				if strings.HasPrefix(p, prefix) && !seen[p] {
					seen[p] = true
					paths = append(paths, p)
				}
			}
		}
		sort.Strings(paths)
		out = append(out, cartLeaf{name: leaf, paths: paths})
	}
	return out
}

func normCartPath(p string) string {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	p = strings.TrimPrefix(path.Clean(p), "./")
	if p == "." {
		return ""
	}
	return p
}

// cartBeforeHorseArtifact recognizes downstream work from structural evidence.
// Safety/fail-closed artifacts are excluded because the doctrine makes the
// safety required to run the spine part of the spine itself.
func cartBeforeHorseArtifact(rel string, body []byte, readable bool) bool {
	p := strings.ToLower(normCartPath(rel))
	if p == "" || cartSafetyPath(p) {
		return false
	}
	for _, seg := range strings.Split(p, "/") {
		if seg == "testdata" || seg == "proof-matrix" || seg == "proof_matrix" || seg == "proofmatrix" {
			return true
		}
	}
	base := strings.TrimSuffix(path.Base(p), path.Ext(p))
	if cartNamedToken(base) {
		return true
	}
	if !readable || !strings.HasSuffix(p, ".go") {
		return false
	}
	f, err := parser.ParseFile(token.NewFileSet(), rel, body, 0)
	if err != nil {
		return false // syntax/build gates own unreadable Go; this advisory must not guess
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || cartSafetyName(fn.Name.Name) {
			continue
		}
		name := strings.ToLower(fn.Name.Name)
		if strings.HasPrefix(name, "benchmark") || strings.HasPrefix(name, "fuzz") ||
			strings.Contains(name, "proofmatrix") || strings.Contains(name, "proof_matrix") ||
			strings.Contains(name, "soak") || strings.Contains(name, "profile") ||
			strings.Contains(name, "perftest") {
			return true
		}
	}
	return false
}

func cartNamedToken(base string) bool {
	base = strings.NewReplacer("-", "_", ".", "_").Replace(strings.ToLower(base))
	bounded := "_" + base + "_"
	for _, token := range []string{"benchmark", "bench", "perf", "profile", "profiling", "soak", "fuzz", "proof_matrix", "proofmatrix"} {
		if strings.Contains(bounded, "_"+token+"_") {
			return true
		}
	}
	return false
}

func cartSafetyPath(p string) bool {
	base := strings.TrimSuffix(path.Base(p), path.Ext(p))
	return cartSafetyName(base)
}

func cartSafetyName(name string) bool {
	n := strings.NewReplacer("-", "", "_", "").Replace(strings.ToLower(name))
	for _, token := range []string{"failclosed", "failsafe", "safety", "unsafe", "refuse", "reject", "deny", "quarantine"} {
		if strings.Contains(n, token) {
			return true
		}
	}
	return false
}

// hasSpineWitnessAttestation accepts an attestation only for its candidate leaf.
// A line staged below that leaf is inherently scoped; an attestation elsewhere
// must name internal/<leaf> explicitly so one leaf cannot waive another.
func hasSpineWitnessAttestation(d *StagedDiff, leaf string) bool {
	prefix := "internal/" + strings.ToLower(leaf) + "/"
	leafRef := "internal/" + strings.ToLower(leaf)
	for _, line := range d.AddedLines() {
		lower := strings.ToLower(line.Text)
		if i := strings.Index(lower, cartBeforeHorseWitnessToken); i >= 0 {
			receipt := strings.TrimSpace(lower[i+len(cartBeforeHorseWitnessToken):])
			underLeaf := strings.HasPrefix(strings.ToLower(normCartPath(line.File)), prefix)
			if receipt != "" && (underLeaf || containsCartLeafRef(receipt, leafRef)) {
				return true
			}
		}
	}
	return false
}

func containsCartLeafRef(text, leafRef string) bool {
	for i := strings.Index(text, leafRef); i >= 0; {
		end := i + len(leafRef)
		if end == len(text) || strings.ContainsRune("/ \x09\x0d\x0a,;:)]}", rune(text[end])) {
			return true
		}
		next := strings.Index(text[i+1:], leafRef)
		if next < 0 {
			return false
		}
		i += next + 1
	}
	return false
}

// leafHasAppliedSpineTest proves that an ordinary, runnable Test function in the
// landing tree references a production declaration from the same leaf. Merely
// staging an empty/synthetic Test name is not enough, and Benchmark/Fuzz functions
// cannot certify themselves as the prerequisite they are supposed to follow.
func leafHasAppliedSpineTest(d *StagedDiff, leaf cartLeaf) bool {
	production := map[string]bool{}
	tests := map[string][]byte{}
	for _, p := range leaf.paths {
		if !strings.HasSuffix(p, ".go") {
			continue
		}
		body, ok := d.FileBytes(p)
		if !ok {
			continue
		}
		if strings.HasSuffix(p, "_test.go") {
			tests[p] = body
			continue
		}
		for name := range goProductionNames(p, body) {
			production[name] = true
		}
	}
	if len(production) == 0 {
		return false
	}
	paths := make([]string, 0, len(tests))
	for p := range tests {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	for _, p := range paths {
		f, err := parser.ParseFile(token.NewFileSet(), p, tests[p], 0)
		if err != nil {
			continue
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || !ordinaryGoTestName(fn.Name.Name) || cartLikeTestName(fn.Name.Name) {
				continue
			}
			drives := false
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.Ident:
					drives = drives || production[x.Name]
				case *ast.SelectorExpr:
					drives = drives || production[x.Sel.Name]
				}
				return !drives
			})
			if drives {
				return true
			}
		}
	}
	return false
}

func goProductionNames(rel string, body []byte) map[string]bool {
	out := map[string]bool{}
	f, err := parser.ParseFile(token.NewFileSet(), rel, body, 0)
	if err != nil {
		return out
	}
	for _, decl := range f.Decls {
		switch x := decl.(type) {
		case *ast.FuncDecl:
			if x.Name.Name != "init" {
				out[x.Name.Name] = true
			}
		case *ast.GenDecl:
			for _, spec := range x.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					out[s.Name.Name] = true
				case *ast.ValueSpec:
					for _, name := range s.Names {
						out[name.Name] = true
					}
				}
			}
		}
	}
	return out
}

func ordinaryGoTestName(name string) bool {
	return strings.HasPrefix(name, "Test") && len(name) > len("Test")
}

func cartLikeTestName(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "proofmatrix") || strings.Contains(lower, "proof_matrix") ||
		strings.Contains(lower, "soak") || strings.Contains(lower, "profile") ||
		strings.Contains(lower, "perftest")
}

func cartPathSummary(paths []string) string {
	const capPaths = 4
	shown := paths
	if len(shown) > capPaths {
		shown = shown[:capPaths]
	}
	result := strings.Join(shown, ", ")
	if len(paths) > len(shown) {
		result += fmt.Sprintf(" (+%d more)", len(paths)-len(shown))
	}
	return result
}
