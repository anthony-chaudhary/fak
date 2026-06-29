// Package affectedtests is the pure core of the `fak affected` fast test gate: given
// the package import graph and the set of CHANGED packages, it computes the exact set
// of packages whose test outcome could change -- so a developer runs `go test` on only
// those, turning the full ~minutes `go test ./...` into a seconds-long pre-commit gate
// WITHOUT dropping coverage on what they changed.
//
// Tier: foundation (1) -- see internal/architest. It is a pure graph primitive: it
// imports only the standard library, has no I/O, and never touches the request path.
// The impure shell (cmd/fak/affected.go) gathers the inputs -- `git diff` for the
// changed files, `go list -json ./...` for the import graph -- and runs `go test` on
// the result; this package owns only the deterministic selection.
//
// THE CORRECTNESS ARGUMENT. A package P's tests can only behave differently than they
// did at the base if P itself changed, or if some package P transitively imports
// changed. Equivalently: the affected set is the changed packages together with all
// their ANCESTORS in the import DAG (every package with an import path leading to a
// changed one). Test imports count as import edges -- if P's _test.go imports changed Q,
// P is affected. So as long as the edge set the shell hands in includes test imports,
// selecting the ancestor closure can never skip a package whose test could newly fail.
// It is conservative in exactly one safe direction: a package imported by no test and
// only by far-away production code is still re-tested (a true edge, not a false one);
// it never DROPS a package that a sound full run would have caught.
//
// THE LIMIT, STATED. This is an IMPORT-graph closure, not a behavioral one. It assumes
// a test's outcome depends only on Go packages reachable through the import graph. A
// test that reads a file at runtime by a path the shell did not map to a package, talks
// to a network service, or depends on build tags the graph did not expand, can change
// without its package being selected. That residue is why `make ci` still runs the full
// `go test ./...` as the authoritative gate -- `fak affected` is the fast INNER loop, not
// a replacement for the full oracle. Naming the limit here is the honesty fence
// (docs/standards/net-true-value.md Q3 scope).
package affectedtests

import "sort"

// Select returns the sorted set of packages whose tests should run given the changed
// packages: the changed packages themselves PLUS every package that transitively imports
// a changed package. edges[p] is the list of packages p directly imports (intra-module,
// and INCLUDING test imports -- the shell is responsible for folding Imports +
// TestImports + XTestImports). A package that appears only as an import target (never as
// an edges key) is handled correctly; a changed package with no importers selects just
// itself.
//
// Pure and deterministic: same inputs -> identical output, always.
func Select(edges map[string][]string, changed []string) []string {
	// Build the REVERSE graph: importedBy[q] lists every p that directly imports q.
	// Seeding a BFS from the changed set over this reverse graph reaches exactly the
	// ancestors of the changed packages -- the packages whose tests could change.
	importedBy := make(map[string][]string, len(edges))
	for p, imps := range edges {
		for _, q := range imps {
			importedBy[q] = append(importedBy[q], p)
		}
	}

	selected := make(map[string]bool, len(changed))
	queue := make([]string, 0, len(changed))
	for _, c := range changed {
		if !selected[c] {
			selected[c] = true
			queue = append(queue, c)
		}
	}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, importer := range importedBy[cur] {
			if !selected[importer] {
				selected[importer] = true
				queue = append(queue, importer)
			}
		}
	}

	out := make([]string, 0, len(selected))
	for p := range selected {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// ChangedPackages maps a set of changed FILE paths to the set of packages they belong
// to, using a precomputed FILE->import-path index (fileToPkg, keyed by each package's
// actual source/embed file at its repo-relative slash path -- the shell builds it from
// `go list`'s GoFiles / TestGoFiles / XTestGoFiles / EmbedFiles / Cgo / ignored-by-build
// lists). Mapping by real source-file membership rather than by directory is what keeps
// the selection both precise and correct at the module root: a top-level Makefile, a
// README, or a doc inside a package directory is NOT one of any package's source files,
// so it maps to nothing -- a docs/build-only change selects an empty set and skips the
// suite, and a non-source file never spuriously drags in the root package.
//
// The result is sorted and de-duplicated. Pure and deterministic.
func ChangedPackages(fileToPkg map[string]string, files []string) []string {
	set := make(map[string]bool)
	for _, f := range files {
		if pkg, ok := fileToPkg[f]; ok {
			set[pkg] = true
		}
	}
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}
