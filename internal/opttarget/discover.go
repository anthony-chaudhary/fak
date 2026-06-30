package opttarget

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// annotationTag is the directive that marks a const tunable as an auto-
// discoverable optimization target. A single-line comment of the form
//
//	// fak:opttarget metric=<name> dir=<higher|lower> sweep=<n,n,...> measurer=<key> [name=<id>]
//
// in the doc comment of a `const <Name> = <int>` declaration is harvested into an
// OptTarget WITHOUT a human enumerating it in any list — adding the comment
// anywhere in the tree adds a target. This is the difference between "10 targets a
// human typed" and "every annotated knob the repo already contains" (epic #1279,
// Phase 1). Discovery is READ-ONLY: it mutates nothing, it only inventories.
const annotationTag = "fak:opttarget"

// DiscoverDir walks root for .go files and harvests every annotationTag-tagged
// const into an OptTarget, returning them sorted by Name. Site.Path is recorded
// relative to root with forward slashes (so scanning the module root yields a
// path like "internal/rsiloop/tunable.go"). Sub-directories named "vendor" or
// "testdata", and dot-directories, are skipped (the standard tree-scan default;
// the root itself is always scanned, so a test may point DiscoverDir straight at a
// fixture dir). A malformed annotation is returned as an aggregated error while
// the well-formed targets are still returned — so a `--check` ratchet sees both
// the inventory and the defects.
func DiscoverDir(root string) ([]OptTarget, error) {
	var targets []OptTarget
	var errs []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path == root {
				return nil
			}
			name := d.Name()
			if name == "vendor" || name == "testdata" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			rel = path
		}
		rel = filepath.ToSlash(rel)
		ts, ferr := discoverFile(path, rel)
		targets = append(targets, ts...)
		errs = append(errs, ferr...)
		return nil
	})
	if walkErr != nil {
		return targets, walkErr
	}
	sort.Slice(targets, func(i, j int) bool { return targets[i].Name < targets[j].Name })
	if len(errs) > 0 {
		sort.Strings(errs)
		return targets, fmt.Errorf("opttarget discovery: %d malformed annotation(s):\n  %s", len(errs), strings.Join(errs, "\n  "))
	}
	return targets, nil
}

// discoverFile parses one source file and returns the OptTargets its annotated
// consts declare, plus any malformed-annotation errors. relPath is the path
// recorded on each target's Site (and Guards.ChangedPaths) — the file a kept
// candidate is allowed to rewrite.
func discoverFile(absPath, relPath string) ([]OptTarget, []string) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, absPath, nil, parser.ParseComments)
	if err != nil {
		return nil, []string{relPath + ": parse: " + err.Error()}
	}
	var targets []OptTarget
	var errs []string
	for _, decl := range f.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.CONST {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok || len(vs.Names) == 0 {
				continue
			}
			doc := vs.Doc
			if doc == nil {
				doc = gd.Doc
			}
			line, found := annotationLine(doc)
			if !found {
				continue
			}
			// A fak:opttarget tag only marks an INT tunable (`const <Name> = <int>`):
			// the int-sweep grammar rewrites that integer literal in place. A tag found
			// on any other const is documentation, not a target — most notably
			// annotationTag's OWN doc comment, which carries the format EXAMPLE
			// (`dir=<higher|lower>`) on a string const. Skip those silently so the
			// walker never false-positives on its own documentation.
			if !isIntLiteralSpec(vs) {
				continue
			}
			t, perr := parseAnnotation(line, vs.Names[0].Name, relPath)
			if perr != nil {
				errs = append(errs, relPath+" ("+vs.Names[0].Name+"): "+perr.Error())
				continue
			}
			if verr := t.Validate(); verr != nil {
				errs = append(errs, relPath+" ("+vs.Names[0].Name+"): "+verr.Error())
				continue
			}
			targets = append(targets, t)
		}
	}
	return targets, errs
}

// isIntLiteralSpec reports whether vs declares exactly one const bound to an
// integer literal (e.g. `const DefaultCacheSize = 4`). The int-sweep grammar
// rewrites that literal in place, so only an int-literal const is a tunable
// target; grouped iota specs, typed expressions, and string/float consts are
// not, and a fak:opttarget tag on one of them is documentation (or a mistake),
// never harvested.
func isIntLiteralSpec(vs *ast.ValueSpec) bool {
	if len(vs.Values) != 1 {
		return false
	}
	lit, ok := vs.Values[0].(*ast.BasicLit)
	return ok && lit.Kind == token.INT
}

// annotationLine returns the text following annotationTag in the first doc line
// that carries it, and whether one was found.
func annotationLine(doc *ast.CommentGroup) (string, bool) {
	if doc == nil {
		return "", false
	}
	for _, c := range doc.List {
		text := strings.TrimPrefix(c.Text, "//")
		text = strings.TrimPrefix(text, "/*")
		if i := strings.Index(text, annotationTag); i >= 0 {
			return strings.TrimSpace(text[i+len(annotationTag):]), true
		}
	}
	return "", false
}

// parseAnnotation turns the `metric=… dir=… sweep=… measurer=… [name=…]` tail of
// an annotation into an OptTarget for const constName declared in relPath. The
// grammar is the bounded int sweep (Phase 1's space); a missing required key, an
// unknown direction, or a non-int sweep value is a hard error so a typo can never
// silently drop a target out of the program.
func parseAnnotation(tail, constName, relPath string) (OptTarget, error) {
	kv := map[string]string{}
	for _, field := range strings.Fields(tail) {
		k, v, ok := strings.Cut(field, "=")
		if !ok || k == "" {
			return OptTarget{}, fmt.Errorf("annotation field %q is not key=value", field)
		}
		kv[k] = v
	}
	metric, ok := kv["metric"]
	if !ok {
		return OptTarget{}, fmt.Errorf("annotation missing metric=")
	}
	measurer, ok := kv["measurer"]
	if !ok {
		return OptTarget{}, fmt.Errorf("annotation missing measurer=")
	}
	var dir Direction
	switch kv["dir"] {
	case "higher":
		dir = HigherBetter
	case "lower":
		dir = LowerBetter
	case "":
		return OptTarget{}, fmt.Errorf("annotation missing dir=")
	default:
		return OptTarget{}, fmt.Errorf("annotation has unknown dir=%q (want higher|lower)", kv["dir"])
	}
	sweepRaw, ok := kv["sweep"]
	if !ok {
		return OptTarget{}, fmt.Errorf("annotation missing sweep=")
	}
	var ints []int
	for _, s := range strings.Split(sweepRaw, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		n, err := strconv.Atoi(s)
		if err != nil {
			return OptTarget{}, fmt.Errorf("annotation sweep value %q is not an int", s)
		}
		ints = append(ints, n)
	}
	name := kv["name"]
	if name == "" {
		name = constName
	}
	ref := kv["baseline"]
	if ref == "" {
		ref = "main"
	}
	return OptTarget{
		Name:        name,
		Metric:      metric,
		Direction:   dir,
		BaselineRef: ref,
		Site:        Site{Path: relPath, Const: constName},
		Grammar:     Grammar{Kind: GrammarIntSweep, Ints: ints},
		Measurer:    measurer,
		Guards:      Guards{ChangedPaths: []string{relPath}},
	}, nil
}

// Check is the discovery RATCHET — the keep-bit for coverage. It returns an error
// naming every required target (by Name) the discovered set no longer contains.
// Wired into a test or `fak opt discover --check`, it turns red the moment an
// annotated tunable is removed or renamed out of the program, so a target can
// never SILENTLY drop out of the RSI loop's reach. (It does not fault EXTRA
// discovered targets — the program is meant to grow; it only floors the known
// set.)
func Check(discovered []OptTarget, required []string) error {
	have := map[string]bool{}
	for _, t := range discovered {
		have[t.Name] = true
	}
	var missing []string
	for _, r := range required {
		if !have[r] {
			missing = append(missing, r)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("discovery ratchet: %d required target(s) no longer discovered: %s",
			len(missing), strings.Join(missing, ", "))
	}
	return nil
}

// MarshalInventory renders discovered targets as the stable JSON inventory the
// scanner emits (the Phase 1 `fak opt discover` payload) — indented and newline-
// terminated so a diff over two runs is reviewable. The targets are emitted in
// the order given (DiscoverDir already sorts by Name), so the bytes are
// deterministic.
func MarshalInventory(targets []OptTarget) ([]byte, error) {
	if targets == nil {
		targets = []OptTarget{}
	}
	b, err := json.MarshalIndent(targets, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
