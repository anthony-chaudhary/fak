package archcheck

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const InternalImportPrefix = "github.com/anthony-chaudhary/fak/internal/"

// Violation records one architectural rule infraction.
type Violation struct {
	FromPackage string `json:"from_package"`
	FromTier    int    `json:"from_tier"`
	ToPackage   string `json:"to_package"`
	ToTier      int    `json:"to_tier"`
	File        string `json:"file"`
	Line        int    `json:"line"`
	Rule        string `json:"rule"`
	Detail      string `json:"detail"`
}

// CheckResult is the structured result returned by arch check.
type CheckResult struct {
	OK              bool        `json:"ok"`
	CheckedPackages int         `json:"checked_packages"`
	CheckedPaths    []string    `json:"checked_paths"`
	Violations      []Violation `json:"violations"`
	ElapsedMS       int64       `json:"elapsed_ms"`
}

func resolveRepoRoot(root string) string {
	if root != "" && root != "." {
		return root
	}
	dir, err := os.Getwd()
	if err != nil {
		return root
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return root
}

// LoadTiers parses the authoritative tier map from internal/architest/architest_test.go.
func LoadTiers(root string) (map[string]int, []string, error) {
	root = resolveRepoRoot(root)
	contractPath := filepath.Join(root, "internal", "architest", "architest_test.go")
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, contractPath, nil, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("archcheck: parse architest contract %s: %w", contractPath, err)
	}

	tiers := make(map[string]int)
	var names []string

	ast.Inspect(f, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
			return true
		}
		lit, ok := vs.Values[0].(*ast.CompositeLit)
		if !ok {
			return true
		}
		switch vs.Names[0].Name {
		case "tier":
			for _, e := range lit.Elts {
				kv, ok := e.(*ast.KeyValueExpr)
				if !ok {
					continue
				}
				k, ok1 := kv.Key.(*ast.BasicLit)
				v, ok2 := kv.Value.(*ast.BasicLit)
				if !ok1 || !ok2 {
					continue
				}
				pkg, err1 := strconv.Unquote(k.Value)
				lvl, err2 := strconv.Atoi(v.Value)
				if err1 == nil && err2 == nil {
					tiers[pkg] = lvl
				}
			}
		case "tierName":
			for _, e := range lit.Elts {
				b, ok := e.(*ast.BasicLit)
				if ok {
					if name, err := strconv.Unquote(b.Value); err == nil {
						names = append(names, name)
					}
				}
			}
		}
		return true
	})

	if len(tiers) == 0 {
		return nil, nil, fmt.Errorf("archcheck: zero tiers found in %s", contractPath)
	}
	return tiers, names, nil
}

// CheckPackage inspects a single package directory for architectural tier violations.
func CheckPackage(root, pkgRel string) (*CheckResult, error) {
	start := time.Now()
	root = resolveRepoRoot(root)
	tiers, names, err := LoadTiers(root)
	if err != nil {
		return nil, err
	}

	res := &CheckResult{
		OK:           true,
		CheckedPaths: []string{pkgRel},
	}

	violations, err := checkSinglePackage(root, pkgRel, tiers, names)
	if err != nil {
		return nil, err
	}
	res.Violations = violations
	res.OK = len(violations) == 0
	res.CheckedPackages = 1
	res.ElapsedMS = time.Since(start).Milliseconds()
	return res, nil
}

// CheckMine inspects Go packages touched by local uncommitted or staged changes.
func CheckMine(root string) (*CheckResult, error) {
	start := time.Now()
	root = resolveRepoRoot(root)
	tiers, names, err := LoadTiers(root)
	if err != nil {
		return nil, err
	}

	pkgPaths, err := discoverMinePackages(root)
	if err != nil {
		return nil, err
	}

	res := &CheckResult{
		OK:           true,
		CheckedPaths: pkgPaths,
	}

	for _, pkg := range pkgPaths {
		v, err := checkSinglePackage(root, pkg, tiers, names)
		if err != nil {
			return nil, err
		}
		res.Violations = append(res.Violations, v...)
	}

	res.CheckedPackages = len(pkgPaths)
	res.OK = len(res.Violations) == 0
	res.ElapsedMS = time.Since(start).Milliseconds()
	return res, nil
}

// CheckAll checks all packages under internal/.
func CheckAll(root string) (*CheckResult, error) {
	start := time.Now()
	root = resolveRepoRoot(root)
	tiers, names, err := LoadTiers(root)
	if err != nil {
		return nil, err
	}

	internalDir := filepath.Join(root, "internal")
	entries, err := os.ReadDir(internalDir)
	if err != nil {
		return nil, fmt.Errorf("archcheck: read internal: %w", err)
	}

	var pkgPaths []string
	for _, e := range entries {
		if e.IsDir() {
			pkgPaths = append(pkgPaths, filepath.ToSlash(filepath.Join("internal", e.Name())))
		}
	}
	sort.Strings(pkgPaths)

	res := &CheckResult{
		OK:           true,
		CheckedPaths: pkgPaths,
	}

	for _, pkg := range pkgPaths {
		v, err := checkSinglePackage(root, pkg, tiers, names)
		if err != nil {
			return nil, err
		}
		res.Violations = append(res.Violations, v...)
	}

	res.CheckedPackages = len(pkgPaths)
	res.OK = len(res.Violations) == 0
	res.ElapsedMS = time.Since(start).Milliseconds()
	return res, nil
}

func checkSinglePackage(root, pkgRel string, tiers map[string]int, names []string) ([]Violation, error) {
	cleanRel := filepath.ToSlash(filepath.Clean(pkgRel))
	cleanRel = strings.TrimPrefix(cleanRel, "./")

	var leafName string
	var isInternal bool
	if strings.HasPrefix(cleanRel, "internal/") {
		parts := strings.Split(cleanRel, "/")
		if len(parts) >= 2 {
			leafName = parts[1]
			isInternal = true
		}
	}

	pkgFull := filepath.Join(root, filepath.FromSlash(cleanRel))
	entries, err := os.ReadDir(pkgFull)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("archcheck: read pkg %s: %w", cleanRel, err)
	}

	var goFiles []os.DirEntry
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") && !strings.HasSuffix(e.Name(), "_test.go") {
			goFiles = append(goFiles, e)
		}
	}
	if len(goFiles) == 0 {
		return nil, nil
	}

	callerTier := 5 // default command / integrator level if outside internal
	if isInternal && leafName != "architest" {
		t, ok := tiers[leafName]
		if !ok {
			return []Violation{
				{
					FromPackage: leafName,
					FromTier:    -1,
					Rule:        "UNTIERED_LEAF",
					Detail:      fmt.Sprintf("package %s has no declared tier in internal/architest/architest_test.go", leafName),
				},
			}, nil
		}
		callerTier = t
	}

	fset := token.NewFileSet()
	var violations []Violation

	for _, e := range goFiles {

		filePath := filepath.Join(pkgFull, e.Name())
		f, err := parser.ParseFile(fset, filePath, nil, parser.ImportsOnly)
		if err != nil {
			return nil, fmt.Errorf("archcheck: parse imports %s: %w", filePath, err)
		}

		fileRel, _ := filepath.Rel(root, filePath)
		fileRel = filepath.ToSlash(fileRel)

		for _, imp := range f.Imports {
			importPath, err := strconv.Unquote(imp.Path.Value)
			if err != nil || !strings.HasPrefix(importPath, InternalImportPrefix) {
				continue
			}

			sub := strings.TrimPrefix(importPath, InternalImportPrefix)
			importedLeaf := strings.Split(sub, "/")[0]
			if importedLeaf == "" || importedLeaf == leafName {
				continue
			}

			importedTier, hasTier := tiers[importedLeaf]
			if !hasTier {
				line := fset.Position(imp.Path.Pos()).Line
				violations = append(violations, Violation{
					FromPackage: leafName,
					FromTier:    callerTier,
					ToPackage:   importedLeaf,
					ToTier:      -1,
					File:        fileRel,
					Line:        line,
					Rule:        "UNKNOWN_IMPORT_TIER",
					Detail:      fmt.Sprintf("imported package %s has no declared tier in architest", importedLeaf),
				})
				continue
			}

			line := fset.Position(imp.Path.Pos()).Line

			// Rule 1: Layered DAG: a package may only import packages whose tier is <= its own.
			if isInternal && importedTier > callerTier {
				violations = append(violations, Violation{
					FromPackage: leafName,
					FromTier:    callerTier,
					ToPackage:   importedLeaf,
					ToTier:      importedTier,
					File:        fileRel,
					Line:        line,
					Rule:        "UPWARD_IMPORT",
					Detail:      fmt.Sprintf("layered-DAG rule violated: %s (tier %d) imports %s (tier %d)", leafName, callerTier, importedLeaf, importedTier),
				})
			}

			// Rule 2: Primitive leaves (tier 1) stay primitive: may only import root/frozen ABI (tier 0).
			if isInternal && callerTier == 1 && importedTier > 0 {
				violations = append(violations, Violation{
					FromPackage: leafName,
					FromTier:    callerTier,
					ToPackage:   importedLeaf,
					ToTier:      importedTier,
					File:        fileRel,
					Line:        line,
					Rule:        "PRIMITIVE_LEAF_PURITY",
					Detail:      fmt.Sprintf("primitive leaf %s (tier 1) may only import root (tier 0), but imports %s (tier %d)", leafName, importedLeaf, importedTier),
				})
			}
		}
	}

	return violations, nil
}

func discoverMinePackages(root string) ([]string, error) {
	cmd := exec.Command("git", "-C", root, "status", "--porcelain=v1", "-z")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git status: %w", err)
	}

	pkgMap := make(map[string]bool)
	entries := bytes.Split(out, []byte{0})
	for _, entry := range entries {
		if len(entry) < 4 {
			continue
		}
		path := string(entry[3:])
		if strings.HasSuffix(path, ".go") {
			dir := filepath.ToSlash(filepath.Dir(path))
			if strings.HasPrefix(dir, "internal/") || strings.HasPrefix(dir, "cmd/") {
				pkgMap[dir] = true
			}
		}
	}

	var pkgs []string
	for p := range pkgMap {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)
	return pkgs, nil
}
