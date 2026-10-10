package buildwitness

import (
	"fmt"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// platformname.go — the filename-implied build-constraint guard (incident 2026-10-10).
//
// Go treats a `_<GOOS>`, `_<GOARCH>` or `_<GOOS>_<GOARCH>` basename suffix (before an optional
// `_test`) as an implicit build constraint. Commit f9d4cca4 added
// internal/ctxmmu/closed_windows.go meaning "closed windows" in the context-window sense; Go read
// it as GOOS=windows, so `fileWindows` was undefined on every Linux build of internal/ctxmmu and its
// importers. It landed from a Windows checkout, where it compiled. Nothing in the source says the
// file is platform-specific, so neither the author nor a reviewer can see the constraint.
//
// The rule this enforces: a .go file whose NAME constrains it to a platform must also say so in
// its SOURCE with a `//go:build` line naming that platform. The explicit line is redundant to the
// compiler but not to a reader; an accidental suffix has no such line and is refused. A reviewed
// exception goes on PlatformNameAllowlist with its reason.

// knownOS and knownArch mirror go/build's syslist.go (the lists goodOSArchFile consults). They are
// deliberately the KNOWN lists, not the supported-port lists: a name like `x_nacl.go` is still
// excluded from every current build. TestPlatformNameListsMatchGoBuild pins them to go/build.
var (
	knownOS = map[string]bool{
		"aix": true, "android": true, "darwin": true, "dragonfly": true, "freebsd": true,
		"hurd": true, "illumos": true, "ios": true, "js": true, "linux": true, "nacl": true,
		"netbsd": true, "openbsd": true, "plan9": true, "solaris": true, "wasip1": true,
		"windows": true, "zos": true,
	}
	knownArch = map[string]bool{
		"386": true, "amd64": true, "amd64p32": true, "arm": true, "armbe": true, "arm64": true,
		"arm64be": true, "loong64": true, "mips": true, "mipsle": true, "mips64": true,
		"mips64le": true, "mips64p32": true, "mips64p32le": true, "ppc": true, "ppc64": true,
		"ppc64le": true, "riscv": true, "riscv64": true, "s390": true, "s390x": true,
		"sparc": true, "sparc64": true, "wasm": true,
	}
)

// PlatformNameAllowlist holds reviewed repo-relative (slash-separated) paths whose filename
// constraint is intentional but which deliberately carry no matching //go:build line. Keep it
// tiny: adding the build line is almost always the better fix. A stale entry (the file is gone
// or now compliant) fails the whole-tree gate so the list cannot rot.
var PlatformNameAllowlist = map[string]string{}

// FilenamePlatformTags returns the build tags a .go basename implies, following
// go/build.(*Context).goodOSArchFile: everything before the first "_" is ignored, a trailing
// "_test" is dropped, then a final `_GOOS_GOARCH`, `_GOOS` or `_GOARCH` element constrains the
// file. It returns nil for a name that implies no constraint.
func FilenamePlatformTags(base string) []string {
	name, _, _ := strings.Cut(base, ".")
	i := strings.Index(name, "_")
	if i < 0 {
		return nil
	}
	l := strings.Split(name[i:], "_")
	if n := len(l); n > 0 && l[n-1] == "test" {
		l = l[:n-1]
	}
	n := len(l)
	if n >= 2 && knownOS[l[n-2]] && knownArch[l[n-1]] {
		return []string{l[n-2], l[n-1]}
	}
	if n >= 1 && (knownOS[l[n-1]] || knownArch[l[n-1]]) {
		return []string{l[n-1]}
	}
	return nil
}

// PlatformNameViolation is one .go file whose name constrains it to a platform its source never
// names.
type PlatformNameViolation struct {
	Path    string   // repo-relative, slash-separated
	Implied []string // tags the filename implies
	Missing []string // implied tags absent from the file's build constraint
	HasLine bool     // the file carries a build constraint, just not one naming Missing
}

func (v PlatformNameViolation) String() string {
	base := path.Base(v.Path)
	what := "has no //go:build line"
	if v.HasLine {
		what = "has a //go:build line that does not name " + strings.Join(v.Missing, ", ")
	}
	return fmt.Sprintf("%s: the filename suffix makes Go compile it ONLY for %s, but the file %s. "+
		"If the platform restriction is intended, add `//go:build %s`; if %q is just a word, rename "+
		"the file so it does not end in a GOOS/GOARCH element",
		v.Path, strings.Join(v.Implied, "/"), what, strings.Join(v.Implied, " && "), base)
}

// CheckPlatformFilename applies the rule to one file's name and source. rel is the repo-relative
// path used for reporting; src is the file content. It returns nil when the name implies no
// platform or the source names every implied tag in a build constraint.
func CheckPlatformFilename(rel string, src []byte) *PlatformNameViolation {
	rel = filepath.ToSlash(rel)
	implied := FilenamePlatformTags(path.Base(rel))
	if len(implied) == 0 {
		return nil
	}
	named, hasLine := buildConstraintTags(src)
	var missing []string
	for _, tag := range implied {
		if !named[tag] {
			missing = append(missing, tag)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return &PlatformNameViolation{Path: rel, Implied: implied, Missing: missing, HasLine: hasLine}
}

// buildConstraintTags returns every tag named by the file's header build constraints
// (`//go:build` or legacy `// +build` lines before the package clause) and whether any exist.
func buildConstraintTags(src []byte) (map[string]bool, bool) {
	named := map[string]bool{}
	f, err := parser.ParseFile(token.NewFileSet(), "", src, parser.PackageClauseOnly|parser.ParseComments)
	if err != nil || f == nil {
		return named, false
	}
	has := false
	for _, group := range f.Comments {
		if group.Pos() > f.Package {
			break
		}
		for _, c := range group.List {
			if !constraint.IsGoBuild(c.Text) && !constraint.IsPlusBuild(c.Text) {
				continue
			}
			expr, err := constraint.Parse(c.Text)
			if err != nil {
				continue
			}
			has = true
			collectConstraintTags(expr, named)
		}
	}
	return named, has
}

func collectConstraintTags(expr constraint.Expr, into map[string]bool) {
	switch e := expr.(type) {
	case *constraint.TagExpr:
		into[e.Tag] = true
	case *constraint.NotExpr:
		collectConstraintTags(e.X, into)
	case *constraint.AndExpr:
		collectConstraintTags(e.X, into)
		collectConstraintTags(e.Y, into)
	case *constraint.OrExpr:
		collectConstraintTags(e.X, into)
		collectConstraintTags(e.Y, into)
	}
}

// CheckPlatformFilenames checks the given repo-relative paths under root (non-.go and missing
// files are skipped) and returns the violations not on PlatformNameAllowlist, sorted by path.
func CheckPlatformFilenames(root string, rels []string) ([]PlatformNameViolation, error) {
	var out []PlatformNameViolation
	for _, rel := range rels {
		rel = filepath.ToSlash(rel)
		if !strings.HasSuffix(rel, ".go") || FilenamePlatformTags(path.Base(rel)) == nil {
			continue
		}
		if _, ok := PlatformNameAllowlist[rel]; ok {
			continue
		}
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if os.IsNotExist(err) {
			continue // a deletion carries no constraint
		}
		if err != nil {
			return nil, err
		}
		if v := CheckPlatformFilename(rel, src); v != nil {
			out = append(out, *v)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// ScanPlatformFilenames walks every .go file under root that the go tool would consider
// (skipping testdata and "_"/"." prefixed directories and files) and returns the relative paths
// of those whose name implies a platform. Feed the result to CheckPlatformFilenames.
func ScanPlatformFilenames(root string) ([]string, error) {
	var rels []string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if p != root && (name == "testdata" || name == "vendor" || strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") || strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") {
			return nil
		}
		if FilenamePlatformTags(name) == nil {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		rels = append(rels, filepath.ToSlash(rel))
		return nil
	})
	sort.Strings(rels)
	return rels, err
}
