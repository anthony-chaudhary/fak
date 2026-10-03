package witness

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const symptomProofCacheVerb = "symptom_proof_v1"

// symptomChangedTestsCheapCacheEligible rejects known external-input tests
// before paying for a parent checkout or dependency enumeration. It only says
// "worth the full proof"; symptomGoClosureDigest remains the authority that
// admits reuse.
func symptomChangedTestsCheapCacheEligible(ctx context.Context, git Runner, repoDir, commit string, tests []string) bool {
	if git == nil {
		git = gitRunner
	}
	for _, rel := range tests {
		if !strings.HasSuffix(rel, "_test.go") {
			return false
		}
		source, code, err := git(ctx, repoDir, "show", commit+":"+rel)
		if err != nil || code != 0 || ctx.Err() != nil {
			return false
		}
		file, err := parser.ParseFile(token.NewFileSet(), rel, source, parser.ImportsOnly)
		if err != nil {
			return false
		}
		for _, spec := range file.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil || symptomKnownExternalInputImport(path) {
				return false
			}
		}
	}
	return len(tests) > 0
}

func symptomKnownExternalInputImport(path string) bool {
	for _, prefix := range []string{"archive/zip", "crypto/rand", "io/ioutil", "math/rand", "net", "os", "plugin", "syscall", "time", "unsafe"} {
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return true
		}
	}
	return false
}

// goListProofPackage is the bounded subset of `go list -deps -test -json`
// needed to prove which source bytes can affect selected Go tests.
type goListProofPackage struct {
	Dir             string
	ImportPath      string
	Name            string
	Standard        bool
	Incomplete      bool
	ForTest         string
	GoFiles         []string
	CgoFiles        []string
	CFiles          []string
	CXXFiles        []string
	MFiles          []string
	HFiles          []string
	FFiles          []string
	SFiles          []string
	SwigFiles       []string
	SwigCXXFiles    []string
	SysoFiles       []string
	EmbedFiles      []string
	TestGoFiles     []string
	TestEmbedFiles  []string
	XTestGoFiles    []string
	XTestEmbedFiles []string
	Imports         []string
	TestImports     []string
	XTestImports    []string
	Error           *struct{ Err string }
	DepsErrors      []struct{ Err string }
}

func (r *Resolver) symptomProofCacheKey(ctx context.Context, candidateDir, parentDir string, selections []symptomSelection, tags []string) (string, bool) {
	if !cacheEnabled() {
		return "", false
	}
	packages, selectors := symptomProofSelections(selections)
	if len(packages) == 0 || len(selectors) == 0 {
		return "", false
	}
	candidate, ok := symptomGoClosureDigest(ctx, candidateDir, packages, tags)
	if !ok {
		return "", false
	}
	parent, ok := symptomGoClosureDigest(ctx, parentDir, packages, tags)
	if !ok {
		return "", false
	}
	environment, ok := symptomExecutionEnvironmentDigest(ctx, candidateDir, parentDir)
	if !ok {
		return "", false
	}
	parts := []string{"candidate=" + candidate, "parent-overlay=" + parent, "environment=" + environment}
	parts = append(parts, "tags="+strings.Join(normalizeTags(tags), ","))
	parts = append(parts, selectors...)
	return WitnessCacheKey(symptomProofCacheVerb, nil, parts...), true
}

func symptomProofSelections(selections []symptomSelection) ([]string, []string) {
	byPackage := make(map[string][]string, len(selections))
	for _, selection := range selections {
		pkg := strings.TrimSpace(selection.Package)
		if pkg == "" {
			continue
		}
		byPackage[pkg] = append(byPackage[pkg], selection.Tests...)
	}
	packages := make([]string, 0, len(byPackage))
	for pkg := range byPackage {
		packages = append(packages, pkg)
	}
	sort.Strings(packages)
	rows := make([]string, 0, len(packages))
	for _, pkg := range packages {
		rows = append(rows, "selection="+pkg+"\x00"+strings.Join(normalizeTags(byPackage[pkg]), "\x00"))
	}
	return packages, rows
}

func symptomGoClosureDigest(ctx context.Context, dir string, packages, tags []string) (string, bool) {
	argv := []string{"go", "list", "-deps", "-test", "-json"}
	if normalized := normalizeTags(tags); len(normalized) > 0 {
		argv = append(argv, "-tags", strings.Join(normalized, ","))
	}
	argv = append(argv, packages...)
	out, code, err := commandRunner(ctx, dir, argv...)
	if err != nil || code != 0 || ctx.Err() != nil {
		return "", false
	}
	dec := json.NewDecoder(strings.NewReader(out))
	listed := make([]goListProofPackage, 0, 64)
	standard := make(map[string]bool)
	for {
		var pkg goListProofPackage
		err := dec.Decode(&pkg)
		if err == io.EOF {
			break
		}
		if err != nil || pkg.Incomplete || pkg.Error != nil || len(pkg.DepsErrors) > 0 {
			return "", false
		}
		listed = append(listed, pkg)
		if pkg.Standard {
			standard[pkg.ImportPath] = true
		}
	}
	rows := make([]string, 0, 128)
	seen := false
	for _, pkg := range listed {
		if pkg.Standard || symptomGeneratedTestMain(pkg) || pkg.Dir == "" {
			continue
		}
		seen = true
		if !symptomPureGoPackage(pkg, standard) {
			return "", false
		}
		for _, group := range []struct {
			kind  string
			files []string
		}{{"go", pkg.GoFiles}, {"test", pkg.TestGoFiles}, {"xtest", pkg.XTestGoFiles}} {
			for _, name := range group.files {
				digest, ok := symptomFileDigest(filepath.Join(pkg.Dir, name))
				if !ok {
					return "", false
				}
				rows = append(rows, pkg.ImportPath+"\x00"+group.kind+"\x00"+filepath.ToSlash(name)+"\x00"+digest)
			}
		}
	}
	if !seen || len(rows) == 0 {
		return "", false
	}
	sort.Strings(rows)
	return symptomDigestStrings(rows), true
}

func symptomGeneratedTestMain(pkg goListProofPackage) bool {
	if pkg.Name != "main" || pkg.ForTest != "" || pkg.Dir == "" || !strings.HasSuffix(pkg.ImportPath, ".test") || len(pkg.GoFiles) == 0 {
		return false
	}
	for _, name := range pkg.GoFiles {
		if !filepath.IsAbs(name) {
			return false
		}
		rel, err := filepath.Rel(pkg.Dir, name)
		if err != nil || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))) {
			return false
		}
	}
	return true
}

func symptomPureGoPackage(pkg goListProofPackage, standard map[string]bool) bool {
	if len(pkg.CgoFiles)+len(pkg.CFiles)+len(pkg.CXXFiles)+len(pkg.MFiles)+len(pkg.HFiles)+
		len(pkg.FFiles)+len(pkg.SFiles)+len(pkg.SwigFiles)+len(pkg.SwigCXXFiles)+
		len(pkg.SysoFiles)+len(pkg.EmbedFiles)+len(pkg.TestEmbedFiles)+len(pkg.XTestEmbedFiles) > 0 {
		return false
	}
	for _, imported := range append(append(append([]string{}, pkg.Imports...), pkg.TestImports...), pkg.XTestImports...) {
		if standard[imported] && !symptomPureStandardImport(imported) {
			return false
		}
	}
	return true
}

func symptomPureStandardImport(path string) bool {
	if path == "testing" {
		return true // the known test harness, not a product input surface
	}
	for _, prefix := range []string{"bytes", "cmp", "encoding", "errors", "fmt", "hash", "math", "reflect", "regexp", "slices", "sort", "strconv", "strings", "sync", "unicode"} {
		if path == prefix || strings.HasPrefix(path, prefix+"/") {
			return !strings.HasPrefix(path, "math/rand")
		}
	}
	return false
}

func symptomExecutionEnvironmentDigest(ctx context.Context, candidateDir, parentDir string) (string, bool) {
	env := filterGitEnv(os.Environ())
	for i, row := range env {
		name, value, ok := strings.Cut(row, "=")
		if ok && symptomPathEnvironmentKey(name) {
			identity, readable := symptomPathIdentity(value)
			if !readable {
				return "", false
			}
			env[i] = name + "=" + identity
		}
	}
	sort.Strings(env)
	rows := []string{"process=" + symptomDigestStrings(env)}
	for _, item := range []struct{ name, dir string }{{"candidate", candidateDir}, {"parent", parentDir}} {
		digest, ok := symptomGoEnvironmentDigest(ctx, item.dir)
		if !ok {
			return "", false
		}
		rows = append(rows, item.name+"="+digest)
	}
	return symptomDigestStrings(rows), true
}

func symptomGoEnvironmentDigest(ctx context.Context, dir string) (string, bool) {
	keys := []string{"GOVERSION", "GOOS", "GOARCH", "GOEXPERIMENT", "CGO_ENABLED", "CC", "CXX", "GOFLAGS", "GOROOT", "GOTOOLDIR", "GOMOD", "GOWORK", "GOENV"}
	argv := append([]string{"go", "env", "-json"}, keys...)
	out, code, err := commandRunner(ctx, dir, argv...)
	if err != nil || code != 0 || ctx.Err() != nil {
		return "", false
	}
	values := map[string]string{}
	if json.Unmarshal([]byte(out), &values) != nil {
		return "", false
	}
	rows := make([]string, 0, len(keys)+1)
	for _, key := range keys {
		value, ok := values[key]
		if !ok {
			return "", false
		}
		if symptomPathEnvironmentKey(key) {
			identity, readable := symptomPathIdentity(value)
			if !readable {
				return "", false
			}
			value = identity
		}
		rows = append(rows, key+"="+value)
	}
	launcher, err := exec.LookPath("go")
	if err != nil {
		return "", false
	}
	launcherDigest, ok := symptomFileDigest(launcher)
	if !ok {
		return "", false
	}
	goBinary := filepath.Join(values["GOROOT"], "bin", "go")
	if filepath.Ext(goBinary) == "" {
		if _, err := os.Stat(goBinary + ".exe"); err == nil {
			goBinary += ".exe"
		}
	}
	binaryDigest, ok := symptomFileDigest(goBinary)
	if !ok {
		return "", false
	}
	rows = append(rows, "go-launcher="+launcherDigest, "go-toolchain="+binaryDigest)
	return symptomDigestStrings(rows), true
}

func symptomPathEnvironmentKey(name string) bool {
	switch strings.ToUpper(name) {
	case "GOMOD", "GOWORK", "GOENV":
		return true
	default:
		return false
	}
}

func symptomPathIdentity(path string) (string, bool) {
	path = strings.TrimSpace(path)
	if path == "" || strings.EqualFold(path, "off") || path == os.DevNull {
		return path, true
	}
	digest, ok := symptomFileDigest(path)
	if ok {
		return "sha256:" + digest, true
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return "missing", true
	}
	return "", false
}

func symptomFileDigest(path string) (string, bool) {
	f, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", false
	}
	return hex.EncodeToString(h.Sum(nil)), true
}

func symptomDigestStrings(rows []string) string {
	h := sha256.New()
	for _, row := range rows {
		_, _ = fmt.Fprintf(h, "%d:%s\n", len(row), row)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (r *Resolver) symptomProofCacheConfirmed(ctx context.Context, key string) bool {
	if ctx.Err() != nil {
		return false
	}
	cache, ok := r.verdictCache(ctx)
	if !ok {
		return false
	}
	entry, hit := cache.Get(key)
	return hit && entry.Verb == symptomProofCacheVerb && entry.Verdict == "confirmed"
}

func (r *Resolver) symptomProofCachePutConfirmed(ctx context.Context, key string) {
	// ResolveSymptomWithDetail converts every post-deadline result to ABSTAIN.
	// Never persist a CONFIRMED verdict the public API is about to discard.
	if ctx.Err() != nil {
		return
	}
	cache, ok := r.verdictCache(ctx)
	if !ok || ctx.Err() != nil {
		return
	}
	cache.Put(CachedVerdict{Key: key, Verb: symptomProofCacheVerb, Subject: "selected red-parent/green-candidate", Verdict: "confirmed"})
}
