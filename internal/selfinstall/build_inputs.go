package selfinstall

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// buildInputOptions is the complete caller-controlled build envelope. Environment
// overrides are applied to both go list and go env and are recorded in the digest.
type buildInputOptions struct {
	Tags       []string
	BuildFlags []string
	Env        []string
}

type buildInputIdentity struct {
	Digest   string
	Envelope map[string]string
}

type goListModule struct {
	Path, Version, Sum, GoMod, GoVersion string
	Dir                                  string
	Replace                              *goListModule
}

type goListPackage struct {
	ImportPath                                                          string
	Dir                                                                 string
	Standard                                                            bool
	Module                                                              *goListModule
	GoFiles, CgoFiles, CFiles, CXXFiles, MFiles, HFiles, FFiles, SFiles []string
	SwigFiles, SwigCXXFiles, SysoFiles, EmbedFiles                      []string
}

var buildInputEnvKeys = []string{
	"GOEXE", "GOOS", "GOARCH", "GOVERSION", "GOROOT", "GOTOOLCHAIN",
	"CGO_ENABLED", "CC", "CXX", "AR", "PKG_CONFIG",
	"GOAMD64", "GO386", "GOARM", "GOARM64", "GOMIPS", "GOMIPS64", "GOPPC64", "GORISCV64", "GOWASM",
	"GOEXPERIMENT", "GOFIPS140", "CGO_CFLAGS", "CGO_CPPFLAGS", "CGO_CXXFLAGS", "CGO_LDFLAGS",
	"GOFLAGS", "GO111MODULE", "GOWORK", "GOMOD", "GOMODCACHE", "GOPROXY", "GONOSUMDB", "GOPRIVATE", "GOVCS",
}

var runBuildInputCommand = runBuildInputGo

func deriveBuildInputIdentity(ctx context.Context, sourceDir, target string, opts buildInputOptions) (buildInputIdentity, error) {
	root, err := filepath.Abs(sourceDir)
	if err != nil {
		return buildInputIdentity{}, fmt.Errorf("resolve source directory: %w", err)
	}
	env := append(os.Environ(), opts.Env...)
	args := []string{"list", "-deps", "-json"}
	if len(opts.Tags) != 0 {
		args = append(args, "-tags="+strings.Join(opts.Tags, ","))
	}
	args = append(args, opts.BuildFlags...)
	args = append(args, target)
	out, err := runBuildInputCommand(ctx, root, env, args...)
	if err != nil {
		return buildInputIdentity{}, fmt.Errorf("enumerate executable input graph: %w", err)
	}
	var packages []goListPackage
	dec := json.NewDecoder(bytes.NewReader(out))
	for dec.More() {
		var pkg goListPackage
		if err := dec.Decode(&pkg); err != nil {
			return buildInputIdentity{}, fmt.Errorf("decode executable input graph: %w", err)
		}
		if !pkg.Standard {
			packages = append(packages, pkg)
		}
	}
	if len(packages) == 0 {
		return buildInputIdentity{}, fmt.Errorf("executable input graph contains no non-standard packages")
	}
	envOut, err := runBuildInputCommand(ctx, root, env, append([]string{"env", "-json"}, buildInputEnvKeys...)...)
	if err != nil {
		return buildInputIdentity{}, fmt.Errorf("read Go build envelope: %w", err)
	}
	envelope := map[string]string{}
	if err := json.Unmarshal(envOut, &envelope); err != nil {
		return buildInputIdentity{}, fmt.Errorf("decode Go build envelope: %w", err)
	}
	for _, pair := range opts.Env {
		key, value, ok := strings.Cut(pair, "=")
		if !ok || key == "" {
			return buildInputIdentity{}, fmt.Errorf("invalid environment override %q", pair)
		}
		envelope["override:"+key] = value
	}
	envelope["tags"] = strings.Join(opts.Tags, ",")
	envelope["build_flags"] = strings.Join(opts.BuildFlags, "\x00")
	for key, value := range envelope {
		envelope[key] = normalizeBuildInputEnv(key, value, root)
	}

	h := sha256.New()
	writeBuildInputRecord(h, "schema", "fak-build-inputs/1")
	keys := make([]string, 0, len(envelope))
	for key := range envelope {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		writeBuildInputRecord(h, "env:"+key, envelope[key])
	}
	sort.Slice(packages, func(i, j int) bool { return packages[i].ImportPath < packages[j].ImportPath })
	modules := map[string]*goListModule{}
	for _, pkg := range packages {
		if pkg.Module != nil {
			modules[buildInputModuleKey(pkg.Module)] = pkg.Module
		}
	}
	moduleKeys := make([]string, 0, len(modules))
	for key := range modules {
		moduleKeys = append(moduleKeys, key)
	}
	sort.Strings(moduleKeys)
	for _, key := range moduleKeys {
		if err := hashBuildInputModule(h, modules[key], root); err != nil {
			return buildInputIdentity{}, err
		}
	}
	for _, pkg := range packages {
		writeBuildInputRecord(h, "package", pkg.ImportPath)
		files := packageBuildInputFiles(pkg)
		sort.Strings(files)
		for _, name := range files {
			path, err := secureBuildInputPath(pkg.Dir, name)
			if err != nil {
				return buildInputIdentity{}, fmt.Errorf("package %s input %q: %w", pkg.ImportPath, name, err)
			}
			body, err := os.ReadFile(path)
			if err != nil {
				return buildInputIdentity{}, fmt.Errorf("read package %s input %q: %w", pkg.ImportPath, name, err)
			}
			writeBuildInputRecord(h, "file:"+pkg.ImportPath+":"+filepath.ToSlash(name), string(body))
		}
	}
	return buildInputIdentity{Digest: "sha256:" + hex.EncodeToString(h.Sum(nil)), Envelope: envelope}, nil
}

func runBuildInputGo(ctx context.Context, dir string, env []string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Dir, cmd.Env = dir, env
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("go %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func packageBuildInputFiles(pkg goListPackage) []string {
	var out []string
	for _, set := range [][]string{pkg.GoFiles, pkg.CgoFiles, pkg.CFiles, pkg.CXXFiles, pkg.MFiles, pkg.HFiles, pkg.FFiles, pkg.SFiles, pkg.SwigFiles, pkg.SwigCXXFiles, pkg.SysoFiles, pkg.EmbedFiles} {
		out = append(out, set...)
	}
	return out
}

func secureBuildInputPath(dir, name string) (string, error) {
	if filepath.IsAbs(name) {
		return "", fmt.Errorf("absolute path is not allowed")
	}
	base, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	path, err := filepath.Abs(filepath.Join(base, name))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(base, path)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes package directory")
	}
	return path, nil
}

func buildInputModuleKey(mod *goListModule) string {
	if mod == nil {
		return ""
	}
	replace := ""
	if mod.Replace != nil {
		replace = strings.Join([]string{mod.Replace.Path, mod.Replace.Version, mod.Replace.Sum, mod.Replace.GoVersion}, "\x00")
	}
	return strings.Join([]string{mod.Path, mod.Version, mod.Sum, mod.GoVersion, replace}, "\x00")
}

func hashBuildInputModule(h interface{ Write([]byte) (int, error) }, mod *goListModule, root string) error {
	if mod == nil {
		return nil
	}
	record := func(m *goListModule) string {
		if m == nil {
			return ""
		}
		path := m.Path
		if m.Dir != "" {
			if rel, err := filepath.Rel(root, m.Dir); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
				path += "@local:" + filepath.ToSlash(rel)
			}
		}
		return strings.Join([]string{path, m.Version, m.Sum, m.GoVersion}, "\x00")
	}
	key := record(mod)
	if mod.Replace != nil {
		key += "\x00replace=" + record(mod.Replace)
	}
	writeBuildInputRecord(h, "module", key)
	metadata := mod.GoMod
	if mod.Replace != nil && mod.Replace.GoMod != "" {
		metadata = mod.Replace.GoMod
	}
	if metadata != "" {
		body, err := os.ReadFile(metadata)
		if err != nil {
			return fmt.Errorf("read module metadata for %s: %w", mod.Path, err)
		}
		writeBuildInputRecord(h, "gomod:"+mod.Path, string(body))
	}
	return nil
}

func normalizeBuildInputEnv(key, value, root string) string {
	if key == "GOMOD" || key == "GOWORK" {
		if value == "" || value == os.DevNull {
			return value
		}
		if rel, err := filepath.Rel(root, value); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return filepath.ToSlash(rel)
		}
	}
	return value
}

func writeBuildInputRecord(w interface{ Write([]byte) (int, error) }, key, value string) {
	fmt.Fprintf(w, "%d:%s%d:%s", len(key), key, len(value), value)
}
