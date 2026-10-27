package conceptcatalog

import (
	"archive/tar"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// This is an explicit, versioned execution contract, not a freshness receipt.
// Default/unknown profiles retain the full export and ordinary invocation.
// The opt-in excludes site startup; equivalence to ordinary startup is witnessed
// for the qualified baseline, not promised for arbitrary future site plugins.
const conceptTreeProfileEnv = "FAK_CONCEPT_TREE_PROFILE"
const conceptTreeProfileV1 = "isolated-v1"
const conceptTreeProfileV2 = "isolated-v2"
const conceptScriptSHA = "f03ebadc6443c4f4a0533118ebb007930b0e70218f0bf3c9230f4fd77d786259"
const conceptPythonSHA = "fa67443527ed9647f760d807e2a38f26340757123e643c4639cf273ed15d5ea7"
const conceptLibPythonSHA = "7e177a680eabb11fcb7d3867f7a1b9c4b93a9bd5b18129ce1af51ec07977b024"
const conceptStdlibSHA = "bcd17fc66d37e32b97d12ceda5d089bf47e7ea6e1b494615c5c434ab17297fc8"
const conceptStdlibV2SHA = "c8646082e9310ad2fa3cb8cd0038c87e63041f95444adba8bffa6163484bff14"
const conceptToolsSHA = "62313b76d15080e1d912367e6a29170fc5dd71489522e446d10719331a65ddb2"
const conceptToolsV2SHA = "4312ceeba9394b80c3f4145dc4afe413d3eb4b085513b9fc7ae93b0b80caeade"

// A materialized tree carries its execution contract explicitly. Sparse roots
// must never be handed to the ordinary generator without this descriptor.
type conceptGitTree struct {
	root    string
	profile *conceptTreeProfile
}
type conceptTreeProfile struct{ tree, python, prefix, stdlibSHA string }

func conceptDigest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func conceptFileDigest(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Runtime layout is derived without executing it. Unknown layouts decline the
// profile before export. No unverified encodings/site module runs in a probe.
func conceptProfileEnvironmentEligible(env []string) bool {
	for _, entry := range env {
		key, value, _ := strings.Cut(entry, "=")
		if value != "" && (strings.HasPrefix(key, "PYTHON") || strings.HasPrefix(key, "LD_")) {
			return false
		}
	}
	return true
}

func conceptRuntimeProfile(stdlibSHA string) (*conceptTreeProfile, error) {
	if runtime.GOOS != "linux" {
		return nil, nil
	}
	env := os.Environ()
	if !conceptProfileEnvironmentEligible(env) {
		return nil, nil
	}
	python, e := ResolvePython()
	if e != nil {
		return nil, nil
	}
	python, e = filepath.EvalSymlinks(python)
	if e != nil {
		return nil, nil
	}
	if filepath.Base(python) != "python3.12" || filepath.Base(filepath.Dir(python)) != "bin" {
		return nil, nil
	}
	p := &conceptTreeProfile{python: python, prefix: filepath.Dir(filepath.Dir(python)), stdlibSHA: stdlibSHA}
	if e = p.verifyRuntime(); e != nil {
		return nil, nil
	}
	// Only after the executable, shared interpreter, stdlib and zip are pinned.
	probe := exec.Command(python, "-I", "-S", "-B", "-c", `import sys,json;print(json.dumps([sys.flags.isolated,sys.flags.no_site,sys.flags.dont_write_bytecode,sys.path]))`)
	probe.Env = env
	windowgate.ConfigureBackgroundCommand(probe)
	out, e := probe.Output()
	if e != nil {
		return nil, fmt.Errorf("qualified Python probe: %w", e)
	}
	var got []json.RawMessage
	if e = json.Unmarshal(out, &got); e != nil || len(got) != 4 {
		return nil, fmt.Errorf("qualified Python probe shape")
	}
	for _, v := range got[:3] {
		if string(v) != "1" {
			return nil, fmt.Errorf("qualified Python isolation flags")
		}
	}
	var paths []string
	if e = json.Unmarshal(got[3], &paths); e != nil {
		return nil, e
	}
	want := []string{filepath.Join(p.prefix, "lib/python312.zip"), filepath.Join(p.prefix, "lib/python3.12"), filepath.Join(p.prefix, "lib/python3.12/lib-dynload")}
	if len(paths) != len(want) {
		return nil, fmt.Errorf("qualified Python search path")
	}
	for i := range paths {
		if paths[i] != want[i] {
			return nil, fmt.Errorf("qualified Python search path")
		}
	}
	return p, nil
}

func (p *conceptTreeProfile) verifyRuntime() error {
	for path, want := range map[string]string{p.python: conceptPythonSHA, filepath.Join(p.prefix, "lib/libpython3.12.so.1.0"): conceptLibPythonSHA} {
		st, e := os.Lstat(path)
		if e != nil || !st.Mode().IsRegular() {
			return fmt.Errorf("profile runtime regular file %s", path)
		}
		got, e := conceptFileDigest(path)
		if e != nil || got != want {
			return fmt.Errorf("profile runtime identity %s", path)
		}
	}
	// Pin the loader-visible alias as well as its backing bytes.
	alias := filepath.Join(p.prefix, "lib/libpython3.12.so")
	target, e := os.Readlink(alias)
	if e != nil || target != "libpython3.12.so.1.0" {
		return fmt.Errorf("profile libpython alias")
	}
	if _, e = os.Lstat(filepath.Join(p.prefix, "lib/python312.zip")); !os.IsNotExist(e) {
		return fmt.Errorf("profile unexpected stdlib zip")
	}
	root := filepath.Join(p.prefix, "lib/python3.12")
	var records []string
	e = filepath.WalkDir(root, func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		rel, e := filepath.Rel(root, path)
		if e != nil {
			return e
		}
		if d.IsDir() {
			if d.Name() == "site-packages" || d.Name() == "dist-packages" {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("profile stdlib symlink %s", rel)
		}
		st, e := d.Info()
		if e != nil {
			return e
		}
		if !st.Mode().IsRegular() {
			return fmt.Errorf("profile stdlib nonregular %s", rel)
		}
		h, e := conceptFileDigest(path)
		if e != nil {
			return e
		}
		records = append(records, filepath.ToSlash(rel)+"\x00"+fmt.Sprint(uint32(st.Mode().Perm()))+"\x00"+h+"\n")
		return nil
	})
	if e != nil {
		return e
	}
	sort.Strings(records)
	if conceptDigest([]byte(strings.Join(records, ""))) != p.stdlibSHA {
		return fmt.Errorf("profile stdlib identity")
	}
	return nil
}

// Each explicit profile admits only its independently qualified tools surface.
// In particular, v2 does not widen v1 or turn an unknown selector into a profile.
func conceptProfilePins(profile string) (tools, stdlib string, selected bool) {
	switch profile {
	case conceptTreeProfileV1:
		return conceptToolsSHA, conceptStdlibSHA, true
	case conceptTreeProfileV2:
		return conceptToolsV2SHA, conceptStdlibV2SHA, true
	default:
		return "", "", false
	}
}

func qualifyConceptTreeProfile(root, tree string) (*conceptTreeProfile, error) {
	toolsSHA, stdlibSHA, selected := conceptProfilePins(os.Getenv(conceptTreeProfileEnv))
	if !selected || runtime.GOOS != "linux" {
		return nil, nil
	}
	// Missing/changed script is ineligible, while failure to inspect the tree is fatal.
	listing, e := git(root, "ls-tree", "-rz", tree, "--", "tools")
	if e != nil {
		return nil, e
	}
	var surface []byte
	scriptPresent := false
	for _, row := range strings.Split(string(listing), "\x00") {
		if row == "" {
			continue
		}
		_, path, ok := strings.Cut(row, "\t")
		if !ok {
			return nil, fmt.Errorf("profile tree entry")
		}
		if path == "tools/concept_disambiguation_scorecard.py" {
			scriptPresent = true
			continue
		}
		if strings.HasPrefix(path, "tools/concept_disambiguation_scorecard.data/") && strings.Count(path, "/") == 2 && strings.HasSuffix(path, ".json") {
			continue
		}
		surface = append(surface, []byte(row)...)
		surface = append(surface, 0)
	}
	if !scriptPresent || conceptDigest(surface) != toolsSHA {
		return nil, nil
	}
	script, e := git(root, "show", tree+":tools/concept_disambiguation_scorecard.py")
	if e != nil {
		return nil, e
	}
	if conceptDigest(script) != conceptScriptSHA {
		return nil, nil
	}
	p, e := conceptRuntimeProfile(stdlibSHA)
	if p != nil {
		p.tree = tree
	}
	return p, e
}

func (p *conceptTreeProfile) command(root, out string, jsonOutput bool) (*exec.Cmd, error) {
	env := os.Environ()
	if !conceptProfileEnvironmentEligible(env) {
		return nil, fmt.Errorf("qualified Python environment changed")
	}
	if e := p.verifyRuntime(); e != nil {
		return nil, e
	}
	h, e := conceptFileDigest(filepath.Join(root, "tools/concept_disambiguation_scorecard.py"))
	if e != nil || h != conceptScriptSHA {
		return nil, fmt.Errorf("qualified archived generator identity")
	}
	args := []string{"-I", "-S", "-B", filepath.Join(root, "tools/concept_disambiguation_scorecard.py"), "--workspace", root, "--markdown-dir", out}
	if jsonOutput {
		args = append(args, "--json")
	}
	cmd := exec.Command(p.python, args...)
	cmd.Env = env
	return cmd, nil
}

func conceptProfileNeedsBytes(name string, size int64) bool {
	p := filepath.ToSlash(name)
	if p == "tools/concept_disambiguation_scorecard.py" || p == "docs/cli-reference.md" || p == GeneratedReadme || p == GeneratedIndex {
		return true
	}
	if strings.HasPrefix(p, "tools/concept_disambiguation_scorecard.data/") && strings.Count(p, "/") == 2 && strings.HasSuffix(p, ".json") {
		return true
	}
	if size > 512*1024 {
		return false
	}
	switch p {
	case "CLAIMS.md", "ARCHITECTURE.md", "README.md", "AGENTS.md", "INDEX.md", "GLOSSARY.md":
		return true
	}
	for _, part := range strings.Split(p, "/") {
		switch part {
		case ".git", ".cache", "node_modules", "testdata", "_registry", "__pycache__", ".pytest_cache", ".ruff_cache", "vendor", ".dispatch-runs", ".goal-runs":
			return false
		}
	}
	return ((strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/")) && strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go")) || (strings.HasPrefix(p, "docs/") && strings.HasSuffix(p, ".md"))
}

func extractConceptProfileTar(archive io.Reader, tmp string) error {
	tr := tar.NewReader(archive)
	for {
		h, e := tr.Next()
		if e == io.EOF {
			return nil
		}
		if e != nil {
			return e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive path %q", h.Name)
		}
		dst := filepath.Join(tmp, clean)
		if h.FileInfo().IsDir() {
			if e = os.MkdirAll(dst, 0755); e != nil {
				return e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e = os.MkdirAll(filepath.Dir(dst), 0755); e != nil {
			return e
		}
		f, e := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return e
		}
		var copyErr error
		if conceptProfileNeedsBytes(clean, h.Size) {
			_, copyErr = io.Copy(f, tr)
		} else {
			_, copyErr = io.Copy(io.Discard, tr)
			if copyErr == nil {
				copyErr = f.Truncate(h.Size)
			}
		}
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
}
