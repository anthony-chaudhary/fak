package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// fak-test:runtime fast est=10ms lane=default
func TestConceptProfilePayloadSelection(t *testing.T) {
	for _, tc := range []struct {
		path string
		size int64
		keep bool
	}{
		{"internal/example/main.go", 524288, true}, {"internal/example/main.go", 524289, false},
		{"internal/example/main_test.go", 10, false}, {"internal/vendor/x.go", 10, false},
		{"cmd/x/x.go", 10, true}, {"docs/x.md", 524288, true}, {"docs/x.md", 524289, false},
		{"docs/cli-reference.md", 900000, true}, {GeneratedReadme, 900000, true}, {GeneratedIndex, 900000, true},
		{"README.md", 10, true}, {"README.md", 524289, false}, {"tools/concept_disambiguation_scorecard.data/_meta.json", 900000, true},
		{"tools/concept_disambiguation_scorecard.data/new.json", 900000, true}, {"tools/concept_disambiguation_scorecard.data/nested/x.json", 10, false},
		{"assets/anchor.bin", 900000, false}, {"internal/x/FUTURE.GO", 10, false},
	} {
		if got := conceptProfileNeedsBytes(tc.path, tc.size); got != tc.keep {
			t.Errorf("%s %d: %v", tc.path, tc.size, got)
		}
	}
}

func profileFixtureTar(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	w := tar.NewWriter(&b)
	for _, x := range []struct {
		path, body string
		mode       int64
		kind       byte
		link       string
	}{
		{"internal/dironly/marker.bin", "omitted", 0755, tar.TypeReg, ""},
		{"internal/symbol/a.go", "package symbol; var Grounding = 1", 0644, tar.TypeReg, ""},
		{"docs/heading.md", "# Preserved Heading\n", 0644, tar.TypeReg, ""},
		{"assets/anchor.bin", strings.Repeat("z", 32768), 0755, tar.TypeReg, ""},
		{"internal/large.go", strings.Repeat("x", 524289), 0644, tar.TypeReg, ""},
		{"link.go", "", 0777, tar.TypeSymlink, "internal/symbol/a.go"},
		{"hard.go", "", 0644, tar.TypeLink, "internal/symbol/a.go"},
	} {
		if e := w.WriteHeader(&tar.Header{Name: x.path, Mode: x.mode, Size: int64(len(x.body)), Typeflag: x.kind, Linkname: x.link}); e != nil {
			t.Fatal(e)
		}
		if _, e := io.WriteString(w, x.body); e != nil {
			t.Fatal(e)
		}
	}
	if e := w.Close(); e != nil {
		t.Fatal(e)
	}
	return b.Bytes()
}

// fak-test:runtime fast est=20ms lane=default
func TestConceptProfileArchiveRepresentation(t *testing.T) {
	raw := profileFixtureTar(t)
	full, sparse := t.TempDir(), t.TempDir()
	if e := extractGitTar(bytes.NewReader(raw), full); e != nil {
		t.Fatal(e)
	}
	if e := extractConceptProfileTar(bytes.NewReader(raw), sparse); e != nil {
		t.Fatal(e)
	}
	var names []string
	e := filepath.WalkDir(full, func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		rel, _ := filepath.Rel(full, path)
		a, e := d.Info()
		if e != nil {
			return e
		}
		b, e := os.Lstat(filepath.Join(sparse, rel))
		if e != nil {
			return e
		}
		if a.Mode() != b.Mode() || (!a.IsDir() && a.Size() != b.Size()) {
			t.Fatalf("metadata %s", rel)
		}
		names = append(names, rel)
		if a.Mode().IsRegular() && conceptProfileNeedsBytes(rel, a.Size()) {
			x, _ := os.ReadFile(path)
			y, _ := os.ReadFile(filepath.Join(sparse, rel))
			if !bytes.Equal(x, y) {
				t.Fatalf("payload %s", rel)
			}
		}
		return nil
	})
	if e != nil {
		t.Fatal(e)
	}
	for _, n := range []string{"link.go", "hard.go"} {
		if _, e := os.Lstat(filepath.Join(sparse, n)); !os.IsNotExist(e) {
			t.Fatal("nonregular materialized", n)
		}
	}
	if len(names) == 0 {
		t.Fatal("empty fixture")
	}
	var b bytes.Buffer
	w := tar.NewWriter(&b)
	w.WriteHeader(&tar.Header{Name: "../unsafe", Mode: 0644, Size: 1})
	w.Write([]byte{'x'})
	w.Close()
	if e := extractConceptProfileTar(bytes.NewReader(b.Bytes()), t.TempDir()); e == nil {
		t.Fatal("unsafe accepted")
	}
	b.Reset()
	w = tar.NewWriter(&b)
	w.WriteHeader(&tar.Header{Name: "omitted.bin", Mode: 0644, Size: 999})
	w.Write([]byte{'x'})
	w.Flush()
	if e := extractConceptProfileTar(bytes.NewReader(b.Bytes()), t.TempDir()); e == nil {
		t.Fatal("omitted truncation accepted")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestConceptProfileUnknownIsGeneric(t *testing.T) {
	for _, value := range []string{"", "unknown"} {
		t.Setenv(conceptTreeProfileEnv, value)
		p, e := qualifyConceptTreeProfile(t.TempDir(), "not-a-tree")
		if e != nil || p != nil {
			t.Fatal(value, p, e)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestConceptProfileVersionedPins(t *testing.T) {
	for _, tc := range []struct{ profile, digest, stdlib string }{
		{conceptTreeProfileV1, "62313b76d15080e1d912367e6a29170fc5dd71489522e446d10719331a65ddb2", "bcd17fc66d37e32b97d12ceda5d089bf47e7ea6e1b494615c5c434ab17297fc8"},
		{conceptTreeProfileV2, "4312ceeba9394b80c3f4145dc4afe413d3eb4b085513b9fc7ae93b0b80caeade", "c8646082e9310ad2fa3cb8cd0038c87e63041f95444adba8bffa6163484bff14"},
		{"", "", ""}, {"isolated-v3", "", ""}, {"isolated-v2 ", "", ""},
	} {
		got, stdlib, selected := conceptProfilePins(tc.profile)
		if got != tc.digest || stdlib != tc.stdlib || selected != (tc.digest != "") {
			t.Fatalf("profile %q: digest=%q selected=%v", tc.profile, got, selected)
		}
	}
}

// This opt-in qualification is deliberately sequential, outside a worker export.
// Raw stdout/artifacts and actual storage accounting are retained by the caller.
// fak-test:runtime integration est=90s lane=default
func TestConceptProfileQualifiedTreeEquivalence(t *testing.T) {
	root, tree := os.Getenv("FAK_TEST_CONCEPT_ROOT"), os.Getenv("FAK_TEST_CONCEPT_TREE")
	evidence := os.Getenv("FAK_TEST_CONCEPT_EVIDENCE")
	if root == "" || tree == "" || evidence == "" {
		t.Skip("explicit pinned-tree qualification only")
	}
	if os.Getenv(conceptTreeProfileEnv) == "" {
		t.Setenv(conceptTreeProfileEnv, conceptTreeProfileV1)
	}
	p, e := qualifyConceptTreeProfile(root, tree)
	if e != nil || p == nil {
		t.Fatalf("profile qualification: %v %v", p, e)
	}
	var oracle map[string][]byte
	var oracleJSON map[string]any
	var oracleStatus int
	for _, mode := range []string{"full-ordinary", "full-isolated", "sparse-isolated"} {
		func() {
			tmp, e := os.MkdirTemp("", "concept-profile-proof-*")
			if e != nil {
				t.Fatal(e)
			}
			defer os.RemoveAll(tmp)
			export := filepath.Join(tmp, "tree")
			if e = os.Mkdir(export, 0755); e != nil {
				t.Fatal(e)
			}
			extract := extractGitTar
			if mode == "sparse-isolated" {
				extract = extractConceptProfileTar
			}
			reused := os.Getenv("FAK_TEST_CONCEPT_ORACLE_ROOT")
			if reused != "" && mode != "sparse-isolated" {
				export = reused
				if e = extractGitTreeUsing(root, tree, export, compareConceptOracleTar); e != nil {
					t.Fatal(e)
				}
			} else if e = extractGitTreeUsing(root, tree, export, extract); e != nil {
				t.Fatal(e)
			}
			out := filepath.Join(tmp, "generated")
			cmd := newInvariantSnapshotCommand(export, out)
			if mode != "full-ordinary" {
				cmd, e = p.command(export, out, true)
				if e != nil {
					t.Fatal(e)
				}
			}
			cmd.Dir = export
			windowgate.ConfigureBackgroundCommand(cmd)
			raw, runErr := cmd.Output()
			if runErr != nil {
				if _, ok := runErr.(*exec.ExitError); !ok {
					t.Fatal(runErr)
				}
			}
			if reused != "" && mode != "sparse-isolated" {
				if e = extractGitTreeUsing(root, tree, export, compareConceptOracleTar); e != nil {
					t.Fatal(e)
				}
			}
			if e = p.verifyRuntime(); e != nil {
				t.Fatal(e)
			}
			dest := filepath.Join(evidence, mode)
			if e = os.MkdirAll(dest, 0755); e != nil {
				t.Fatal(e)
			}
			if e = os.WriteFile(filepath.Join(dest, "stdout.json"), raw, 0644); e != nil {
				t.Fatal(e)
			}
			var payload map[string]any
			if e = json.Unmarshal(raw, &payload); e != nil {
				t.Fatal(e)
			}
			if payload["workspace"] != export {
				t.Fatal("unexpected workspace field")
			}
			delete(payload, "workspace")
			artifacts := map[string][]byte{}
			for _, a := range generatedArtifacts {
				v, e := os.ReadFile(filepath.Join(out, a.Name))
				if e != nil {
					t.Fatal(e)
				}
				artifacts[a.Name] = v
				if e = os.WriteFile(filepath.Join(dest, a.Name), v, 0644); e != nil {
					t.Fatal(e)
				}
			}
			var allocated, logical int64
			walkErr := filepath.Walk(export, func(path string, st os.FileInfo, e error) error {
				if e != nil {
					t.Fatal(e)
				}
				if path == filepath.Join(export, ".git") {
					if st.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}
				if st.Mode().IsRegular() {
					logical += st.Size()
				}
				v := reflect.ValueOf(st.Sys())
				if v.Kind() == reflect.Pointer && !v.IsNil() {
					f := v.Elem().FieldByName("Blocks")
					if f.IsValid() && f.CanInt() {
						allocated += f.Int() * 512
					}
				}
				return nil
			})
			if walkErr != nil {
				t.Fatal(walkErr)
			}
			status := 0
			if runErr != nil {
				status = runErr.(*exec.ExitError).ExitCode()
			}
			metrics, _ := json.Marshal(map[string]any{"tree": tree, "allocated_bytes": allocated, "logical_bytes": logical, "exit_code": status})
			if e = os.WriteFile(filepath.Join(dest, "storage.json"), metrics, 0644); e != nil {
				t.Fatal(e)
			}
			t.Logf("%s allocated=%d logical=%d exit=%d", mode, allocated, logical, status)
			if oracle == nil {
				oracle = artifacts
				oracleJSON = payload
				oracleStatus = status
			} else if status != oracleStatus || !reflect.DeepEqual(oracle, artifacts) || !reflect.DeepEqual(oracleJSON, payload) {
				t.Fatal("snapshot mismatch", mode)
			}
		}()
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestConceptProfileEnvironmentDrift(t *testing.T) {
	for _, key := range []string{"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT"} {
		if conceptProfileEnvironmentEligible([]string{key + "=changed"}) {
			t.Fatal("accepted", key)
		}
	}
	if !conceptProfileEnvironmentEligible([]string{"PATH=/bin", "PYTHONPATH="}) {
		t.Fatal("ordinary environment refused")
	}
	t.Setenv("LD_AUDIT", "changed")
	// The selected descriptor fails before touching even an invalid runtime.
	if _, err := (&conceptTreeProfile{}).command(t.TempDir(), t.TempDir(), false); err == nil || !strings.Contains(err.Error(), "environment changed") {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestConceptProfileUnknownImportSurfaceFallsBack(t *testing.T) {
	t.Setenv(conceptTreeProfileEnv, conceptTreeProfileV1)
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"concept_disambiguation_scorecard.py", "json.py"} {
		if err := os.WriteFile(filepath.Join(root, "tools", name), []byte("raise RuntimeError('must never run')\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	gitInitFixture(t, root)
	p, err := qualifyConceptTreeProfile(root, gitWriteTree(t, root))
	if err != nil || p != nil {
		t.Fatalf("unknown import surface: %v %v", p, err)
	}
	// Unknown profile materializes the ordinary full payload, never placeholders.
	tree, cleanup, err := materializeGitTree(root, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	got, err := os.ReadFile(filepath.Join(tree.root, "tools/json.py"))
	if err != nil || tree.profile != nil || !bytes.Contains(got, []byte("must never run")) {
		t.Fatalf("fallback payload: %q %v", got, err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestConceptProfileDefaultPreservesCommitSubstitution(t *testing.T) {
	t.Setenv(conceptTreeProfileEnv, "")
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".gitattributes"), []byte("version.txt export-subst\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "version.txt"), []byte("$Format:%H$\n"), 0644); err != nil {
		t.Fatal(err)
	}
	gitInitFixture(t, root)
	commit, err := git(root, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	tree, cleanup, err := materializeGitTree(root, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	got, err := os.ReadFile(filepath.Join(tree.root, "version.txt"))
	if err != nil || string(got) != string(commit) || tree.profile != nil {
		t.Fatalf("commit substitution: %q want %q: %v", got, commit, err)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestConceptProfileSelectedScriptCorruptionIsFatal(t *testing.T) {
	if os.Getenv("FAK_TEST_CONCEPT_ROOT") == "" {
		t.Skip("explicit runtime qualification only")
	}
	version := os.Getenv(conceptTreeProfileEnv)
	if version == "" {
		version = conceptTreeProfileV1
	}
	_, stdlib, selected := conceptProfilePins(version)
	if !selected {
		t.Fatal("unknown explicit profile")
	}
	p, err := conceptRuntimeProfile(stdlib)
	if err != nil || p == nil {
		t.Fatalf("runtime: %v %v", p, err)
	}
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "tools/concept_disambiguation_scorecard.py"), []byte("raise RuntimeError('corrupted archived generator')\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd, err := p.command(root, t.TempDir(), true)
	if cmd != nil || err == nil || !strings.Contains(err.Error(), "archived generator identity") {
		t.Fatalf("selected corruption must fail without fallback: %v %v", cmd, err)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestConceptProfileCurrentTreeEligibility(t *testing.T) {
	root, tree := os.Getenv("FAK_TEST_CONCEPT_ROOT"), os.Getenv("FAK_TEST_CONCEPT_TREE")
	if root == "" || tree == "" {
		t.Skip("explicit pinned-tree eligibility only")
	}
	if os.Getenv(conceptTreeProfileEnv) == "" {
		t.Setenv(conceptTreeProfileEnv, conceptTreeProfileV1)
	}
	p, err := qualifyConceptTreeProfile(root, tree)
	if err != nil || p == nil || p.tree != tree {
		t.Fatalf("current tree ineligible: %v %v", p, err)
	}
}

// compareConceptOracleTar is a test-only, read-only alternative to allocating a
// second full checkout. The caller must hold the existing checkout quiescent.
// Compare the actual archive stream, not cached Git stat data: this also refuses
// an oracle changed by export-subst, export-ignore, a peer edit or an extra file.
func compareConceptOracleTar(archive io.Reader, root string) error {
	return compareConceptOraclePayload(archive, root, false)
}

func compareConceptOraclePayload(archive io.Reader, root string, sparse bool) error {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil || !bytes.Contains(status, []byte("Umask:\t0022\n")) {
		return fmt.Errorf("oracle requires the native 022 extraction umask: %v", err)
	}
	st, err := os.Lstat(root)
	if err != nil || !st.IsDir() {
		return fmt.Errorf("oracle root is not a directory: %v", err)
	}
	seen := map[string]bool{".": true}
	declared := map[string]bool{}
	tr := tar.NewReader(archive)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		name := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(name) || name == ".." || strings.HasPrefix(name, ".."+string(filepath.Separator)) || name == ".git" || strings.HasPrefix(name, ".git"+string(filepath.Separator)) {
			return fmt.Errorf("unsafe oracle archive path %q", h.Name)
		}
		if !h.FileInfo().IsDir() && h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if declared[name] {
			return fmt.Errorf("duplicate oracle path %q", name)
		}
		declared[name] = true
		seen[name] = true
		// Match MkdirAll even when an archive omits explicit parent headers.
		for parent := filepath.Dir(name); parent != "."; parent = filepath.Dir(parent) {
			st, err := os.Lstat(filepath.Join(root, parent))
			if err != nil || !st.IsDir() || st.Mode().Perm() != 0755 {
				return fmt.Errorf("oracle parent directory %q: %v", parent, err)
			}
			seen[parent] = true
		}
		path := filepath.Join(root, name)
		st, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if h.FileInfo().IsDir() {
			if !st.IsDir() || st.Mode().Perm() != 0755 {
				return fmt.Errorf("oracle directory %q", name)
			}
			continue
		}
		// The native exporter creates files under the qualified 022 umask.
		if !st.Mode().IsRegular() || st.Size() != h.Size || st.Mode().Perm() != h.FileInfo().Mode().Perm()&^0022 {
			return fmt.Errorf("oracle file metadata %q", name)
		}
		hash := sha256.New()
		if sparse && !conceptProfileNeedsBytes(name, h.Size) {
			if _, err = io.Copy(io.Discard, tr); err != nil {
				return err
			}
			// The native sparse exporter truncates omitted files to zero-filled holes.
			var zeros [32 * 1024]byte
			for left := h.Size; left > 0; {
				n := min(left, int64(len(zeros)))
				_, _ = hash.Write(zeros[:n])
				left -= n
			}
		} else if _, err = io.Copy(hash, tr); err != nil {
			return err
		}
		got, err := conceptFileDigest(path)
		if err != nil || got != hex.EncodeToString(hash.Sum(nil)) {
			return fmt.Errorf("oracle payload %q: %v", name, err)
		}
	}
	return filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if name == ".git" {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !seen[name] {
			return fmt.Errorf("extra oracle path %q", name)
		}
		return nil
	})
}

// fak-test:runtime fast est=50ms lane=default
func TestConceptProfileReusedOracleRefusesDrift(t *testing.T) {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil || !bytes.Contains(status, []byte("Umask:\t0022\n")) {
		t.Skip("explicit Linux 022 oracle qualification only")
	}
	var raw bytes.Buffer
	w := tar.NewWriter(&raw)
	if err := w.WriteHeader(&tar.Header{Name: "a.txt", Mode: 0664, Size: 4, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("same")); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"exact", "changed", "extra", "symlink", "executable"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			p := filepath.Join(root, "a.txt")
			if err := os.WriteFile(p, []byte("same"), 0644); err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "changed":
				if err := os.WriteFile(p, []byte("diff"), 0644); err != nil {
					t.Fatal(err)
				}
			case "extra":
				if err := os.WriteFile(filepath.Join(root, "extra"), nil, 0644); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := os.Remove(p); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink("elsewhere", p); err != nil {
					t.Fatal(err)
				}
			case "executable":
				if err := os.Chmod(p, 0755); err != nil {
					t.Fatal(err)
				}
			}
			err := compareConceptOracleTar(bytes.NewReader(raw.Bytes()), root)
			if (err == nil) != (mode == "exact") {
				t.Fatalf("%s: %v", mode, err)
			}
		})
	}
}

// This continuation does not replay completed full arms or allocate another
// tree. It consumes caller-pinned evidence from an interrupted native proof,
// verifies every byte/mode/path of its retained sparse export against the SAME
// immutable archive, and runs only the unfinished sparse arm. It never retires
// the adopted input unless a separately authorized exact native proof root is
// explicitly supplied for normal producer retirement after successful proof.
// fak-test:runtime integration est=60s lane=default
func TestConceptProfileInterruptedSparseContinuation(t *testing.T) {
	root, tree := os.Getenv("FAK_TEST_CONCEPT_ROOT"), os.Getenv("FAK_TEST_CONCEPT_TREE")
	partial, prior := os.Getenv("FAK_TEST_CONCEPT_PARTIAL_TREE"), os.Getenv("FAK_TEST_CONCEPT_PRIOR_EVIDENCE")
	manifest, evidence := os.Getenv("FAK_TEST_CONCEPT_PRIOR_MANIFEST"), os.Getenv("FAK_TEST_CONCEPT_EVIDENCE")
	retire := os.Getenv("FAK_TEST_CONCEPT_RETIRE_PARTIAL_ROOT")
	if root == "" || tree == "" || partial == "" || prior == "" || manifest == "" || evidence == "" {
		t.Skip("explicit interrupted-proof continuation only")
	}
	if len(tree) != 40 && len(tree) != 64 {
		t.Fatal("immutable tree object ID required")
	}
	if _, err := hex.DecodeString(tree); err != nil {
		t.Fatal("immutable tree object ID required", err)
	}
	// Resolve output parents before any write; string inequality alone would
	// permit an alias or a descendant of the adopted source/evidence tree.
	absolute, err := filepath.Abs(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = os.Lstat(absolute); !os.IsNotExist(err) {
		t.Fatal("new evidence path required", err)
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(absolute))
	if err != nil {
		t.Fatal(err)
	}
	evidence = filepath.Join(parent, filepath.Base(absolute))
	if !filepath.IsAbs(os.TempDir()) {
		t.Fatal("absolute owned TMPDIR required")
	}
	temporary, err := filepath.EvalSymlinks(os.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	under := func(path, base string) bool {
		rel, e := filepath.Rel(base, path)
		return e == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	inputs := []string{root, partial, prior, manifest}
	var adopted os.FileInfo
	if retire != "" {
		adopted, err = conceptPartialOwner(retire, partial)
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, retire)
	}
	for _, input := range inputs {
		if !filepath.IsAbs(input) {
			t.Fatal("absolute immutable input required", input)
		}
		canonical, e := filepath.EvalSymlinks(input)
		if e != nil {
			t.Fatal(e)
		}
		for _, output := range []string{evidence, temporary} {
			if under(output, canonical) || under(canonical, output) {
				t.Fatalf("output %q overlaps input %q", output, canonical)
			}
		}
	}
	p, err := qualifyConceptTreeProfile(root, tree)
	if err != nil || p == nil {
		t.Fatalf("profile qualification: %v %v", p, err)
	}
	rawManifest, err := os.ReadFile(manifest)
	if err != nil {
		t.Fatal(err)
	}
	var pins map[string]string
	if err = json.Unmarshal(rawManifest, &pins); err != nil || len(pins) != 8 {
		t.Fatal("exact prior evidence pins required", err)
	}
	var wantJSON map[string]any
	var wantArtifacts map[string][]byte
	var wantStatus int
	for _, mode := range []string{"full-ordinary", "full-isolated"} {
		read := func(name string) []byte {
			key := mode + "/" + name
			b, e := os.ReadFile(filepath.Join(prior, mode, name))
			if e != nil || len(pins[key]) != 64 || conceptDigest(b) != pins[key] {
				t.Fatalf("prior evidence %s: %v", key, e)
			}
			return b
		}
		var storage struct {
			Tree     string `json:"tree"`
			ExitCode int    `json:"exit_code"`
		}
		if err = json.Unmarshal(read("storage.json"), &storage); err != nil || storage.Tree != tree {
			t.Fatal("prior tree mismatch", err)
		}
		var payload map[string]any
		if err = json.Unmarshal(read("stdout.json"), &payload); err != nil || payload["workspace"] != root {
			t.Fatal("prior workspace", err)
		}
		delete(payload, "workspace")
		artifacts := map[string][]byte{}
		for _, art := range generatedArtifacts {
			artifacts[art.Name] = read(art.Name)
		}
		if wantJSON == nil {
			wantJSON, wantArtifacts, wantStatus = payload, artifacts, storage.ExitCode
		} else if !reflect.DeepEqual(wantJSON, payload) || !reflect.DeepEqual(wantArtifacts, artifacts) || wantStatus != storage.ExitCode {
			t.Fatal("prior full arms disagree")
		}
	}
	compare := func(r io.Reader, dest string) error { return compareConceptOraclePayload(r, dest, true) }
	if err = extractGitTreeUsing(root, tree, partial, compare); err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	cmd, err := p.command(partial, out, true)
	if err != nil {
		t.Fatal(err)
	}
	cmd.Dir = partial
	windowgate.ConfigureBackgroundCommand(cmd)
	raw, runErr := cmd.Output()
	status := 0
	if runErr != nil {
		exit, ok := runErr.(*exec.ExitError)
		if !ok {
			t.Fatal(runErr)
		}
		status = exit.ExitCode()
	}
	if err = p.verifyRuntime(); err != nil {
		t.Fatal(err)
	}
	if err = extractGitTreeUsing(root, tree, partial, compare); err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err = json.Unmarshal(raw, &payload); err != nil || payload["workspace"] != partial {
		t.Fatal("sparse workspace", err)
	}
	delete(payload, "workspace")
	if status != wantStatus || !reflect.DeepEqual(wantJSON, payload) {
		t.Fatal("sparse snapshot differs from full oracles")
	}
	if err = os.Mkdir(evidence, 0700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(evidence, "stdout.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	for _, art := range generatedArtifacts {
		b, e := os.ReadFile(filepath.Join(out, art.Name))
		if e != nil || !bytes.Equal(b, wantArtifacts[art.Name]) {
			t.Fatal("sparse artifact", art.Name, e)
		}
		if e = os.WriteFile(filepath.Join(evidence, art.Name), b, 0600); e != nil {
			t.Fatal(e)
		}
	}
	record, _ := json.Marshal(map[string]any{"tree": tree, "exit_code": status, "prior_pins": pins, "partial_tree": partial, "retired": false})
	if err = os.WriteFile(filepath.Join(evidence, "continuation.json"), record, 0600); err != nil {
		t.Fatal(err)
	}
	if retire != "" {
		preserved := map[string][]byte{"stdout.json": raw, "continuation.json": record}
		for name, body := range wantArtifacts {
			preserved[name] = body
		}
		for name, want := range preserved {
			got, e := os.ReadFile(filepath.Join(evidence, name))
			if e != nil || !bytes.Equal(got, want) {
				t.Fatal("continuation evidence readback", name, e)
			}
		}
		// Fail closed before retirement unless every external evidence file and
		// its directory entries have been synchronized successfully.
		for name := range preserved {
			if e := syncConceptProofPath(filepath.Join(evidence, name)); e != nil {
				t.Fatal(e)
			}
		}
		for name, want := range pins {
			path := filepath.Join(prior, filepath.FromSlash(name))
			got, e := conceptFileDigest(path)
			if e != nil || got != want {
				t.Fatal("prior evidence changed", name, e)
			}
			if e = syncConceptProofPath(path); e != nil {
				t.Fatal(e)
			}
		}
		for _, dir := range []string{evidence, filepath.Dir(evidence), filepath.Join(prior, "full-ordinary"), filepath.Join(prior, "full-isolated"), prior, filepath.Dir(prior)} {
			if e := syncConceptProofPath(dir); e != nil {
				t.Fatal(e)
			}
		}
		current, e := conceptPartialOwner(retire, partial)
		if e != nil || !os.SameFile(adopted, current) {
			t.Fatal("adopted producer root changed", e)
		}
		// All source bytes are now witnessed recoverable from the pinned archive;
		// complete evidence is durable outside this single-child owned input root.
		if e = os.RemoveAll(retire); e != nil {
			t.Fatal(e)
		}
		if _, e = os.Lstat(retire); !os.IsNotExist(e) {
			t.Fatal("producer retirement not confirmed", e)
		}
		retired, _ := json.Marshal(map[string]any{"retired": true, "root": retire, "tree": tree, "prior_pins": pins})
		if e = syncConceptProofPath(filepath.Dir(retire)); e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(filepath.Join(evidence, "retirement.json"), retired, 0600); e != nil {
			t.Fatal(e)
		}
		if e = syncConceptProofPath(filepath.Join(evidence, "retirement.json")); e != nil {
			t.Fatal(e)
		}
		if e = syncConceptProofPath(evidence); e != nil {
			t.Fatal(e)
		}
	}
}

func syncConceptProofPath(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// Only the exact single-child shape minted by the native qualification producer
// is eligible. Caller authority and positive writer/consumer drain are required
// independently; names and hashes never manufacture ownership.
func conceptPartialOwner(root, tree string) (os.FileInfo, error) {
	if !filepath.IsAbs(root) || filepath.Clean(root) != root || tree != filepath.Join(root, "tree") {
		return nil, fmt.Errorf("exact absolute native producer root required")
	}
	suffix := strings.TrimPrefix(filepath.Base(root), "concept-profile-proof-")
	if suffix == filepath.Base(root) || suffix == "" || strings.Trim(suffix, "0123456789") != "" {
		return nil, fmt.Errorf("native proof root name required")
	}
	st, err := os.Lstat(root)
	if err != nil || !st.IsDir() {
		return nil, fmt.Errorf("native proof directory: %v", err)
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 1 || entries[0].Name() != "tree" || !entries[0].IsDir() {
		return nil, fmt.Errorf("native proof root has unknown contents: %v", err)
	}
	return st, nil
}

// fak-test:runtime fast est=20ms lane=default
func TestConceptProfilePartialRetirementBoundary(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concept-profile-proof-123")
	tree := filepath.Join(root, "tree")
	if err := os.MkdirAll(tree, 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := conceptPartialOwner(root, tree); err != nil {
		t.Fatal(err)
	}
	if _, err := conceptPartialOwner(root, root); err == nil {
		t.Fatal("wrong child accepted")
	}
	if err := os.WriteFile(filepath.Join(root, "evidence"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := conceptPartialOwner(root, tree); err == nil {
		t.Fatal("unknown sibling accepted")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestConceptProfileSparseOracleRejectsOmittedPayload(t *testing.T) {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil || !bytes.Contains(status, []byte("Umask:\t0022\n")) {
		t.Skip("explicit Linux 022 oracle qualification only")
	}
	var archive bytes.Buffer
	w := tar.NewWriter(&archive)
	if err = w.WriteHeader(&tar.Header{Name: "asset.bin", Mode: 0664, Size: 4, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err = w.Write([]byte("data")); err != nil {
		t.Fatal(err)
	}
	if err = w.Close(); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	file := filepath.Join(root, "asset.bin")
	if err = os.WriteFile(file, make([]byte, 4), 0644); err != nil {
		t.Fatal(err)
	}
	if err = compareConceptOraclePayload(bytes.NewReader(archive.Bytes()), root, true); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(file, []byte("data"), 0644); err != nil {
		t.Fatal(err)
	}
	if err = compareConceptOraclePayload(bytes.NewReader(archive.Bytes()), root, true); err == nil {
		t.Fatal("omitted payload bytes accepted")
	}
}
