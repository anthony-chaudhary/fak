package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"encoding/json"
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

// This opt-in qualification is deliberately sequential, outside a worker export.
// Raw stdout/artifacts and actual storage accounting are retained by the caller.
// fak-test:runtime integration est=90s lane=default
func TestConceptProfileQualifiedTreeEquivalence(t *testing.T) {
	root, tree := os.Getenv("FAK_TEST_CONCEPT_ROOT"), os.Getenv("FAK_TEST_CONCEPT_TREE")
	evidence := os.Getenv("FAK_TEST_CONCEPT_EVIDENCE")
	if root == "" || tree == "" || evidence == "" {
		t.Skip("explicit pinned-tree qualification only")
	}
	t.Setenv(conceptTreeProfileEnv, conceptTreeProfileV1)
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
			if e = extractGitTreeUsing(root, tree, export, extract); e != nil {
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
	p, err := conceptRuntimeProfile()
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
	t.Setenv(conceptTreeProfileEnv, conceptTreeProfileV1)
	p, err := qualifyConceptTreeProfile(root, tree)
	if err != nil || p == nil || p.tree != tree {
		t.Fatalf("current tree ineligible: %v %v", p, err)
	}
}
