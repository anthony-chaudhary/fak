package cudaarch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var fixtureFiles = []string{
	"internal/compute/cuda_arch.txt",
	"internal/compute/build_cuda.sh",
	"scripts/ci.ps1",
	"tools/build_cuda_windows.ps1",
	"Dockerfile.cuda",
	"docs/cuda-dev.md",
}

func fixture(tb testing.TB) string {
	tb.Helper()
	root := filepath.Clean(filepath.Join("..", ".."))
	dstRoot := tb.TempDir()
	for _, rel := range fixtureFiles {
		contents, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			tb.Fatal(err)
		}
		dst := filepath.Join(dstRoot, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			tb.Fatal(err)
		}
		if err := os.WriteFile(dst, contents, 0o644); err != nil {
			tb.Fatal(err)
		}
	}
	return dstRoot
}

func replaceFixture(tb testing.TB, root, rel, old, replacement string) {
	tb.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	contents, err := os.ReadFile(path)
	if err != nil {
		tb.Fatal(err)
	}
	updated := strings.Replace(string(contents), old, replacement, 1)
	if updated == string(contents) {
		tb.Fatalf("fixture %s did not contain %q", rel, old)
	}
	if err := os.WriteFile(path, []byte(updated), 0o644); err != nil {
		tb.Fatal(err)
	}
}

func TestCurrentTreeIsValid(t *testing.T) {
	errors, err := Validate(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(errors) != 0 {
		t.Fatalf("Validate() errors = %v", errors)
	}
}

func TestEntrypointsUseGoValidator(t *testing.T) {
	root := fixture(t)
	for _, rel := range []string{"internal/compute/build_cuda.sh", "scripts/ci.ps1"} {
		contents, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		text := string(contents)
		if strings.Contains(text, "tools/cuda_arch_matrix.py") {
			t.Errorf("%s still invokes retired Python validator", rel)
		}
		if !strings.Contains(text, "go test ./internal/cudaarch") {
			t.Errorf("%s does not invoke the Go CUDA architecture validator", rel)
		}
	}
}
func TestRejectsMissingSM120(t *testing.T) {
	root := fixture(t)
	replaceFixture(t, root, "internal/compute/cuda_arch.txt", "sm_120\n", "")
	assertErrorContains(t, root, "PTX floor")
}

func TestRejectsMissingPTXFloor(t *testing.T) {
	root := fixture(t)
	replaceFixture(t, root, "internal/compute/build_cuda.sh", `GENCODE+=(-gencode "arch=compute_${PTX_CC},code=compute_${PTX_CC}")`, "")
	assertErrorContains(t, root, "compute_${PTX_CC}")
}

func TestRejectsDocDrift(t *testing.T) {
	root := fixture(t)
	replaceFixture(t, root, "docs/cuda-dev.md", "cuda-build-sm120", "cuda-build-future")
	assertErrorContains(t, root, "docs")
}

func TestRejectsDuplicateAndMalformedArchitectures(t *testing.T) {
	root := fixture(t)
	path := filepath.Join(root, "internal", "compute", "cuda_arch.txt")
	if err := os.WriteFile(path, []byte("sm_80\nsm_80\nsm_future\nsm_120\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	assertErrorContains(t, root, "unique architecture list")
	assertErrorContains(t, root, "sm_<digits>")
}

func assertErrorContains(t *testing.T, root, want string) {
	t.Helper()
	errors, err := Validate(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, got := range errors {
		if strings.Contains(got, want) {
			return
		}
	}
	t.Fatalf("Validate() errors = %v, want substring %q", errors, want)
}

func TestParseArchitectures(t *testing.T) {
	t.Run("ValidMatrix", func(t *testing.T) {
		text := "  sm_80 \n\n sm_90\nsm_120\n"
		arches, errs := ParseArchitectures(text)
		if len(errs) != 0 {
			t.Fatalf("unexpected errors: %v", errs)
		}
		if len(arches) != 3 || arches[0] != "sm_80" || arches[1] != "sm_90" || arches[2] != "sm_120" {
			t.Fatalf("unexpected arches: %v", arches)
		}
	})

	t.Run("EmptyText", func(t *testing.T) {
		arches, errs := ParseArchitectures("   \n\n  ")
		if len(arches) != 0 {
			t.Fatalf("expected 0 arches, got %v", arches)
		}
		if len(errs) == 0 {
			t.Fatal("expected error on empty architecture list")
		}
	})

	t.Run("Duplicates", func(t *testing.T) {
		_, errs := ParseArchitectures("sm_80\nsm_80\nsm_120")
		if len(errs) == 0 || !strings.Contains(errs[0], "unique architecture list") {
			t.Fatalf("expected duplicate error, got %v", errs)
		}
	})

	t.Run("MalformedArch", func(t *testing.T) {
		_, errs := ParseArchitectures("sm_80\nsm_abc\nsm_120")
		if len(errs) == 0 || !strings.Contains(errs[0], "sm_<digits>") {
			t.Fatalf("expected malformed error, got %v", errs)
		}
	})
}

func TestValidateSourceFile(t *testing.T) {
	t.Run("MatchesAllNeedles", func(t *testing.T) {
		content := "alpha beta gamma"
		errs := ValidateSourceFile("test", content, []string{"alpha", "gamma"})
		if len(errs) != 0 {
			t.Fatalf("unexpected errors: %v", errs)
		}
	})

	t.Run("StripsBOM", func(t *testing.T) {
		content := "\xef\xbb\xbfalpha beta"
		errs := ValidateSourceFile("test", content, []string{"alpha"})
		if len(errs) != 0 {
			t.Fatalf("unexpected errors: %v", errs)
		}
	})

	t.Run("ReportsMissingNeedles", func(t *testing.T) {
		content := "alpha beta"
		errs := ValidateSourceFile("test", content, []string{"alpha", "delta"})
		if len(errs) != 1 || !strings.Contains(errs[0], "delta") {
			t.Fatalf("expected missing delta error, got %v", errs)
		}
	})
}
