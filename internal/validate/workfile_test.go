package validate

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestPrepareValidateWorkfilePreservesCompanionsAndDirectives(t *testing.T) {
	t.Setenv("GOWORK", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	parent := t.TempDir()
	source := filepath.Join(parent, "modules", "private")
	public := filepath.Join(parent, "modules", "public")
	companion := filepath.Join(parent, "modules", "companion")
	extracted := filepath.Join(t.TempDir(), "candidate")
	for root, identity := range map[string]string{source: "fixture.test/private", public: publicModulePath, companion: "fixture.test/companion", extracted: "fixture.test/private"} {
		writeWorkfileFixture(t, filepath.Join(root, "go.mod"), "module "+identity+"\n\ngo 1.26\n")
	}
	workfile := filepath.Join(parent, "go.work")
	original := "// retain workspace policy\ngo 1.26\ntoolchain go1.26.9\ngodebug panicnil=1\nuse (\n ./modules/private\n ./modules/public\n ./modules/companion\n)\nreplace fixture.test/local v1.0.0 => ./replacement\nreplace fixture.test/upstream v1.0.0 => fixture.test/mirror v1.1.0\n"
	writeWorkfileFixture(t, workfile, original)
	if err := prepareValidateWorkfile(context.Background(), source, extracted, false); err != nil {
		t.Fatal(err)
	}
	got := readWorkfileFixture(t, filepath.Join(extracted, "go.work"))
	wantUses := map[string]bool{public: true, companion: true, extracted: true}
	if len(got.Use) != len(wantUses) {
		t.Fatalf("use entries=%v want=%v", got.Use, wantUses)
	}
	for _, use := range got.Use {
		if !wantUses[use.DiskPath] {
			t.Fatalf("unexpected workspace use %q", use.DiskPath)
		}
		delete(wantUses, use.DiskPath)
	}
	if len(wantUses) != 0 {
		t.Fatalf("missing workspace uses: %v", wantUses)
	}
	wantReplacements := map[string]string{
		"fixture.test/local@v1.0.0":    filepath.Join(parent, "replacement") + "@",
		"fixture.test/upstream@v1.0.0": "fixture.test/mirror@v1.1.0",
	}
	actual := make(map[string]string)
	for _, replacement := range got.Replace {
		actual[replacement.Old.Path+"@"+replacement.Old.Version] = replacement.New.Path + "@" + replacement.New.Version
	}
	if !reflect.DeepEqual(actual, wantReplacements) {
		t.Fatalf("replacements=%v want=%v", actual, wantReplacements)
	}
	data, err := os.ReadFile(filepath.Join(extracted, "go.work"))
	if err != nil {
		t.Fatal(err)
	}
	for _, directive := range []string{"// retain workspace policy", "go 1.26", "toolchain go1.26.9", "godebug panicnil=1"} {
		if !strings.Contains(string(data), directive) {
			t.Fatalf("lost directive/comment %q in %s", directive, data)
		}
	}
	unchanged, err := os.ReadFile(workfile)
	if err != nil || string(unchanged) != original {
		t.Fatalf("source workspace changed: %q, %v", unchanged, err)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestPrepareValidateWorkfileRefusesInvalidBindingBeforeWrite(t *testing.T) {
	t.Setenv("GOWORK", "")
	t.Setenv("GOTOOLCHAIN", "local")
	t.Setenv("GOPROXY", "off")
	for _, kind := range []string{"missing_target", "duplicate_target", "wrong_source", "candidate_identity", "missing_companion", "unsupported_replace", "same_root"} {
		t.Run(kind, func(t *testing.T) {
			parent := t.TempDir()
			source, public := filepath.Join(parent, "private"), filepath.Join(parent, "public")
			extracted := filepath.Join(t.TempDir(), "candidate")
			for root, identity := range map[string]string{source: "fixture.test/private", public: publicModulePath, extracted: "fixture.test/private"} {
				writeWorkfileFixture(t, filepath.Join(root, "go.mod"), "module "+identity+"\n\ngo 1.26\n")
			}
			use := "use (\n ./private\n ./public\n)\n"
			suffix := ""
			switch kind {
			case "missing_target":
				use = "use ./public\n"
			case "duplicate_target":
				use = "use (\n ./private\n ./private/.\n ./public\n)\n"
			case "wrong_source":
				writeWorkfileFixture(t, filepath.Join(parent, "other", "go.mod"), "module fixture.test/private\n\ngo 1.26\n")
				use = "use (\n ./other\n ./public\n)\n"
			case "candidate_identity":
				writeWorkfileFixture(t, filepath.Join(extracted, "go.mod"), "module fixture.test/different\n\ngo 1.26\n")
			case "missing_companion":
				use = "use (\n ./private\n ./public\n ./missing\n)\n"
			case "unsupported_replace":
				suffix = "replace fixture.test/local => ./path@name\n"
			case "same_root":
				extracted = source
			}
			original := "go 1.26\n" + use + suffix
			workfile := filepath.Join(parent, "go.work")
			writeWorkfileFixture(t, workfile, original)
			privateWorkfile := filepath.Join(extracted, "go.work")
			if err := prepareValidateWorkfile(context.Background(), source, extracted, false); err == nil {
				t.Fatal("invalid binding accepted")
			}
			if _, err := os.Stat(privateWorkfile); !os.IsNotExist(err) {
				t.Fatalf("invalid binding wrote candidate workspace: %v", err)
			}
			got, err := os.ReadFile(workfile)
			if err != nil || string(got) != original {
				t.Fatal("invalid binding changed source workspace")
			}
		})
	}
}

func writeWorkfileFixture(t *testing.T, path, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readWorkfileFixture(t *testing.T, path string) validateWorkfile {
	t.Helper()
	cmd := exec.Command("go", "work", "edit", "-json", path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("inspect workspace: %v: %s", err, out)
	}
	var got validateWorkfile
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	return got
}
