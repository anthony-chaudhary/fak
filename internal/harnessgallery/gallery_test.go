package harnessgallery

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestBuiltinsAreValidAndCoverDistinctNeeds(t *testing.T) {
	got := Builtins()
	if len(got) != 4 {
		t.Fatalf("blueprints=%d", len(got))
	}
	if err := Validate(got); err != nil {
		t.Fatal(err)
	}
	want := []string{"cited-research", "coding-workspace", "incident-operations", "readonly-support"}
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("ids=%v", got)
		}
	}
	seams := map[string]bool{}
	for _, b := range got {
		seams[b.Seam] = true
	}
	if len(seams) != len(got) {
		t.Fatalf("gallery collapsed distinct needs onto %d seams", len(seams))
	}
}

func TestValidateRejectsContradictionAndUnsafeArtifact(t *testing.T) {
	b := Builtins()[0]
	b.ExcludedCapabilities = append(b.ExcludedCapabilities, b.RequiredCapabilities[0])
	if err := Validate([]Blueprint{b}); err == nil || !strings.Contains(err.Error(), "both requires and excludes") {
		t.Fatalf("err=%v", err)
	}
	b = Builtins()[0]
	b.OwnedArtifacts = []string{"../escape"}
	if err := Validate([]Blueprint{b}); err == nil || !strings.Contains(err.Error(), "unsafe") {
		t.Fatalf("err=%v", err)
	}
	if err := Validate([]Blueprint{Builtins()[0], Builtins()[0]}); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("err=%v", err)
	}
}

func TestInitEveryPackIsIdempotentAndPreservesUserFiles(t *testing.T) {
	for _, b := range Builtins() {
		t.Run(b.ID, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "pack")
			first, err := Init(b.ID, dir)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first.Created, []string{"harness.pack.json", "README.md"}) {
				t.Fatalf("first=%+v", first)
			}
			path := filepath.Join(dir, "README.md")
			body, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"decision scaffold", "## What to do next", "fak harness gallery selfcheck", b.Witness} {
				if !strings.Contains(string(body), want) {
					t.Fatalf("README missing %q:\n%s", want, body)
				}
			}
			custom := []byte("custom user instructions\n")
			if err := os.WriteFile(path, custom, 0o644); err != nil {
				t.Fatal(err)
			}
			second, err := Init(b.ID, dir)
			if err != nil {
				t.Fatal(err)
			}
			if len(second.Created) != 0 || len(second.Preserved) != 2 {
				t.Fatalf("second=%+v", second)
			}
			body, err = os.ReadFile(path)
			if err != nil || string(body) != string(custom) {
				t.Fatalf("body=%q err=%v", body, err)
			}
		})
	}
}
