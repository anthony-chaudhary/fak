package buildwitness

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildAdmissionOverlayMasksOnlyExcludedArtifacts(t *testing.T) {
	root := t.TempDir()
	manifest := filepath.Join(root, "unit.json")
	data := `{"schema":"fak.work-delivery/v1","id":"unit","axes":{"authoring":"recorded","compile_admission":"excluded","verification":"unverified","integration":"unintegrated","release":"not_ready"},"artifacts":[{"path":"fixture/recorded.go","kind":"go-source"}]}`
	if err := os.WriteFile(manifest, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	plan, err := BuildAdmissionOverlay(root, manifest)
	if err != nil {
		t.Fatal(err)
	}
	key := filepath.Join(root, "fixture", "recorded.go")
	if _, ok := plan.Overlay.Replace[key]; !ok {
		t.Fatalf("overlay = %#v, missing %s", plan.Overlay, key)
	}
	if len(plan.CompileSet.Admitted) != 0 || len(plan.CompileSet.Excluded) != 1 {
		t.Fatalf("compile set = %#v", plan.CompileSet)
	}
}
