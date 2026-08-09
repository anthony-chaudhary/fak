package unwiredscore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pinRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.test/pins\n")
	write("internal/orphan/orphan.go", "package orphan\nfunc Run() {}\n")
	write("internal/orphan/doc.go", "package orphan // NOT WIRED: no external importer\n")
	return root
}

func TestClaimPinBidirectionalAndFiresOnSuccess(t *testing.T) {
	root := pinRepo(t)
	pin := ClaimPin{Package: "internal/orphan", ProsePath: "internal/orphan/doc.go", ClaimText: "NOT WIRED: no external importer", Predicate: ExternalImporter}
	if err := CheckClaimPin(root, pin); err != nil {
		t.Fatalf("unwired pin: %v", err)
	}

	// A string/comment mention is not an AST import and must not fire.
	if err := os.WriteFile(filepath.Join(root, "fixture.go"), []byte("package pins\nvar _ = `example.test/pins/internal/orphan` // import is only prose\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckClaimPin(root, pin); err != nil {
		t.Fatalf("fixture string counted as caller: %v", err)
	}

	if err := os.WriteFile(filepath.Join(root, "caller.go"), []byte("package pins\nimport _ \"example.test/pins/internal/orphan\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckClaimPin(root, pin); err == nil || !strings.Contains(err.Error(), "delete prose") || !strings.Contains(err.Error(), pin.ClaimText) {
		t.Fatalf("wired package must fire with actionable prose, got %v", err)
	}
	if err := os.Remove(filepath.Join(root, "caller.go")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal/orphan/doc.go"), []byte("package orphan\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckClaimPin(root, pin); err == nil || !strings.Contains(err.Error(), "remains but prose") {
		t.Fatalf("deleted prose with live pin must fail bidirectionally, got %v", err)
	}
}

func TestMeasurementPinTracksProseAndValue(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "STATUS.md"), []byte("There are 3 unwired packages.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	measured := 3
	pin := MeasurementPin{ProsePath: "STATUS.md", ClaimText: "There are 3 unwired packages.", Want: 3, Measure: func(string) (int, error) { return measured, nil }}
	if err := CheckMeasurementPin(root, pin); err != nil {
		t.Fatalf("matching measurement: %v", err)
	}
	measured = 4
	if err := CheckMeasurementPin(root, pin); err == nil || !strings.Contains(err.Error(), "measurement is 4") {
		t.Fatalf("stale number must name live measurement, got %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "STATUS.md"), []byte("No count here.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckMeasurementPin(root, pin); err == nil || !strings.Contains(err.Error(), "remains but prose") {
		t.Fatalf("removed measurement prose with live pin must fail, got %v", err)
	}
}
