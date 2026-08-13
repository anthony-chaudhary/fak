package managedinventory

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func repositoryRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func loadProductionCatalog(t *testing.T) Catalog {
	t.Helper()
	c, err := Load(filepath.Join(repositoryRoot(t), filepath.FromSlash(DefaultSourceRel)))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestCommittedInventoryAndReportStayInSync(t *testing.T) {
	c := loadProductionCatalog(t)
	if ds := Validate(c, Registrations()); len(ds) != 0 {
		for _, d := range ds {
			t.Error(d.Error())
		}
		return
	}
	want := RenderMarkdown(c)
	got, err := os.ReadFile(filepath.Join(repositoryRoot(t), filepath.FromSlash(DefaultReportRel)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("generated portability report is stale; run go run ./cmd/managedinventory --write")
	}
}

func TestRenderMarkdownIsDeterministic(t *testing.T) {
	c := loadProductionCatalog(t)
	a := RenderMarkdown(c)
	b := RenderMarkdown(c)
	if !bytes.Equal(a, b) {
		t.Fatal("two renders of one catalog produced different bytes")
	}
}

func TestRegisteredNewObjectWithoutInventoryRowFails(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "registrations-with-new-object.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	if err := json.Unmarshal(b, &ids); err != nil {
		t.Fatal(err)
	}
	regs := make([]Registration, 0, len(ids))
	for _, id := range ids {
		regs = append(regs, Registration{ID: id})
	}
	ds := Validate(loadProductionCatalog(t), regs)
	joined := make([]string, 0, len(ds))
	for _, d := range ds {
		joined = append(joined, d.Error())
	}
	got := strings.Join(joined, "\n")
	if !strings.Contains(got, "MISSING_INVENTORY_ROW: object future-managed-object.id") {
		t.Fatalf("registered-new-object fixture did not fail at the drift gate:\n%s", got)
	}
}

func TestCountGrepOutput(t *testing.T) {
	lines, files := CountGrepOutput([]byte("deadbeef:a/x.go:1:skill\ndeadbeef:a/x.go:2:loop\ndeadbeef:b/y.go:9:policy\n"))
	if lines != 3 || files != 2 {
		t.Fatalf("lines/files=%d/%d, want 3/2", lines, files)
	}
}
