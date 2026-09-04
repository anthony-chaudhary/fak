package marketplace

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T, root string) Descriptor {
	t.Helper()
	artifact := []byte("compiled fixture\n")
	if err := os.WriteFile(filepath.Join(root, "fixture.bin"), artifact, 0o600); err != nil {
		t.Fatal(err)
	}
	return ComputeAdapter.Descriptor(Descriptor{
		ID: "example.org/echo", Module: "internal/compute@r3+gabcdef0",
		Compatibility: Compatibility{Min: 1, Max: 1}, Artifact: "fixture.bin", ArtifactSHA256: SHA256(artifact),
		Trust: TrustCompiled, OnError: ErrorClosed, Capabilities: []string{"gpu.execute"},
	})
}

func catalogJSON(t *testing.T, ds ...Descriptor) []byte {
	t.Helper()
	b, err := json.Marshal(Catalog{Schema: Schema, Extensions: ds})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestParseEnumeratesWithoutExecuting(t *testing.T) {
	root := t.TempDir()
	d := fixture(t, root)
	d.Witness = Witness{Required: true, Command: []string{"does-not-run-during-parse"}, ResultSHA256: strings.Repeat("a", 64)}
	c, err := Parse(catalogJSON(t, d), map[string]int{"fak-compute": 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Extensions) != 1 || c.Extensions[0].ID != d.ID {
		t.Fatalf("unexpected catalog: %+v", c)
	}
}

func TestVerifyArtifactAndWitness(t *testing.T) {
	root := t.TempDir()
	d := fixture(t, root)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	argv := []string{exe, "-test.run=^TestWitnessHelper$", "-test.v=false", "--", "witness"}
	out, err := exec.Command(argv[0], argv[1:]...).Output()
	if err != nil {
		t.Fatal(err)
	}
	d.Witness = Witness{Required: true, Command: argv, ResultSHA256: SHA256(out)}
	ctx := context.Background()
	c, err := Parse(catalogJSON(t, d), map[string]int{"fak-compute": 1})
	if err != nil {
		t.Fatal(err)
	}
	r, err := Verify(ctx, c, VerifyOptions{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Valid || len(r.Verified) != 1 {
		t.Fatalf("unexpected report: %+v", r)
	}
}

func TestRefusals(t *testing.T) {
	root := t.TempDir()
	base := fixture(t, root)
	versions := map[string]int{"fak-compute": 1}
	tests := []struct {
		name   string
		mutate func(*Descriptor)
		want   string
	}{
		{"duplicate", func(*Descriptor) {}, "duplicate identity"},
		{"unknown seam", func(d *Descriptor) { d.Seam = "mystery" }, "unknown seam"},
		{"incompatible", func(d *Descriptor) { d.Compatibility.Min = 2; d.Compatibility.Max = 3 }, "incompatible ABI"},
		{"missing witness", func(d *Descriptor) { d.Witness = Witness{Required: true} }, "required witness"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := base
			tt.mutate(&d)
			ds := []Descriptor{d}
			if tt.name == "duplicate" {
				ds = append(ds, d)
			}
			_, err := Parse(catalogJSON(t, ds...), versions)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v want %q", err, tt.want)
			}
		})
	}
	bad := base
	bad.ArtifactSHA256 = strings.Repeat("0", 64)
	c, err := Parse(catalogJSON(t, bad), versions)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Verify(context.Background(), c, VerifyOptions{Root: root})
	if err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("err=%v", err)
	}
}

func TestVerifyRefusesArtifactEscape(t *testing.T) {
	root := t.TempDir()
	d := fixture(t, root)
	d.Artifact = "../outside"
	c, err := Parse(catalogJSON(t, d), map[string]int{"fak-compute": 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = Verify(context.Background(), c, VerifyOptions{Root: root})
	if err == nil || !strings.Contains(err.Error(), "escapes catalog root") {
		t.Fatalf("err=%v", err)
	}
}
func TestJSONGolden(t *testing.T) {
	root := t.TempDir()
	d := fixture(t, root)
	c, err := Parse(catalogJSON(t, d), map[string]int{"fak-compute": 1})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	want, err := os.ReadFile("testdata/catalog.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("golden mismatch\nGOT:\n%s\nWANT:\n%s", got, want)
	}
}

func TestWitnessHelper(t *testing.T) {
	for i, a := range os.Args {
		if a == "--" && i+1 < len(os.Args) {
			fmt.Println(os.Args[i+1])
			return
		}
	}
	t.Skip("helper only")
}
