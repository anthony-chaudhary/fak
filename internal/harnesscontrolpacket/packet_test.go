package harnesscontrolpacket

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCreateAndVerifyKeepsArmsIsolated(t *testing.T) {
	root := t.TempDir()
	binary := write(t, root, "source/fak", "binary")
	receipt := write(t, root, "source/receipt.json", "{}")
	for _, arm := range []string{"default-control", "scratch"} {
		materials := filepath.Join(root, "materials", arm)
		write(t, root, "materials/"+arm+"/arm-card.md", arm)
		write(t, root, "materials/"+arm+"/task-card.md", "task")
		if arm == "default-control" {
			for _, name := range []string{"kernel-component.txt", "product.json", "product.lock.json", "selection.json"} {
				write(t, root, "materials/"+arm+"/"+name, name)
			}
		}
		out := filepath.Join(root, "packet", arm)
		manifest, err := Create(CreateOptions{Arm: arm, MaterialsDir: materials, BinaryPath: binary, ReceiptPath: receipt, OutputDir: out, SourceCommit: "abc123", BinaryVersion: "0.44-study"})
		if err != nil {
			t.Fatal(err)
		}
		if err := Verify(out, manifest); err != nil {
			t.Fatal(err)
		}
		if arm == "scratch" {
			for _, file := range manifest.Files {
				if strings.Contains(file.Path, "product") || strings.Contains(file.Path, "kernel") {
					t.Fatalf("scratch leaked %s", file.Path)
				}
			}
		}
	}
}

func TestVerifyDetectsTamper(t *testing.T) {
	root := t.TempDir()
	materials := filepath.Join(root, "materials")
	write(t, root, "materials/arm-card.md", "card")
	write(t, root, "materials/task-card.md", "task")
	binary := write(t, root, "source/fak", "binary")
	receipt := write(t, root, "source/receipt.json", "{}")
	out := filepath.Join(root, "packet")
	manifest, err := Create(CreateOptions{Arm: "scratch", MaterialsDir: materials, BinaryPath: binary, ReceiptPath: receipt, OutputDir: out, SourceCommit: "abc123", BinaryVersion: "study"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "task-card.md"), []byte("edited"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, manifest); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("error=%v", err)
	}
}

func write(t *testing.T, root, relative, contents string) string {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
